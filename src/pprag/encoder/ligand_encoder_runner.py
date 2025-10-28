#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/ligand_encoder_runner.py: CLI for batch encoding ligand molecules.

This script:
1. Loads pre-built LigandGraph objects from pickle files
2. Initializes LigandEncoder model (optionally from checkpoint)
3. Batch processes all ligands through the encoder
4. Saves ligand embeddings (z_lig, h_atoms) for downstream tasks

Output format:
    - Global embeddings: {ligand_id}.pt containing z_lig tensor
    - Per-atom embeddings: {ligand_id}_atoms.pt containing h_atoms tensor
    - Metadata: encodings_metadata.json with encoding statistics

Author: Aaryesh Deshpande
Last Modified: 10/27/2025
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from ligand_encoder import LigandEncoder
from pprag.dataio.schema import LigandGraph


app = typer.Typer(
    name="ligand-encoder",
    help="Batch encode ligand molecules using pre-trained or random-initialized encoder",
    add_completion=False,
)
console = Console()


class LigandGraphDataset(Dataset):
    """PyTorch Dataset for loading LigandGraph pickle files."""

    def __init__(self, graph_dir: Path, ligand_ids: Optional[List[str]] = None):
        """
        Initialize dataset.

        Args:
            graph_dir: Directory containing {ligand_id}.pkl files
            ligand_ids: Optional list of specific ligand IDs to load
        """
        self.graph_dir = graph_dir

        if ligand_ids:
            self.ligand_files = [graph_dir / f"{lid}.pkl" for lid in ligand_ids]
        else:
            self.ligand_files = sorted(graph_dir.glob("*.pkl"))

        # Filter to existing files
        self.ligand_files = [f for f in self.ligand_files if f.exists()]

    def __len__(self) -> int:
        return len(self.ligand_files)

    def __getitem__(self, idx: int) -> tuple:
        """Load LigandGraph from pickle."""
        pkl_path = self.ligand_files[idx]

        with open(pkl_path, 'rb') as f:
            lig_graph = pickle.load(f)

        if not isinstance(lig_graph, LigandGraph):
            raise TypeError(f"Expected LigandGraph, got {type(lig_graph)}")

        return lig_graph, pkl_path.stem  # Return graph and ligand_id


def collate_ligand_graphs(batch: List[tuple]) -> tuple:
    """
    Collate function for DataLoader - handles variable-sized graphs.

    Returns batched tensors with proper indexing for PyG-style processing.
    """
    graphs, ligand_ids = zip(*batch)

    # Stack node features
    x_list = [torch.from_numpy(g.x).float() for g in graphs]
    pos_list = [torch.from_numpy(g.pos).float() for g in graphs]

    # Handle edges with batch offset
    edge_index_list = []
    edge_attr_list = []
    node_offset = 0

    for g in graphs:
        edge_index = torch.from_numpy(g.edge_index).long()
        edge_attr = torch.from_numpy(g.edge_attr).float()

        # Add offset to edge indices
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)

        node_offset += g.x.shape[0]

    # Concatenate all
    x = torch.cat(x_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, graphs[0].edge_attr.shape[1]))

    # Create batch tensor (node assignment)
    batch_tensor = torch.cat([torch.full((len(x_list[i]),), i, dtype=torch.long)
                              for i in range(len(graphs))])

    # Handle hypergraph data
    hyperedge_attr_list = []
    hyperedge_members_batched = []
    hyperedge_offset = 0    # noqa
    node_offset = 0

    for g in graphs:
        if g.hyperedge_attr is not None and g.hyperedge_members is not None:
            hyperedge_attr_list.append(torch.from_numpy(g.hyperedge_attr).float())

            # Adjust member indices with node offset
            for members in g.hyperedge_members:
                adjusted_members = [m + node_offset for m in members]
                hyperedge_members_batched.append(adjusted_members)

            node_offset += g.x.shape[0]
        else:
            # No hypergraph for this molecule
            pass

    hyperedge_attr = torch.cat(hyperedge_attr_list, dim=0) if hyperedge_attr_list else None
    hyperedge_members = hyperedge_members_batched if hyperedge_members_batched else None

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'pos': pos,
        'batch': batch_tensor,
        'hyperedge_attr': hyperedge_attr,
        'hyperedge_members': hyperedge_members,
        'ligand_ids': ligand_ids,
        'n_atoms_per_mol': [g.x.shape[0] for g in graphs]
    }


@torch.no_grad()
def encode_batch(model: nn.Module, batch_data: dict, device: torch.device) -> Dict[str, List]:
    """
    Encode a batch of ligands.

    Returns:
        Dictionary with lists of tensors for each ligand in batch
    """
    # Move to device
    x = batch_data['x'].to(device)
    edge_index = batch_data['edge_index'].to(device)
    edge_attr = batch_data['edge_attr'].to(device)
    pos = batch_data['pos'].to(device)
    batch_tensor = batch_data['batch'].to(device)

    hyperedge_attr = batch_data['hyperedge_attr'].to(device) if batch_data['hyperedge_attr'] is not None else None
    hyperedge_members = batch_data['hyperedge_members']

    # Forward pass
    z_lig, h_atoms, _ = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        hyperedge_attr=hyperedge_attr,
        hyperedge_members=hyperedge_members,
        batch=batch_tensor,
        return_attn=False
    )

    # Split per-molecule results
    batch_size = int(batch_tensor.max().item()) + 1
    n_atoms_per_mol = batch_data['n_atoms_per_mol']

    z_lig_list = []
    h_atoms_list = []

    atom_offset = 0
    for i in range(batch_size):
        # Global embedding (already per-molecule)
        z_lig_list.append(z_lig[i].cpu())

        # Per-atom embeddings
        n_atoms = n_atoms_per_mol[i]
        h_atoms_mol = h_atoms[atom_offset:atom_offset + n_atoms].cpu()
        h_atoms_list.append(h_atoms_mol)
        atom_offset += n_atoms

    return {
        'z_lig': z_lig_list,
        'h_atoms': h_atoms_list,
        'ligand_ids': batch_data['ligand_ids']
    }


@app.command()
def main(
    graph_dir: Path = typer.Option(
        ...,
        "--graph-dir",
        "-g",
        help="Directory containing LigandGraph pickle files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for ligand embeddings",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint (optional - uses random init if not provided)",
        exists=False,
    ),
    ligand_ids_json: Optional[Path] = typer.Option(
        None,
        "--ligand-ids",
        "-l",
        help="JSON file with list of ligand IDs to encode (optional - encodes all if not provided)",
        exists=False,
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for encoding",
        min=1,
        max=512,
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="Number of DataLoader workers",
        min=0,
        max=32,
    ),
    d_node: int = typer.Option(
        44,
        "--d-node",
        help="Input node feature dimension",
    ),
    d_edge: int = typer.Option(
        6,
        "--d-edge",
        help="Edge feature dimension",
    ),
    d_hyper_feat: int = typer.Option(
        15,
        "--d-hyper-feat",
        help="Hyperedge feature dimension",
    ),
    d_model: int = typer.Option(
        256,
        "--d-model",
        help="Model hidden dimension",
    ),
    n_gnn_layers: int = typer.Option(
        3,
        "--n-gnn-layers",
        help="Number of GNN layers",
    ),
    n_hyper_layers: int = typer.Option(
        2,
        "--n-hyper-layers",
        help="Number of hypergraph layers",
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available",
    ),
    save_atom_embeddings: bool = typer.Option(
        True,
        "--save-atoms/--no-save-atoms",
        help="Save per-atom embeddings (can be large)",
    ),
) -> None:
    """
    Batch encode ligand molecules using LigandEncoder.

    This script loads pre-built LigandGraph objects, processes them through
    the encoder, and saves the resulting embeddings for downstream use.

    Example:
    \b
        python3 src/pprag/encoder/ligand_encoder_runner.py \\
            --graph-dir Output/graphs/ligands \\
            --output-dir Output/embeddings/ligands \\
            --checkpoint models/ligand_encoder_epoch10.pt \\
            --batch-size 32 \\
            --gpu
    """

    console.print("\n[bold cyan]Ligand Encoder - Batch Processing[/bold cyan]\n")

    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        console.print(f"[green]✓[/green] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        console.print("[yellow]⚠[/yellow] Using CPU (this may be slow)")

    # Load ligand IDs if provided
    ligand_ids = None
    if ligand_ids_json and ligand_ids_json.exists():
        with open(ligand_ids_json) as f:
            ligand_ids = json.load(f)
        console.print(f"[cyan]•[/cyan] Loaded {len(ligand_ids)} ligand IDs from filter file")

    # Create dataset
    console.print(f"[cyan]•[/cyan] Loading graphs from: {graph_dir}")
    dataset = LigandGraphDataset(graph_dir, ligand_ids)

    if len(dataset) == 0:
        console.print("[red]✗[/red] No ligand graphs found!", style="bold red")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Found {len(dataset)} ligand graphs")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ligand_graphs,
        pin_memory=(device.type == 'cuda')
    )

    # Initialize model
    console.print("\n[bold yellow]Initializing Model[/bold yellow]")
    model = LigandEncoder(
        d_node=d_node,
        d_edge=d_edge,
        d_hyper_feat=d_hyper_feat,
        d_model=d_model,
        n_gnn_layers=n_gnn_layers,
        n_hyper_layers=n_hyper_layers,
        n_heads=4,
        dropout=0.1
    )

    # Load checkpoint if provided
    if checkpoint and checkpoint.exists():
        console.print(f"[cyan]•[/cyan] Loading checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        console.print("[green]✓[/green] Checkpoint loaded successfully")
    else:
        console.print("[yellow]⚠[/yellow] No checkpoint provided - using random initialization")

    model = model.to(device)
    model.eval()

    # Display model info
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[cyan]•[/cyan] Model parameters: {n_params:,}")
    console.print(f"[cyan]•[/cyan] Output dimension: {d_model}")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    global_dir = output_dir / "global"
    global_dir.mkdir(exist_ok=True)

    if save_atom_embeddings:
        atoms_dir = output_dir / "atoms"
        atoms_dir.mkdir(exist_ok=True)

    # Batch processing
    console.print("\n[bold yellow]Encoding Ligands[/bold yellow]")

    all_ligand_ids = []
    encoding_stats = {
        'n_ligands': 0,
        'n_atoms_total': 0,
        'd_model': d_model,
        'has_atom_embeddings': save_atom_embeddings,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(dataset)} ligands...",
            total=len(dataloader)
        )

        for batch_idx, batch_data in enumerate(dataloader):
            # Encode batch
            results = encode_batch(model, batch_data, device)

            # Save individual ligand embeddings
            for z_lig, h_atoms, ligand_id in zip(
                results['z_lig'],
                results['h_atoms'],
                results['ligand_ids']
            ):
                # Save global embedding
                torch.save(z_lig, global_dir / f"{ligand_id}.pt")

                # Save per-atom embeddings if requested
                if save_atom_embeddings:
                    torch.save(h_atoms, atoms_dir / f"{ligand_id}_atoms.pt")

                all_ligand_ids.append(ligand_id)
                encoding_stats['n_atoms_total'] += h_atoms.shape[0]

            encoding_stats['n_ligands'] += len(results['ligand_ids'])
            progress.update(task, advance=1)

    # Save metadata
    metadata = {
        'encoding_stats': encoding_stats,
        'model_config': {
            'd_node': d_node,
            'd_edge': d_edge,
            'd_hyper_feat': d_hyper_feat,
            'd_model': d_model,
            'n_gnn_layers': n_gnn_layers,
            'n_hyper_layers': n_hyper_layers,
        },
        'ligand_ids': all_ligand_ids,
        'checkpoint': str(checkpoint) if checkpoint else None,
    }

    metadata_path = output_dir / "encodings_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Display summary
    console.print("\n[bold green]✓ Encoding Complete![/bold green]\n")

    summary_table = Table(title="Encoding Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Ligands encoded", f"{encoding_stats['n_ligands']:,}")
    summary_table.add_row("Total atoms", f"{encoding_stats['n_atoms_total']:,}")
    summary_table.add_row("Embedding dimension", str(d_model))
    summary_table.add_row("Avg atoms per ligand", f"{encoding_stats['n_atoms_total'] / encoding_stats['n_ligands']:.1f}")

    console.print(summary_table)
    console.print(f"\n[green]✓[/green] Saved metadata to: [cyan]{metadata_path}[/cyan]")
    console.print()


if __name__ == "__main__":
    app()
