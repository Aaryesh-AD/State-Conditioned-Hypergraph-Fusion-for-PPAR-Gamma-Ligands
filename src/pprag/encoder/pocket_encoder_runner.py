#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
encoder/pocket_encoder_runner.py: CLI for batch encoding protein pockets.

This script:
1. Loads pre-built PocketGraph objects from pickle files
2. Initializes PocketEncoder model (optionally from checkpoint)
3. Batch processes all pockets through the encoder with state conditioning
4. Saves pocket embeddings (z_poc, h_residues) for downstream tasks

Output format:
    - Global embeddings: {target_id}__{state}.pt containing z_poc tensor
    - Per-residue embeddings: {target_id}__{state}_residues.pt containing h_residues tensor
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
from pocket_encoder import PocketEncoder
from pprag.dataio.schema import PocketGraph


app = typer.Typer(
    name="pocket-encoder",
    help="Batch encode protein pockets with state conditioning using pre-trained or random-initialized encoder",
    add_completion=False,
)
console = Console()


class PocketGraphDataset(Dataset):
    """PyTorch Dataset for loading PocketGraph pickle files."""

    def __init__(self, graph_dir: Path, target_ids: Optional[List[str]] = None):
        """
        Initialize dataset.

        Args:
            graph_dir: Directory containing {target_id}__{state}.pkl files
            target_ids: Optional list of specific target IDs to load (loads all states for each target)
        """
        self.graph_dir = graph_dir

        if target_ids:
            # Load specific targets (all states)
            self.pocket_files = []
            for tid in target_ids:
                # Try both agonist and antagonist states
                for state in ["agonist", "antagonist"]:
                    pkl_path = graph_dir / f"{tid}__{state}.pkl"
                    if pkl_path.exists():
                        self.pocket_files.append(pkl_path)
        else:
            self.pocket_files = sorted(graph_dir.glob("*.pkl"))

        # Filter to existing files
        self.pocket_files = [f for f in self.pocket_files if f.exists()]

    def __len__(self) -> int:
        return len(self.pocket_files)

    def __getitem__(self, idx: int) -> tuple:
        """Load PocketGraph from pickle."""
        pkl_path = self.pocket_files[idx]

        with open(pkl_path, 'rb') as f:
            poc_graph = pickle.load(f)

        if not isinstance(poc_graph, PocketGraph):
            raise TypeError(f"Expected PocketGraph, got {type(poc_graph)}")

        # Extract target_id and state from filename: {target_id}__{state}.pkl
        filename = pkl_path.stem
        target_id, state = filename.rsplit('__', 1)

        return poc_graph, target_id, state


def collate_pocket_graphs(batch: List[tuple]) -> dict:
    """
    Collate function for DataLoader - handles variable-sized graphs.

    Returns batched tensors with proper indexing for PyG-style processing.
    """
    graphs, target_ids, states = zip(*batch)

    # Stack node features
    x_list = [torch.from_numpy(g.x).float() for g in graphs]
    pos_ca_list = [torch.from_numpy(g.pos_ca).float() for g in graphs]

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
    pos_ca = torch.cat(pos_ca_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, graphs[0].edge_attr.shape[1]))

    # Create batch tensor - node assignment
    batch_tensor = torch.cat([torch.full((len(x_list[i]),), i, dtype=torch.long)
                              for i in range(len(graphs))])

    # State IDs
    state_ids = torch.tensor([g.state_id for g in graphs], dtype=torch.long)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'pos_ca': pos_ca,
        'batch': batch_tensor,
        'state_ids': state_ids,
        'target_ids': target_ids,
        'states': states,
        'n_residues_per_pocket': [g.x.shape[0] for g in graphs]
    }


@torch.no_grad()
def encode_batch(model: nn.Module, batch_data: dict, device: torch.device) -> Dict[str, List]:
    """
    Encode a batch of pockets.

    Returns:
        Dictionary with lists of tensors for each pocket in batch
    """
    x = batch_data['x'].to(device)
    edge_index = batch_data['edge_index'].to(device)
    edge_attr = batch_data['edge_attr'].to(device)
    batch_tensor = batch_data['batch'].to(device)   # noqa
    state_ids = batch_data['state_ids'].to(device)

    # Forward pass - process each pocket separately to handle state conditioning properly
    batch_size = len(batch_data['target_ids'])
    n_residues_per_pocket = batch_data['n_residues_per_pocket']

    z_poc_list = []
    h_residues_list = []

    node_offset = 0
    edge_offset = 0

    for i in range(batch_size):
        # Extract data for this pocket
        n_residues = n_residues_per_pocket[i]

        # Node features
        x_i = x[node_offset:node_offset + n_residues]

        # Find edges for this pocket
        mask = (batch_data['batch'] == i).cpu().numpy()
        edge_mask = mask[batch_data['edge_index'][0].cpu().numpy()]

        edge_index_i = edge_index[:, edge_offset:edge_offset + edge_mask.sum()] - node_offset
        edge_attr_i = edge_attr[edge_offset:edge_offset + edge_mask.sum()]

        # State ID
        state_id_i = state_ids[i:i + 1]

        # Forward pass for this pocket
        z_poc_i, h_residues_i = model(
            x=x_i,
            edge_index=edge_index_i,
            edge_attr=edge_attr_i,
            state_id=state_id_i,
            batch=None  # Single pocket
        )

        z_poc_list.append(z_poc_i.squeeze(0).cpu())  # Remove batch dim
        h_residues_list.append(h_residues_i.cpu())

        node_offset += n_residues
        edge_offset += edge_mask.sum()

    return {
        'z_poc': z_poc_list,
        'h_residues': h_residues_list,
        'target_ids': batch_data['target_ids'],
        'states': batch_data['states']
    }


@app.command()
def main(
    graph_dir: Path = typer.Option(
        ...,
        "--graph-dir",
        "-g",
        help="Directory containing PocketGraph pickle files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for pocket embeddings",
    ),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint (optional - uses random init if not provided)",
        exists=False,
    ),
    target_ids_json: Optional[Path] = typer.Option(
        None,
        "--target-ids",
        "-t",
        help="JSON file with list of target IDs to encode (optional - encodes all if not provided)",
        exists=False,
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Batch size for encoding (smaller than ligands due to larger graphs)",
        min=1,
        max=256,
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
        30,
        "--d-node",
        help="Input node feature dimension",
    ),
    d_edge: int = typer.Option(
        19,
        "--d-edge",
        help="Edge feature dimension",
    ),
    d_model: int = typer.Option(
        256,
        "--d-model",
        help="Model hidden dimension",
    ),
    n_layers: int = typer.Option(
        4,
        "--n-layers",
        help="Number of GNN layers",
    ),
    use_state_token: bool = typer.Option(
        True,
        "--state-token/--no-state-token",
        help="Use state conditioning (agonist/antagonist)",
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available",
    ),
    save_residue_embeddings: bool = typer.Option(
        True,
        "--save-residues/--no-save-residues",
        help="Save per-residue embeddings (can be large)",
    ),
) -> None:
    """
    Batch encode protein pockets using PocketEncoder with state conditioning.

    This script loads pre-built PocketGraph objects, processes them through
    the encoder with proper state tokens (agonist/antagonist), and saves
    the resulting embeddings for downstream use.

    Example:
    \b
        python3 src/pprag/encoder/pocket_encoder_runner.py \\
            --graph-dir Output/graphs/pockets \\
            --output-dir Output/embeddings/pockets \\
            --checkpoint models/pocket_encoder_epoch10.pt \\
            --batch-size 16 \\
            --state-token \\
            --gpu
    """

    console.print("\n[bold cyan]Pocket Encoder - Batch Processing[/bold cyan]\n")

    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        console.print(f"[green]✓[/green] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        console.print("[yellow]⚠[/yellow] Using CPU (this may be slow)")

    # Load target IDs if provided
    target_ids = None
    if target_ids_json and target_ids_json.exists():
        with open(target_ids_json) as f:
            target_ids = json.load(f)
        console.print(f"[cyan]•[/cyan] Loaded {len(target_ids)} target IDs from filter file")

    # Create dataset
    console.print(f"[cyan]•[/cyan] Loading graphs from: {graph_dir}")
    dataset = PocketGraphDataset(graph_dir, target_ids)

    if len(dataset) == 0:
        console.print("[red]✗[/red] No pocket graphs found!", style="bold red")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Found {len(dataset)} pocket graphs")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pocket_graphs,
        pin_memory=(device.type == 'cuda')
    )

    # Initialize model
    console.print("\n[bold yellow]Initializing Model[/bold yellow]")
    model = PocketEncoder(
        d_node=d_node,
        d_edge=d_edge,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        dropout=0.1,
        use_state_token=use_state_token,
        d_state=16,
        n_states=2  # agonist/antagonist
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
    console.print(f"[cyan]•[/cyan] State conditioning: {'Enabled' if use_state_token else 'Disabled'}")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    global_dir = output_dir / "global"
    global_dir.mkdir(exist_ok=True)

    if save_residue_embeddings:
        residues_dir = output_dir / "residues"
        residues_dir.mkdir(exist_ok=True)

    # Batch processing
    console.print("\n[bold yellow]Encoding Pockets[/bold yellow]")

    all_pocket_ids = []
    state_counts = {'agonist': 0, 'antagonist': 0}
    encoding_stats = {
        'n_pockets': 0,
        'n_residues_total': 0,
        'd_model': d_model,
        'has_residue_embeddings': save_residue_embeddings,
        'state_conditioning': use_state_token,
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
            f"Processing {len(dataset)} pockets...",
            total=len(dataloader)
        )

        for batch_idx, batch_data in enumerate(dataloader):
            # Encode batch
            results = encode_batch(model, batch_data, device)

            # Save individual pocket embeddings
            for z_poc, h_residues, target_id, state in zip(
                results['z_poc'],
                results['h_residues'],
                results['target_ids'],
                results['states']
            ):
                pocket_id = f"{target_id}__{state}"

                # Save global embedding
                torch.save(z_poc, global_dir / f"{pocket_id}.pt")

                # Save per-residue embeddings if requested
                if save_residue_embeddings:
                    torch.save(h_residues, residues_dir / f"{pocket_id}_residues.pt")

                all_pocket_ids.append(pocket_id)
                state_counts[state] = state_counts.get(state, 0) + 1
                encoding_stats['n_residues_total'] += h_residues.shape[0]

            encoding_stats['n_pockets'] += len(results['target_ids'])
            progress.update(task, advance=1)

    # Save metadata
    metadata = {
        'encoding_stats': encoding_stats,
        'state_counts': state_counts,
        'model_config': {
            'd_node': d_node,
            'd_edge': d_edge,
            'd_model': d_model,
            'n_layers': n_layers,
            'use_state_token': use_state_token,
        },
        'pocket_ids': all_pocket_ids,
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

    summary_table.add_row("Pockets encoded", f"{encoding_stats['n_pockets']:,}")
    summary_table.add_row("  └─ Agonist state", f"{state_counts.get('agonist', 0):,}")
    summary_table.add_row("  └─ Antagonist state", f"{state_counts.get('antagonist', 0):,}")
    summary_table.add_row("Total residues", f"{encoding_stats['n_residues_total']:,}")
    summary_table.add_row("Embedding dimension", str(d_model))
    summary_table.add_row("Avg residues per pocket", f"{encoding_stats['n_residues_total'] / encoding_stats['n_pockets']:.1f}")
    summary_table.add_row("State conditioning", "Enabled" if use_state_token else "Disabled")

    console.print(summary_table)
    console.print(f"\n[green]✓[/green] Saved metadata to: [cyan]{metadata_path}[/cyan]")

    if use_state_token:
        console.print("\n[bold cyan]ℹ State Conditioning:[/bold cyan]")
        console.print("  Each pocket was encoded with its corresponding state token:")
        console.print("  • Agonist structures → state_id=0")
        console.print("  • Antagonist structures → state_id=1")

    console.print()


if __name__ == "__main__":
    app()
