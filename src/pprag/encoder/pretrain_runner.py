#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
encoder/pretrain_runner.py: CLI for contrastive pretraining of ligand and pocket encoders.

This script implements self-supervised pretraining using two strategies:

1. GraphCL (Graph Contrastive Learning):
   - For ligand encoders
   - Creates augmented views via node dropping, edge perturbation, feature masking
   - Uses InfoNCE loss to learn invariant representations

2. State-Aware Contrastive Learning:
   - For pocket encoders
   - Learns state-specific representations (agonist vs antagonist)
   - Encourages same-state similarity and cross-state discrimination

Features:
- Flexible augmentation strategies
- Learning rate scheduling with warmup
- Checkpoint saving and resumption
- Comprehensive logging with Rich console output
- Gradient accumulation for large batch sizes
- Mixed precision training (optional)

Author: Aaryesh Deshpande
Last Modified: 10/28/2025
"""

import pickle
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from pprag.encoder.ligand_encoder import LigandEncoder
from pprag.encoder.pocket_encoder import PocketEncoder
from pprag.encoder.pretrain import (
    ContrastivePretrainer,
    ContrastiveConfig,
)
from pprag.dataio.schema import LigandGraph, PocketGraph

# Legacy module mapping for pickle compatibility
_LEGACY_MODULE_MAP = {
    "schema": "pprag.dataio.schema",
    "dataio.schema": "pprag.dataio.schema",
    "pprag.schema": "pprag.dataio.schema",
}


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = _LEGACY_MODULE_MAP.get(module, module)
        return super().find_class(module, name)


def _compat_pickle_load(f):
    try:
        return pickle.load(f)
    except ModuleNotFoundError:
        f.seek(0)
        return _CompatUnpickler(f).load()


app = typer.Typer(
    name="pretrain",
    help="Self-supervised contrastive pretraining for ligand and pocket encoders",
    add_completion=False,
)
console = Console()


# DATASET CLASSES
class LigandPretrainDataset(Dataset):
    """Dataset for ligand contrastive pretraining."""

    def __init__(self, graph_dir: Path, ligand_ids: Optional[List[str]] = None):
        self.graph_dir = graph_dir

        if ligand_ids:
            self.ligand_files = [graph_dir / f"{lid}.pkl" for lid in ligand_ids]
        else:
            self.ligand_files = sorted(graph_dir.glob("*.pkl"))

        self.ligand_files = [f for f in self.ligand_files if f.exists()]

    def __len__(self) -> int:
        return len(self.ligand_files)

    def __getitem__(self, idx: int) -> LigandGraph:
        pkl_path = self.ligand_files[idx]

        with open(pkl_path, 'rb') as f:
            data = _compat_pickle_load(f)

        # Handle different pickle formats
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Empty list in pickle: {pkl_path}")
            elif len(data) == 1:
                lig_graph = data[0]
            else:
                lig_graph = None
                for item in data:
                    if isinstance(item, LigandGraph):
                        lig_graph = item
                        break
                if lig_graph is None:
                    raise TypeError(f"No LigandGraph found in {pkl_path}")
        elif isinstance(data, LigandGraph):
            lig_graph = data
        else:
            raise TypeError(f"Expected LigandGraph, got {type(data).__name__}")

        if not isinstance(lig_graph, LigandGraph):
            raise TypeError(f"Expected LigandGraph, got {type(lig_graph)}")

        return lig_graph


class PocketPretrainDataset(Dataset):
    """Dataset for pocket contrastive pretraining with state labels."""

    def __init__(self, graph_dir: Path, target_ids: Optional[List[str]] = None):
        self.graph_dir = graph_dir

        if target_ids:
            self.pocket_files = []
            for tid in target_ids:
                for state in ["agonist", "antagonist"]:
                    pkl_path = graph_dir / f"{tid}__{state}.pkl"
                    if pkl_path.exists():
                        self.pocket_files.append(pkl_path)
        else:
            self.pocket_files = sorted(graph_dir.glob("*.pkl"))

        self.pocket_files = [f for f in self.pocket_files if f.exists()]

    def __len__(self) -> int:
        return len(self.pocket_files)

    def __getitem__(self, idx: int) -> Tuple[PocketGraph, int]:
        pkl_path = self.pocket_files[idx]

        with open(pkl_path, 'rb') as f:
            data = _compat_pickle_load(f)

        # Handle different pickle formats
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Empty list in pickle: {pkl_path}")
            elif len(data) == 1:
                poc_graph = data[0]
            else:
                poc_graph = None
                for item in data:
                    if isinstance(item, PocketGraph):
                        poc_graph = item
                        break
                if poc_graph is None:
                    raise TypeError(f"No PocketGraph found in {pkl_path}")
        elif isinstance(data, PocketGraph):
            poc_graph = data
        else:
            raise TypeError(f"Expected PocketGraph, got {type(data).__name__}")

        if not isinstance(poc_graph, PocketGraph):
            raise TypeError(f"Expected PocketGraph, got {type(poc_graph)}")

        return poc_graph, poc_graph.state_id


# COLLATE FUNCTIONS
def collate_ligand_graphs(batch: List[LigandGraph]) -> dict:
    """Collate ligand graphs for contrastive learning."""
    x_list = [torch.from_numpy(g.x).float() for g in batch]
    pos_list = [torch.from_numpy(g.pos).float() for g in batch]

    edge_index_list = []
    edge_attr_list = []
    node_offset = 0

    for g in batch:
        edge_index = torch.from_numpy(g.edge_index).long()
        edge_attr = torch.from_numpy(g.edge_attr).float()
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        node_offset += g.x.shape[0]

    x = torch.cat(x_list, dim=0)
    pos = torch.cat(pos_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, batch[0].edge_attr.shape[1]))

    batch_tensor = torch.cat([torch.full((len(x_list[i]),), i, dtype=torch.long)
                              for i in range(len(batch))])

    # Hypergraph data (optional)
    hyperedge_attr_list = []
    hyperedge_members_batched = []
    node_offset = 0

    for g in batch:
        if g.hyperedge_attr is not None and g.hyperedge_members is not None:
            hyperedge_attr_list.append(torch.from_numpy(g.hyperedge_attr).float())
            for members in g.hyperedge_members:
                adjusted_members = [m + node_offset for m in members]
                hyperedge_members_batched.append(adjusted_members)
            node_offset += g.x.shape[0]

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
    }


def collate_pocket_graphs(batch: List[Tuple[PocketGraph, int]]) -> dict:
    """Collate pocket graphs with state labels."""
    graphs, states = zip(*batch)

    x_list = [torch.from_numpy(g.x).float() for g in graphs]
    pos_ca_list = [torch.from_numpy(g.pos_ca).float() for g in graphs]

    edge_index_list = []
    edge_attr_list = []
    node_offset = 0

    for g in graphs:
        edge_index = torch.from_numpy(g.edge_index).long()
        edge_attr = torch.from_numpy(g.edge_attr).float()
        edge_index_list.append(edge_index + node_offset)
        edge_attr_list.append(edge_attr)
        node_offset += g.x.shape[0]

    x = torch.cat(x_list, dim=0)
    pos_ca = torch.cat(pos_ca_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, graphs[0].edge_attr.shape[1]))

    batch_tensor = torch.cat([torch.full((len(x_list[i]),), i, dtype=torch.long)
                              for i in range(len(graphs))])

    states_tensor = torch.tensor(states, dtype=torch.long)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'pos_ca': pos_ca,
        'batch': batch_tensor,
        'states': states_tensor,
    }


# TRAINING FUNCTIONS
def train_ligand_epoch(
    model: ContrastivePretrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aug_type1: str = 'random',
    aug_type2: str = 'random',
    grad_accum_steps: int = 1,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train one epoch of ligand contrastive learning."""
    model.train()
    epoch_loss = 0.0
    epoch_pos_sim = 0.0
    n_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, batch_data in enumerate(dataloader):
        # Move to device
        x = batch_data['x'].to(device)
        edge_index = batch_data['edge_index'].to(device)
        edge_attr = batch_data['edge_attr'].to(device)
        pos = batch_data['pos'].to(device)
        batch_tensor = batch_data['batch'].to(device)
        hyperedge_attr = batch_data['hyperedge_attr'].to(device) if batch_data['hyperedge_attr'] is not None else None
        hyperedge_members = batch_data['hyperedge_members']

        # Mixed precision forward
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, metrics = model.graphcl_loss(
                    x, edge_index, edge_attr,
                    pos=pos,
                    hyperedge_attr=hyperedge_attr,
                    hyperedge_members=hyperedge_members,
                    batch=batch_tensor,
                    aug_type1=aug_type1,
                    aug_type2=aug_type2,
                )
                loss = loss / grad_accum_steps
        else:
            loss, metrics = model.graphcl_loss(
                x, edge_index, edge_attr,
                pos=pos,
                hyperedge_attr=hyperedge_attr,
                hyperedge_members=hyperedge_members,
                batch=batch_tensor,
                aug_type1=aug_type1,
                aug_type2=aug_type2,
            )
            loss = loss / grad_accum_steps

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss += metrics['loss']
        epoch_pos_sim += metrics['pos_similarity']
        n_batches += 1

    return {
        'loss': epoch_loss / n_batches,
        'pos_similarity': epoch_pos_sim / n_batches,
    }


def train_pocket_epoch(
    model: ContrastivePretrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train one epoch of pocket state-aware contrastive learning."""
    model.train()
    epoch_loss = 0.0
    epoch_ago_sim = 0.0
    epoch_ant_sim = 0.0
    epoch_cross_sim = 0.0
    n_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, batch_data in enumerate(dataloader):
        # Move to device
        x = batch_data['x'].to(device)
        edge_index = batch_data['edge_index'].to(device)
        edge_attr = batch_data['edge_attr'].to(device)
        batch_tensor = batch_data['batch'].to(device)
        states = batch_data['states'].to(device)

        # Mixed precision forward
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, metrics = model.state_contrastive_loss(
                    x, edge_index, edge_attr,
                    states=states,
                    batch=batch_tensor,
                )
                loss = loss / grad_accum_steps
        else:
            loss, metrics = model.state_contrastive_loss(
                x, edge_index, edge_attr,
                states=states,
                batch=batch_tensor,
            )
            loss = loss / grad_accum_steps

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss += metrics['loss']
        epoch_ago_sim += metrics['agonist_similarity']
        epoch_ant_sim += metrics['antagonist_similarity']
        epoch_cross_sim += metrics['cross_state_similarity']
        n_batches += 1

    return {
        'loss': epoch_loss / n_batches,
        'agonist_similarity': epoch_ago_sim / n_batches,
        'antagonist_similarity': epoch_ant_sim / n_batches,
        'cross_state_similarity': epoch_cross_sim / n_batches,
    }


# CLI
@app.command(name="ligand")
def train_ligand_pretraining(
    graph_dir: Path = typer.Option(
        ...,
        "--graph-dir",
        "-g",
        help="Directory containing ligand graph pickle files",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for checkpoints and logs",
    ),
    ligand_ids_json: Optional[Path] = typer.Option(
        None,
        "--ligand-ids",
        "-l",
        help="JSON file with ligand IDs for training (optional - uses all if not provided)",
        exists=False,
    ),
    n_epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
        min=1,
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size",
        min=1,
    ),
    learning_rate: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help="Temperature for contrastive loss",
    ),
    projection_dim: int = typer.Option(
        128,
        "--projection-dim",
        help="Projection head output dimension",
    ),
    aug_type1: str = typer.Option(
        "random",
        "--aug1",
        help="Augmentation type for view 1: random, node_drop, edge_pert, feat_mask, all",
    ),
    aug_type2: str = typer.Option(
        "random",
        "--aug2",
        help="Augmentation type for view 2: random, node_drop, edge_pert, feat_mask, all",
    ),
    warmup_epochs: int = typer.Option(
        10,
        "--warmup",
        help="Number of warmup epochs",
    ),
    grad_accum_steps: int = typer.Option(
        1,
        "--grad-accum",
        help="Gradient accumulation steps",
        min=1,
    ),
    checkpoint_every: int = typer.Option(
        10,
        "--checkpoint-every",
        help="Save checkpoint every N epochs",
        min=1,
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from checkpoint",
        exists=False,
    ),
    use_amp: bool = typer.Option(
        False,
        "--amp/--no-amp",
        help="Use automatic mixed precision training",
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="DataLoader workers",
        min=0,
    ),
    # Model architecture
    d_node: int = typer.Option(44, "--d-node", help="Node feature dimension"),
    d_edge: int = typer.Option(6, "--d-edge", help="Edge feature dimension"),
    d_hyper_feat: int = typer.Option(15, "--d-hyper-feat", help="Hyperedge feature dimension"),
    d_model: int = typer.Option(256, "--d-model", help="Model hidden dimension"),
    n_gnn_layers: int = typer.Option(3, "--n-gnn-layers", help="Number of GNN layers"),
    n_hyper_layers: int = typer.Option(2, "--n-hyper-layers", help="Number of hypergraph layers"),
) -> None:
    """
    Pretrain ligand encoder using GraphCL (Graph Contrastive Learning).

    This trains the encoder to learn robust molecular representations by:
    - Creating augmented views of molecules (node dropping, edge perturbation, etc.)
    - Maximizing agreement between views of the same molecule
    - Minimizing agreement between different molecules

    Example:
    \b
        python3 -m src/pprag/encoder/pretrain_runner ligand \\
            --graph-dir Output/graphs/ligands \\
            --output-dir Output/pretrain/ligand \\
            --epochs 100 \\
            --batch-size 32 \\
            --lr 1e-4 \\
            --temperature 0.1 \\
            --aug1 random \\
            --aug2 random \\
            --gpu
    """
    console.print("\n[bold cyan]Ligand Encoder Pretraining (GraphCL)[/bold cyan]\n")

    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        console.print(f"[green]✓[/green] Using GPU: {torch.cuda.get_device_name(0)}")
        if use_amp:
            console.print("[green]✓[/green] Mixed precision training enabled")
    else:
        device = torch.device('cpu')
        console.print("[yellow]⚠[/yellow] Using CPU (training will be slow)")
        if use_amp:
            console.print("[yellow]⚠[/yellow] AMP disabled on CPU")
            use_amp = False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Load dataset
    console.print(f"\n[cyan]•[/cyan] Loading dataset from: {graph_dir}")
    ligand_ids = None
    if ligand_ids_json and ligand_ids_json.exists():
        with open(ligand_ids_json) as f:
            ligand_ids = json.load(f)
        console.print(f"[cyan]•[/cyan] Using {len(ligand_ids)} ligands from filter file")

    dataset = LigandPretrainDataset(graph_dir, ligand_ids)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} ligands")

    if len(dataset) == 0:
        console.print("[red]✗[/red] No ligands found!", style="bold red")
        raise typer.Exit(code=1)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_ligand_graphs,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,  # Drop incomplete batches for contrastive learning
    )

    # Initialize model
    console.print("\n[bold yellow]Initializing Model[/bold yellow]")
    encoder = LigandEncoder(
        d_node=d_node,
        d_edge=d_edge,
        d_hyper_feat=d_hyper_feat,
        d_model=d_model,
        n_gnn_layers=n_gnn_layers,
        n_hyper_layers=n_hyper_layers,
        n_heads=4,
        dropout=0.1,
    )

    config = ContrastiveConfig(
        temperature=temperature,
        projection_dim=projection_dim,
        use_projection_head=True,
        normalize_embeddings=True,
    )
    model = ContrastivePretrainer(encoder, config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[cyan]•[/cyan] Model parameters: {n_params:,}")
    console.print(f"[cyan]•[/cyan] Encoder d_model: {d_model}")
    console.print(f"[cyan]•[/cyan] Projection dim: {projection_dim}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate schedule: warmup + cosine decay
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs, eta_min=learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Resume from checkpoint
    start_epoch = 0
    if resume_from and resume_from.exists():
        console.print(f"\n[cyan]•[/cyan] Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        console.print(f"[green]✓[/green] Resumed from epoch {start_epoch}")

    # Training loop
    console.print("\n[bold yellow]Training[/bold yellow]")
    console.print(f"[cyan]•[/cyan] Epochs: {n_epochs}")
    console.print(f"[cyan]•[/cyan] Batch size: {batch_size}")
    console.print(f"[cyan]•[/cyan] Learning rate: {learning_rate}")
    console.print(f"[cyan]•[/cyan] Temperature: {temperature}")
    console.print(f"[cyan]•[/cyan] Augmentations: {aug_type1}, {aug_type2}")
    console.print(f"[cyan]•[/cyan] Gradient accumulation: {grad_accum_steps} steps\n")

    training_log = []

    for epoch in range(start_epoch, n_epochs):
        # Train one epoch
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Epoch {epoch + 1}/{n_epochs}",
                total=len(dataloader)
            )

            metrics = train_ligand_epoch(
                model, dataloader, optimizer, device,
                aug_type1=aug_type1,
                aug_type2=aug_type2,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
            )

            progress.update(task, completed=len(dataloader))

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        log_entry = {
            'epoch': epoch + 1,
            'loss': metrics['loss'],
            'pos_similarity': metrics['pos_similarity'],
            'learning_rate': current_lr,
        }
        training_log.append(log_entry)

        # Print epoch summary
        console.print(
            f"[green]✓[/green] Epoch {epoch + 1}: "
            f"Loss={metrics['loss']:.4f}, "
            f"PosSim={metrics['pos_similarity']:.4f}, "
            f"LR={current_lr:.2e}"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == n_epochs:
            ckpt_path = ckpt_dir / f"ligand_pretrain_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),  # Save encoder separately
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'd_node': d_node,
                    'd_edge': d_edge,
                    'd_hyper_feat': d_hyper_feat,
                    'd_model': d_model,
                    'n_gnn_layers': n_gnn_layers,
                    'n_hyper_layers': n_hyper_layers,
                    'temperature': temperature,
                    'projection_dim': projection_dim,
                },
                'metrics': log_entry,
            }, ckpt_path)
            console.print(f"[cyan]•[/cyan] Saved checkpoint: {ckpt_path.name}")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    # Save final encoder only
    encoder_path = output_dir / "ligand_encoder_pretrained.pt"
    torch.save(model.encoder.state_dict(), encoder_path)

    console.print("\n[bold green]✓ Pretraining Complete![/bold green]")
    console.print(f"[green]✓[/green] Saved encoder to: [cyan]{encoder_path}[/cyan]")
    console.print(f"[green]✓[/green] Training log: [cyan]{log_path}[/cyan]\n")


@app.command(name="pocket")
def train_pocket_pretraining(
    graph_dir: Path = typer.Option(
        ...,
        "--graph-dir",
        "-g",
        help="Directory containing pocket graph pickle files",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for checkpoints and logs",
    ),
    target_ids_json: Optional[Path] = typer.Option(
        None,
        "--target-ids",
        "-l",
        help="JSON file with target IDs for training (optional - uses all if not provided)",
        exists=False,
    ),
    n_epochs: int = typer.Option(
        100,
        "--epochs",
        "-e",
        help="Number of training epochs",
        min=1,
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Batch size (smaller for pockets due to size)",
        min=1,
    ),
    learning_rate: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help="Temperature for contrastive loss",
    ),
    projection_dim: int = typer.Option(
        128,
        "--projection-dim",
        help="Projection head output dimension",
    ),
    warmup_epochs: int = typer.Option(
        10,
        "--warmup",
        help="Number of warmup epochs",
    ),
    grad_accum_steps: int = typer.Option(
        2,
        "--grad-accum",
        help="Gradient accumulation steps",
        min=1,
    ),
    checkpoint_every: int = typer.Option(
        10,
        "--checkpoint-every",
        help="Save checkpoint every N epochs",
        min=1,
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from checkpoint",
        exists=False,
    ),
    use_amp: bool = typer.Option(
        False,
        "--amp/--no-amp",
        help="Use automatic mixed precision training",
    ),
    use_gpu: bool = typer.Option(
        True,
        "--gpu/--cpu",
        help="Use GPU if available",
    ),
    num_workers: int = typer.Option(
        4,
        "--num-workers",
        "-w",
        help="DataLoader workers",
        min=0,
    ),
    # Model architecture
    d_node: int = typer.Option(30, "--d-node", help="Node feature dimension"),
    d_edge: int = typer.Option(19, "--d-edge", help="Edge feature dimension"),
    d_model: int = typer.Option(256, "--d-model", help="Model hidden dimension"),
    n_layers: int = typer.Option(4, "--n-layers", help="Number of GNN layers"),
    use_state_token: bool = typer.Option(True, "--state-token/--no-state-token", help="Use state conditioning"),
) -> None:
    """
    Pretrain pocket encoder using state-aware contrastive learning.

    This trains the encoder to learn state-specific pocket representations by:
    - Pulling same-state pockets together (agonist with agonist, antagonist with antagonist)
    - Pushing different-state pockets apart
    - Learning functional differences between conformational states

    Example:
    \b
        python3 -m src/pprag/encoder/pretrain_runner pocket \\
            --graph-dir Output/graphs/pockets \\
            --output-dir Output/pretrain/pocket \\
            --epochs 100 \\
            --batch-size 16 \\
            --lr 1e-4 \\
            --temperature 0.1 \\
            --gpu
    """
    console.print("\n[bold cyan]Pocket Encoder Pretraining (State-Aware)[/bold cyan]\n")

    # Setup device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        console.print(f"[green]✓[/green] Using GPU: {torch.cuda.get_device_name(0)}")
        if use_amp:
            console.print("[green]✓[/green] Mixed precision training enabled")
    else:
        device = torch.device('cpu')
        console.print("[yellow]⚠[/yellow] Using CPU (training will be slow)")
        if use_amp:
            console.print("[yellow]⚠[/yellow] AMP disabled on CPU")
            use_amp = False

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Load dataset
    console.print(f"\n[cyan]•[/cyan] Loading dataset from: {graph_dir}")
    target_ids = None
    if target_ids_json and target_ids_json.exists():
        with open(target_ids_json) as f:
            target_ids = json.load(f)
        console.print(f"[cyan]•[/cyan] Using {len(target_ids)} targets from filter file")

    dataset = PocketPretrainDataset(graph_dir, target_ids)
    console.print(f"[green]✓[/green] Loaded {len(dataset)} pockets")

    if len(dataset) == 0:
        console.print("[red]✗[/red] No pockets found!", style="bold red")
        raise typer.Exit(code=1)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pocket_graphs,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # Initialize model
    console.print("\n[bold yellow]Initializing Model[/bold yellow]")
    encoder = PocketEncoder(
        d_node=d_node,
        d_edge=d_edge,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        dropout=0.1,
        use_state_token=use_state_token,
        d_state=16,
    )

    config = ContrastiveConfig(
        temperature=temperature,
        projection_dim=projection_dim,
        use_projection_head=True,
        normalize_embeddings=True,
    )
    model = ContrastivePretrainer(encoder, config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[cyan]•[/cyan] Model parameters: {n_params:,}")
    console.print(f"[cyan]•[/cyan] Encoder d_model: {d_model}")
    console.print(f"[cyan]•[/cyan] Projection dim: {projection_dim}")
    console.print(f"[cyan]•[/cyan] State conditioning: {use_state_token}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs, eta_min=learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Resume from checkpoint
    start_epoch = 0
    if resume_from and resume_from.exists():
        console.print(f"\n[cyan]•[/cyan] Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        console.print(f"[green]✓[/green] Resumed from epoch {start_epoch}")

    # Training loop
    console.print("\n[bold yellow]Training[/bold yellow]")
    console.print(f"[cyan]•[/cyan] Epochs: {n_epochs}")
    console.print(f"[cyan]•[/cyan] Batch size: {batch_size}")
    console.print(f"[cyan]•[/cyan] Learning rate: {learning_rate}")
    console.print(f"[cyan]•[/cyan] Temperature: {temperature}")
    console.print(f"[cyan]•[/cyan] Gradient accumulation: {grad_accum_steps} steps\n")

    training_log = []

    for epoch in range(start_epoch, n_epochs):
        # Train one epoch
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Epoch {epoch + 1}/{n_epochs}",
                total=len(dataloader)
            )

            metrics = train_pocket_epoch(
                model, dataloader, optimizer, device,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
            )

            progress.update(task, completed=len(dataloader))

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        log_entry = {
            'epoch': epoch + 1,
            'loss': metrics['loss'],
            'agonist_similarity': metrics['agonist_similarity'],
            'antagonist_similarity': metrics['antagonist_similarity'],
            'cross_state_similarity': metrics['cross_state_similarity'],
            'learning_rate': current_lr,
        }
        training_log.append(log_entry)

        # Print epoch summary
        console.print(
            f"[green]✓[/green] Epoch {epoch + 1}: "
            f"Loss={metrics['loss']:.4f}, "
            f"AgoSim={metrics['agonist_similarity']:.4f}, "
            f"AntSim={metrics['antagonist_similarity']:.4f}, "
            f"CrossSim={metrics['cross_state_similarity']:.4f}, "
            f"LR={current_lr:.2e}"
        )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == n_epochs:
            ckpt_path = ckpt_dir / f"pocket_pretrain_epoch{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'd_node': d_node,
                    'd_edge': d_edge,
                    'd_model': d_model,
                    'n_layers': n_layers,
                    'use_state_token': use_state_token,
                    'temperature': temperature,
                    'projection_dim': projection_dim,
                },
                'metrics': log_entry,
            }, ckpt_path)
            console.print(f"[cyan]•[/cyan] Saved checkpoint: {ckpt_path.name}")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    # Save final encoder only
    encoder_path = output_dir / "pocket_encoder_pretrained.pt"
    torch.save(model.encoder.state_dict(), encoder_path)

    console.print("\n[bold green]✓ Pretraining Complete![/bold green]")
    console.print(f"[green]✓[/green] Saved encoder to: [cyan]{encoder_path}[/cyan]")
    console.print(f"[green]✓[/green] Training log: [cyan]{log_path}[/cyan]\n")


if __name__ == "__main__":
    app()
