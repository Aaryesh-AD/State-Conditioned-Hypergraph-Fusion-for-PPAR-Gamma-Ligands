#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/build_dataset.py: Build graph dataset from ligands and pockets.

This script processes the entire PPAR-gamma dataset to create graph representations:
1. Loads ligand and pocket metadata from CSVs
2. Builds ligand atom graphs with pharmacophore hypergraphs
3. Builds pocket residue graphs with state conditioning
4. Saves processed graphs to disk (pickle format)

For optimal performance, first generate pocket selections using prep_main.py,
then provide the directory via --pockets-pkl. This avoids redundant computation.

Usage:
    python3 src/pprag/dataio/build_dataset.py \
        --ligands-csv Data/meta/ligands_clean.csv \
        --pockets-csv Data/meta/pockets.csv \
        --output-dir Output/graphs \
        --splits-dir Output/splits \
        --pockets-pkl Output/pockets

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""
import os
import sys
import traceback
import pickle
import json
import signal
from pathlib import Path
from typing import Dict, Optional, Set, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from pprag.dataio.load_labels import load_ligands_csv, load_target_csv
from pprag.dataio.schema import LigandRow, TargetRow, FeatureSpec, global_seed


class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


app = typer.Typer(
    name="build-dataset",
    help="Build graph dataset from PPAR-gamma ligands and pockets",
    add_completion=False,
)
console = Console()

SEED = global_seed()


def process_ligand(row: LigandRow, fe: FeatureSpec) -> Optional[Dict]:
    """Process a single ligand and return graph data."""
    try:
        from pprag.graphs.ligand_builder import build_ligand_graph as _build_ligand_graph
    except ImportError as ie:
        print(f"Import error in worker: {ie}", file=sys.stderr)
        print(f"sys.path: {sys.path}", file=sys.stderr)
        print(f"cwd: {os.getcwd()}", file=sys.stderr)
        return None

    try:
        graph = _build_ligand_graph(row, fe)
        if graph is None:
            return None

        # Convert to serializable dict
        graph_dict = {
            'ligand_id': graph.ligand_id,
            'x': graph.x,
            'edge_index': graph.edge_index,
            'edge_attr': graph.edge_attr,
            'pos': graph.pos,
            'props': graph.props,
            'hyperedge_members': graph.hyperedge_members,
            'hyperedge_attr': graph.hyperedge_attr,
            # Add metadata
            'smiles': row.smiles,
            'class_label': row.class_label,
            'is_decoy': row.is_decoy,
        }
        return graph_dict
    except Exception as e:
        print(f"Error processing ligand {row.ligand_id}: {e}", file=sys.stderr)

        traceback.print_exc()
        return None


def process_pocket(row: TargetRow, fe: FeatureSpec, pockets_pkl_dir: Optional[Path]) -> Optional[Dict]:
    """Process a single pocket and return graph data."""
    # Ensure imports work in worker process
    try:
        from pprag.graphs.pocket_builder import build_pocket_graph as _build_pocket_graph
        from pprag.dataio.pocket_select import build_pocket_select as _build_pocket_select
        from pprag.dataio.schema import STATE_TO_ID as _STATE_TO_ID
    except ImportError as ie:
        print(f"Import error in worker: {ie}", file=sys.stderr)
        return None

    try:
        # Try to load pre-computed pocket selection first
        ps = None
        if pockets_pkl_dir:
            pocket_file = pockets_pkl_dir / f"{row.target_id}__{row.state}.pkl"
            if pocket_file.exists():
                with open(pocket_file, 'rb') as f:
                    ps = pickle.load(f)
            else:
                print(f"Warning: Pre-computed pocket not found: {pocket_file}", file=sys.stderr)

        # If not found, build pocket selection on-the-fly
        if ps is None:
            ps = _build_pocket_select(
                target_id=row.target_id,
                protein_mol2=row.pdb_path,
                chain=None,  # Auto-detect chain
                radius=10.0,  # Default radius if building on-the-fly
                ligand_mol2=row.ligand_path,
                centers_json=None,
            )

        # Build pocket graph
        state_id = _STATE_TO_ID[row.state]
        graph = _build_pocket_graph(ps, state_id, fe)

        # Convert to serializable dict
        graph_dict = {
            'pocket_id': graph.target_id,  # PocketGraph uses target_id, not pocket_id
            'x': graph.x,
            'edge_index': graph.edge_index,
            'edge_attr': graph.edge_attr,
            'pos': graph.pos_ca,  # PocketGraph uses pos_ca, not pos
            'pos_sc': graph.pos_sc,  # Include side-chain positions
            'state_id': graph.state_id,
            # Add metadata
            'target_id': row.target_id,
            'state': row.state,
            'n_residues': len(ps.feats),
            'residue_ids': graph.residue_ids,
        }
        return graph_dict
    except Exception as e:
        print(f"Warning: Error processing pocket {row.target_id}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def load_split_ids(splits_dir: Path) -> Dict[str, Set[str]]:
    """Load train/val/test split IDs from JSON files."""
    splits = {}
    for split_name in ['train_ids', 'val_ids', 'test_ids']:
        split_file = splits_dir / f"{split_name}.json"
        if split_file.exists():
            with open(split_file) as f:
                splits[split_name.replace('_ids', '')] = set(json.load(f))
        else:
            console.print(f"[yellow]Warning: {split_file} not found, skipping split[/yellow]")
    return splits


@app.command()
def main(
    ligands_csv: Path = typer.Option(
        ...,
        "--ligands-csv",
        "-l",
        help="Path to ligands CSV file",
        exists=True,
    ),
    pockets_csv: Path = typer.Option(
        ...,
        "--pockets-csv",
        "-p",
        help="Path to pockets CSV file",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for processed graphs",
    ),
    splits_dir: Optional[Path] = typer.Option(
        None,
        "--splits-dir",
        "-s",
        help="Directory containing train/val/test split JSONs (optional)",
    ),
    pockets_pkl_dir: Optional[Path] = typer.Option(
        None,
        "--pockets-pkl",
        help="Directory containing pre-computed PocketSelect pickle files (recommended for speed)",
    ),
    radius: float = typer.Option(
        10.0,
        "--radius",
        "-r",
        help="Pocket selection radius in Angstroms (only used if building pockets on-the-fly)",
        min=3.0,
        max=20.0,
    ),
    n_workers: int = typer.Option(
        4,
        "--n-workers",
        "-w",
        help="Number of parallel workers",
        min=1,
        max=32,
    ),
    max_ligands: Optional[int] = typer.Option(
        None,
        "--max-ligands",
        help="Maximum number of ligands to process (for testing)",
    ),
    max_pockets: Optional[int] = typer.Option(
        None,
        "--max-pockets",
        help="Maximum number of pockets to process (for testing)",
    ),
) -> None:
    """
    Build graph dataset from PPAR-gamma ligands and pockets.

    This processes all ligands and pockets into graph representations ready
    for the GNN encoders. Graphs are saved as pickle files organized by split.

    For best performance, use the pre-computed pocket selections via --pockets-pkl
    (generated by prep_main.py). Otherwise, pockets will be computed on-the-fly.
    """
    console.print(Panel.fit(
        "[bold cyan]PPAR-Gamma Graph Dataset Builder[/bold cyan]\n"
        "Building ligand atom graphs with pharmacophore hypergraphs\n"
        "and pocket residue graphs with state conditioning",
        border_style="cyan"
    ))

    # Validate pockets_pkl_dir if provided
    if pockets_pkl_dir and not pockets_pkl_dir.exists():
        console.print(f"[yellow]Warning: Pockets directory not found: {pockets_pkl_dir}[/yellow]")
        console.print("[yellow]Will compute pocket selections on-the-fly (slower)[/yellow]\n")
        pockets_pkl_dir = None
    elif pockets_pkl_dir:
        console.print(f"[green]Using pre-computed pockets from: {pockets_pkl_dir}[/green]\n")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir = output_dir / "ligands"
    pockets_dir = output_dir / "pockets"
    ligands_dir.mkdir(exist_ok=True)
    pockets_dir.mkdir(exist_ok=True)

    # Load data
    console.print("\n[bold]Loading metadata...[/bold]")
    ligand_rows = load_ligands_csv(ligands_csv)
    pocket_rows = load_target_csv(pockets_csv)

    if max_ligands:
        ligand_rows = ligand_rows[:max_ligands]
    if max_pockets:
        pocket_rows = pocket_rows[:max_pockets]

    console.print(f"Loaded {len(ligand_rows)} ligands")
    console.print(f"Loaded {len(pocket_rows)} pockets")

    # Load splits if provided
    splits = {}
    if splits_dir and splits_dir.exists():
        console.print(f"\n[bold]Loading splits from {splits_dir}...[/bold]")
        splits = load_split_ids(splits_dir)
        for split_name, ids in splits.items():
            console.print(f"{split_name}: {len(ids)} ligands")

    # Define feature specification
    fe = FeatureSpec(
        d_lig_node=44,  # From get_atom_features
        d_lig_edge=6,   # From get_bond_features (without 3D distances)
        d_poc_node=30,  # From build_residue_features
        d_poc_edge=19,  # From build_edge_features
        use_pharmacophore_hypergraph=True,
        use_3d_distances=False,
        hyperedge_types=(
            "ring", "donor_group", "acceptor_group",
            "cation_center", "anion_center", "halogen_donor",
            "aromatic_cluster", "rigid_fragment", "flexible_linker"
        ),
    )

    # Process Ligands
    console.print(f"\n[bold green]Processing {len(ligand_rows)} ligands...[/bold green]")

    # Force sequential processing due to RDKit segfaults in worker processes
    if n_workers > 1:
        console.print("[yellow] Warning: Old impl used Multiprocessing, but can cause worker crashes due to RDKit C-level segfaults.[/yellow]")
        console.print("[yellow] Forcing sequential mode (n_workers=1) for stability.[/yellow]")
        console.print("[yellow] This will be slower and take time[/yellow]\n")
        n_workers = 1

    ligand_graphs: List[Dict[str, Any]] = []
    ligand_stats: Dict[str, Any] = {
        'success': 0,
        'failed': 0,
        'by_split': {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Building ligand graphs...", total=len(ligand_rows))

        if n_workers == 1:
            # Sequential processing (safer, easier to debug) so RDKit doesn't segfault and hence do it instead
            console.print("[yellow]Running in sequential mode (n_workers=1)[/yellow]")
            from pprag.graphs.ligand_builder import build_ligand_graph as _build_ligand_graph

            # Track progress and save checkpoint
            checkpoint_file = output_dir / ".build_checkpoint.txt"
            start_idx = 0

            # Resume from checkpoint if exists
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    start_idx = int(f.read().strip())
                console.print(f"[cyan]Resuming from ligand {start_idx}[/cyan]")
                console.print(f"[dim]The previous run crashed on ligand: {ligand_rows[start_idx - 1].ligand_id if start_idx > 0 else 'unknown'}[/dim]")
                console.print("[dim]Skipping that molecule and continuing...[/dim]\n")

            for idx, row in enumerate(ligand_rows):
                if idx < start_idx:
                    progress.update(task, advance=1)
                    continue

                # Save checkpoint before processing (so we can resume after crash)
                with open(checkpoint_file, 'w') as f:
                    f.write(str(idx))

                # Set up signal handler for timeouts
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout per molecule

                try:
                    graph = _build_ligand_graph(row, fe)
                    signal.alarm(0)  # Cancel alarm

                    if graph is not None:
                        ligand_result = {
                            'ligand_id': graph.ligand_id,
                            'x': graph.x,
                            'edge_index': graph.edge_index,
                            'edge_attr': graph.edge_attr,
                            'pos': graph.pos,
                            'props': graph.props,
                            'hyperedge_members': graph.hyperedge_members,
                            'hyperedge_attr': graph.hyperedge_attr,
                            'smiles': row.smiles,
                            'class_label': row.class_label,
                            'is_decoy': row.is_decoy,
                        }
                        ligand_graphs.append(ligand_result)
                        ligand_stats['success'] += 1

                        # Track by split
                        ligand_id = ligand_result['ligand_id']
                        found_split = False
                        for split_name, split_ids in splits.items():
                            if ligand_id in split_ids:
                                ligand_stats['by_split'][split_name] += 1
                                found_split = True
                                break
                        if not found_split:
                            ligand_stats['by_split']['unknown'] += 1
                    else:
                        ligand_stats['failed'] += 1
                except (Exception, TimeoutError) as e:
                    signal.alarm(0)  # Cancel alarm
                    console.print(f"[red]Error processing {row.ligand_id}: {e}[/red]")
                    ligand_stats['failed'] += 1

                progress.update(task, advance=1)

            # Clean up checkpoint file when done
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        else:
            console.print(f"[cyan]Running in parallel mode with {n_workers} workers[/cyan]")
            try:
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    ligand_futures: Dict[Any, LigandRow] = {
                        executor.submit(process_ligand, row, fe): row
                        for row in ligand_rows
                    }

                    for future in as_completed(ligand_futures):
                        row = ligand_futures[future]
                        try:
                            ligand_result_parallel: Optional[Dict[str, Any]] = future.result()

                            if ligand_result_parallel is not None:
                                ligand_graphs.append(ligand_result_parallel)
                                ligand_stats['success'] += 1

                                # Track by split
                                ligand_id = ligand_result_parallel['ligand_id']
                                found_split = False
                                for split_name, split_ids in splits.items():
                                    if ligand_id in split_ids:
                                        ligand_stats['by_split'][split_name] += 1
                                        found_split = True
                                        break
                                if not found_split:
                                    ligand_stats['by_split']['unknown'] += 1
                            else:
                                ligand_stats['failed'] += 1
                        except Exception as e:
                            console.print(f"[red]Error processing {row.ligand_id}: {e}[/red]")
                            ligand_stats['failed'] += 1

                        progress.update(task, advance=1)
            except Exception as e:
                console.print(f"[red]Fatal error in parallel processing: {e}[/red]")
                console.print("[yellow]Try running with --n-workers 1 for debugging[/yellow]")
                raise

    # Save ligand graphs
    console.print(f"\n[bold]Saving {len(ligand_graphs)} ligand graphs...[/bold]")

    # Save all ligands
    all_ligands_file = ligands_dir / "all_ligands.pkl"
    with open(all_ligands_file, 'wb') as f:
        pickle.dump(ligand_graphs, f)
    console.print(f" Saved all ligands to {all_ligands_file}")

    # Save by split if available
    if splits:
        for split_name, split_ids in splits.items():
            split_graphs = [g for g in ligand_graphs if g['ligand_id'] in split_ids]
            split_file = ligands_dir / f"{split_name}_ligands.pkl"
            with open(split_file, 'wb') as f:
                pickle.dump(split_graphs, f)
            console.print(f" Saved {len(split_graphs)} {split_name} ligands to {split_file}")

    # Process Pockets
    console.print(f"\n[bold green]Processing {len(pocket_rows)} pockets...[/bold green]")

    pocket_graphs: List[Dict[str, Any]] = []
    pocket_stats: Dict[str, Any] = {
        'success': 0,
        'failed': 0,
        'by_state': {'agonist': 0, 'antagonist': 0}
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Building pocket graphs...", total=len(pocket_rows))

        # Process in parallel (fewer workers due to MDAnalysis memory usage)
        with ProcessPoolExecutor(max_workers=max(1, n_workers // 2)) as executor:
            pocket_futures: Dict[Any, TargetRow] = {
                executor.submit(process_pocket, row, fe, pockets_pkl_dir): row
                for row in pocket_rows
            }

            for future in as_completed(pocket_futures):
                pocket_row = pocket_futures[future]     # noqa
                pocket_result: Optional[Dict[str, Any]] = future.result()

                if pocket_result is not None:
                    pocket_graphs.append(pocket_result)
                    pocket_stats['success'] += 1
                    pocket_stats['by_state'][pocket_result['state']] += 1
                else:
                    pocket_stats['failed'] += 1

                progress.update(task, advance=1)

    # Save pocket graphs
    console.print(f"\n[bold]Saving {len(pocket_graphs)} pocket graphs...[/bold]")

    # Save all pockets
    all_pockets_file = pockets_dir / "all_pockets.pkl"
    with open(all_pockets_file, 'wb') as f:
        pickle.dump(pocket_graphs, f)
    console.print(f"  Saved all pockets to {all_pockets_file}")

    # Save by state
    for state in ['agonist', 'antagonist']:
        state_graphs = [g for g in pocket_graphs if g['state'] == state]
        state_file = pockets_dir / f"{state}_pockets.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(state_graphs, f)
        console.print(f"   Saved {len(state_graphs)} {state} pockets to {state_file}")

    # Save metadata
    metadata = {
        'feature_spec': asdict(fe),
        'radius': radius,
        'n_ligands_total': len(ligand_rows),
        'n_ligands_processed': ligand_stats['success'],
        'n_pockets_total': len(pocket_rows),
        'n_pockets_processed': pocket_stats['success'],
        'ligand_stats': ligand_stats,
        'pocket_stats': pocket_stats,
    }

    metadata_file = output_dir / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    console.print(f"\n   Saved metadata to {metadata_file}")

    # Print summary table
    console.print("\n[bold cyan]Dataset Build Summary[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")

    table.add_row("Ligands", str(ligand_stats['success']), str(ligand_stats['failed']))
    table.add_row("Pockets", str(pocket_stats['success']), str(pocket_stats['failed']))

    console.print(table)

    if splits:
        console.print("\n[bold cyan]Ligands by Split[/bold cyan]")
        split_table = Table(show_header=True, header_style="bold magenta")
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Count", justify="right", style="green")

        for split_name, count in ligand_stats['by_split'].items():
            if count > 0:
                split_table.add_row(split_name.capitalize(), str(count))

        console.print(split_table)

    console.print("\n[bold green]✓ Dataset build complete![/bold green]")
    console.print(f"[dim]Output directory: {output_dir.absolute()}[/dim]")


if __name__ == "__main__":
    app()
