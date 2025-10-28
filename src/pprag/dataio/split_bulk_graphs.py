#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utility script -> For encoding efficiency
dataio/split_bulk_graphs.py: Convert bulk graph pickle files to individual files.

This script takes the bulk pickle files created by build_dataset.py
(all_pockets.pkl, all_ligands.pkl, etc.) and splits them into individual
pickle files that can be loaded by the encoder runners.

For pockets: Creates {target_id}__{state}.pkl files containing PocketGraph objects
For ligands: Creates {ligand_id}.pkl files containing LigandGraph objects

Author: Aaryesh Deshpande
Last Modified: 10/27/2025
"""

import pickle
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pprag.dataio.schema import PocketGraph, LigandGraph


app = typer.Typer(
    name="split-bulk-graphs",
    help="Split bulk pickle files into individual graph files",
    add_completion=False,
)
console = Console()


def dict_to_pocket_graph(data: dict) -> PocketGraph:
    """Convert dictionary representation to PocketGraph object."""
    return PocketGraph(
        target_id=data['target_id'],
        state_id=data['state_id'],
        x=data['x'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        pos_ca=data['pos'],  # Dictionary uses 'pos' key
        pos_sc=data.get('pos_sc'),
        residue_ids=data['residue_ids']
    )


def dict_to_ligand_graph(data: dict) -> LigandGraph:
    """Convert dictionary representation to LigandGraph object."""
    return LigandGraph(
        ligand_id=data['ligand_id'],
        x=data['x'],
        edge_index=data['edge_index'],
        edge_attr=data['edge_attr'],
        pos=data['pos'],
        props=data['props'],
        hyperedge_members=data.get('hyperedge_members'),
        hyperedge_attr=data.get('hyperedge_attr')
    )


def split_pockets_internal(bulk_file: Path, output_dir: Path, overwrite: bool) -> None:
    """Internal function to split pocket pickle files."""
    console.print("\n[bold cyan]Splitting Pocket Graphs[/bold cyan]\n")
    console.print(f"[cyan]•[/cyan] Loading bulk file: {bulk_file}")

    # Load bulk file
    with open(bulk_file, 'rb') as f:
        pocket_dicts = pickle.load(f)

    if not isinstance(pocket_dicts, list):
        console.print(f"[red]✗[/red] Expected list, got {type(pocket_dicts)}", style="bold red")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Loaded {len(pocket_dicts)} pocket graphs")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each pocket
    stats = {'success': 0, 'skipped': 0, 'failed': 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Splitting {len(pocket_dicts)} pockets...",
            total=len(pocket_dicts)
        )

        for data in pocket_dicts:
            try:
                target_id = data['target_id']
                state = data['state']
                output_file = output_dir / f"{target_id}__{state}.pkl"

                # Check if exists
                if output_file.exists() and not overwrite:
                    stats['skipped'] += 1
                    progress.update(task, advance=1)
                    continue

                # Convert dict to PocketGraph object
                pocket_graph = dict_to_pocket_graph(data)

                # Save individual file
                with open(output_file, 'wb') as f:
                    pickle.dump(pocket_graph, f)

                stats['success'] += 1

            except Exception as e:
                console.print(f"[red]Error processing pocket {data.get('target_id', 'unknown')}: {e}[/red]")
                stats['failed'] += 1

            progress.update(task, advance=1)

    # Summary
    console.print("\n[bold green]✓ Splitting Complete![/bold green]")
    console.print(f"  Success: {stats['success']}")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Failed: {stats['failed']}")
    console.print(f"\n[cyan]Output directory:[/cyan] {output_dir}\n")


@app.command()
def split_pockets(
    bulk_file: Path = typer.Option(
        ...,
        "--bulk-file",
        "-i",
        help="Path to bulk pockets pickle file (e.g., all_pockets.pkl)",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for individual pocket files",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing files",
    ),
) -> None:
    """
    Split bulk pocket pickle file into individual files.

    Example:
        python src/pprag/dataio/split_bulk_graphs.py split-pockets \\
            --bulk-file Output/graphs/pockets/all_pockets.pkl \\
            --output-dir Output/graphs/pockets
    """
    split_pockets_internal(bulk_file, output_dir, overwrite)


def split_ligands_internal(bulk_file: Path, output_dir: Path, overwrite: bool) -> None:
    """Internal function to split ligand pickle files."""
    console.print("\n[bold cyan]Splitting Ligand Graphs[/bold cyan]\n")
    console.print(f"[cyan]•[/cyan] Loading bulk file: {bulk_file}")

    # Load bulk file
    with open(bulk_file, 'rb') as f:
        ligand_dicts = pickle.load(f)

    if not isinstance(ligand_dicts, list):
        console.print(f"[red]✗[/red] Expected list, got {type(ligand_dicts)}", style="bold red")
        raise typer.Exit(code=1)

    console.print(f"[green]✓[/green] Loaded {len(ligand_dicts)} ligand graphs")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each ligand
    stats = {'success': 0, 'skipped': 0, 'failed': 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Splitting {len(ligand_dicts)} ligands...",
            total=len(ligand_dicts)
        )

        for data in ligand_dicts:
            try:
                ligand_id = data['ligand_id']
                output_file = output_dir / f"{ligand_id}.pkl"

                # Check if exists
                if output_file.exists() and not overwrite:
                    stats['skipped'] += 1
                    progress.update(task, advance=1)
                    continue

                # Convert dict to LigandGraph object
                ligand_graph = dict_to_ligand_graph(data)

                # Save individual file
                with open(output_file, 'wb') as f:
                    pickle.dump(ligand_graph, f)

                stats['success'] += 1

            except Exception as e:
                console.print(f"[red]Error processing ligand {data.get('ligand_id', 'unknown')}: {e}[/red]")
                stats['failed'] += 1

            progress.update(task, advance=1)

    # Summary
    console.print("\n[bold green]✓ Splitting Complete![/bold green]")
    console.print(f"  Success: {stats['success']}")
    console.print(f"  Skipped: {stats['skipped']}")
    console.print(f"  Failed: {stats['failed']}")
    console.print(f"\n[cyan]Output directory:[/cyan] {output_dir}\n")


@app.command()
def split_ligands(
    bulk_file: Path = typer.Option(
        ...,
        "--bulk-file",
        "-i",
        help="Path to bulk ligands pickle file (e.g., all_ligands.pkl)",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for individual ligand files",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing files",
    ),
) -> None:
    """
    Split bulk ligand pickle file into individual files.

    Example:
        python split_bulk_graphs.py split-ligands \\
            --bulk-file Output/graphs/ligands/all_ligands.pkl \\
            --output-dir Output/graphs/ligands
    """
    split_ligands_internal(bulk_file, output_dir, overwrite)


@app.command()
def split_all(
    graphs_dir: Path = typer.Option(
        ...,
        "--graphs-dir",
        "-g",
        help="Base graphs directory (contains pockets/ and ligands/ subdirs)",
        exists=True,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing files",
    ),
) -> None:
    """
    Split all bulk files (pockets and ligands) in one command.

    Example:
        python split_bulk_graphs.py split-all \\
            --graphs-dir Output/graphs
    """
    console.print("\n[bold cyan]Splitting All Graph Files[/bold cyan]\n")

    # Process pockets
    pockets_bulk = graphs_dir / "pockets" / "all_pockets.pkl"
    if pockets_bulk.exists():
        console.print("[bold yellow]Processing Pockets...[/bold yellow]")
        try:
            # Call split_pockets directly as a function
            split_pockets_internal(
                bulk_file=pockets_bulk,
                output_dir=graphs_dir / "pockets",
                overwrite=overwrite
            )
        except Exception as e:
            console.print(f"[red]Failed to split pockets: {e}[/red]")
    else:
        console.print(f"[yellow]⚠[/yellow] Pockets file not found: {pockets_bulk}")

    # Process ligands
    ligands_bulk = graphs_dir / "ligands" / "all_ligands.pkl"
    if ligands_bulk.exists():
        console.print("\n[bold yellow]Processing Ligands...[/bold yellow]")
        try:
            # Call split_ligands directly as a function
            split_ligands_internal(
                bulk_file=ligands_bulk,
                output_dir=graphs_dir / "ligands",
                overwrite=overwrite
            )
        except Exception as e:
            console.print(f"[red]Failed to split ligands: {e}[/red]")
    else:
        console.print(f"[yellow]⚠[/yellow] Ligands file not found: {ligands_bulk}")

    console.print("\n[bold green]✓ All files processed![/bold green]\n")


if __name__ == "__main__":
    app()
