#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/prep_main.py: Main CLI for data preparation pipeline.

This script handles:
1. Murcko scaffold-based train/val/test splits
2. Phase partitioning (pretrain, zero-shot, few-shot)
3. Pocket selection with RDKit-based feature calculations

The pocket selection now uses RDKit for accurate molecular property calculations:
- Hydropathy: Calculated from TPSA and molecular weight (fallback: Kyte-Doolittle)
- Charge: Formal charge from SMILES representation (fallback: pH 7.4 lookup)
- HBD/HBA: Automatic counting via RDKit descriptors
- SASA: FreeSASA algorithm (fallback: neighbor-counting approximation)

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import pickle
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from load_labels import load_target_csv
from splits import murcko_scaffold_split, write_phase_partitions
from pocket_select import build_pocket_select
from schema import global_seed

# Initialize Typer app and Rich console
app = typer.Typer(
    name="prep",
    help="PPAR-gamma Data Preparation Pipeline - Splits, phases, and pocket feature extraction",
    add_completion=False,
)
console = Console()

SEED = global_seed()


@app.command()
def main(
    ligands_csv: Path = typer.Option(
        ...,
        "--ligands-csv",
        "-l",
        help="Path to ligands CSV file (e.g., data/meta/ligands.csv)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    pockets_csv: Path = typer.Option(
        ...,
        "--pockets-csv",
        "-p",
        help="Path to pockets/targets CSV file (e.g., data/meta/pockets.csv)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    meta_out: Path = typer.Option(
        ...,
        "--meta-out",
        "-m",
        help="Output directory for split JSONs (train/val/test/pretrain/zero-shot/few-shot)",
    ),
    pocket_out: Optional[Path] = typer.Option(
        None,
        "--pocket-out",
        "-o",
        help="Output directory for PocketSelect pickle files (optional but recommended)",
    ),
    radius: float = typer.Option(
        10.0,
        "--radius",
        "-r",
        help="Pocket selection radius in Angstroms around ligand center",
        min=3.0,
        max=20.0,
    ),
    centers_json: Optional[Path] = typer.Option(
        None,
        "--centers-json",
        "-c",
        help="Optional JSON file with custom pocket centers: {target_id: [x, y, z]}",
        exists=False,  # May not exist yet
        file_okay=True,
        dir_okay=False,
    ),
    seed: int = typer.Option(
        SEED,
        "--seed",
        "-s",
        help="Random seed for reproducible scaffold splitting",
    ),
    train_frac: float = typer.Option(
        0.8,
        "--train-frac",
        help="Fraction of scaffolds for training set",
        min=0.1,
        max=0.9,
    ),
    val_frac: float = typer.Option(
        0.1,
        "--val-frac",
        help="Fraction of scaffolds for validation set",
        min=0.05,
        max=0.5,
    ),
) -> None:
    """
    Prepare PPAR-gamma dataset: scaffold splits, phase partitions, and pocket features.

    This pipeline performs:

    \b
    1. Murcko scaffold-based splitting into train/val/test sets
    2. Phase partitioning for different training strategies:
       - Pretrain: Agonist ligands only (no decoys)
       - Zero-shot: Antagonist ligands + decoys from test set
       - Few-shot pool: All 9 antagonist ligands (no decoys)
    3. Pocket selection with RDKit-based feature calculations:
       - SASA via FreeSASA algorithm
       - Hydropathy from TPSA + molecular weight
       - Formal charge from SMILES structure
       - HBD/HBA from automatic RDKit counting

    Example:
    \b
        python3 src/pprag/dataio/prep_main.py \\
            --ligands-csv Data/meta/ligands.csv \\
            --pockets-csv Data/meta/pockets.csv \\
            --meta-out Output/splits \\
            --pocket-out Output/pockets \\
            --radius 10.0 \\
            --seed 16
    """

    # Validate fractions
    if train_frac + val_frac >= 1.0:
        console.print(
            "[red]Error:[/red] train_frac + val_frac must be < 1.0",
            style="bold red"
        )
        raise typer.Exit(code=1)

    # Create output directories
    meta_out.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]═══ PPAR-gamma Data Preparation Pipeline ═══[/bold cyan]\n")

    # ===== STEP 1: Scaffold Splitting =====
    console.print("[bold yellow]Step 1:[/bold yellow] Murcko Scaffold-Based Splitting")
    console.print(f"  • Ligands CSV: [cyan]{ligands_csv}[/cyan]")
    console.print(f"  • Random seed: [cyan]{seed}[/cyan]")
    console.print(f"  • Split ratios: [cyan]{train_frac:.1%}[/cyan] train / "
                  f"[cyan]{val_frac:.1%}[/cyan] val / "
                  f"[cyan]{1 - train_frac - val_frac:.1%}[/cyan] test")
    console.print("  [dim]Note: RDKit warnings for invalid SMILES are suppressed[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Computing Murcko scaffolds...", total=None)
        train_ids, val_ids, test_ids = murcko_scaffold_split(
            ligands_csv,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac
        )
        progress.update(task, completed=True)

    # Display split statistics
    table = Table(title="Split Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", justify="left")
    table.add_column("Count", justify="right", style="green")
    table.add_row("Train", str(len(train_ids)))
    table.add_row("Validation", str(len(val_ids)))
    table.add_row("Test", str(len(test_ids)))
    table.add_row("Total", str(len(train_ids) + len(val_ids) + len(test_ids)), style="bold")
    console.print(table)
    console.print()

    # ===== STEP 2: Phase Partitioning =====
    console.print("[bold yellow]Step 2:[/bold yellow] Phase Partitioning")
    console.print("  • Generating pretrain/zero-shot/few-shot subsets\n")

    files = write_phase_partitions(ligands_csv, meta_out, train_ids, val_ids, test_ids)

    # Display phase statistics
    phase_table = Table(title="Phase Partitions", show_header=True, header_style="bold magenta")
    phase_table.add_column("Phase", style="cyan", justify="left")
    phase_table.add_column("File", style="white", justify="left")
    phase_table.add_column("Purpose", style="yellow", justify="left")

    phase_descriptions = {
        "train_ids": "Standard training split",
        "val_ids": "Validation split",
        "test_ids": "Test split",
        "pretrain_ids": "Agonist ligands only (no decoys)",
        "zero_shot_ids": "Antagonists + decoys from test",
        "fewshot_pool": "All 9 antagonist ligands (no decoys)",
    }

    for name, path in files.items():
        phase_table.add_row(name, path.name, phase_descriptions.get(name, ""))

    console.print(phase_table)
    console.print(f"\n  ✓ Saved to: [cyan]{meta_out}[/cyan]\n")

    # ===== STEP 3: Pocket Selection =====
    if pocket_out:
        console.print("[bold yellow]Step 3:[/bold yellow] Pocket Selection & Feature Calculation")
        console.print(f"  • Pockets CSV: [cyan]{pockets_csv}[/cyan]")
        console.print(f"  • Selection radius: [cyan]{radius:.1f} Å[/cyan]")
        if centers_json:
            console.print(f"  • Custom centers: [cyan]{centers_json}[/cyan]")
        console.print("\n  [dim]Using RDKit for molecular property calculations:[/dim]")
        console.print("    • SASA: FreeSASA algorithm (fallback: neighbor counting)")
        console.print("    • Hydropathy: TPSA + MW (fallback: Kyte-Doolittle)")
        console.print("    • Charge: Formal charge from SMILES (fallback: pH 7.4 lookup)")
        console.print("    • HBD/HBA: Automatic RDKit counting\n")

        pocket_out.mkdir(parents=True, exist_ok=True)
        pockets = load_target_csv(pockets_csv)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Processing {len(pockets)} pocket structures...",
                total=len(pockets)
            )

            for i, t in enumerate(pockets, 1):
                progress.update(
                    task,
                    description=f"Processing pocket {i}/{len(pockets)}: {t.target_id} ({t.state})"
                )

                ps = build_pocket_select(
                    target_id=t.target_id,
                    protein_mol2=t.pdb_path,
                    chain=None,  # Set if MOL2 has chain IDs
                    radius=radius,
                    ligand_mol2=t.ligand_path,
                    centers_json=str(centers_json) if centers_json else None
                )

                output_file = pocket_out / f"{t.target_id}__{t.state}.pkl"
                with open(output_file, "wb") as f:
                    pickle.dump(ps, f)

                progress.advance(task)

        console.print(f"\n  ✓ Saved {len(pockets)} PocketSelect objects to: [cyan]{pocket_out}[/cyan]\n")
    else:
        console.print("[bold yellow]Step 3:[/bold yellow] Pocket Selection")
        console.print("  [yellow]Skipped[/yellow] (use --pocket-out to enable)\n")

    # ===== Summary =====
    console.print("[bold green]✓ Data preparation complete![/bold green]\n")
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Verify split files in: [cyan]{}[/cyan]".format(meta_out))
    if pocket_out:
        console.print("  2. Check pocket features in: [cyan]{}[/cyan]".format(pocket_out))
        console.print("  3. Proceed with graph construction and model training")
    else:
        console.print("  2. Run again with --pocket-out to generate pocket features")
        console.print("  3. Then proceed with graph construction and model training")
    console.print()


if __name__ == "__main__":
    app()
