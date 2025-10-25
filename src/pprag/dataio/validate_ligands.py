#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/validate_ligands.py: Validate and clean ligand SMILES dataset.

This script checks all ligands for chemical validity and creates a cleaned CSV
with only valid molecules. It reports issues and optionally attempts fixes.

Functions:
    validate_smiles: Check if a SMILES string is valid
    fix_smiles: Attempt to repair common SMILES issues
    validate_and_clean_csv: Process entire ligands CSV and create cleaned version

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import csv
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from rdkit import Chem
from rdkit import RDLogger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import typer

# Disable RDKit warnings during validation
RDLogger.DisableLog('rdApp.*')  # noqa

app = typer.Typer(help="Validate and clean ligand SMILES dataset")
console = Console()


def validate_smiles(smi: str, ligand_id: str = "") -> Tuple[bool, str, Optional[str]]:
    """
    Validate a SMILES string and return detailed status.

    Args:
        smi: SMILES string to validate
        ligand_id: Ligand identifier for error reporting

    Returns:
        Tuple of (is_valid, error_type, error_message)
        - is_valid: True if SMILES is chemically valid
        - error_type: Category of error (valence, kekulization, parse, empty)
        - error_message: Detailed error description
    """
    if not smi or smi.strip() == "":
        return False, "empty", "Empty SMILES string"

    # Try standard parsing
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return False, "parse", "Cannot parse SMILES syntax"

    # Try sanitization to catch chemical errors
    try:
        Chem.SanitizeMol(mol)
        return True, "valid", None
    except Chem.AtomValenceException as e:
        return False, "valence", str(e)
    except Chem.KekulizeException as e:
        return False, "kekulization", str(e)
    except Exception as e:
        return False, "other", str(e)


def fix_smiles(smi: str) -> Optional[str]:
    """
    Attempt to fix common SMILES issues.

    Strategies:
    1. Normalize charge states
    2. Remove explicit hydrogens
    3. Canonical SMILES regeneration
    4. InChI round-trip conversion

    Returns:
        Fixed SMILES string if successful, None otherwise
    """
    # Strategy 1: Try to parse and regenerate canonical SMILES
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            # Remove explicit hydrogens
            mol = Chem.RemoveHs(mol, sanitize=False)
            # Try to sanitize
            Chem.SanitizeMol(mol)
            # Generate canonical SMILES
            fixed = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            # Validate the fixed version
            if validate_smiles(fixed)[0]:
                return fixed
    except Exception:
        pass

    # Strategy 2: InChI round-trip (can fix some structural issues)
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            inchi = Chem.MolToInchi(mol)
            if inchi:
                mol_fixed = Chem.MolFromInchi(inchi)
                if mol_fixed is not None:
                    fixed = Chem.MolToSmiles(mol_fixed, canonical=True, isomericSmiles=True)
                    if validate_smiles(fixed)[0]:
                        return fixed
    except Exception:
        pass

    # Strategy 3: Try without charge specification
    try:
        # Remove charge annotations and try again
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            # Reset all formal charges
            for atom in mol.GetAtoms():
                atom.SetFormalCharge(0)
            Chem.SanitizeMol(mol)
            fixed = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if validate_smiles(fixed)[0]:
                return fixed
    except Exception:
        pass

    return None


@app.command()
def validate_and_clean_csv(
    input_csv: Path = typer.Argument(..., help="Input ligands CSV file"),
    output_csv: Path = typer.Argument(..., help="Output cleaned CSV file"),
    report_file: Path = typer.Option("ligands_validation_report.txt", help="Validation report file"),
    attempt_fix: bool = typer.Option(True, help="Attempt to fix invalid SMILES"),
    remove_invalid: bool = typer.Option(True, help="Remove invalid ligands from output"),
) -> None:
    """
    Validate all ligands in CSV and create a cleaned version.

    This will:
    1. Check every SMILES string for chemical validity
    2. Attempt to fix common issues (if --attempt-fix)
    3. Create cleaned CSV with valid ligands only (if --remove-invalid)
    4. Generate detailed validation report
    """
    console.print("\n[bold cyan]═══ Ligand Dataset Validation & Cleaning ═══[/bold cyan]\n")
    console.print(f"  • Input:  [cyan]{input_csv}[/cyan]")
    console.print(f"  • Output: [cyan]{output_csv}[/cyan]")
    console.print(f"  • Report: [cyan]{report_file}[/cyan]")
    console.print(f"  • Attempt fixes: [cyan]{attempt_fix}[/cyan]")
    console.print(f"  • Remove invalid: [cyan]{remove_invalid}[/cyan]\n")

    # Read input CSV
    rows = []
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV file has no header row or is empty")
        rows = list(reader)

    total = len(rows)
    console.print(f"[bold]Found {total} ligands to validate[/bold]\n")

    # Validation tracking
    valid_count = 0
    fixed_count = 0
    invalid_count = 0
    error_types: Dict[str, List[str]] = {
        "valence": [],
        "kekulization": [],
        "parse": [],
        "empty": [],
        "other": []
    }

    cleaned_rows = []

    # Validate each ligand
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Validating ligands...", total=total)

        for row in rows:
            ligand_id = row['ligand_id']
            original_smiles = row['smiles']

            # Validate
            is_valid, error_type, error_msg = validate_smiles(original_smiles, ligand_id)

            if is_valid:
                valid_count += 1
                cleaned_rows.append(row)
            else:
                # Try to fix if requested
                if attempt_fix:
                    fixed_smiles = fix_smiles(original_smiles)
                    if fixed_smiles:
                        fixed_count += 1
                        row_copy = row.copy()
                        row_copy['smiles'] = fixed_smiles
                        cleaned_rows.append(row_copy)
                    else:
                        invalid_count += 1
                        error_types[error_type].append(f"{ligand_id}: {error_msg}")
                        if not remove_invalid:
                            cleaned_rows.append(row)
                else:
                    invalid_count += 1
                    error_types[error_type].append(f"{ligand_id}: {error_msg}")
                    if not remove_invalid:
                        cleaned_rows.append(row)

            progress.update(task, advance=1)

    # Display summary statistics
    console.print("\n[bold green]✓ Validation Complete[/bold green]\n")

    table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right")

    table.add_row("Total ligands", str(total), "100.0%")
    table.add_row("Valid (original)", str(valid_count), f"{valid_count / total * 100:.1f}%")
    if attempt_fix:
        table.add_row("Fixed successfully", str(fixed_count), f"{fixed_count / total * 100:.1f}%", style="yellow")
    table.add_row("Invalid (unfixable)", str(invalid_count), f"{invalid_count / total * 100:.1f}%", style="red")
    table.add_row("─" * 20, "─" * 8, "─" * 12)
    table.add_row("Final dataset", str(len(cleaned_rows)), f"{len(cleaned_rows) / total * 100:.1f}%", style="bold green")

    console.print(table)

    # Error breakdown
    if invalid_count > 0:
        console.print("\n[bold yellow]Invalid Ligand Breakdown:[/bold yellow]")
        error_table = Table(show_header=True, header_style="bold red")
        error_table.add_column("Error Type", style="cyan")
        error_table.add_column("Count", justify="right")

        for err_type, ligands in error_types.items():
            if ligands:
                error_table.add_row(err_type.capitalize(), str(len(ligands)))

        console.print(error_table)

    # Write cleaned CSV
    console.print(f"\n[bold]Writing cleaned dataset to {output_csv}[/bold]")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    console.print(f"  ✓ Wrote {len(cleaned_rows)} ligands\n")

    # Write detailed report
    console.print(f"[bold]Writing validation report to {report_file}[/bold]")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LIGAND DATASET VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input file: {input_csv}\n")
        f.write(f"Output file: {output_csv}\n")
        f.write(f"Total ligands processed: {total}\n")
        f.write(f"Valid ligands: {valid_count} ({valid_count / total * 100:.1f}%)\n")
        if attempt_fix:
            f.write(f"Fixed ligands: {fixed_count} ({fixed_count / total * 100:.1f}%)\n")
        f.write(f"Invalid ligands: {invalid_count} ({invalid_count / total * 100:.1f}%)\n")
        f.write(f"Final dataset size: {len(cleaned_rows)} ({len(cleaned_rows) / total * 100:.1f}%)\n\n")
        f.write("=" * 80 + "\n")
        f.write("INVALID LIGANDS DETAILS\n")
        f.write("=" * 80 + "\n\n")

        for err_type, ligands in error_types.items():
            if ligands:
                f.write(f"\n{err_type.upper()} ERRORS ({len(ligands)}):\n")
                f.write("-" * 80 + "\n")
                for entry in ligands:
                    f.write(f"  {entry}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    console.print("  ✓ Report saved\n")

    # Final recommendations
    if invalid_count > 0:
        console.print("[bold yellow]⚠ Recommendations:[/bold yellow]")
        console.print(f"  • Review {report_file} for details on invalid ligands")
        console.print(f"  • {invalid_count} ligands could not be fixed automatically")
        console.print("  • Consider manually reviewing/correcting these in the source database")
        if not remove_invalid:
            console.print("  • Invalid ligands were kept in output (use --remove-invalid to exclude)")
    else:
        console.print("[bold green]✓ All ligands are valid! Dataset is clean.[/bold green]")


@app.command(name="quick-clean")
def quick_clean() -> None:
    """
    Quick clean with default paths - validates and cleans Data/meta/ligands.csv
    """
    input_csv = Path("Data/meta/ligands.csv")
    output_csv = Path("Data/meta/ligands_clean.csv")
    report_file = Path("Data/meta/ligands_validation_report.txt")

    if not input_csv.exists():
        console.print(f"[bold red]Error: {input_csv} not found![/bold red]")
        console.print("Run from project root directory or specify paths manually.")
        raise typer.Exit(1)

    validate_and_clean_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        report_file=report_file,
        attempt_fix=True,
        remove_invalid=True
    )


if __name__ == "__main__":
    app()
