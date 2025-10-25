# Ligand Dataset Validation & Cleaning

## Quick Start

To validate and clean your ligands dataset with default settings:

```bash
# From project root directory
python3 src/pprag/dataio/validate_ligands.py quick-clean
```

This will:

- Read `Data/meta/ligands.csv`
- Validate all SMILES strings for chemical validity
- Attempt to fix common issues (valence errors, kekulization)
- Create cleaned dataset: `Data/meta/ligands_clean.csv`
- Generate report: `Data/meta/ligands_validation_report.txt`

## Custom Paths

For custom input/output paths:

```bash
python3 src/pprag/dataio/validate_ligands.py \
    path/to/input.csv \
    path/to/output_clean.csv \
    --report-file path/to/report.txt
```

## Options

- `--attempt-fix / --no-attempt-fix` - Try to fix invalid SMILES (default: True)
- `--remove-invalid / --no-remove-invalid` - Remove unfixable ligands (default: True)

## What Gets Fixed

The script attempts 3 repair strategies:

1. **Canonical regeneration**: Parse → remove H → regenerate SMILES
2. **InChI round-trip**: SMILES → InChI → SMILES (normalizes structure)
3. **Charge reset**: Remove all formal charges and re-parse

## Common Issues Detected

- ❌ **Valence errors**: Atoms with too many bonds (e.g., N with 4 bonds)
- ❌ **Kekulization errors**: Aromatic rings that can't assign bond orders
- ❌ **Parse errors**: Invalid SMILES syntax
- ❌ **Empty SMILES**: Missing molecular data

## Output

### Cleaned CSV

Same format as input, but with:

- Invalid ligands removed
- Fixed SMILES (when possible) replacing originals

### Validation Report

Detailed text report containing:

- Summary statistics
- List of all invalid ligands with error details
- Error type breakdown

## After Cleaning

Update your prep_main.py command to use the cleaned dataset:

```bash
python3 src/pprag/dataio/prep_main.py \
    --ligands-csv Data/meta/ligands_clean.csv \
    --pockets-csv Data/meta/pockets.csv \
    --meta-out Output/splits \
    --pocket-out Output/pockets \
    --radius 10.0
```

---
