# PPAR-Gamma Data I/O Module

Data preparation pipeline for PPAR-Gamma agonist/antagonist prediction using pocket-based graph neural networks.

## Overview

This module handles:

1. **Dataset cataloging** - Loading ligand and target metadata from CSV files
2. **Scaffold-based splitting** - Murcko scaffold splitting for train/val/test sets
3. **Phase partitioning** - Organizing data for pretrain/zero-shot/few-shot learning
4. **Pocket selection** - RDKit-based feature extraction from protein structures
5. **Graph construction** - Converting molecules and pockets to graph representations

---

## Module Structure

```bash
dataio/
├── schema.py          # Data structures and constants
├── load_labels.py     # CSV loading utilities
├── murcko.py          # Murcko scaffold utilities
├── splits.py          # Dataset splitting logic
├── pocket_select.py   # Pocket feature extraction (RDKit-based)
├── prep_main.py       # Main CLI for data preparation
└── README.md          # This file
```

---

## Features

### RDKit-Based Molecular Property Calculations

The pocket selection pipeline using **RDKit** for accurate, chemistry-aware feature calculations:

| Property | Method | Fallback |
|----------|--------|----------|
| **SASA** | FreeSASA algorithm | Neighbor-counting approximation |
| **Hydropathy** | TPSA + Molecular Weight | Kyte-Doolittle scale |
| **Charge** | Formal charge from SMILES | pH 7.4 lookup table |
| **HBD/HBA** | Automatic RDKit counting | N/A |

---

## Usage

### Command-Line Interface (Typer)

The main data preparation script using **Typer** for CLI:

```bash
python -m pprag.dataio.prep_main \
    --ligands-csv data/meta/ligands.csv \
    --pockets-csv data/meta/pockets.csv \
    --meta-out Output/splits \
    --pocket-out Output/pockets \
    --radius 10.0 \
    --seed 16
```

#### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--ligands-csv` | `-l` | Path to ligands CSV file |
| `--pockets-csv` | `-p` | Path to pockets/targets CSV file |
| `--meta-out` | `-m` | Output directory for split JSONs |

#### Optional Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--pocket-out` | `-o` | `None` | Output directory for PocketSelect pickles |
| `--radius` | `-r` | `10.0` | Pocket selection radius (Å) |
| `--centers-json` | `-c` | `None` | Custom pocket centers JSON |
| `--seed` | `-s` | `16` | Random seed for splitting |
| `--train-frac` | - | `0.8` | Training set fraction |
| `--val-frac` | - | `0.1` | Validation set fraction |

#### Get Help

```bash
python -m pprag.dataio.prep_main --help
```

---

## Pipeline Steps

### Step 1: Murcko Scaffold-Based Splitting

Splits ligands into train/val/test sets based on their **Murcko scaffolds** to ensure:

- No data leakage between splits
- Model generalizes to novel scaffolds
- Chemically meaningful evaluation

**Output files:**

- `train_ids.json`
- `val_ids.json`
- `test_ids.json`

### Step 2: Phase Partitioning

Organizes ligands for different learning paradigms:

| Phase | Description | Use Case |
|-------|-------------|----------|
| **Pretrain** | Agonist ligands only (no decoys) | Self-supervised pretraining |
| **Zero-shot** | Antagonists + decoys from test set | Evaluate without antagonist training |
| **Few-shot pool** | All 9 antagonist ligands (no decoys) | Few-shot learning experiments |

**Output files:**

- `pretrain_ids.json`
- `zero_shot_ids.json`
- `fewshot_pool.json`

### Step 3: Pocket Selection & Feature Extraction

For each protein target:

1. Load protein structure (MOL2 format)
2. Find binding site center (from co-crystal ligand or custom JSON)
3. Select residues within radius (default: 10 Angstrom)
4. Calculate features for each residue using **RDKit**

**Features calculated per residue:**

- **Amino acid identity** - One-hot encoded (20 standard AAs)
- **SASA** - Solvent accessible surface area
- **Hydropathy** - Hydrophobicity score (-5.0 to +5.0)
- **Charge** - Formal charge class (-1, 0, +1)
- **HBD/HBA** - Hydrogen bond donor/acceptor counts
- **3D coordinates** - CA and side-chain centroids

**Output:** Pickle files named `{target_id}__{state}.pkl`

---

## References

1. **Murcko Scaffolds**: Bemis, G. W., & Murcko, M. A. (1996). *J. Med. Chem.*
2. **Kyte-Doolittle**: Kyte, J., & Doolittle, R. F. (1982). *J. Mol. Biol.*
3. **FreeSASA**: Mitternacht, S. (2016). *F1000Research*
4. **RDKit**: [www.rdkit.org](https://www.rdkit.org/)

---

## Authors

- **Aaryesh Deshpande** - Initial implementation and integration.
- Last Updated: October 25, 2025

---
