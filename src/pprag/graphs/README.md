# Graph Builder Pipeline - README

## Overview

The main goal here is to convert raw molecular data (MOL2 files for ligands and PDB files for proteins) into graph representations that our GNN models can actually work with.

## The Files

### 1. `ligand_builder.py` - Building Ligand Graphs

This is where we build the ligand representations. Each ligand becomes a graph where:

- **Nodes** = atoms (with features like atomic number, charge, aromaticity, etc.)
- **Edges** = bonds (single, double, aromatic, etc.)
- **Hypergraphs** = pharmacophore groups (functional groups that matter for binding)

#### What makes this tricky

The pharmacophore hypergraph part was actually pretty complex. Instead of just treating atoms and bonds as a simple graph, We also detecting important chemical patterns like:

- Hydrogen bond donors/acceptors
- Aromatic rings
- Charged groups (cations/anions)
- Flexible linkers vs rigid fragments

**Big headache**: RDKit sometimes crashes with segfaults when processing certain molecules, especially when accessing 3D coordinates. I had to add a TON of try-except blocks and progressive sanitization strategies to handle weird edge cases like:

- Molecules with missing hydrogens
- Failed kekulization (aromaticity issues)
- Charged groups with valence problems

The `load_mol_from_mol2()` function alone has like 5 different fallback strategies just to avoid crashes!

#### Key Functions Ligand Graphs

- `load_mol_from_mol2()` - Safely loads MOL2 files (with lots of error handling)
- `get_atom_features()` - Extracts around 44 dimensional feature vectors per atom
- `get_bond_features()` - Extracts bond type and distance features
- `detect_pharmacophores()` - SMARTS pattern matching to find functional groups
- `build_ligand_graph()` - Puts it all together into a LigandGraph object

### 2. `pocket_builder.py` - Building Pocket Graphs

This builds the protein binding site (pocket) representations.

- **Nodes** = amino acid residues in the pocket
- **Edges** = spatial proximity (CA-CA distance < 10 Angstroms, plus kNN for connectivity)

#### Why residue-level?

I chose residue-level graphs instead of atom-level for proteins because:

1. Proteins are HUGE - atom-level would be computationally insane
2. Residue-level captures the important chemistry (amino acid identity, charge, hydrophobicity)
3. It's a common approach in protein-ligand interaction modeling

#### Features tracking

- Amino acid type (one-hot encoded, 20 possibilities)
- Solvent accessible surface area (SASA)
- Hydropathy (Kyte-Doolittle scale)
- Charge class (-1, 0, +1)
- H-bond donor/acceptor counts
- Secondary structure (helix/sheet/coil - though we are using simple heuristics for now instead of DSSP)

#### Key Functions Protein Pocket Graphs

- `rbf_encode_distance()` - Radial basis function encoding for distances
- `build_residue_features()` - Creates ~30 dimensional feature vectors per residue
- `build_edges_distance()` - Distance-based edge construction with kNN fallback
- `build_edge_features()` - Edge features with RBF distances and sequential separation flags
- `build_pocket_graph()` - Assembles the final PocketGraph

### 3. `build_dataset.py` - The Main Pipeline

This is the orchestrator script that processes the entire dataset. It:

1. Loads metadata CSVs (ligands and pockets)
2. Builds graphs for all ligands using `ligand_builder.py`
3. Builds graphs for all pockets using `pocket_builder.py`
4. Saves everything as pickle files organized by train/val/test splits
5. Generates metadata about the dataset

#### Design Decisions

**Sequential Processing**: Originally I tried multiprocessing for speed, but RDKit has these annoying C-level segfaults that crash worker processes unpredictably. So I ended up forcing sequential processing for ligands with:

- Checkpoint system (saves progress after each molecule)
- 30-second timeout per molecule (using signal handlers)
- Resume capability (if it crashes, you can restart from where it left off)

Not elegant, but it works and won't lose progress if RDKit decides to segfault on molecule like #847 out of 1000.

**Pocket Pre-computation**: For pockets, you can pass in pre-computed PocketSelect files via `--pockets-pkl`. This is way faster than computing pocket selections on-the-fly during graph building.

#### Usage Example

```bash
python3 src/pprag/dataio/build_dataset.py \
    --ligands-csv Data/meta/ligands_clean.csv \
    --pockets-csv Data/meta/pockets.csv \
    --output-dir Output/graphs \
    --splits-dir Output/splits \
    --pockets-pkl Output/pockets \
    --n-workers 1
```

#### Output Structure

```bash
Output/graphs/
├── ligands/
│   ├── all_ligands.pkl
│   ├── train_ligands.pkl
│   ├── val_ligands.pkl
│   └── test_ligands.pkl
├── pockets/
│   ├── all_pockets.pkl
│   ├── agonist_pockets.pkl
│   └── antagonist_pockets.pkl
└── dataset_metadata.json
```

## The Feature Specification

I'm using a `FeatureSpec` dataclass to keep track of all the dimensions:

- `d_lig_node = 44` (ligand atom features)
- `d_lig_edge = 6` (bond features without 3D distances)
- `d_poc_node = 30` (pocket residue features)
- `d_poc_edge = 19` (pocket edge features with RBF + contact flags)

> [!NOTE]
> Make sure to have proper input files in correct locations if you want to reproduce the dataset building.

-----
