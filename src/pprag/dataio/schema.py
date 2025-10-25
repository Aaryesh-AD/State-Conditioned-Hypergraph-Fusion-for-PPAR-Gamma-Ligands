#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/schema.py: Schema definitions for data I/O and model-ready objects.

Current Definitions:
- Row-level catalogs for ligands and targets (processed dataset CSVs)
- Feature specs (to keep d_model, featurization flags centralized)
- Graph containers for ligand and pocket (residue) graphs
- Hypergraph structures (incidence lists) for ligand pharmacophores

Classes:
    LigandRow: Row in ligand catalog CSV
    TargetRow: Row in target catalog CSV
    FeatureSpec: Feature dimensions and flags for featurization
    ResidueFeats: Features for a single residue in pocket selection
    PocketSelect: Selected pocket residues for a target chain
    LigandGraph: Graph container for ligand atom graph + hypergraph
    PocketGraph: Graph container for pocket residue graph

# TODO: Add this as we go along:
- Sample wrappers for attention, contrastive pretraining and finetuning
- Outputs/diagnostics (soft contact maps, logits, etc.)

Author: Aaryesh Deshpande
Last Modified: 10/24/2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Literal
import numpy as np


def global_seed() -> int:
    """Global seed for reproducibility across modules."""
    return 16


# Catalog row definitions
@dataclass
class LigandRow:
    ligand_id: str
    smiles: str
    mol2_path: str
    class_label: Literal["agonist", "antagonist"]    # from directory name
    is_decoy: int                                    # 0/1


@dataclass
class TargetRow:
    target_id: str
    mol2_path: str                                   # protein MOL2 path
    state: Literal["agonist", "antagonist"]          # from directory name
    ligand_path: Optional[str] = None                # co-complexed ligand MOL2


# Feature specifications
@dataclass
class FeatureSpec:
    """Dims and flags for builds persistent - for reproducibility."""

    # ligand atom features
    d_lig_node: int = 96
    d_lig_edge: int = 8
    use_3d_distances: bool = True

    # ligand hypergraph
    use_pharmacophore_hypergraph: bool = True
    hyperedge_types: Tuple[str, ...] = (
        "ring", "donor_group", "acceptor_group",
        "cation_center", "anion_center", "halogen_donor", "aromatic_cluster",
        "rigid_fragment", "flexible_linker"
    )

    # pocket residue node features
    d_poc_node: int = 64
    d_poc_edge: int = 6

    # state conditioning
    use_state_token: bool = True
    d_state: int = 16

    # attention heads
    # TODO: Shift to a new dataclass for model spec/config later
    n_heads: int = 4
    d_model: int = 256


# Pocket selection
@dataclass
class ResidueFeats:
    resid: int
    aa_idx: int                 # 0..19; map non-canon to nearest or 0
    sasa: float                 # solvent accessible surface area
    hydropathy: float           # Kyte–Doolittle
    charge_class: int           # {-1,0,1}
    hbd: int                    # donor count (side chain)
    hba: int                    # acceptor count
    ca_xyz: np.ndarray          # (3,)
    sc_xyz: Optional[np.ndarray] = None  # (3,) side-chain centroid


@dataclass
class PocketSelect:
    target_id: str
    chain: str
    residues: List[int]
    feats: List[ResidueFeats]


# Graph containers
@dataclass
class LigandGraph:
    """Atom graph + coordinates (+ optional hypergraph)."""
    ligand_id: str

    # Atom graph
    x: np.ndarray               # (N_a, d_lig_node) atom features
    edge_index: np.ndarray      # (2, E) int64
    edge_attr: np.ndarray       # (E, d_lig_edge)
    pos: np.ndarray             # (N_a, 3) 3D coords from MOL2 conformer

    # Optional global props (qed/logP etc.)
    props: Dict[str, float] = field(default_factory=dict)

    # Hypergraph
    # Incidence list representation: for each hyperedge, the atom indices it contains
    hyperedge_members: Optional[List[List[int]]] = None
    # Hyperedge feature matrix (type one-hot, size, 3D centroid, etc.)
    hyperedge_attr: Optional[np.ndarray] = None      # (H, d_h)


@dataclass
class PocketGraph:
    """Residue graph with residue-level features + 3D anchors."""
    target_id: str
    state_id: int                                   # 0=agonist, 1=antagonist
    x: np.ndarray                                   # (N_r, d_poc_node) residue features
    edge_index: np.ndarray                          # (2, E_r)
    edge_attr: np.ndarray                           # (E_r, d_poc_edge) distances/flags
    pos_ca: np.ndarray                              # (N_r, 3) CA coords
    pos_sc: Optional[np.ndarray] = None             # (N_r, 3) side-chain centroids (optional, can be NaN)
    residue_ids: Optional[np.ndarray] = None        # (N_r,) original residue numbers for debugging


# State mappings enum like
STATE_TO_ID: Dict[str, int] = {"agonist": 0, "antagonist": 1, "partial": 2, "apo": 3}   # extendable in-case we add more
ID_TO_STATE: Dict[int, str] = {v: k for k, v in STATE_TO_ID.items()}


# Amino acid constants and mappings
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}

# Three-letter to one-letter amino acid code conversion
AA3_TO_AA1: Dict[str, str] = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Amino acid side-chain SMILES for RDKit-based calculations
# These represent the side chain structure (without backbone)
AA_SIDECHAIN_SMILES: Dict[str, str] = {
    'A': 'C',                          # Alanine: methyl
    'C': 'CS',                         # Cysteine: thiol
    'D': 'CC(=O)[O-]',                 # Aspartate: carboxylate (charged)
    'E': 'CCC(=O)[O-]',                # Glutamate: carboxylate (charged)
    'F': 'Cc1ccccc1',                  # Phenylalanine: benzyl
    'G': '[H]',                        # Glycine: hydrogen
    'H': 'Cc1c[nH]cn1',                # Histidine: imidazole (can be protonated)
    'I': 'C(C)C(C)C',                  # Isoleucine
    'K': 'CCCC[NH3+]',                 # Lysine: amine (charged)
    'L': 'CC(C)C',                     # Leucine
    'M': 'CCSC',                       # Methionine: thioether
    'N': 'CC(=O)N',                    # Asparagine: amide
    'P': 'C1CCCN1',                    # Proline: pyrrolidine (cyclic)
    'Q': 'CCC(=O)N',                   # Glutamine: amide
    'R': 'CCCCNC(=[NH2+])N',           # Arginine: guanidinium (charged)
    'S': 'CO',                         # Serine: hydroxyl
    'T': 'C(C)O',                      # Threonine: hydroxyl
    'V': 'C(C)C',                      # Valine
    'W': 'Cc1c[nH]c2ccccc12',          # Tryptophan: indole
    'Y': 'Cc1ccc(O)cc1',               # Tyrosine: phenol
}

# Full amino acid SMILES (with backbone) for more accurate property calculations
# Format: NCC(R)C(=O)O where R is the side chain
AA_FULL_SMILES: Dict[str, str] = {
    'A': 'NCC(C)C(=O)O',                          # Alanine
    'C': 'NCC(CS)C(=O)O',                         # Cysteine
    'D': 'NCC(CC(=O)[O-])C(=O)O',                 # Aspartate
    'E': 'NCC(CCC(=O)[O-])C(=O)O',                # Glutamate
    'F': 'NCC(Cc1ccccc1)C(=O)O',                  # Phenylalanine
    'G': 'NCC(=O)O',                              # Glycine
    'H': 'NCC(Cc1c[nH]cn1)C(=O)O',                # Histidine
    'I': 'NCC(C(C)CC)C(=O)O',                     # Isoleucine
    'K': 'NCC(CCCC[NH3+])C(=O)O',                 # Lysine
    'L': 'NCC(CC(C)C)C(=O)O',                     # Leucine
    'M': 'NCC(CCSC)C(=O)O',                       # Methionine
    'N': 'NCC(CC(=O)N)C(=O)O',                    # Asparagine
    'P': 'N1CC(C(=O)O)CC1',                       # Proline
    'Q': 'NCC(CCC(=O)N)C(=O)O',                   # Glutamine
    'R': 'NCC(CCCCNC(=[NH2+])N)C(=O)O',           # Arginine
    'S': 'NCC(CO)C(=O)O',                         # Serine
    'T': 'NCC(C(C)O)C(=O)O',                      # Threonine
    'V': 'NCC(C(C)C)C(=O)O',                      # Valine
    'W': 'NCC(Cc1c[nH]c2ccccc12)C(=O)O',          # Tryptophan
    'Y': 'NCC(Cc1ccc(O)cc1)C(=O)O',               # Tyrosine
}

# Kyte-Doolittle hydropathy index (positive=hydrophobic)
# Kept as reference but prefer RDKit LogP-based calculation
KD_HYDROPATHY: Dict[str, float] = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7,
    'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5,
    'N': -3.5, 'K': -3.9, 'R': -4.5
}

# Charge classification at physiological pH (~7.4)
# Kept as reference but prefer RDKit formal charge calculation
CHARGE_MAP_PH7: Dict[str, int] = {
    'K': 1, 'R': 1, 'H': 1,  # positive (H can be protonated)
    'D': -1, 'E': -1,         # negative
}
