#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/schema.py: Schema definitions for data I/O and model-ready objects.

Current Definitions:
- Row-level catalogs for ligands and targets (processed dataset CSVs)
- Feature specs (to keep d_model, featurization flags centralized)
- Graph containers for ligand and pocket (residue) graphs
- Hypergraph structures (incidence lists) for ligand pharmacophores

TODO: Add this as we go along:
- Sample wrappers for attention, contrastive pretraining and finetuning
- Outputs/diagnostics (soft contact maps, logits, etc.)

Author: Aaryesh Deshpande
Last Modified: 10/24/2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Literal
import numpy as np


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
