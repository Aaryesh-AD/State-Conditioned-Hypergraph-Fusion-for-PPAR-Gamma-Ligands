#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graphs/ligand_builder.py: Build ligand atom graphs with pharmacophore hypergraphs.

This module constructs molecular graph representations from MOL2 files with:
- Atom-level graph (nodes=atoms, edges=bonds)
- Pharmacophore hypergraph (hyperedges for functional groups and structural motifs)

Hyperedge types:
    - ring: aromatic/aliphatic ring systems
    - donor_group: hydrogen bond donors (SMARTS-based)
    - acceptor_group: hydrogen bond acceptors
    - cation_center: positively charged atoms/groups
    - anion_center: negatively charged atoms/groups
    - halogen_donor: halogen bond donors (X-C=O patterns)
    - aromatic_cluster: fused aromatic ring systems
    - rigid_fragment: rigid structural units (rings + amides)
    - flexible_linker: rotatable bond segments

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import sys
import traceback
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem import rdPartialCharges
from pprag.dataio.schema import LigandRow, LigandGraph, FeatureSpec
from pprag.dataio.schema import (
    HBD_SMARTS,
    HBA_SMARTS,
    CATION_SMARTS,
    ANION_SMARTS,
    HALOGEN_DONOR_SMARTS,
    global_seed,
)

SEED = global_seed()


def load_mol_from_mol2(mol2_path: str | Path) -> Optional[Chem.Mol]:
    """
    Load molecule from MOL2 file with 3D coordinates. Segfault-safe.

    Handles common MOL2 issues:
    - Missing explicit hydrogens
    - Charged amidine groups
    - Kekulization failures
    """
    try:
        # First try: keep hydrogens (preferred for formal charge estimation)
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
        if mol is None:
            # Second try: remove hydrogens
            mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=True)

        if mol is not None:
            # Validate molecule has atoms
            if mol.GetNumAtoms() == 0:
                return None

            # Validate molecule has reasonable size (< 500 atoms to avoid memory issues)
            if mol.GetNumAtoms() > 500:
                print(f"Warning: Large molecule ({mol.GetNumAtoms()} atoms) in {mol2_path}",
                      file=sys.stderr)

            # Progressive sanitization strategy to handle problematic molecules
            sanitize_success = False

            # Try 1: Full sanitization
            try:
                Chem.SanitizeMol(mol)
                sanitize_success = True
            except Chem.KekulizeException:
                # Kekulization failed - try to fix aromaticity
                try:
                    # Skip kekulization but do other sanitization steps
                    Chem.SanitizeMol(mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE))
                    # Set all aromatic bonds to AROMATIC type
                    for bond in mol.GetBonds():
                        if bond.GetIsAromatic():
                            bond.SetBondType(Chem.BondType.AROMATIC)
                    sanitize_success = True
                except Exception as e:
                    print(f"Kekulization and aromatic bond fix failed for {mol2_path}: {e}",
                          file=sys.stderr)
            except Chem.AtomValenceException as e:
                # Valence issues (e.g., charged amidine with isFixed atom)
                try:
                    # Try cleaning up formal charges and re-sanitizing
                    for atom in mol.GetAtoms():
                        # Reset problematic formal charges
                        if atom.GetFormalCharge() != 0:
                            # Check if charge is reasonable for this atom type
                            atomic_num = atom.GetAtomicNum()
                            charge = atom.GetFormalCharge()
                            # Keep charges that make sense
                            if not ((atomic_num == 7 and -1 <= charge <= 1) or  # noqa N: -1,0,+1
                                    (atomic_num == 8 and -1 <= charge <= 0) or   # noqa O: -1,0
                                    (atomic_num == 16 and -1 <= charge <= 0)):   # S: -1,0
                                atom.SetFormalCharge(0)

                    # Try sanitization again
                    Chem.SanitizeMol(mol)
                    sanitize_success = True
                except Exception:
                    print(f"Charged amidine/valence fix failed for {mol2_path}: {e}",
                          file=sys.stderr)
            except Exception as e:
                print(f"Sanitization failed for {mol2_path}: {e}", file=sys.stderr)

            # If all sanitization attempts failed, try minimal sanitization
            if not sanitize_success:
                try:
                    Chem.SanitizeMol(mol,
                                     sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
                    sanitize_success = True
                except Exception:
                    # If even minimal sanitization fails, reject the molecule
                    print(f"All sanitization attempts failed for {mol2_path}, rejecting molecule",
                          file=sys.stderr)
                    return None

            # Add explicit hydrogens if missing (for formal charge estimation)
            # This helps with molecules that have "no explicit hydrogens" warnings
            try:
                if mol.GetNumAtoms() > 0:
                    num_hs = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
                    if num_hs == 0:  # No explicit hydrogens
                        # Add them for better charge calculation
                        mol = Chem.AddHs(mol, addCoords=True)
            except Exception:
                # If adding Hs fails, continue without them
                pass

            # Compute Gasteiger charges if needed (wrap in try-except to prevent crashes)
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except Exception:
                # Gasteiger can fail for some molecules, that's ok
                pass

        return mol
    except Exception as e:
        print(f"Error loading MOL2 {mol2_path}: {e}", file=sys.stderr)
        return None


def get_atom_features(atom: Chem.Atom, mol: Chem.Mol, use_charges: bool = True) -> np.ndarray:
    """
    Extract atom features for graph neural network.

    Features (approx estimated 96 dims):
        - Atomic number (one-hot: C,N,O,S,F,Cl,Br,I,P,other = 10 dims)
        - Formal charge (clipped to [-2,2], shifted+scaled = 5 dims one-hot)
        - Aromatic flag (1 dim)
        - Hybridization (sp, sp2, sp3, other = 4 dims one-hot)
        - Degree (0-5, 6 dims one-hot)
        - In ring flag (1 dim)
        - Chiral tag (None, R, S, other = 4 dims one-hot)
        - Hydrogen count (0-4, 5 dims one-hot)
        - Implicit valence (0-6, 7 dims one-hot)
        - Gasteiger charge (1 dim, optional)
    """
    # Atomic number one-hot (C=6, N=7, O=8, S=16, F=9, Cl=17, Br=35, I=53, P=15)
    Z = atom.GetAtomicNum()
    z_allowed = [6, 7, 8, 16, 9, 17, 35, 53, 15]  # C,N,O,S,F,Cl,Br,I,P
    z_feats = [int(Z == z) for z in z_allowed] + [int(Z not in z_allowed)]  # 10 dims

    # Formal charge (clipped to [-2, 2])
    fc = max(-2, min(2, atom.GetFormalCharge()))
    fc_feats = [int(fc == i) for i in range(-2, 3)]  # 5 dims

    # Aromatic
    aromatic = [int(atom.GetIsAromatic())]  # 1 dim

    # Hybridization
    hyb = atom.GetHybridization()
    hyb_feats = [
        int(hyb == Chem.HybridizationType.SP),
        int(hyb == Chem.HybridizationType.SP2),
        int(hyb == Chem.HybridizationType.SP3),
        int(hyb not in [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3])
    ]  # 4 dims

    # Degree (0-5)
    deg = atom.GetDegree()
    deg_feats = [int(deg == i) for i in range(6)]  # 6 dims

    # In ring
    in_ring = [int(atom.IsInRing())]  # 1 dim

    # Chiral tag
    chiral = atom.GetChiralTag()
    chiral_feats = [
        int(chiral == Chem.ChiralType.CHI_UNSPECIFIED),
        int(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
        int(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
        int(chiral not in [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW])
    ]  # 4 dims

    # Hydrogen count (0-4)
    h_count = min(4, atom.GetTotalNumHs())
    h_feats = [int(h_count == i) for i in range(5)]  # 5 dims

    # Implicit valence (0-6)
    try:
        # GetValance() IMPLICIT and EXPLICIT for total
        implicit_val = atom.GetValence(Chem.ValenceType.IMPLICIT)
        explicit_val = atom.GetValence(Chem.ValenceType.EXPLICIT)
        total_val = implicit_val + explicit_val

        val = min(6, total_val - explicit_val)
    except Exception:
        # Fallback to 0 if calculation fails
        val = 0
    val_feats = [int(val == i) for i in range(7)]  # 7 dims

    # Gasteiger partial charge
    gasteiger = [0.0]
    if use_charges:
        try:
            gasteiger = [float(atom.GetProp('_GasteigerCharge'))]
            if np.isnan(gasteiger[0]) or np.isinf(gasteiger[0]):
                gasteiger = [0.0]
        except Exception:
            gasteiger = [0.0]

    # Concatenate: 10+5+1+4+6+1+4+5+7+1 = 44 base dims
    feats = (z_feats + fc_feats + aromatic + hyb_feats + deg_feats + in_ring + chiral_feats + h_feats + val_feats + gasteiger)

    return np.array(feats, dtype=np.float32)


def get_bond_features(bond: Chem.Bond, use_3d_dist: bool = False,
                      pos: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract bond features for graph edges.

    Features:
        - Bond type (single, double, triple, aromatic = 4 dims one-hot)
        - Conjugated flag (1 dim)
        - In ring flag (1 dim)
        - Distance bucket (optional, 8 bins from 0-3 Angstrom if use_3d_dist)
    """
    # Bond type
    bt = bond.GetBondType()
    type_feats = [
        int(bt == Chem.BondType.SINGLE),
        int(bt == Chem.BondType.DOUBLE),
        int(bt == Chem.BondType.TRIPLE),
        int(bt == Chem.BondType.AROMATIC),
    ]  # 4 dims

    # Conjugated
    conj = [int(bond.GetIsConjugated())]  # 1 dim

    # In ring
    in_ring = [int(bond.IsInRing())]  # 1 dim

    feats = type_feats + conj + in_ring  # 6 dims base

    # Optional distance bucket
    if use_3d_dist and pos is not None:
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = np.linalg.norm(pos[i] - pos[j])
        # 8 bins: [0-0.5, 0.5-1.0, ..., 3.0-3.5, >3.5]

        bins = np.linspace(0, 3.5, 8)
        dist_feats = [int(dist >= bins[k] and (k == 7 or dist < bins[k + 1])) for k in range(8)]
        feats += dist_feats  # +8 dims = 14 total

    return np.array(feats, dtype=np.float32)


# NOTE: Just a caution, this function in testing has been known to cause segfaults occasionally, f*ck C++
def detect_pharmacophores(mol: Chem.Mol) -> Dict[str, List[List[int]]]:
    """
    Detect pharmacophore groups and structural motifs in molecule.

    Returns:
        Dictionary mapping hyperedge type to list of atom index lists.
        Each list represents one hyperedge (group of atoms).
    """
    groups: Dict[str, List[List[int]]] = {
        "ring": [],
        "donor_group": [],
        "acceptor_group": [],
        "cation_center": [],
        "anion_center": [],
        "halogen_donor": [],
        "aromatic_cluster": [],
        "rigid_fragment": [],
        "flexible_linker": [],
    }

    # 1. Rings (aromatic and aliphatic)
    ring_info = mol.GetRingInfo()
    aromatic_rings = []
    for ring in ring_info.AtomRings():
        ring_list = list(ring)
        groups["ring"].append(ring_list)

        # Check if aromatic for clustering
        is_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list)
        if is_aromatic:
            aromatic_rings.append(ring_list)

    # 2. H-bond donors (SMARTS-based)
    for smarts in HBD_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                groups["donor_group"].append(list(match))

    # 3. H-bond acceptors
    for smarts in HBA_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                groups["acceptor_group"].append(list(match))

    # 4. Cation centers
    for smarts in CATION_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                groups["cation_center"].append(list(match))

    # 5. Anion centers
    for smarts in ANION_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                groups["anion_center"].append(list(match))

    # 6. Halogen bond donors
    for smarts in HALOGEN_DONOR_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                groups["halogen_donor"].append(list(match))

    # 7. Aromatic clusters (fused rings)
    if len(aromatic_rings) > 1 and mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        # Compute ring centroids

        # NOTE: Here is where the segfaults can happen sometimes (found after stress testing)
        # Might be cuz of boundary issues between Python and C++ in RDKit
        centroid_list = []
        for ring in aromatic_rings:
            coords = []
            for atom_idx in ring:
                pos = conf.GetAtomPosition(atom_idx)
                coords.append([pos.x, pos.y, pos.z])
            coords_array = np.array(coords)
            centroid_list.append(coords_array.mean(axis=0))
        centroids: np.ndarray = np.array(centroid_list)

        # Cluster nearby aromatic rings (distance < 2 Angstrom)
        clustered = [False] * len(aromatic_rings)
        for i in range(len(aromatic_rings)):
            if clustered[i]:
                continue

            cluster = set(aromatic_rings[i])

            for j in range(i + 1, len(aromatic_rings)):
                if clustered[j]:
                    continue
                dist = np.linalg.norm(centroids[i] - centroids[j])
                if dist < 2.0:  # Fused or close rings
                    cluster.update(aromatic_rings[j])
                    clustered[j] = True

            if len(cluster) > len(aromatic_rings[i]):  # Only if clustered
                groups["aromatic_cluster"].append(sorted(cluster))
            clustered[i] = True

    # 8. Rigid fragments (rings + amide bonds)
    rigid_atoms = set()
    for ring in ring_info.AtomRings():
        rigid_atoms.update(ring)

    # Add amide bonds as rigid
    amide_patt = Chem.MolFromSmarts("[C](=O)[N]")
    if amide_patt:
        matches = mol.GetSubstructMatches(amide_patt)
        for match in matches:
            rigid_atoms.update(match)

    if rigid_atoms:
        groups["rigid_fragment"].append(sorted(rigid_atoms))

    # 9. Flexible linkers (rotatable bonds)
    rot_bonds_pattern = Lipinski.RotatableBondSmarts
    if rot_bonds_pattern:
        matches = mol.GetSubstructMatches(rot_bonds_pattern)
        for match in matches:
            # Get atoms connected by this rotatable bond
            bond_atoms = list(match)
            # Expand to include adjacent atoms (the linker segment)
            linker = set(bond_atoms)
            for idx in bond_atoms:
                atom = mol.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    # Only add if not in rigid fragment
                    if n_idx not in rigid_atoms:
                        linker.add(n_idx)
            if len(linker) > 2:  # At least 3 atoms
                groups["flexible_linker"].append(sorted(linker))

    return groups


def build_hyperedge_features(mol: Chem.Mol, hyperedge_members: List[List[int]],
                             hyperedge_types: List[str],
                             type_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Build feature matrix for hyperedges.

    Features per hyperedge:
        - Type one-hot (9 types)
        - Size (1 dim)
        - Centroid 3D coordinates (3 dims)
        - Mean aromaticity (1 dim)
        - Mean formal charge (1 dim)
    Total: 9 + 1 + 3 + 1 + 1 = 15 dims
    """
    n_types = len(type_to_idx)
    feats = []

    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    for members, htype in zip(hyperedge_members, hyperedge_types):
        # Type one-hot
        type_feats = [0.0] * n_types
        type_feats[type_to_idx[htype]] = 1.0

        # Size
        size = float(len(members))

        # Centroid (if 3D available)
        # NOTE: Here also segfaults can happen
        if conf:
            coords = []
            for atom_idx in members:
                pos = conf.GetAtomPosition(atom_idx)
                coords.append([pos.x, pos.y, pos.z])
            coords_array = np.array(coords)
            centroid = coords_array.mean(axis=0)
        else:
            centroid = np.array([0.0, 0.0, 0.0])

        # Mean aromaticity
        aromatic = np.mean([mol.GetAtomWithIdx(i).GetIsAromatic() for i in members])

        # Mean formal charge
        charge = np.mean([mol.GetAtomWithIdx(i).GetFormalCharge() for i in members])

        # Concatenate
        feat_vec = type_feats + [size] + list(centroid) + [aromatic, charge]
        feats.append(feat_vec)

    return np.array(feats, dtype=np.float32)


def build_ligand_graph(row: LigandRow, fe: FeatureSpec) -> Optional[LigandGraph]:
    """
    Build complete ligand graph with atom graph and pharmacophore hypergraph.

    Args:
        row: LigandRow with ligand metadata and MOL2 path
        fe: FeatureSpec with feature dimensions and flags

    Returns:
        LigandGraph with atom features, bonds, 3D coords, and hypergraph
        Returns None if molecule cannot be loaded or is invalid
    """
    try:
        # Load molecule with 3D coordinates
        mol = load_mol_from_mol2(row.mol2_path)
        if mol is None:
            return None

        # Validate molecule has atoms
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None

        # Extract atom features
        x_list = []
        for atom in mol.GetAtoms():
            try:
                feat = get_atom_features(atom, mol)
                x_list.append(feat)
            except Exception as e:
                # Skip problematic atoms
                print(f"Error getting atom features for {row.ligand_id}: {e}", file=sys.stderr)
                return None

        if len(x_list) == 0:
            return None

        x = np.stack(x_list, axis=0)  # (N, d_atom)

        # Get 3D coordinates
        # NOTE: Here also segfaults can happen, culprit seems to be GetAtomPosition and GetConformer
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            pos_list = []
            for i in range(n_atoms):
                atom_pos = conf.GetAtomPosition(i)
                pos_list.append([atom_pos.x, atom_pos.y, atom_pos.z])
            pos = np.array(pos_list, dtype=np.float32)
        else:
            # Fallback: generate 3D if missing (shouldn't happen with MOL2)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=SEED)     # type: ignore[attr-defined]
                conf = mol.GetConformer()
                pos_list = []
                for i in range(n_atoms):
                    atom_pos = conf.GetAtomPosition(i)
                    pos_list.append([atom_pos.x, atom_pos.y, atom_pos.z])
                pos = np.array(pos_list, dtype=np.float32)
            except Exception:
                # If 3D generation fails, use zeros
                pos = np.zeros((n_atoms, 3), dtype=np.float32)

        # Extract bond features
        edge_index_list = []
        edge_attr_list = []
        for bond in mol.GetBonds():
            try:
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                # Add both directions for undirected graph
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])

                feat = get_bond_features(bond, fe.use_3d_distances, pos)
                edge_attr_list.append(feat)
                edge_attr_list.append(feat)  # Same features for both directions
            except Exception as e:
                # Skip problematic bonds
                print(f"Error getting bond features for {row.ligand_id}: {e}", file=sys.stderr)
                continue

        if len(edge_index_list) > 0:
            edge_index = np.array(edge_index_list, dtype=np.int64).T  # (2, E)
            edge_attr = np.stack(edge_attr_list, axis=0)  # (E, d_edge)
        else:
            # Single atom molecule
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 6), dtype=np.float32)

        # Build hypergraph if enabled
        hyperedge_members = None
        hyperedge_attr = None

        if fe.use_pharmacophore_hypergraph:
            try:
                pharmacophores = detect_pharmacophores(mol)

                # Flatten into lists
                hyperedge_members = []
                hyperedge_types = []

                for htype in fe.hyperedge_types:
                    if htype in pharmacophores:
                        for group in pharmacophores[htype]:
                            if len(group) > 0:  # Only add non-empty groups
                                hyperedge_members.append(group)
                                hyperedge_types.append(htype)

                # Build hyperedge features
                if len(hyperedge_members) > 0:
                    type_to_idx = {t: i for i, t in enumerate(fe.hyperedge_types)}
                    hyperedge_attr = build_hyperedge_features(mol, hyperedge_members,
                                                              hyperedge_types, type_to_idx)
            except Exception as e:
                # If pharmacophore detection fails, continue without hypergraph
                print(f"Warning: Pharmacophore detection failed for {row.ligand_id}: {e}",
                      file=sys.stderr)

        # Compute some global properties
        props = {}
        try:
            props["mw"] = Descriptors.MolWt(mol)    # type: ignore[attr-defined]
            props["logp"] = Descriptors.MolLogP(mol)    # type: ignore[attr-defined]
            props["tpsa"] = Descriptors.TPSA(mol)   # type: ignore[attr-defined]
            props["n_hbd"] = Descriptors.NumHDonors(mol)    # type: ignore[attr-defined]
            props["n_hba"] = Descriptors.NumHAcceptors(mol)   # type: ignore[attr-defined]
            props["n_rotatable"] = Descriptors.NumRotatableBonds(mol)   # type: ignore[attr-defined]
        except Exception:
            pass

        return LigandGraph(
            ligand_id=row.ligand_id,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            props=props,
            hyperedge_members=hyperedge_members,
            hyperedge_attr=hyperedge_attr
        )
    except Exception as e:
        # Catch any RDKit errors that might cause segfaults
        print(f"Error building ligand graph for {row.ligand_id}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None
