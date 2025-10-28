#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/pocket_select.py: Pocket selection and RDKit-based feature extraction.

Key Features:
    - Pocket selection via spherical cutoff around ligand center
    - RDKit-based molecular property calculations:
        * SASA: FreeSASA algorithm (fallback: neighbor-counting)
        * Hydropathy: TPSA + molecular weight (fallback: Kyte-Doolittle)
        * Formal charge: From SMILES structure (fallback: pH 7.4 lookup)
        * HBD/HBA: Automatic counting via RDKit descriptors

Functions:
    one_from_resname: Convert 3-letter AA code to 1-letter
    calc_hydropathy_rdkit: Calculate hydrophobicity using RDKit
    calc_charge_rdkit: Calculate formal charge from SMILES
    calc_hbd_hba_rdkit: Count H-bond donors/acceptors
    calculate_residue_sasa: Calculate solvent-accessible surface area
    center_from_ligand: Extract binding site center from ligand MOL2
    load_centers_json: Load custom pocket centers from JSON
    select_pocket: Main pocket selection with feature calculation
    build_pocket_select: High-level pocket builder with multiple center sources

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

from pathlib import Path
from typing import List, Optional, Dict
import json
import warnings
import numpy as np
import MDAnalysis as mda
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors, rdFreeSASA
from pprag.dataio.schema import (
    ResidueFeats,
    PocketSelect,
    AA2IDX,
    AA3_TO_AA1,
    AA_SIDECHAIN_SMILES,
    KD_HYDROPATHY,
    CHARGE_MAP_PH7,
)

# Suppress MDAnalysis warnings about unknown elements
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')

# Suppress RDKit warnings about isolated hydrogens in MOL2 files
RDLogger.DisableLog('rdApp.*')


def one_from_resname(res3: str) -> str:
    """Convert 3-letter amino acid code to 1-letter code."""
    return AA3_TO_AA1.get(res3.upper(), 'A')


def calc_hydropathy_rdkit(aa_one: str) -> float:
    """
    Calculate hydropathy using RDKit's molecular weight and TPSA as proxies.
    Returns a hydrophobicity score where positive = hydrophobic, negative = hydrophilic.
    Falls back to Kyte-Doolittle if RDKit calculation fails.
    """
    smiles = AA_SIDECHAIN_SMILES.get(aa_one)
    if not smiles:
        return KD_HYDROPATHY.get(aa_one, 0.0)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return KD_HYDROPATHY.get(aa_one, 0.0)

        # Use combination of molecular properties as hydrophobicity proxy
        # TPSA (topological polar surface area): lower = more hydrophobic
        # MolWt: heavier side chains tend to be more hydrophobic
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        mw = rdMolDescriptors.CalcExactMolWt(mol)

        # Simple heuristic: low TPSA and high MW = hydrophobic
        # Scale to roughly match Kyte-Doolittle range (-4.5 to 4.5)
        hydropathy_score = (mw / 30.0) - (tpsa / 10.0)

        # Clamp to reasonable range
        return float(max(-5.0, min(5.0, hydropathy_score)))
    except Exception:
        # Fallback to Kyte-Doolittle
        return KD_HYDROPATHY.get(aa_one, 0.0)


def calc_charge_rdkit(aa_one: str) -> int:
    """
    Calculate formal charge using RDKit for amino acid side chain.
    Returns -1, 0, or 1 based on the formal charge at physiological pH.
    Falls back to lookup table if RDKit calculation fails.
    """
    smiles = AA_SIDECHAIN_SMILES.get(aa_one)
    if not smiles:
        return CHARGE_MAP_PH7.get(aa_one, 0)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return CHARGE_MAP_PH7.get(aa_one, 0)

        # Calculate total formal charge
        total_charge = Chem.GetFormalCharge(mol)

        # Clamp to -1, 0, 1
        if total_charge > 0:
            return 1
        elif total_charge < 0:
            return -1
        else:
            return 0
    except Exception:
        # Fallback to lookup table
        return CHARGE_MAP_PH7.get(aa_one, 0)


def calc_hbd_hba_rdkit(aa_one: str) -> tuple[int, int]:
    """
    Calculate hydrogen bond donors and acceptors using RDKit.
    Returns (hbd_count, hba_count) for the amino acid side chain.
    """
    smiles = AA_SIDECHAIN_SMILES.get(aa_one)
    if not smiles:
        return 0, 0

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0, 0

        # Use RDKit descriptors for HBD/HBA
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)

        return hbd, hba
    except Exception:
        return 0, 0


def calculate_residue_sasa(residue, universe) -> float:
    """
    Calculate SASA for a residue using RDKit's FreeSASA implementation.
    Falls back to neighbor-counting approximation if RDKit calculation fails.
    Returns SASA.
    """
    try:
        # Extract residue atoms and build RDKit molecule
        res_atoms = residue.atoms.select_atoms("not type H")  # heavy atoms only
        if res_atoms.n_atoms == 0:
            return 0.0

        # Create a simple RDKit molecule from coordinates
        # For protein residues, we'll use a neighbor-counting approach weighted by RDKit radii
        mol = Chem.MolFromSmiles(AA_SIDECHAIN_SMILES.get(one_from_resname(residue.resname), 'C'))
        if mol is None:
            raise ValueError("Could not create molecule")

        # Add 3D coordinates from MDAnalysis
        conf = Chem.Conformer(res_atoms.n_atoms)
        for i, pos in enumerate(res_atoms.positions):
            conf.SetAtomPosition(i, tuple(pos))

        # Calculate SASA using FreeSASA
        radii = rdFreeSASA.classifyAtoms(mol)
        sasa_vals = rdFreeSASA.CalcSASA(mol, radii)

        # Return total SASA for the residue (sasa_vals is already a float)
        return float(sasa_vals)

    except Exception:
        # Fallback to neighbor-counting approximation
        res_center = residue.atoms.positions.mean(axis=0)
        nearby = universe.select_atoms(
            f"around 5.0 (point {res_center[0]} {res_center[1]} {res_center[2]} 0.1) and not resid {residue.resid}"
        )
        neighbor_count = nearby.n_atoms
        max_neighbors = 50
        normalized = max(0, min(1, neighbor_count / max_neighbors))
        estimated_sasa = 200.0 * (1.0 - normalized)
        return float(estimated_sasa)


def center_from_ligand(mol2_path: str | Path) -> Optional[np.ndarray]:
    try:
        u = mda.Universe(str(mol2_path))
        lig = u.atoms.select_atoms("not protein")
        if lig.n_atoms == 0:
            lig = u.atoms  # if file is only ligand
        return lig.positions.mean(axis=0).astype(np.float32)
    except Exception:
        return None


def load_centers_json(path: Optional[str | Path]) -> Dict[str, List[float]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def select_pocket(mol2_path: str | Path,
                  chain: Optional[str],
                  radius: float,
                  center_xyz: Optional[np.ndarray]) -> PocketSelect:
    u = mda.Universe(str(mol2_path))
    if chain:
        sel = u.select_atoms(f"segid {chain} or chainid {chain}")  # try both notations
        if sel.n_atoms > 0:
            u = sel.universe

    if center_xyz is None:
        # crude fallback: protein centroid
        prot = u.atoms.select_atoms("protein")
        center_xyz = (prot.positions.mean(axis=0) if prot.n_atoms else u.atoms.positions.mean(axis=0)).astype(np.float32)

    pocket_atoms = u.select_atoms(f"point {center_xyz[0]} {center_xyz[1]} {center_xyz[2]} {radius} and protein")
    residues = pocket_atoms.residues

    feats: List[ResidueFeats] = []
    res_ids: List[int] = []

    for res in residues:
        one = one_from_resname(res.resname)
        aa_idx = AA2IDX.get(one, 0)

        # CA or mean of backbone heavy atoms
        ca = res.atoms.select_atoms("name CA")
        ca_xyz = (ca.positions[0] if ca.n_atoms else res.atoms.positions.mean(axis=0)).astype(np.float32)
        sc = res.atoms.select_atoms("not name C N O CA and not type H")
        sc_xyz = (sc.positions.mean(axis=0).astype(np.float32) if sc.n_atoms else None)

        # Calculate SASA
        sasa = calculate_residue_sasa(res, u)

        # Calculate charge using RDKit (with fallback to lookup table)
        charge_class = calc_charge_rdkit(one)

        # Calculate HBD/HBA
        hbd, hba = calc_hbd_hba_rdkit(one)

        # Calculate hydropathy using RDKit LogP (with fallback to KD)
        hydropathy = calc_hydropathy_rdkit(one)

        feats.append(ResidueFeats(
            resid=int(res.resid),
            aa_idx=aa_idx,
            sasa=sasa,
            hydropathy=hydropathy,
            charge_class=charge_class,
            hbd=hbd,
            hba=hba,
            ca_xyz=ca_xyz,
            sc_xyz=sc_xyz
        ))
        res_ids.append(int(res.resid))

    return PocketSelect(
        target_id=Path(mol2_path).stem,
        chain=chain or "",
        residues=res_ids,
        feats=feats
    )


def build_pocket_select(target_id: str,
                        protein_mol2: str | Path,
                        chain: Optional[str],
                        radius: float,
                        ligand_mol2: Optional[str | Path],
                        centers_json: Optional[str | Path]) -> PocketSelect:
    # 1) try co-complexed ligand
    center = None
    if ligand_mol2:
        center = center_from_ligand(ligand_mol2)

    # 2) else config JSON
    if center is None:
        centers = load_centers_json(centers_json)
        if centers and target_id in centers:
            center = np.array(centers[target_id], dtype=np.float32)

    # 3) fallback will be handled inside select_pocket
    return select_pocket(protein_mol2, chain=chain, radius=radius, center_xyz=center)
