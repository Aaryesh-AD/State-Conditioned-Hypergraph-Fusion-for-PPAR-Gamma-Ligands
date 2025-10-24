#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/murcko.py: Functions to compute and group by Murcko scaffolds.

Murcko Scaffold and ligand Normalization Utilities

Functions:
    scaffold_of_smiles: Extract Murcko scaffold from a SMILES string
    add_scaffolds: Generate scaffolds for a list of SMILES strings
    group_by_scaffold: Group molecular IDs by their Murcko scaffold
    canonical_smiles_from_mol2: Get canonical isomeric SMILES from a MOL2 file

Author: Aaryesh Deshpande
Last Modified: 10/24/2025
"""

from typing import Dict, List
from rdkit import Chem
from pathlib import Path
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_of_smiles(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc, isomericSmiles=False, canonical=True) if sc else ""


def add_scaffolds(smiles: List[str]) -> List[str]:
    return [scaffold_of_smiles(s) for s in smiles]


def group_by_scaffold(ids: List[str], smiles: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for lig_id, smi in zip(ids, smiles):
        sc = scaffold_of_smiles(smi)
        groups.setdefault(sc, []).append(lig_id)
    return groups


def canonical_smiles_from_mol2(path: str | Path) -> str | None:
    """
    Returns canonical isomeric SMILES for the first molecule in a MOL2 file.
    """
    try:
        mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    except Exception:
        mol = None
    if mol is None:
        try:
            mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=True)
        except Exception:
            mol = None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
