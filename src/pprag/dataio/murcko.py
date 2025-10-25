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

import sys
from typing import Dict, List
from rdkit import Chem
from pathlib import Path
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging

# Suppress RDKit warnings for invalid SMILES during scaffold extraction
# (many public datasets have some malformed SMILES strings)
logging.getLogger('rdkit').setLevel(logging.ERROR)


def scaffold_of_smiles(smi: str) -> str:
    """
    Extract Murcko scaffold from a SMILES string.
    Returns empty string if SMILES is invalid or has no scaffold.

    Attempts multiple parsing strategies to handle problematic SMILES:
    1. Standard sanitization
    2. Without sanitization (then fix manually)
    3. With explicit hydrogen removal
    """
    # Strategy 1: Standard parsing with sanitization
    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
        if m is not None:
            sc = MurckoScaffold.GetScaffoldForMol(m)
            return Chem.MolToSmiles(sc, isomericSmiles=False, canonical=True) if sc else ""
    except Exception:
        pass

    # Strategy 2: Parse without sanitization, then try to fix
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is not None:
            # Try to fix common issues
            Chem.SanitizeMol(m, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            # Remove hydrogens that might cause valence issues
            m = Chem.RemoveHs(m, sanitize=False)
            # Try to kekulize
            Chem.Kekulize(m, clearAromaticFlags=True)
            # Now sanitize fully
            Chem.SanitizeMol(m)
            sc = MurckoScaffold.GetScaffoldForMol(m)
            return Chem.MolToSmiles(sc, isomericSmiles=False, canonical=True) if sc else ""
    except Exception:
        pass

    # Strategy 3: Try InChI conversion as last resort (can fix some issues)
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is not None:
            # Convert to InChI and back (this can fix some structural issues)
            inchi = Chem.MolToInchi(m)
            if inchi:
                m_fixed = Chem.MolFromInchi(inchi)
                if m_fixed is not None:
                    sc = MurckoScaffold.GetScaffoldForMol(m_fixed)
                    return Chem.MolToSmiles(sc, isomericSmiles=False, canonical=True) if sc else ""
    except Exception:
        pass

    # All strategies failed - return empty scaffold
    return ""


def add_scaffolds(smiles: List[str]) -> List[str]:
    return [scaffold_of_smiles(s) for s in smiles]


def group_by_scaffold(ids: List[str], smiles: List[str]) -> Dict[str, List[str]]:
    """
    Group molecular IDs by their Murcko scaffold.

    Returns:
        Dictionary mapping scaffold SMILES to list of molecule IDs.
        Molecules that fail parsing are grouped under empty string key.
    """
    groups: Dict[str, List[str]] = {}
    failed_count = 0

    for lig_id, smi in zip(ids, smiles):
        sc = scaffold_of_smiles(smi)
        if sc == "" and smi != "":  # Empty scaffold from non-empty SMILES = parsing failure
            failed_count += 1
        groups.setdefault(sc, []).append(lig_id)

    # Log summary to stderr if there were failures
    if failed_count > 0:
        total = len(ids)
        success_rate = (total - failed_count) / total * 100
        print(f"  [Scaffold parsing: {total - failed_count}/{total} successful ({success_rate:.1f}%)]",
              file=sys.stderr)

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
