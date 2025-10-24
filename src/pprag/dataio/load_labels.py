#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/load_labels.py: Load ligand and target labels from CSV files.

Author: Aaryesh Deshpande
Last Modified: 10/24/2025
"""

import csv
from pathlib import Path
from typing import List, cast, Literal
from .schema import LigandRow, TargetRow


def load_ligands_csv(csv_path: str | Path) -> List[LigandRow]:
    rows: List[LigandRow] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            class_label = d["class_label"].strip()
            if class_label not in ["agonist", "antagonist"]:
                raise ValueError(f"Invalid class_label value: {class_label}. Expected 'agonist' or 'antagonist'")
            rows.append(LigandRow(
                ligand_id=d["ligand_id"].strip(),
                smiles=d["smiles"].strip(),
                mol2_path=d["mol2_path"].strip(),
                class_label=cast(Literal["agonist", "antagonist"], class_label),        # "agonist" or "antagonist"
                is_decoy=int(d["is_decoy"]),
            ))
    return rows


def load_target_csv(csv_path: str | Path) -> List[TargetRow]:
    rows: List[TargetRow] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            state = d["state"].strip()
            if state not in ["agonist", "antagonist"]:
                raise ValueError(f"Invalid state value: {state}. Expected 'agonist' or 'antagonist'")
            rows.append(TargetRow(
                target_id=d["target_id"].strip(),
                mol2_path=d["mol2_path"].strip(),      # protein MOL2
                state=cast(Literal["agonist", "antagonist"], state),  # "agonist" or "antagonist"
                ligand_path=d.get("ligand_path") or None
            ))
    return rows
