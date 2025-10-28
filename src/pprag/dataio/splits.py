#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataio/splits.py: Dataset splitting and phase partitioning utilities.

Handles the creation of train/val/test splits using Murcko scaffold-based
splitting to ensure no data leakage and proper generalization to novel chemical scaffolds.
It also partitions the data into different phases for pretraining, zero-shot, and few-shot
learning scenarios specific to the PPAR-gamma agonist/antagonist prediction task.

Split Strategy:
    - Murcko scaffold-based splitting prevents data leakage
    - Ensures model generalizes to novel chemical scaffolds
    - Maintains class balance where possible

Phase Partitions:
    - Pretrain: Agonist ligands only (no decoys) for self-supervised learning
    - Zero-shot: Antagonists + decoys from test set for evaluation without training
    - Few-shot pool: All 9 antagonist ligands (no decoys) for few-shot experiments

Functions:
    murcko_scaffold_split: Split ligands by Murcko scaffolds into train/val/test
    write_phase_partitions: Generate phase-specific subsets and save to JSON

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

from typing import List, Tuple, Dict
import random
import json
from pathlib import Path
from pprag.dataio.load_labels import load_ligands_csv
from pprag.dataio.murcko import group_by_scaffold
from pprag.dataio.schema import global_seed

SEED = global_seed()


def murcko_scaffold_split(lig_csv: str | Path,
                          seed: int = SEED,
                          train_frac: float = 0.8,
                          val_frac: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
    rows = load_ligands_csv(lig_csv)
    ids = [r.ligand_id for r in rows]
    smiles = [r.smiles for r in rows]
    groups = group_by_scaffold(ids, smiles)

    scafs = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(scafs)
    n = len(scafs)
    n_tr = int(n * train_frac)
    n_val = int(n * val_frac)

    split = {"train": set(scafs[:n_tr]),
             "val": set(scafs[n_tr: n_tr + n_val]),
             "test": set(scafs[n_tr + n_val:])}

    tr_ids, va_ids, te_ids = [], [], []
    # place ligands by their scaffold group
    for scaf, lid_list in groups.items():
        bucket = "train" if scaf in split["train"] else ("val" if scaf in split["val"] else "test")
        if bucket == "train":
            tr_ids.extend(lid_list)
        elif bucket == "val":
            va_ids.extend(lid_list)
        else:
            te_ids.extend(lid_list)
    return tr_ids, va_ids, te_ids


def write_phase_partitions(lig_csv: str | Path, out_dir: str | Path,
                           train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_ligands_csv(lig_csv)
    row_by_id = {r.ligand_id: r for r in rows}
    # pretrain set: agonist ligands only, no decoys
    pretrain_ids = []
    for lid in train_ids:
        if row_by_id[lid].class_label == "agonist" and row_by_id[lid].is_decoy == 0:
            pretrain_ids.append(lid)

    # zero-shot set: antagonists + antagonist decoys (use test side by default)
    zero_shot_ids = []
    for lid in test_ids:
        if row_by_id[lid].class_label == "antagonist":
            zero_shot_ids.append(lid)

    # few-shot pool: the 9 antagonist ligands (no decoys)
    fewshot_pool = []
    all_ids = train_ids + val_ids + test_ids
    for lid in all_ids:
        if row_by_id[lid].class_label == "antagonist" and row_by_id[lid].is_decoy == 0:
            fewshot_pool.append(lid)

    def dump(lst: List[str], name: str) -> Path:
        p = out / f"{name}.json"
        p.write_text(json.dumps(lst, indent=2))
        return p

    files = {
        "train_ids": dump(train_ids, "train_ids"),
        "val_ids": dump(val_ids, "val_ids"),
        "test_ids": dump(test_ids, "test_ids"),
        "pretrain_ids": dump(pretrain_ids, "pretrain_ids"),
        "zero_shot_ids": dump(zero_shot_ids, "zero_shot_ids"),
        "fewshot_pool": dump(fewshot_pool, "fewshot_pool"),
    }
    return files
