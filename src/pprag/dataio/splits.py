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
from collections import defaultdict
from pprag.dataio.load_labels import load_ligands_csv
from pprag.dataio.murcko import group_by_scaffold
from pprag.dataio.schema import global_seed

SEED = global_seed()

try:
    from murcko import murcko_smiles
    _HAS_RD = True
except Exception:
    _HAS_RD = False


def _get_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES; fall back to raw SMILES if RDKit/helper not available."""
    if _HAS_RD:
        try:
            return murcko_smiles(smiles)
        except Exception:
            pass
    return smiles


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


def stratified_scaffold_split(
    df,                           # pandas DataFrame with columns: ligand_id, smiles, class_label, is_decoy
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = SEED,
    label_mode: str = "4way",     # "4way": agonist, antagonist, agonist_decoy, antagonist_decoy
):
    """
    Keep entire Murcko scaffolds intact AND stratify class proportions across train/val/test.
    Returns: {"train": [...ligand_ids...], "val": [...], "test": [...]}
    """
    assert abs(sum(frac) - 1.0) < 1e-6, "fractions must sum to 1.0"
    rng = random.Random(seed)

    # Normalize label to classes stratify on
    def norm_label(row):
        class_label = str(row["class_label"]).strip().lower()
        is_decoy = int(row["is_decoy"]) == 1
        if label_mode == "4way":
            if "agonist" in class_label and is_decoy:
                return "agonist_decoy"
            if "antagonist" in class_label and is_decoy:
                return "antagonist_decoy"
            if "agonist" in class_label:
                return "agonist"
            if "antagonist" in class_label:
                return "antagonist"
            return "unknown"
        else:
            if "agonist" in class_label:
                return "agonist"
            if "antagonist" in class_label:
                return "antagonist"
            return "unknown"

    df = df.copy()
    if "scaffold" not in df.columns:
        df["scaffold"] = df["smiles"].map(_get_scaffold)

    df["label_norm"] = df.apply(norm_label, axis=1)
    df = df.loc[df["label_norm"] != "unknown"].reset_index(drop=True)

    # Build scaffold groups per label
    groups_by_label: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(lambda: defaultdict(list))  # label -> scaffold -> [ligand_id]
    for _, row in df.iterrows():
        groups_by_label[row["label_norm"]][row["scaffold"]].append(row["ligand_id"])

    labels = sorted(groups_by_label.keys())
    label_counts = {lbl: sum(len(v) for v in groups_by_label[lbl].values()) for lbl in labels}

    # Target counts per split, per label
    targets = {
        lbl: {
            "train": int(round(label_counts[lbl] * frac[0])),
            "val": int(round(label_counts[lbl] * frac[1])),
            "test": int(round(label_counts[lbl] * frac[2])),
        } for lbl in labels
    }
    # fix rounding drift
    for lbl in labels:
        tot = label_counts[lbl]
        alloc = targets[lbl]["train"] + targets[lbl]["val"] + targets[lbl]["test"]
        if alloc != tot:
            targets[lbl]["train"] += (tot - alloc)

    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    assigned = {lbl: {"train": 0, "val": 0, "test": 0} for lbl in labels}

    # Greedy per-label assignment of whole scaffolds
    for lbl in labels:
        items = [(scf, len(ids)) for scf, ids in groups_by_label[lbl].items()]
        rng.shuffle(items)
        items.sort(key=lambda x: x[1], reverse=True)  # place big scaffolds first
        for scf, sz in items:
            need = {sp: targets[lbl][sp] - assigned[lbl][sp] for sp in ("train", "val", "test")}
            # best split = highest remaining need (tie-breaker: train > val > test)
            best = max(need.items(), key=lambda kv: (kv[1], {"train": 2, "val": 1, "test": 0}[kv[0]]))[0]
            if need[best] <= 0:
                best = min(assigned[lbl].items(), key=lambda kv: kv[1])[0]
            out[best].extend(groups_by_label[lbl][scf])
            assigned[lbl][best] += sz

    # Minimal guarantee: if a split lacks a class entirely, move the smallest scaffold of that class to it
    for lbl in labels:
        present = {sp: any(df.loc[df["ligand_id"].isin(out[sp]), "label_norm"] == lbl) for sp in ("train", "val", "test")}
        if all(present.values()):
            continue
        # Build scaffold→(size,split,ids) among already assigned ligands of this label
        scf_info = []
        for sp in ("train", "val", "test"):
            ligs = [lid for lid in out[sp] if df.loc[df["ligand_id"] == lid, "label_norm"].iat[0] == lbl]
            tmp = defaultdict(list)
            for lid in ligs:
                scf = df.loc[df["ligand_id"] == lid, "scaffold"].iat[0]
                tmp[scf].append(lid)
            for scf, ids in tmp.items():
                scf_info.append((scf, len(ids), sp, ids))
        scf_info.sort(key=lambda x: x[1])  # smallest first

        for sp in ("train", "val", "test"):
            if not present[sp]:
                # move the smallest available scaffold of this label from any other split
                for scf, sz, donor, ids in scf_info:
                    if donor == sp:
                        continue
                    for lid in ids:
                        out[donor].remove(lid)
                    out[sp].extend(ids)
                    present[sp] = True
                    break

    return out


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
