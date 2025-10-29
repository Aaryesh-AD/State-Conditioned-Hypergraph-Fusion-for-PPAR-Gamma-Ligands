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


import random
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
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
    df: pd.DataFrame,
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 16,
    label_mode: str = "4way",
) -> Dict[str, List[str]]:
    """
    Stratified scaffold-based split that maintains scaffold integrity and class balance.

    Args:
        df: DataFrame with columns [ligand_id, smiles, class_label, is_decoy]
        frac: Tuple of (train_frac, val_frac, test_frac), must sum to 1.0
        seed: Random seed for reproducibility
        label_mode: "4way" (agonist/antagonist/agonist_decoy/antagonist_decoy) or "2way"

    Returns:
        Dictionary with keys ["train", "val", "test"], values are lists of ligand_ids

    Strategy:
        1. Group ligands by scaffold (entire scaffold, not per-label)
        2. Characterize each scaffold by its label composition
        3. Greedily assign scaffolds to splits to balance class proportions
        4. Each scaffold's ALL ligands go to the same split (scaffold integrity)
    """
    assert abs(sum(frac) - 1.0) < 1e-6, f"fractions must sum to 1.0, got {sum(frac)}"
    rng = random.Random(seed)

    # Label Normalization
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
        else:  # 2way mode
            if "agonist" in class_label:
                return "agonist"
            if "antagonist" in class_label:
                return "antagonist"
            return "unknown"

    # Prepare dataframe
    df = df.copy()

    # Compute scaffolds if not already present
    if "scaffold" not in df.columns:
        from pprag.dataio.murcko import scaffold_of_smiles
        df["scaffold"] = df["smiles"].apply(scaffold_of_smiles)

    df["label_norm"] = df.apply(norm_label, axis=1)
    df = df.loc[df["label_norm"] != "unknown"].reset_index(drop=True)

    # Group ALL ligands by scaffold, regardless of label
    scaffold_groups: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        scaffold_groups[row["scaffold"]].append(row["ligand_id"])

    # For each scaffold, count how many ligands of each label it contains
    from typing import TypedDict

    class ScaffoldInfo(TypedDict):
        scaffold: str
        ligand_ids: list[str]
        size: int
        label_counts: dict[str, int]

    scaffold_info: list[ScaffoldInfo] = []
    for scaffold, ligand_ids in scaffold_groups.items():
        # Get label distribution for this scaffold
        ligand_ids_list: list[str] = list(ligand_ids)
        scaffold_df = df[df["ligand_id"].isin(ligand_ids_list)]
        label_counts_dict: dict[str, int] = dict(Counter(scaffold_df["label_norm"]))

        scaffold_info.append({
            "scaffold": str(scaffold),
            "ligand_ids": ligand_ids_list,
            "size": int(len(ligand_ids_list)),
            "label_counts": label_counts_dict,
        })

    # Shuffle for randomness, then sort by size (greedy works better with large first)
    rng.shuffle(scaffold_info)
    scaffold_info.sort(key=lambda x: x["size"], reverse=True)

    all_labels = sorted(set(df["label_norm"]))

    # Total count per label
    total_label_counts = {
        lbl: int((df["label_norm"] == lbl).sum())
        for lbl in all_labels
    }

    # Target counts per split per label
    targets = {
        split: {
            lbl: int(round(total_label_counts[lbl] * f))
            for lbl in all_labels
        }
        for split, f in zip(["train", "val", "test"], frac)
    }

    # adjust train to ensure exact total
    for lbl in all_labels:
        total_target = sum(targets[sp][lbl] for sp in ["train", "val", "test"])
        diff = total_label_counts[lbl] - total_target
        if diff != 0:
            targets["train"][lbl] += diff

    # Greedy Scaffold Assignment
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    assigned: Dict[str, Dict[str, int]] = {
        split: {lbl: 0 for lbl in all_labels}
        for split in ["train", "val", "test"]
    }

    for scf_data in scaffold_info:
        ligand_ids = scf_data["ligand_ids"]
        label_counts = dict(scf_data["label_counts"])

        # Calculate "need score" for each split
        # Score = sum of (remaining need for each label) * (count of that label in scaffold)
        # Higher score = this split needs this scaffold's labels more urgently
        scores = {}
        for split in ["train", "val", "test"]:
            score = 0
            for lbl, count in label_counts.items():
                remaining_need = targets[split][lbl] - assigned[split][lbl]
                score += remaining_need * count
            scores[split] = score

        # Assign scaffold to split with highest score
        # Tie-breaker: prefer train > val > test
        best_split = max(
            scores.items(),
            key=lambda kv: (kv[1], {"train": 2, "val": 1, "test": 0}[kv[0]])
        )[0]

        # If all needs are negative (overallocated), assign to least full split
        if scores[best_split] < 0:
            best_split = min(
                assigned.items(),
                key=lambda kv: sum(kv[1].values())  # Total ligands assigned to this split
            )[0]

        # Assign ENTIRE scaffold to this split
        out[best_split].extend(ligand_ids)

        # Update assigned counts for each label in this scaffold
        for lbl, count in label_counts.items():
            assigned[best_split][lbl] += count

    # Check if each split has representation of each class
    for split in ["train", "val", "test"]:
        split_labels = set(df[df["ligand_id"].isin(out[split])]["label_norm"])
        missing = set(all_labels) - split_labels
        if missing:
            import sys
            print(
                f"Warning: {split} split missing classes: {missing}",
                file=sys.stderr
            )
            print(
                "  This may occur with very small datasets or highly imbalanced classes.",
                file=sys.stderr
            )
            print(
                "  Consider using --no-stratified for pure scaffold-based splitting.",
                file=sys.stderr
            )

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
