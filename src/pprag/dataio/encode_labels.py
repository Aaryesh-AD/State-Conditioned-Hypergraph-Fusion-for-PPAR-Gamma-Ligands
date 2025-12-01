#!/usr/bin/env python3
"""
Utility Post add
Encode class labels from `Data/meta/ligands_clean.csv` into two CSVs:
- `Data/meta/ligands_labels_encoded.csv` (ligand_id,label_id)
- `Data/meta/ligands_labels_onehot.csv` (ligand_id,agonist,antagonist,agonist_decoy,antagonist_decoy)

Mapping used:
  agonist -> 1
  antagonist -> 2
  agonist decoy -> 3
  antagonist decoy -> 4
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'Data' / 'meta' / 'ligands_clean.csv'
OUT_ENCODED = ROOT / 'Data' / 'meta' / 'ligands_labels_encoded.csv'
OUT_ONEHOT = ROOT / 'Data' / 'meta' / 'ligands_labels_onehot.csv'

LABEL_TO_ID = {
    'agonist': 1,
    'antagonist': 2,
    'agonist decoy': 3,
    'antagonist decoy': 4,
    'agonist_decoy': 3,
    'antagonist_decoy': 4,
}

ONEHOT_KEYS = ['agonist', 'antagonist', 'agonist_decoy', 'antagonist_decoy']


def normalize_label(s: str) -> str:
    if s is None:
        return ''
    return s.strip().lower().replace('-', ' ').replace('/', ' ')


def label_to_onehot(label: str):
    norm = normalize_label(label)
    # accept both 'agonist decoy' and 'agonist_decoy'
    if norm in ('agonist decoy', 'agonist_decoy'):
        return {'agonist': 0, 'antagonist': 0, 'agonist_decoy': 1, 'antagonist_decoy': 0}
    if norm in ('antagonist decoy', 'antagonist_decoy'):
        return {'agonist': 0, 'antagonist': 0, 'agonist_decoy': 0, 'antagonist_decoy': 1}
    if norm == 'agonist':
        return {'agonist': 1, 'antagonist': 0, 'agonist_decoy': 0, 'antagonist_decoy': 0}
    if norm == 'antagonist':
        return {'agonist': 0, 'antagonist': 1, 'agonist_decoy': 0, 'antagonist_decoy': 0}
    # Unknown label -> zeros
    return {'agonist': 0, 'antagonist': 0, 'agonist_decoy': 0, 'antagonist_decoy': 0}


def label_to_id(label: str):
    norm = normalize_label(label)
    return LABEL_TO_ID.get(norm, 0)


def main():
    if not INPUT.exists():
        print(f"Input file not found: {INPUT}")
        return

    rows = []
    with INPUT.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            ligand_id = r.get('ligand_id') or r.get('ligand') or r.get('ligand id')
            class_label = r.get('class_label') or r.get('label') or r.get('class')
            is_decoy_val = r.get('is_decoy') or r.get('isdecoy') or r.get('is_decoy?')
            if not ligand_id:
                # skip rows without id
                continue
            ligand_id = ligand_id.strip()
            class_label = (class_label.strip() if class_label else '')
            # determine decoy boolean robustly
            is_decoy = False
            if is_decoy_val is not None:
                s = str(is_decoy_val).strip()
                if s == '1' or s.lower() in ('true', 't', 'yes', 'y'):
                    is_decoy = True
            # compute effective label using both class_label and is_decoy
            norm = normalize_label(class_label)
            if is_decoy:
                if norm == 'agonist':
                    effective = 'agonist decoy'
                elif norm == 'antagonist':
                    effective = 'antagonist decoy'
                else:
                    # unknown class but flagged as decoy -> treat as unknown decoy
                    effective = ''
            else:
                if norm == 'agonist':
                    effective = 'agonist'
                elif norm == 'antagonist':
                    effective = 'antagonist'
                else:
                    effective = ''

            rows.append((ligand_id, effective))

    # Write encoded (id) CSV
    with OUT_ENCODED.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ligand_id', 'label_id'])
        for lid, lbl in rows:
            writer.writerow([lid, label_to_id(lbl)])

    # Write one-hot CSV
    with OUT_ONEHOT.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ligand_id'] + ONEHOT_KEYS)
        for lid, lbl in rows:
            oh = label_to_onehot(lbl)
            writer.writerow([lid, oh['agonist'], oh['antagonist'], oh['agonist_decoy'], oh['antagonist_decoy']])

    print(f"Processed {len(rows)} rows")
    print(f"Wrote: {OUT_ENCODED}")
    print(f"Wrote: {OUT_ONEHOT}")


if __name__ == '__main__':
    main()
