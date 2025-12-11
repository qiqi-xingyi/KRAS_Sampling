# --*-- conding:utf-8 --*--
# @time:11/1/25 02:51
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prepare_splits.py


import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "training_data"
INPUT_FILE = DATA_DIR / "all_examples_grouped.jsonl"

TRAIN_OUT = DATA_DIR / "train.jsonl"
VALID_OUT = DATA_DIR / "valid.jsonl"
TEST_OUT  = DATA_DIR / "test.jsonl"

FEATURE_LIST_OUT = DATA_DIR / "feature_list.txt"
SCALER_OUT = DATA_DIR / "scaler.json"

# Split config
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15
SEED = 42

# Identifier/label fields to exclude from feature normalization
ID_FIELDS = {
    "group_id", "protein", "sequence", "bitstring", "residues"
}
LABEL_FIELDS = {
    "rel", "rmsd"
}
EXCLUDE_FIELDS = ID_FIELDS | LABEL_FIELDS


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            recs.append(obj)
    return recs


def split_by_protein(records: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    # Collect unique proteins
    proteins = sorted({str(r.get("protein")) for r in records})
    rng = random.Random(SEED)
    rng.shuffle(proteins)

    n = len(proteins)
    n_train = int(round(n * TRAIN_RATIO))
    n_valid = int(round(n * VALID_RATIO))
    # ensure full coverage
    n_test = max(0, n - n_train - n_valid)

    train_set = set(proteins[:n_train])
    valid_set = set(proteins[n_train:n_train + n_valid])
    test_set  = set(proteins[n_train + n_valid:])

    def pick(pset):
        return [r for r in records if str(r.get("protein")) in pset]

    return pick(train_set), pick(valid_set), pick(test_set)


def select_feature_columns(records: List[dict]) -> List[str]:
    # Choose numeric fields present in train set and not in EXCLUDE_FIELDS
    cand = set()
    for r in records:
        for k, v in r.items():
            if k in EXCLUDE_FIELDS:
                continue
            if is_number(v):
                cand.add(k)
    # stable order
    return sorted(cand)


def compute_scaler(train_records: List[dict], feat_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    # compute mean and std over train only
    for col in feat_cols:
        vals = [float(r[col]) for r in train_records if col in r and is_number(r[col])]
        if len(vals) == 0:
            stats[col] = {"mean": 0.0, "std": 1.0}
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = var ** 0.5
        if std == 0.0:
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
    return stats


def apply_scaler(records: List[dict], feat_cols: List[str], scaler: Dict[str, Dict[str, float]]) -> None:
    for r in records:
        for col in feat_cols:
            if col in r and is_number(r[col]):
                s = scaler[col]
                r[col] = (float(r[col]) - s["mean"]) / s["std"]


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def label_stats(records: List[dict]) -> Dict[str, int]:
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in records:
        rel = r.get("rel", None)
        if isinstance(rel, int) and rel in hist:
            hist[rel] += 1
    return {str(k): v for k, v in hist.items()}


def main():
    records = load_records(INPUT_FILE)

    # Basic sanity: keep only records with labels
    records = [r for r in records if "rel" in r and "rmsd" in r]

    # Split by protein (pdbid)
    train_recs, valid_recs, test_recs = split_by_protein(records)

    # Feature selection on train
    feat_cols = select_feature_columns(train_recs)

    # Compute scaler on train, apply to all splits
    scaler = compute_scaler(train_recs, feat_cols)
    apply_scaler(train_recs, feat_cols, scaler)
    apply_scaler(valid_recs, feat_cols, scaler)
    apply_scaler(test_recs,  feat_cols, scaler)

    # Write outputs
    write_jsonl(TRAIN_OUT, train_recs)
    write_jsonl(VALID_OUT, valid_recs)
    write_jsonl(TEST_OUT,  test_recs)

    # Save feature list and scaler for later use
    with FEATURE_LIST_OUT.open("w", encoding="utf-8") as f:
        for c in feat_cols:
            f.write(c + "\n")
    with SCALER_OUT.open("w", encoding="utf-8") as f:
        json.dump({"features": feat_cols, "scaler": scaler}, f, indent=2)

    # Stats
    def count(rec_list): return len(rec_list)
    print("== Split summary ==")
    print(f"train: {count(train_recs)} rows, rel hist {label_stats(train_recs)}")
    print(f"valid: {count(valid_recs)} rows, rel hist {label_stats(valid_recs)}")
    print(f"test : {count(test_recs)} rows, rel hist {label_stats(test_recs)}")
    print(f"features: {len(feat_cols)} -> saved to {FEATURE_LIST_OUT}")
    print(f"scaler  : saved to {SCALER_OUT}")
    print(f"outputs : {TRAIN_OUT.name}, {VALID_OUT.name}, {TEST_OUT.name}")

if __name__ == "__main__":
    main()
