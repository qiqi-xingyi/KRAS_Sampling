# --*-- conding:utf-8 --*--
# @time:12/18/25 22:32
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:check_ckpt.py

# Purpose:
#   1) Inspect scaler object stored in GRN checkpoint
#   2) Verify whether inference ranking differs between:
#        - identity transform
#        - checkpoint scaler (if usable) / rebuilt scaler
#   3) Print quick metrics: score correlation, Spearman, top-k overlap
#
# Usage:
#   python check_grn_scaler.py \
#       --ckpt checkpoints_full/grn_best.pt \
#       --input prepared_dataset/kras_all_grn_input.jsonl \
#       --sample_n 20000 \
#       --topk 50
#
# Notes:
#   - This script does not modify any files; it only prints diagnostics.
#   - If sklearn is missing, it will still attempt a lightweight scaler rebuild.

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Import your inferencer (checkpoint-driven). Assumes Ghost_RMSD is importable.
from Ghost_RMSD import GRNInferencer


def _load_jsonl(path: Path, max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and len(rows) >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_rows(path: Path, sample_n: int, seed: int) -> List[Dict[str, Any]]:
    """
    Reservoir sampling from JSONL (O(N) streaming, O(sample_n) memory).
    """
    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if len(sample) < sample_n:
                sample.append(obj)
            else:
                j = rng.randint(0, i)
                if j < sample_n:
                    sample[j] = obj
    return sample


def _identity_scaler(expected_dim: int):
    class _Id:
        def __init__(self, dim: int):
            self.n_features_in_ = dim

        def transform(self, X):
            return X

    return _Id(expected_dim)


def _describe_scaler(s: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "type": str(type(s)),
        "has_transform": bool(hasattr(s, "transform")),
    }
    if hasattr(s, "__dict__"):
        d = getattr(s, "__dict__", {})
        # keep small
        info["attrs"] = sorted([k for k in d.keys() if not k.startswith("_")])[:50]
    if isinstance(s, dict):
        info["dict_keys"] = sorted(list(s.keys()))[:80]
        # common nested patterns
        if "state" in s and isinstance(s["state"], dict):
            info["state_keys"] = sorted(list(s["state"].keys()))[:80]
    return info


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation via rank transform (ties: average rank).
    """
    if x.size == 0 or y.size == 0:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum()))
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _topk_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame, k: int) -> float:
    """
    Compare top-k per (pdb_id, group_id) using bitstring sets.
    Returns average overlap ratio in [0,1].
    """
    need = ["pdb_id", "group_id", "bitstring", "score"]
    for c in need:
        if c not in df_a.columns or c not in df_b.columns:
            raise ValueError(f"Missing required columns for topk overlap: {need}")

    key_cols = ["pdb_id", "group_id"]
    a_top = (
        df_a.sort_values(key_cols + ["score"], ascending=[True, True, False])
            .groupby(key_cols).head(k)
    )
    b_top = (
        df_b.sort_values(key_cols + ["score"], ascending=[True, True, False])
            .groupby(key_cols).head(k)
    )

    overlaps = []
    for key, sub_a in a_top.groupby(key_cols, sort=False):
        sub_b = b_top[(b_top["pdb_id"] == key[0]) & (b_top["group_id"] == key[1])]
        sa = set(sub_a["bitstring"].astype(str).tolist())
        sb = set(sub_b["bitstring"].astype(str).tolist())
        if not sa and not sb:
            continue
        inter = len(sa & sb)
        denom = max(1, min(len(sa), len(sb)))
        overlaps.append(inter / denom)
    return float(np.mean(overlaps)) if overlaps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="Ghost_RMSD/checkpoints_full/grn_best.pt")
    ap.add_argument("--input", type=str, default="prepared_dataset/kras_all_grn_input.jsonl")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--sample_n", type=int, default=20000,
                    help="How many rows to sample from input for comparison.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=50,
                    help="Top-k overlap per (pdb_id,group_id) to report.")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    inp_path = Path(args.input)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")
    if not inp_path.exists():
        raise FileNotFoundError(f"Input jsonl not found: {inp_path.resolve()}")

    # ---- 1) Inspect scaler in checkpoint ----
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    scaler_obj = ckpt.get("scaler", None)
    print("=== Checkpoint scaler inspection ===")
    print("ckpt keys:", sorted(list(ckpt.keys())))
    print("scaler info:", json.dumps(_describe_scaler(scaler_obj), indent=2, ensure_ascii=False))

    # Also print args if present
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, dict):
        print("ckpt args (subset):", {k: ckpt_args.get(k) for k in ["batch_size", "dropout", "score_mode", "use_rank_head"]})

    # ---- 2) Sample rows ----
    print("\n=== Sampling input rows ===")
    rows = _sample_rows(inp_path, sample_n=int(args.sample_n), seed=int(args.seed))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Sampled dataframe is empty. Check input file.")

    # Ensure group_id exists for fair comparison
    if "group_id" not in df.columns:
        df["group_id"] = 0

    print(f"[OK] sampled {len(df)} rows")
    print("columns:", sorted(df.columns.tolist()))

    # ---- 3) Run inference with checkpoint scaler (default inferencer) ----
    print("\n=== Inference with checkpoint scaler (default) ===")
    inf_ckpt = GRNInferencer(ckpt=str(ckpt_path), device=str(args.device), verbose=True)
    df_ckpt = inf_ckpt.predict_df(df, topk=None)
    print(f"[OK] scored {len(df_ckpt)} rows (ckpt scaler path)")

    # ---- 4) Run inference with identity scaler (force) ----
    print("\n=== Inference with identity scaler (forced) ===")
    inf_id = GRNInferencer(ckpt=str(ckpt_path), device=str(args.device), verbose=False)
    # force identity scaler with correct expected dimension
    inf_id.scaler = _identity_scaler(expected_dim=len(inf_id.base_feature_names))
    df_id = inf_id.predict_df(df, topk=None)
    print(f"[OK] scored {len(df_id)} rows (identity scaler path)")

    # ---- 5) Compare scores ----
    # Align by row identity (bitstring + pdb_id + group_id); fallback to index alignment if duplicates
    join_cols = ["pdb_id", "group_id", "bitstring"]
    df_ck = df_ckpt[join_cols + ["score"]].copy()
    df_ii = df_id[join_cols + ["score"]].copy()
    merged = df_ck.merge(df_ii, on=join_cols, how="inner", suffixes=("_ckpt", "_id"))

    if merged.empty:
        print("[WARN] Could not align rows by (pdb_id,group_id,bitstring). Falling back to positional alignment.")
        n = min(len(df_ckpt), len(df_id))
        s_ck = df_ckpt["score"].to_numpy()[:n]
        s_id = df_id["score"].to_numpy()[:n]
    else:
        s_ck = merged["score_ckpt"].to_numpy()
        s_id = merged["score_id"].to_numpy()

    corr = float(np.corrcoef(s_ck, s_id)[0, 1]) if s_ck.size >= 2 else float("nan")
    sp = _spearman(s_ck, s_id)

    print("\n=== Comparison ===")
    print(f"Aligned rows: {int(s_ck.size)}")
    print(f"Pearson corr(score_ckpt, score_id): {corr:.6f}")
    print(f"Spearman corr(score_ckpt, score_id): {sp:.6f}")
    print(f"mean(score_ckpt)={float(np.mean(s_ck)):.6f} | std={float(np.std(s_ck)):.6f}")
    print(f"mean(score_id)  ={float(np.mean(s_id)):.6f} | std={float(np.std(s_id)):.6f}")

    k = int(args.topk)
    if k > 0:
        ov = _topk_overlap(df_ckpt, df_id, k=k)
        print(f"Avg top-{k} overlap per (pdb_id,group_id): {ov:.4f}")

    # Also report how many unique fragments in sample
    n_frags = df[join_cols[:2]].drop_duplicates().shape[0]
    print(f"Fragments (unique pdb_id,group_id) in sample: {n_frags}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
