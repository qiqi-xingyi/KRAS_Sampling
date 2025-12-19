# --*-- conding:utf-8 --*--
# @time:12/18/25  (patched)
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prediction.py

# Run full GRN inference on prepared JSONL inputs and save ranked results.
#
# Input JSONL (recommended):
#   prepared_dataset/kras_all_grn_input.jsonl
#
# Output:
#   out_root/
#     grn_ranked_all.parquet
#     grn_ranked_all.csv              (optional)
#     per_fragment/
#       <pdb_id>_ranked.parquet
#     top<k>.jsonl                    (optional)
#     meta.json
#
# Notes:
# - Ranking is performed within (pdb_id, group_id). If group_id is missing, it defaults to 0.
# - For very large N (e.g., 1.2M), parquet is strongly recommended.
# - GRNInferencer is now checkpoint-driven; batch_size/score_mode are taken from ckpt args by default.
#   This script allows overriding them (optional), but you can simply omit those args.

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from Ghost_RMSD import GRNInferencer


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write parquet to {path}. "
            f"Install pyarrow or fastparquet. Original error: {e}"
        )


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _save_topk_jsonl(df_ranked: pd.DataFrame, out_path: Path, k: int) -> None:
    if k <= 0:
        return
    need_cols = ["pdb_id", "group_id", "bitstring", "sequence", "score", "rank_in_group"]
    for c in need_cols:
        if c not in df_ranked.columns:
            raise ValueError(f"Missing required column in ranked df: {c}")

    top = (
        df_ranked.sort_values(["pdb_id", "group_id", "score"], ascending=[True, True, False])
                 .groupby(["pdb_id", "group_id"])
                 .head(k)
                 .reset_index(drop=True)
    )

    _ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in top.iterrows():
            obj = {c: r[c] for c in need_cols}
            obj["group_id"] = int(obj["group_id"])
            obj["rank_in_group"] = int(obj["rank_in_group"])
            obj["score"] = float(obj["score"])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", type=str, default="prepared_dataset/kras_all_grn_input.jsonl",
                    help="GRN input jsonl file.")
    ap.add_argument("--ckpt", type=str, default="checkpoints_full/grn_best.pt",
                    help="GRN checkpoint path.")
    ap.add_argument("--out_root", type=str, default="grn_outputs",
                    help="Output directory root.")
    ap.add_argument("--device", type=str, default="auto",
                    help="Device: auto|cpu|cuda|mps")

    # Optional overrides (if omitted, GRNInferencer uses ckpt args / defaults)
    ap.add_argument("--batch_size", type=int, default=-1,
                    help="Override inference batch size. Use -1 to use ckpt/default.")
    ap.add_argument("--score_mode", type=str, default="",
                    help="Override score_mode. Use '' to use ckpt/default.")

    ap.add_argument("--topk", type=int, default=0,
                    help="If >0, also save top-k per fragment as JSONL.")
    ap.add_argument("--per_fragment", action="store_true",
                    help="Also write one ranked parquet per pdb_id.")
    ap.add_argument("--write_csv", action="store_true",
                    help="Also export a CSV (large; not recommended for 1.2M rows).")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")

    out_root = Path(args.out_root) / f"run_{_now_tag()}"
    _ensure_dir(out_root)

    # Build inferencer kwargs (checkpoint-driven by default)
    infer_kwargs = {
        "ckpt": args.ckpt,
        "device": args.device,
    }
    # Only override if user provided meaningful values
    if int(args.batch_size) and int(args.batch_size) > 0:
        infer_kwargs["batch_size"] = int(args.batch_size)
    if isinstance(args.score_mode, str) and args.score_mode.strip():
        infer_kwargs["score_mode"] = str(args.score_mode).strip()

    # Run inference
    inf = GRNInferencer(**infer_kwargs)
    df_ranked = inf.predict_jsonl(inp, topk=None)  # full ranking
    print(f"[INFO] Ranked rows: {len(df_ranked)}")

    # Save metadata for reproducibility (record effective settings)
    meta = {
        "input": str(inp),
        "ckpt": str(args.ckpt),
        "device": args.device,
        "batch_size_effective": int(getattr(inf, "batch_size", -1)),
        "score_mode_effective": str(getattr(inf, "score_mode", "")),
        "topk": int(args.topk),
        "per_fragment": bool(args.per_fragment),
        "write_csv": bool(args.write_csv),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Write combined result
    all_parquet = out_root / "grn_ranked_all.parquet"
    _safe_write_parquet(df_ranked, all_parquet)
    print(f"[OK] Wrote: {all_parquet}")

    if args.write_csv:
        all_csv = out_root / "grn_ranked_all.csv"
        _safe_write_csv(df_ranked, all_csv)
        print(f"[OK] Wrote: {all_csv}")

    # Optional: per-fragment outputs
    if args.per_fragment:
        per_dir = out_root / "per_fragment"
        _ensure_dir(per_dir)
        for pdb_id, sub in df_ranked.groupby("pdb_id", sort=True):
            out_p = per_dir / f"{pdb_id}_ranked.parquet"
            _safe_write_parquet(sub.reset_index(drop=True), out_p)
        print(f"[OK] Wrote per-fragment parquet to: {per_dir}")

    # Optional: top-k JSONL for downstream steps
    if args.topk and args.topk > 0:
        topk_path = out_root / f"top{int(args.topk)}.jsonl"
        _save_topk_jsonl(df_ranked, topk_path, int(args.topk))
        print(f"[OK] Wrote top-k jsonl: {topk_path}")

    print(f"[DONE] Outputs under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
