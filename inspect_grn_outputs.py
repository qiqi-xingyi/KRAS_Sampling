# --*-- conding:utf-8 --*--
# @time:12/18/25 22:34
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:inspect_grn_outputs.py


# Inspect GRN inference parquet output and export readable summaries.
#
# Usage:
#   python inspect_grn_outputs.py \
#     --parquet grn_outputs/run_20251218_223025/grn_ranked_all.parquet \
#     --out_dir grn_outputs/run_20251218_223025/inspect \
#     --topk 50
#
# Optional:
#   python inspect_grn_outputs.py ... --pdb_id 4LPK_WT_1 --export_csv

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd


REQUIRED_COLS = ["pdb_id", "group_id", "bitstring", "sequence", "score", "rank_in_group"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path.resolve()}")
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet: {path}. "
            f"Install pyarrow or fastparquet. Original error: {e}"
        )
    return df


def _check_cols(df: pd.DataFrame) -> None:
    miss = [c for c in REQUIRED_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}. Existing columns: {list(df.columns)}")


def _print_global_summary(df: pd.DataFrame) -> None:
    _check_cols(df)

    n_rows = len(df)
    n_pdb = df["pdb_id"].nunique(dropna=True)
    n_groups = df[["pdb_id", "group_id"]].drop_duplicates().shape[0]

    print("=== GRN inference parquet summary ===")
    print(f"Rows: {n_rows}")
    print(f"Unique pdb_id: {n_pdb}")
    print(f"Unique (pdb_id, group_id): {n_groups}")
    print("Columns:", sorted(df.columns.tolist()))
    print()

    # score distribution
    s = df["score"].astype(float)
    print("=== Score distribution (global) ===")
    print(s.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())
    print()

    # per fragment size
    sizes = df.groupby(["pdb_id", "group_id"]).size().reset_index(name="n")
    print("=== Fragment sizes (top 10 by n) ===")
    print(sizes.sort_values("n", ascending=False).head(10).to_string(index=False))
    print()


def _export_topk_per_fragment(df: pd.DataFrame, out_jsonl: Path, k: int) -> None:
    _check_cols(df)
    if k <= 0:
        return

    top = (
        df.sort_values(["pdb_id", "group_id", "score"], ascending=[True, True, False])
          .groupby(["pdb_id", "group_id"], sort=True)
          .head(k)
          .reset_index(drop=True)
    )

    _ensure_dir(out_jsonl.parent)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, r in top.iterrows():
            obj = {c: r[c] for c in REQUIRED_COLS}
            obj["group_id"] = int(obj["group_id"])
            obj["rank_in_group"] = int(obj["rank_in_group"])
            obj["score"] = float(obj["score"])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] Top-{k} per fragment saved: {out_jsonl} (rows={len(top)})")


def _print_fragment_preview(df: pd.DataFrame, pdb_id: str, k: int = 20) -> pd.DataFrame:
    sub = df[df["pdb_id"].astype(str) == str(pdb_id)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for pdb_id={pdb_id}. Available examples: {df['pdb_id'].unique()[:10]}")
    sub = sub.sort_values(["group_id", "score"], ascending=[True, False])
    print(f"=== Preview pdb_id={pdb_id} (top {k} by score) ===")
    cols_show = ["pdb_id", "group_id", "rank_in_group", "score", "bitstring", "sequence"]
    print(sub[cols_show].head(k).to_string(index=False))
    print()

    # per group score stats
    print(f"=== Score distribution for pdb_id={pdb_id} (per group_id) ===")
    stats = (
        sub.groupby("group_id")["score"]
           .apply(lambda x: pd.Series({
               "n": int(x.shape[0]),
               "min": float(x.min()),
               "p50": float(x.median()),
               "p95": float(x.quantile(0.95)),
               "max": float(x.max()),
               "mean": float(x.mean()),
               "std": float(x.std(ddof=0)),
           }))
           .reset_index()
    )
    print(stats.to_string(index=False))
    print()
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, required=True,
                    help="Path to grn_ranked_all.parquet")
    ap.add_argument("--out_dir", type=str, default="inspect_out",
                    help="Output directory for readable exports.")
    ap.add_argument("--topk", type=int, default=50,
                    help="Export top-k per fragment to JSONL.")
    ap.add_argument("--pdb_id", type=str, default="",
                    help="If set, preview this specific pdb_id.")
    ap.add_argument("--export_csv", action="store_true",
                    help="If set with --pdb_id, export that fragment as CSV.")
    ap.add_argument("--preview_k", type=int, default=20,
                    help="How many top rows to print for --pdb_id preview.")
    args = ap.parse_args()

    pq = Path(args.parquet)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    df = _load_parquet(pq)
    _print_global_summary(df)

    # Export topk per fragment (lightweight)
    if int(args.topk) > 0:
        out_topk = out_dir / f"top{int(args.topk)}_per_fragment.jsonl"
        _export_topk_per_fragment(df, out_topk, int(args.topk))

    # Optional: inspect one fragment
    if args.pdb_id.strip():
        sub = _print_fragment_preview(df, args.pdb_id.strip(), k=int(args.preview_k))
        if args.export_csv:
            out_csv = out_dir / f"{args.pdb_id.strip()}_all_ranked.csv"
            sub.to_csv(out_csv, index=False)
            print(f"[OK] Exported fragment CSV: {out_csv}")

    # Save a small meta summary
    meta = {
        "parquet": str(pq),
        "rows": int(len(df)),
        "unique_pdb_id": int(df["pdb_id"].nunique()),
        "unique_fragments": int(df[["pdb_id", "group_id"]].drop_duplicates().shape[0]),
        "export_topk": int(args.topk),
    }
    with (out_dir / "inspect_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Inspect outputs under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
