# --*-- coding:utf-8 --*--
# @time:12/18/25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prediction.py

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

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


def _hash_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:10]


def _iter_group_keys(df: pd.DataFrame) -> Iterable[tuple[str, int, str]]:
    for (pdb_id, group_id, sequence), _ in df.groupby(["pdb_id", "group_id", "sequence"], sort=True):
        yield str(pdb_id), int(group_id), str(sequence)


def _write_group_jsonl(
    df_ranked: pd.DataFrame,
    out_dir: Path,
    pdb_id: str,
    group_id: int,
    sequence: str,
    include_all_columns: bool,
) -> Path:
    sub = df_ranked[
        (df_ranked["pdb_id"] == pdb_id)
        & (df_ranked["group_id"] == group_id)
        & (df_ranked["sequence"] == sequence)
    ].copy()

    sub = sub.sort_values(["rank_in_group", "score"], ascending=[True, False]).reset_index(drop=True)

    seq_tag = _hash_seq(sequence)
    out_path = out_dir / f"{pdb_id}_g{group_id}_seq{seq_tag}.ranked.jsonl"
    _ensure_dir(out_path.parent)

    if include_all_columns:
        cols = list(sub.columns)
    else:
        cols = ["pdb_id", "group_id", "sequence", "bitstring", "score", "rank_in_group"]
        for c in cols:
            if c not in sub.columns:
                raise ValueError(f"Missing required column in ranked df: {c}")

    with out_path.open("w", encoding="utf-8") as f:
        for _, r in sub.iterrows():
            obj = {c: r[c] for c in cols}
            if "group_id" in obj:
                obj["group_id"] = int(obj["group_id"])
            if "rank_in_group" in obj:
                obj["rank_in_group"] = int(obj["rank_in_group"])
            if "score" in obj:
                obj["score"] = float(obj["score"])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default="prepared_dataset/kras_all_grn_input.jsonl",
        help="GRN input JSONL file.",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="Ghost_RMSD/checkpoints_full/grn_best.pt",
        help="GRN checkpoint path.",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="grn_outputs",
        help="Output directory root.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto|cpu|cuda|mps",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Inference batch size.",
    )
    ap.add_argument(
        "--score_mode",
        type=str,
        default=None,
        help="Override score_mode. If None, use checkpoint args.",
    )
    ap.add_argument(
        "--write_all_parquet",
        action="store_true",
        help="Also write a combined parquet for all rows.",
    )
    ap.add_argument(
        "--per_sequence_jsonl",
        action="store_true",
        help="Write one ranked JSONL per (pdb_id, group_id, sequence).",
    )
    ap.add_argument(
        "--include_all_columns",
        action="store_true",
        help="If set, per-sequence JSONL will include all columns, not just core fields.",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")

    out_root = Path(args.out_root) / f"run_{_now_tag()}"
    _ensure_dir(out_root)

    meta = {
        "input": str(inp),
        "ckpt": str(args.ckpt),
        "device": args.device,
        "batch_size": int(args.batch_size),
        "score_mode_override": args.score_mode,
        "write_all_parquet": bool(args.write_all_parquet),
        "per_sequence_jsonl": bool(args.per_sequence_jsonl),
        "include_all_columns": bool(args.include_all_columns),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    inf = GRNInferencer(
        ckpt=args.ckpt,
        device=args.device,
        batch_size=int(args.batch_size),
        score_mode=args.score_mode,
    )

    df_ranked = inf.predict_jsonl(inp, topk=None)
    print(f"[INFO] Ranked rows: {len(df_ranked)}")

    if args.write_all_parquet:
        all_parquet = out_root / "grn_ranked_all.parquet"
        _safe_write_parquet(df_ranked, all_parquet)
        print(f"[OK] Wrote: {all_parquet}")

    if args.per_sequence_jsonl:
        out_dir = out_root / "per_sequence"
        _ensure_dir(out_dir)

        total_groups = 0
        for pdb_id, group_id, sequence in _iter_group_keys(df_ranked):
            p = _write_group_jsonl(
                df_ranked=df_ranked,
                out_dir=out_dir,
                pdb_id=pdb_id,
                group_id=group_id,
                sequence=sequence,
                include_all_columns=bool(args.include_all_columns),
            )
            total_groups += 1
            if total_groups % 50 == 0:
                print(f"[INFO] Written per-sequence files: {total_groups} (last: {p.name})")

        print(f"[OK] Wrote per-sequence ranked JSONL files: {total_groups} groups -> {out_dir}")

    print(f"[DONE] Outputs under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
