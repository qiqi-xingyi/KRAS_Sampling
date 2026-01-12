# --*-- conding:utf-8 --*--
# @time:1/11/26 22:13
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:inspect_schema.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Schema scanner for KRAS_sampling_results/analysis_closed_loop

Fixes:
- Excludes output directory (e.g., _schema_report) from scanning to avoid self-reference.
- Deduplicates columns_to_files mapping lists.
- Robust CSV parsing: delimiter sniffing, encoding fallback, bad line tolerant.
- Adds automatic "no omissions" validation and fails fast if any indexable file is not parsed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


# ----------------------------
# Utilities
# ----------------------------
def iso_mtime(p: Path) -> str:
    ts = p.stat().st_mtime
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def read_text_head(path: Path, n_bytes: int = 65536) -> bytes:
    with path.open("rb") as f:
        return f.read(n_bytes)


def sniff_delimiter(sample: bytes) -> Optional[str]:
    # Try csv.Sniffer first, then fallback to heuristic counts.
    try:
        text = sample.decode("utf-8", errors="ignore")
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(text, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        pass

    # Heuristic: pick delimiter with most occurrences in the first non-empty line
    try:
        text = sample.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if line.strip():
                counts = {d: line.count(d) for d in [",", "\t", ";", "|"]}
                best = max(counts, key=counts.get)
                return best if counts[best] > 0 else None
    except Exception:
        return None
    return None


def safe_json_load(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj, None
        # If it's a list, we still keep top-level type info as keys_or_cols
        return {"__type__": type(obj).__name__}, None
    except Exception as e:
        try:
            with path.open("r", encoding="latin-1") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj, None
            return {"__type__": type(obj).__name__}, None
        except Exception as e2:
            return None, f"error:{type(e2).__name__}"


def safe_read_csv_columns(
    path: Path,
    max_rows_for_dtype: int = 200,
) -> Tuple[Optional[List[str]], Optional[Dict[str, str]], Optional[str], Optional[str]]:
    """
    Returns: (columns, dtype_preview, delimiter_used, error_status)
    We intentionally only read a small number of rows to infer dtypes, not the whole file.
    """
    sample = read_text_head(path, n_bytes=65536)
    delim = sniff_delimiter(sample)

    # Try a few robust configurations
    attempts = []

    # 1) pandas auto sep inference (python engine) is often robust
    attempts.append({"sep": None, "engine": "python", "encoding": "utf-8"})
    attempts.append({"sep": None, "engine": "python", "encoding": "latin-1"})

    # 2) If we have a sniffed delimiter, try that explicitly (faster / more stable)
    if delim is not None:
        attempts.insert(0, {"sep": delim, "engine": "python", "encoding": "utf-8"})
        attempts.insert(1, {"sep": delim, "engine": "python", "encoding": "latin-1"})

    last_err = None
    for kw in attempts:
        try:
            df = pd.read_csv(
                path,
                nrows=max_rows_for_dtype,
                sep=kw["sep"],
                engine=kw["engine"],
                encoding=kw["encoding"],
                on_bad_lines="skip",
                low_memory=True,
            )
            cols = [str(c) for c in df.columns.tolist()]
            # Deduplicate raw columns for reporting purposes
            # (CSV may contain duplicate headers; keep the first occurrence)
            seen = set()
            cols_unique = []
            for c in cols:
                if c not in seen:
                    cols_unique.append(c)
                    seen.add(c)

            dtypes = {c: str(df[c].dtype) for c in cols_unique if c in df.columns}
            used = kw["sep"] if kw["sep"] is not None else "auto"
            return cols_unique, dtypes, used, None
        except Exception as e:
            last_err = e

    return None, None, None, f"error:{type(last_err).__name__}"


def estimate_row_count_fast(path: Path) -> Optional[int]:
    """
    Row counting can be expensive for multi-GB files.
    We do a size-based policy:
      - <= 200MB: count lines in binary quickly
      - > 200MB: return None (unknown)
    """
    try:
        size = path.stat().st_size
        if size > 200 * 1024 * 1024:
            return None
        n = 0
        with path.open("rb") as f:
            for _ in f:
                n += 1
        # If it has a header, row_count ~= n-1 (best effort)
        return max(n - 1, 0)
    except Exception:
        return None


# ----------------------------
# Data model
# ----------------------------
@dataclass
class FileIndexRow:
    rel_path: str
    abs_path: str
    ext: str
    size_bytes: int
    mtime: str
    n_cols: int
    cols_or_keys: str
    dtype_preview: str
    row_count: Optional[int]
    parse_status: str
    extra: str = ""  # delimiter or other notes


# ----------------------------
# Main logic
# ----------------------------
def collect_files(
    analysis_dir: Path,
    out_dir: Path,
    exclude_dirs: List[str],
    include_exts: Tuple[str, ...] = (".csv", ".json"),
) -> List[Path]:
    """
    Recursively collect files to index, excluding:
      - out_dir
      - any directory name in exclude_dirs
    """
    analysis_dir = analysis_dir.resolve()
    out_dir = out_dir.resolve()

    excluded_names = set([x.strip().rstrip("/").rstrip("\\") for x in exclude_dirs if x.strip()])

    files: List[Path] = []
    for p in analysis_dir.rglob("*"):
        if p.is_dir():
            continue

        # Exclude by extension
        if p.suffix.lower() not in include_exts:
            continue

        # Exclude output directory itself
        if is_within(p, out_dir):
            continue

        # Exclude any directory by name
        # (match any parent folder name)
        skip = False
        for parent in p.parents:
            if parent == analysis_dir.parent:
                break
            if parent.name in excluded_names:
                skip = True
                break
        if skip:
            continue

        files.append(p)

    files.sort(key=lambda x: str(x))
    return files


def build_schema_report(
    analysis_dir: Path,
    out_dir: Path,
    exclude_dirs: List[str],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(
        analysis_dir=analysis_dir,
        out_dir=out_dir,
        exclude_dirs=exclude_dirs,
        include_exts=(".csv", ".json"),
    )

    index_rows: List[FileIndexRow] = []

    # mappings
    columns_to_files: Dict[str, List[str]] = {}
    normalized_to_raw: Dict[str, List[str]] = {}

    parsed_ok = 0
    parsed_fail = 0

    for fp in files:
        rel = fp.resolve().relative_to(analysis_dir.resolve()).as_posix()
        ext = fp.suffix.lower()
        size = fp.stat().st_size
        mtime = iso_mtime(fp)

        if ext == ".json":
            obj, err = safe_json_load(fp)
            if err is None and obj is not None:
                keys = list(obj.keys())
                keys_str = "|".join(keys[:200])
                n_cols = len(keys)
                dtype_preview = ""  # not needed for json
                row = FileIndexRow(
                    rel_path=rel,
                    abs_path=str(fp),
                    ext=ext,
                    size_bytes=size,
                    mtime=mtime,
                    n_cols=n_cols,
                    cols_or_keys=keys_str,
                    dtype_preview=dtype_preview,
                    row_count=None,
                    parse_status="ok",
                    extra="",
                )
                index_rows.append(row)
                parsed_ok += 1

                # Update schema mappings
                for k in keys:
                    nk = normalize_key(str(k))
                    columns_to_files.setdefault(nk, []).append(rel)
                    normalized_to_raw.setdefault(nk, []).append(str(k))
            else:
                row = FileIndexRow(
                    rel_path=rel,
                    abs_path=str(fp),
                    ext=ext,
                    size_bytes=size,
                    mtime=mtime,
                    n_cols=0,
                    cols_or_keys="",
                    dtype_preview="",
                    row_count=None,
                    parse_status=err or "error",
                    extra="",
                )
                index_rows.append(row)
                parsed_fail += 1

        elif ext == ".csv":
            cols, dtypes, delim_used, err = safe_read_csv_columns(fp)
            if err is None and cols is not None:
                cols_str = "|".join(cols[:400])
                dtype_str = ""
                if dtypes:
                    # keep compact
                    pairs = [f"{k}:{v}" for k, v in list(dtypes.items())[:60]]
                    dtype_str = "|".join(pairs)

                row_count = estimate_row_count_fast(fp)
                row = FileIndexRow(
                    rel_path=rel,
                    abs_path=str(fp),
                    ext=ext,
                    size_bytes=size,
                    mtime=mtime,
                    n_cols=len(cols),
                    cols_or_keys=cols_str,
                    dtype_preview=dtype_str,
                    row_count=row_count,
                    parse_status="ok",
                    extra=f"sep={delim_used}",
                )
                index_rows.append(row)
                parsed_ok += 1

                for c in cols:
                    nc = normalize_key(str(c))
                    columns_to_files.setdefault(nc, []).append(rel)
                    normalized_to_raw.setdefault(nc, []).append(str(c))
            else:
                row = FileIndexRow(
                    rel_path=rel,
                    abs_path=str(fp),
                    ext=ext,
                    size_bytes=size,
                    mtime=mtime,
                    n_cols=0,
                    cols_or_keys="",
                    dtype_preview="",
                    row_count=None,
                    parse_status=err or "error",
                    extra="",
                )
                index_rows.append(row)
                parsed_fail += 1

    # Deduplicate mapping values (bugfix: avoid duplicate file paths in lists)
    for k, lst in columns_to_files.items():
        columns_to_files[k] = sorted(list(dict.fromkeys(lst)))

    for k, lst in normalized_to_raw.items():
        # Dedup raw names while preserving order
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        normalized_to_raw[k] = out

    # Write outputs
    files_index_csv = out_dir / "files_index.csv"
    columns_to_files_json = out_dir / "columns_to_files.json"
    normalized_to_raw_json = out_dir / "normalized_to_raw_columns.json"
    summary_json = out_dir / "summary.json"

    # CSV
    df = pd.DataFrame([r.__dict__ for r in index_rows])
    df.to_csv(files_index_csv, index=False)

    # JSON
    with columns_to_files_json.open("w", encoding="utf-8") as f:
        json.dump(columns_to_files, f, indent=2, ensure_ascii=False)

    with normalized_to_raw_json.open("w", encoding="utf-8") as f:
        json.dump(normalized_to_raw, f, indent=2, ensure_ascii=False)

    # Validation: no omissions
    # Rule:
    #   - Every collected file must appear in files_index.csv (should always be true)
    #   - All collected files must have parse_status == ok
    indexable_rel = [p.resolve().relative_to(analysis_dir.resolve()).as_posix() for p in files]
    indexed_rel = set(df["rel_path"].tolist())

    missing_in_index = sorted([x for x in indexable_rel if x not in indexed_rel])

    bad = df[df["parse_status"] != "ok"][["rel_path", "parse_status"]].copy()
    bad_list = [{"rel_path": r, "parse_status": s} for r, s in bad.values.tolist()]

    no_omissions = (len(missing_in_index) == 0) and (len(bad_list) == 0)

    summary = {
        "analysis_dir": str(analysis_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "excluded_dirs": exclude_dirs,
        "n_files_scanned": len(files),
        "n_files_indexed": int(df.shape[0]),
        "n_parsed_ok": parsed_ok,
        "n_parsed_fail": parsed_fail,
        "missing_in_index": missing_in_index,
        "bad_files": bad_list,
        "no_omissions": no_omissions,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def default_project_root() -> Path:
    # If this script is in tools/, project root is its parent.
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Path to analysis_closed_loop. Default: <PROJECT_ROOT>/KRAS_sampling_results/analysis_closed_loop",
    )
    parser.add_argument(
        "--out_dir_name",
        type=str,
        default="_schema_report",
        help="Output directory name under analysis_dir.",
    )
    parser.add_argument(
        "--exclude_dirs",
        type=str,
        default="_schema_report,figs,figs_redraw,plots_addon,exported_pdb,reps_pdbs",
        help="Comma-separated directory names to exclude anywhere in the tree.",
    )
    args = parser.parse_args()

    project_root = default_project_root()

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else (project_root / "KRAS_sampling_results" / "analysis_closed_loop")
    out_dir = analysis_dir / args.out_dir_name
    exclude_dirs = [x.strip() for x in args.exclude_dirs.split(",") if x.strip()]

    if not analysis_dir.exists():
        print(f"[ERROR] ANALYSIS_DIR not found: {analysis_dir}")
        return 1

    # Always exclude output dir name (even if user forgets)
    if args.out_dir_name not in exclude_dirs:
        exclude_dirs = [args.out_dir_name] + exclude_dirs

    print(f"[INFO] PROJECT_ROOT: {project_root}")
    print(f"[INFO] ANALYSIS_DIR:  {analysis_dir.resolve()}")
    print(f"[INFO] OUT_DIR:      {out_dir.resolve()}")
    print(f"[INFO] EXCLUDE_DIRS: {exclude_dirs}")

    summary = build_schema_report(
        analysis_dir=analysis_dir,
        out_dir=out_dir,
        exclude_dirs=exclude_dirs,
    )

    # Print validation result
    if summary["no_omissions"]:
        print("[OK] Schema report complete. No omissions. All indexable files parsed successfully.")
        print(f"[OK] files_index.csv: {Path(summary['out_dir']) / 'files_index.csv'}")
        print(f"[OK] columns_to_files.json: {Path(summary['out_dir']) / 'columns_to_files.json'}")
        print(f"[OK] normalized_to_raw_columns.json: {Path(summary['out_dir']) / 'normalized_to_raw_columns.json'}")
        print(f"[OK] summary.json: {Path(summary['out_dir']) / 'summary.json'}")
        return 0

    print("[ERROR] Schema report validation FAILED.")
    if summary["missing_in_index"]:
        print(f"[ERROR] Missing in index ({len(summary['missing_in_index'])}):")
        for x in summary["missing_in_index"][:200]:
            print(f"  - {x}")
        if len(summary["missing_in_index"]) > 200:
            print("  ... (truncated)")

    if summary["bad_files"]:
        print(f"[ERROR] Bad files ({len(summary['bad_files'])}):")
        for item in summary["bad_files"][:200]:
            print(f"  - {item['rel_path']}  ({item['parse_status']})")
        if len(summary["bad_files"]) > 200:
            print("  ... (truncated)")

    # Non-zero exit for CI / automation
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


