# --*-- conding:utf-8 --*--
# @time:1/11/26 22:13
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:inspect_schema.py


import argparse
import csv as _csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import pandas as pd


SKIP_DIR_NAMES = {
    "figs", "fig", "figures", "plots", "plot", "figs_redraw", "plots_addon"
}
SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".pdb", ".pdbqt", ".pml"
}


def project_root_from_tools() -> Path:
    # If this script is in tools/, project root is tools/..
    return Path(__file__).resolve().parents[1]


def sniff_delimiter(path: Path, default: str = ",") -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(8192)
        if not sample.strip():
            return default
        dialect = _csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return default


def safe_stat(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def normalize_col(name: str) -> str:
    # Normalize to help detect near-duplicates across files
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def read_csv_schema(path: Path, infer_rows: int = 200) -> Tuple[List[str], Dict[str, str]]:
    delim = sniff_delimiter(path)
    # Header only for columns
    df0 = pd.read_csv(path, sep=delim, nrows=0, engine="python")
    cols = list(df0.columns)

    # Infer dtypes from a small sample
    dtypes: Dict[str, str] = {}
    if infer_rows > 0:
        dfi = pd.read_csv(path, sep=delim, nrows=infer_rows, engine="python", low_memory=False)
        for c in cols:
            dtypes[c] = str(dfi[c].dtype) if c in dfi.columns else "unknown"
    else:
        for c in cols:
            dtypes[c] = "unknown"

    return cols, dtypes


def read_json_schema(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, list) and obj:
        # If list of dicts, union keys
        keys = set()
        for item in obj[:50]:
            if isinstance(item, dict):
                keys.update(item.keys())
        return sorted(keys)
    return []


def read_jsonl_schema(path: Path, max_lines: int = 2000) -> List[str]:
    keys = set()
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    keys.update(obj.keys())
            except Exception:
                pass
            n += 1
            if n >= max_lines:
                break
    return sorted(keys)


def count_rows_fast(path: Path) -> int:
    # Counts data rows for CSV/TSV-like files (excluding header).
    # This is optional because it can be slow for very large files.
    with path.open("rb") as f:
        n = 0
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    # Approx: number of lines - 1 header (>=0)
    return max(0, n - 1)


def iter_data_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in SKIP_DIR_NAMES:
                # skip known figure directories
                continue
            continue
        if p.suffix.lower() in SKIP_EXTS:
            continue
        # Skip obvious caches
        if p.name.startswith("."):
            continue
        files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Inspect schema (columns/keys) for analysis_closed_loop outputs.")
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="KRAS_sampling_results/analysis_closed_loop",
        help="Path to analysis_closed_loop (relative to project root by default).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="KRAS_sampling_results/analysis_closed_loop/_schema_report",
        help="Output directory (relative to project root by default).",
    )
    parser.add_argument("--infer_rows", type=int, default=200, help="Rows to sample for dtype inference (CSV).")
    parser.add_argument("--count_rows", action="store_true", help="Count CSV data rows (can be slow).")

    args = parser.parse_args()

    root = project_root_from_tools()
    analysis_dir = (root / args.analysis_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not analysis_dir.exists():
        raise FileNotFoundError(f"analysis_dir not found: {analysis_dir}")

    files = iter_data_files(analysis_dir)

    file_records: List[Dict[str, Any]] = []
    columns_to_files: Dict[str, List[str]] = {}
    norm_to_raw: Dict[str, List[str]] = {}

    for fp in files:
        rel = str(fp.relative_to(analysis_dir))
        ext = fp.suffix.lower()

        record: Dict[str, Any] = {
            "rel_path": rel,
            "abs_path": str(fp),
            "ext": ext,
            **safe_stat(fp),
            "n_cols": 0,
            "cols_or_keys": "",
            "dtype_preview": "",
            "row_count": "",
            "parse_status": "ok",
        }

        try:
            if ext in {".csv", ".tsv"}:
                cols, dtypes = read_csv_schema(fp, infer_rows=args.infer_rows)
                record["n_cols"] = len(cols)
                record["cols_or_keys"] = "|".join([str(c) for c in cols])
                record["dtype_preview"] = "|".join([f"{c}:{dtypes.get(c,'')}" for c in cols[:60]])
                if args.count_rows:
                    record["row_count"] = str(count_rows_fast(fp))

                for c in cols:
                    columns_to_files.setdefault(c, []).append(rel)
                    nc = normalize_col(c)
                    norm_to_raw.setdefault(nc, [])
                    if c not in norm_to_raw[nc]:
                        norm_to_raw[nc].append(c)

            elif ext == ".json":
                keys = read_json_schema(fp)
                record["n_cols"] = len(keys)
                record["cols_or_keys"] = "|".join(keys)
                for k in keys:
                    columns_to_files.setdefault(k, []).append(rel)
                    nk = normalize_col(k)
                    norm_to_raw.setdefault(nk, [])
                    if k not in norm_to_raw[nk]:
                        norm_to_raw[nk].append(k)

            elif ext in {".jsonl", ".ndjson"}:
                keys = read_jsonl_schema(fp)
                record["n_cols"] = len(keys)
                record["cols_or_keys"] = "|".join(keys)
                for k in keys:
                    columns_to_files.setdefault(k, []).append(rel)
                    nk = normalize_col(k)
                    norm_to_raw.setdefault(nk, [])
                    if k not in norm_to_raw[nk]:
                        norm_to_raw[nk].append(k)
            else:
                # Unknown text-like file, just record it
                record["parse_status"] = "skipped_unknown_ext"

        except Exception as e:
            record["parse_status"] = f"error:{type(e).__name__}"
            record["cols_or_keys"] = ""
            record["dtype_preview"] = ""

        file_records.append(record)

    # Save per-file index
    df = pd.DataFrame(file_records)
    df.to_csv(out_dir / "files_index.csv", index=False)

    # Save column->files mapping
    with (out_dir / "columns_to_files.json").open("w", encoding="utf-8") as f:
        json.dump(columns_to_files, f, indent=2, ensure_ascii=False)

    # Save normalized name map (helps align basin_id vs basin, etc.)
    with (out_dir / "normalized_to_raw_columns.json").open("w", encoding="utf-8") as f:
        json.dump(norm_to_raw, f, indent=2, ensure_ascii=False)

    # Convenience summary
    summary = {
        "analysis_dir": str(analysis_dir),
        "n_files_scanned": len(files),
        "n_files_indexed": len(file_records),
        "out_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[OK] Schema report written to:", out_dir)
    print("[OK] files_index.csv / columns_to_files.json / normalized_to_raw_columns.json")


if __name__ == "__main__":
    main()
