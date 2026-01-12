# --*-- conding:utf-8 --*--
# @time:1/11/26 22:13
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:inspect_schema.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd


SKIP_DIR_NAMES = {"figs", "fig", "figures", "plots", "plot", "figs_redraw", "plots_addon"}
SKIP_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf", ".pdb", ".pdbqt", ".pml"}

CANDIDATE_DELIMS = [",", "\t", ";", "|"]


def project_root_from_tools() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_stat(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("__", "_")


def first_data_line(path: Path, max_lines: int = 50) -> Optional[str]:
    """
    Return the first non-empty, non-comment line as header candidate.
    """
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            s = line.strip("\n").strip("\r").strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            return s
    return None


def split_header_best(line: str) -> Tuple[List[str], str]:
    """
    Try multiple delimiters and pick the one that yields the most columns.
    Also handles whitespace-separated header as a fallback.
    """
    best_cols: List[str] = [line.strip()]
    best_delim = "RAW"

    for d in CANDIDATE_DELIMS:
        cols = [c.strip() for c in line.split(d)]
        cols = [c for c in cols if c != ""]
        if len(cols) > len(best_cols):
            best_cols = cols
            best_delim = d

    # whitespace fallback (only if it seems like a table header)
    ws_cols = [c.strip() for c in line.split()]
    ws_cols = [c for c in ws_cols if c != ""]
    if len(ws_cols) > len(best_cols) and len(ws_cols) >= 2:
        best_cols = ws_cols
        best_delim = "WHITESPACE"

    return best_cols, best_delim


def read_csv_schema_robust(path: Path, infer_rows: int = 200) -> Tuple[List[str], Dict[str, str], str, str]:
    """
    Return (columns, dtype_preview_map, delimiter_used, parse_mode)
    parse_mode: header_only | pandas_sample | header_fallback
    """
    header = first_data_line(path)
    if header is None:
        return [], {}, "EMPTY", "header_fallback"

    cols, delim = split_header_best(header)

    # If we only got 1 column, still try pandas with common separators to see if file has real header
    dtypes: Dict[str, str] = {c: "unknown" for c in cols}

    # Try pandas sample only when we have >=2 columns, otherwise it may be misleading.
    if infer_rows > 0 and len(cols) >= 2:
        try:
            sep = "\t" if delim == "WHITESPACE" else ("," if delim == "RAW" else delim)
            # pandas can't directly use WHITESPACE as sep here; use delim_whitespace
            if delim == "WHITESPACE":
                dfi = pd.read_csv(path, nrows=infer_rows, engine="python", delim_whitespace=True, low_memory=False)
            else:
                dfi = pd.read_csv(path, sep=sep, nrows=infer_rows, engine="python", low_memory=False, on_bad_lines="skip")

            # Re-take columns from pandas if it parsed more convincingly
            if len(dfi.columns) > len(cols):
                cols = list(map(str, dfi.columns))
                dtypes = {c: str(dfi[c].dtype) for c in cols}
                return cols, dtypes, "PANDAS_AUTO", "pandas_sample"

            # normal dtype inference
            for c in cols:
                if c in dfi.columns:
                    dtypes[c] = str(dfi[c].dtype)
            return cols, dtypes, delim, "pandas_sample"
        except Exception:
            # If pandas fails, keep header-only result
            return cols, dtypes, delim, "header_only"

    return cols, dtypes, delim, "header_only"


def read_json_schema(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.keys())
    if isinstance(obj, list) and obj:
        keys = set()
        for item in obj[:50]:
            if isinstance(item, dict):
                keys.update(item.keys())
        return sorted(keys)
    return []


def read_jsonl_schema(path: Path, max_lines: int = 2000) -> List[str]:
    keys = set()
    n = 0
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    keys.update(obj.keys())
            except Exception:
                pass
            n += 1
            if n >= max_lines:
                break
    return sorted(keys)


def iter_data_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            if p.name in SKIP_DIR_NAMES:
                continue
            continue
        if p.suffix.lower() in SKIP_EXTS:
            continue
        if p.name.startswith("."):
            continue
        files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Inspect schema (columns/keys) for analysis_closed_loop outputs.")
    parser.add_argument("--analysis_dir", type=str, default="KRAS_sampling_results/analysis_closed_loop")
    parser.add_argument("--out_dir", type=str, default="KRAS_sampling_results/analysis_closed_loop/_schema_report")
    parser.add_argument("--infer_rows", type=int, default=200)

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
            "parse_status": "ok",
            "extra": "",
        }

        try:
            if ext in {".csv", ".tsv"}:
                cols, dtypes, delim_used, mode = read_csv_schema_robust(fp, infer_rows=args.infer_rows)
                record["n_cols"] = len(cols)
                record["cols_or_keys"] = "|".join([str(c) for c in cols])
                record["dtype_preview"] = "|".join([f"{c}:{dtypes.get(c,'')}" for c in cols[:60]])
                record["extra"] = f"delim={delim_used};mode={mode}"

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
                record["parse_status"] = "skipped_unknown_ext"

        except Exception as e:
            record["parse_status"] = f"error:{type(e).__name__}"
            record["cols_or_keys"] = ""
            record["dtype_preview"] = ""
            record["extra"] = str(e)[:200]

        file_records.append(record)

    df = pd.DataFrame(file_records)
    df.to_csv(out_dir / "files_index.csv", index=False)

    with (out_dir / "columns_to_files.json").open("w", encoding="utf-8") as f:
        json.dump(columns_to_files, f, indent=2, ensure_ascii=False)

    with (out_dir / "normalized_to_raw_columns.json").open("w", encoding="utf-8") as f:
        json.dump(norm_to_raw, f, indent=2, ensure_ascii=False)

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

