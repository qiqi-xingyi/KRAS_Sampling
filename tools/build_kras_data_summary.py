# --*-- conding:utf-8 --*--
# @time:1/11/26 23:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:build_kras_data_summary.py

# build_kras_data_summary.py
# Purpose:
#   1) Collect (copy) all indexable analysis outputs from:
#        KRAS_sampling_results/analysis_closed_loop
#      into:
#        KRAS_analysis/data_summary/raw
#      preserving relative paths.
#   2) Normalize column names using:
#        analysis_closed_loop/_schema_report/normalized_to_raw_columns.json
#   3) Produce merged/enriched summary tables into:
#        KRAS_analysis/data_summary/merged
#   4) Perform strict completeness checks (no missing copied files, no missing key outputs).

from __future__ import annotations

import json
import shutil
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_ok(msg: str) -> None:
    print(f"[OK] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_err(msg: str) -> None:
    print(f"[ERROR] {msg}")

def find_project_root(start: Path) -> Path:
    """
    Find project root by walking up from 'start' until we see KRAS_sampling_results folder.
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "KRAS_sampling_results").exists():
            return p
    raise FileNotFoundError(
        "Cannot locate project root. Expected a parent directory containing 'KRAS_sampling_results'. "
        f"Start={start}"
    )

def parse_sep(extra: str) -> str:
    # files_index.csv stores something like: sep=,
    if not extra:
        return ","
    extra = extra.strip().strip('"')
    if "sep=" in extra:
        return extra.split("sep=", 1)[1].strip()
    return ","

def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p: Path, obj: dict) -> None:
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_csv_safely(p: Path, sep: str = ",") -> pd.DataFrame:
    # Keep default dtype inference; avoid blowing up memory with dtype=str everywhere.
    return pd.read_csv(p, sep=sep)

def normalize_columns(df: pd.DataFrame, raw_to_norm: Dict[str, str]) -> pd.DataFrame:
    # Rename only columns that are in mapping
    rename_map = {c: raw_to_norm[c] for c in df.columns if c in raw_to_norm}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def build_raw_to_norm(normalized_to_raw: Dict[str, List[str]]) -> Dict[str, str]:
    """
    normalized_to_raw example:
      {"basin_id": ["basin_id"], "g12c": ["G12C"], ...}
    Build raw->normalized map.
    """
    raw_to_norm: Dict[str, str] = {}
    for norm, raw_list in normalized_to_raw.items():
        for raw in raw_list:
            # If duplicates occur, keep the first mapping (stable behavior).
            raw_to_norm.setdefault(raw, norm)
    return raw_to_norm


# ----------------------------
# Config
# ----------------------------

@dataclass
class Paths:
    project_root: Path
    analysis_dir: Path
    schema_dir: Path
    out_root: Path
    out_raw: Path
    out_merged: Path
    out_meta: Path


def make_paths(project_root: Path) -> Paths:
    analysis_dir = project_root / "KRAS_sampling_results" / "analysis_closed_loop"
    schema_dir = analysis_dir / "_schema_report"
    out_root = project_root / "KRAS_analysis" / "data_summary"
    out_raw = out_root / "raw"
    out_merged = out_root / "merged"
    out_meta = out_root / "_meta"
    return Paths(
        project_root=project_root,
        analysis_dir=analysis_dir,
        schema_dir=schema_dir,
        out_root=out_root,
        out_raw=out_raw,
        out_merged=out_merged,
        out_meta=out_meta,
    )


# ----------------------------
# Core pipeline
# ----------------------------

def copy_all_indexed_files(paths: Paths) -> Tuple[pd.DataFrame, List[str]]:
    """
    Copy all files in schema report files_index.csv into out_raw preserving rel paths.
    Returns:
      - df_index: the files_index dataframe
      - missing_after_copy: any rel_path missing after copy
    """
    files_index_csv = paths.schema_dir / "files_index.csv"
    if not files_index_csv.exists():
        raise FileNotFoundError(f"Missing schema index: {files_index_csv}")

    df_index = pd.read_csv(files_index_csv)
    required_cols = {"rel_path", "abs_path", "ext", "parse_status", "extra"}
    missing = required_cols - set(df_index.columns)
    if missing:
        raise ValueError(f"files_index.csv missing columns: {sorted(missing)}")

    ensure_dir(paths.out_raw)

    copied = 0
    skipped = 0
    for _, r in df_index.iterrows():
        rel_path = str(r["rel_path"])
        abs_path = Path(str(r["abs_path"]))
        if not abs_path.exists():
            log_warn(f"Source missing on disk (will report omission): {abs_path}")
            continue

        dst = paths.out_raw / rel_path
        ensure_dir(dst.parent)

        # Avoid re-copy if same size+mtime (fast path); correctness validated later via existence check.
        try:
            if dst.exists() and dst.stat().st_size == abs_path.stat().st_size:
                skipped += 1
            else:
                shutil.copy2(abs_path, dst)
                copied += 1
        except Exception as e:
            raise RuntimeError(f"Failed to copy {abs_path} -> {dst}: {e}") from e

    # Verify completeness: every rel_path must exist in out_raw
    missing_after_copy: List[str] = []
    for rel_path in df_index["rel_path"].astype(str).tolist():
        if not (paths.out_raw / rel_path).exists():
            missing_after_copy.append(rel_path)

    log_ok(f"Copied files: copied={copied}, skipped={skipped}, total_indexed={len(df_index)}")
    return df_index, missing_after_copy


def load_normalization_maps(paths: Paths) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    norm_json = paths.schema_dir / "normalized_to_raw_columns.json"
    if not norm_json.exists():
        raise FileNotFoundError(f"Missing normalization map: {norm_json}")
    normalized_to_raw = load_json(norm_json)
    raw_to_norm = build_raw_to_norm(normalized_to_raw)
    return normalized_to_raw, raw_to_norm


def build_basin_master(
    paths: Paths,
    raw_to_norm: Dict[str, str],
) -> Path:
    """
    Build a basin-level master table by merging the core basin CSVs and optional addon story.
    Output: KRAS_analysis/data_summary/merged/basin_master.csv
    """
    ensure_dir(paths.out_merged)

    def read_norm_csv(rel: str) -> Optional[pd.DataFrame]:
        p = paths.out_raw / rel
        if not p.exists():
            return None
        # In raw copy we lost the sep hint; use schema files_index.csv if needed.
        # Most files here are standard CSV with commas.
        df = read_csv_safely(p, sep=",")
        df = normalize_columns(df, raw_to_norm)
        return df

    # Core
    df_stats = read_norm_csv("basin_stats.csv")
    df_delta = read_norm_csv("basin_delta_summary.csv")
    df_occ = read_norm_csv("basin_occupancy.csv")

    if df_stats is None:
        raise FileNotFoundError("Missing required file in raw: basin_stats.csv")
    if df_delta is None:
        raise FileNotFoundError("Missing required file in raw: basin_delta_summary.csv")
    if df_occ is None:
        raise FileNotFoundError("Missing required file in raw: basin_occupancy.csv")

    # Optional
    df_story = read_norm_csv("addons/key_basin_story.csv")
    # If both story and ranked exist, prefer ranked as "already sorted"
    df_story_ranked = read_norm_csv("addons/key_basin_story_ranked.csv")
    if df_story_ranked is not None:
        df_story = df_story_ranked

    # Merge sequence (basin_id is the anchor key)
    if "basin_id" not in df_stats.columns:
        raise ValueError("basin_stats.csv missing basin_id after normalization.")
    if "basin_id" not in df_delta.columns:
        raise ValueError("basin_delta_summary.csv missing basin_id after normalization.")
    if "basin_id" not in df_occ.columns:
        raise ValueError("basin_occupancy.csv missing basin_id after normalization.")

    # Ensure basin_id is int when possible (robust join)
    for df in [df_stats, df_delta, df_occ]:
        try:
            df["basin_id"] = df["basin_id"].astype("int64")
        except Exception:
            pass

    df_master = df_stats.merge(df_delta, on="basin_id", how="left", suffixes=("", "_delta"))
    df_master = df_master.merge(df_occ, on="basin_id", how="left", suffixes=("", "_occ"))

    if df_story is not None and "basin_id" in df_story.columns:
        try:
            df_story["basin_id"] = df_story["basin_id"].astype("int64")
        except Exception:
            pass

        # Avoid duplicating columns aggressively: keep only story columns not already present
        story_cols = [c for c in df_story.columns if c == "basin_id" or c not in set(df_master.columns)]
        df_master = df_master.merge(df_story[story_cols], on="basin_id", how="left")

    out_p = paths.out_merged / "basin_master.csv"
    df_master.to_csv(out_p, index=False)
    log_ok(f"Wrote basin master: {out_p} (rows={len(df_master)}, cols={len(df_master.columns)})")
    return out_p


def build_representatives_enriched(
    paths: Paths,
    raw_to_norm: Dict[str, str],
    basin_master_csv: Path,
) -> Path:
    """
    Enrich representatives.csv with basin-level metrics.
    Output: KRAS_analysis/data_summary/merged/representatives_enriched.csv
    """
    reps_p = paths.out_raw / "representatives.csv"
    if not reps_p.exists():
        raise FileNotFoundError("Missing required file in raw: representatives.csv")

    df_reps = read_csv_safely(reps_p, sep=",")
    df_reps = normalize_columns(df_reps, raw_to_norm)

    df_basin = pd.read_csv(basin_master_csv)
    # Keep the basin columns compact but useful
    keep = [c for c in df_basin.columns if c in {
        "basin_id",
        "label",
        "mass",
        "e_total_mean",
        "e_total_std",
        "backbone_rmsd_mean",
        "backbone_rmsd_std",
        "wt", "g12c", "g12d",
        "delta_g12c_minus_wt", "delta_g12d_minus_wt",
        "abs_delta_g12c", "abs_delta_g12d",
        "rank_abs_g12c", "rank_abs_g12d",
    }]
    if "basin_id" not in keep:
        keep = ["basin_id"] + keep

    df_enriched = df_reps.merge(df_basin[keep], on="basin_id", how="left", suffixes=("", "_basin"))

    out_p = paths.out_merged / "representatives_enriched.csv"
    df_enriched.to_csv(out_p, index=False)
    log_ok(f"Wrote enriched representatives: {out_p} (rows={len(df_enriched)}, cols={len(df_enriched.columns)})")
    return out_p


def build_points_enriched_chunked(
    paths: Paths,
    raw_to_norm: Dict[str, str],
    basin_master_csv: Path,
    chunksize: int = 500_000,
) -> Path:
    """
    Enrich the large points table with basin-level summary metrics.
    Uses chunked processing to avoid memory spikes.

    Input preference:
      1) merged_points_with_basin.csv
      2) merged_points.csv (if above missing)

    Output:
      KRAS_analysis/data_summary/merged/points_enriched.csv
    """
    p1 = paths.out_raw / "merged_points_with_basin.csv"
    p2 = paths.out_raw / "merged_points.csv"
    if p1.exists():
        points_p = p1
    elif p2.exists():
        points_p = p2
    else:
        raise FileNotFoundError("Missing required points table: merged_points_with_basin.csv or merged_points.csv")

    df_basin = pd.read_csv(basin_master_csv)
    # Minimal basin columns to add to every point (keep file size reasonable)
    basin_keep = [c for c in df_basin.columns if c in {
        "basin_id",
        "label",
        "mass",
        "e_total_mean",
        "backbone_rmsd_mean",
        "wt", "g12c", "g12d",
        "delta_g12c_minus_wt", "delta_g12d_minus_wt",
    }]
    if "basin_id" not in basin_keep:
        basin_keep = ["basin_id"] + basin_keep

    df_basin_small = df_basin[basin_keep].copy()
    # For safety: ensure join key is consistent
    try:
        df_basin_small["basin_id"] = df_basin_small["basin_id"].astype("int64")
    except Exception:
        pass

    out_p = paths.out_merged / "points_enriched.csv"
    if out_p.exists():
        out_p.unlink()

    wrote_header = False
    total_rows = 0

    log_info(f"Chunk-joining points table: {points_p.name} -> {out_p.name} (chunksize={chunksize})")

    # Use pandas chunked reading
    for chunk in pd.read_csv(points_p, chunksize=chunksize):
        chunk = normalize_columns(chunk, raw_to_norm)

        if "basin_id" not in chunk.columns:
            # If we used merged_points.csv without basin_id, this is not mergeable
            raise ValueError(
                f"{points_p.name} lacks 'basin_id' after normalization; cannot enrich. "
                "Use merged_points_with_basin.csv for enrichment."
            )

        try:
            chunk["basin_id"] = chunk["basin_id"].astype("int64")
        except Exception:
            pass

        merged = chunk.merge(df_basin_small, on="basin_id", how="left", suffixes=("", "_basin"))
        merged.to_csv(out_p, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        total_rows += len(merged)
        log_info(f"  processed rows={total_rows}")

    log_ok(f"Wrote enriched points: {out_p} (rows={total_rows})")
    return out_p


def write_run_summary(
    paths: Paths,
    df_index: pd.DataFrame,
    missing_after_copy: List[str],
    outputs: List[Path],
) -> Path:
    """
    Write KRAS_analysis/data_summary/_meta/run_summary.json and manifest with hashes.
    """
    ensure_dir(paths.out_meta)

    # Hash outputs + copied raw files (fast but potentially heavy if raw is huge; your set is small).
    manifest_rows = []

    # Raw copied files
    for rel in df_index["rel_path"].astype(str).tolist():
        p = paths.out_raw / rel
        if not p.exists():
            continue
        manifest_rows.append({
            "group": "raw",
            "rel_path": rel,
            "abs_path": str(p.resolve()),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    # Generated outputs
    for p in outputs:
        manifest_rows.append({
            "group": "merged",
            "rel_path": str(p.relative_to(paths.out_root)),
            "abs_path": str(p.resolve()),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p) if p.exists() else None,
        })

    df_manifest = pd.DataFrame(manifest_rows)
    manifest_csv = paths.out_meta / "manifest_sha256.csv"
    df_manifest.to_csv(manifest_csv, index=False)

    summary = {
        "timestamp_utc": utc_now_iso(),
        "project_root": str(paths.project_root),
        "analysis_dir": str(paths.analysis_dir),
        "schema_dir": str(paths.schema_dir),
        "out_root": str(paths.out_root),
        "indexed_files": int(len(df_index)),
        "missing_after_copy": missing_after_copy,
        "no_omissions_after_copy": len(missing_after_copy) == 0,
        "generated_outputs": [str(p.resolve()) for p in outputs],
        "manifest_csv": str(manifest_csv.resolve()),
    }
    out_json = paths.out_meta / "run_summary.json"
    save_json(out_json, summary)

    log_ok(f"Wrote run summary: {out_json}")
    log_ok(f"Wrote hash manifest: {manifest_csv}")
    return out_json


def main() -> None:
    project_root = find_project_root(Path.cwd())
    paths = make_paths(project_root)

    log_info(f"PROJECT_ROOT: {paths.project_root}")
    log_info(f"ANALYSIS_DIR:  {paths.analysis_dir}")
    log_info(f"SCHEMA_DIR:    {paths.schema_dir}")
    log_info(f"OUT_ROOT:      {paths.out_root}")

    if not paths.analysis_dir.exists():
        raise FileNotFoundError(f"Missing analysis directory: {paths.analysis_dir}")
    if not paths.schema_dir.exists():
        raise FileNotFoundError(f"Missing schema directory: {paths.schema_dir}")

    ensure_dir(paths.out_root)
    ensure_dir(paths.out_raw)
    ensure_dir(paths.out_merged)
    ensure_dir(paths.out_meta)

    # 1) Copy everything that schema indexed
    df_index, missing_after_copy = copy_all_indexed_files(paths)
    if missing_after_copy:
        log_err("Omissions detected after copy:")
        for rel in missing_after_copy:
            log_err(f"  - {rel}")
        raise RuntimeError("Copy completeness check failed (missing files in out_raw).")
    log_ok("Copy completeness check passed (no missing files).")

    # 2) Load normalization mapping
    normalized_to_raw, raw_to_norm = load_normalization_maps(paths)

    # Save mapping for convenience
    save_json(paths.out_meta / "normalized_to_raw_columns.json", normalized_to_raw)
    save_json(paths.out_meta / "raw_to_normalized_columns.json", raw_to_norm)
    log_ok("Saved normalization maps into _meta/")

    # 3) Build merged outputs
    outputs: List[Path] = []

    basin_master_csv = build_basin_master(paths, raw_to_norm)
    outputs.append(basin_master_csv)

    reps_enriched_csv = build_representatives_enriched(paths, raw_to_norm, basin_master_csv)
    outputs.append(reps_enriched_csv)

    points_enriched_csv = build_points_enriched_chunked(paths, raw_to_norm, basin_master_csv, chunksize=500_000)
    outputs.append(points_enriched_csv)

    # 4) Save final run summary + hashes, and hard-check outputs exist
    for p in outputs:
        if not p.exists():
            raise RuntimeError(f"Expected output missing: {p}")

    summary_json = write_run_summary(paths, df_index, missing_after_copy, outputs)
    outputs.append(summary_json)

    log_ok("All done. Final outputs are under: KRAS_analysis/data_summary/")


if __name__ == "__main__":
    main()
