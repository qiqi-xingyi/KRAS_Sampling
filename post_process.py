# --*-- coding:utf-8 --*--
# @time:10/21/25 14:23 (patched)
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:post_process.py
#
# Batch runner:
# - Iterate subfolders under ./quantum_data
# - Infer protein_id from CSV column 'pdbid' (preferred) or 'protein'
# - Group multiple tasks (e.g., KRAS_4LPK_WT_1/2/3) by the same protein_id
# - Stage inputs into pp_result/_staging/<protein_id>/ with unique filenames
# - Run QSAD pipeline once per protein_id
# - Results saved to ./pp_result/<protein_id>/<protein_id>/{decoded,energies,features}.jsonl

import os
import json
import csv
import traceback
import shutil
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions
from qsadpp.feature_calculator import FeatureConfig

BASE_DIR = Path("./KRAS_sampling_results")
OUT_ROOT = Path("./pp_result")
STAGING_ROOT = OUT_ROOT / "_staging"

INCLUDE_PATTERNS = [
    "samples_*.csv",
    "*.csv",
    "*.jsonl",
    "*.pdb",
    "*.pdb.gz",
]

EXCLUDE_PATTERNS = [
    "*timing*.csv",
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def discover_targets(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def _match_any(name: str, patterns: List[str]) -> bool:
    return any(glob.fnmatch.fnmatch(name, pat) for pat in patterns)


def infer_protein_id(src_dir: Path) -> str:
    """
    Infer protein_id for output folder naming.
    Priority:
      1) read one row from a samples_*.csv (or any *.csv) and use column 'pdbid' if present
      2) fallback: column 'protein' if present
      3) fallback: parse folder name like 'KRAS_4LPK_WT_1' -> '4LPK_WT'
      4) fallback: folder name itself
    """
    # 1) find a csv to probe
    csvs = sorted(src_dir.glob("samples_*.csv"))
    if not csvs:
        csvs = sorted(src_dir.glob("*.csv"))

    if csvs:
        try:
            df0 = pd.read_csv(csvs[0], nrows=1)
            if "pdbid" in df0.columns and len(df0) > 0:
                v = str(df0.loc[0, "pdbid"]).strip()
                if v and v.lower() != "nan":
                    return v
            if "protein" in df0.columns and len(df0) > 0:
                v = str(df0.loc[0, "protein"]).strip()
                if v and v.lower() != "nan":
                    return v
        except Exception:
            pass

    # 2) parse folder name: KRAS_4LPK_WT_1 -> 4LPK_WT
    name = src_dir.name.strip()
    m = re.match(r"^KRAS_([0-9A-Za-z]{4}_[0-9A-Za-z]+)_(\d+)$", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"  # 4LPK_WT_1 / 4LPK_WT_2 / 4LPK_WT_3

    # 3) fallback
    return name


def stage_input(src_dir: Path, staging_dir: Path, prefix: str, use_symlink: bool = True) -> None:
    """
    Copy or symlink only the files matching INCLUDE_PATTERNS
    while skipping those matching EXCLUDE_PATTERNS.

    To avoid filename collisions when merging multiple task folders into one staging_dir,
    destination filenames are prefixed with '<prefix>__'.
    """
    ensure_dir(staging_dir)
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if _match_any(f.name, EXCLUDE_PATTERNS):
            continue
        if not _match_any(f.name, INCLUDE_PATTERNS):
            continue

        dst = staging_dir / f"{prefix}__{f.name}"
        try:
            if use_symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(f.resolve())
            else:
                shutil.copy2(f, dst)
        except Exception:
            shutil.copy2(f, dst)


def make_cfg(pdb_dir: Path, out_dir: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        pdb_dir=str(pdb_dir),
        reader_options=ReaderOptions(
            chunksize=100_000,
            strict=True,
            categorize_strings=True,
            include_all_csv=False,
        ),
        fifth_bit=False,
        out_dir=str(out_dir),
        compute_features=True,
        feature_from="decoded",
        combined_feature_name="features.jsonl",
        feature_config=FeatureConfig(
            output_format="jsonl",
        ),
    )


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("protein_id,status,message\n", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ensure_dir(OUT_ROOT)
    ensure_dir(STAGING_ROOT)

    targets = discover_targets(BASE_DIR)
    if not targets:
        print(f"[WARN] No subfolders under {BASE_DIR}")
        return

    # Group task folders by inferred protein_id (e.g., 4LPK_WT)
    groups: Dict[str, List[Path]] = {}
    for task_dir in targets:
        pid = infer_protein_id(task_dir)
        groups.setdefault(pid, []).append(task_dir)

    aggregate: List[Dict[str, Any]] = []

    for protein_id, task_dirs in sorted(groups.items(), key=lambda x: x[0]):
        out_dir = OUT_ROOT / protein_id
        staging_dir = STAGING_ROOT / protein_id
        ensure_dir(out_dir)

        print(f"==> Processing protein_id={protein_id} from {len(task_dirs)} task folder(s)")
        try:
            # Prepare staging dir (merged tasks)
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            ensure_dir(staging_dir)

            for td in task_dirs:
                stage_input(td, staging_dir, prefix=td.name, use_symlink=True)

            # Run orchestrator once per protein_id
            cfg = make_cfg(staging_dir, out_dir)
            runner = PipelineOrchestrator(cfg)
            summary = runner.run()

            print(f"[OK] {protein_id}")
            row: Dict[str, Any] = {
                "protein_id": protein_id,
                "status": "ok",
                "out_dir": str(out_dir),
                "task_dirs": ",".join([d.name for d in task_dirs]),
            }
            if isinstance(summary, dict):
                for k in ("groups", "decoded_rows", "energy_rows", "feature_rows", "decoded_all", "energies_all", "features_all"):
                    if k in summary:
                        row[k] = summary[k]
            aggregate.append(row)

            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary if isinstance(summary, dict) else {"summary": str(summary)},
                          f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[FAIL] {protein_id}: {e}")
            aggregate.append({
                "protein_id": protein_id,
                "status": "fail",
                "message": str(e),
                "out_dir": str(out_dir),
                "task_dirs": ",".join([d.name for d in task_dirs]),
            })
            with (out_dir / "error.log").open("w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exc()))

    write_csv(OUT_ROOT / "qsad_rmsd_summary.csv", aggregate)
    write_jsonl(OUT_ROOT / "summary.jsonl", aggregate)
    print(f"\nAll done. Summary at {OUT_ROOT}/qsad_rmsd_summary.csv & summary.jsonl")


if __name__ == "__main__":
    main()
