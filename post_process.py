# --*-- conding:utf-8 --*--
# @time:10/21/25 14:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:post_process.py

# process_all.py
# Batch runner: iterate through subfolders under ./quantum_data and run the QSAD pipeline.
# Results are saved to ./pp_result/<pdbid>; qsad_rmsd_summary.csv/jsonl are generated at the end.

import os
import json
import csv
import traceback
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Any

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions
from qsadpp.feature_calculator import FeatureConfig

BASE_DIR = Path("./quantum_data")
OUT_ROOT = Path("./pp_result")
STAGING_ROOT = OUT_ROOT / "_staging"

INCLUDE_PATTERNS = [
    "samples_*.csv",
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


def stage_input(src_dir: Path, staging_dir: Path, use_symlink: bool = True) -> None:
    """
    Copy or symlink only the files matching INCLUDE_PATTERNS
    while skipping those matching EXCLUDE_PATTERNS.
    """
    ensure_dir(staging_dir)
    for f in sorted(src_dir.iterdir()):
        if not f.is_file():
            continue
        if _match_any(f.name, EXCLUDE_PATTERNS):
            continue
        if not _match_any(f.name, INCLUDE_PATTERNS):
            continue
        dst = staging_dir / f.name
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
        path.write_text("pdb_id,status,message\n", encoding="utf-8")
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

    aggregate: List[Dict[str, Any]] = []

    for pdb_path in targets:
        pdb_id = pdb_path.name
        out_dir = OUT_ROOT / pdb_id
        staging_dir = STAGING_ROOT / pdb_id
        ensure_dir(out_dir)

        print(f"==> Processing {pdb_id}")
        try:
            # Prepare a temporary folder excluding timing files
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            stage_input(pdb_path, staging_dir, use_symlink=True)

            # Run orchestrator
            cfg = make_cfg(staging_dir, out_dir)
            runner = PipelineOrchestrator(cfg)
            summary = runner.run()

            print(f"[OK] {pdb_id}")
            row = {"pdb_id": pdb_id, "status": "ok", "out_dir": str(out_dir)}
            if isinstance(summary, dict):
                for k in ("num_decoded", "num_energy", "num_feature", "time_sec"):
                    if k in summary:
                        row[k] = summary[k]
            aggregate.append(row)

            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary if isinstance(summary, dict) else {"summary": str(summary)},
                          f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[FAIL] {pdb_id}: {e}")
            aggregate.append({
                "pdb_id": pdb_id,
                "status": "fail",
                "message": str(e),
                "out_dir": str(out_dir),
            })
            with (out_dir / "error.log").open("w", encoding="utf-8") as f:
                f.write("".join(traceback.format_exc()))

    write_csv(OUT_ROOT / "qsad_rmsd_summary.csv", aggregate)
    write_jsonl(OUT_ROOT / "summary.jsonl", aggregate)
    print(f"\nAll done. Summary at {OUT_ROOT}/qsad_rmsd_summary.csv & summary.jsonl")


if __name__ == "__main__":
    main()
