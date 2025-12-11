# --*-- conding:utf-8 --*--
# @time:11/1/25 01:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data_post_process.py

import os
import re
import json
import shutil
from pathlib import Path

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions
from qsadpp.feature_calculator import FeatureConfig

# --------- Paths ---------
ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "sampling_data"       # contains samples_*.csv
WORK_DIR  = ROOT / "tmp_work"            # per-protein staging dir (created/removed)
OUT_DIR   = ROOT / "training_data"    # new output root directory

# Aggregated corpus outputs
ALL_ENERGIES = OUT_DIR / "all_energies.jsonl"
ALL_FEATURES = OUT_DIR / "all_features.jsonl"
PROCESSED_TXT = OUT_DIR / "processed.txt"   # list of pdb_ids already aggregated

# --------- Utilities ---------
PDB_RE = re.compile(r"^samples_(?P<pdb>[A-Za-z0-9]+)_.*\.csv$")

def list_training_files():
    for p in sorted(TRAIN_DIR.glob("samples_*.csv")):
        m = PDB_RE.match(p.name)
        if not m:
            continue
        yield m.group("pdb"), p

def append_file(src: Path, dst: Path):
    if not src.exists():
        return 0
    n = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("a", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)
            n += 1
    return n

def load_processed():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PROCESSED_TXT.exists():
        return set()
    with PROCESSED_TXT.open("r", encoding="utf-8") as f:
        return set(x.strip() for x in f if x.strip())

def mark_processed(pdb_id: str):
    with PROCESSED_TXT.open("a", encoding="utf-8") as f:
        f.write(pdb_id + "\n")

def run_orchestrator_on_single_csv(pdb_id: str, csv_path: Path):
    # Stage this single CSV into its own directory so SampleReader sees only one protein
    WORK_DIR.mkdir(exist_ok=True, parents=True)
    stage_dir = WORK_DIR / pdb_id
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged = stage_dir / csv_path.name
    shutil.copy2(csv_path, staged)

    # Configure pipeline
    cfg = OrchestratorConfig(
        pdb_dir=str(stage_dir),
        reader_options=ReaderOptions(
            chunksize=100_000,
            strict=True,
            categorize_strings=True,
            include_all_csv=False,
        ),
        fifth_bit=False,
        out_dir=str(OUT_DIR),

        # enable features
        compute_features=True,
        feature_from="decoded",
        combined_feature_name="features.jsonl",
        feature_config=FeatureConfig(output_format="jsonl"),
    )

    runner = PipelineOrchestrator(cfg)
    summary = runner.run()

    protein_root = OUT_DIR / pdb_id
    energies_path = protein_root / "energies.jsonl"
    features_path = protein_root / "features.jsonl"

    return {
        "summary": summary,
        "energies": energies_path,
        "features": features_path,
        "protein_root": protein_root
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Create or keep aggregated files (resume-safe)
    ALL_ENERGIES.touch(exist_ok=True)
    ALL_FEATURES.touch(exist_ok=True)

    processed = load_processed()
    print(f"[INFO] Aggregated outputs: {ALL_ENERGIES.name}, {ALL_FEATURES.name}")
    print(f"[INFO] Already processed: {len(processed)} proteins")

    total_new_energy = 0
    total_new_feat = 0
    done = 0

    for pdb_id, csv_path in list_training_files():
        if pdb_id in processed:
            print(f"[SKIP] {pdb_id}: already aggregated.")
            continue

        print(f"[RUN ] {pdb_id}: post-processing {csv_path.name} ...")
        result = run_orchestrator_on_single_csv(pdb_id, csv_path)

        # Append this protein's results into global corpus
        nE = append_file(result["energies"], ALL_ENERGIES)
        nF = append_file(result["features"], ALL_FEATURES)
        total_new_energy += nE
        total_new_feat += nF

        mark_processed(pdb_id)
        done += 1

        # Cleanup
        try:
            shutil.rmtree(result["protein_root"])
        except Exception:
            pass
        try:
            shutil.rmtree(WORK_DIR / pdb_id)
        except Exception:
            pass

        print(f"[OK  ] {pdb_id}: +{nE} energy lines, +{nF} feature lines (appended).")

    print(f"\n[SUMMARY] proteins processed this run: {done}")
    print(f"[SUMMARY] appended: {total_new_energy} energies, {total_new_feat} features")
    print(f"[OUTPUT ] energies -> {ALL_ENERGIES}")
    print(f"[OUTPUT ] features -> {ALL_FEATURES}")

if __name__ == "__main__":
    main()
