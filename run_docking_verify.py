# --*-- conding:utf-8 --*--
# @time:12/25/25 01:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_docking_verify.py


"""
IDE-friendly runner for docking_verify.

- No command-line parameters required.
- Just press Run in your IDE.
- You can still run it in terminal as: python run_docking_verify_ide.py

IMPORTANT:
- Set WORKDIR to your project root (the folder that contains docking_verify/ and docking_data/).
- Update VINA_BIN / OBABEL_BIN if they are not on PATH.
"""

from __future__ import annotations

import os
from pathlib import Path

from docking_verify import DockingPipeline, PipelineConfig, BoxConfig, VinaParams


# =============================
# User-editable configuration
# =============================

# Project root directory (set this to your repo root)
# If None, uses current working directory.
WORKDIR: str | None = None
# Example (uncomment and modify):
# WORKDIR = "/Users/yuqizhang/Desktop/Code/KRAS_QSAD"

CASES_CSV = Path("docking_data/cases.csv")
RESULT_ROOT = Path("docking_result")

# Groups to run. Set to None to run all groups in cases.csv.
TARGET_GROUPS = ["4LPK_WT", "6OIM_G12C", "9C41_G12D"]

WT_GROUP_KEY = "4LPK_WT"

# Tool paths
VINA_BIN = "vina"
# Example:
# VINA_BIN = "/opt/anaconda3/envs/docking/bin/vina"

OBABEL_BIN = "obabel"

# Vina parameters
VINA_PARAMS = VinaParams(
    exhaustiveness=16,
    num_modes=20,
    energy_range=3,
    cpu=8,
)

# Repeats / seeds
N_REPEATS = 5
BASE_SEED = 0
SEED_LIST = None  # e.g. [0, 1, 2, 3, 4] (overrides N_REPEATS/BASE_SEED)

# Docking box policy (ligand-centered)
BOX_CFG = BoxConfig(
    margin=10.0,
    min_size=20.0,
    max_size=None,
    fixed_size=None,  # e.g. (20.0, 20.0, 20.0)
    select_ligand_instance="largest",  # or "first"
)

# Behavior
OVERWRITE = False
STRICT = True  # set False to continue even if a group fails

# Optional: OpenBabel ligand pH (mostly affects ligand hydrogenation)
LIGAND_PH = None  # e.g. 7.4


# =============================
# Runner
# =============================
def main() -> None:
    if WORKDIR is not None:
        os.chdir(WORKDIR)

    cfg = PipelineConfig(
        cases_csv=CASES_CSV,
        result_root=RESULT_ROOT,
        obabel_bin=OBABEL_BIN,
        vina_bin=VINA_BIN,
        wt_group_key=WT_GROUP_KEY,
        vina_params=VINA_PARAMS,
        n_repeats=N_REPEATS,
        seed_list=SEED_LIST,
        base_seed=BASE_SEED,
        box=BOX_CFG,
        overwrite=OVERWRITE,
        strict=STRICT,
        ligand_ph=LIGAND_PH,
    )

    pipeline = DockingPipeline(cfg)


    if TARGET_GROUPS is None:
        results = pipeline.run_all()
    else:
        results = pipeline.run_all(target_groups=TARGET_GROUPS)

    summary_dir = cfg.result_root / cfg.pipeline_step_dirname
    print(f"[OK] Completed groups: {len(results)}")
    print(f"[OK] Pipeline summary CSV: {summary_dir / 'pipeline_summary.csv'}")
    print(f"[OK] Pipeline report JSON: {summary_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()

