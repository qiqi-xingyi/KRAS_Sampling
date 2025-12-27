# --*-- conding:utf-8 --*--
# @time:12/25/25 01:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_docking_verify.py

# run_docking_verify.py
"""
Top-level runner script for docking_verify.

This script lives OUTSIDE the docking_verify package and calls the pipeline dispatcher
to execute the full validation workflow end-to-end.

Example:
  python run_docking_verify.py \
    --cases_csv /Users/yuqizhang/Desktop/Code/KRAS_QSAD/docking_data/cases.csv \
    --result_root /Users/yuqizhang/Desktop/Code/KRAS_QSAD/docking_result \
    --wt_group_key 4LPK_WT \
    --groups 4LPK_WT 6OIM_G12C 9C41_G12D \
    --n_repeats 5 \
    --base_seed 0 \
    --vina_bin /opt/anaconda3/envs/docking/bin/vina \
    --obabel_bin obabel

If --groups is omitted, it will run all groups found in cases.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from docking_verify import (
    DockingPipeline,
    PipelineConfig,
    BoxConfig,
    VinaParams,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run docking_verify end-to-end pipeline.")
    p.add_argument("--cases_csv", type=Path, required=True, help="Path to docking_data/cases.csv")
    p.add_argument("--result_root", type=Path, default=Path("docking_result"), help="Output root folder")

    p.add_argument("--groups", nargs="*", default=None, help="Optional list of target_group_key to run")
    p.add_argument("--wt_group_key", type=str, default="4LPK_WT", help="WT group key for fallback fragments")

    # Tools
    p.add_argument("--vina_bin", type=str, default="vina", help="Path to AutoDock Vina executable")
    p.add_argument("--obabel_bin", type=str, default="obabel", help="Path to OpenBabel obabel executable")

    # Vina params
    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--num_modes", type=int, default=20)
    p.add_argument("--energy_range", type=int, default=3)
    p.add_argument("--cpu", type=int, default=8)

    # Repeats / seeds
    p.add_argument("--n_repeats", type=int, default=5, help="Number of repeated dockings per group (if no seed_list)")
    p.add_argument("--base_seed", type=int, default=0, help="Seeds = base_seed + i")
    p.add_argument(
        "--seed_list",
        type=int,
        nargs="*",
        default=None,
        help="Explicit seeds (overrides n_repeats/base_seed)",
    )

    # Box config
    p.add_argument("--margin", type=float, default=10.0, help="Å margin around ligand bbox")
    p.add_argument("--min_size", type=float, default=20.0, help="Minimum box size per axis (Å)")
    p.add_argument("--max_size", type=float, default=None, help="Maximum box size per axis (Å)")
    p.add_argument(
        "--fixed_size",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="If provided, use fixed box size (Å) instead of ligand bbox sizing",
    )
    p.add_argument(
        "--ligand_instance_policy",
        type=str,
        default="largest",
        choices=["largest", "first"],
        help="How to choose ligand instance when multiple same resname exist",
    )

    # Behavior
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing vina runs")
    p.add_argument("--non_strict", action="store_true", help="Do not fail-fast; keep going per group")

    # Optional OpenBabel ligand pH
    p.add_argument("--ligand_ph", type=float, default=None, help="Optional OpenBabel -p <pH> for ligand hydrogenation")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    vina_params = VinaParams(
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        cpu=args.cpu,
    )

    box_cfg = BoxConfig(
        margin=float(args.margin),
        min_size=float(args.min_size),
        max_size=float(args.max_size) if args.max_size is not None else None,
        fixed_size=tuple(args.fixed_size) if args.fixed_size is not None else None,
        select_ligand_instance=str(args.ligand_instance_policy),
    )

    cfg = PipelineConfig(
        cases_csv=args.cases_csv,
        result_root=args.result_root,
        obabel_bin=args.obabel_bin,
        vina_bin=args.vina_bin,
        wt_group_key=args.wt_group_key,
        vina_params=vina_params,
        n_repeats=int(args.n_repeats),
        seed_list=args.seed_list,
        base_seed=int(args.base_seed),
        box=box_cfg,
        overwrite=bool(args.overwrite),
        strict=not bool(args.non_strict),
        ligand_ph=args.ligand_ph,
    )

    pipeline = DockingPipeline(cfg)

    if args.groups and len(args.groups) > 0:
        results = pipeline.run_all(target_groups=args.groups)
    else:
        results = pipeline.run_all()

    # Print a minimal pointer to the final summary
    summary_dir = cfg.result_root / cfg.pipeline_step_dirname
    print(f"[OK] Completed groups: {len(results)}")
    print(f"[OK] Pipeline summary CSV: {summary_dir / 'pipeline_summary.csv'}")
    print(f"[OK] Pipeline report JSON: {summary_dir / 'pipeline_report.json'}")


if __name__ == "__main__":
    main()
