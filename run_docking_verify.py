# --*-- conding:utf-8 --*--
# @time:12/25/25 01:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_docking_verify.py

"""
IDE-clickable runner for docking_verify.

How to use:
- Put this file in your repo root (or a place where `import docking_verify` works).
- Edit the CONFIG section below.
- Click Run in your IDE.

This runner does not use argparse; it is designed for direct execution.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import docking_verify as dv


# =========================
# CONFIG (edit me)
# =========================

# Path to cases.csv produced by docking_verify.dataset
CASES_CSV = Path("docking_data/cases.csv")

# Output directory (will be created)
OUT_DIR = Path("dock_out")

# Executables (if not absolute paths, must be discoverable in PATH)
VINA_EXE = "vina"
OBABEL_EXE = "obabel"

# Modes:
# - receptor_mode: "hybrid" or "crystal"
# - ligand_mode  : "from_crystal" or "external"
RECEPTOR_MODE = "hybrid"
LIGAND_MODE = "from_crystal"

# If ligand_mode == "external", set this to your ligand file (sdf/mol2/pdb/...)
EXTERNAL_LIGAND = None  # e.g., Path("ligands/my_ligand.sdf")

# Repeats and box
SEEDS = [0, 1, 2, 3, 4]
BOX_SIZE = (20.0, 20.0, 20.0)  # (sx, sy, sz)

# Vina parameters
VINA_PARAMS = dv.VinaParams(
    exhaustiveness=16,
    num_modes=20,
    energy_range=3,
    cpu=8,
)

# Receptor stripping behavior
# Keep these HET residue names in receptor stripping (rarely needed, e.g. MG, ZN)
KEEP_HET_RESNAMES_IN_RECEPTOR = []  # e.g., ["MG"]
REMOVE_WATER = True

# Robustness controls
RESUME = True      # skip steps if outputs exist
STRICT = False     # True -> stop immediately on first failure
POCKET_HIT_MARGIN = 0.0  # Å

# =========================
# END CONFIG
# =========================


def _resolve_exe(exe: str, label: str) -> str:
    p = Path(exe)
    if p.exists() and p.is_file():
        return str(p.resolve())

    found = shutil.which(exe)
    if not found:
        raise RuntimeError(
            f"[{label}] Cannot find executable '{exe}'.\n"
            f"- If it's installed in your conda env, make sure your IDE is using that interpreter.\n"
            f"- Or set {label}_EXE to an absolute path."
        )
    return found


def main() -> None:
    # Basic validations
    cases_csv = CASES_CSV.expanduser().resolve()
    out_dir = OUT_DIR.expanduser().resolve()

    if not cases_csv.exists():
        raise FileNotFoundError(f"cases.csv not found: {cases_csv}")

    vina_exe = _resolve_exe(VINA_EXE, "VINA")
    obabel_exe = _resolve_exe(OBABEL_EXE, "OBABEL")

    external_ligand = None
    if LIGAND_MODE == "external":
        if EXTERNAL_LIGAND is None:
            raise RuntimeError("LIGAND_MODE='external' but EXTERNAL_LIGAND is None.")
        external_ligand = Path(EXTERNAL_LIGAND).expanduser().resolve()
        if not external_ligand.exists():
            raise FileNotFoundError(f"external ligand not found: {external_ligand}")

    print("=== docking_verify IDE runner ===")
    print(f"cases_csv      : {cases_csv}")
    print(f"out_dir        : {out_dir}")
    print(f"receptor_mode  : {RECEPTOR_MODE}")
    print(f"ligand_mode    : {LIGAND_MODE}")
    if external_ligand is not None:
        print(f"external_ligand: {external_ligand}")
    print(f"vina           : {vina_exe}")
    print(f"obabel         : {obabel_exe}")
    print(f"seeds          : {SEEDS}")
    print(f"box_size       : {BOX_SIZE}")
    print(f"vina_params    : {VINA_PARAMS}")
    print(f"resume         : {RESUME}")
    print(f"strict         : {STRICT}")
    print("===============================")

    reports_dir = dv.run_pipeline(
        cases_csv=cases_csv,
        out_dir=out_dir,
        vina_exe=vina_exe,
        obabel_exe=obabel_exe,
        receptor_mode=RECEPTOR_MODE,
        ligand_mode=LIGAND_MODE,
        external_ligand=str(external_ligand) if external_ligand is not None else None,
        seeds=SEEDS,
        box_size=BOX_SIZE,
        vina_params=VINA_PARAMS,
        keep_het_resnames_in_receptor=KEEP_HET_RESNAMES_IN_RECEPTOR,
        remove_water=REMOVE_WATER,
        resume=RESUME,
        margin_for_pocket_hit=POCKET_HIT_MARGIN,
        strict=STRICT,
    )

    print("\n=== Done ===")
    print(f"Reports dir: {reports_dir}")
    print(f"- runs.csv       : {Path(reports_dir) / 'runs.csv'}")
    print(f"- summary.csv    : {Path(reports_dir) / 'summary.csv'}")
    print(f"- aggregate.json : {Path(reports_dir) / 'aggregate.json'}")
    print(f"Case statuses    : {out_dir / 'status'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FAILED]", type(e).__name__, str(e), file=sys.stderr)
        raise
