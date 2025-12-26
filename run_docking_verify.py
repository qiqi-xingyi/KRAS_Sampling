# --*-- conding:utf-8 --*--
# @time:12/25/25 01:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_docking_verify.py

# --*-- conding:utf-8 --*--
# @time:12/26/25 00:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_docking_verify.py

"""
IDE-clickable runner for docking_verify (template-fit hybrid pipeline).

Pipeline (minimal & robust):
- Extract receptor_base = crystal protein with ligand removed
- Build hybrid receptor by template-fit (Kabsch CA alignment + graft) using decoded.jsonl
- Convert receptor to rigid PDBQT via OpenBabel (-xr + sanitize)
- Extract ligand from crystal and convert to PDBQT via OpenBabel
- Run Vina multi-seed
- Write reports: dock_out/reports/summary.csv and aggregate.json

Usage:
- Put this file in your repo root (or wherever `import docking_verify` works)
- Edit CONFIG below
- Click Run in your IDE
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import docking_verify as dv


# =========================
# CONFIG (edit me)
# =========================

# Path to cases.csv
CASES_CSV = Path("docking_data/cases.csv")

# Output directory
OUT_DIR = Path("dock_out")

# Executables (absolute paths recommended if IDE PATH is messy)
VINA_EXE = "vina"
OBABEL_EXE = "obabel"

# Modes:
# - receptor_mode: "hybrid_templatefit" or "crystal"
# - ligand_mode  : only "from_crystal" supported in this minimal runner
RECEPTOR_MODE = "hybrid_templatefit"
LIGAND_MODE = "from_crystal"

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

# Receptor stripping behavior (optional)
KEEP_HET_RESNAMES_IN_RECEPTOR = []  # e.g., ["MG", "ZN"]
REMOVE_WATER = True

# Robustness controls
RESUME = False      # skip steps if outputs exist
STRICT = False     # True -> stop immediately on first failure

# =========================
# END CONFIG
# =========================


def _resolve_exe(exe: str, label: str) -> str:
    """
    Resolve executable path:
    - if exe is an existing file -> absolute path
    - else search PATH using shutil.which
    """
    p = Path(exe)
    if p.exists() and p.is_file():
        return str(p.resolve())

    found = shutil.which(exe)
    if not found:
        raise RuntimeError(
            f"[{label}] Cannot find executable '{exe}'.\n"
            f"- If it's installed in your conda env, ensure your IDE uses that interpreter.\n"
            f"- Or set {label}_EXE to an absolute path."
        )
    return found


def main() -> None:
    cases_csv = CASES_CSV.expanduser().resolve()
    out_dir = OUT_DIR.expanduser().resolve()

    if not cases_csv.exists():
        raise FileNotFoundError(f"cases.csv not found: {cases_csv}")

    vina_exe = _resolve_exe(VINA_EXE, "VINA")
    obabel_exe = _resolve_exe(OBABEL_EXE, "OBABEL")

    if RECEPTOR_MODE not in ("hybrid_templatefit", "crystal"):
        raise RuntimeError(
            f"Unsupported RECEPTOR_MODE={RECEPTOR_MODE}. "
            "Use 'hybrid_templatefit' or 'crystal'."
        )
    if LIGAND_MODE != "from_crystal":
        raise RuntimeError("This runner only supports LIGAND_MODE='from_crystal'.")

    print("=== docking_verify IDE runner (template-fit) ===")
    print(f"cases_csv      : {cases_csv}")
    print(f"out_dir        : {out_dir}")
    print(f"receptor_mode  : {RECEPTOR_MODE}")
    print(f"ligand_mode    : {LIGAND_MODE}")
    print(f"vina           : {vina_exe}")
    print(f"obabel         : {obabel_exe}")
    print(f"seeds          : {SEEDS}")
    print(f"box_size       : {BOX_SIZE}")
    print(f"vina_params    : {VINA_PARAMS}")
    print(f"remove_water   : {REMOVE_WATER}")
    print(f"keep_het       : {KEEP_HET_RESNAMES_IN_RECEPTOR}")
    print(f"resume         : {RESUME}")
    print(f"strict         : {STRICT}")
    print("==============================================")

    res = dv.run_pipeline(
        cases_csv=cases_csv,
        out_dir=out_dir,
        vina_exe=vina_exe,
        obabel_exe=obabel_exe,
        receptor_mode=RECEPTOR_MODE,
        ligand_mode=LIGAND_MODE,
        seeds=SEEDS,
        box_size=BOX_SIZE,
        vina_params=VINA_PARAMS,
        keep_het_resnames_in_receptor=KEEP_HET_RESNAMES_IN_RECEPTOR,
        remove_water=REMOVE_WATER,
        resume=RESUME,
        strict=STRICT,
    )

    print("\n=== Done ===")
    # new run_pipeline returns dict with summary_csv/aggregate_json/n_cases
    print(f"summary_csv    : {res.get('summary_csv')}")
    print(f"aggregate_json : {res.get('aggregate_json')}")
    print(f"n_cases        : {res.get('n_cases')}")
    print(f"Case statuses  : {out_dir / 'status'}")
    print(f"Vina outputs   : {out_dir / 'vina_out'}")
    print(f"PDBQT outputs  : {out_dir / 'pdbqt'}")
    print(f"Receptors      : {out_dir / 'receptors'}")
    print(f"Ligands        : {out_dir / 'ligands'}")
    print(f"Reports        : {out_dir / 'reports'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FAILED]", type(e).__name__, str(e), file=sys.stderr)
        raise


