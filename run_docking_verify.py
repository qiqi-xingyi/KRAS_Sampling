# --*-- conding:utf-8 --*--
# @time:12/25/25 01:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:run_docking_verify.py

"""
IDE-clickable runner for docking_verify (updated for new pipeline).

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
MEEKO_EXE = "mk_prepare_receptor.py"   # from meeko package
PULCHRA_EXE = "pulchra"                # required for hybrid_allatom
SCWRL_EXE = None                       # e.g. "Scwrl4" or "/abs/path/Scwrl4" (optional)

# Modes:
# - receptor_mode: "hybrid_allatom" or "crystal"
# - ligand_mode  : "from_crystal"
RECEPTOR_MODE = "hybrid_allatom"
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

# Robustness controls
RESUME = True      # skip steps if outputs exist
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
    meeko_exe = _resolve_exe(MEEKO_EXE, "MEEKO")

    # pulchra is required when receptor_mode == hybrid_allatom
    if RECEPTOR_MODE == "hybrid_allatom":
        pulchra_exe = _resolve_exe(PULCHRA_EXE, "PULCHRA")
    else:
        pulchra_exe = _resolve_exe(PULCHRA_EXE, "PULCHRA")  # still resolve for consistency

    scwrl_exe = None
    if SCWRL_EXE:
        scwrl_exe = _resolve_exe(str(SCWRL_EXE), "SCWRL")

    print("=== docking_verify IDE runner (updated) ===")
    print(f"cases_csv            : {cases_csv}")
    print(f"out_dir              : {out_dir}")
    print(f"receptor_mode        : {RECEPTOR_MODE}")
    print(f"ligand_mode          : {LIGAND_MODE}")
    print(f"vina                 : {vina_exe}")
    print(f"obabel               : {obabel_exe}")
    print(f"mk_prepare_receptor   : {meeko_exe}")
    print(f"pulchra              : {pulchra_exe}")
    print(f"scwrl                : {scwrl_exe}")
    print(f"seeds                : {SEEDS}")
    print(f"box_size             : {BOX_SIZE}")
    print(f"vina_params          : {VINA_PARAMS}")
    print(f"resume               : {RESUME}")
    print(f"strict               : {STRICT}")
    print("==========================================")

    res = dv.run_pipeline(
        cases_csv=cases_csv,
        out_dir=out_dir,
        receptor_mode=RECEPTOR_MODE,
        ligand_mode=LIGAND_MODE,
        vina_exe=vina_exe,
        obabel_exe=obabel_exe,
        mk_prepare_receptor_exe=meeko_exe,
        pulchra_exe=pulchra_exe,
        scwrl_exe=scwrl_exe,
        seeds=SEEDS,
        box_size=BOX_SIZE,
        vina_params=VINA_PARAMS,
        resume=RESUME,
        strict=STRICT,
    )

    print("\n=== Done ===")
    # new run_pipeline returns dict with summary_csv/aggregate_json/n_cases
    try:
        print(f"summary_csv    : {res.get('summary_csv')}")
        print(f"aggregate_json : {res.get('aggregate_json')}")
        print(f"n_cases        : {res.get('n_cases')}")
    except Exception:
        print(res)

    print(f"Case statuses  : {out_dir / 'status'}")
    print(f"Vina outputs   : {out_dir / 'vina_out'}")
    print(f"PDBQT outputs  : {out_dir / 'pdbqt'}")
    print(f"Receptors      : {out_dir / 'receptors'}")
    print(f"Rebuild        : {out_dir / 'rebuild'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FAILED]", type(e).__name__, str(e), file=sys.stderr)
        raise

