# --*-- conding:utf-8 --*--
# @time:1/7/26 00:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_addons_make_plots.py

"""
Generate ONLY addon outputs for KRAS closed-loop analysis:
1) Per-residue displacement curves (WT vs G12C, WT vs G12D) from exported basin PDBs
2) Energy-term waterfall plots from addons/basin_energy_contrast.csv (LONG format)

Expected paths (relative to project root):
KRAS_sampling_results/analysis_closed_loop/addons/
  - basin_energy_contrast.csv
  - exported_pdb/
      basin01_WT.pdb, basin01_G12C.pdb, basin01_G12D.pdb, ...
Output will be written to:
KRAS_sampling_results/analysis_closed_loop/addons/plots_addon/
KRAS_sampling_results/analysis_closed_loop/addons/per_residue/
"""

from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config (edit here)
# ----------------------------
PROJECT_ROOT = Path(" ").resolve()

ANALYSIS_DIR = PROJECT_ROOT / "KRAS_sampling_results" / "analysis_closed_loop"
ADDONS_DIR = ANALYSIS_DIR / "addons"
EXPORTED_PDB_DIR = ADDONS_DIR / "exported_pdb"
ENERGY_CONTRAST_CSV = ADDONS_DIR / "basin_energy_contrast.csv"

# if None -> auto-detect basins from exported_pdb + energy file intersection
BASINS_TO_PLOT: Optional[List[int]] = None  # e.g., [1, 2, 5, 6]

# which labels to compare
LABELS = ("WT", "G12C", "G12D")

# Only these terms go into "energy waterfall"
ENERGY_TERMS_ORDER = [
    "E_steric",
    "E_geom",
    "E_bond",
    "E_mj",
    "E_dihedral",
    "E_hydroph",
    "E_cbeta",
    "E_rama",
    "E_total",  # keep total last (or move to first if you prefer)
]

# Output dirs
OUT_PLOTS_DIR = ADDONS_DIR / "plots_addon"
OUT_PER_RES_DIR = ADDONS_DIR / "per_residue"


# ----------------------------
# Utils: PDB parsing (CA only)
# ----------------------------
def _pdb_ca_records(pdb_path: Path) -> List[Tuple[Tuple[str, int, str], np.ndarray]]:
    """
    Return list of ((chain, resseq, icode), coord[3]) for CA atoms in file order.
    """
    recs = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21].strip() or " "
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip() or ""
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (chain, resseq, icode)
            recs.append((key, np.array([x, y, z], dtype=float)))
    return recs


def match_ca_by_residue(
    wt_path: Path, mut_path: Path
) -> Tuple[List[Tuple[str, int, str]], np.ndarray, np.ndarray]:
    """
    Match CA coords by residue key intersection (chain, resseq, icode).
    Returns matched keys, X_wt (N,3), X_mut (N,3).
    """
    wt_recs = _pdb_ca_records(wt_path)
    mut_recs = _pdb_ca_records(mut_path)

    wt_map = {k: v for k, v in wt_recs}
    mut_map = {k: v for k, v in mut_recs}

    keys = [k for k, _ in wt_recs if k in mut_map]  # keep WT order
    if len(keys) < 3:
        raise ValueError(f"Too few matched CA residues: {wt_path.name} vs {mut_path.name}")

    X_wt = np.stack([wt_map[k] for k in keys], axis=0)
    X_mut = np.stack([mut_map[k] for k in keys], axis=0)
    return keys, X_wt, X_mut


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align Q to P with Kabsch. Return Q_aligned and RMSD.
    P, Q: (N,3)
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    H = Q0.T @ P0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # ensure right-handed rotation
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    Q_aligned = (Q0 @ R) + Pc
    diff = P - Q_aligned
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return Q_aligned, rmsd


# ----------------------------
# Addon 1: per-residue displacement
# ----------------------------
def plot_per_residue_displacement_for_basin(basin_id: int) -> None:
    OUT_PER_RES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    wt_pdb = EXPORTED_PDB_DIR / f"basin{basin_id:02d}_WT.pdb"
    if not wt_pdb.exists():
        print(f"[WARN] Missing WT PDB for basin {basin_id:02d}: {wt_pdb}")
        return

    for mut in ("G12C", "G12D"):
        mut_pdb = EXPORTED_PDB_DIR / f"basin{basin_id:02d}_{mut}.pdb"
        if not mut_pdb.exists():
            print(f"[WARN] Missing {mut} PDB for basin {basin_id:02d}: {mut_pdb}")
            continue

        keys, X_wt, X_mut = match_ca_by_residue(wt_pdb, mut_pdb)
        X_mut_aligned, rmsd = kabsch_align(X_wt, X_mut)

        disp = np.linalg.norm(X_mut_aligned - X_wt, axis=1)

        # x axis: residue number (resseq)
        resseq = np.array([k[1] for k in keys], dtype=int)

        # save CSV
        out_csv = OUT_PER_RES_DIR / f"basin{basin_id:02d}_WT_vs_{mut}_per_residue_displacement.csv"
        pd.DataFrame(
            {
                "chain": [k[0] for k in keys],
                "resseq": resseq,
                "icode": [k[2] for k in keys],
                "displacement_A": disp,
            }
        ).to_csv(out_csv, index=False)

        # plot
        plt.figure()
        plt.plot(resseq, disp)
        plt.xlabel("Residue index (resseq)")
        plt.ylabel("CA displacement (Å)")
        plt.title(f"Basin {basin_id:02d}: WT vs {mut} per-residue displacement (CA)\nKabsch CA RMSD = {rmsd:.3f} Å")
        plt.tight_layout()

        out_png = OUT_PLOTS_DIR / f"per_residue_basin{basin_id:02d}_WT_vs_{mut}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()

        print(f"[OK] per-residue displacement: basin {basin_id:02d} WT vs {mut} | CA_RMSD={rmsd:.3f} Å")
        print(f"      - CSV: {out_csv}")
        print(f"      - PNG: {out_png}")


# ----------------------------
# Addon 2: energy-term waterfall (LONG format)
# ----------------------------
def plot_energy_waterfall_for_basin(basin_id: int, df: pd.DataFrame) -> None:
    """
    df is long-format energy contrast table with columns:
      basin_id, term, WT, G12C, G12D, delta_G12C_minus_WT, delta_G12D_minus_WT, ...
    """
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    d = df[df["basin_id"] == basin_id].copy()
    if d.empty:
        print(f"[WARN] No energy contrast rows for basin {basin_id:02d}. Skip waterfall.")
        return

    # keep only energy terms
    d = d[d["term"].isin(ENERGY_TERMS_ORDER)].copy()
    if d.empty:
        print(f"[WARN] No energy terms for basin {basin_id:02d}. Skip waterfall.")
        return

    # ensure order
    d["term"] = pd.Categorical(d["term"], categories=ENERGY_TERMS_ORDER, ordered=True)
    d = d.sort_values("term")

    # need delta columns
    if "delta_G12C_minus_WT" not in d.columns or "delta_G12D_minus_WT" not in d.columns:
        print(f"[WARN] Missing delta columns in basin_energy_contrast.csv. Skip waterfall (basin {basin_id:02d}).")
        return

    x = np.arange(len(d))
    width = 0.38

    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, d["delta_G12C_minus_WT"].values, width, label="G12C - WT")
    plt.bar(x + width / 2, d["delta_G12D_minus_WT"].values, width, label="G12D - WT")
    plt.axhline(0.0, linewidth=1)

    plt.xticks(x, d["term"].astype(str).tolist(), rotation=30, ha="right")
    plt.ylabel("Δ energy term (variant - WT)")
    plt.title(f"Basin {basin_id:02d}: Energy-term deltas vs WT")
    plt.legend()
    plt.tight_layout()

    out_png = OUT_PLOTS_DIR / f"waterfall_basin{basin_id:02d}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] energy waterfall: basin {basin_id:02d} -> {out_png}")


def load_energy_contrast_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Energy contrast CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"basin_id", "term"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} does not look like the expected LONG format. "
            f"Need columns at least {sorted(required)}; got {df.columns.tolist()}"
        )

    # force basin_id int
    df["basin_id"] = df["basin_id"].astype(int)
    return df


def detect_basins(df_energy: pd.DataFrame) -> List[int]:
    basins_energy = sorted(df_energy["basin_id"].dropna().astype(int).unique().tolist())

    basins_pdb = []
    if EXPORTED_PDB_DIR.exists():
        for p in EXPORTED_PDB_DIR.glob("basin*_WT.pdb"):
            m = re.match(r"basin(\d+)_WT\.pdb$", p.name)
            if m:
                basins_pdb.append(int(m.group(1)))
    basins_pdb = sorted(set(basins_pdb))

    # intersection if both available, else whichever exists
    if basins_energy and basins_pdb:
        return sorted(set(basins_energy).intersection(basins_pdb))
    return basins_energy or basins_pdb


def main() -> None:
    print("=== KRAS addon plots (per-residue + waterfall) ===")
    print(f"[INFO] ANALYSIS_DIR: {ANALYSIS_DIR}")
    print(f"[INFO] ADDONS_DIR:   {ADDONS_DIR}")
    print(f"[INFO] PDB_DIR:      {EXPORTED_PDB_DIR}")
    print(f"[INFO] ENERGY_CSV:   {ENERGY_CONTRAST_CSV}")

    df_energy = load_energy_contrast_csv(ENERGY_CONTRAST_CSV)

    basins = BASINS_TO_PLOT if BASINS_TO_PLOT is not None else detect_basins(df_energy)
    if not basins:
        raise RuntimeError("No basins detected to plot.")

    print(f"[INFO] basins to plot: {basins}")

    # 1) per-residue displacement
    for bid in basins:
        plot_per_residue_displacement_for_basin(bid)

    # 2) energy waterfall
    for bid in basins:
        plot_energy_waterfall_for_basin(bid, df_energy)

    print("=== Done. ===")


if __name__ == "__main__":
    main()
