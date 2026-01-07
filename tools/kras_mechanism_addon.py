# --*-- conding:utf-8 --*--
# @time:1/6/26 23:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_mechanism_addon.py

# Add-on only (NO re-run of embedding/basin pipeline):
# 1) Per-residue displacement curves (from exported CA-only PDBs)
# 2) Energy-term waterfall plots (from basin_energy_contrast.csv)
#
# Works with BOTH directory conventions:
#   analysis_closed_loop/addon/...
#   analysis_closed_loop/addons/...
#
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# User-tunable parameters
# -----------------------------
ANALYSIS_REL = Path("KRAS_sampling_results") / "analysis_closed_loop"

# If you want to force basins, set list like [1,2,5,6]; otherwise auto-detect from exported_pdb filenames
FORCED_BASINS: Optional[List[int]] = None

FIG_DPI = 300
SAVE_PDF = True

# Energy terms expected in your contrast csv (it has delta_E_xxx columns)
ENERGY_TERMS = [
    "E_steric", "E_geom", "E_bond", "E_mj", "E_dihedral",
    "E_hydroph", "E_cbeta", "E_rama", "E_total"
]


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    # tools/ -> project root
    return Path(__file__).resolve().parent.parent


def resolve_analysis_dir() -> Path:
    root = project_root_from_tools_dir()
    d = root / ANALYSIS_REL
    if not d.exists():
        raise FileNotFoundError(f"Cannot find analysis dir: {d}")
    return d


def resolve_addon_dir(analysis_dir: Path) -> Path:
    """
    Prefer existing directory among:
      analysis_closed_loop/addon
      analysis_closed_loop/addons
    If both exist, prefer 'addon' because your screenshot shows that.
    """
    addon = analysis_dir / "addon"
    addons = analysis_dir / "addons"
    if addon.exists():
        return addon
    if addons.exists():
        return addons
    # create canonical 'addon'
    addon.mkdir(parents=True, exist_ok=True)
    return addon


def resolve_exported_pdb_dir(analysis_dir: Path) -> Path:
    """
    exported_pdb could be under addon/ or addons/
    """
    candidates = [
        analysis_dir / "addon" / "exported_pdb",
        analysis_dir / "addons" / "exported_pdb",
        analysis_dir / "addon" / "exported_pdb",  # keep
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find exported_pdb directory. Expected one of:\n"
        f"  {analysis_dir/'addon'/'exported_pdb'}\n"
        f"  {analysis_dir/'addons'/'exported_pdb'}"
    )


def resolve_energy_contrast_csv(analysis_dir: Path) -> Path:
    """
    Your declared truth path is:
      KRAS_sampling_results/analysis_closed_loop/addons/basin_energy_contrast.csv

    But screenshot suggests 'addon/' exists.
    So: prefer addons/basin_energy_contrast.csv if it exists, else fallback addon/...
    """
    candidates = [
        analysis_dir / "addons" / "basin_energy_contrast.csv",
        analysis_dir / "addon" / "basin_energy_contrast.csv",
        analysis_dir / "addons" / "basin_delta_summary.csv",  # legacy fallback
        analysis_dir / "addon" / "basin_delta_summary.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find basin_energy_contrast.csv.\n"
        "Tried:\n  - analysis_closed_loop/addons/basin_energy_contrast.csv\n"
        "  - analysis_closed_loop/addon/basin_energy_contrast.csv"
    )


# -----------------------------
# PDB parsing (CA-only)
# -----------------------------
def parse_ca_pdb(pdb_path: Path) -> pd.DataFrame:
    rows = []
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            try:
                resseq_i = int(resseq)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            rows.append((chain, resseq_i, icode, resname, x, y, z))

    if not rows:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    df = pd.DataFrame(rows, columns=["chain", "resseq", "icode", "resname", "x", "y", "z"])
    return df.reset_index(drop=True)


# -----------------------------
# Geometry: Kabsch alignment
# -----------------------------
def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    C = P0.T @ Q0
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Qc - Pc @ R
    return R, t


def align_and_displacement(wt_df: pd.DataFrame, mut_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["chain", "resseq", "icode"]
    wt = wt_df.copy()
    mut = mut_df.copy()
    wt["key"] = wt[key_cols].astype(str).agg("|".join, axis=1)
    mut["key"] = mut[key_cols].astype(str).agg("|".join, axis=1)

    merged = wt.merge(mut, on="key", suffixes=("_wt", "_mut"))
    if len(merged) < 3:
        raise ValueError("Too few shared CA atoms to align (need >= 3).")

    P = merged[["x_mut", "y_mut", "z_mut"]].to_numpy(dtype=float)
    Q = merged[["x_wt", "y_wt", "z_wt"]].to_numpy(dtype=float)

    R, t = kabsch_align(P, Q)
    P_aligned = P @ R + t

    dxyz = P_aligned - Q
    disp = np.linalg.norm(dxyz, axis=1)

    out = pd.DataFrame({
        "chain": merged["chain_wt"].values,
        "resseq": merged["resseq_wt"].values.astype(int),
        "icode": merged["icode_wt"].values,
        "resname": merged["resname_wt"].values,
        "dx": dxyz[:, 0],
        "dy": dxyz[:, 1],
        "dz": dxyz[:, 2],
        "disp": disp,
    })

    # preserve WT file order
    wt_order = {k: i for i, k in enumerate(wt["key"].tolist())}
    out["order"] = [wt_order.get(f"{c}|{r}|{i}", 10**9) for c, r, i in zip(out["chain"], out["resseq"], out["icode"])]
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return out


# -----------------------------
# Basin discovery from exported_pdb
# -----------------------------
def discover_basins(exported_dir: Path) -> List[int]:
    """
    Detect basins from filenames like:
      basin01_WT.pdb, basin02_G12D.pdb, basin06_G12C.pdb
    """
    if FORCED_BASINS is not None:
        return sorted(set(int(x) for x in FORCED_BASINS))

    pat = re.compile(r"^basin(\d+)_")
    basins = set()
    for p in exported_dir.glob("basin*_WT.pdb"):
        m = pat.match(p.stem)
        if m:
            basins.add(int(m.group(1)))
    if not basins:
        raise FileNotFoundError(f"No basin*_WT.pdb found in {exported_dir}")
    return sorted(basins)


def basin_pdb_paths(exported_dir: Path, basin_id: int) -> Tuple[Path, Path, Path]:
    """
    Return (WT, G12C, G12D) paths for basin
    """
    wt = exported_dir / f"basin{basin_id:02d}_WT.pdb"
    g12c = exported_dir / f"basin{basin_id:02d}_G12C.pdb"
    g12d = exported_dir / f"basin{basin_id:02d}_G12D.pdb"
    return wt, g12c, g12d


# -----------------------------
# Plot helpers
# -----------------------------
def plot_displacement_curve(per_res: pd.DataFrame, title: str, out_png: Path):
    x = np.arange(len(per_res)) + 1
    y = per_res["disp"].to_numpy(dtype=float)

    plt.figure(figsize=(9.2, 3.8))
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("Residue index (fragment order)")
    plt.ylabel("CA displacement (Å)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI)
    if SAVE_PDF:
        plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


def plot_energy_waterfall_for_basin(df: pd.DataFrame, basin_id: int, out_png: Path):
    row = df[df["basin_id"] == basin_id]
    if row.empty:
        print(f"[WARN] basin {basin_id:02d}: not found in {out_png.name} input table. Skip.")
        return
    row = row.iloc[0]

    # detect available terms by checking columns
    terms = []
    for t in ENERGY_TERMS:
        c1 = f"delta_{t}_G12C_minus_WT"
        c2 = f"delta_{t}_G12D_minus_WT"
        if c1 in df.columns and c2 in df.columns:
            terms.append(t)

    if not terms:
        print("[WARN] No delta energy-term columns found. Skip waterfall.")
        return

    x = np.arange(len(terms))
    g12c = np.array([float(row[f"delta_{t}_G12C_minus_WT"]) for t in terms], dtype=float)
    g12d = np.array([float(row[f"delta_{t}_G12D_minus_WT"]) for t in terms], dtype=float)

    plt.figure(figsize=(9.2, 4.6))
    width = 0.38
    plt.bar(x - width/2, g12c, width=width, label="G12C − WT")
    plt.bar(x + width/2, g12d, width=width, label="G12D − WT")
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, terms, rotation=35, ha="right")
    plt.ylabel("Δ Energy term")
    plt.title(f"Energy-term contrast (basin {basin_id:02d})")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(out_png, dpi=FIG_DPI)
    if SAVE_PDF:
        plt.savefig(out_png.with_suffix(".pdf"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    analysis_dir = resolve_analysis_dir()

    exported_dir = resolve_exported_pdb_dir(analysis_dir)
    energy_csv = resolve_energy_contrast_csv(analysis_dir)

    # Where to write results: put them under the existing addon/addons folder
    addon_dir = resolve_addon_dir(analysis_dir)
    out_disp = addon_dir / "displacement"
    out_disp.mkdir(parents=True, exist_ok=True)
    out_water = addon_dir / "energy_waterfall"
    out_water.mkdir(parents=True, exist_ok=True)

    basins = discover_basins(exported_dir)
    print(f"[INFO] exported_pdb={exported_dir}")
    print(f"[INFO] energy_csv={energy_csv}")
    print(f"[INFO] basins={basins}")

    # --- Displacement curves
    all_rows = []
    for bid in basins:
        wt_p, c_p, d_p = basin_pdb_paths(exported_dir, bid)
        if not wt_p.exists():
            print(f"[WARN] Missing {wt_p}. Skip basin {bid:02d}.")
            continue

        wt_df = parse_ca_pdb(wt_p)

        # G12C
        if c_p.exists():
            c_df = parse_ca_pdb(c_p)
            disp_c = align_and_displacement(wt_df, c_df)
            disp_c["pair"] = "G12C-vs-WT"
            disp_c["basin_id"] = bid
            all_rows.append(disp_c)

            plot_displacement_curve(
                disp_c,
                title=f"Per-residue CA displacement (basin {bid:02d}) | G12C vs WT",
                out_png=out_disp / f"disp_curve_basin{bid:02d}_G12C_vs_WT.png",
            )
        else:
            print(f"[WARN] Missing {c_p} (G12C).")

        # G12D
        if d_p.exists():
            d_df = parse_ca_pdb(d_p)
            disp_d = align_and_displacement(wt_df, d_df)
            disp_d["pair"] = "G12D-vs-WT"
            disp_d["basin_id"] = bid
            all_rows.append(disp_d)

            plot_displacement_curve(
                disp_d,
                title=f"Per-residue CA displacement (basin {bid:02d}) | G12D vs WT",
                out_png=out_disp / f"disp_curve_basin{bid:02d}_G12D_vs_WT.png",
            )
        else:
            print(f"[WARN] Missing {d_p} (G12D).")

    if all_rows:
        out_csv = out_disp / "per_residue_displacement.csv"
        pd.concat(all_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"[OK] per_residue_displacement.csv -> {out_csv}")
    else:
        print("[WARN] No displacement results generated.")

    # --- Energy waterfall
    df = pd.read_csv(energy_csv)
    if "basin_id" not in df.columns:
        raise ValueError(f"{energy_csv} missing required column: basin_id")
    df["basin_id"] = df["basin_id"].astype(int)

    for bid in basins:
        plot_energy_waterfall_for_basin(
            df, bid, out_png=out_water / f"waterfall_basin{bid:02d}.png"
        )
    print(f"[OK] energy waterfalls -> {out_water}")

    print("[DONE]")


if __name__ == "__main__":
    main()



