# --*-- conding:utf-8 --*--
# @time:1/12/26 03:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_E_representatives.py

# plot_E_structures.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
FIG_SUBDIR = "E_reps_structures"   # outputs to KRAS_analysis/figs/E_reps_structures/
REPS_DIR_NAME = "reps_pdbs"        # located under KRAS_analysis/data_used/reps_pdbs

# Which basins to include (None = auto-detect from filenames)
BASINS: Optional[List[int]] = None

# Plot controls
HEATMAP_FIGSIZE = (9.2, 7.6)
BAR_FIGSIZE = (7.6, 4.6)
LINE_FIGSIZE = (7.6, 4.6)

# If True, use residue sequence order (CA atoms) for matching.
# If False, attempt to match by residue sequence number (resseq). (Order is safer for fragments.)
MATCH_BY_ORDER = True


# -----------------------------
# Path helpers
# -----------------------------
def kras_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# PDB parsing (CA-only)
# -----------------------------
def parse_pdb_ca_coords(pdb_path: Path) -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    """
    Minimal PDB parser: extract CA atoms.
    Returns:
      coords: (N,3) float
      meta:   list of (chain, resseq, resname) for each CA in the same order
    """
    coords: List[List[float]] = []
    meta: List[Tuple[str, int, str]] = []

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "?"
            try:
                resseq = int(line[22:26])
            except Exception:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue

            coords.append([x, y, z])
            meta.append((chain, resseq, resname))

    if not coords:
        raise ValueError(f"No CA atoms found in: {pdb_path}")

    return np.asarray(coords, dtype=float), meta


# -----------------------------
# Kabsch alignment + RMSD
# -----------------------------
def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align P onto Q using Kabsch.
    P, Q: (N,3)
    Returns:
      P_aligned: (N,3)
      R: rotation matrix (3,3)
    """
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must have shape (N,3) and be the same shape.")

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Proper rotation (avoid reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    P_aligned = Pc @ R + Q.mean(axis=0, keepdims=True)
    return P_aligned, R


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    d = P - Q
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def align_and_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]:
    P_aligned, _ = kabsch_align(P, Q)
    return rmsd(P_aligned, Q), P_aligned


# -----------------------------
# Filename parsing
# -----------------------------
_RE = re.compile(r"^(WT|G12C|G12D)_basin(\d+)\.pdb$", re.IGNORECASE)


def parse_rep_name(p: Path) -> Tuple[str, int]:
    m = _RE.match(p.name)
    if not m:
        raise ValueError(f"Unexpected rep pdb filename: {p.name} (expected WT_basinX.pdb etc.)")
    label = m.group(1).upper()
    basin = int(m.group(2))
    return label, basin


# -----------------------------
# Matching CA lists between structures
# -----------------------------
def match_coords(
    A: np.ndarray, metaA: List[Tuple[str, int, str]],
    B: np.ndarray, metaB: List[Tuple[str, int, str]],
    by_order: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, str]]]:
    """
    Return matched CA coords (A_m, B_m) with the same length.
    Default: match by CA order (safest for equal-length fragments).
    If by_order=False: match by (chain, resseq) intersection.
    """
    if by_order:
        n = min(len(A), len(B))
        return A[:n].copy(), B[:n].copy(), metaA[:n]

    # match by (chain, resseq)
    idxA = {(c, r): i for i, (c, r, rn) in enumerate(metaA)}
    idxB = {(c, r): i for i, (c, r, rn) in enumerate(metaB)}
    keys = sorted(set(idxA.keys()) & set(idxB.keys()), key=lambda x: (x[0], x[1]))
    if not keys:
        raise ValueError("No overlapping (chain, resseq) keys between structures.")
    A_m = np.stack([A[idxA[k]] for k in keys], axis=0)
    B_m = np.stack([B[idxB[k]] for k in keys], axis=0)
    meta_m = [(k[0], k[1], metaA[idxA[k]][2]) for k in keys]
    return A_m, B_m, meta_m


# -----------------------------
# Plotting
# -----------------------------
def plot_rmsd_heatmap(labels: List[str], M: np.ndarray, out_png: Path, out_pdf: Path):
    plt.figure(figsize=HEATMAP_FIGSIZE)
    ax = plt.gca()
    im = ax.imshow(M, aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, label="CA RMSD (Å)")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title("Representative structures: pairwise CA RMSD")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_basin_rmsd_bars(
    basins: List[int],
    rmsd_g12c_wt: List[float],
    rmsd_g12d_wt: List[float],
    rmsd_g12c_g12d: List[float],
    out_png: Path,
    out_pdf: Path,
):
    x = np.arange(len(basins), dtype=float)
    width = 0.22
    offs = [-width, 0.0, width]

    plt.figure(figsize=BAR_FIGSIZE)
    ax = plt.gca()

    ax.bar(x + offs[0], rmsd_g12c_wt, width=width, label="G12C vs WT")
    ax.bar(x + offs[1], rmsd_g12d_wt, width=width, label="G12D vs WT")
    ax.bar(x + offs[2], rmsd_g12c_g12d, width=width, label="G12C vs G12D")

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in basins])
    ax.set_xlabel("Basin ID")
    ax.set_ylabel("CA RMSD (Å)")
    ax.set_title("Per-basin representative structural differences")
    ax.legend(frameon=False, ncol=3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_per_residue_displacement(
    residue_index: np.ndarray,
    disp_g12c: np.ndarray,
    disp_g12d: np.ndarray,
    basin: int,
    out_png: Path,
    out_pdf: Path,
):
    plt.figure(figsize=LINE_FIGSIZE)
    ax = plt.gca()

    ax.plot(residue_index, disp_g12c, label="G12C − WT (aligned)")
    ax.plot(residue_index, disp_g12d, label="G12D − WT (aligned)")

    ax.set_xlabel("Residue index (CA order)")
    ax.set_ylabel("Displacement (Å)")
    ax.set_title(f"Basin {basin}: per-residue CA displacement")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    root = kras_root_from_script()
    data_used = root / "data_used"
    reps_dir = data_used / REPS_DIR_NAME
    if not reps_dir.exists():
        raise FileNotFoundError(f"Cannot find reps directory: {reps_dir}")

    out_dir = ensure_dir(root / "figs" / FIG_SUBDIR)

    # Collect PDBs
    pdbs = sorted(reps_dir.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"No .pdb files found under: {reps_dir}")

    reps: Dict[Tuple[str, int], Path] = {}
    for p in pdbs:
        lab, basin = parse_rep_name(p)
        reps[(lab, basin)] = p

    basins_all = sorted({b for (_, b) in reps.keys()})
    basins = basins_all if BASINS is None else [b for b in BASINS if b in basins_all]
    if not basins:
        raise ValueError("No basins available after filtering.")

    # Load coords
    coords: Dict[Tuple[str, int], np.ndarray] = {}
    metas: Dict[Tuple[str, int], List[Tuple[str, int, str]]] = {}

    for lab in ["WT", "G12C", "G12D"]:
        for b in basins:
            key = (lab, b)
            if key not in reps:
                continue
            c, m = parse_pdb_ca_coords(reps[key])
            coords[key] = c
            metas[key] = m

    # Build label list for RMSD matrix (existing keys only)
    keys_sorted = sorted(coords.keys(), key=lambda x: (x[1], x[0]))  # by basin then label
    label_names = [f"{k[0]}_basin{k[1]}" for k in keys_sorted]

    # Pairwise RMSD matrix
    n = len(keys_sorted)
    M = np.zeros((n, n), dtype=float)

    for i, ki in enumerate(keys_sorted):
        Ai, mi = coords[ki], metas[ki]
        for j, kj in enumerate(keys_sorted):
            Aj, mj = coords[kj], metas[kj]
            A_m, B_m, _ = match_coords(Ai, mi, Aj, mj, by_order=MATCH_BY_ORDER)
            r, _ = align_and_rmsd(A_m, B_m)
            M[i, j] = r

    heat_png = out_dir / "E1_pairwise_rmsd_heatmap.png"
    heat_pdf = out_dir / "E1_pairwise_rmsd_heatmap.pdf"
    plot_rmsd_heatmap(label_names, M, heat_png, heat_pdf)

    # Per-basin RMSD summary
    rmsd_g12c_wt: List[float] = []
    rmsd_g12d_wt: List[float] = []
    rmsd_g12c_g12d: List[float] = []

    per_residue_rows = []

    for b in basins:
        k_wt = ("WT", b)
        k_c = ("G12C", b)
        k_d = ("G12D", b)

        if k_wt not in coords:
            continue

        # WT as reference
        W, mW = coords[k_wt], metas[k_wt]

        # G12C vs WT
        if k_c in coords:
            C, mC = coords[k_c], metas[k_c]
            Wm, Cm, meta = match_coords(W, mW, C, mC, by_order=MATCH_BY_ORDER)
            r_c, C_aligned = align_and_rmsd(Cm, Wm)
            rmsd_g12c_wt.append(r_c)
            disp_c = np.linalg.norm(C_aligned - Wm, axis=1)
        else:
            rmsd_g12c_wt.append(float("nan"))
            disp_c = None

        # G12D vs WT
        if k_d in coords:
            D, mD = coords[k_d], metas[k_d]
            Wm2, Dm, meta2 = match_coords(W, mW, D, mD, by_order=MATCH_BY_ORDER)
            r_d, D_aligned = align_and_rmsd(Dm, Wm2)
            rmsd_g12d_wt.append(r_d)
            disp_d = np.linalg.norm(D_aligned - Wm2, axis=1)
        else:
            rmsd_g12d_wt.append(float("nan"))
            disp_d = None

        # G12C vs G12D
        if (k_c in coords) and (k_d in coords):
            C, mC = coords[k_c], metas[k_c]
            D, mD = coords[k_d], metas[k_d]
            Cm2, Dm2, _ = match_coords(C, mC, D, mD, by_order=MATCH_BY_ORDER)
            r_cd, _ = align_and_rmsd(Cm2, Dm2)
            rmsd_g12c_g12d.append(r_cd)
        else:
            rmsd_g12c_g12d.append(float("nan"))

        # Per-residue displacement plot (only if both mutants present)
        if disp_c is not None and disp_d is not None:
            idx = np.arange(1, len(disp_c) + 1, dtype=int)

            out_png = out_dir / f"E3_basin{b}_per_residue_displacement.png"
            out_pdf = out_dir / f"E3_basin{b}_per_residue_displacement.pdf"
            plot_per_residue_displacement(idx, disp_c, disp_d, b, out_png, out_pdf)

            for i, (dc, dd) in enumerate(zip(disp_c, disp_d), start=1):
                per_residue_rows.append(
                    {"basin_id": b, "res_index": i, "disp_G12C_minus_WT_A": float(dc), "disp_G12D_minus_WT_A": float(dd)}
                )

    bar_png = out_dir / "E2_per_basin_rmsd_bars.png"
    bar_pdf = out_dir / "E2_per_basin_rmsd_bars.pdf"
    plot_basin_rmsd_bars(basins, rmsd_g12c_wt, rmsd_g12d_wt, rmsd_g12c_g12d, bar_png, bar_pdf)

    # Save a CSV for downstream table/fig tweaks
    summary_rows = []
    for i, b in enumerate(basins):
        summary_rows.append(
            {
                "basin_id": b,
                "rmsd_G12C_vs_WT_A": rmsd_g12c_wt[i] if i < len(rmsd_g12c_wt) else np.nan,
                "rmsd_G12D_vs_WT_A": rmsd_g12d_wt[i] if i < len(rmsd_g12d_wt) else np.nan,
                "rmsd_G12C_vs_G12D_A": rmsd_g12c_g12d[i] if i < len(rmsd_g12c_g12d) else np.nan,
            }
        )
    df_sum = pd.DataFrame(summary_rows)
    sum_csv = out_dir / "E_summary_rmsd_by_basin.csv"
    df_sum.to_csv(sum_csv, index=False)

    if per_residue_rows:
        df_pr = pd.DataFrame(per_residue_rows)
        pr_csv = out_dir / "E_per_residue_displacement.csv"
        df_pr.to_csv(pr_csv, index=False)
    else:
        pr_csv = None

    # Manifest
    manifest = out_dir / "E_manifest.json"
    payload = {
        "inputs": {
            "reps_dir": str(reps_dir),
            "rep_files": {f"{k[0]}_basin{k[1]}": str(v) for k, v in sorted(reps.items())},
        },
        "params": {
            "FIG_SUBDIR": FIG_SUBDIR,
            "MATCH_BY_ORDER": MATCH_BY_ORDER,
            "basins": basins,
        },
        "outputs": [
            str(heat_png), str(heat_pdf),
            str(bar_png), str(bar_pdf),
            str(sum_csv),
        ] + ([] if pr_csv is None else [str(pr_csv)]),
        "notes": {
            "E1": "Pairwise CA RMSD heatmap across all representative structures (Kabsch-aligned).",
            "E2": "Per-basin RMSD bars: G12C vs WT, G12D vs WT, and G12C vs G12D.",
            "E3": "Per-residue CA displacement curves (mutant aligned to WT), one plot per basin when both mutants exist.",
        },
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in payload["outputs"]:
        print("  -", Path(p).name)
    print("  -", manifest.name)


if __name__ == "__main__":
    main()