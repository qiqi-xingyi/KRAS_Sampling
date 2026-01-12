# --*-- conding:utf-8 --*--
# @time:1/11/26 21:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_analysis.py

#!/usr/bin/env python3
"""
Unified plotting script for your KRAS sampling analysis.

What it does (auto-detect / best-effort):
1) Sampling distribution maps (WT / mutants) + diff maps (mut - WT)
2) Basin mass contrast lollipop plot (Top |Δmass|)
3) Per-residue displacement line plot (with optional percentile band)
4) Energy term contrast waterfall plot (mut - WT)

It will search for data files under ANALYSIS_DIR/addons by default and save figures to FIG_DIR.

How to run:
  python plot_all.py
or run in your IDE.

Notes:
- This script is designed to be robust to slightly different column names.
- If a required file is missing or columns can’t be inferred, that plot is skipped with a warning.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# =========================
# User parameters (edit here)
# =========================

# If you want to hardcode your project root, set it here; otherwise it uses the script location.
PROJECT_ROOT_OVERRIDE: Optional[Path] = None

# Your analysis folder (relative to project root)
ANALYSIS_REL = Path("KRAS_sampling_results") / "analysis_closed_loop"
ADDONS_REL = ANALYSIS_REL / "addons"

# Output figures folder (relative to analysis)
FIG_REL = ANALYSIS_REL / "figs_redraw"

# Conditions you care about (used for ordering and pairing vs WT)
WT_KEYS = ["WT", "wildtype", "wild_type"]
MUTANT_ORDER = ["G12C", "G12D"]  # add more if needed, e.g. ["G12C", "G12D", "G13D"]

# Save formats
SAVE_PNG = True
SAVE_PDF = True
DPI = 300

# Top-K basins to show
TOPK_BASINS = 10

# Distribution plot settings
DIST_CMAP = "viridis"
DIFF_CMAP = "coolwarm"
DIST_NBINS_DEFAULT = 160  # for point-cloud hist2d fallback

# Global style (keep consistent across all figs)
plt.rcParams.update({
    "figure.figsize": (10.5, 7.0),
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
})


# =========================
# Helpers
# =========================

def _project_root() -> Path:
    """
    Detect repo root even if this script is inside tools/.
    Priority:
      1) PROJECT_ROOT_OVERRIDE if set
      2) parent of this script if we are in /tools
      3) this script directory otherwise
    """
    if PROJECT_ROOT_OVERRIDE is not None:
        return PROJECT_ROOT_OVERRIDE.resolve()

    here = Path(__file__).resolve()

    # If script is in .../KRAS_QSAD/tools/plot_all.py -> root is .../KRAS_QSAD
    if here.parent.name.lower() == "tools":
        return here.parent.parent

    # If script is deeper under tools/, still try to locate tools in ancestors
    for p in here.parents:
        if p.name.lower() == "tools":
            return p.parent

    # Fallback: script folder
    return here.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, outpath_no_ext: Path) -> None:
    if SAVE_PNG:
        fig.savefig(outpath_no_ext.with_suffix(".png"), bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(outpath_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _lower(s: str) -> str:
    return str(s).strip().lower()


def _is_wt(name: str) -> bool:
    ln = _lower(name)
    return any(k in ln for k in WT_KEYS)


def _guess_condition_from_filename(fp: Path) -> str:
    name = fp.stem
    # Common patterns: *_WT*, *_G12C*, etc.
    for key in ["WT"] + MUTANT_ORDER:
        if re.search(rf"(?:^|[^A-Za-z0-9]){re.escape(key)}(?:$|[^A-Za-z0-9])", name, flags=re.I):
            return key.upper()
    # Fallback: try detect GxxX patterns
    m = re.search(r"(G\d{2}[A-Z])", name, flags=re.I)
    if m:
        return m.group(1).upper()
    return name.upper()


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # Fuzzy contains
    for cand in candidates:
        for lc, orig in cols.items():
            if cand.lower() in lc:
                return orig
    return None


def _as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_grid(Z: np.ndarray) -> np.ndarray:
    z = np.array(Z, dtype=float)
    total = np.nansum(z)
    if total > 0:
        z = z / total
    return z


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # Jensen-Shannon divergence (base e)
    p = np.array(p, dtype=float).ravel()
    q = np.array(q, dtype=float).ravel()
    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.array(p, dtype=float).ravel()
    q = np.array(q, dtype=float).ravel()
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return 0.5 * float(np.sum(np.abs(p - q)))


# =========================
# 1) Distribution: auto-detect formats
# =========================

@dataclass
class DistGrid:
    x_edges: np.ndarray  # len = nx+1
    y_edges: np.ndarray  # len = ny+1
    Z: np.ndarray        # shape = (ny, nx), matches pcolormesh convention


def _load_distribution_files(addons_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Searches for CSV files likely containing 2D distributions.

    Supported styles:
    A) single file with columns: condition, x, y, w (point cloud)
    B) per-condition file(s): inferred by filename, with columns x,y,(w)
    C) grid file with columns: x_bin, y_bin, value or x_center,y_center,value
    """
    cands = []
    for pat in [
        "*dist*.csv",
        "*density*.csv",
        "*distribution*.csv",
        "*embedding*.csv",
        "*umap*.csv",
        "*tsne*.csv",
        "*pca2*.csv",
        "*pca_2*.csv",
        "*proj2*.csv",
    ]:
        cands.extend(addons_dir.glob(pat))

    # De-dup and keep existing
    files = sorted({p.resolve() for p in cands if p.is_file()})

    if not files:
        return {}

    # Prefer "one file with condition column" if present
    for fp in files:
        try:
            df = _read_csv(fp)
        except Exception:
            continue
        cond_col = _pick_col(df, ["condition", "group", "label", "case", "variant"])
        x_col = _pick_col(df, ["x", "x2", "pc1", "dim1", "u", "umap1", "tsne1"])
        y_col = _pick_col(df, ["y", "y2", "pc2", "dim2", "v", "umap2", "tsne2"])
        if cond_col and x_col and y_col:
            # Split into condition dict
            out: Dict[str, pd.DataFrame] = {}
            for cond, sub in df.groupby(cond_col):
                out[str(cond)] = sub.copy()
            _info(f"Found combined distribution file: {fp.name} (split into {len(out)} conditions)")
            return out

    # Otherwise treat each file as one condition
    out: Dict[str, pd.DataFrame] = {}
    for fp in files:
        try:
            df = _read_csv(fp)
        except Exception:
            continue
        x_col = _pick_col(df, ["x", "x2", "pc1", "dim1", "u", "umap1", "tsne1", "x_center", "xbin", "x_bin"])
        y_col = _pick_col(df, ["y", "y2", "pc2", "dim2", "v", "umap2", "tsne2", "y_center", "ybin", "y_bin"])
        if not (x_col and y_col):
            continue
        cond = _guess_condition_from_filename(fp)
        out[cond] = df.copy()
    if out:
        _info(f"Found per-file distributions: {len(out)} files")
    return out


def _dist_df_to_grid(df: pd.DataFrame, nbins: int = DIST_NBINS_DEFAULT) -> Optional[DistGrid]:
    """
    Convert a distribution dataframe to a regular grid.
    Supports:
      - point cloud: x,y,(w)
      - binned grid: x_bin,y_bin,value or x_center,y_center,value
    """
    x_col = _pick_col(df, ["x", "pc1", "dim1", "umap1", "tsne1", "x_center", "xbin", "x_bin"])
    y_col = _pick_col(df, ["y", "pc2", "dim2", "umap2", "tsne2", "y_center", "ybin", "y_bin"])
    w_col = _pick_col(df, ["w", "weight", "prob", "p", "count", "density", "value", "z"])

    if not (x_col and y_col):
        return None

    x = _as_numeric(df[x_col]).to_numpy()
    y = _as_numeric(df[y_col]).to_numpy()

    if w_col is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = _as_numeric(df[w_col]).fillna(0).to_numpy()

    # If x/y look like small-integer bins, try pivot grid
    # Heuristic: integer-like and limited unique values
    def _is_bin_like(arr: np.ndarray) -> bool:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return False
        frac = np.mean(np.abs(arr - np.round(arr)) < 1e-8)
        nunique = len(np.unique(arr))
        return (frac > 0.95) and (nunique < 2000)

    if _is_bin_like(x) and _is_bin_like(y):
        # pivot table on (y,x)
        tmp = pd.DataFrame({"x": x, "y": y, "w": w})
        piv = tmp.pivot_table(index="y", columns="x", values="w", aggfunc="sum", fill_value=0.0)
        x_vals = np.array(piv.columns, dtype=float)
        y_vals = np.array(piv.index, dtype=float)
        Z = piv.to_numpy()

        # Convert centers/bins to edges (simple: midpoints)
        def centers_to_edges(c: np.ndarray) -> np.ndarray:
            c = np.sort(c)
            if c.size == 1:
                return np.array([c[0] - 0.5, c[0] + 0.5])
            mids = 0.5 * (c[:-1] + c[1:])
            left = c[0] - (mids[0] - c[0])
            right = c[-1] + (c[-1] - mids[-1])
            return np.concatenate([[left], mids, [right]])

        x_edges = centers_to_edges(x_vals)
        y_edges = centers_to_edges(y_vals)
        return DistGrid(x_edges=x_edges, y_edges=y_edges, Z=_normalize_grid(Z))

    # Fallback: hist2d from point cloud
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[finite]
    y = y[finite]
    w = w[finite]
    if x.size == 0:
        return None

    H, x_edges, y_edges = np.histogram2d(x, y, bins=nbins, weights=w)
    # histogram2d returns shape (nx, ny); we want (ny, nx) for pcolormesh
    Z = H.T
    return DistGrid(x_edges=x_edges, y_edges=y_edges, Z=_normalize_grid(Z))


def plot_distributions_and_diffs(
    dist_map: Dict[str, pd.DataFrame],
    out_dir: Path,
    title_prefix: str = "Sampling distribution",
) -> None:
    if not dist_map:
        _warn("No distribution CSVs found. Skipping distribution plots.")
        return

    # Normalize condition names
    normalized: Dict[str, pd.DataFrame] = {}
    for k, v in dist_map.items():
        nk = k.upper()
        normalized[nk] = v
    dist_map = normalized

    # Identify WT
    wt_key = None
    for k in dist_map.keys():
        if _is_wt(k):
            wt_key = k
            break
    if wt_key is None:
        # fallback: choose first non-mutant
        for k in dist_map.keys():
            if "G" not in k:
                wt_key = k
                break
    if wt_key is None:
        wt_key = sorted(dist_map.keys())[0]
        _warn(f"WT not detected; using {wt_key} as baseline.")

    # Build grids for each condition
    grids: Dict[str, DistGrid] = {}
    for cond, df in dist_map.items():
        g = _dist_df_to_grid(df)
        if g is None:
            _warn(f"Could not convert distribution df to grid for {cond}; skipping.")
            continue
        grids[cond] = g

    if wt_key not in grids:
        _warn("WT baseline grid not available. Skipping distribution plots.")
        return

    # Make a common grid shape by rebinning onto WT grid (nearest / simple)
    wt = grids[wt_key]
    x_edges = wt.x_edges
    y_edges = wt.y_edges

    def rebin_to_wt(g: DistGrid) -> np.ndarray:
        # Re-histogram approximate: sample at cell centers from g, then histogram into WT bins.
        # This keeps script dependency-free and robust.
        gx = 0.5 * (g.x_edges[:-1] + g.x_edges[1:])
        gy = 0.5 * (g.y_edges[:-1] + g.y_edges[1:])
        Xc, Yc = np.meshgrid(gx, gy)
        W = g.Z
        # flatten as pseudo-point cloud
        x = Xc.ravel()
        y = Yc.ravel()
        w = W.ravel()
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=w)
        return _normalize_grid(H.T)

    Z_wt = rebin_to_wt(wt)

    # Order conditions
    ordered = [wt_key] + [m for m in MUTANT_ORDER if m in grids] + [k for k in sorted(grids) if k not in ([wt_key] + MUTANT_ORDER)]
    ordered = [k for k in ordered if k in grids]

    # 1) Distribution panels
    n = len(ordered)
    fig = plt.figure(figsize=(4.2 * n, 3.6))
    for i, cond in enumerate(ordered, start=1):
        ax = fig.add_subplot(1, n, i)
        Z = rebin_to_wt(grids[cond])
        im = ax.pcolormesh(x_edges, y_edges, Z, shading="auto", cmap=DIST_CMAP)
        ax.set_title(cond)
        ax.set_xlabel("Dim-1")
        if i == 1:
            ax.set_ylabel("Dim-2")
        ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.02)
    cbar.set_label("Probability mass")
    fig.suptitle(title_prefix, y=1.02)
    _save_fig(fig, out_dir / "fig1_distribution")

    # 2) Diff maps vs WT (only for mutants in MUTANT_ORDER that exist)
    diffs: List[Tuple[str, np.ndarray]] = []
    for m in MUTANT_ORDER:
        if m in grids:
            Zm = rebin_to_wt(grids[m])
            diffs.append((m, Zm - Z_wt))
    if not diffs:
        _warn("No mutant distributions available for diff maps. Skipping diff plot.")
        return

    vmax = max(float(np.max(np.abs(d))) for _, d in diffs)
    vmax = max(vmax, 1e-12)

    fig = plt.figure(figsize=(4.6 * len(diffs), 4.0))
    for i, (m, D) in enumerate(diffs, start=1):
        ax = fig.add_subplot(1, len(diffs), i)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.pcolormesh(x_edges, y_edges, D, shading="auto", cmap=DIFF_CMAP, norm=norm)
        jsd = _js_divergence(Z_wt, Z_wt + D)  # since D = Zm - Zwt
        tv = _total_variation(Z_wt, Z_wt + D)
        ax.set_title(f"{m} - {wt_key}\nJSD={jsd:.3f}  TV={tv:.3f}")
        ax.set_xlabel("Dim-1")
        if i == 1:
            ax.set_ylabel("Dim-2")
        ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=fig.axes, fraction=0.03, pad=0.02)
    cbar.set_label("Δ probability mass")
    fig.suptitle("Distribution difference maps", y=1.02)
    _save_fig(fig, out_dir / "fig2_distribution_diff")


# =========================
# 2) Basin lollipop (Δmass)
# =========================

def _find_basin_contrast_csv(addons_dir: Path) -> Optional[Path]:
    # Your earlier path suggests this exact file exists:
    # KRAS_sampling_results/analysis_closed_loop/addons/basin_energy_contrast.csv
    exact = addons_dir / "basin_energy_contrast.csv"
    if exact.exists():
        return exact
    # fallback search
    cands = list(addons_dir.glob("*basin*contrast*.csv")) + list(addons_dir.glob("*basin*energy*.csv"))
    return cands[0] if cands else None


def plot_basin_lollipop(path: Path, out_dir: Path) -> None:
    df = _read_csv(path)

    basin_col = _pick_col(df, ["basin", "basin_id", "cluster", "cluster_id", "state"])
    if basin_col is None:
        # fallback: first non-numeric-ish column
        for c in df.columns:
            if df[c].dtype == object:
                basin_col = c
                break
    if basin_col is None:
        _warn(f"Could not identify basin id column in {path.name}. Skipping basin plot.")
        return

    # Try to find Δmass column, else compute from mutant/wt mass columns
    delta_col = _pick_col(df, ["delta_mass", "d_mass", "mass_delta", "diff_mass", "delta_prob", "dprob"])
    wt_mass_col = _pick_col(df, ["wt_mass", "mass_wt", "prob_wt", "p_wt"])
    mut_mass_col = _pick_col(df, ["mut_mass", "mass_mut", "prob_mut", "p_mut", "variant_mass"])

    plot_rows: List[Tuple[str, float]] = []

    if delta_col is not None:
        tmp = df[[basin_col, delta_col]].copy()
        tmp[delta_col] = _as_numeric(tmp[delta_col]).fillna(0.0)
        for _, r in tmp.iterrows():
            plot_rows.append((str(r[basin_col]), float(r[delta_col])))
        title = "Basin probability mass change (mut - WT)"
    elif (wt_mass_col is not None) and (mut_mass_col is not None):
        tmp = df[[basin_col, wt_mass_col, mut_mass_col]].copy()
        tmp[wt_mass_col] = _as_numeric(tmp[wt_mass_col]).fillna(0.0)
        tmp[mut_mass_col] = _as_numeric(tmp[mut_mass_col]).fillna(0.0)
        for _, r in tmp.iterrows():
            plot_rows.append((str(r[basin_col]), float(r[mut_mass_col] - r[wt_mass_col])))
        title = "Basin probability mass change (computed mut - WT)"
    else:
        _warn(f"Could not infer Δmass columns in {path.name}. Expected delta_mass or (wt_mass, mut_mass). Skipping.")
        return

    if not plot_rows:
        _warn("No basin rows found. Skipping.")
        return

    # Sort by |Δmass| and take TopK
    plot_rows.sort(key=lambda t: abs(t[1]), reverse=True)
    plot_rows = plot_rows[:TOPK_BASINS]
    basins = [b for b, _ in plot_rows][::-1]
    dmass = [v for _, v in plot_rows][::-1]

    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(basins) + 2.0))
    y = np.arange(len(basins))

    ax.hlines(y, 0, dmass, linewidth=2.0)
    ax.scatter(dmass, y, s=60)

    ax.axvline(0, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(basins)
    ax.set_xlabel("Δ mass")
    ax.set_title(f"{title}  (Top {len(basins)} basins by |Δmass|)")
    fig.tight_layout()
    _save_fig(fig, out_dir / "fig3_basin_lollipop")


# =========================
# 3) Per-residue displacement
# =========================

def _find_displacement_csv(addons_dir: Path) -> Optional[Path]:
    # Common names you might have
    for name in [
        "per_residue_displacement.csv",
        "residue_displacement.csv",
        "displacement_per_residue.csv",
        "rmsf_like.csv",
        "backbone_displacement.csv",
    ]:
        p = addons_dir / name
        if p.exists():
            return p
    cands = list(addons_dir.glob("*displacement*residue*.csv")) + list(addons_dir.glob("*per_residue*csv"))
    return cands[0] if cands else None


def plot_per_residue_displacement(path: Path, out_dir: Path) -> None:
    df = _read_csv(path)

    res_col = _pick_col(df, ["resi", "resid", "residue", "residue_id", "res_index", "position", "index"])
    if res_col is None:
        _warn(f"Could not find residue index column in {path.name}. Skipping displacement plot.")
        return

    cond_col = _pick_col(df, ["condition", "group", "variant", "case", "label"])
    basin_col = _pick_col(df, ["basin", "basin_id", "cluster", "cluster_id", "state"])
    disp_col = _pick_col(df, ["displacement", "disp", "delta", "dr", "rmsf", "dist", "distance"])
    mean_col = _pick_col(df, ["mean", "avg", "mu"])
    q25_col = _pick_col(df, ["q25", "p25", "percentile_25"])
    q75_col = _pick_col(df, ["q75", "p75", "percentile_75"])
    std_col = _pick_col(df, ["std", "sigma"])

    df = df.copy()
    df[res_col] = _as_numeric(df[res_col])

    # Strategy:
    # - If mean/q25/q75 exist, use them directly.
    # - Else if disp exists with multiple samples, aggregate.
    if mean_col is None:
        if disp_col is None:
            _warn(f"No displacement/mean columns found in {path.name}. Skipping.")
            return
        df[disp_col] = _as_numeric(df[disp_col])
        grp_cols = [res_col]
        if cond_col: grp_cols.insert(0, cond_col)
        if basin_col: grp_cols.insert(0, basin_col)

        agg = df.groupby(grp_cols)[disp_col].agg(["mean", "quantile"]).reset_index(drop=False)
        # pandas quantile in agg is awkward; do custom percentiles:
        def pct(s, p):
            s = pd.to_numeric(s, errors="coerce").dropna()
            if len(s) == 0:
                return np.nan
            return float(np.percentile(s, p))
        g = df.groupby(grp_cols)[disp_col]
        agg = g.agg(
            mean="mean",
            q25=lambda s: pct(s, 25),
            q75=lambda s: pct(s, 75),
        ).reset_index()
        mean_col, q25_col, q75_col = "mean", "q25", "q75"
        df_plot = agg
    else:
        df_plot = df
        df_plot[mean_col] = _as_numeric(df_plot[mean_col])
        if q25_col: df_plot[q25_col] = _as_numeric(df_plot[q25_col])
        if q75_col: df_plot[q75_col] = _as_numeric(df_plot[q75_col])
        if std_col: df_plot[std_col] = _as_numeric(df_plot[std_col])

    # Choose a single basin if basin exists: pick the one with most rows
    chosen_basin = None
    if basin_col:
        counts = df_plot[basin_col].value_counts()
        if len(counts) > 0:
            chosen_basin = counts.index[0]
            df_plot = df_plot[df_plot[basin_col] == chosen_basin].copy()
            _info(f"Displacement plot: using basin={chosen_basin} (most data rows).")

    # Choose conditions: WT and mutants if possible
    if cond_col is None:
        # assume already Δ vs WT, or single curve
        curves = {"DISPLACEMENT": df_plot}
    else:
        curves = {}
        # normalize condition strings
        df_plot["_COND_"] = df_plot[cond_col].astype(str).str.upper()
        # pick WT baseline
        wt = None
        for c in df_plot["_COND_"].unique():
            if _is_wt(c):
                wt = c
                break
        if wt is None:
            wt = sorted(df_plot["_COND_"].unique())[0]
        # pick mutants in order
        wanted = [wt] + [m for m in MUTANT_ORDER if m in set(df_plot["_COND_"].unique())]
        for c in wanted:
            curves[c] = df_plot[df_plot["_COND_"] == c].copy()

    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    for label, sub in curves.items():
        sub = sub.sort_values(res_col)
        x = sub[res_col].to_numpy()
        y = sub[mean_col].to_numpy()
        ax.plot(x, y, linewidth=2.0, label=label)

        # band: q25-q75 if present, else mean ± std if present
        if q25_col and q75_col and (q25_col in sub.columns) and (q75_col in sub.columns):
            y1 = sub[q25_col].to_numpy()
            y2 = sub[q75_col].to_numpy()
            ax.fill_between(x, y1, y2, alpha=0.2)
        elif std_col and (std_col in sub.columns):
            s = sub[std_col].to_numpy()
            ax.fill_between(x, y - s, y + s, alpha=0.15)

    ax.set_xlabel("Residue index")
    ax.set_ylabel("Displacement (Å)")
    title = "Per-residue displacement"
    if chosen_basin is not None:
        title += f" (basin {chosen_basin})"
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save_fig(fig, out_dir / "fig4_per_residue_displacement")


# =========================
# 4) Energy term waterfall
# =========================

def _find_energy_terms_csv(addons_dir: Path) -> Optional[Path]:
    for name in [
        "energy_terms_contrast.csv",
        "energy_decomposition.csv",
        "energy_terms.csv",
        "term_energy_contrast.csv",
    ]:
        p = addons_dir / name
        if p.exists():
            return p
    cands = list(addons_dir.glob("*energy*term*csv")) + list(addons_dir.glob("*decomp*csv"))
    return cands[0] if cands else None


def _waterfall(ax, labels: List[str], deltas: List[float], title: str) -> None:
    deltas = np.array(deltas, dtype=float)
    order = np.argsort(np.abs(deltas))[::-1]
    labels = [labels[i] for i in order]
    deltas = deltas[order]

    cum = np.cumsum(deltas)
    starts = np.concatenate([[0.0], cum[:-1]])

    x = np.arange(len(labels) + 1)
    # bars for terms
    ax.bar(np.arange(len(labels)), deltas, bottom=starts, width=0.7)
    # final total bar
    ax.bar(len(labels), cum[-1] if len(cum) else 0.0, width=0.7)

    ax.axhline(0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["TOTAL"], rotation=30, ha="right")
    ax.set_ylabel("Δ Energy")
    ax.set_title(title)


def plot_energy_waterfall(path: Path, out_dir: Path) -> None:
    df = _read_csv(path)

    term_col = _pick_col(df, ["term", "energy_term", "component", "name"])
    if term_col is None:
        _warn(f"Could not find term column in {path.name}. Skipping energy plot.")
        return

    cond_col = _pick_col(df, ["condition", "variant", "group", "label", "case"])
    wt_col = _pick_col(df, ["wt", "e_wt", "energy_wt", "value_wt"])
    mut_col = _pick_col(df, ["mut", "e_mut", "energy_mut", "value_mut"])
    delta_col = _pick_col(df, ["delta", "dE", "delta_energy", "diff", "contrast"])

    df = df.copy()
    df[term_col] = df[term_col].astype(str)

    if delta_col is None:
        if wt_col is None or mut_col is None:
            _warn(f"Expected delta column OR (wt, mut) columns in {path.name}. Skipping.")
            return
        df[wt_col] = _as_numeric(df[wt_col]).fillna(0.0)
        df[mut_col] = _as_numeric(df[mut_col]).fillna(0.0)
        df["_DELTA_"] = df[mut_col] - df[wt_col]
        delta_col = "_DELTA_"
    else:
        df[delta_col] = _as_numeric(df[delta_col]).fillna(0.0)

    # If file contains multiple mutants, split by condition
    if cond_col is not None:
        df["_COND_"] = df[cond_col].astype(str).str.upper()
        mutants = [m for m in MUTANT_ORDER if m in set(df["_COND_"].unique())]
        if not mutants:
            # fallback: any non-WT
            mutants = [c for c in sorted(df["_COND_"].unique()) if not _is_wt(c)]
            mutants = mutants[:2]
        if not mutants:
            _warn("No mutant condition found in energy terms CSV. Plotting all rows as one waterfall.")
            mutants = ["ALL"]

        fig, axes = plt.subplots(1, len(mutants), figsize=(5.2 * len(mutants), 4.0), squeeze=False)
        for i, m in enumerate(mutants):
            ax = axes[0, i]
            sub = df if m == "ALL" else df[df["_COND_"] == m].copy()
            labels = sub[term_col].tolist()
            deltas = sub[delta_col].tolist()
            _waterfall(ax, labels, deltas, title=f"Energy term Δ (mut - WT): {m}")
        fig.tight_layout()
        _save_fig(fig, out_dir / "fig5_energy_waterfall")
    else:
        labels = df[term_col].tolist()
        deltas = df[delta_col].tolist()
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        _waterfall(ax, labels, deltas, title="Energy term Δ (mut - WT)")
        fig.tight_layout()
        _save_fig(fig, out_dir / "fig5_energy_waterfall")


# =========================
# Main
# =========================

def main() -> None:
    root = _project_root()
    analysis_dir = root / ANALYSIS_REL
    addons_dir = root / ADDONS_REL
    fig_dir = root / FIG_REL

    _info(f"PROJECT_ROOT: {root}")
    _info(f"ANALYSIS_DIR:  {analysis_dir}")
    _info(f"ADDONS_DIR:    {addons_dir}")
    _info(f"FIG_DIR:       {fig_dir}")

    if not addons_dir.exists():
        _warn(f"ADDONS_DIR does not exist: {addons_dir}")
        return

    _ensure_dir(fig_dir)

    # 1) Distributions
    dist_map = _load_distribution_files(addons_dir)
    plot_distributions_and_diffs(dist_map, fig_dir)

    # 2) Basin lollipop
    basin_csv = _find_basin_contrast_csv(addons_dir)
    if basin_csv is None:
        _warn("Could not find basin_energy_contrast.csv (or similar). Skipping basin plot.")
    else:
        _info(f"Using basin contrast file: {basin_csv.name}")
        plot_basin_lollipop(basin_csv, fig_dir)

    # 3) Per-residue displacement
    disp_csv = _find_displacement_csv(addons_dir)
    if disp_csv is None:
        _warn("Could not find per-residue displacement CSV. Skipping displacement plot.")
    else:
        _info(f"Using displacement file: {disp_csv.name}")
        plot_per_residue_displacement(disp_csv, fig_dir)

    # 4) Energy term waterfall
    energy_csv = _find_energy_terms_csv(addons_dir)
    if energy_csv is None:
        _warn("Could not find energy term CSV. Skipping energy plot.")
    else:
        _info(f"Using energy term file: {energy_csv.name}")
        plot_energy_waterfall(energy_csv, fig_dir)

    _info("Done. Figures saved to FIG_DIR.")


if __name__ == "__main__":
    main()
