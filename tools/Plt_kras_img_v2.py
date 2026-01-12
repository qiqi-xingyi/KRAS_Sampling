# --*-- conding:utf-8 --*--
# @time:1/12/26 00:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_kras_img_v2.py

# Output:
#   <project_root>/KRAS_analysis/figs/*.png + *.pdf   (600 dpi)
#
# Input (fixed):
#   <project_root>/KRAS_analysis/data_summary/merged/
#       points_enriched.csv
#       basin_master.csv
#       representatives_enriched.csv
#
# Notes:
# - This script is robust to column name variations.
# - It auto-detects:
#     system label column (WT/G12C/G12D),
#     basin id column (0..6),
#     embedding coords (z1/z2 or tsne1/tsne2 etc),
#     RMSD column,
#     weight/prob/count column.

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon

# ----------------------------
# Global style (journal-like)
# ----------------------------
def set_style():
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.color": "#d0d0d0",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "font.size": 12,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


# ----------------------------
# Path helpers
# ----------------------------
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Resolve project root robustly.
    We search upwards for a folder containing 'KRAS_analysis'.
    """
    if start is None:
        start = Path(__file__).resolve()
    cur = start if start.is_dir() else start.parent
    for _ in range(8):
        if (cur / "KRAS_analysis").exists():
            return cur
        cur = cur.parent
    # fallback: tools/ -> root
    return Path(__file__).resolve().parent.parent


def ensure_outdir(root: Path) -> Path:
    outdir = root / "KRAS_analysis" / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ----------------------------
# Column auto-detection
# ----------------------------
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def find_basin_col(df: pd.DataFrame) -> str:
    # basin id usually has 'basin' in name
    for c in df.columns:
        if re.search(r"\bbasin\b", c, re.IGNORECASE) or re.search(r"basin_", c, re.IGNORECASE):
            # exclude basin energy columns like basin_energy if needed; but those are still fine
            return c
    # try common names
    c = pick_first_existing(df, ["basin_id", "basin", "basin_index", "cluster", "cluster_id"])
    if c is not None:
        return c
    raise ValueError("Cannot find basin column in dataframe (expected something like basin/basin_id/cluster_id).")


def find_embed_cols(df: pd.DataFrame) -> Tuple[str, str]:
    # Try common pairs
    pairs = [
        ("z1", "z2"),
        ("tsne1", "tsne2"),
        ("t_sne_1", "t_sne_2"),
        ("x", "y"),
        ("embed_x", "embed_y"),
        ("embedding_x", "embedding_y"),
        ("axis1", "axis2"),
    ]
    lower = {c.lower(): c for c in df.columns}
    for a, b in pairs:
        if a in lower and b in lower:
            return lower[a], lower[b]

    # Heuristic: any two numeric columns containing 'tsne' or 'embed' or 'z'
    def score(col: str) -> int:
        s = 0
        if re.search(r"tsne", col, re.IGNORECASE): s += 3
        if re.search(r"embed", col, re.IGNORECASE): s += 3
        if re.fullmatch(r"z\d+", col, re.IGNORECASE): s += 3
        if re.search(r"axis", col, re.IGNORECASE): s += 1
        return s

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ranked = sorted(num_cols, key=lambda c: score(c), reverse=True)
    if len(ranked) >= 2 and score(ranked[0]) > 0 and score(ranked[1]) > 0:
        return ranked[0], ranked[1]

    raise ValueError("Cannot find embedding coordinate columns (need two numeric columns like z1/z2 or tsne1/tsne2).")


def find_rmsd_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"rmsd", c, re.IGNORECASE):
            return c
    return None


def find_weight_col(df: pd.DataFrame) -> Optional[str]:
    # Priority: weight > prob > count
    for cand in ["weight", "prob", "probability", "count", "freq", "frequency"]:
        c = pick_first_existing(df, [cand])
        if c is not None:
            return c
    return None


def infer_system_col(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a column whose values contain WT/G12C/G12D.
    """
    keys = ["WT", "G12C", "G12D"]
    best = (None, 0)
    for c in df.columns:
        if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        s = df[c].astype(str)
        hits = sum(s.str.contains(k, case=False, na=False).sum() for k in keys)
        if hits > best[1]:
            best = (c, hits)
    return best[0]


def normalize_system_values(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper()
    # map common variants
    s = s.replace({
        "WILD": "WT",
        "WILDTYPE": "WT",
        "WILD_TYPE": "WT",
        "WT_1": "WT",
    })
    # reduce to canonical by substring
    out = []
    for v in s.tolist():
        if "G12C" in v:
            out.append("G12C")
        elif "G12D" in v:
            out.append("G12D")
        elif "WT" in v:
            out.append("WT")
        else:
            out.append(v)
    return pd.Series(out, index=series.index)


# ----------------------------
# Utilities
# ----------------------------
def safe_int_basin(x) -> Optional[int]:
    try:
        v = int(float(x))
        return v
    except Exception:
        return None


def weighted_mean_grid(
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return grid of weighted mean(value) over 2D bins.
    """
    # sum w*val
    H_num, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[list(xlim), list(ylim)], weights=weight * value
    )
    # sum w
    H_den, _, _ = np.histogram2d(
        x, y, bins=bins, range=[list(xlim), list(ylim)], weights=weight
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        M = H_num / (H_den + 1e-12)
    M[H_den <= 0] = np.nan

    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            Mn = np.nan_to_num(M, nan=np.nanmedian(M[np.isfinite(M)]) if np.isfinite(M).any() else 0.0)
            Mn = gaussian_filter(Mn, sigma=float(smooth_sigma), mode="nearest")
            # keep empty bins as NaN
            Mn[H_den <= 0] = np.nan
            M = Mn
        except Exception:
            pass

    return M, xedges, yedges


def density_grid(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float = 0.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[list(xlim), list(ylim)], weights=weight
    )
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=float(smooth_sigma), mode="nearest")
        except Exception:
            pass
    if normalize:
        s = float(np.sum(H))
        if s > 0:
            H = H / s
    return H, xedges, yedges


def convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Monotonic chain convex hull.
    points: (N,2)
    return hull vertices in order.
    """
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)


# ----------------------------
# Ring plots (equal 7 sectors, no overlapping labels)
# ----------------------------
def ring_equal7(ax: plt.Axes,
               values: np.ndarray,
               title: str,
               cmap: str,
               vmin: float,
               vmax: float,
               cbar_ax: Optional[plt.Axes] = None,
               cbar_label: str = "",
               show_values: bool = False):
    """
    Draw a 7-sector equal ring. Color encodes values (not wedge size).
    Basin ids 0..6 are labeled outside; no overlap.
    """
    n = 7
    assert len(values) == n
    angles = np.linspace(0, 2*np.pi, n+1)

    # constant wedge sizes
    sizes = np.ones(n) / n

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = mpl.cm.get_cmap(cmap)
    colors = [cm(norm(v)) for v in values]

    # donut
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2.0),
        radius=1.0,
    )
    ax.set(aspect="equal")
    ax.set_title(title, pad=18)

    # basin labels outside at fixed radius
    # center angle for each wedge
    for i in range(n):
        theta = (angles[i] + angles[i+1]) / 2.0
        # pie uses degrees with startangle; convert to actual display theta with startangle=90
        # We'll compute in degrees for placement
        deg = 90 - (i + 0.5) * (360 / n)
        rad = np.deg2rad(deg)

        r_lab = 1.18
        x = r_lab * np.cos(rad)
        y = r_lab * np.sin(rad)

        ha = "left" if x >= 0 else "right"
        ax.text(x, y, f"{i}", ha=ha, va="center", fontsize=14, fontweight="bold")

        if show_values:
            r_val = 0.88
            xv = r_val * np.cos(rad)
            yv = r_val * np.sin(rad)
            ax.text(xv, yv, f"{values[i]:.2f}", ha="center", va="center", fontsize=10)

    if cbar_ax is not None:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        cb = plt.colorbar(sm, cax=cbar_ax)
        cb.set_label(cbar_label, fontsize=14)
        cb.ax.tick_params(labelsize=12)

    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------
# Main plotting functions
# ----------------------------
def plot_basin_occupancy_rings(outdir: Path,
                              occ: Dict[str, np.ndarray]):
    """
    3 rings: WT/G12C/G12D occupancy (prob mass per basin)
    """
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    vmax = max(float(np.nanmax(v)) for v in occ.values())
    vmax = max(vmax, 1e-9)

    ring_equal7(axes[0], occ["WT"], "Occupancy by basin (WT)", cmap="viridis",
                vmin=0.0, vmax=vmax, cbar_ax=None)
    ring_equal7(axes[1], occ["G12C"], "Occupancy by basin (G12C)", cmap="viridis",
                vmin=0.0, vmax=vmax, cbar_ax=None)
    ring_equal7(axes[2], occ["G12D"], "Occupancy by basin (G12D)", cmap="viridis",
                vmin=0.0, vmax=vmax, cbar_ax=cax, cbar_label="Occupancy (probability mass)")

    fig.savefig(outdir / "fig_D_basin_occupancy_rings.png", dpi=600)
    fig.savefig(outdir / "fig_D_basin_occupancy_rings.pdf")
    plt.close(fig)


def plot_basin_shift_rings(outdir: Path,
                           delta_c: np.ndarray,
                           delta_d: np.ndarray):
    """
    2 rings: (G12C-WT) and (G12D-WT), diverging colormap centered at 0
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    vmax = float(np.nanmax(np.abs(np.concatenate([delta_c, delta_d]))))
    vmax = max(vmax, 1e-9)

    ring_equal7(ax1, delta_c, "Distribution shift (G12C − WT)", cmap="coolwarm",
                vmin=-vmax, vmax=vmax, cbar_ax=None)
    ring_equal7(ax2, delta_d, "Distribution shift (G12D − WT)", cmap="coolwarm",
                vmin=-vmax, vmax=vmax, cbar_ax=cax, cbar_label="Δ occupancy (mut − WT)")

    fig.savefig(outdir / "fig_E_delta_rings.png", dpi=600)
    fig.savefig(outdir / "fig_E_delta_rings.pdf")
    plt.close(fig)


def plot_rmsd_heatmap(outdir: Path,
                      df_points: pd.DataFrame,
                      system: str,
                      xcol: str,
                      ycol: str,
                      rmsd_col: str,
                      wcol: Optional[str],
                      bins: int = 220,
                      smooth_sigma: float = 1.2):
    """
    RMSD heatmap (weighted mean RMSD in bins) + density contours.
    """
    d = df_points[df_points["__system__"] == system].copy()
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol]) & np.isfinite(d[rmsd_col])]

    if d.empty:
        print(f"[WARN] RMSD heatmap skipped for {system}: no valid rows.")
        return

    w = np.ones(len(d), dtype=float)
    if wcol is not None and wcol in d.columns:
        ww = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.sum(ww) > 0:
            w = ww

    x = d[xcol].to_numpy(float)
    y = d[ycol].to_numpy(float)
    r = pd.to_numeric(d[rmsd_col], errors="coerce").to_numpy(float)

    # shared limits per system plot for better usage: use global limits from the whole df_points
    xmin, xmax = np.nanmin(df_points[xcol].to_numpy(float)), np.nanmax(df_points[xcol].to_numpy(float))
    ymin, ymax = np.nanmin(df_points[ycol].to_numpy(float)), np.nanmax(df_points[ycol].to_numpy(float))
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    M, xedges, yedges = weighted_mean_grid(x, y, r, w, bins=bins, xlim=xlim, ylim=ylim, smooth_sigma=smooth_sigma)
    H, _, _ = density_grid(x, y, w, bins=bins, xlim=xlim, ylim=ylim, smooth_sigma=smooth_sigma, normalize=True)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    # RMSD heatmap
    im = ax.imshow(
        M.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Mean backbone RMSD", fontsize=16)
    cb.ax.tick_params(labelsize=12)

    # density contours on top (so you see where points actually are)
    # use log1p for stable contours
    D = np.log1p(H)
    xs = 0.5 * (xedges[:-1] + xedges[1:])
    ys = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    levels = np.nanpercentile(D[D > 0], [70, 85, 95]) if np.any(D > 0) else [0.1, 0.2, 0.3]
    ax.contour(Xg, Yg, D, levels=levels, colors="white", linewidths=1.2, alpha=0.9)

    ax.set_title(f"Embedding RMSD heatmap ({system})", pad=12)
    ax.set_xlabel("Structure axis 1 (embedding)")
    ax.set_ylabel("Structure axis 2 (embedding)")
    ax.grid(True)

    fig.savefig(outdir / f"fig_B_rmsd_heatmap_{system}.png", dpi=600)
    fig.savefig(outdir / f"fig_B_rmsd_heatmap_{system}.pdf")
    plt.close(fig)


def plot_representatives_with_basin_blocks(outdir: Path,
                                           df_points: pd.DataFrame,
                                           df_reps: pd.DataFrame,
                                           xcol: str,
                                           ycol: str,
                                           basin_col: str,
                                           wcol: Optional[str],
                                           bins: int = 220,
                                           smooth_sigma: float = 1.2):
    """
    Background: pooled density heatmap (log1p)
    Basin blocks: convex hull per basin (semi-transparent)
    Representatives: red X
    """
    d = df_points.copy()
    d = d[np.isfinite(d[xcol]) & np.isfinite(d[ycol])]
    if d.empty:
        print("[WARN] Representatives plot skipped: points empty.")
        return

    # weights
    w = np.ones(len(d), dtype=float)
    if wcol is not None and wcol in d.columns:
        ww = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.sum(ww) > 0:
            w = ww

    x = d[xcol].to_numpy(float)
    y = d[ycol].to_numpy(float)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    H, xedges, yedges = density_grid(x, y, w, bins=bins, xlim=xlim, ylim=ylim, smooth_sigma=smooth_sigma, normalize=True)
    D = np.log1p(H)

    fig, ax = plt.subplots(figsize=(9.0, 7.2))

    # background density
    ax.imshow(
        D.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        alpha=0.95,
        interpolation="nearest",
    )

    # basin blocks (convex hull)
    # choose a categorical colormap, then apply alpha in patch
    cat_cmap = mpl.cm.get_cmap("tab10")
    for b in range(7):
        dd = d[pd.to_numeric(d[basin_col], errors="coerce").fillna(-1).astype(int) == b]
        if len(dd) < 10:
            continue
        pts = dd[[xcol, ycol]].to_numpy(float)
        hull = convex_hull(pts)
        if len(hull) >= 3:
            poly = Polygon(
                hull, closed=True,
                facecolor=cat_cmap(b % 10),
                edgecolor="white",
                linewidth=2.0,
                alpha=0.22,
            )
            ax.add_patch(poly)

        # label at centroid
        cx, cy = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
        ax.text(cx, cy, f"{b}", ha="center", va="center",
                fontsize=16, fontweight="bold", color="black")

    # representatives overlay
    # auto-detect rep coord cols: prefer same xcol/ycol; else find_embed_cols on reps
    rx, ry = None, None
    if xcol in df_reps.columns and ycol in df_reps.columns:
        rx, ry = xcol, ycol
    else:
        rx, ry = find_embed_cols(df_reps)

    ax.scatter(
        df_reps[rx].to_numpy(float),
        df_reps[ry].to_numpy(float),
        s=260,
        marker="x",
        linewidths=4.0,
        color="red",
        alpha=0.95,
        zorder=5,
    )

    # label reps by basin if available
    rep_basin = None
    try:
        rep_basin = find_basin_col(df_reps)
    except Exception:
        rep_basin = None

    if rep_basin is not None:
        for _, row in df_reps.iterrows():
            try:
                bx = float(row[rx]); by = float(row[ry])
            except Exception:
                continue
            bid = safe_int_basin(row.get(rep_basin, None))
            if bid is None:
                continue
            ax.text(bx, by, f"{bid}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color="black", zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Representatives on embedding (density + basin blocks)", pad=12)
    ax.set_xlabel("Structure axis 1 (embedding)")
    ax.set_ylabel("Structure axis 2 (embedding)")
    ax.grid(True)

    fig.savefig(outdir / "fig_H_representatives_density_basinblocks.png", dpi=600)
    fig.savefig(outdir / "fig_H_representatives_density_basinblocks.pdf")
    plt.close(fig)


# ----------------------------
# Data loading
# ----------------------------
def load_required_csvs(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "KRAS_analysis" / "data_summary" / "merged"
    points_csv = base / "points_enriched.csv"
    basin_csv = base / "basin_master.csv"
    reps_csv = base / "representatives_enriched.csv"

    missing = [p for p in [points_csv, basin_csv, reps_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

    points = pd.read_csv(points_csv)
    basins = pd.read_csv(basin_csv)
    reps = pd.read_csv(reps_csv)
    return points, basins, reps


def compute_basin_occupancy(points: pd.DataFrame,
                            basin_col: str,
                            system_col: str,
                            wcol: Optional[str]) -> Dict[str, np.ndarray]:
    """
    occupancy[system][b] = normalized probability mass in basin b
    """
    out: Dict[str, np.ndarray] = {}
    for system in ["WT", "G12C", "G12D"]:
        d = points[points["__system__"] == system].copy()
        d = d[pd.to_numeric(d[basin_col], errors="coerce").notna()]
        b = pd.to_numeric(d[basin_col], errors="coerce").astype(int).to_numpy()

        w = np.ones(len(d), dtype=float)
        if wcol is not None and wcol in d.columns:
            ww = pd.to_numeric(d[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if np.sum(ww) > 0:
                w = ww

        vec = np.zeros(7, dtype=float)
        for bi in range(7):
            m = (b == bi)
            if np.any(m):
                vec[bi] = float(np.sum(w[m]))

        s = float(np.sum(vec))
        if s > 0:
            vec = vec / s
        out[system] = vec
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    set_style()
    root = find_project_root()
    outdir = ensure_outdir(root)

    points, basins, reps = load_required_csvs(root)

    # Detect columns in points
    basin_col = find_basin_col(points)
    xcol, ycol = find_embed_cols(points)
    rmsd_col = find_rmsd_col(points)
    wcol = find_weight_col(points)

    sys_col = infer_system_col(points)
    if sys_col is None:
        raise ValueError(
            "Cannot infer system column (need values containing WT/G12C/G12D in some string column)."
        )

    points["__system__"] = normalize_system_values(points[sys_col])
    # filter only those we need
    points = points[points["__system__"].isin(["WT", "G12C", "G12D"])].copy()

    # Clean basin to int 0..6 if possible
    points["__basin__"] = pd.to_numeric(points[basin_col], errors="coerce")
    points = points[points["__basin__"].between(0, 6, inclusive="both") | points["__basin__"].isna()].copy()

    # Repoint basin_col to cleaned basin if original is messy
    basin_col_clean = "__basin__"

    print("[INFO] Detected columns:")
    print(f"  system_col = {sys_col}")
    print(f"  basin_col  = {basin_col}  (using cleaned {basin_col_clean})")
    print(f"  embed_cols = ({xcol}, {ycol})")
    print(f"  rmsd_col   = {rmsd_col}")
    print(f"  weight_col = {wcol}")

    # 1) Basin occupancy rings + shift rings
    occ = compute_basin_occupancy(points, basin_col_clean, "__system__", wcol)
    plot_basin_occupancy_rings(outdir, occ)

    delta_c = occ["G12C"] - occ["WT"]
    delta_d = occ["G12D"] - occ["WT"]
    plot_basin_shift_rings(outdir, delta_c, delta_d)

    # 2) RMSD heatmap overlays (per mutant, or all three if你愿意)
    if rmsd_col is not None:
        for system in ["WT", "G12C", "G12D"]:
            plot_rmsd_heatmap(
                outdir=outdir,
                df_points=points,
                system=system,
                xcol=xcol,
                ycol=ycol,
                rmsd_col=rmsd_col,
                wcol=wcol,
                bins=220,
                smooth_sigma=1.2,
            )
    else:
        print("[WARN] No RMSD column detected; RMSD heatmaps skipped.")

    # 3) Representatives plot: density background + basin blocks + reps
    # Detect and normalize system in reps too (optional, not mandatory)
    rep_sys_col = infer_system_col(reps)
    if rep_sys_col is not None:
        reps["__system__"] = normalize_system_values(reps[rep_sys_col])
    plot_representatives_with_basin_blocks(
        outdir=outdir,
        df_points=points,
        df_reps=reps,
        xcol=xcol,
        ycol=ycol,
        basin_col=basin_col_clean,
        wcol=wcol,
        bins=220,
        smooth_sigma=1.2,
    )

    # Save a small manifest (for reproducibility)
    manifest = {
        "input_dir": str((root / "KRAS_analysis" / "data_summary" / "merged").resolve()),
        "output_dir": str(outdir.resolve()),
        "detected_columns": {
            "points_system_col": sys_col,
            "points_basin_col": basin_col,
            "points_embed_cols": [xcol, ycol],
            "points_rmsd_col": rmsd_col,
            "points_weight_col": wcol,
            "reps_system_col": rep_sys_col,
        }
    }
    with open(outdir / "fig_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Figures saved to: {outdir}")


if __name__ == "__main__":
    main()
