# --*-- conding:utf-8 --*--
# @time:1/12/26 13:49
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_C_marginals.py

# plot_C_marginals.py
# ------------------------------------------------------------
# C: Sampling distribution with marginal projections
# - Main: smoothed density background (Gaussian) + points colored by basin (B palette)
# - Marginals: WT baseline as gray filled curve; mutants overlay their curve on top of WT
# - Light gray background, no "all-scatter" look
#
# Input:
#   KRAS_analysis/data_used/*merged_points_with_basin.csv (e.g., 04_merged_points_with_basin.csv)
#     required cols: z1, z2, label, basin_id
#     optional: weight/prob/count/mass for weighting
#
# Output (overwrite, fixed filenames):
#   KRAS_analysis/figs/C_marginals/
#     C1_marginals_WT.png/.pdf
#     C2_marginals_G12C.png/.pdf
#     C3_marginals_G12D.png/.pdf
#     C_manifest.json
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


# -----------------------------
# Config
# -----------------------------
TARGET_LABELS = ["WT", "G12C", "G12D"]

# Visual
FIGSIZE = (8.4, 7.2)
BG_FACECOLOR = "#f3f3f3"          # light gray panel background
DENSITY_CMAP = "Greys"            # soft grayscale density
DENSITY_ALPHA = 0.80
POINT_SIZE = 9
POINT_ALPHA = 0.33

# Basin colors (must match B: stable mapping by sorted basin_id)
BASIN_PALETTE = "tab20"

# Density smoothing
BINS_2D = 240
SMOOTH_SIGMA_2D = 1.25            # in bin units

# Marginals (1D)
BINS_1D = 140
SMOOTH_SIGMA_1D = 2.2             # in bin units
WT_FILL_ALPHA = 0.25
WT_LINE_ALPHA = 0.55
MUT_LINE_W = 2.0
WT_LINE_W = 1.6

# Performance
SEED = 0
MAX_POINTS_PER_LABEL = 120000     # scatter subsample; set None to disable


# -----------------------------
# Utilities
# -----------------------------
def root_dir() -> Path:
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, suffix: str) -> Path:
    # match both "merged_points_with_basin.csv" and "04_merged_points_with_basin.csv"
    cands = sorted(data_used_dir.glob(f"*{suffix}"))
    if not cands:
        raise FileNotFoundError(f"Cannot find *{suffix} under: {data_used_dir}")
    return cands[0]


def detect_weight_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["weight", "prob", "p_mass", "mass", "occupancy", "count"]:
        if c in df.columns:
            return c
    return None


def gaussian_smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return y
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore
        return gaussian_filter1d(y, sigma=float(sigma), mode="nearest")
    except Exception:
        s = float(sigma)
        radius = max(1, int(3 * s))
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-0.5 * (x / s) ** 2)
        k = k / (k.sum() + 1e-12)
        return np.convolve(y, k, mode="same")


def gaussian_smooth_2d(H: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return H
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(H, sigma=float(sigma), mode="nearest")
    except Exception:
        s = float(sigma)
        radius = max(1, int(3 * s))
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-0.5 * (x / s) ** 2)
        k = k / (k.sum() + 1e-12)
        H0 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, H)
        H1 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, H0)
        return H1


def subsample(df: pd.DataFrame, max_points: Optional[int], wcol: Optional[str], seed: int) -> pd.DataFrame:
    if max_points is None or len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    if wcol and wcol in df.columns:
        w = pd.to_numeric(df[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s > 0:
            p = w / s
            idx = rng.choice(len(df), size=int(max_points), replace=False, p=p)
            return df.iloc[idx].copy()
    idx = rng.choice(len(df), size=int(max_points), replace=False)
    return df.iloc[idx].copy()


def hist2d_smoothed(z1: np.ndarray, z2: np.ndarray, w: Optional[np.ndarray],
                    bins: int, xlim: Tuple[float, float], ylim: Tuple[float, float], sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, xedges, yedges = np.histogram2d(
        z1, z2,
        bins=bins,
        range=[list(xlim), list(ylim)],
        weights=w
    )
    H = gaussian_smooth_2d(H, sigma=sigma)
    s = float(H.sum())
    if s > 0:
        H = H / s
    return H, xedges, yedges


def hist1d_smoothed(x: np.ndarray, w: Optional[np.ndarray], bins: int, xlim: Tuple[float, float], sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    y, edges = np.histogram(x, bins=bins, range=list(xlim), weights=w, density=False)
    y = y.astype(float)
    if y.sum() > 0:
        y = y / (y.sum() + 1e-12)
    y = gaussian_smooth_1d(y, sigma=sigma)
    if y.sum() > 0:
        y = y / (y.sum() + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return y, centers


def basin_color_map(basin_ids_sorted: List[int]) -> Dict[int, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(BASIN_PALETTE, max(len(basin_ids_sorted), 1))
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for i, b in enumerate(basin_ids_sorted):
        out[b] = cmap(i)
    return out


def mutation_line_color(label: str) -> Tuple[float, float, float, float]:
    # Use default matplotlib cycle colors, but keep deterministic mapping
    # WT not used as "mutant overlay"
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    if label == "G12C":
        return plt.matplotlib.colors.to_rgba(cycle[0])
    if label == "G12D":
        return plt.matplotlib.colors.to_rgba(cycle[1])
    return plt.matplotlib.colors.to_rgba("#555555")


# -----------------------------
# Plot
# -----------------------------
def plot_one_panel(
    df_all: pd.DataFrame,
    df_wt: pd.DataFrame,
    label: str,
    out_png: Path,
    out_pdf: Path,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    wcol: Optional[str],
    bmap: Dict[int, Tuple[float, float, float, float]],
):
    from matplotlib.gridspec import GridSpec

    sub = df_all[df_all["label"] == label].copy()
    if sub.empty:
        raise ValueError(f"No data for label={label}")

    # weights (optional)
    w_sub = None
    w_wt = None
    if wcol and wcol in sub.columns:
        w_sub = pd.to_numeric(sub[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w_wt = pd.to_numeric(df_wt[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    z1 = sub["z1"].to_numpy(dtype=float)
    z2 = sub["z2"].to_numpy(dtype=float)
    z1_wt = df_wt["z1"].to_numpy(dtype=float)
    z2_wt = df_wt["z2"].to_numpy(dtype=float)

    # Density background (smoothed)
    H, xedges, yedges = hist2d_smoothed(z1, z2, w_sub, BINS_2D, xlim, ylim, SMOOTH_SIGMA_2D)
    D = np.log1p(H)

    # Marginals
    yx_wt, cx = hist1d_smoothed(z1_wt, w_wt, BINS_1D, xlim, SMOOTH_SIGMA_1D)
    yy_wt, cy = hist1d_smoothed(z2_wt, w_wt, BINS_1D, ylim, SMOOTH_SIGMA_1D)

    if label == "WT":
        yx_mut, yy_mut = None, None
    else:
        yx_mut, _ = hist1d_smoothed(z1, w_sub, BINS_1D, xlim, SMOOTH_SIGMA_1D)
        yy_mut, _ = hist1d_smoothed(z2, w_sub, BINS_1D, ylim, SMOOTH_SIGMA_1D)

    # Layout: top marginal + main + right marginal
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = GridSpec(2, 2, height_ratios=[1.15, 5.0], width_ratios=[5.0, 1.15], hspace=0.05, wspace=0.05)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # Style
    for ax in [ax_top, ax_main, ax_right]:
        ax.set_facecolor(BG_FACECOLOR)
        for sp in ax.spines.values():
            sp.set_color("#777777")
            sp.set_linewidth(0.8)

    # --- Main: density background ---
    im = ax_main.imshow(
        D.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap=DENSITY_CMAP,
        alpha=DENSITY_ALPHA,
        zorder=1,
    )

    # --- Main: basin-colored points overlay ---
    # keep stable basin draw order
    basin_ids = sorted(sub["basin_id"].unique().tolist())
    for b in basin_ids:
        part = sub[sub["basin_id"] == b]
        if part.empty:
            continue
        ax_main.scatter(
            part["z1"].to_numpy(),
            part["z2"].to_numpy(),
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=[bmap.get(int(b), (0.2, 0.2, 0.2, 1.0))],
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )

    # basin id labels (optional but helps)
    cent = sub.groupby("basin_id")[["z1", "z2"]].mean()
    for b, row in cent.iterrows():
        t = ax_main.text(float(row["z1"]), float(row["z2"]), str(int(b)),
                         ha="center", va="center", fontsize=11, color="#111111", zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white", alpha=0.9)])

    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.set_xlabel("Structure axis 1 (t-SNE, Hamming)")
    ax_main.set_ylabel("Structure axis 2 (t-SNE, Hamming)")
    ax_main.set_title(f"{label} sampling map with marginal projections")

    # --- Top marginal (X) ---
    ax_top.fill_between(cx, 0, yx_wt, color="#666666", alpha=WT_FILL_ALPHA, linewidth=0.0)
    ax_top.plot(cx, yx_wt, color="#666666", alpha=WT_LINE_ALPHA, linewidth=WT_LINE_W, label="WT (baseline)")
    if label != "WT" and yx_mut is not None:
        c_mut = mutation_line_color(label)
        ax_top.plot(cx, yx_mut, color=c_mut, linewidth=MUT_LINE_W, label=f"{label}")
    ax_top.set_xlim(xlim)
    ax_top.set_xticks([])
    ax_top.set_ylabel("mass")
    ax_top.grid(False)

    # --- Right marginal (Y) ---
    # (plot horizontally: value vs y)
    ax_right.fill_betweenx(cy, 0, yy_wt, color="#666666", alpha=WT_FILL_ALPHA, linewidth=0.0)
    ax_right.plot(yy_wt, cy, color="#666666", alpha=WT_LINE_ALPHA, linewidth=WT_LINE_W)
    if label != "WT" and yy_mut is not None:
        c_mut = mutation_line_color(label)
        ax_right.plot(yy_mut, cy, color=c_mut, linewidth=MUT_LINE_W)
    ax_right.set_ylim(ylim)
    ax_right.set_yticks([])
    ax_right.set_xlabel("mass")
    ax_right.grid(False)

    # Legend only on mutant plots (keeps WT clean)
    if label != "WT":
        ax_top.legend(frameon=False, loc="upper right", fontsize=10)

    # Add a subtle colorbar for density (optional; comment out if you want even cleaner)
    cbar = fig.colorbar(im, ax=[ax_main, ax_right], fraction=0.035, pad=0.02)
    cbar.set_label("log(1 + smoothed probability mass)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close(fig)


def main():
    root = root_dir()
    data_used = root / "data_used"
    out_dir = root / "figs" / "C_marginals"
    out_dir.mkdir(parents=True, exist_ok=True)

    # overwrite-only (fixed filenames)
    fixed_outputs = [
        out_dir / "C1_marginals_WT.png",
        out_dir / "C1_marginals_WT.pdf",
        out_dir / "C2_marginals_G12C.png",
        out_dir / "C2_marginals_G12C.pdf",
        out_dir / "C3_marginals_G12D.png",
        out_dir / "C3_marginals_G12D.pdf",
        out_dir / "C_manifest.json",
    ]
    for p in fixed_outputs:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    path_points = pick_file(data_used, "merged_points_with_basin.csv")
    df = pd.read_csv(path_points)

    # validate / cast
    for c in ["z1", "z2", "label", "basin_id"]:
        if c not in df.columns:
            raise ValueError(f"{path_points.name} missing column: {c}")

    df = df.copy()
    df["label"] = df["label"].astype(str)
    df["z1"] = pd.to_numeric(df["z1"], errors="coerce")
    df["z2"] = pd.to_numeric(df["z2"], errors="coerce")
    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce")
    df = df.dropna(subset=["z1", "z2", "basin_id", "label"]).copy()
    df["basin_id"] = df["basin_id"].astype(int)

    wcol = detect_weight_column(df)

    # subsample per label for scatter (keeps speed)
    parts = []
    for lab in TARGET_LABELS:
        sub = df[df["label"] == lab].copy()
        if sub.empty:
            continue
        sub = subsample(sub, MAX_POINTS_PER_LABEL, wcol, seed=SEED)
        parts.append(sub)
    df_plot = pd.concat(parts, ignore_index=True) if parts else df

    # global x/y limits for strict comparability
    xmin, xmax = float(df_plot["z1"].min()), float(df_plot["z1"].max())
    ymin, ymax = float(df_plot["z2"].min()), float(df_plot["z2"].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # global basin color mapping (stable)
    basin_ids_sorted = sorted(df["basin_id"].unique().tolist())
    bmap = basin_color_map(basin_ids_sorted)

    # WT baseline dataframe for marginals (use FULL WT from df for stability)
    df_wt_full = df[df["label"] == "WT"].copy()
    if df_wt_full.empty:
        raise ValueError("Cannot find WT rows in merged_points_with_basin.csv")

    # output mapping
    out_map = {
        "WT": (out_dir / "C1_marginals_WT.png", out_dir / "C1_marginals_WT.pdf"),
        "G12C": (out_dir / "C2_marginals_G12C.png", out_dir / "C2_marginals_G12C.pdf"),
        "G12D": (out_dir / "C3_marginals_G12D.png", out_dir / "C3_marginals_G12D.pdf"),
    }

    for lab in TARGET_LABELS:
        if not (df_plot["label"] == lab).any():
            print(f"[WARN] label '{lab}' not found, skip.")
            continue
        out_png, out_pdf = out_map[lab]
        plot_one_panel(
            df_all=df_plot,
            df_wt=df_wt_full,
            label=lab,
            out_png=out_png,
            out_pdf=out_pdf,
            xlim=xlim,
            ylim=ylim,
            wcol=wcol,
            bmap=bmap,
        )

    # manifest
    manifest = {
        "inputs": {"merged_points_with_basin": str(path_points)},
        "params": {
            "BINS_2D": BINS_2D,
            "SMOOTH_SIGMA_2D": SMOOTH_SIGMA_2D,
            "BINS_1D": BINS_1D,
            "SMOOTH_SIGMA_1D": SMOOTH_SIGMA_1D,
            "MAX_POINTS_PER_LABEL": MAX_POINTS_PER_LABEL,
            "PALETTE": BASIN_PALETTE,
            "BG_FACECOLOR": BG_FACECOLOR,
            "weight_col": wcol,
        },
        "outputs": [str(p) for p in fixed_outputs],
        "notes": [
            "Main panel: smoothed density background + basin-colored point overlay.",
            "Marginals: WT baseline (gray fill/line); mutants overlay colored curve.",
            "Axis limits shared across labels for comparability.",
            "Fixed filenames only; overwritten each run.",
        ],
    }
    with open(out_dir / "C_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in fixed_outputs:
        print("  -", p.name)


if __name__ == "__main__":
    main()
