# plot_C_marginals.py
# ------------------------------------------------------------
# C: Colored (basin) blurred map + marginal projections
# - Main: per-basin 2D histogram -> Gaussian blur -> colored RGBA composite
# - Optional: overlay light scatter for texture (can turn off)
# - Marginals: WT baseline as gray filled curve; mutants overlay colored curve
#
# Input:
#   KRAS_analysis/data_used/*merged_points_with_basin.csv
#     required: z1, z2, label, basin_id
#     optional weights: weight/prob/count/mass...
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

# Layout / style
FIGSIZE = (8.4, 7.2)
BG_FACECOLOR = "#f3f3f3"   # light gray panel background
AX_SPINE_COLOR = "#777777"

# Basin colors (must match B: stable mapping by sorted basin_id)
BASIN_PALETTE = "tab20"

# Main (colored blur)
BINS_2D = 260
SMOOTH_SIGMA_2D = 1.0         # stronger blur to form blobs
ALPHA_POWER = 0.65            # alpha = (norm_density)**ALPHA_POWER (smaller -> more filled)
ALPHA_MAX = 0.95              # clamp
COMPOSITE_MODE = "add"        # "add" or "max" (add looks more like heat overlay)

# Optional scatter overlay (to keep point texture)
OVERLAY_SCATTER = True
MAX_SCATTER_POINTS_PER_LABEL = 45000
POINT_SIZE = 7
POINT_ALPHA = 0.10

# Marginals
BINS_1D = 140
SMOOTH_SIGMA_1D = 2.2
WT_FILL_ALPHA = 0.22
WT_LINE_ALPHA = 0.55
WT_LINE_W = 1.6
MUT_LINE_W = 2.1

# Performance
SEED = 0
MAX_POINTS_FOR_DENSITY_PER_LABEL = 180000  # used for density construction; set None to disable


# -----------------------------
# Utilities
# -----------------------------
def root_dir() -> Path:
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, suffix: str) -> Path:
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


def basin_color_map(basin_ids_sorted: List[int]) -> Dict[int, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(BASIN_PALETTE, max(len(basin_ids_sorted), 1))
    out: Dict[int, Tuple[float, float, float, float]] = {}
    for i, b in enumerate(basin_ids_sorted):
        out[b] = cmap(i)
    return out


def mutation_line_color(label: str) -> Tuple[float, float, float, float]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    if label == "G12C":
        return plt.matplotlib.colors.to_rgba(cycle[0])
    if label == "G12D":
        return plt.matplotlib.colors.to_rgba(cycle[1])
    return plt.matplotlib.colors.to_rgba("#555555")


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


# -----------------------------
# Colored blur composite
# -----------------------------
def build_colored_blur_rgba(
    df_label: pd.DataFrame,
    basin_ids_global: List[int],
    bmap: Dict[int, Tuple[float, float, float, float]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    bins: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns rgba image (Ny,Nx,4) and xedges,yedges
    """
    z1 = df_label["z1"].to_numpy(dtype=float)
    z2 = df_label["z2"].to_numpy(dtype=float)

    # use optional weights only for density mass
    w = None
    # weights handled outside: we already subsample by weight; density mass can remain unweighted for appearance
    # If you want, you can restore weighted histogram here.

    # common grid
    xedges = np.linspace(xlim[0], xlim[1], bins + 1, dtype=float)
    yedges = np.linspace(ylim[0], ylim[1], bins + 1, dtype=float)

    # composite buffer
    rgba = np.zeros((bins, bins, 4), dtype=float)

    # per-basin blurred densities
    # normalize each basin by its own max (stable blob visibility), then alpha-power
    for b in basin_ids_global:
        part = df_label[df_label["basin_id"] == b]
        if part.empty:
            continue

        H, _, _ = np.histogram2d(
            part["z1"].to_numpy(dtype=float),
            part["z2"].to_numpy(dtype=float),
            bins=[xedges, yedges],
            weights=w,
        )
        H = gaussian_smooth_2d(H, sigma=sigma)

        m = float(H.max())
        if m <= 0:
            continue
        Hn = H / (m + 1e-12)  # [0,1]

        # alpha shaping
        A = np.power(Hn, ALPHA_POWER)
        A = np.clip(A, 0.0, ALPHA_MAX)

        r, g, bb, _ = bmap[b]
        layer = np.zeros_like(rgba)
        layer[..., 0] = r
        layer[..., 1] = g
        layer[..., 2] = bb
        layer[..., 3] = A

        if COMPOSITE_MODE == "max":
            # per-channel "over" approx: keep max alpha and corresponding color mass (simple)
            # better: alpha-composite; but max gives crisp basin identity
            mask = layer[..., 3] > rgba[..., 3]
            rgba[mask, 0] = layer[mask, 0]
            rgba[mask, 1] = layer[mask, 1]
            rgba[mask, 2] = layer[mask, 2]
            rgba[mask, 3] = layer[mask, 3]
        else:
            # additive blending with alpha accumulation (looks like colorful heat)
            rgba[..., 0] += layer[..., 0] * layer[..., 3]
            rgba[..., 1] += layer[..., 1] * layer[..., 3]
            rgba[..., 2] += layer[..., 2] * layer[..., 3]
            rgba[..., 3] += layer[..., 3] * 0.85

    # clamp & normalize color by alpha to avoid overbright
    rgba[..., 3] = np.clip(rgba[..., 3], 0.0, 1.0)
    # avoid division by zero
    denom = np.clip(rgba[..., 3][..., None], 1e-6, 1.0)
    rgba[..., 0:3] = np.clip(rgba[..., 0:3] / denom, 0.0, 1.0)

    return rgba, xedges, yedges


# -----------------------------
# Plot
# -----------------------------
def plot_one_panel(
    df_all: pd.DataFrame,
    df_wt_full: pd.DataFrame,
    df_density_full: pd.DataFrame,
    label: str,
    out_png: Path,
    out_pdf: Path,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    wcol: Optional[str],
    basin_ids_global: List[int],
    bmap: Dict[int, Tuple[float, float, float, float]],
):
    from matplotlib.gridspec import GridSpec

    sub_scatter = df_all[df_all["label"] == label].copy()
    sub_density = df_density_full[df_density_full["label"] == label].copy()
    if sub_scatter.empty or sub_density.empty:
        raise ValueError(f"No data for label={label}")

    # WT baseline for marginals uses FULL WT
    wt = df_wt_full.copy()

    # weights (marginals)
    w_wt = None
    w_sub = None
    if wcol and wcol in wt.columns:
        w_wt = pd.to_numeric(wt[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if wcol and wcol in sub_density.columns:
        w_sub = pd.to_numeric(sub_density[wcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Marginals
    yx_wt, cx = hist1d_smoothed(wt["z1"].to_numpy(dtype=float), w_wt, BINS_1D, xlim, SMOOTH_SIGMA_1D)
    yy_wt, cy = hist1d_smoothed(wt["z2"].to_numpy(dtype=float), w_wt, BINS_1D, ylim, SMOOTH_SIGMA_1D)

    if label == "WT":
        yx_mut, yy_mut = None, None
    else:
        yx_mut, _ = hist1d_smoothed(sub_density["z1"].to_numpy(dtype=float), w_sub, BINS_1D, xlim, SMOOTH_SIGMA_1D)
        yy_mut, _ = hist1d_smoothed(sub_density["z2"].to_numpy(dtype=float), w_sub, BINS_1D, ylim, SMOOTH_SIGMA_1D)

    # Colored blurred RGBA
    rgba, xedges, yedges = build_colored_blur_rgba(
        df_label=sub_density,
        basin_ids_global=basin_ids_global,
        bmap=bmap,
        xlim=xlim,
        ylim=ylim,
        bins=BINS_2D,
        sigma=SMOOTH_SIGMA_2D,
    )

    # Layout
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = GridSpec(2, 2, height_ratios=[1.15, 5.0], width_ratios=[5.0, 1.15], hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    for ax in [ax_top, ax_main, ax_right]:
        ax.set_facecolor(BG_FACECOLOR)
        for sp in ax.spines.values():
            sp.set_color(AX_SPINE_COLOR)
            sp.set_linewidth(0.8)

    # Main: show colored RGBA "blob" (this is the key change)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax_main.imshow(
        rgba.transpose(1, 0, 2),  # histogram2d uses x-y; align with extent
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )

    # Optional texture scatter (very light)
    if OVERLAY_SCATTER and not sub_scatter.empty:
        for b in basin_ids_global:
            part = sub_scatter[sub_scatter["basin_id"] == b]
            if part.empty:
                continue
            ax_main.scatter(
                part["z1"].to_numpy(),
                part["z2"].to_numpy(),
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                c=[bmap[b]],
                edgecolors="none",
                rasterized=True,
                zorder=3,
            )

    # Basin id labels (helps reading)
    cent = sub_scatter.groupby("basin_id")[["z1", "z2"]].mean()
    for b, row in cent.iterrows():
        t = ax_main.text(
            float(row["z1"]), float(row["z2"]), str(int(b)),
            ha="center", va="center",
            fontsize=11,
            color="#111111",
            zorder=6
        )
        t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white", alpha=0.9)])

    ax_main.set_xlim(xlim)
    ax_main.set_ylim(ylim)
    ax_main.set_xlabel("Structure axis 1 (t-SNE, Hamming)")
    ax_main.set_ylabel("Structure axis 2 (t-SNE, Hamming)")
    ax_main.set_title(f"{label} colored basin blur + marginal projections")

    # Top marginal (X)
    ax_top.fill_between(cx, 0, yx_wt, color="#666666", alpha=WT_FILL_ALPHA, linewidth=0.0)
    ax_top.plot(cx, yx_wt, color="#666666", alpha=WT_LINE_ALPHA, linewidth=WT_LINE_W, label="WT (baseline)")
    if label != "WT" and yx_mut is not None:
        c_mut = mutation_line_color(label)
        ax_top.plot(cx, yx_mut, color=c_mut, linewidth=MUT_LINE_W, label=f"{label}")
    ax_top.set_xlim(xlim)
    ax_top.set_xticks([])
    ax_top.set_ylabel("mass")
    ax_top.grid(False)

    # Right marginal (Y)
    ax_right.fill_betweenx(cy, 0, yy_wt, color="#666666", alpha=WT_FILL_ALPHA, linewidth=0.0)
    ax_right.plot(yy_wt, cy, color="#666666", alpha=WT_LINE_ALPHA, linewidth=WT_LINE_W)
    if label != "WT" and yy_mut is not None:
        c_mut = mutation_line_color(label)
        ax_right.plot(yy_mut, cy, color=c_mut, linewidth=MUT_LINE_W)
    ax_right.set_ylim(ylim)
    ax_right.set_yticks([])
    ax_right.set_xlabel("mass")
    ax_right.grid(False)

    if label != "WT":
        ax_top.legend(frameon=False, loc="upper right", fontsize=10)

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

    # Scatter subset
    parts_scatter = []
    for lab in TARGET_LABELS:
        sub = df[df["label"] == lab].copy()
        if sub.empty:
            continue
        sub = subsample(sub, MAX_SCATTER_POINTS_PER_LABEL, wcol, seed=SEED)
        parts_scatter.append(sub)
    df_scatter = pd.concat(parts_scatter, ignore_index=True) if parts_scatter else df

    # Density subset (can be larger)
    parts_density = []
    for lab in TARGET_LABELS:
        sub = df[df["label"] == lab].copy()
        if sub.empty:
            continue
        sub = subsample(sub, MAX_POINTS_FOR_DENSITY_PER_LABEL, wcol, seed=SEED + 17)
        parts_density.append(sub)
    df_density = pd.concat(parts_density, ignore_index=True) if parts_density else df

    # global limits for comparability
    xmin, xmax = float(df_density["z1"].min()), float(df_density["z1"].max())
    ymin, ymax = float(df_density["z2"].min()), float(df_density["z2"].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # global basin colors (consistent with B)
    basin_ids_global = sorted(df["basin_id"].unique().tolist())
    bmap = basin_color_map(basin_ids_global)

    # WT full for baseline marginals
    df_wt_full = df[df["label"] == "WT"].copy()
    if df_wt_full.empty:
        raise ValueError("Cannot find WT rows in merged_points_with_basin.csv")

    out_map = {
        "WT": (out_dir / "C1_marginals_WT.png", out_dir / "C1_marginals_WT.pdf"),
        "G12C": (out_dir / "C2_marginals_G12C.png", out_dir / "C2_marginals_G12C.pdf"),
        "G12D": (out_dir / "C3_marginals_G12D.png", out_dir / "C3_marginals_G12D.pdf"),
    }

    for lab in TARGET_LABELS:
        if not (df_density["label"] == lab).any():
            print(f"[WARN] label '{lab}' not found, skip.")
            continue
        out_png, out_pdf = out_map[lab]
        plot_one_panel(
            df_all=df_scatter,
            df_wt_full=df_wt_full,
            df_density_full=df_density,
            label=lab,
            out_png=out_png,
            out_pdf=out_pdf,
            xlim=xlim,
            ylim=ylim,
            wcol=wcol,
            basin_ids_global=basin_ids_global,
            bmap=bmap,
        )

    manifest = {
        "inputs": {"merged_points_with_basin": str(path_points)},
        "params": {
            "BINS_2D": BINS_2D,
            "SMOOTH_SIGMA_2D": SMOOTH_SIGMA_2D,
            "ALPHA_POWER": ALPHA_POWER,
            "ALPHA_MAX": ALPHA_MAX,
            "COMPOSITE_MODE": COMPOSITE_MODE,
            "OVERLAY_SCATTER": OVERLAY_SCATTER,
            "BINS_1D": BINS_1D,
            "SMOOTH_SIGMA_1D": SMOOTH_SIGMA_1D,
            "MAX_POINTS_FOR_DENSITY_PER_LABEL": MAX_POINTS_FOR_DENSITY_PER_LABEL,
            "MAX_SCATTER_POINTS_PER_LABEL": MAX_SCATTER_POINTS_PER_LABEL,
            "PALETTE": BASIN_PALETTE,
            "weight_col": wcol,
        },
        "outputs": [str(p) for p in fixed_outputs],
        "notes": [
            "Main panel is a colored blur: per-basin 2D hist -> Gaussian blur -> RGBA composite using basin colors.",
            "Marginals: WT baseline gray filled; mutants overlay colored curves.",
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
