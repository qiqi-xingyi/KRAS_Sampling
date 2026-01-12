# plot_B_basins.py
# ------------------------------------------------------------
# KRAS basin visualization (Plan B)
# White background + light density (alpha-encoded) + pastel basin masks (frosted)
#
# Input (under KRAS_analysis/data_used):
#   *_merged_points_with_basin.csv   (required)
#     expected columns: z1, z2, label, basin_id, [weight/prob/count/mass... optional]
#
# Output:
#   KRAS_analysis/figs/B_basins/
#     B1_density_with_basins_WT.png/.pdf
#     B2_density_with_basins_G12C.png/.pdf
#     B3_density_with_basins_G12D.png/.pdf
#     B4_basin_legend.png/.pdf
#     B_manifest.json
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
FIG_SUBDIR = "B_basins"
TARGET_LABELS = ["WT", "G12C", "G12D"]

# density
BINS = 240
SMOOTH_SIGMA = 1.2  # in bin units

# basin region rendering
GRID_N = 520
MASK_ALPHA = 0.18          # pastel overlay opacity
MASK_BLUR_SIGMA = 2.4      # softness for mask edges
PALETTE = "Pastel1"        # pastel colors on white

# density style on white background
DENSITY_CMAP = "Greys"     # 0 -> white, high -> black
DENSITY_ALPHA_MAX = 0.85   # maximum opacity for high density
DENSITY_ALPHA_GAMMA = 0.55 # <1: emphasize low-density haze, >1: emphasize peaks

# subtle glass veil (optional; keep tiny)
GLASS_VEIL_ALPHA = 0.03

# boundary style
BOUNDARY_COLOR = "#FFFFFF"
BOUNDARY_LW = 1.4
BOUNDARY_ALPHA = 0.95


# -----------------------------
# Path helpers
# -----------------------------
def kras_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, preferred_name: str) -> Path:
    candidates = sorted(data_used_dir.glob(f"*_{preferred_name}"))
    if candidates:
        return candidates[0]
    p = data_used_dir / preferred_name
    if p.exists():
        return p
    candidates = sorted(data_used_dir.glob(f"*{preferred_name}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot find '{preferred_name}' under: {data_used_dir}")


# -----------------------------
# Smoothing utilities
# -----------------------------
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


def gaussian_blur_2d(M: np.ndarray, sigma: float) -> np.ndarray:
    if sigma is None or sigma <= 0:
        return M
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        return gaussian_filter(M.astype(float), sigma=float(sigma), mode="nearest")
    except Exception:
        s = float(sigma)
        radius = max(1, int(3 * s))
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-0.5 * (x / s) ** 2)
        k = k / (k.sum() + 1e-12)
        M0 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, M.astype(float))
        M1 = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, M0)
        return M1


# -----------------------------
# Density utilities
# -----------------------------
def hist2d_weighted(
    Z: np.ndarray,
    weights: np.ndarray,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float = 0.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, xedges, yedges = np.histogram2d(
        Z[:, 0], Z[:, 1],
        bins=bins,
        range=[list(xlim), list(ylim)],
        weights=weights,
    )
    if smooth_sigma and smooth_sigma > 0:
        H = gaussian_smooth_2d(H, sigma=float(smooth_sigma))
    if normalize:
        s = float(H.sum())
        if s > 0:
            H = H / s
    return H, xedges, yedges


# -----------------------------
# Basin partition (Voronoi by centroid)
# -----------------------------
def compute_basin_centroids(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    tmp = df.copy()
    tmp["basin_id"] = pd.to_numeric(tmp["basin_id"], errors="coerce")
    tmp = tmp.dropna(subset=["basin_id"]).copy()
    tmp["basin_id"] = tmp["basin_id"].astype(int)
    g = tmp.groupby("basin_id")[["z1", "z2"]].mean()
    return {int(i): (float(r["z1"]), float(r["z2"])) for i, r in g.iterrows()}


def build_voronoi_grid_labels(
    centroids: Dict[int, Tuple[float, float]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    basin_ids = sorted(centroids.keys())
    C = np.array([centroids[b] for b in basin_ids], dtype=float)  # (K,2)

    xs = np.linspace(xlim[0], xlim[1], grid_n, dtype=float)
    ys = np.linspace(ylim[0], ylim[1], grid_n, dtype=float)
    Xg, Yg = np.meshgrid(xs, ys)
    P = np.stack([Xg, Yg], axis=-1)  # (Ny,Nx,2)

    d2 = np.sum((P[..., None, :] - C[None, None, :, :]) ** 2, axis=-1)
    idx = np.argmin(d2, axis=-1)  # (Ny,Nx)

    grid_labels = np.zeros_like(idx, dtype=int)
    for i, b in enumerate(basin_ids):
        grid_labels[idx == i] = b

    return grid_labels, xs, ys


# -----------------------------
# Basin overlay rendering
# -----------------------------
def make_basin_overlay_rgba(
    grid_labels: np.ndarray,
    alpha: float = 0.18,
    blur_sigma: float = 2.4,
    palette: str = "Pastel1",
) -> np.ndarray:
    unique_basins = np.unique(grid_labels).astype(int)
    cmap = plt.get_cmap(palette, max(len(unique_basins), 1))

    H, W = grid_labels.shape
    overlay = np.zeros((H, W, 4), dtype=float)

    for i, b in enumerate(unique_basins):
        mask = (grid_labels == b).astype(float)
        mask = gaussian_blur_2d(mask, sigma=float(blur_sigma))
        mmax = float(mask.max())
        if mmax > 0:
            mask = mask / mmax

        r, g, bb, _ = cmap(i)
        overlay[..., 0] += r * mask * alpha
        overlay[..., 1] += g * mask * alpha
        overlay[..., 2] += bb * mask * alpha
        overlay[..., 3] += mask * alpha

    # keep it airy on white background
    overlay[..., 3] = np.clip(overlay[..., 3], 0.0, 0.42)
    overlay[..., 0:3] = np.clip(overlay[..., 0:3], 0.0, 1.0)
    return overlay


def draw_basin_boundaries(ax: plt.Axes, grid_labels: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    unique_basins = np.unique(grid_labels).astype(int)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    for b in unique_basins:
        mask = (grid_labels == b).astype(float)
        ax.contour(
            mask,
            levels=[0.5],
            colors=BOUNDARY_COLOR,
            linewidths=BOUNDARY_LW,
            alpha=BOUNDARY_ALPHA,
            origin="lower",
            extent=extent,
            zorder=6,
        )


# -----------------------------
# IO helpers
# -----------------------------
def detect_weight_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["weight", "prob", "p_mass", "mass", "occupancy", "count"]:
        if c in df.columns:
            return c
    return None


def load_points_with_basin(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["z1", "z2", "label", "basin_id"]:
        if c not in df.columns:
            raise ValueError(f"{path.name} missing column: {c}")

    df = df.copy()
    df["label"] = df["label"].astype(str)

    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce")
    df = df.dropna(subset=["basin_id"]).copy()
    df["basin_id"] = df["basin_id"].astype(int)

    df["z1"] = pd.to_numeric(df["z1"], errors="coerce")
    df["z2"] = pd.to_numeric(df["z2"], errors="coerce")
    df = df.dropna(subset=["z1", "z2"]).copy()
    return df


# -----------------------------
# Core: white background density rendering
# -----------------------------
def density_to_rgba_on_white(D: np.ndarray, vmax: float, cmap_name: str) -> np.ndarray:
    """
    Convert density grid D (nonnegative) to an RGBA image rendered on white:
    - color from cmap (Greys is ideal: low -> white)
    - alpha proportional to normalized density (gamma curve)
    """
    norm = np.clip(D / (vmax + 1e-12), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)  # (H,W,4)

    # alpha encoding -> "light haze" on white
    a = (norm ** float(DENSITY_ALPHA_GAMMA)) * float(DENSITY_ALPHA_MAX)
    rgba[..., 3] = np.clip(a, 0.0, 1.0)

    return rgba


def plot_density_with_basins_white(
    df: pd.DataFrame,
    label: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_labels: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    centroids: Dict[int, Tuple[float, float]],
    weight_col: Optional[str],
):
    sub = df[df["label"] == label].copy()
    if sub.empty:
        raise ValueError(f"No rows for label={label}")

    Z = sub[["z1", "z2"]].to_numpy(dtype=float)

    if weight_col and weight_col in sub.columns:
        w = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if weight_col == "count":
            w = np.maximum(w, 0.0)
    else:
        w = np.ones(len(sub), dtype=float)
    w = w / (w.sum() + 1e-12)

    H, xedges, yedges = hist2d_weighted(
        Z, w,
        bins=bins,
        xlim=xlim,
        ylim=ylim,
        smooth_sigma=SMOOTH_SIGMA,
        normalize=True,
    )
    D = np.log1p(H)

    # robust vmax so the haze is visible but not overblown
    if np.any(D > 0):
        vmax = float(np.percentile(D[D > 0], 99.2))
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-12)

    fig = plt.figure(figsize=(7.6, 6.2), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # 1) density as RGBA on white (no dark background)
    rgba = density_to_rgba_on_white(D.T, vmax=vmax, cmap_name=DENSITY_CMAP)
    ax.imshow(
        rgba,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        interpolation="bilinear",
        zorder=1,
    )

    # colorbar (use ScalarMappable for correct scale)
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=plt.get_cmap(DENSITY_CMAP))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label("log(1 + probability mass)")

    # 2) pastel basin masks (frosted)
    overlay = make_basin_overlay_rgba(
        grid_labels=grid_labels,
        alpha=MASK_ALPHA,
        blur_sigma=MASK_BLUR_SIGMA,
        palette=PALETTE,
    )
    ax.imshow(
        overlay,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        interpolation="bilinear",
        zorder=3,
    )

    # 3) subtle veil (optional)
    ax.add_patch(
        Rectangle(
            (xlim[0], ylim[0]),
            xlim[1] - xlim[0],
            ylim[1] - ylim[0],
            facecolor="white",
            alpha=GLASS_VEIL_ALPHA,
            edgecolor="none",
            zorder=4,
        )
    )

    # 4) boundaries
    draw_basin_boundaries(ax, grid_labels=grid_labels, xs=xs, ys=ys)

    # 5) basin ids with stroke (dark text on white)
    for bid, (cx, cy) in centroids.items():
        txt = ax.text(
            cx, cy, str(bid),
            ha="center", va="center",
            fontsize=12,
            color="#111111",
            zorder=8,
        )
        txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white", alpha=0.95)])

    ax.set_title(title)
    ax.set_xlabel("Structure axis 1 (t-SNE, Hamming)")
    ax.set_ylabel("Structure axis 2 (t-SNE, Hamming)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close(fig)


def plot_basin_legend(basin_ids: List[int], out_png: Path, out_pdf: Path, palette: str):
    cmap = plt.get_cmap(palette, max(len(basin_ids), 1))
    patches = []
    for i, b in enumerate(basin_ids):
        r, g, bb, _ = cmap(i)
        patches.append(Patch(facecolor=(r, g, bb, 0.35), edgecolor="none", label=f"Basin {b}"))

    plt.figure(figsize=(7.8, 2.3), facecolor="white")
    ax = plt.gca()
    ax.axis("off")
    ax.legend(
        handles=patches,
        ncol=min(6, max(1, len(patches))),
        frameon=False,
        loc="center",
        fontsize=11,
    )
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
    out_dir = root / "figs" / FIG_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    points_path = pick_file(data_used, "merged_points_with_basin.csv")
    df = load_points_with_basin(points_path)
    weight_col = detect_weight_column(df)

    # global axis limits
    xmin, xmax = float(df["z1"].min()), float(df["z1"].max())
    ymin, ymax = float(df["z2"].min()), float(df["z2"].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # basins (global centroids)
    centroids = compute_basin_centroids(df)
    grid_labels, xs, ys = build_voronoi_grid_labels(
        centroids=centroids,
        xlim=xlim,
        ylim=ylim,
        grid_n=GRID_N,
    )
    basin_ids_sorted = sorted(centroids.keys())

    outputs: List[str] = []

    # B1/B2/B3
    for i, lab in enumerate(TARGET_LABELS, start=1):
        if not (df["label"] == lab).any():
            print(f"[WARN] label '{lab}' not found, skip.")
            continue

        out_png = out_dir / f"B{i}_density_with_basins_{lab}.png"
        out_pdf = out_dir / f"B{i}_density_with_basins_{lab}.pdf"

        plot_density_with_basins_white(
            df=df,
            label=lab,
            out_png=out_png,
            out_pdf=out_pdf,
            title=f"{lab} density with basin masks",
            bins=BINS,
            xlim=xlim,
            ylim=ylim,
            grid_labels=grid_labels,
            xs=xs,
            ys=ys,
            centroids=centroids,
            weight_col=weight_col,
        )
        outputs += [str(out_png), str(out_pdf)]

    # B4 legend
    leg_png = out_dir / "B4_basin_legend.png"
    leg_pdf = out_dir / "B4_basin_legend.pdf"
    plot_basin_legend(basin_ids_sorted, leg_png, leg_pdf, palette=PALETTE)
    outputs += [str(leg_png), str(leg_pdf)]

    # manifest
    manifest = out_dir / "B_manifest.json"
    payload = {
        "inputs": {"merged_points_with_basin": str(points_path)},
        "params": {
            "BINS": BINS,
            "SMOOTH_SIGMA": SMOOTH_SIGMA,
            "GRID_N": GRID_N,
            "MASK_ALPHA": MASK_ALPHA,
            "MASK_BLUR_SIGMA": MASK_BLUR_SIGMA,
            "PALETTE": PALETTE,
            "DENSITY_CMAP": DENSITY_CMAP,
            "DENSITY_ALPHA_MAX": DENSITY_ALPHA_MAX,
            "DENSITY_ALPHA_GAMMA": DENSITY_ALPHA_GAMMA,
            "GLASS_VEIL_ALPHA": GLASS_VEIL_ALPHA,
            "weight_col": weight_col,
        },
        "basins": basin_ids_sorted,
        "outputs": outputs,
        "notes": {
            "theme": "White background. Density rendered as alpha-encoded haze on white. Pastel frosted basin masks.",
            "partition": "Basin regions are Voronoi partitions over basin centroids in embedding space.",
        },
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in outputs:
        print("  -", Path(p).name)
    print("  -", manifest.name)


if __name__ == "__main__":
    main()
