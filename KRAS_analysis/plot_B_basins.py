# plot_B_basins.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
DENSITY_BINS = 220            # histogram bins for density background
SMOOTH_SIGMA = 1.2            # light smoothing (bin units)
GRID_N = 360                  # resolution for basin boundary grid (GRID_N x GRID_N)
KNN_K = 25                    # kNN neighbors for grid classification
CHUNK = 25000                 # chunk size for grid neighbor queries


# -----------------------------
# Path helpers
# -----------------------------
def kras_root_from_script() -> Path:
    # This script is placed in KRAS_analysis/
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, preferred_name: str) -> Path:
    """
    Prefer the numbered filename in data_used (e.g., 04_merged_points_with_basin.csv),
    but also support the raw base name if present.
    """
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
# Smoothing
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


# -----------------------------
# Density grid
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
        Z[:, 0],
        Z[:, 1],
        bins=bins,
        range=[list(xlim), list(ylim)],
        weights=weights,
    )
    if smooth_sigma and smooth_sigma > 0:
        H = gaussian_smooth_2d(H, sigma=smooth_sigma)

    if normalize:
        s = float(H.sum())
        if s > 0:
            H = H / s
    return H, xedges, yedges


# -----------------------------
# Basin boundary grid (kNN voting)
# -----------------------------
def compute_basin_label_grid(
    coords: np.ndarray,
    basin_ids: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_n: int,
    k: int,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid in (z1, z2) and assign each grid cell a basin_id via inverse-distance
    weighted kNN voting.
    Returns:
      grid_labels: (grid_n, grid_n) int
      xs: x coordinates (grid_n,)
      ys: y coordinates (grid_n,)
    """
    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])

    nn = NearestNeighbors(n_neighbors=min(k, len(coords)), algorithm="auto")
    nn.fit(coords)

    unique_basins = np.unique(basin_ids).astype(int)
    basin_to_index = {b: i for i, b in enumerate(unique_basins)}

    out_labels = np.empty((grid_points.shape[0],), dtype=int)

    for start in range(0, grid_points.shape[0], chunk):
        end = min(grid_points.shape[0], start + chunk)
        pts = grid_points[start:end]

        dists, idxs = nn.kneighbors(pts, return_distance=True)
        neigh_basins = basin_ids[idxs]  # (M, k)

        # inverse-distance weights
        w = 1.0 / (dists + 1e-6)  # (M, k)

        M, kk = neigh_basins.shape
        score = np.zeros((M, len(unique_basins)), dtype=float)

        # accumulate weights for each basin
        for j in range(kk):
            bj = neigh_basins[:, j].astype(int)
            wj = w[:, j]
            col = np.fromiter((basin_to_index[int(b)] for b in bj), count=M, dtype=int)
            score[np.arange(M), col] += wj

        out_labels[start:end] = unique_basins[np.argmax(score, axis=1)]

    grid_labels = out_labels.reshape((grid_n, grid_n))
    return grid_labels, xs, ys


def basin_centroids(
    df: pd.DataFrame,
    weight_col: Optional[str],
) -> Dict[int, Tuple[float, float]]:
    cent: Dict[int, Tuple[float, float]] = {}
    for bid, g in df.groupby("basin_id"):
        if weight_col and weight_col in g.columns:
            w = g[weight_col].to_numpy(dtype=float)
        else:
            w = np.ones(len(g), dtype=float)
        w = w / (w.sum() + 1e-12)
        z1 = float(np.sum(g["z1"].to_numpy(dtype=float) * w))
        z2 = float(np.sum(g["z2"].to_numpy(dtype=float) * w))
        cent[int(bid)] = (z1, z2)
    return cent


# -----------------------------
# Plot helpers
# -----------------------------
def draw_basin_boundaries(ax: plt.Axes, grid_labels: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """
    Draw boundaries by contouring each basin mask at level=0.5.
    """
    unique_basins = np.unique(grid_labels).astype(int)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    for b in unique_basins:
        mask = (grid_labels == b).astype(float)
        ax.contour(
            mask,
            levels=[0.5],
            colors="black",
            linewidths=0.8,
            origin="lower",
            extent=extent,
        )


def plot_density_with_basins(
    df: pd.DataFrame,
    label: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
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
        w = sub[weight_col].to_numpy(dtype=float)
    else:
        w = np.ones(len(sub), dtype=float)
    w = w / (w.sum() + 1e-12)

    H, xedges, yedges = hist2d_weighted(
        Z, w,
        bins=DENSITY_BINS,
        xlim=xlim,
        ylim=ylim,
        smooth_sigma=SMOOTH_SIGMA,
        normalize=True,
    )
    D = np.log1p(H)

    plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()

    # IMPORTANT: capture the mappable returned by imshow
    im = ax.imshow(
        D.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    # IMPORTANT: bind the colorbar explicitly to the mappable and axis
    plt.colorbar(im, ax=ax, label="log(1 + probability mass)")

    draw_basin_boundaries(ax, grid_labels=grid_labels, xs=xs, ys=ys)

    # basin numbers
    for bid, (cx, cy) in centroids.items():
        ax.text(cx, cy, str(bid), ha="center", va="center", fontsize=11, color="black")

    ax.set_title(title)
    ax.set_xlabel("Structure axis 1 (t-SNE, Hamming)")
    ax.set_ylabel("Structure axis 2 (t-SNE, Hamming)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_basin_map_with_reps(
    df_all: pd.DataFrame,
    reps: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_labels: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    centroids: Dict[int, Tuple[float, float]],
):
    plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()

    # light pooled point background
    ax.scatter(df_all["z1"], df_all["z2"], s=3, alpha=0.15)

    draw_basin_boundaries(ax, grid_labels=grid_labels, xs=xs, ys=ys)

    for bid, (cx, cy) in centroids.items():
        ax.text(cx, cy, str(bid), ha="center", va="center", fontsize=11, color="black")

    reps = reps.copy()
    for col in ["scope", "label", "basin_id", "z1", "z2"]:
        if col not in reps.columns:
            raise ValueError(f"representatives.csv missing required column: {col}")

    def _scatter_subset(sub: pd.DataFrame, marker: str, lab: str):
        if sub.empty:
            return
        ax.scatter(sub["z1"], sub["z2"], s=70, marker=marker, label=lab, alpha=0.95)

    within = reps[reps["scope"].astype(str) == "within_label"]
    global_ = reps[reps["scope"].astype(str) == "global"]

    _scatter_subset(within[within["label"] == "WT"], marker="o", lab="WT reps")
    _scatter_subset(within[within["label"] == "G12D"], marker="s", lab="G12D reps")
    _scatter_subset(within[within["label"] == "G12C"], marker="^", lab="G12C reps")
    _scatter_subset(global_, marker="X", lab="Global reps")

    ax.set_title(title)
    ax.set_xlabel("Structure axis 1 (t-SNE, Hamming)")
    ax.set_ylabel("Structure axis 2 (t-SNE, Hamming)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(frameon=False, loc="best")
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
    figs = root / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    merged_path = pick_file(data_used, "merged_points_with_basin.csv")
    reps_path = pick_file(data_used, "representatives.csv")

    df = pd.read_csv(merged_path)
    reps = pd.read_csv(reps_path)

    required = {"z1", "z2", "label", "basin_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{merged_path.name} missing columns: {sorted(missing)}")

    # optional weight column
    weight_col = None
    for cand in ["weight", "p_mass"]:
        if cand in df.columns:
            weight_col = cand
            break

    # shared axis limits
    xmin, xmax = float(df["z1"].min()), float(df["z1"].max())
    ymin, ymax = float(df["z2"].min()), float(df["z2"].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    coords = df[["z1", "z2"]].to_numpy(dtype=float)
    basin_ids = df["basin_id"].to_numpy(dtype=int)

    print(f"[INFO] Using: {merged_path}")
    print(f"[INFO] Using: {reps_path}")
    print(f"[INFO] Points: {len(df)} | basins: {len(np.unique(basin_ids))}")
    print(f"[INFO] Grid: {GRID_N}x{GRID_N} | kNN: k={KNN_K}")

    grid_labels, xs, ys = compute_basin_label_grid(
        coords=coords,
        basin_ids=basin_ids,
        xlim=xlim,
        ylim=ylim,
        grid_n=GRID_N,
        k=KNN_K,
        chunk=CHUNK,
    )

    centroids = basin_centroids(df, weight_col=weight_col)

    out_wt_png = figs / "B1_basins_overlay_WT.png"
    out_wt_pdf = figs / "B1_basins_overlay_WT.pdf"
    out_d_png = figs / "B2_basins_overlay_G12D.png"
    out_d_pdf = figs / "B2_basins_overlay_G12D.pdf"
    out_rep_png = figs / "B3_basins_representatives.png"
    out_rep_pdf = figs / "B3_basins_representatives.pdf"

    plot_density_with_basins(
        df=df,
        label="WT",
        out_png=out_wt_png,
        out_pdf=out_wt_pdf,
        title="WT density with basin boundaries",
        xlim=xlim,
        ylim=ylim,
        grid_labels=grid_labels,
        xs=xs,
        ys=ys,
        centroids=centroids,
        weight_col=weight_col,
    )

    plot_density_with_basins(
        df=df,
        label="G12D",
        out_png=out_d_png,
        out_pdf=out_d_pdf,
        title="G12D density with basin boundaries",
        xlim=xlim,
        ylim=ylim,
        grid_labels=grid_labels,
        xs=xs,
        ys=ys,
        centroids=centroids,
        weight_col=weight_col,
    )

    plot_basin_map_with_reps(
        df_all=df,
        reps=reps,
        out_png=out_rep_png,
        out_pdf=out_rep_pdf,
        title="Basin map with representative points",
        xlim=xlim,
        ylim=ylim,
        grid_labels=grid_labels,
        xs=xs,
        ys=ys,
        centroids=centroids,
    )

    # manifest for traceability
    manifest = figs / "B_manifest.json"
    payload = {
        "inputs": {
            "merged_points_with_basin": str(merged_path),
            "representatives": str(reps_path),
        },
        "params": {
            "DENSITY_BINS": DENSITY_BINS,
            "SMOOTH_SIGMA": SMOOTH_SIGMA,
            "GRID_N": GRID_N,
            "KNN_K": KNN_K,
            "CHUNK": CHUNK,
            "weight_col": weight_col,
        },
        "outputs": [
            str(out_wt_png), str(out_wt_pdf),
            str(out_d_png), str(out_d_pdf),
            str(out_rep_png), str(out_rep_pdf),
        ],
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n[DONE] Saved:")
    for p in payload["outputs"]:
        print(f"  - {Path(p).name}")
    print(f"  - {manifest.name}")


if __name__ == "__main__":
    main()
