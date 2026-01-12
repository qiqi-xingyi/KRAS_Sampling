# plot_B_basins.py
# ------------------------------------------------------------
# KRAS basin visualization (Plan B - points colored by basin)
#
# Change vs previous:
# - REMOVE density heatmap (already shown elsewhere)
# - Color points by basin_id directly (clear basin regions)
# - Optional: keep white basin boundaries (Voronoi over centroids) as guide
#
# Input (under KRAS_analysis/data_used):
#   *_merged_points_with_basin.csv   (required)
#     expected columns: z1, z2, label, basin_id, [weight/prob/count/mass... optional]
#
# Output (overwrite, filenames unchanged):
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
from matplotlib.patches import Patch


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
FIG_SUBDIR = "B_basins"
TARGET_LABELS = ["WT", "G12C", "G12D"]

# scatter style
SEED = 0
MAX_POINTS_PER_LABEL = 60000   # subsample to keep figure responsive (set None to disable)
POINT_SIZE = 9
POINT_ALPHA = 0.28
EDGE_ALPHA = 0.95

# palette for basins (categorical)
PALETTE = "tab20"

# show boundaries as guidance (Voronoi over centroids)
DRAW_BOUNDARIES = True
GRID_N = 520
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


def weighted_subsample_df(
    df: pd.DataFrame,
    max_points: Optional[int],
    weight_col: Optional[str],
    seed: int,
) -> pd.DataFrame:
    if max_points is None or len(df) <= max_points:
        return df

    rng = np.random.default_rng(seed)

    # Prefer keeping heavier points if a weight column exists
    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s > 0:
            p = w / s
            idx = rng.choice(len(df), size=int(max_points), replace=False, p=p)
            return df.iloc[idx].copy()

    # Fallback: uniform random
    idx = rng.choice(len(df), size=int(max_points), replace=False)
    return df.iloc[idx].copy()


# -----------------------------
# Basin centroids + Voronoi boundaries (optional)
# -----------------------------
def compute_basin_centroids(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    g = df.groupby("basin_id")[["z1", "z2"]].mean()
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
# Plotting
# -----------------------------
def plot_scatter_by_basin(
    df_all: pd.DataFrame,
    label: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cmap_name: str,
    draw_boundaries: bool,
    grid_labels: Optional[np.ndarray],
    xs: Optional[np.ndarray],
    ys: Optional[np.ndarray],
):
    sub = df_all[df_all["label"] == label].copy()
    if sub.empty:
        raise ValueError(f"No rows for label={label}")

    basin_ids = sorted(sub["basin_id"].unique().tolist())
    cmap = plt.get_cmap(cmap_name, max(len(basin_ids), 1))

    fig = plt.figure(figsize=(7.6, 6.2), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # Scatter each basin in its own color
    for i, b in enumerate(basin_ids):
        part = sub[sub["basin_id"] == b]
        if part.empty:
            continue
        color = cmap(i)
        ax.scatter(
            part["z1"].to_numpy(),
            part["z2"].to_numpy(),
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=[color],
            edgecolors="none",
            rasterized=True,   # keep PDF small
            zorder=3,
        )

    # Optional: boundaries as a guide (global Voronoi)
    if draw_boundaries and grid_labels is not None and xs is not None and ys is not None:
        draw_basin_boundaries(ax, grid_labels=grid_labels, xs=xs, ys=ys)

    # Basin id labels at centroids (computed within this label for better placement)
    cent = compute_basin_centroids(sub)
    for b, (cx, cy) in cent.items():
        txt = ax.text(
            cx, cy, str(b),
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
        patches.append(Patch(facecolor=(r, g, bb, 0.65), edgecolor="none", label=f"Basin {b}"))

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

    # Overwrite-only policy: remove only known outputs
    fixed_outputs = [
        out_dir / "B1_density_with_basins_WT.png",
        out_dir / "B1_density_with_basins_WT.pdf",
        out_dir / "B2_density_with_basins_G12C.png",
        out_dir / "B2_density_with_basins_G12C.pdf",
        out_dir / "B3_density_with_basins_G12D.png",
        out_dir / "B3_density_with_basins_G12D.pdf",
        out_dir / "B4_basin_legend.png",
        out_dir / "B4_basin_legend.pdf",
        out_dir / "B_manifest.json",
    ]
    for p in fixed_outputs:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    points_path = pick_file(data_used, "merged_points_with_basin.csv")
    df = load_points_with_basin(points_path)
    weight_col = detect_weight_column(df)

    # subsample per label (optional)
    parts = []
    for lab in TARGET_LABELS:
        sub = df[df["label"] == lab].copy()
        if sub.empty:
            continue
        sub = weighted_subsample_df(sub, MAX_POINTS_PER_LABEL, weight_col, seed=SEED)
        parts.append(sub)
    df_plot = pd.concat(parts, ignore_index=True) if parts else df

    # global axis limits (consistent across labels)
    xmin, xmax = float(df_plot["z1"].min()), float(df_plot["z1"].max())
    ymin, ymax = float(df_plot["z2"].min()), float(df_plot["z2"].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # global boundaries grid (optional)
    grid_labels = xs = ys = None
    if DRAW_BOUNDARIES:
        centroids_global = compute_basin_centroids(df)  # use full df for stable boundaries
        grid_labels, xs, ys = build_voronoi_grid_labels(
            centroids=centroids_global,
            xlim=xlim,
            ylim=ylim,
            grid_n=GRID_N,
        )
        basin_ids_sorted = sorted(centroids_global.keys())
    else:
        basin_ids_sorted = sorted(df["basin_id"].unique().tolist())

    # B1/B2/B3 (filenames unchanged)
    out_map = {
        "WT": (out_dir / "B1_density_with_basins_WT.png", out_dir / "B1_density_with_basins_WT.pdf"),
        "G12C": (out_dir / "B2_density_with_basins_G12C.png", out_dir / "B2_density_with_basins_G12C.pdf"),
        "G12D": (out_dir / "B3_density_with_basins_G12D.png", out_dir / "B3_density_with_basins_G12D.pdf"),
    }

    for lab in TARGET_LABELS:
        if not (df_plot["label"] == lab).any():
            print(f"[WARN] label '{lab}' not found, skip.")
            continue

        out_png, out_pdf = out_map[lab]
        plot_scatter_by_basin(
            df_all=df_plot,
            label=lab,
            out_png=out_png,
            out_pdf=out_pdf,
            title=f"{lab} points colored by basin",
            xlim=xlim,
            ylim=ylim,
            cmap_name=PALETTE,
            draw_boundaries=DRAW_BOUNDARIES,
            grid_labels=grid_labels,
            xs=xs,
            ys=ys,
        )

    # B4 legend (unchanged filename)
    leg_png = out_dir / "B4_basin_legend.png"
    leg_pdf = out_dir / "B4_basin_legend.pdf"
    plot_basin_legend(basin_ids_sorted, leg_png, leg_pdf, palette=PALETTE)

    # manifest (unchanged filename)
    manifest = out_dir / "B_manifest.json"
    payload = {
        "inputs": {"merged_points_with_basin": str(points_path)},
        "params": {
            "SEED": SEED,
            "MAX_POINTS_PER_LABEL": MAX_POINTS_PER_LABEL,
            "POINT_SIZE": POINT_SIZE,
            "POINT_ALPHA": POINT_ALPHA,
            "PALETTE": PALETTE,
            "DRAW_BOUNDARIES": DRAW_BOUNDARIES,
            "GRID_N": GRID_N,
            "weight_col": weight_col,
        },
        "notes": {
            "figure": "Density heatmap removed. Points are directly colored by basin_id for clarity.",
            "boundaries": "Optional Voronoi boundaries over global basin centroids are drawn as a guide.",
            "overwrite_policy": "Outputs overwrite fixed filenames only; no new files are created.",
        },
        "outputs": [str(p) for p in fixed_outputs if p.name != "B_manifest.json"] + [str(manifest)],
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in fixed_outputs:
        print("  -", p.name)


if __name__ == "__main__":
    main()
