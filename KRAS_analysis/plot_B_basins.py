# plot_B_basins.py
# ------------------------------------------------------------
# KRAS basin visualization (Plan B - points colored by basin + TRUE basin boundaries)
#
# Key change:
# - Basin boundaries are NOT Voronoi wedges.
# - We rasterize basin regions by kNN classification on a grid using (z1,z2)->basin_id from data.
# - Then draw contour boundaries of those regions.
#
# Input (under KRAS_analysis/data_used):
#   merged_points_with_basin.csv (or *_merged_points_with_basin.csv)
#     required columns: z1, z2, label, basin_id
#     optional: weight/prob/count/mass... for subsampling
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
MAX_POINTS_PER_LABEL = 80000   # keep responsiveness; set None to disable
POINT_SIZE = 10
POINT_ALPHA = 0.30

# categorical colors
PALETTE = "tab20"

# true basin boundary rendering (kNN on grid)
DRAW_BOUNDARIES = True
GRID_N = 520                   # boundary grid resolution (higher = smoother but slower)
KNN_K = 9                       # kNN neighbors (odd number recommended)
BOUNDARY_COLOR = "#000000"
BOUNDARY_LW = 1.6
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

    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s > 0:
            p = w / s
            idx = rng.choice(len(df), size=int(max_points), replace=False, p=p)
            return df.iloc[idx].copy()

    idx = rng.choice(len(df), size=int(max_points), replace=False)
    return df.iloc[idx].copy()


# -----------------------------
# Basin centroids (for labels only)
# -----------------------------
def compute_basin_centroids(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    g = df.groupby("basin_id")[["z1", "z2"]].mean()
    return {int(i): (float(r["z1"]), float(r["z2"])) for i, r in g.iterrows()}


# -----------------------------
# TRUE boundary grid by kNN classification
# -----------------------------
def _knn_predict_grid_labels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Predict basin_id on a (ys,xs) grid using kNN over labeled points.
    Weighted vote by inverse distance.
    Returns grid_labels with shape (len(ys), len(xs)).
    """
    k = int(max(1, k))
    if k > len(X_train):
        k = len(X_train)

    # Prefer scipy cKDTree for speed if available; fallback to sklearn NearestNeighbors.
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(X_train)
        # query returns (d, idx) shapes: (M,k)
        Xg, Yg = np.meshgrid(xs, ys)
        Q = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        d, idx = tree.query(Q, k=k, workers=-1)
    except Exception:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(X_train)
        Xg, Yg = np.meshgrid(xs, ys)
        Q = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        d, idx = nn.kneighbors(Q, return_distance=True)

    # Ensure 2D shapes
    if k == 1:
        d = d.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    neigh_labels = y_train[idx]  # (M,k)

    # weighted vote: w = 1/(d+eps)
    eps = 1e-6
    w = 1.0 / (d + eps)

    # For each query, accumulate weights per label
    # Efficient loop (M can be ~270k for 520^2); use per-row unique accumulation.
    M = neigh_labels.shape[0]
    out = np.empty(M, dtype=int)
    for i in range(M):
        labs = neigh_labels[i]
        wi = w[i]
        # accumulate
        score: Dict[int, float] = {}
        for lab, ww in zip(labs, wi):
            lab_int = int(lab)
            score[lab_int] = score.get(lab_int, 0.0) + float(ww)
        # argmax
        out[i] = max(score.items(), key=lambda kv: kv[1])[0]

    return out.reshape(len(ys), len(xs))


def build_true_boundary_grid(
    df_points: pd.DataFrame,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    grid_n: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], grid_n, dtype=float)
    ys = np.linspace(ylim[0], ylim[1], grid_n, dtype=float)

    X_train = df_points[["z1", "z2"]].to_numpy(dtype=float)
    y_train = df_points["basin_id"].to_numpy(dtype=int)

    grid_labels = _knn_predict_grid_labels(X_train, y_train, xs, ys, k=k)
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
            zorder=8,
        )


# -----------------------------
# Plotting
# -----------------------------
def plot_scatter_by_basin_with_boundaries(
    df_all: pd.DataFrame,
    df_for_boundaries: pd.DataFrame,
    label: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cmap_name: str,
    draw_boundaries: bool,
):
    sub = df_all[df_all["label"] == label].copy()
    if sub.empty:
        raise ValueError(f"No rows for label={label}")

    basin_ids = sorted(sub["basin_id"].unique().tolist())
    cmap = plt.get_cmap(cmap_name, max(len(basin_ids), 1))

    fig = plt.figure(figsize=(7.6, 6.2), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    # points colored by basin
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

    # TRUE basin boundaries from kNN grid using points of THIS label
    if draw_boundaries:
        grid_labels, xs, ys = build_true_boundary_grid(
            df_points=df_for_boundaries[df_for_boundaries["label"] == label].copy(),
            xlim=xlim,
            ylim=ylim,
            grid_n=GRID_N,
            k=KNN_K,
        )
        draw_basin_boundaries(ax, grid_labels=grid_labels, xs=xs, ys=ys)

    # basin ids at centroids (within this label)
    cent = compute_basin_centroids(sub)
    for b, (cx, cy) in cent.items():
        txt = ax.text(
            cx, cy, str(b),
            ha="center", va="center",
            fontsize=12,
            color="#111111",
            zorder=10,
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
        patches.append(Patch(facecolor=(r, g, bb, 0.75), edgecolor="none", label=f"Basin {b}"))

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

    # df_plot: for scatter (may subsample)
    parts = []
    for lab in TARGET_LABELS:
        sub = df[df["label"] == lab].copy()
        if sub.empty:
            continue
        sub = weighted_subsample_df(sub, MAX_POINTS_PER_LABEL, weight_col, seed=SEED)
        parts.append(sub)
    df_plot = pd.concat(parts, ignore_index=True) if parts else df

    # df_bound: for boundaries (use MORE points for smoother region; but cap to avoid huge runtime)
    # Here we reuse df_plot if you want speed; or use df (full) for best fidelity.
    df_bound = df  # best fidelity

    # global axis limits (consistent across labels)
    xmin, xmax = float(df_plot["z1"].min()), float(df_plot["z1"].max())
    ymin, ymax = float(df_plot["z2"].min()), float(df_plot["z2"].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # outputs mapping (filenames unchanged)
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
        plot_scatter_by_basin_with_boundaries(
            df_all=df_plot,
            df_for_boundaries=df_bound,
            label=lab,
            out_png=out_png,
            out_pdf=out_pdf,
            title=f"{lab} basins (points colored) with true boundaries",
            xlim=xlim,
            ylim=ylim,
            cmap_name=PALETTE,
            draw_boundaries=DRAW_BOUNDARIES,
        )

    # legend basin ids (global)
    basin_ids_sorted = sorted(df["basin_id"].unique().tolist())
    leg_png = out_dir / "B4_basin_legend.png"
    leg_pdf = out_dir / "B4_basin_legend.pdf"
    plot_basin_legend(basin_ids_sorted, leg_png, leg_pdf, palette=PALETTE)

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
            "KNN_K": KNN_K,
            "BOUNDARY_LW": BOUNDARY_LW,
            "weight_col": weight_col,
        },
        "notes": {
            "figure": "Density heatmap removed. Points colored by basin_id.",
            "boundaries": "True basin boundaries: kNN classification on grid using (kNN over labeled points) per label.",
            "overwrite_policy": "Outputs overwrite fixed filenames only; no new files are created.",
        },
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in fixed_outputs:
        print("  -", p.name)


if __name__ == "__main__":
    main()
