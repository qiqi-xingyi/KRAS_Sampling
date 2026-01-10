# --*-- conding:utf-8 --*--
# @time:1/9/26 19:39
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Qdock_plt_sampling.py

# Batch plot sampling distributions for EACH sampling result folder.
# - For every folder under <project_root>/QDock_sampling_results/KRAS_*:
#     - read samples_*_group{0..9}_ibm.csv
#     - aggregate bitstrings
#     - subsample
#     - t-SNE embedding (Hamming)
#     - density map (smoothed) + optional scatter
#
# Outputs (per case folder):
# <project_root>/QDock_sampling_results/plots_each/<CASE_NAME>/
#   density.png/pdf
#   scatter.png/pdf            (optional)
#   embedding_points.csv
#   metrics.json
#
# Run: click Run in IDE.

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
ROOT_DIRNAME = "QDock_sampling_results"  # your new folder
CASE_PREFIX = "KRAS_"                   # only process folders starting with this
MAX_POINTS_PER_CASE = 2000              # max unique bitstrings after aggregation
BINS = 220                              # 2D histogram bins for density maps
SEED = 0                                # embedding seed
SMOOTH_SIGMA = 1.2                      # light smoothing for density maps (bin units)
MAKE_SCATTER = True                     # also save scatter plot


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    # tools/plt_sampling_batch.py -> tools -> project_root
    return Path(__file__).resolve().parent.parent


# -----------------------------
# IO
# -----------------------------
def find_group_csvs(folder: Path) -> List[Path]:
    """Only group0..9, ignore all_*.csv and other files."""
    files: List[Path] = []
    for g in range(10):
        files.extend(sorted(folder.glob(f"samples_*_group{g}_ibm.csv")))
    return files


def read_sampling_csvs(files: List[Path]) -> pd.DataFrame:
    """
    Read csvs and return minimal DF: [bitstring, weight].
    Weight uses prob if present else count.
    """
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "bitstring" not in df.columns:
            raise ValueError(f"Missing 'bitstring' column in: {f}")

        if "prob" in df.columns:
            w = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
        elif "count" in df.columns:
            w = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
        else:
            raise ValueError(f"Neither 'prob' nor 'count' exists in: {f}")

        tmp = pd.DataFrame({"bitstring": df["bitstring"].astype(str), "weight": w})
        dfs.append(tmp)

    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["bitstring", "weight"])
    out = out[out["bitstring"].str.fullmatch(r"[01]+", na=False)]
    out = out[out["weight"] > 0]
    return out


def aggregate_bitstrings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate identical bitstrings and normalize weights to sum=1."""
    agg = df.groupby("bitstring", as_index=False)["weight"].sum()
    s = float(agg["weight"].sum())
    if s > 0:
        agg["weight"] = agg["weight"] / s
    return agg.sort_values("weight", ascending=False).reset_index(drop=True)


def weighted_subsample(agg: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    """
    Subsample unique bitstrings to keep embedding tractable.
    Keeps top 70% by weight, samples remaining 30% by weight.
    """
    if len(agg) <= max_points:
        return agg

    rng = np.random.default_rng(seed)

    keep_top = int(max_points * 0.7)
    keep_top = max(1, min(keep_top, max_points - 1))

    top = agg.iloc[:keep_top].copy()
    rest = agg.iloc[keep_top:].copy()
    if rest.empty:
        return top

    rest_w = rest["weight"].to_numpy()
    rest_w = rest_w / (rest_w.sum() + 1e-12)

    k = max_points - keep_top
    idx = rng.choice(len(rest), size=k, replace=False, p=rest_w)
    samp = rest.iloc[idx].copy()

    out = pd.concat([top, samp], ignore_index=True)
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    out["weight"] = out["weight"] / (out["weight"].sum() + 1e-12)
    return out


# -----------------------------
# Embedding
# -----------------------------
def bitstrings_to_binary_matrix(bitstrings: List[str]) -> np.ndarray:
    """Convert list of '0101..' -> (N, L) uint8."""
    if not bitstrings:
        return np.zeros((0, 0), dtype=np.uint8)

    L = len(bitstrings[0])
    X = np.empty((len(bitstrings), L), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        if len(s) != L:
            raise ValueError(f"Bitstring length mismatch within one case: {len(s)} vs {L}")
        X[i] = np.fromiter((1 if c == "1" else 0 for c in s), count=L, dtype=np.uint8)
    return X


def compute_embedding_tsne(X: np.ndarray, seed: int) -> np.ndarray:
    """
    t-SNE embedding with Hamming metric. Compatible with sklearn variants:
    some use n_iter, others use max_iter.
    """
    n = X.shape[0]
    if n < 3:
        # trivial fallback: put them on a line
        z = np.zeros((n, 2), dtype=float)
        if n >= 1:
            z[:, 0] = np.arange(n, dtype=float)
        return z

    perplexity = min(50, max(5, (n - 1) // 3))

    kwargs = dict(
        n_components=2,
        metric="hamming",
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
        verbose=1,
    )

    sig = inspect.signature(TSNE.__init__)
    if "n_iter" in sig.parameters:
        kwargs["n_iter"] = 2000
    elif "max_iter" in sig.parameters:
        kwargs["max_iter"] = 2000

    tsne = TSNE(**kwargs)
    return tsne.fit_transform(X)


# -----------------------------
# Density utilities
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


def hist2d_weighted(
    Z: np.ndarray,
    weights: np.ndarray,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float = 0.0,
    normalize: bool = True,
):
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
# Plotting
# -----------------------------
def plot_density(
    Z: np.ndarray,
    weights: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float,
):
    H, xedges, yedges = hist2d_weighted(
        Z, weights,
        bins=bins, xlim=xlim, ylim=ylim,
        smooth_sigma=smooth_sigma,
        normalize=True,
    )
    D = np.log1p(H)

    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(
        D.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    plt.colorbar(label="log(1 + probability mass)")
    plt.title(title)
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_scatter(
    Z: np.ndarray,
    weights: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
):
    # size ~ weight for a quick sanity view (clipped)
    w = weights.astype(float)
    w = w / (w.max() + 1e-12)
    s = 5 + 40 * np.sqrt(w)

    plt.figure(figsize=(7.2, 6.0))
    plt.scatter(Z[:, 0], Z[:, 1], s=s, alpha=0.55)
    plt.title(title)
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Per-case pipeline
# -----------------------------
def process_one_case(case_dir: Path, out_root: Path):
    files = find_group_csvs(case_dir)
    if not files:
        print(f"[SKIP] {case_dir.name}: no group0..9 CSVs found")
        return

    df = read_sampling_csvs(files)
    agg = aggregate_bitstrings(df)
    if agg.empty:
        print(f"[SKIP] {case_dir.name}: empty after filtering")
        return

    agg = weighted_subsample(agg, max_points=MAX_POINTS_PER_CASE, seed=SEED)
    bitlen = len(agg.loc[0, "bitstring"])

    X = bitstrings_to_binary_matrix(agg["bitstring"].tolist())
    Z = compute_embedding_tsne(X, seed=SEED)

    # axis limits per-case
    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    weights = agg["weight"].to_numpy(dtype=float)
    s = float(weights.sum())
    if s > 0:
        weights = weights / s

    out_dir = out_root / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    title_density = f"{case_dir.name} sampling density (t-SNE, Hamming)"
    plot_density(
        Z, weights,
        out_png=out_dir / "density.png",
        out_pdf=out_dir / "density.pdf",
        title=title_density,
        bins=BINS,
        xlim=xlim,
        ylim=ylim,
        smooth_sigma=SMOOTH_SIGMA,
    )

    if MAKE_SCATTER:
        title_scatter = f"{case_dir.name} embedding scatter (size~weight)"
        plot_scatter(
            Z, weights,
            out_png=out_dir / "scatter.png",
            out_pdf=out_dir / "scatter.pdf",
            title=title_scatter,
        )

    emb = agg.copy()
    emb["z1"] = Z[:, 0]
    emb["z2"] = Z[:, 1]
    emb.to_csv(out_dir / "embedding_points.csv", index=False)

    metrics = {
        "case": case_dir.name,
        "files": [p.name for p in files],
        "unique_points_used": int(len(agg)),
        "bitlen": int(bitlen),
        "embedding": {
            "method": "t-SNE",
            "metric": "hamming",
            "seed": int(SEED),
            "max_points_per_case": int(MAX_POINTS_PER_CASE),
        },
        "density": {
            "bins": int(BINS),
            "smooth_sigma": float(SMOOTH_SIGMA),
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] {case_dir.name}: files={len(files)} points={len(agg)} -> {out_dir}")


# -----------------------------
# Main
# -----------------------------
def main():
    proj_root = project_root_from_tools_dir()
    root = proj_root / ROOT_DIRNAME
    if not root.exists():
        raise FileNotFoundError(f"Cannot find {ROOT_DIRNAME} at:\n  {root}")

    out_root = root / "plots_each"
    out_root.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(CASE_PREFIX)])
    if not case_dirs:
        raise RuntimeError(f"No case folders found under {root} with prefix '{CASE_PREFIX}'")

    print(f"[FOUND] {len(case_dirs)} case folders under {root}")
    for case_dir in case_dirs:
        try:
            process_one_case(case_dir, out_root)
        except Exception as e:
            print(f"[FAIL] {case_dir.name}: {type(e).__name__}: {e}")

    print(f"[DONE] All outputs saved under:\n  {out_root}")


if __name__ == "__main__":
    main()
