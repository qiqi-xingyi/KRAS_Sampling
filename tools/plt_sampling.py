# --*-- conding:utf-8 --*--
# @time:12/19/25 00:27
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_sampling.py

# tools/plt_sampling.py
# ------------------------------------------------------------
# KRAS sampling distribution visualization (Plan A)
# - Pooled structure-space embedding using bitstring Hamming distance (t-SNE)
# - Per-metadata density maps (WT / G12C / G12D) with light Gaussian smoothing
# - Difference maps (G12C - WT, G12D - WT)
# - Jensen–Shannon divergence (JSD) quantification + metrics.json
#
# Directory layout assumed (fixed):
# <project_root>/
#   KRAS_sampling_results/
#     KRAS_4LPK_WT_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#     KRAS_6OIM_G12C_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#     KRAS_9C41_G12D_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#   tools/
#     plt_sampling.py   <-- this file
#
# Run: click Run in IDE.
# Outputs:
# <project_root>/KRAS_sampling_results/plots_A/
#   density_WT.png/pdf
#   density_G12C.png/pdf
#   density_G12D.png/pdf
#   diff_G12C_minus_WT.png/pdf
#   diff_G12D_minus_WT.png/pdf
#   scatter_all.png/pdf
#   embedding_points.csv
#   metrics.json
# ------------------------------------------------------------

from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# -----------------------------
# Fixed metadata (folders)
# -----------------------------
TARGET_DIRS = {
    "WT": "KRAS_4LPK_WT_1",
    "G12C": "KRAS_6OIM_G12C_1",
    "G12D": "KRAS_9C41_G12D_1",
}

# -----------------------------
# Default parameters (IDE-friendly)
# -----------------------------
MAX_POINTS_PER_SET = 2000     # max unique bitstrings per label after aggregation
BINS = 220                    # 2D histogram bins for density maps
SEED = 0                      # embedding seed
SMOOTH_SIGMA = 1.2            # light smoothing for density maps (in bin units)

# Optional robustness check (OFF by default to keep one-click run fast)
ENABLE_STABILITY = False
STABILITY_SEEDS = [0, 1, 2]


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    # tools/plt_sampling.py -> tools -> project_root
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
    L = len(bitstrings[0])
    X = np.empty((len(bitstrings), L), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        X[i] = np.fromiter((1 if c == "1" else 0 for c in s), count=L, dtype=np.uint8)
    return X


def compute_embedding_tsne(X: np.ndarray, seed: int) -> np.ndarray:
    """
    t-SNE embedding with Hamming metric. Compatible with sklearn variants:
    some use n_iter, others use max_iter.
    """
    n = X.shape[0]
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
# Density utilities (smoothing + JSD)
# -----------------------------
def gaussian_smooth_2d(H: np.ndarray, sigma: float) -> np.ndarray:
    """
    Light 2D Gaussian smoothing for histogram maps.
    Tries scipy first; falls back to numpy separable convolution.
    """
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted 2D histogram, optional smoothing, optional normalization (sum=1)."""
    H, xedges, yedges = np.histogram2d(
        Z[:, 0], Z[:, 1],
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


def jsd_from_grids(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence between two nonnegative grids.
    Returns JSD in bits.
    """
    p = P.astype(float).ravel() + eps
    q = Q.astype(float).ravel() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    jsd_nats = 0.5 * (kl(p, m) + kl(q, m))
    return jsd_nats / np.log(2.0)


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
    smooth_sigma: float = 1.2,
):
    H, xedges, yedges = hist2d_weighted(
        Z, weights, bins=bins, xlim=xlim, ylim=ylim,
        smooth_sigma=smooth_sigma, normalize=True
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


def plot_diff_density(
    Z_mut: np.ndarray,
    w_mut: np.ndarray,
    Z_wt: np.ndarray,
    w_wt: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float = 1.2,
):
    Hm, xedges, yedges = hist2d_weighted(
        Z_mut, w_mut, bins=bins, xlim=xlim, ylim=ylim,
        smooth_sigma=smooth_sigma, normalize=True
    )
    Hw, _, _ = hist2d_weighted(
        Z_wt, w_wt, bins=bins, xlim=xlim, ylim=ylim,
        smooth_sigma=smooth_sigma, normalize=True
    )

    D = Hm - Hw
    vmax = float(np.max(np.abs(D))) if np.any(D) else 1.0

    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(
        D.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label="prob(mut) − prob(WT)")
    plt.title(title)
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_scatter_all(Z: np.ndarray, labels: np.ndarray, out_png: Path, out_pdf: Path):
    plt.figure(figsize=(7.2, 6.0))
    for lab in ["WT", "G12C", "G12D"]:
        m = labels == lab
        if np.any(m):
            plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=lab)
    plt.title("KRAS pooled structure-space embedding (t-SNE, Hamming)")
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    proj_root = project_root_from_tools_dir()
    root = proj_root / "KRAS_sampling_results"
    if not root.exists():
        raise FileNotFoundError(
            f"Cannot find KRAS_sampling_results at expected path:\n  {root}\n"
            f"Make sure your repo layout matches the fixed convention."
        )

    out_dir = root / "plots_A"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load & aggregate each set
    per_set: Dict[str, pd.DataFrame] = {}
    bitlen = None

    for label, folder_name in TARGET_DIRS.items():
        folder = root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        files = find_group_csvs(folder)
        if not files:
            raise FileNotFoundError(f"No group0..9 CSVs found under: {folder}")

        df = read_sampling_csvs(files)
        agg = aggregate_bitstrings(df)
        agg = weighted_subsample(agg, max_points=MAX_POINTS_PER_SET, seed=SEED)

        if agg.empty:
            raise RuntimeError(f"No valid rows after filtering in: {folder}")

        L = len(agg.loc[0, "bitstring"])
        if bitlen is None:
            bitlen = L
        elif L != bitlen:
            raise ValueError(f"Bitstring length mismatch: got {L}, expected {bitlen} (label={label})")

        per_set[label] = agg
        print(f"[{label}] files={len(files)} unique_bitstrings={len(agg)}")

    # 2) pool and embed once (shared coordinate system)
    pooled = pd.concat([df.assign(label=lab) for lab, df in per_set.items()], ignore_index=True)
    X = bitstrings_to_binary_matrix(pooled["bitstring"].tolist())
    labels = pooled["label"].to_numpy()
    weights = pooled["weight"].to_numpy(dtype=float)

    print(f"[POOL] total_unique_points={len(pooled)} bitlen={X.shape[1]}")

    Z = compute_embedding_tsne(X, seed=SEED)

    # shared axis limits
    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    Z_by = {lab: Z[labels == lab] for lab in TARGET_DIRS.keys()}
    w_by = {lab: weights[labels == lab] for lab in TARGET_DIRS.keys()}

    # normalize inside each label
    for lab in w_by:
        s = float(w_by[lab].sum())
        if s > 0:
            w_by[lab] = w_by[lab] / s

    # 3) compute JSD metrics on smoothed histogram grids (same bins / same limits)
    H_wt, _, _ = hist2d_weighted(Z_by["WT"], w_by["WT"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
    H_c, _, _ = hist2d_weighted(Z_by["G12C"], w_by["G12C"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
    H_d, _, _ = hist2d_weighted(Z_by["G12D"], w_by["G12D"], BINS, xlim, ylim, SMOOTH_SIGMA, True)

    jsd_g12c_wt = jsd_from_grids(H_c, H_wt)
    jsd_g12d_wt = jsd_from_grids(H_d, H_wt)

    metrics = {
        "embedding": {
            "method": "t-SNE",
            "metric": "hamming",
            "seed": int(SEED),
            "max_points_per_set": int(MAX_POINTS_PER_SET),
        },
        "density": {
            "bins": int(BINS),
            "smooth_sigma": float(SMOOTH_SIGMA),
        },
        "jsd_bits": {
            "G12C_vs_WT": float(jsd_g12c_wt),
            "G12D_vs_WT": float(jsd_g12d_wt),
        },
    }

    print(f"[METRIC] JSD(G12C, WT) = {jsd_g12c_wt:.4f} bits")
    print(f"[METRIC] JSD(G12D, WT) = {jsd_g12d_wt:.4f} bits")

    # 4) density plots (smoothed)
    for lab in ["WT", "G12C", "G12D"]:
        if lab == "WT":
            title = "KRAS structure-space sampling density (WT)"
        elif lab == "G12C":
            title = f"KRAS structure-space sampling density (G12C) | JSD vs WT = {jsd_g12c_wt:.3f} bits"
        else:
            title = f"KRAS structure-space sampling density (G12D) | JSD vs WT = {jsd_g12d_wt:.3f} bits"

        plot_density(
            Z_by[lab],
            w_by[lab],
            out_png=out_dir / f"density_{lab}.png",
            out_pdf=out_dir / f"density_{lab}.pdf",
            title=title,
            bins=BINS,
            xlim=xlim,
            ylim=ylim,
            smooth_sigma=SMOOTH_SIGMA,
        )

    # 5) difference maps (smoothed)
    plot_diff_density(
        Z_mut=Z_by["G12C"], w_mut=w_by["G12C"],
        Z_wt=Z_by["WT"], w_wt=w_by["WT"],
        out_png=out_dir / "diff_G12C_minus_WT.png",
        out_pdf=out_dir / "diff_G12C_minus_WT.pdf",
        title="KRAS density difference: G12C − WT",
        bins=BINS, xlim=xlim, ylim=ylim,
        smooth_sigma=SMOOTH_SIGMA,
    )
    plot_diff_density(
        Z_mut=Z_by["G12D"], w_mut=w_by["G12D"],
        Z_wt=Z_by["WT"], w_wt=w_by["WT"],
        out_png=out_dir / "diff_G12D_minus_WT.png",
        out_pdf=out_dir / "diff_G12D_minus_WT.pdf",
        title="KRAS density difference: G12D − WT",
        bins=BINS, xlim=xlim, ylim=ylim,
        smooth_sigma=SMOOTH_SIGMA,
    )

    # 6) pooled scatter sanity check
    plot_scatter_all(
        Z,
        labels,
        out_png=out_dir / "scatter_all.png",
        out_pdf=out_dir / "scatter_all.pdf",
    )

    # 7) save embedding points table
    emb = pooled.copy()
    emb["z1"] = Z[:, 0]
    emb["z2"] = Z[:, 1]
    emb.to_csv(out_dir / "embedding_points.csv", index=False)

    # 8) optional stability check (multiple seeds)
    if ENABLE_STABILITY:
        stability: Dict[str, Dict[str, float]] = {}
        for sd in STABILITY_SEEDS:
            print(f"[STABILITY] recomputing t-SNE with seed={sd} ...")
            Z_sd = compute_embedding_tsne(X, seed=sd)
            Z_by_sd = {lab: Z_sd[labels == lab] for lab in TARGET_DIRS.keys()}

            H_wt_sd, _, _ = hist2d_weighted(Z_by_sd["WT"], w_by["WT"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
            H_c_sd, _, _ = hist2d_weighted(Z_by_sd["G12C"], w_by["G12C"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
            H_d_sd, _, _ = hist2d_weighted(Z_by_sd["G12D"], w_by["G12D"], BINS, xlim, ylim, SMOOTH_SIGMA, True)

            stability[str(sd)] = {
                "G12C_vs_WT": float(jsd_from_grids(H_c_sd, H_wt_sd)),
                "G12D_vs_WT": float(jsd_from_grids(H_d_sd, H_wt_sd)),
            }

        metrics["stability_jsd_bits_by_seed"] = stability
        print("[STABILITY] JSD across seeds:", stability)

    # 9) save metrics.json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Saved plots + embedding_points.csv + metrics.json to:\n  {out_dir}")


if __name__ == "__main__":
    main()
