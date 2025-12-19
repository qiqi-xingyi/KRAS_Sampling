# --*-- conding:utf-8 --*--
# @time:12/19/25 00:27
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_sampling.py

#!/usr/bin/env python3
"""
KRAS sampling distribution visualization (Plan A: structure-space embedding from distances)

What this script does:
1) Collects sampling CSVs for three metadata sets:
   - WT1  : KRAS_4LPK_WT_1
   - G12C : KRAS_6OIM_G12C_1
   - G12D : KRAS_9C41_G12D_1
   Only reads files: samples_*_group{0..9}_ibm.csv (ignores *_all_*.csv and other files)

2) Builds a structure-space embedding using bitstring Hamming distance (TSNE with metric="hamming")
   - Embedding is fitted ONCE on pooled data -> coordinates are directly comparable across WT/G12C/G12D

3) Plots weighted density heatmaps (log1p) for each set + difference maps (mut - WT)

Outputs:
- <out_dir>/density_WT.png/pdf
- <out_dir>/density_G12C.png/pdf
- <out_dir>/density_G12D.png/pdf
- <out_dir>/diff_G12C_minus_WT.png/pdf
- <out_dir>/diff_G12D_minus_WT.png/pdf
- <out_dir>/scatter_all.png/pdf

Run:
python plot_kras_structure_density.py --root KRAS_sampling_results --out KRAS_sampling_results/plots_A
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# -----------------------------
# Config: which folders to use
# -----------------------------
TARGET_DIRS = {
    "WT": "KRAS_4LPK_WT_1",
    "G12C": "KRAS_6OIM_G12C_1",
    "G12D": "KRAS_9C41_G12D_1",
}


def find_group_csvs(folder: Path) -> list[Path]:
    # Only group0..group9, ignore all_*.csv
    files = []
    for g in range(10):
        files.extend(sorted(folder.glob(f"samples_*_group{g}_ibm.csv")))
    return files


def read_sampling_csvs(files: list[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # minimal columns needed
        need = {"bitstring"}
        if not need.issubset(df.columns):
            raise ValueError(f"Missing required columns in {f}: need={need}")
        # weight: prefer prob, fallback to count
        if "prob" in df.columns:
            df["weight"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
        elif "count" in df.columns:
            df["weight"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
        else:
            raise ValueError(f"No 'prob' or 'count' in {f}")
        df = df[["bitstring", "weight"]].copy()
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["bitstring", "weight"])
    out = pd.concat(dfs, ignore_index=True)
    # sanitize
    out["bitstring"] = out["bitstring"].astype(str)
    out = out[out["bitstring"].str.fullmatch(r"[01]+", na=False)]
    out = out[out["weight"] > 0]
    return out


def aggregate_bitstrings(df: pd.DataFrame) -> pd.DataFrame:
    # Sum weights for identical bitstrings
    agg = df.groupby("bitstring", as_index=False)["weight"].sum()
    # Normalize to probability mass (sum=1)
    s = float(agg["weight"].sum())
    if s > 0:
        agg["weight"] = agg["weight"] / s
    return agg.sort_values("weight", ascending=False).reset_index(drop=True)


def weighted_subsample(agg: pd.DataFrame, max_points: int, seed: int = 0) -> pd.DataFrame:
    if len(agg) <= max_points:
        return agg

    rng = np.random.default_rng(seed)

    # Keep top portion deterministically, sample the rest for diversity
    keep_top = int(max_points * 0.7)
    keep_top = max(1, min(keep_top, max_points - 1))

    top = agg.iloc[:keep_top].copy()
    rest = agg.iloc[keep_top:].copy()

    if rest.empty:
        return top

    rest_w = rest["weight"].to_numpy()
    rest_w = rest_w / rest_w.sum()

    k = max_points - keep_top
    idx = rng.choice(len(rest), size=k, replace=False, p=rest_w)
    samp = rest.iloc[idx].copy()

    out = pd.concat([top, samp], ignore_index=True)
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    # renormalize after subsampling
    out["weight"] = out["weight"] / out["weight"].sum()
    return out


def bitstrings_to_binary_matrix(bitstrings: list[str]) -> np.ndarray:
    # Convert list of '0101...' strings -> (N, L) uint8 matrix
    L = len(bitstrings[0])
    X = np.empty((len(bitstrings), L), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        # assume s is valid binary string
        X[i] = np.fromiter((1 if c == "1" else 0 for c in s), count=L, dtype=np.uint8)
    return X


def compute_embedding_tsne(X: np.ndarray, random_state: int = 0) -> np.ndarray:
    n = X.shape[0]
    # reasonable perplexity bound
    perplexity = min(50, max(5, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        metric="hamming",
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        n_iter=2000,
        random_state=random_state,
        verbose=1,
    )
    Z = tsne.fit_transform(X)
    return Z


def plot_density(
    Z: np.ndarray,
    weights: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
):
    H, xedges, yedges = np.histogram2d(
        Z[:, 0], Z[:, 1],
        bins=bins,
        range=[list(xlim), list(ylim)],
        weights=weights,
    )
    # log density for visibility; weights already sum to 1 within each label if you pass normalized
    D = np.log1p(H)

    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(
        D.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )
    plt.colorbar(label="log(1 + density)")
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
):
    Hm, xedges, yedges = np.histogram2d(
        Z_mut[:, 0], Z_mut[:, 1],
        bins=bins,
        range=[list(xlim), list(ylim)],
        weights=w_mut,
    )
    Hw, _, _ = np.histogram2d(
        Z_wt[:, 0], Z_wt[:, 1],
        bins=bins,
        range=[list(xlim), list(ylim)],
        weights=w_wt,
    )
    # normalize each to sum 1 so the diff is comparable
    if Hm.sum() > 0:
        Hm = Hm / Hm.sum()
    if Hw.sum() > 0:
        Hw = Hw / Hw.sum()

    D = Hm - Hw
    vmax = np.max(np.abs(D)) if np.any(D) else 1.0

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
    plt.colorbar(label="density(mut) - density(WT)")
    plt.title(title)
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_scatter_all(
    Z_all: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    out_png: Path,
    out_pdf: Path,
    title: str,
):
    plt.figure(figsize=(7.2, 6.0))
    # Use alpha scaled by weight to hint density without forcing colors too much
    # (Still keeps everything in one shared coordinate system.)
    for lab in ["WT", "G12C", "G12D"]:
        m = labels == lab
        if not np.any(m):
            continue
        # alpha mapping
        w = weights[m]
        a = np.clip(0.05 + 0.95 * (w / (w.max() + 1e-12)), 0.05, 1.0)
        plt.scatter(Z_all[m, 0], Z_all[m, 1], s=6, alpha=float(np.median(a)), label=lab)

    plt.title(title)
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder, e.g., KRAS_sampling_results")
    ap.add_argument("--out", type=str, default=None, help="Output folder (default: <root>/plots_A)")
    ap.add_argument("--max_points_per_set", type=int, default=2000, help="Max unique bitstrings per set (after aggregation)")
    ap.add_argument("--bins", type=int, default=220, help="Bins for density heatmap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else (root / "plots_A")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read + aggregate each set
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
        agg = weighted_subsample(agg, max_points=args.max_points_per_set, seed=args.seed)

        if agg.empty:
            raise RuntimeError(f"No valid rows after filtering in: {folder}")

        # ensure same bitstring length
        L = len(agg.loc[0, "bitstring"])
        if bitlen is None:
            bitlen = L
        elif L != bitlen:
            raise ValueError(f"Bitstring length mismatch: got {L}, expected {bitlen} (label={label})")

        per_set[label] = agg
        print(f"[{label}] files={len(files)} unique_bitstrings={len(agg)}")

    # 2) Pool and embed once
    all_rows = []
    for label, agg in per_set.items():
        tmp = agg.copy()
        tmp["label"] = label
        all_rows.append(tmp)
    pooled = pd.concat(all_rows, ignore_index=True)

    X = bitstrings_to_binary_matrix(pooled["bitstring"].tolist())
    labels = pooled["label"].to_numpy()
    weights = pooled["weight"].to_numpy(dtype=float)

    print(f"[POOL] total_unique_points={len(pooled)} bitlen={X.shape[1]}")

    Z = compute_embedding_tsne(X, random_state=args.seed)

    # shared axis limits
    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # 3) Plot density per label
    Z_by = {lab: Z[labels == lab] for lab in TARGET_DIRS.keys()}
    w_by = {lab: weights[labels == lab] for lab in TARGET_DIRS.keys()}

    # normalize weights inside each label for comparable density color scale
    for lab in w_by:
        s = float(w_by[lab].sum())
        if s > 0:
            w_by[lab] = w_by[lab] / s

    for lab in ["WT", "G12C", "G12D"]:
        plot_density(
            Z_by[lab],
            w_by[lab],
            out_png=out_dir / f"density_{lab}.png",
            out_pdf=out_dir / f"density_{lab}.pdf",
            title=f"KRAS structure-space sampling density ({lab})",
            bins=args.bins,
            xlim=xlim,
            ylim=ylim,
        )

    # 4) Difference maps (mut - WT)
    plot_diff_density(
        Z_mut=Z_by["G12C"], w_mut=w_by["G12C"],
        Z_wt=Z_by["WT"], w_wt=w_by["WT"],
        out_png=out_dir / "diff_G12C_minus_WT.png",
        out_pdf=out_dir / "diff_G12C_minus_WT.pdf",
        title="KRAS density difference: G12C − WT",
        bins=args.bins, xlim=xlim, ylim=ylim,
    )
    plot_diff_density(
        Z_mut=Z_by["G12D"], w_mut=w_by["G12D"],
        Z_wt=Z_by["WT"], w_wt=w_by["WT"],
        out_png=out_dir / "diff_G12D_minus_WT.png",
        out_pdf=out_dir / "diff_G12D_minus_WT.pdf",
        title="KRAS density difference: G12D − WT",
        bins=args.bins, xlim=xlim, ylim=ylim,
    )

    # 5) One combined scatter (quick sanity check)
    plot_scatter_all(
        Z_all=Z,
        labels=labels,
        weights=weights,
        out_png=out_dir / "scatter_all.png",
        out_pdf=out_dir / "scatter_all.pdf",
        title="KRAS pooled structure-space embedding (t-SNE, Hamming)",
    )

    # Save embedding table for reuse
    emb = pooled.copy()
    emb["z1"] = Z[:, 0]
    emb["z2"] = Z[:, 1]
    emb.to_csv(out_dir / "embedding_points.csv", index=False)
    print(f"[DONE] saved plots + embedding_points.csv to: {out_dir}")


if __name__ == "__main__":
    main()
