# --*-- conding:utf-8 --*--
# @time:12/19/25 00:27
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_sampling.py


# It assumes your folder layout is:
# <project_root>/
#   KRAS_sampling_results/
#     KRAS_4LPK_WT_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#     KRAS_6OIM_G12C_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#     KRAS_9C41_G12D_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#   tool/
#     plot_kras_sampling_density_A.py   <-- this file
#
# Outputs:
# <project_root>/KRAS_sampling_results/plots_A/
#   density_WT.png/pdf, density_G12C.png/pdf, density_G12D.png/pdf
#   diff_G12C_minus_WT.png/pdf, diff_G12D_minus_WT.png/pdf
#   scatter_all.png/pdf
#   embedding_points.csv

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


# -----------------------------
# Fixed config (your metadata)
# -----------------------------
TARGET_DIRS = {
    "WT": "KRAS_4LPK_WT_1",
    "G12C": "KRAS_6OIM_G12C_1",
    "G12D": "KRAS_9C41_G12D_1",
}

# Default params (safe for IDE-click run)
MAX_POINTS_PER_SET = 2000   # max unique bitstrings per label after aggregation
BINS = 220                  # heatmap bins
SEED = 0


def project_root_from_tool_dir() -> Path:
    # tool/this_file.py -> project_root/tool -> project_root
    return Path(__file__).resolve().parent.parent


def find_group_csvs(folder: Path) -> list[Path]:
    files: list[Path] = []
    for g in range(10):
        files.extend(sorted(folder.glob(f"samples_*_group{g}_ibm.csv")))
    return files


def read_sampling_csvs(files: list[Path]) -> pd.DataFrame:
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
    agg = df.groupby("bitstring", as_index=False)["weight"].sum()
    s = float(agg["weight"].sum())
    if s > 0:
        agg["weight"] = agg["weight"] / s
    return agg.sort_values("weight", ascending=False).reset_index(drop=True)


def weighted_subsample(agg: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
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
    rest_w = rest_w / rest_w.sum()

    k = max_points - keep_top
    idx = rng.choice(len(rest), size=k, replace=False, p=rest_w)
    samp = rest.iloc[idx].copy()

    out = pd.concat([top, samp], ignore_index=True)
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    out["weight"] = out["weight"] / out["weight"].sum()
    return out


def bitstrings_to_binary_matrix(bitstrings: list[str]) -> np.ndarray:
    L = len(bitstrings[0])
    X = np.empty((len(bitstrings), L), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        X[i] = np.fromiter((1 if c == "1" else 0 for c in s), count=L, dtype=np.uint8)
    return X


def compute_embedding_tsne(X: np.ndarray, seed: int) -> np.ndarray:
    n = X.shape[0]
    perplexity = min(50, max(5, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        metric="hamming",
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        n_iter=2000,
        random_state=seed,
        verbose=1,
    )
    return tsne.fit_transform(X)


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

    if Hm.sum() > 0:
        Hm = Hm / Hm.sum()
    if Hw.sum() > 0:
        Hw = Hw / Hw.sum()

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
    plt.colorbar(label="density(mut) - density(WT)")
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


def main():
    proj_root = project_root_from_tool_dir()
    root = proj_root / "KRAS_sampling_results"
    if not root.exists():
        raise FileNotFoundError(
            f"Cannot find KRAS_sampling_results at expected path:\n  {root}\n"
            f"Make sure your repo layout matches the screenshot."
        )

    out_dir = root / "plots_A"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load & aggregate
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
    pooled = pd.concat(
        [df.assign(label=lab) for lab, df in per_set.items()],
        ignore_index=True
    )
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

    # normalize inside each label so density maps are comparable
    for lab in w_by:
        s = float(w_by[lab].sum())
        if s > 0:
            w_by[lab] = w_by[lab] / s

    # 3) density plots
    for lab in ["WT", "G12C", "G12D"]:
        plot_density(
            Z_by[lab],
            w_by[lab],
            out_png=out_dir / f"density_{lab}.png",
            out_pdf=out_dir / f"density_{lab}.pdf",
            title=f"KRAS structure-space sampling density ({lab})",
            bins=BINS,
            xlim=xlim,
            ylim=ylim,
        )

    # 4) difference maps
    plot_diff_density(
        Z_mut=Z_by["G12C"], w_mut=w_by["G12C"],
        Z_wt=Z_by["WT"], w_wt=w_by["WT"],
        out_png=out_dir / "diff_G12C_minus_WT.png",
        out_pdf=out_dir / "diff_G12C_minus_WT.pdf",
        title="KRAS density difference: G12C − WT",
        bins=BINS, xlim=xlim, ylim=ylim,
    )
    plot_diff_density(
        Z_mut=Z_by["G12D"], w_mut=w_by["G12D"],
        Z_wt=Z_by["WT"], w_wt=w_by["WT"],
        out_png=out_dir / "diff_G12D_minus_WT.png",
        out_pdf=out_dir / "diff_G12D_minus_WT.pdf",
        title="KRAS density difference: G12D − WT",
        bins=BINS, xlim=xlim, ylim=ylim,
    )

    # 5) pooled scatter sanity check
    plot_scatter_all(
        Z,
        labels,
        out_png=out_dir / "scatter_all.png",
        out_pdf=out_dir / "scatter_all.pdf",
    )

    # save embedding table for later overlays (energy/RMSD/topK, etc.)
    emb = pooled.copy()
    emb["z1"] = Z[:, 0]
    emb["z2"] = Z[:, 1]
    emb.to_csv(out_dir / "embedding_points.csv", index=False)

    print(f"[DONE] Saved plots + embedding_points.csv to:\n  {out_dir}")


if __name__ == "__main__":
    main()
