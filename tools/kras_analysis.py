# --*-- conding:utf-8 --*--
# @time:1/6/26 21:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_analysis.py

# Closed-loop KRAS analysis using:
# 1) Sampling csvs (bitstring + prob/count) in KRAS_sampling_results/
# 2) Postprocess jsonl in pp_result/<fragment_id>/<inner_dir>/
#    - decoded.jsonl (positions)
#    - energies.jsonl (E_total + components)
#    - features.jsonl (Rg, contact_density, etc.)
#    - backbone_rmsd.jsonl (rmsd, scale_factor)
#
# Outputs:
# <project_root>/KRAS_sampling_results/analysis_closed_loop/
#   embedding_points.csv
#   merged_points.csv
#   metrics.json
#   metrics_bootstrap.csv (optional)
#   metrics_stability_by_seed.csv (optional)
#   basin_assignments.csv
#   basin_occupancy.csv
#   basin_stats.csv
#   representatives.csv
#   representative_structures/*.pdb
#   plots/*.png + *.pdf
#
# Run: click Run in IDE.

from __future__ import annotations

import json
import math
import inspect
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Iterable, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


# -----------------------------
# Fixed metadata (folders)
# -----------------------------
TARGET_DIRS = {
    "WT":   "KRAS_4LPK_WT_1",
    "G12C": "KRAS_6OIM_G12C_1",
    "G12D": "KRAS_9C41_G12D_1",
}

# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
MAX_POINTS_PER_SET = 2000      # max unique bitstrings per label for embedding/plots
SEED = 0                       # embedding seed
BINS = 220                     # 2D histogram bins
SMOOTH_SIGMA = 1.2             # smoothing sigma in bin units

# Metrics
ENABLE_BOOTSTRAP = True
BOOTSTRAP_REPS = 200
BOOTSTRAP_SAMPLE_N = 4000
BOOTSTRAP_SEED = 123

ENABLE_STABILITY = True
STABILITY_SEEDS = [0, 1, 2, 3, 4]

# Basin analysis
ENABLE_BASINS = True
GMM_K_MIN = 2
GMM_K_MAX = 10
GMM_COV = "full"
GMM_SEED = 0

# Representative export
DEFAULT_SCALE_FACTOR = 3.8
TOP_REP_PER_BASIN = 1          # export top-N reps per basin (global)

SAVE_PDF = True


# -----------------------------
# Helpers: paths
# -----------------------------
def project_root_from_tools_dir() -> Path:
    # tools/kras_closed_loop_analysis.py -> tools -> project_root
    return Path(__file__).resolve().parent.parent


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Helpers: IO
# -----------------------------
def find_group_csvs(folder: Path) -> List[Path]:
    files: List[Path] = []
    for g in range(10):
        files.extend(sorted(folder.glob(f"samples_*_group{g}_ibm.csv")))
    return files


def read_sampling_csvs(files: List[Path]) -> pd.DataFrame:
    """
    Read csvs and return minimal DF: [bitstring, weight].
    weight uses prob if present else count.
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


def iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl_by_bitstring(path: Path, keep: Set[str]) -> Dict[str, dict]:
    """
    Load jsonl into dict keyed by bitstring, only keeping bitstrings in 'keep'.
    If duplicates exist, last one wins.
    """
    out: Dict[str, dict] = {}
    for obj in iter_jsonl(path):
        bs = obj.get("bitstring")
        if isinstance(bs, str) and bs in keep:
            out[bs] = obj
    return out


# -----------------------------
# Postprocess path resolution
# -----------------------------
def resolve_pp_inner_dir(pp_fragment_dir: Path) -> Path:
    """
    pp_result/<fragment_id>/ contains one inner dir like 4LPK_WT or 6OIM_G12C.
    We'll pick the first directory inside.
    """
    if not pp_fragment_dir.exists():
        raise FileNotFoundError(f"Missing pp_result fragment dir: {pp_fragment_dir}")
    candidates = [p for p in pp_fragment_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No inner dir under: {pp_fragment_dir}")
    # deterministic
    return sorted(candidates, key=lambda x: x.name)[0]


def resolve_postprocess_files(project_root: Path, fragment_id: str) -> Dict[str, Path]:
    """
    Returns paths to decoded/energies/features/backbone_rmsd for one fragment_id.
    """
    pp_fragment = project_root / "pp_result" / fragment_id
    inner = resolve_pp_inner_dir(pp_fragment)
    files = {
        "inner_dir": inner,
        "decoded": inner / "decoded.jsonl",
        "energies": inner / "energies.jsonl",
        "features": inner / "features.jsonl",
        "backbone_rmsd": inner / "backbone_rmsd.jsonl",
    }
    for k in ["decoded", "energies", "features", "backbone_rmsd"]:
        if not files[k].exists():
            raise FileNotFoundError(f"Missing file for {fragment_id}: {files[k]}")
    return files


# -----------------------------
# Embedding
# -----------------------------
def bitstrings_to_binary_matrix(bitstrings: List[str]) -> np.ndarray:
    L = len(bitstrings[0])
    X = np.empty((len(bitstrings), L), dtype=np.uint8)
    for i, s in enumerate(bitstrings):
        X[i] = np.fromiter((1 if c == "1" else 0 for c in s), count=L, dtype=np.uint8)
    return X


def compute_embedding_tsne(X: np.ndarray, seed: int) -> np.ndarray:
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

    return TSNE(**kwargs).fit_transform(X)


# -----------------------------
# Density + metrics
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def jsd_bits_from_grids(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    p = P.astype(float).ravel() + eps
    q = Q.astype(float).ravel() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    jsd_nats = 0.5 * (kl(p, m) + kl(q, m))
    return jsd_nats / np.log(2.0)


def tv_distance_from_grids(P: np.ndarray, Q: np.ndarray) -> float:
    p = P.astype(float).ravel()
    q = Q.astype(float).ravel()
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return 0.5 * float(np.sum(np.abs(p - q)))


# -----------------------------
# Plotting helpers
# -----------------------------
def save_fig(base: Path):
    plt.tight_layout()
    plt.savefig(str(base.with_suffix(".png")), dpi=300)
    if SAVE_PDF:
        plt.savefig(str(base.with_suffix(".pdf")))
    plt.close()


def plot_density_map(H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, title: str, out_base: Path):
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
    save_fig(out_base)


def plot_diff_map(D: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, title: str, out_base: Path):
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
    save_fig(out_base)


def plot_scatter_labels(Z: np.ndarray, labels: np.ndarray, out_base: Path, title: str):
    plt.figure(figsize=(7.2, 6.0))
    for lab in ["WT", "G12C", "G12D"]:
        m = labels == lab
        if np.any(m):
            plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=lab)
    plt.title(title)
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.legend(frameon=False)
    save_fig(out_base)


def plot_weighted_hist_overlay(
    df: pd.DataFrame,
    col: str,
    out_base: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
):
    plt.figure(figsize=(7.2, 4.8))
    for lab in ["WT", "G12C", "G12D"]:
        sub = df[df["label"] == lab]
        x = pd.to_numeric(sub[col], errors="coerce").to_numpy()
        w = pd.to_numeric(sub["p_mass"], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x = x[m]
        w = w[m]
        if len(x) == 0:
            continue
        # weighted histogram as density
        hist, edges = np.histogram(x, bins=bins, weights=w, density=False)
        hist = hist / (hist.sum() + 1e-12)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("probability mass (binned)")
    plt.legend(frameon=False)
    save_fig(out_base)


def plot_basin_occupancy(occ_df: pd.DataFrame, out_base: Path):
    df = occ_df.sort_values("basin_id").reset_index(drop=True)
    x = np.arange(len(df))
    width = 0.28
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(x - width, df["WT"], width, label="WT")
    plt.bar(x, df["G12C"], width, label="G12C")
    plt.bar(x + width, df["G12D"], width, label="G12D")
    plt.xticks(x, df["basin_id"].astype(int).tolist())
    plt.ylabel("probability mass")
    plt.xlabel("basin id")
    plt.title("Basin occupancy (probability mass) by variant")
    plt.legend(frameon=False)
    save_fig(out_base)


# -----------------------------
# Bootstrap + stability
# -----------------------------
def weighted_resample_indices(rng: np.random.Generator, p: np.ndarray, n: int) -> np.ndarray:
    p = p.astype(float)
    p = p / (p.sum() + 1e-12)
    return rng.choice(len(p), size=int(n), replace=True, p=p)


def bootstrap_metrics_on_embedding(
    Z_by: Dict[str, np.ndarray],
    w_by: Dict[str, np.ndarray],
    bins: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    smooth_sigma: float,
    reps: int,
    sample_n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(reps):
        Zs = {}
        ws = {}
        for lab in ["WT", "G12C", "G12D"]:
            idx = weighted_resample_indices(rng, w_by[lab], sample_n)
            Zs[lab] = Z_by[lab][idx]
            ws[lab] = np.ones(len(idx), dtype=float) / max(1, len(idx))

        H_wt, _, _ = hist2d_weighted(Zs["WT"], ws["WT"], bins, xlim, ylim, smooth_sigma, True)
        H_c, _, _  = hist2d_weighted(Zs["G12C"], ws["G12C"], bins, xlim, ylim, smooth_sigma, True)
        H_d, _, _  = hist2d_weighted(Zs["G12D"], ws["G12D"], bins, xlim, ylim, smooth_sigma, True)

        rows.append({
            "rep": r,
            "JSD_bits_G12C_vs_WT": jsd_bits_from_grids(H_c, H_wt),
            "JSD_bits_G12D_vs_WT": jsd_bits_from_grids(H_d, H_wt),
            "TV_G12C_vs_WT": tv_distance_from_grids(H_c, H_wt),
            "TV_G12D_vs_WT": tv_distance_from_grids(H_d, H_wt),
        })
    return pd.DataFrame(rows)


def ci95(x: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# -----------------------------
# Basin analysis
# -----------------------------
def fit_gmm_bic(Z: np.ndarray, kmin: int, kmax: int, cov: str, seed: int) -> Tuple[GaussianMixture, pd.DataFrame]:
    records = []
    best = None
    best_bic = None
    for k in range(kmin, kmax + 1):
        gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=seed, reg_covar=1e-6)
        gmm.fit(Z)
        bic = gmm.bic(Z)
        records.append({"K": k, "BIC": float(bic)})
        if best is None or bic < best_bic:
            best = gmm
            best_bic = bic
    assert best is not None
    return best, pd.DataFrame(records).sort_values("K")


def basin_occupancy(labels: np.ndarray, weights: np.ndarray, basin_id: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"label": labels, "weight": weights, "basin_id": basin_id})
    out = []
    for lab in ["WT", "G12C", "G12D"]:
        sub = df[df["label"] == lab].copy()
        s = float(sub["weight"].sum())
        if s > 0:
            sub["weight"] = sub["weight"] / s
        occ = sub.groupby("basin_id", as_index=False)["weight"].sum().rename(columns={"weight": lab})
        out.append(occ)

    merged = out[0]
    for t in out[1:]:
        merged = merged.merge(t, on="basin_id", how="outer")
    merged = merged.fillna(0.0)
    return merged.sort_values("basin_id").reset_index(drop=True)


def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]
    w = w[m]
    if len(x) == 0:
        return float("nan"), float("nan")
    w = w / (w.sum() + 1e-12)
    mu = float(np.sum(w * x))
    var = float(np.sum(w * (x - mu) ** 2))
    return mu, math.sqrt(max(0.0, var))


# -----------------------------
# Representative structure export (CA-only PDB)
# -----------------------------
def write_ca_pdb(
    out_path: Path,
    sequence: str,
    positions: List[List[float]],
    scale_factor: float,
    chain_id: str = "A",
):
    # CA atoms only, residue numbering starts at 1
    # positions length should match len(sequence)
    n = min(len(sequence), len(positions))
    with open(out_path, "w", encoding="utf-8") as f:
        atom_id = 1
        for i in range(n):
            resi = i + 1
            aa = sequence[i]
            # simple 3-letter mapping (fallback GLY)
            aa3 = {
                "A": "ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN","G":"GLY",
                "H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER",
                "T":"THR","W":"TRP","Y":"TYR","V":"VAL"
            }.get(aa, "GLY")
            x, y, z = (scale_factor * float(positions[i][0]),
                       scale_factor * float(positions[i][1]),
                       scale_factor * float(positions[i][2]))
            f.write(
                f"ATOM  {atom_id:5d}  CA  {aa3:>3s} {chain_id}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            atom_id += 1
        f.write("END\n")


# -----------------------------
# Main
# -----------------------------
def main():
    proj_root = project_root_from_tools_dir()
    sampling_root = proj_root / "KRAS_sampling_results"
    if not sampling_root.exists():
        raise FileNotFoundError(f"Cannot find KRAS_sampling_results at: {sampling_root}")

    out_dir = ensure_dir(sampling_root / "analysis_closed_loop")
    plot_dir = ensure_dir(out_dir / "plots")
    rep_dir = ensure_dir(out_dir / "representative_structures")

    # 1) load sampling (aggregate + subsample) for embedding
    per_set: Dict[str, pd.DataFrame] = {}
    bitlen: Optional[int] = None
    fragment_id_by_label: Dict[str, str] = {}

    for label, folder_name in TARGET_DIRS.items():
        folder = sampling_root / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing sampling folder: {folder}")

        files = find_group_csvs(folder)
        if not files:
            raise FileNotFoundError(f"No group0..9 sampling CSVs under: {folder}")

        df = read_sampling_csvs(files)
        agg = aggregate_bitstrings(df)
        agg = weighted_subsample(agg, max_points=MAX_POINTS_PER_SET, seed=SEED)

        if agg.empty:
            raise RuntimeError(f"No valid sampling rows after filtering for: {label}")

        L = len(agg.loc[0, "bitstring"])
        if bitlen is None:
            bitlen = L
        elif L != bitlen:
            raise ValueError(f"Bitstring length mismatch: got {L}, expected {bitlen} (label={label})")

        per_set[label] = agg
        fragment_id_by_label[label] = folder_name.replace("KRAS_", "")
        print(f"[SAMPLING {label}] unique_bitstrings={len(agg)} fragment_id={fragment_id_by_label[label]}")

    pooled = pd.concat([df.assign(label=lab) for lab, df in per_set.items()], ignore_index=True)
    all_bitstrings = set(pooled["bitstring"].astype(str).tolist())

    # 2) load postprocess jsonl (energies/features/rmsd) only for needed bitstrings
    energies_map: Dict[str, dict] = {}
    features_map: Dict[str, dict] = {}
    rmsd_map: Dict[str, dict] = {}
    # decoded is large; load later only for representatives

    for label in ["WT", "G12C", "G12D"]:
        fragment_id = fragment_id_by_label[label]
        files = resolve_postprocess_files(proj_root, fragment_id)
        print(f"[POST {label}] inner_dir={files['inner_dir']}")

        # keep only bitstrings belonging to this label
        keep = set(per_set[label]["bitstring"].astype(str).tolist())

        energies_map.update(load_jsonl_by_bitstring(files["energies"], keep))
        features_map.update(load_jsonl_by_bitstring(files["features"], keep))
        rmsd_map.update(load_jsonl_by_bitstring(files["backbone_rmsd"], keep))

    # 3) pooled t-SNE
    X = bitstrings_to_binary_matrix(pooled["bitstring"].tolist())
    labels = pooled["label"].to_numpy()
    weights = pooled["weight"].to_numpy(dtype=float)

    print(f"[POOL] N={len(pooled)} bitlen={X.shape[1]}")
    Z = compute_embedding_tsne(X, seed=SEED)

    emb = pooled.copy()
    emb["z1"] = Z[:, 0]
    emb["z2"] = Z[:, 1]
    emb.to_csv(out_dir / "embedding_points.csv", index=False)

    # shared limits
    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    Z_by = {lab: Z[labels == lab] for lab in TARGET_DIRS.keys()}
    w_by = {lab: weights[labels == lab].copy() for lab in TARGET_DIRS.keys()}
    for lab in w_by:
        s = float(w_by[lab].sum())
        if s > 0:
            w_by[lab] = w_by[lab] / s

    # 4) density grids + metrics
    H_wt, xedges, yedges = hist2d_weighted(Z_by["WT"], w_by["WT"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
    H_c,  _, _          = hist2d_weighted(Z_by["G12C"], w_by["G12C"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
    H_d,  _, _          = hist2d_weighted(Z_by["G12D"], w_by["G12D"], BINS, xlim, ylim, SMOOTH_SIGMA, True)

    metrics = {
        "embedding": {"method": "t-SNE", "metric": "hamming", "seed": int(SEED), "max_points_per_set": int(MAX_POINTS_PER_SET)},
        "density": {"bins": int(BINS), "smooth_sigma": float(SMOOTH_SIGMA)},
        "grid_metrics": {
            "JSD_bits_G12C_vs_WT": float(jsd_bits_from_grids(H_c, H_wt)),
            "JSD_bits_G12D_vs_WT": float(jsd_bits_from_grids(H_d, H_wt)),
            "TV_G12C_vs_WT": float(tv_distance_from_grids(H_c, H_wt)),
            "TV_G12D_vs_WT": float(tv_distance_from_grids(H_d, H_wt)),
        },
    }
    print("[METRICS]", metrics["grid_metrics"])

    # plots: density + diff + scatter
    plot_density_map(H_wt, xedges, yedges, "KRAS structure-space sampling density (WT)", plot_dir / "density_WT")
    plot_density_map(H_c,  xedges, yedges,
                     f"KRAS structure-space sampling density (G12C) | JSD={metrics['grid_metrics']['JSD_bits_G12C_vs_WT']:.3f} bits",
                     plot_dir / "density_G12C")
    plot_density_map(H_d,  xedges, yedges,
                     f"KRAS structure-space sampling density (G12D) | JSD={metrics['grid_metrics']['JSD_bits_G12D_vs_WT']:.3f} bits",
                     plot_dir / "density_G12D")
    plot_diff_map(H_c - H_wt, xedges, yedges, "KRAS density difference: G12C − WT", plot_dir / "diff_G12C_minus_WT")
    plot_diff_map(H_d - H_wt, xedges, yedges, "KRAS density difference: G12D − WT", plot_dir / "diff_G12D_minus_WT")
    plot_scatter_labels(Z, labels, plot_dir / "scatter_all", "KRAS pooled structure-space embedding (t-SNE, Hamming)")

    # 5) merge sampling points with energies/features/rmsd
    merged_rows = []
    missing = {"energies": 0, "features": 0, "rmsd": 0}
    for _, row in emb.iterrows():
        bs = str(row["bitstring"])
        lab = str(row["label"])
        out = dict(row)

        e = energies_map.get(bs)
        if e is None:
            missing["energies"] += 1
        else:
            # keep all E_* keys
            for k, v in e.items():
                if k.startswith("E_"):
                    out[k] = v

        ft = features_map.get(bs)
        if ft is None:
            missing["features"] += 1
        else:
            for k, v in ft.items():
                if k not in ("bitstring",):
                    out[f"feat_{k}"] = v

        rr = rmsd_map.get(bs)
        if rr is None:
            missing["rmsd"] += 1
        else:
            out["backbone_rmsd"] = rr.get("rmsd")
            out["scale_factor"] = rr.get("scale_factor")

        merged_rows.append(out)

    merged = pd.DataFrame(merged_rows)
    # normalize prob mass within each label for scalar distribution plots
    merged["p_mass"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    for lab in ["WT", "G12C", "G12D"]:
        m = merged["label"] == lab
        s = float(merged.loc[m, "p_mass"].sum())
        if s > 0:
            merged.loc[m, "p_mass"] = merged.loc[m, "p_mass"] / s

    merged.to_csv(out_dir / "merged_points.csv", index=False)
    print("[MERGE] missing:", missing)

    # 6) energy / feature / rmsd distribution overlays (weighted)
    # choose a small set of key columns (you can add more)
    key_cols = [
        ("E_total", "Total energy"),
        ("E_steric", "Steric energy"),
        ("E_geom", "Geom energy"),
        ("E_mj", "Miyazawa-Jernigan energy"),
        ("backbone_rmsd", "Backbone RMSD"),
        ("feat_Rg", "Radius of gyration (Rg)"),
        ("feat_contact_density", "Contact density"),
        ("feat_clash_count", "Clash count"),
        ("feat_packing_rep_count", "Packing repulsive count"),
        ("feat_rama_allowed_ratio", "Rama allowed ratio"),
    ]
    for col, title in key_cols:
        if col in merged.columns:
            plot_weighted_hist_overlay(
                merged, col,
                plot_dir / f"dist_{col}",
                title=f"Weighted distribution: {title}",
                xlabel=title,
                bins=60,
            )

    # 7) basin analysis on embedding + basin occupancy shifts + basin stats
    basin_id = None
    occ_df = None
    basin_stats = None
    reps_df = None

    if ENABLE_BASINS:
        gmm, bic_df = fit_gmm_bic(Z, GMM_K_MIN, GMM_K_MAX, GMM_COV, GMM_SEED)
        K = int(gmm.n_components)
        basin_id = gmm.predict(Z).astype(int)
        bic_df.to_csv(out_dir / "gmm_bic.csv", index=False)
        metrics["basin"] = {"method": "GMM", "covariance_type": GMM_COV, "K": K, "seed": int(GMM_SEED)}

        # assignments
        basin_assign = emb.copy()
        basin_assign["basin_id"] = basin_id
        basin_assign.to_csv(out_dir / "basin_assignments.csv", index=False)

        # occupancy
        occ_df = basin_occupancy(labels, weights, basin_id)
        occ_df.to_csv(out_dir / "basin_occupancy.csv", index=False)
        plot_basin_occupancy(occ_df, plot_dir / "basin_occupancy")

        # basin stats: weighted mean/std of energies/features/rmsd by (label, basin)
        stat_rows = []
        for lab in ["WT", "G12C", "G12D"]:
            m_lab = merged["label"] == lab
            for bid in sorted(merged.loc[m_lab, "basin_id"].dropna().unique()):
                m = m_lab & (merged["basin_id"] == bid)
                w = merged.loc[m, "p_mass"].to_numpy(dtype=float)
                rec = {"label": lab, "basin_id": int(bid), "mass": float(w.sum())}

                for col, _ in key_cols:
                    if col in merged.columns:
                        x = pd.to_numeric(merged.loc[m, col], errors="coerce").to_numpy(dtype=float)
                        mu, sd = weighted_mean_std(x, w)
                        rec[f"{col}_mean"] = mu
                        rec[f"{col}_std"] = sd
                stat_rows.append(rec)

        basin_stats = pd.DataFrame(stat_rows).sort_values(["label", "basin_id"])
        basin_stats.to_csv(out_dir / "basin_stats.csv", index=False)

        # representatives: top weight per basin (global and per label)
        reps = []
        tmp = merged.copy()
        tmp["basin_id"] = basin_id
        # global top per basin
        for bid, sub in tmp.groupby("basin_id"):
            sub2 = sub.sort_values("weight", ascending=False).head(TOP_REP_PER_BASIN)
            for _, r in sub2.iterrows():
                reps.append({
                    "scope": "global",
                    "basin_id": int(bid),
                    "label": str(r["label"]),
                    "bitstring": str(r["bitstring"]),
                    "weight": float(r["weight"]),
                    "p_mass": float(r["p_mass"]),
                    "z1": float(r["z1"]),
                    "z2": float(r["z2"]),
                    "E_total": float(r.get("E_total", np.nan)),
                    "backbone_rmsd": float(r.get("backbone_rmsd", np.nan)),
                    "scale_factor": float(r.get("scale_factor", np.nan)) if pd.notna(r.get("scale_factor", np.nan)) else np.nan,
                })
        reps_df = pd.DataFrame(reps).sort_values(["basin_id", "scope", "weight"], ascending=[True, True, False])
        reps_df.to_csv(out_dir / "representatives.csv", index=False)

    # 8) Bootstrap CIs
    if ENABLE_BOOTSTRAP:
        print(f"[BOOT] reps={BOOTSTRAP_REPS} sample_n={BOOTSTRAP_SAMPLE_N} ...")
        boot_df = bootstrap_metrics_on_embedding(
            Z_by=Z_by, w_by=w_by,
            bins=BINS, xlim=xlim, ylim=ylim,
            smooth_sigma=SMOOTH_SIGMA,
            reps=BOOTSTRAP_REPS,
            sample_n=BOOTSTRAP_SAMPLE_N,
            seed=BOOTSTRAP_SEED,
        )
        boot_df.to_csv(out_dir / "metrics_bootstrap.csv", index=False)
        metrics["bootstrap_ci95"] = {
            col: {"lo": ci95(boot_df[col].to_numpy())[0], "hi": ci95(boot_df[col].to_numpy())[1]}
            for col in ["JSD_bits_G12C_vs_WT", "JSD_bits_G12D_vs_WT", "TV_G12C_vs_WT", "TV_G12D_vs_WT"]
        }

    # 9) Stability across embedding seeds
    if ENABLE_STABILITY:
        rows = []
        for sd in STABILITY_SEEDS:
            print(f"[STABILITY] embedding seed={sd} ...")
            Z_sd = compute_embedding_tsne(X, seed=sd)
            Z_by_sd = {lab: Z_sd[labels == lab] for lab in TARGET_DIRS.keys()}
            H_wt_sd, _, _ = hist2d_weighted(Z_by_sd["WT"], w_by["WT"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
            H_c_sd,  _, _ = hist2d_weighted(Z_by_sd["G12C"], w_by["G12C"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
            H_d_sd,  _, _ = hist2d_weighted(Z_by_sd["G12D"], w_by["G12D"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
            rows.append({
                "seed": sd,
                "JSD_bits_G12C_vs_WT": jsd_bits_from_grids(H_c_sd, H_wt_sd),
                "JSD_bits_G12D_vs_WT": jsd_bits_from_grids(H_d_sd, H_wt_sd),
                "TV_G12C_vs_WT": tv_distance_from_grids(H_c_sd, H_wt_sd),
                "TV_G12D_vs_WT": tv_distance_from_grids(H_d_sd, H_wt_sd),
            })
        stab = pd.DataFrame(rows)
        stab.to_csv(out_dir / "metrics_stability_by_seed.csv", index=False)
        metrics["stability_seeds"] = STABILITY_SEEDS

    # 10) Export representative PDBs (CA-only) using decoded.jsonl
    if ENABLE_BASINS and reps_df is not None and len(reps_df) > 0:
        # Collect all needed bitstrings to decode
        need_decode: Dict[str, Set[str]] = {"WT": set(), "G12C": set(), "G12D": set()}
        for _, r in reps_df.iterrows():
            need_decode[str(r["label"])].add(str(r["bitstring"]))

        for lab in ["WT", "G12C", "G12D"]:
            fragment_id = fragment_id_by_label[lab]
            files = resolve_postprocess_files(proj_root, fragment_id)
            keep = need_decode[lab]
            if not keep:
                continue

            decoded_map = load_jsonl_by_bitstring(files["decoded"], keep)
            # need sequence; try from decoded or energies_map
            for bs in keep:
                dec = decoded_map.get(bs)
                if dec is None:
                    continue
                seq = dec.get("sequence") or energies_map.get(bs, {}).get("sequence") or "G" * len(dec.get("main_positions", []))
                positions = dec.get("main_positions", [])
                # scale factor: prefer rmsd_map if exists
                sf = rmsd_map.get(bs, {}).get("scale_factor", DEFAULT_SCALE_FACTOR)
                try:
                    sf = float(sf)
                except Exception:
                    sf = DEFAULT_SCALE_FACTOR

                # find basin id
                bid = None
                sub = reps_df[(reps_df["label"] == lab) & (reps_df["bitstring"] == bs)]
                if len(sub) > 0:
                    bid = int(sub.iloc[0]["basin_id"])
                else:
                    bid = -1

                out_pdb = rep_dir / f"{lab}_basin{bid}_wtop.pdb"
                write_ca_pdb(out_pdb, str(seq), positions, sf, chain_id="A")

        print(f"[REP] Exported CA-only PDBs to: {rep_dir}")

    # Save metrics.json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Outputs saved to:\n  {out_dir}")


if __name__ == "__main__":
    main()
