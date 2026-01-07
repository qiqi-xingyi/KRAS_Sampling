# --*-- conding:utf-8 --*--
# @time:1/6/26 23:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_mechanism_addon.py

# Closed-loop KRAS analysis for quantum sampling:
# - Load sampling CSVs (group0..9) for WT / G12C / G12D
# - Load postprocessed JSONLs (energies/features/backbone_rmsd/decoded)
# - Merge by bitstring
# - Pooled embedding (t-SNE, Hamming) + density maps + difference maps
# - Grid metrics (JSD / Total Variation) + stability across seeds + bootstrap CI
# - Basin discovery (GMM on embedding) + occupancy + basin stats
# - Basin deltas + top basins (union)
# - Basin energy contrast (long table) saved to:
#     KRAS_sampling_results/analysis_closed_loop/addons/basin_energy_contrast.csv
# - Energy "waterfall" style term-contrast plots from basin_energy_contrast.csv
# - Representative structures per basin + export CA-only PDB for WT/G12C/G12D
# - Basin-pair CA RMSD (WT vs G12C, WT vs G12D)
# - Optional distribution tests (KS/Wasserstein) inside key basins
#
# Directory layout assumed (fixed, relative to project root):
# <project_root>/
#   KRAS_sampling_results/
#     KRAS_4LPK_WT_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#     KRAS_6OIM_G12C_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#     KRAS_9C41_G12D_1/
#       samples_*_group0_ibm.csv ... samples_*_group9_ibm.csv
#   pp_result/
#     4LPK_WT_1/4LPK_WT/{energies.jsonl,features.jsonl,backbone_rmsd.jsonl,decoded.jsonl}
#     6OIM_G12C_1/6OIM_G12C/{...}
#     9C41_G12D_1/9C41_G12D/{...}
#   tools/
#     kras_analysis_closed_loop.py   <-- this file
#
# Run: click Run in IDE (or python tools/kras_analysis_closed_loop.py)
#
# Outputs:
#   KRAS_sampling_results/analysis_closed_loop/
#     metrics.json
#     metrics_stability_by_seed.csv
#     embedding_points.csv
#     basin_occupancy.csv
#     basin_stats.csv
#     basin_delta_summary.csv
#     top_basins.json
#     representatives.csv
#     basin_pair_ca_rmsd.csv
#     basin_dist_tests.csv (if scipy available)
#     plots/*.png + *.pdf
#     addons/basin_energy_contrast.csv
#     addons/waterfall_terms_basinXX_*.png
#     addons/exported_pdb/basinXX_LABEL.pdb
#
from __future__ import annotations

import json
import math
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


# -----------------------------
# User-configurable section
# -----------------------------
TARGET = {
    "WT": {
        "sampling_folder": "KRAS_4LPK_WT_1",
        "fragment_id": "4LPK_WT_1",
        "pp_inner": "4LPK_WT",
    },
    "G12C": {
        "sampling_folder": "KRAS_6OIM_G12C_1",
        "fragment_id": "6OIM_G12C_1",
        "pp_inner": "6OIM_G12C",
    },
    "G12D": {
        "sampling_folder": "KRAS_9C41_G12D_1",
        "fragment_id": "9C41_G12D_1",
        "pp_inner": "9C41_G12D",
    },
}

# NOTE:
# - WT pocket may have 3 fragments, but here we ONLY analyze the mutated-pocket fragment
#   that matches G12C/G12D. So WT uses fragment_id=4LPK_WT_1.

MAX_POINTS_PER_SET = 2000
BINS = 220
SEED = 0
SMOOTH_SIGMA = 1.2

# stability check
STABILITY_SEEDS = [0, 1, 2, 3, 4]

# basin model
BASIN_K = 7
BASIN_SEED = 0

# bootstrap for grid metrics
BOOTSTRAP_N = 300   # increase if you want tighter CI, but slower
BOOTSTRAP_SEED = 0

# which basins to emphasize: top |delta occupancy| per mutant, then union
TOPK_BASINS_PER_MUT = 3

# export representative CA-only pdb
EXPORT_PDB = True

# optional: distribution tests inside "union basins"
ENABLE_DIST_TESTS = True


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    root: Path
    kras_sampling_root: Path
    pp_root: Path
    out_root: Path
    plots: Path
    addons: Path
    exported_pdb: Path


def make_paths() -> Paths:
    pr = project_root_from_tools_dir()
    kras = pr / "KRAS_sampling_results"
    pp = pr / "pp_result"
    out = kras / "analysis_closed_loop"
    plots = out / "plots"
    addons = out / "addons"   # IMPORTANT: "addons" (plural)
    exported = addons / "exported_pdb"

    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    addons.mkdir(parents=True, exist_ok=True)
    exported.mkdir(parents=True, exist_ok=True)
    return Paths(pr, kras, pp, out, plots, addons, exported)


# -----------------------------
# Sampling IO
# -----------------------------
def find_group_csvs(folder: Path) -> List[Path]:
    files: List[Path] = []
    for g in range(10):
        files.extend(sorted(folder.glob(f"samples_*_group{g}_ibm.csv")))
    return files


def read_sampling_csvs(files: List[Path]) -> pd.DataFrame:
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
    rest_w = rest_w / (rest_w.sum() + 1e-12)

    k = max_points - keep_top
    idx = rng.choice(len(rest), size=k, replace=False, p=rest_w)
    samp = rest.iloc[idx].copy()

    out = pd.concat([top, samp], ignore_index=True)
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    out["weight"] = out["weight"] / (out["weight"].sum() + 1e-12)
    return out


def load_sampling_set(paths: Paths, label: str) -> pd.DataFrame:
    folder = paths.kras_sampling_root / TARGET[label]["sampling_folder"]
    if not folder.exists():
        raise FileNotFoundError(f"Missing sampling folder: {folder}")

    files = find_group_csvs(folder)
    if not files:
        raise FileNotFoundError(f"No group0..9 CSVs found under: {folder}")

    df = read_sampling_csvs(files)
    agg = aggregate_bitstrings(df)
    agg = weighted_subsample(agg, max_points=MAX_POINTS_PER_SET, seed=SEED)

    if agg.empty:
        raise RuntimeError(f"No valid rows after sampling filtering for label={label}")

    return agg


# -----------------------------
# Postprocess IO (jsonl)
# -----------------------------
def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_post_tables(paths: Paths, label: str) -> Dict[str, pd.DataFrame]:
    frag = TARGET[label]["fragment_id"]
    inner = TARGET[label]["pp_inner"]
    inner_dir = paths.pp_root / frag / inner
    if not inner_dir.exists():
        raise FileNotFoundError(f"Missing pp_result inner dir: {inner_dir}")

    print(f"[POST {label}] inner_dir={inner_dir}")

    energies = pd.DataFrame(read_jsonl(inner_dir / "energies.jsonl"))
    features = pd.DataFrame(read_jsonl(inner_dir / "features.jsonl"))
    rmsd = pd.DataFrame(read_jsonl(inner_dir / "backbone_rmsd.jsonl"))

    # decoded (used for PDB export)
    decoded = pd.DataFrame(read_jsonl(inner_dir / "decoded.jsonl"))

    # normalize columns
    for df in [energies, features, rmsd, decoded]:
        if "bitstring" in df.columns:
            df["bitstring"] = df["bitstring"].astype(str)

    # ensure rmsd col is named consistently
    if "rmsd" in rmsd.columns and "backbone_rmsd" not in rmsd.columns:
        rmsd = rmsd.rename(columns={"rmsd": "backbone_rmsd"})

    return {
        "energies": energies,
        "features": features,
        "rmsd": rmsd,
        "decoded": decoded,
    }


def merge_sampling_and_post(
    label: str,
    sampling: pd.DataFrame,
    post: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = sampling.copy()
    out["label"] = label
    out["fragment_id"] = TARGET[label]["fragment_id"]

    miss = {"energies": 0, "features": 0, "rmsd": 0}

    # merge energies
    if "energies" in post and not post["energies"].empty:
        cols = [c for c in post["energies"].columns if c not in ("sequence", "fifth_bit", "main_vectors", "side_vectors", "main_positions", "side_positions")]
        out = out.merge(post["energies"][cols], on="bitstring", how="left")
        miss["energies"] = int(out["E_total"].isna().sum()) if "E_total" in out.columns else int(out.isna().any(axis=1).sum())

    # merge features
    if "features" in post and not post["features"].empty:
        cols = [c for c in post["features"].columns if c not in ("sequence",)]
        out = out.merge(post["features"][cols], on="bitstring", how="left")
        miss["features"] = int(out["length"].isna().sum()) if "length" in out.columns else int(out.isna().any(axis=1).sum())

    # merge rmsd
    if "rmsd" in post and not post["rmsd"].empty:
        cols = [c for c in post["rmsd"].columns if c not in ("sequence", "decoded_file", "line_index")]
        out = out.merge(post["rmsd"][cols], on="bitstring", how="left")
        miss["rmsd"] = int(out["backbone_rmsd"].isna().sum()) if "backbone_rmsd" in out.columns else int(out.isna().any(axis=1).sum())

    print(f"[MERGE] {label} missing: {miss}")
    return out


# -----------------------------
# Embedding (t-SNE Hamming)
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

    tsne = TSNE(**kwargs)
    return tsne.fit_transform(X)


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


def jsd_from_grids(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    p = P.astype(float).ravel() + eps
    q = Q.astype(float).ravel() + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * (np.log(a) - np.log(b))))

    jsd_nats = 0.5 * (kl(p, m) + kl(q, m))
    return jsd_nats / np.log(2.0)


def tv_from_grids(P: np.ndarray, Q: np.ndarray) -> float:
    p = P.astype(float).ravel()
    q = Q.astype(float).ravel()
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return 0.5 * float(np.sum(np.abs(p - q)))


# -----------------------------
# Plotting helpers
# -----------------------------
def save_fig(path_png: Path, path_pdf: Path):
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.savefig(path_pdf)
    plt.close()


def plot_density_map(paths: Paths, H: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, title: str, name: str):
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
    save_fig(paths.plots / f"{name}.png", paths.plots / f"{name}.pdf")


def plot_diff_map(paths: Paths, D: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, title: str, name: str):
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
    save_fig(paths.plots / f"{name}.png", paths.plots / f"{name}.pdf")


def plot_scatter_all(paths: Paths, Z: np.ndarray, labels: np.ndarray):
    plt.figure(figsize=(7.2, 6.0))
    for lab in ["WT", "G12C", "G12D"]:
        m = labels == lab
        if np.any(m):
            plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=lab)
    plt.title("KRAS pooled structure-space embedding (t-SNE, Hamming)")
    plt.xlabel("Structure axis 1 (t-SNE, Hamming)")
    plt.ylabel("Structure axis 2 (t-SNE, Hamming)")
    plt.legend(frameon=False)
    save_fig(paths.plots / "scatter_all.png", paths.plots / "scatter_all.pdf")


# -----------------------------
# Basin analysis utilities
# -----------------------------
def weighted_mean_std(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    m = float(np.sum(w * x))
    v = float(np.sum(w * (x - m) ** 2))
    return m, float(math.sqrt(max(v, 0.0)))


def add_basin_id(emb: pd.DataFrame, K: int, seed: int) -> Tuple[pd.DataFrame, GaussianMixture]:
    Z = emb[["z1", "z2"]].to_numpy(dtype=float)
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=seed)
    gmm.fit(Z)
    emb = emb.copy()
    emb["basin_id"] = gmm.predict(Z).astype(int)
    return emb, gmm


def basin_occupancy(emb: pd.DataFrame) -> pd.DataFrame:
    # sum weights within basin per label
    piv = (
        emb.groupby(["basin_id", "label"], as_index=False)["weight"].sum()
        .pivot(index="basin_id", columns="label", values="weight")
        .fillna(0.0)
        .reset_index()
    )
    for lab in ["WT", "G12C", "G12D"]:
        if lab not in piv.columns:
            piv[lab] = 0.0
    return piv[["basin_id", "WT", "G12C", "G12D"]].sort_values("basin_id")


def compute_basin_stats(emb: pd.DataFrame) -> pd.DataFrame:
    # metrics to summarize if present
    energy_terms = [
        "E_total", "E_steric", "E_geom", "E_bond", "E_mj", "E_dihedral", "E_hydroph", "E_cbeta", "E_rama"
    ]
    feat_terms = [
        "Rg", "end_to_end", "contact_density", "clash_count", "packing_rep_count", "rama_allowed_ratio"
    ]
    rmsd_terms = ["backbone_rmsd"]
    cols = []
    for c in energy_terms + feat_terms + rmsd_terms:
        if c in emb.columns:
            cols.append(c)

    rows = []
    for (lab, bid), g in emb.groupby(["label", "basin_id"]):
        w = g["weight"].to_numpy(dtype=float)
        w = w / (w.sum() + 1e-12)
        row = {"label": lab, "basin_id": int(bid), "mass": float(g["weight"].sum())}
        for c in cols:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            msk = np.isfinite(x)
            if not np.any(msk):
                row[f"{c}_mean"] = np.nan
                row[f"{c}_std"] = np.nan
            else:
                ww = w[msk]
                ww = ww / (ww.sum() + 1e-12)
                mm, ss = weighted_mean_std(x[msk], ww)
                row[f"{c}_mean"] = mm
                row[f"{c}_std"] = ss
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["label", "basin_id"]).reset_index(drop=True)
    return df


def pick_top_basins(occ: pd.DataFrame, topk: int) -> Tuple[Dict[str, List[int]], List[int], pd.DataFrame]:
    d = occ.copy()
    d["delta_G12C_minus_WT"] = d["G12C"] - d["WT"]
    d["delta_G12D_minus_WT"] = d["G12D"] - d["WT"]
    d["abs_delta_G12C"] = np.abs(d["delta_G12C_minus_WT"])
    d["abs_delta_G12D"] = np.abs(d["delta_G12D_minus_WT"])

    d["rank_abs_G12C"] = d["abs_delta_G12C"].rank(ascending=False, method="first").astype(int)
    d["rank_abs_G12D"] = d["abs_delta_G12D"].rank(ascending=False, method="first").astype(int)

    top_c = d.sort_values("abs_delta_G12C", ascending=False)["basin_id"].head(topk).astype(int).tolist()
    top_d = d.sort_values("abs_delta_G12D", ascending=False)["basin_id"].head(topk).astype(int).tolist()
    union = sorted(set(top_c) | set(top_d))

    info = {"G12C_top": top_c, "G12D_top": top_d, "union": union}
    return info, union, d.sort_values("abs_delta_G12D", ascending=False).reset_index(drop=True)


# -----------------------------
# Energy contrast (long table) + waterfall plotting
# -----------------------------
def build_basin_energy_contrast(basin_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Return long table:
      basin_id, term, WT, G12C, G12D, delta_G12C_minus_WT, delta_G12D_minus_WT
    """
    # infer available terms from basin_stats columns
    mean_cols = [c for c in basin_stats.columns if c.endswith("_mean")]
    terms = [c[:-5] for c in mean_cols]  # drop _mean

    # pivot into wide: index=(basin_id, term) columns=label values=mean
    rows = []
    for term in terms:
        sub = basin_stats[["label", "basin_id", f"{term}_mean"]].copy()
        sub = sub.rename(columns={f"{term}_mean": "value"})
        sub["term"] = term
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)

    piv = (
        long.pivot_table(index=["basin_id", "term"], columns="label", values="value", aggfunc="mean")
        .reset_index()
        .fillna(np.nan)
    )

    for lab in ["WT", "G12C", "G12D"]:
        if lab not in piv.columns:
            piv[lab] = np.nan

    piv["delta_G12C_minus_WT"] = piv["G12C"] - piv["WT"]
    piv["delta_G12D_minus_WT"] = piv["G12D"] - piv["WT"]

    # nicer ordering: show energies first if present
    def term_priority(t: str) -> int:
        energy_order = {
            "E_total": 0, "E_steric": 1, "E_geom": 2, "E_bond": 3, "E_mj": 4, "E_dihedral": 5, "E_hydroph": 6, "E_cbeta": 7, "E_rama": 8
        }
        if t in energy_order:
            return energy_order[t]
        if t.startswith("feat_"):
            return 50
        if t.endswith("_rmsd") or "rmsd" in t:
            return 60
        return 100

    piv = piv.sort_values(["basin_id", "term"], key=lambda s: s.map(term_priority) if s.name == "term" else s)
    return piv


def find_energy_contrast_csv(paths: Paths) -> Optional[Path]:
    candidates = [
        paths.addons / "basin_energy_contrast.csv",
        paths.out_root / "addons" / "basin_energy_contrast.csv",
        paths.out_root / "addon" / "basin_energy_contrast.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: scan
    for p in paths.out_root.rglob("*.csv"):
        if p.name == "basin_energy_contrast.csv":
            return p
    return None


def plot_waterfall_from_energy_contrast(
    paths: Paths,
    df: pd.DataFrame,
    focus_basins: List[int],
):
    # only keep "energy-like" terms for waterfall plot
    # (you can widen this if you want)
    keep_terms = {
        "E_steric", "E_geom", "E_bond", "E_mj", "E_dihedral", "E_hydroph", "E_cbeta", "E_rama", "E_total"
    }
    df = df[df["term"].isin(keep_terms)].copy()

    for bid in focus_basins:
        sub = df[df["basin_id"] == bid].copy()
        if sub.empty:
            continue

        # G12D - WT
        sub_d = sub.sort_values("delta_G12D_minus_WT", key=lambda s: s.abs(), ascending=False)
        plt.figure(figsize=(8.6, 4.6))
        plt.bar(sub_d["term"], sub_d["delta_G12D_minus_WT"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("ΔE (G12D − WT)")
        plt.title(f"Energy-term contrast (basin {bid:02d}): G12D − WT")
        plt.tight_layout()
        plt.savefig(paths.addons / f"waterfall_terms_basin{bid:02d}_G12D_minus_WT.png", dpi=300)
        plt.close()

        # G12C - WT
        sub_c = sub.sort_values("delta_G12C_minus_WT", key=lambda s: s.abs(), ascending=False)
        plt.figure(figsize=(8.6, 4.6))
        plt.bar(sub_c["term"], sub_c["delta_G12C_minus_WT"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("ΔE (G12C − WT)")
        plt.title(f"Energy-term contrast (basin {bid:02d}): G12C − WT")
        plt.tight_layout()
        plt.savefig(paths.addons / f"waterfall_terms_basin{bid:02d}_G12C_minus_WT.png", dpi=300)
        plt.close()

    print(f"[OK] Energy term-contrast plots saved to: {paths.addons}")


# -----------------------------
# Representatives + PDB export + CA RMSD
# -----------------------------
def choose_representatives(emb: pd.DataFrame) -> pd.DataFrame:
    """
    Representative per (label, basin): pick max-weight bitstring.
    Also "global" representative per basin across all labels: pick max-weight overall.
    """
    rows = []

    for bid, g in emb.groupby("basin_id"):
        gg = g.sort_values("weight", ascending=False).iloc[0]
        rows.append({
            "scope": "global",
            "basin_id": int(bid),
            "label": gg["label"],
            "fragment_id": gg["fragment_id"],
            "bitstring": gg["bitstring"],
            "uid": f'{gg["label"]}|{gg["bitstring"]}',
            "weight": float(gg["weight"]),
            "p_mass": float(gg["weight"]),  # alias
            "z1": float(gg["z1"]),
            "z2": float(gg["z2"]),
            "E_total": float(gg["E_total"]) if "E_total" in gg and pd.notna(gg["E_total"]) else np.nan,
            "backbone_rmsd": float(gg["backbone_rmsd"]) if "backbone_rmsd" in gg and pd.notna(gg["backbone_rmsd"]) else np.nan,
            "scale_factor": float(gg["scale_factor"]) if "scale_factor" in gg and pd.notna(gg["scale_factor"]) else np.nan,
        })

    for (lab, bid), g in emb.groupby(["label", "basin_id"]):
        gg = g.sort_values("weight", ascending=False).iloc[0]
        rows.append({
            "scope": "within_label",
            "basin_id": int(bid),
            "label": lab,
            "fragment_id": gg["fragment_id"],
            "bitstring": gg["bitstring"],
            "uid": f'{lab}|{gg["bitstring"]}',
            "weight": float(gg["weight"]),
            "p_mass": float(g["weight"].sum()),
            "z1": float(gg["z1"]),
            "z2": float(gg["z2"]),
            "E_total": float(gg["E_total"]) if "E_total" in gg and pd.notna(gg["E_total"]) else np.nan,
            "backbone_rmsd": float(gg["backbone_rmsd"]) if "backbone_rmsd" in gg and pd.notna(gg["backbone_rmsd"]) else np.nan,
            "scale_factor": float(gg["scale_factor"]) if "scale_factor" in gg and pd.notna(gg["scale_factor"]) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["scope", "basin_id", "label"]).reset_index(drop=True)


AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}


def export_ca_only_pdb(
    out_pdb: Path,
    sequence: str,
    coords: np.ndarray,
):
    """
    Write CA-only PDB using sequence residues.
    coords: (N, 3) float
    """
    lines = []
    atom_id = 1
    for i, (x, y, z) in enumerate(coords, start=1):
        aa = AA3.get(sequence[i-1], "GLY") if i-1 < len(sequence) else "GLY"
        # ATOM formatting
        line = (
            f"ATOM  {atom_id:5d}  CA  {aa:>3s} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        lines.append(line)
        atom_id += 1
    lines.append("END")
    out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_decoded_map(decoded_df: pd.DataFrame) -> Dict[str, dict]:
    m: Dict[str, dict] = {}
    if decoded_df is None or decoded_df.empty:
        return m
    for _, r in decoded_df.iterrows():
        b = str(r.get("bitstring", ""))
        if not b:
            continue
        m[b] = dict(r)
    return m


def read_ca_coords_from_pdb(pdb_path: Path) -> np.ndarray:
    coords = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords, dtype=float)


def ca_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    n = min(len(P), len(Q))
    if n == 0:
        return float("nan")
    P = P[:n]
    Q = Q[:n]
    d = P - Q
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def export_representative_pdbs(
    paths: Paths,
    reps: pd.DataFrame,
    post_by_label: Dict[str, Dict[str, pd.DataFrame]],
):
    decoded_maps = {}
    seq_maps = {}
    for lab in ["WT", "G12C", "G12D"]:
        decoded_df = post_by_label[lab]["decoded"]
        decoded_maps[lab] = get_decoded_map(decoded_df)

        # sequence: take first available
        seq = None
        if "sequence" in decoded_df.columns and not decoded_df["sequence"].isna().all():
            seq = str(decoded_df["sequence"].dropna().iloc[0])
        seq_maps[lab] = seq or ""

    # export only within_label representatives per basin per label
    sub = reps[reps["scope"] == "within_label"].copy()
    for _, r in sub.iterrows():
        lab = r["label"]
        bid = int(r["basin_id"])
        bs = str(r["bitstring"])
        rec = decoded_maps.get(lab, {}).get(bs, None)
        if rec is None:
            continue

        seq = str(rec.get("sequence", seq_maps.get(lab, "")))
        pos = rec.get("main_positions", None)
        if pos is None:
            continue

        coords = np.array(pos, dtype=float)
        out_pdb = paths.exported_pdb / f"basin{bid:02d}_{lab}.pdb"
        export_ca_only_pdb(out_pdb, seq, coords)

    print(f"[OK] Exported CA-only representative PDBs to: {paths.exported_pdb}")


def compute_basin_pair_ca_rmsd(paths: Paths, focus_basins: List[int]) -> pd.DataFrame:
    rows = []
    for bid in focus_basins:
        p_wt = paths.exported_pdb / f"basin{bid:02d}_WT.pdb"
        p_c = paths.exported_pdb / f"basin{bid:02d}_G12C.pdb"
        p_d = paths.exported_pdb / f"basin{bid:02d}_G12D.pdb"

        if p_wt.exists() and p_c.exists():
            rmsd_wc = ca_rmsd(read_ca_coords_from_pdb(p_wt), read_ca_coords_from_pdb(p_c))
            rows.append({"basin_id": bid, "pair": "G12C-vs-WT", "ca_rmsd": rmsd_wc,
                         "WT_pdb": str(p_wt), "G12C_pdb": str(p_c), "G12D_pdb": ""})
            print(f"[OK] basin {bid:02d} WT vs G12C: CA_RMSD={rmsd_wc:.3f} Å")

        if p_wt.exists() and p_d.exists():
            rmsd_wd = ca_rmsd(read_ca_coords_from_pdb(p_wt), read_ca_coords_from_pdb(p_d))
            rows.append({"basin_id": bid, "pair": "G12D-vs-WT", "ca_rmsd": rmsd_wd,
                         "WT_pdb": str(p_wt), "G12C_pdb": "", "G12D_pdb": str(p_d)})
            print(f"[OK] basin {bid:02d} WT vs G12D: CA_RMSD={rmsd_wd:.3f} Å")

    return pd.DataFrame(rows)


# -----------------------------
# Bootstrap CI for grid metrics
# -----------------------------
def bootstrap_grid_metrics(
    Z_by: Dict[str, np.ndarray],
    w_by: Dict[str, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def resample(Z: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(Z)
        if n == 0:
            return Z, w
        p = w / (w.sum() + 1e-12)
        idx = rng.choice(n, size=n, replace=True, p=p)
        Zr = Z[idx]
        wr = np.ones(n, dtype=float) / n
        return Zr, wr

    jsd_c = []
    jsd_d = []
    tv_c = []
    tv_d = []

    for _ in range(BOOTSTRAP_N):
        Z_wt, w_wt = resample(Z_by["WT"], w_by["WT"])
        Z_c, w_c = resample(Z_by["G12C"], w_by["G12C"])
        Z_d, w_d = resample(Z_by["G12D"], w_by["G12D"])

        H_wt, _, _ = hist2d_weighted(Z_wt, w_wt, BINS, xlim, ylim, SMOOTH_SIGMA, True)
        H_c, _, _ = hist2d_weighted(Z_c, w_c, BINS, xlim, ylim, SMOOTH_SIGMA, True)
        H_d, _, _ = hist2d_weighted(Z_d, w_d, BINS, xlim, ylim, SMOOTH_SIGMA, True)

        jsd_c.append(jsd_from_grids(H_c, H_wt))
        jsd_d.append(jsd_from_grids(H_d, H_wt))
        tv_c.append(tv_from_grids(H_c, H_wt))
        tv_d.append(tv_from_grids(H_d, H_wt))

    def ci95(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr, dtype=float)
        return {"lo": float(np.quantile(a, 0.025)), "hi": float(np.quantile(a, 0.975))}

    return {
        "JSD_bits_G12C_vs_WT": ci95(jsd_c),
        "JSD_bits_G12D_vs_WT": ci95(jsd_d),
        "TV_G12C_vs_WT": ci95(tv_c),
        "TV_G12D_vs_WT": ci95(tv_d),
    }


# -----------------------------
# Optional distribution tests (KS/Wasserstein)
# -----------------------------
def dist_tests_within_basins(
    emb: pd.DataFrame,
    basins: List[int],
    metrics: List[str],
) -> pd.DataFrame:
    try:
        from scipy.stats import ks_2samp  # type: ignore
        from scipy.stats import wasserstein_distance  # type: ignore
    except Exception:
        print("[WARN] scipy not available; skip distribution tests.")
        return pd.DataFrame(columns=["basin_id", "metric", "pair", "n_a", "n_b", "ks_stat", "ks_pvalue", "wasserstein", "mean_a", "mean_b", "delta_mean_a_minus_b"])

    rows = []
    for bid in basins:
        g = emb[emb["basin_id"] == bid].copy()
        for metric in metrics:
            if metric not in g.columns:
                continue

            for pair in [("WT", "G12C"), ("WT", "G12D")]:
                a = g[g["label"] == pair[0]][metric].to_numpy(dtype=float)
                b = g[g["label"] == pair[1]][metric].to_numpy(dtype=float)
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if len(a) < 10 or len(b) < 10:
                    continue

                ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
                wd = wasserstein_distance(a, b)
                rows.append({
                    "basin_id": int(bid),
                    "metric": metric,
                    "pair": f"{pair[0]}-vs-{pair[1]}",
                    "n_a": int(len(a)),
                    "n_b": int(len(b)),
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein": float(wd),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "delta_mean_a_minus_b": float(np.mean(a) - np.mean(b)),
                })

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    paths = make_paths()

    # 1) load sampling + post for each label
    sampling_by = {}
    post_by = {}
    merged_by = {}

    bitlen = None
    for lab in ["WT", "G12C", "G12D"]:
        s = load_sampling_set(paths, lab)
        sampling_by[lab] = s
        print(f"[SAMPLING {lab}] unique_bitstrings={len(s)} fragment_id={TARGET[lab]['fragment_id']}")

        post = load_post_tables(paths, lab)
        post_by[lab] = post

        m = merge_sampling_and_post(lab, s, post)
        merged_by[lab] = m

        if bitlen is None:
            bitlen = len(m.iloc[0]["bitstring"])
        else:
            if len(m.iloc[0]["bitstring"]) != bitlen:
                raise ValueError(f"Bitstring length mismatch for {lab}")

    # pool
    pooled = pd.concat([merged_by[lab] for lab in ["WT", "G12C", "G12D"]], ignore_index=True)
    pooled = pooled.reset_index(drop=True)

    X = bitstrings_to_binary_matrix(pooled["bitstring"].tolist())
    labels = pooled["label"].to_numpy()
    weights = pooled["weight"].to_numpy(dtype=float)

    print(f"[POOL] N={len(pooled)} bitlen={X.shape[1]}")

    # 2) t-SNE embedding seed=SEED for main coordinate system
    Z = compute_embedding_tsne(X, seed=SEED)
    pooled["z1"] = Z[:, 0]
    pooled["z2"] = Z[:, 1]

    # shared axis limits
    xmin, xmax = float(Z[:, 0].min()), float(Z[:, 0].max())
    ymin, ymax = float(Z[:, 1].min()), float(Z[:, 1].max())
    padx = 0.05 * (xmax - xmin + 1e-12)
    pady = 0.05 * (ymax - ymin + 1e-12)
    xlim = (xmin - padx, xmax + padx)
    ylim = (ymin - pady, ymax + pady)

    # per label Z/w
    Z_by = {lab: pooled.loc[pooled["label"] == lab, ["z1", "z2"]].to_numpy(dtype=float) for lab in ["WT", "G12C", "G12D"]}
    w_by = {lab: pooled.loc[pooled["label"] == lab, "weight"].to_numpy(dtype=float) for lab in ["WT", "G12C", "G12D"]}
    for lab in w_by:
        s = float(w_by[lab].sum())
        if s > 0:
            w_by[lab] = w_by[lab] / s

    # 3) density + grid metrics
    H_wt, xedges, yedges = hist2d_weighted(Z_by["WT"], w_by["WT"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
    H_c, _, _ = hist2d_weighted(Z_by["G12C"], w_by["G12C"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
    H_d, _, _ = hist2d_weighted(Z_by["G12D"], w_by["G12D"], BINS, xlim, ylim, SMOOTH_SIGMA, True)

    jsd_c = jsd_from_grids(H_c, H_wt)
    jsd_d = jsd_from_grids(H_d, H_wt)
    tv_c = tv_from_grids(H_c, H_wt)
    tv_d = tv_from_grids(H_d, H_wt)

    print("[METRICS]", {"JSD_bits_G12C_vs_WT": jsd_c, "JSD_bits_G12D_vs_WT": jsd_d, "TV_G12C_vs_WT": tv_c, "TV_G12D_vs_WT": tv_d})

    # plots
    plot_scatter_all(paths, Z, labels)
    plot_density_map(paths, H_wt, xedges, yedges, "KRAS sampling density (WT)", "density_WT")
    plot_density_map(paths, H_c, xedges, yedges, f"KRAS sampling density (G12C) | JSD vs WT = {jsd_c:.3f} bits", "density_G12C")
    plot_density_map(paths, H_d, xedges, yedges, f"KRAS sampling density (G12D) | JSD vs WT = {jsd_d:.3f} bits", "density_G12D")

    plot_diff_map(paths, (H_c - H_wt), xedges, yedges, "KRAS density difference: G12C − WT", "diff_G12C_minus_WT")
    plot_diff_map(paths, (H_d - H_wt), xedges, yedges, "KRAS density difference: G12D − WT", "diff_G12D_minus_WT")

    # save embedding points
    pooled.to_csv(paths.out_root / "embedding_points.csv", index=False)

    # 4) stability across seeds (re-run t-SNE; reuse xlim/ylim from seed0 to compare)
    stab_rows = []
    for sd in STABILITY_SEEDS:
        Z_sd = compute_embedding_tsne(X, seed=sd) if sd != SEED else Z
        Z_by_sd = {
            lab: Z_sd[labels == lab] for lab in ["WT", "G12C", "G12D"]
        }
        H_wt_sd, _, _ = hist2d_weighted(Z_by_sd["WT"], w_by["WT"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
        H_c_sd, _, _ = hist2d_weighted(Z_by_sd["G12C"], w_by["G12C"], BINS, xlim, ylim, SMOOTH_SIGMA, True)
        H_d_sd, _, _ = hist2d_weighted(Z_by_sd["G12D"], w_by["G12D"], BINS, xlim, ylim, SMOOTH_SIGMA, True)

        stab_rows.append({
            "seed": sd,
            "JSD_bits_G12C_vs_WT": jsd_from_grids(H_c_sd, H_wt_sd),
            "JSD_bits_G12D_vs_WT": jsd_from_grids(H_d_sd, H_wt_sd),
            "TV_G12C_vs_WT": tv_from_grids(H_c_sd, H_wt_sd),
            "TV_G12D_vs_WT": tv_from_grids(H_d_sd, H_wt_sd),
        })

    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(paths.out_root / "metrics_stability_by_seed.csv", index=False)

    # 5) bootstrap CI for grid metrics
    ci = bootstrap_grid_metrics(Z_by, w_by, xlim, ylim)

    # 6) basin discovery + occupancy + stats
    pooled_b, gmm = add_basin_id(pooled, K=BASIN_K, seed=BASIN_SEED)

    occ = basin_occupancy(pooled_b)
    occ.to_csv(paths.out_root / "basin_occupancy.csv", index=False)

    bstats = compute_basin_stats(pooled_b)
    bstats.to_csv(paths.out_root / "basin_stats.csv", index=False)

    top_info, union_basins, occ_ranked = pick_top_basins(occ, topk=TOPK_BASINS_PER_MUT)
    (paths.out_root / "top_basins.json").write_text(json.dumps(top_info, indent=2), encoding="utf-8")

    # basin delta summary (wide, per basin)
    occ_ranked.to_csv(paths.out_root / "basin_delta_summary.csv", index=False)

    # 7) representatives + export pdb + basin RMSD
    reps = choose_representatives(pooled_b)
    reps.to_csv(paths.out_root / "representatives.csv", index=False)

    if EXPORT_PDB:
        export_representative_pdbs(paths, reps, post_by_label=post_by)

    basin_rmsd_df = compute_basin_pair_ca_rmsd(paths, focus_basins=union_basins)
    basin_rmsd_df.to_csv(paths.out_root / "basin_pair_ca_rmsd.csv", index=False)

    # 8) energy contrast (long table) -> addons/basin_energy_contrast.csv
    energy_contrast = build_basin_energy_contrast(bstats)
    energy_contrast_path = paths.addons / "basin_energy_contrast.csv"
    energy_contrast.to_csv(energy_contrast_path, index=False)

    # plot "waterfall" term contrast from that file
    ec_path = find_energy_contrast_csv(paths)
    if ec_path is None:
        print("[WARN] No basin_energy_contrast.csv found. Skip energy waterfall.")
    else:
        ec = pd.read_csv(ec_path)
        plot_waterfall_from_energy_contrast(paths, ec, focus_basins=union_basins)

    # 9) optional distribution tests (inside union basins)
    if ENABLE_DIST_TESTS:
        # prefer a few key metrics (you can add more)
        test_metrics = []
        for m in ["E_total", "backbone_rmsd", "E_hydroph", "E_steric"]:
            if m in pooled_b.columns:
                test_metrics.append(m)
        dt = dist_tests_within_basins(pooled_b, union_basins, test_metrics)
        dt.to_csv(paths.out_root / "basin_dist_tests.csv", index=False)

    # 10) write metrics.json
    metrics = {
        "analysis_set": {lab: TARGET[lab]["fragment_id"] for lab in ["WT", "G12C", "G12D"]},
        "embedding": {"method": "t-SNE", "metric": "hamming", "seed": int(SEED), "max_points_per_set": int(MAX_POINTS_PER_SET)},
        "density": {"bins": int(BINS), "smooth_sigma": float(SMOOTH_SIGMA)},
        "grid_metrics": {
            "JSD_bits_G12C_vs_WT": float(jsd_c),
            "JSD_bits_G12D_vs_WT": float(jsd_d),
            "TV_G12C_vs_WT": float(tv_c),
            "TV_G12D_vs_WT": float(tv_d),
        },
        "basin": {"method": "GMM", "covariance_type": "full", "K": int(BASIN_K), "seed": int(BASIN_SEED)},
        "bootstrap_ci95": ci,
        "stability_seeds": STABILITY_SEEDS,
        "top_basins": top_info,
    }
    (paths.out_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[DONE] All outputs saved to:\n  {paths.out_root}")
    print(f"[DONE] Plots saved to:\n  {paths.plots}")
    print(f"[DONE] Addons saved to:\n  {paths.addons}")


if __name__ == "__main__":
    main()

