# --*-- conding:utf-8 --*--
# @time:1/11/26 21:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_analysis.py

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log(f"[WARN] Failed to read CSV: {path} ({e})")
        return None


def savefig_all(fig: plt.Figure, out_base: Path, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(str(out_base.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def pick_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def normalize_label(x: str) -> str:
    s = str(x).strip()
    u = s.upper()
    if "WT" in u and "G12" not in u:
        return "WT"
    if "G12C" in u:
        return "G12C"
    if "G12D" in u:
        return "G12D"
    return s


def infer_label_from_row(row: pd.Series, label_col: Optional[str], pdbid_col: Optional[str]) -> str:
    if label_col and label_col in row and pd.notna(row[label_col]):
        return normalize_label(str(row[label_col]))
    if pdbid_col and pdbid_col in row and pd.notna(row[pdbid_col]):
        return normalize_label(str(row[pdbid_col]))
    return "UNKNOWN"


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def find_col_contains(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        cl = c.lower()
        for p in patterns:
            if p.lower() in cl:
                return c
    return None


def sort_basins_naturally(basins: List[str]) -> List[str]:
    def key_fn(x: str):
        m = re.search(r"(\d+)", str(x))
        return (int(m.group(1)) if m else 10**9, str(x))
    return sorted(basins, key=key_fn)


@dataclass
class Paths:
    project_root: Path
    analysis_dir: Path
    addons_dir: Path
    fig_dir: Path


def resolve_paths(analysis_dir_arg: Optional[str]) -> Paths:
    # Script is in tools/, and tools/ is sibling of KRAS_sampling_results/
    project_root = Path(__file__).resolve().parent.parent

    if analysis_dir_arg:
        analysis_dir = Path(analysis_dir_arg).expanduser().resolve()
    else:
        analysis_dir = project_root / "KRAS_sampling_results" / "analysis_closed_loop"

    addons_dir = analysis_dir / "addons"
    fig_dir = analysis_dir / "figs_redraw"
    ensure_dir(fig_dir)

    log(f"[INFO] PROJECT_ROOT: {project_root}")
    log(f"[INFO] ANALYSIS_DIR:  {analysis_dir}")
    log(f"[INFO] ADDONS_DIR:   {addons_dir}")
    log(f"[INFO] FIG_DIR:      {fig_dir}")
    if not addons_dir.exists():
        log(f"[WARN] ADDONS_DIR does not exist: {addons_dir}")

    return Paths(project_root, analysis_dir, addons_dir, fig_dir)


# -----------------------------
# Plotters
# -----------------------------
def plot_distribution_energy_rmsd(paths: Paths) -> None:
    src = pick_first_existing([
        paths.analysis_dir / "merged_points_with_basin.csv",
        paths.analysis_dir / "merged_points.csv",
    ])
    if not src:
        log("[WARN] No merged_points*.csv found. Skipping distribution plots.")
        return

    df = read_csv_safe(src)
    if df is None or len(df) == 0:
        log("[WARN] merged_points CSV is empty. Skipping distribution plots.")
        return

    # Try to infer key columns robustly
    pdbid_col = find_col(df, ["pdb_id", "pdbid", "case_id", "id"])
    label_col = find_col(df, ["label", "variant", "mut", "mutation", "group_label"])
    basin_col = find_col(df, ["basin_id", "basin", "cluster", "component"])
    rmsd_col = find_col_contains(df, ["rmsd"])
    energy_col = find_col(df, ["E_total", "energy", "score", "E", "total_energy"])
    mass_col = find_col(df, ["p_mass", "prob_mass", "mass", "prob", "weight"])

    if rmsd_col is None:
        log(f"[WARN] Could not find an RMSD column in {src.name}. Columns: {list(df.columns)}")
        return
    if energy_col is None:
        log(f"[WARN] Could not find an energy column in {src.name}. Columns: {list(df.columns)}")
        return

    # Create a canonical label column
    df["_label"] = df.apply(lambda r: infer_label_from_row(r, label_col, pdbid_col), axis=1)

    # Filter invalid numeric rows
    df = df.copy()
    df = df[np.isfinite(pd.to_numeric(df[rmsd_col], errors="coerce"))]
    df = df[np.isfinite(pd.to_numeric(df[energy_col], errors="coerce"))]
    df[rmsd_col] = pd.to_numeric(df[rmsd_col], errors="coerce")
    df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")

    label_order = ["WT", "G12C", "G12D"]
    labels = [x for x in label_order if x in set(df["_label"].unique())] + \
             [x for x in sorted(df["_label"].unique()) if x not in label_order]

    # 1) Faceted hexbin: RMSD vs Energy per label
    n = len(labels)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, lab in enumerate(labels):
        ax = axes[i]
        sub = df[df["_label"] == lab]
        x = sub[rmsd_col].to_numpy()
        y = sub[energy_col].to_numpy()

        if len(sub) == 0:
            ax.set_axis_off()
            continue

        hb = ax.hexbin(x, y, gridsize=60, bins="log", mincnt=1)
        ax.set_title(f"{lab} (n={len(sub)})")
        ax.set_xlabel(rmsd_col)
        ax.set_ylabel(energy_col)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log10(count)")

    for j in range(len(labels), len(axes)):
        axes[j].set_axis_off()

    savefig_all(fig, paths.fig_dir / "dist_energy_vs_rmsd_hexbin_by_label")

    # 2) If basin exists, a clean scatter of representatives by basin and label
    if basin_col and basin_col in df.columns:
        # Sample to keep it readable if too large
        max_points = 60000
        d2 = df
        if len(d2) > max_points:
            d2 = d2.sample(n=max_points, random_state=0)

        fig2, ax2 = plt.subplots(figsize=(8.5, 6.5))
        ax2.scatter(
            d2[rmsd_col].to_numpy(),
            d2[energy_col].to_numpy(),
            s=6,
            alpha=0.35,
        )
        ax2.set_title("All samples (scatter) – Energy vs RMSD")
        ax2.set_xlabel(rmsd_col)
        ax2.set_ylabel(energy_col)
        savefig_all(fig2, paths.fig_dir / "dist_energy_vs_rmsd_scatter_all")

    # 3) Mass vs basin (if available)
    if mass_col and mass_col in df.columns and basin_col and basin_col in df.columns:
        dmass = df.copy()
        dmass[mass_col] = pd.to_numeric(dmass[mass_col], errors="coerce")
        dmass = dmass[np.isfinite(dmass[mass_col])]

        g = (
            dmass.groupby(["_label", basin_col])[mass_col]
            .sum()
            .reset_index()
        )
        # Normalize within label to show occupancy distribution
        g["_norm"] = g.groupby("_label")[mass_col].transform(lambda x: x / max(x.sum(), 1e-12))

        basins = sort_basins_naturally([str(x) for x in g[basin_col].unique()])
        fig3, ax3 = plt.subplots(figsize=(10.5, 5.5))
        width = 0.25
        x = np.arange(len(basins))

        for k, lab in enumerate([l for l in ["WT", "G12C", "G12D"] if l in set(g["_label"])]):
            sub = g[g["_label"] == lab].set_index(basin_col)
            y = [float(sub.loc[b, "_norm"]) if b in sub.index else 0.0 for b in basins]
            ax3.bar(x + (k - 1) * width, y, width=width, label=lab)

        ax3.set_xticks(x)
        ax3.set_xticklabels(basins, rotation=0)
        ax3.set_ylabel("Normalized mass within label")
        ax3.set_xlabel("Basin")
        ax3.set_title("Basin occupancy distribution (from merged_points)")
        ax3.legend()
        savefig_all(fig3, paths.fig_dir / "basin_occupancy_from_merged_points_norm")


def plot_basin_occupancy_delta(paths: Paths) -> None:
    # Prefer delta_basin_mass.csv (you have it)
    delta_path = paths.addons_dir / "delta_basin_mass.csv"
    occ_path = paths.analysis_dir / "basin_occupancy.csv"

    df = read_csv_safe(delta_path)
    if df is None:
        df = read_csv_safe(occ_path)

    if df is None or len(df) == 0:
        log("[WARN] No delta_basin_mass.csv or basin_occupancy.csv found/usable. Skipping occupancy delta plot.")
        return

    basin_col = find_col(df, ["basin_id", "basin", "cluster"])
    if basin_col is None:
        basin_col = df.columns[0]

    # Try to find delta columns
    col_c = find_col_contains(df, ["delta", "g12c"]) or find_col(df, ["delta_mass_G12C_minus_WT", "delta_G12C_minus_WT"])
    col_d = find_col_contains(df, ["delta", "g12d"]) or find_col(df, ["delta_mass_G12D_minus_WT", "delta_G12D_minus_WT"])

    # If the file is occupancy, compute deltas
    if (col_c is None or col_d is None) and ("WT" in df.columns and "G12C" in df.columns and "G12D" in df.columns):
        df = df.copy()
        df["delta_G12C_minus_WT"] = pd.to_numeric(df["G12C"], errors="coerce") - pd.to_numeric(df["WT"], errors="coerce")
        df["delta_G12D_minus_WT"] = pd.to_numeric(df["G12D"], errors="coerce") - pd.to_numeric(df["WT"], errors="coerce")
        col_c, col_d = "delta_G12C_minus_WT", "delta_G12D_minus_WT"

    if col_c is None or col_d is None:
        log(f"[WARN] Could not infer delta columns in {df.shape}. Columns: {list(df.columns)}")
        return

    df = df.copy()
    df[basin_col] = df[basin_col].astype(str)
    df[col_c] = pd.to_numeric(df[col_c], errors="coerce")
    df[col_d] = pd.to_numeric(df[col_d], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_c, col_d])

    basins = sort_basins_naturally(df[basin_col].unique().tolist())
    df = df.set_index(basin_col).reindex(basins).reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(df))
    width = 0.38
    ax.bar(x - width / 2, df[col_c].to_numpy(), width=width, label="G12C - WT")
    ax.bar(x + width / 2, df[col_d].to_numpy(), width=width, label="G12D - WT")
    ax.axhline(0.0, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(df[basin_col].tolist())
    ax.set_xlabel("Basin")
    ax.set_ylabel("Δ mass")
    ax.set_title("Basin occupancy shift (Δ mass)")
    ax.legend()
    savefig_all(fig, paths.fig_dir / "basin_occupancy_delta_mass")


def parse_basin_energy_contrast(df: pd.DataFrame) -> Tuple[str, List[str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      basin_col, basins, data
      data[basin][key] = delta value, where key in {"G12C", "G12D"} for each term
    Supports:
      - Long format: columns include basin_id, term, delta_G12C_minus_WT, delta_G12D_minus_WT
      - Wide format: columns like delta_E_steric_G12C_minus_WT, delta_E_steric_G12D_minus_WT
    """
    basin_col = find_col(df, ["basin_id", "basin", "cluster"]) or df.columns[0]

    # Long format?
    term_col = find_col(df, ["term", "energy_term", "component", "name"])
    long_c = find_col_contains(df, ["delta", "g12c"])
    long_d = find_col_contains(df, ["delta", "g12d"])

    if term_col and long_c and long_d:
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        basins = sort_basins_naturally([str(x) for x in df[basin_col].unique()])
        terms = [str(x) for x in df[term_col].unique()]
        # Keep term order as appeared
        terms_seen = []
        for t in df[term_col].tolist():
            ts = str(t)
            if ts not in terms_seen:
                terms_seen.append(ts)

        data: Dict[str, Dict[str, float]] = {}
        # We'll store per basin a dict term->(c,d) by encoding keys "term|G12C" "term|G12D"
        by_basin: Dict[str, Dict[str, float]] = {}
        for b in basins:
            by_basin[b] = {}

        for _, r in df.iterrows():
            b = str(r[basin_col])
            t = str(r[term_col])
            c = float(pd.to_numeric(r[long_c], errors="coerce"))
            d = float(pd.to_numeric(r[long_d], errors="coerce"))
            if np.isfinite(c):
                by_basin[b][f"{t}|G12C"] = c
            if np.isfinite(d):
                by_basin[b][f"{t}|G12D"] = d

        return basin_col, terms_seen, by_basin

    # Wide format
    cols = list(df.columns)
    # Identify delta columns: *G12C_minus_WT* and *G12D_minus_WT*
    c_cols = [c for c in cols if re.search(r"g12c.*minus.*wt", c, flags=re.IGNORECASE) and "delta" in c.lower()]
    d_cols = [c for c in cols if re.search(r"g12d.*minus.*wt", c, flags=re.IGNORECASE) and "delta" in c.lower()]

    if not c_cols and not d_cols:
        raise ValueError("Unsupported basin_energy_contrast.csv format")

    # Infer term names by stripping suffix
    def term_name(col: str) -> str:
        s = col
        s = re.sub(r"(?i)_?g12c_?minus_?wt", "", s)
        s = re.sub(r"(?i)_?g12d_?minus_?wt", "", s)
        s = re.sub(r"(?i)^delta_?", "", s)
        return s

    terms = []
    for c in c_cols + d_cols:
        t = term_name(c)
        if t not in terms:
            terms.append(t)

    basins = sort_basins_naturally([str(x) for x in df[basin_col].unique()])
    by_basin: Dict[str, Dict[str, float]] = {b: {} for b in basins}

    for _, r in df.iterrows():
        b = str(r[basin_col])
        for c in c_cols:
            t = term_name(c)
            val = float(pd.to_numeric(r[c], errors="coerce"))
            if np.isfinite(val):
                by_basin[b][f"{t}|G12C"] = val
        for c in d_cols:
            t = term_name(c)
            val = float(pd.to_numeric(r[c], errors="coerce"))
            if np.isfinite(val):
                by_basin[b][f"{t}|G12D"] = val

    return basin_col, terms, by_basin


def plot_waterfall(ax: plt.Axes, terms: List[str], deltas: List[float], title: str) -> None:
    # Classic waterfall: incremental bars + final total
    terms2 = terms + ["TOTAL"]
    vals = list(deltas) + [float(np.nansum(deltas))]

    running = 0.0
    starts = []
    heights = []
    for v in deltas:
        starts.append(running)
        heights.append(v)
        running += v

    # Bars for increments
    x = np.arange(len(terms2))
    ax.bar(x[:len(deltas)], heights, bottom=starts, width=0.75)
    # Total bar from zero
    ax.bar(x[-1], vals[-1], width=0.75)

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(terms2, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Δ value")


def plot_energy_contrast_waterfalls(paths: Paths) -> None:
    p = paths.addons_dir / "basin_energy_contrast.csv"
    df = read_csv_safe(p)
    if df is None or len(df) == 0:
        log("[WARN] basin_energy_contrast.csv missing/empty. Skipping waterfall plots.")
        return

    try:
        basin_col, terms, by_basin = parse_basin_energy_contrast(df)
    except Exception as e:
        log(f"[WARN] Could not parse basin_energy_contrast.csv: {e}")
        return

    basins = sort_basins_naturally(list(by_basin.keys()))
    # Keep only basins that have any entries
    basins = [b for b in basins if len(by_basin[b]) > 0]

    # For each basin, create a 1x2 figure: G12C-WT and G12D-WT
    for b in basins:
        # Build delta lists aligned to terms
        deltas_c = [by_basin[b].get(f"{t}|G12C", np.nan) for t in terms]
        deltas_d = [by_basin[b].get(f"{t}|G12D", np.nan) for t in terms]

        # Drop terms that are all NaN for both comparisons
        keep_terms = []
        keep_c = []
        keep_d = []
        for t, vc, vd in zip(terms, deltas_c, deltas_d):
            if np.isfinite(vc) or np.isfinite(vd):
                keep_terms.append(t)
                keep_c.append(0.0 if not np.isfinite(vc) else float(vc))
                keep_d.append(0.0 if not np.isfinite(vd) else float(vd))

        if len(keep_terms) == 0:
            continue

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 4.8))
        plot_waterfall(axes[0], keep_terms, keep_c, title=f"Basin {b}: G12C - WT")
        plot_waterfall(axes[1], keep_terms, keep_d, title=f"Basin {b}: G12D - WT")
        savefig_all(fig, paths.fig_dir / f"waterfall_basin{b}")


def plot_per_residue_displacement(paths: Paths) -> None:
    # Prefer the "overall" file, then fall back to per-basin files
    overall = paths.addons_dir / "displacement" / "per_residue_displacement.csv"
    df = read_csv_safe(overall)

    if df is not None and len(df) > 0:
        res_col = find_col(df, ["resi", "residue", "residue_id", "res_index", "resnum"]) or df.columns[0]
        basin_col = find_col(df, ["basin_id", "basin", "cluster"])  # might not exist

        # Identify displacement columns
        # We accept columns containing "disp" or "displacement" and also containing G12C/G12D and WT
        disp_cols = [c for c in df.columns if re.search(r"disp|displacement", c, flags=re.IGNORECASE)]
        if not disp_cols:
            # fall back to numeric columns excluding res_col/basin_col
            disp_cols = [c for c in df.columns if c not in {res_col, basin_col}]

        df2 = df.copy()
        df2[res_col] = pd.to_numeric(df2[res_col], errors="coerce")
        df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=[res_col])
        df2 = df2.sort_values(res_col)

        def plot_one(sub: pd.DataFrame, title: str, out: Path) -> None:
            fig, ax = plt.subplots(figsize=(10.8, 4.6))
            for c in disp_cols:
                y = pd.to_numeric(sub[c], errors="coerce").to_numpy()
                if np.all(~np.isfinite(y)):
                    continue
                ax.plot(sub[res_col].to_numpy(), y, linewidth=1.6, label=c)
            ax.set_title(title)
            ax.set_xlabel("Residue index")
            ax.set_ylabel("Displacement")
            ax.legend(fontsize=8, ncol=1)
            savefig_all(fig, out)

        if basin_col and basin_col in df2.columns:
            basins = sort_basins_naturally([str(x) for x in df2[basin_col].unique()])
            for b in basins:
                sub = df2[df2[basin_col].astype(str) == str(b)]
                if len(sub) == 0:
                    continue
                plot_one(sub, f"Per-residue displacement (Basin {b})", paths.fig_dir / f"per_residue_displacement_basin{b}")
        else:
            plot_one(df2, "Per-residue displacement (overall)", paths.fig_dir / "per_residue_displacement_overall")

        return

    # If overall is missing/unusable, plot each per-basin CSV in addons/per_residue/
    per_dir = paths.addons_dir / "per_residue"
    if not per_dir.exists():
        log("[WARN] No per-residue displacement CSV found. Skipping displacement plot.")
        return

    files = sorted(per_dir.glob("*_per_residue_displacement.csv"))
    if not files:
        log("[WARN] No per-basin per_residue_displacement CSVs found. Skipping displacement plot.")
        return

    for f in files:
        d = read_csv_safe(f)
        if d is None or len(d) == 0:
            continue
        res_col = find_col(d, ["resi", "residue", "residue_id", "res_index", "resnum"]) or d.columns[0]
        y_col = find_col_contains(d, ["disp", "displacement"]) or d.columns[1]

        d2 = d.copy()
        d2[res_col] = pd.to_numeric(d2[res_col], errors="coerce")
        d2[y_col] = pd.to_numeric(d2[y_col], errors="coerce")
        d2 = d2.replace([np.inf, -np.inf], np.nan).dropna(subset=[res_col, y_col]).sort_values(res_col)

        fig, ax = plt.subplots(figsize=(10.8, 4.6))
        ax.plot(d2[res_col].to_numpy(), d2[y_col].to_numpy(), linewidth=1.8)
        ax.set_title(f.name.replace("_per_residue_displacement.csv", ""))
        ax.set_xlabel("Residue index")
        ax.set_ylabel(y_col)
        savefig_all(fig, paths.fig_dir / f"per_residue_{f.stem}")


def plot_mechanism_ci(paths: Paths) -> None:
    p = paths.addons_dir / "mechanism" / "occupancy_delta_ci.csv"
    df = read_csv_safe(p)
    if df is None or len(df) == 0:
        return

    basin_col = find_col(df, ["basin_id", "basin", "cluster"]) or df.columns[0]

    # Expect columns like: delta_mean_G12C_minus_WT, lo, hi (or similar)
    mean_c = find_col_contains(df, ["mean", "g12c"]) or find_col_contains(df, ["delta", "g12c"])
    lo_c = find_col_contains(df, ["lo", "lower", "ci_low", "lcl", "g12c"])
    hi_c = find_col_contains(df, ["hi", "upper", "ci_high", "ucl", "g12c"])

    mean_d = find_col_contains(df, ["mean", "g12d"]) or find_col_contains(df, ["delta", "g12d"])
    lo_d = find_col_contains(df, ["lo", "lower", "ci_low", "lcl", "g12d"])
    hi_d = find_col_contains(df, ["hi", "upper", "ci_high", "ucl", "g12d"])

    if mean_c is None or mean_d is None:
        return

    df2 = df.copy()
    df2[basin_col] = df2[basin_col].astype(str)
    basins = sort_basins_naturally(df2[basin_col].unique().tolist())
    df2 = df2.set_index(basin_col).reindex(basins).reset_index()

    x = np.arange(len(df2))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.2, 5.2))

    def err(mean, lo, hi):
        if lo is None or hi is None:
            return None
        m = pd.to_numeric(df2[mean], errors="coerce").to_numpy()
        l = pd.to_numeric(df2[lo], errors="coerce").to_numpy()
        h = pd.to_numeric(df2[hi], errors="coerce").to_numpy()
        e1 = m - l
        e2 = h - m
        return np.vstack([e1, e2])

    y_c = pd.to_numeric(df2[mean_c], errors="coerce").to_numpy()
    y_d = pd.to_numeric(df2[mean_d], errors="coerce").to_numpy()

    e_c = err(mean_c, lo_c, hi_c)
    e_d = err(mean_d, lo_d, hi_d)

    ax.bar(x - width / 2, y_c, width=width, yerr=e_c, capsize=3, label="G12C - WT")
    ax.bar(x + width / 2, y_d, width=width, yerr=e_d, capsize=3, label="G12D - WT")
    ax.axhline(0.0, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(df2[basin_col].tolist())
    ax.set_xlabel("Basin")
    ax.set_ylabel("Δ mass (mean ± CI)")
    ax.set_title("Occupancy shift with CI (bootstrap)")
    ax.legend()
    savefig_all(fig, paths.fig_dir / "occupancy_delta_ci")


def plot_metrics(paths: Paths) -> None:
    p = paths.analysis_dir / "metrics_bootstrap.csv"
    df = read_csv_safe(p)
    if df is None or len(df) == 0:
        return

    # Make a compact view of all numeric metric columns as boxplots
    non_metric_cols = set([c for c in df.columns if re.search(r"seed|case|id|label|variant", c, flags=re.IGNORECASE)])
    metric_cols = [c for c in df.columns if c not in non_metric_cols]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c]) or np.issubdtype(df[c].dtype, np.number)]

    if not metric_cols:
        return

    data = []
    names = []
    for c in metric_cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        data.append(v)
        names.append(c)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(9.5, 0.65 * len(names)), 5.2))
    ax.boxplot(data, labels=names, showfliers=False)
    ax.set_title("Bootstrap metrics (boxplot)")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    savefig_all(fig, paths.fig_dir / "metrics_bootstrap_boxplot")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_dir", type=str, default=None,
                    help="Override analysis_closed_loop directory path")
    args = ap.parse_args()

    paths = resolve_paths(args.analysis_dir)

    # Core plots from your actual files
    plot_distribution_energy_rmsd(paths)
    plot_basin_occupancy_delta(paths)
    plot_energy_contrast_waterfalls(paths)
    plot_per_residue_displacement(paths)

    # Optional nicer “mechanism” plot if present
    plot_mechanism_ci(paths)

    # Optional: metrics overview
    plot_metrics(paths)

    log(f"[INFO] Done. Figures saved to: {paths.fig_dir}")


if __name__ == "__main__":
    main()


