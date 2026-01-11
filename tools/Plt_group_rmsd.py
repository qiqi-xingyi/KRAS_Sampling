# --*-- conding:utf-8 --*--
# @time:1/9/26 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_group_rmsd.py

# ------------------------------------------------------------
# Top-journal style RMSD comparison plots (paired set, N=20):
#
# Figures produced (on the SAME case set = Sampling cases):
#   Fig1. Grouped bar chart (per-case, 4 methods)
#   Fig2. Paired delta RMSD (baseline - sampling): box + jitter + median annotation
#   FigS1. (Optional) 3-panel paired scatter: Sampling vs AF2/AF3/VQE with y=x and r, rho

# Inputs:
#   <project_root>/QDock_RMSD/
#     af2_rmsd_summary.txt
#     af3_rmsd_summary.txt
#     q_rmsd_summary.txt
#     backbone_rmsd_min.csv   (sampling-based min RMSD per pdb_id)

# Outputs:
#   <project_root>/QDock_RMSD/merged_rmsd_on_sampling_ids.csv
#   <project_root>/QDock_RMSD/plots_topjournal/
#     fig1_grouped_bar.png/.pdf
#     fig2_delta_box_jitter.png/.pdf
#     figs1_scatter_triplet.png/.pdf

# Notes:
# - Assumes your statement: for your 20 cases, all other 3 methods have results too.
#   Script still handles missing values robustly (skips those points).
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER INTERFACES (EDIT HERE)
# ============================================================

# Method styles (consistent across all plots)
METHODS = [
    ("sampling_rmsd", "Our"),
    ("af2_rmsd", "ColabFold"),
    ("af3_rmsd", "AF3"),
    ("vqe_rmsd", "VQE"),
]

METHOD_COLORS = {
    "sampling_rmsd": "#6E6E6E",  # gray
    "af2_rmsd":      "#2F6BFF",  # blue
    "af3_rmsd":      "#22A06B",  # green
    "vqe_rmsd":      "#E24A33",  # red
}

# Font sizes (top-journal-ish, clean)
FONT = {
    "title": 13,
    "axis": 12,
    "tick": 10,
    "legend": 10,
    "annot": 10,
}

# Export
DPI = 300

# Case ordering in Fig1
SORT_BY = "sampling_rmsd"   # "sampling_rmsd" or "pdb_id"

# Fig1 grouped bar
BAR_WIDTH = 0.18
GROUP_GAP = 0.40          # extra spacing between groups
ROTATE_X = 45

# Fig2 delta plot
DELTA_ORDER = [
    ("af2_rmsd", "AF2 - Sampling"),
    ("af3_rmsd", "AF3 - Sampling"),
    ("vqe_rmsd", "VQE - Sampling"),
]
SHOW_OUTLIERS = True
JITTER_WIDTH = 0.10
RANDOM_SEED = 0

# Optional scatter triplet
ENABLE_SCATTER_TRIPLET = True


# ============================================================
# Helpers
# ============================================================

def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def apply_rcparams():
    plt.rcParams["axes.titlesize"] = FONT["title"]
    plt.rcParams["axes.labelsize"] = FONT["axis"]
    plt.rcParams["xtick.labelsize"] = FONT["tick"]
    plt.rcParams["ytick.labelsize"] = FONT["tick"]
    plt.rcParams["legend.fontsize"] = FONT["legend"]
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["savefig.bbox"] = "tight"


def read_rmsd_kv_txt(path: Path) -> Dict[str, float]:
    """Parse: pdb_id <space/tab> rmsd ; ignore comments (#) and blanks."""
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            pid = parts[0].strip().lower()
            try:
                val = float(parts[1])
            except Exception:
                continue
            out[pid] = val
    return out


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def save_fig(fig: plt.Figure, out_png: Path, out_pdf: Path):
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    # rank transform (no scipy dependency)
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    return pearson_r(rx, ry)


def median_text(v: np.ndarray) -> str:
    if len(v) == 0:
        return "median=nan"
    return f"median={np.median(v):.3f}"


# ============================================================
# Plot: Fig1 grouped bar (per-case)
# ============================================================

def plot_fig1_grouped_bar(df: pd.DataFrame, out_png: Path, out_pdf: Path):
    d = df.copy()

    if SORT_BY == "pdb_id":
        d = d.sort_values("pdb_id").reset_index(drop=True)
    else:
        d = d.sort_values("sampling_rmsd").reset_index(drop=True)

    n = len(d)
    case_labels = d["pdb_id"].tolist()

    # custom x positions with extra gap between groups (top-journal readability)
    x = np.arange(n, dtype=float) * (1.0 + GROUP_GAP)

    fig_w = max(10.0, 0.45 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    offsets = np.linspace(-1.5 * BAR_WIDTH, 1.5 * BAR_WIDTH, num=len(METHODS))

    for (col, label), off in zip(METHODS, offsets):
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        mask = ~np.isnan(y)
        if not np.any(mask):
            continue
        ax.bar(
            x[mask] + off,
            y[mask],
            width=BAR_WIDTH,
            color=METHOD_COLORS.get(col, None),
            label=label,
            linewidth=0.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=ROTATE_X, ha="right")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title(f"Per-case RMSD comparison (paired set, n={n})")

    # y range with padding
    y_all = []
    for col, _ in METHODS:
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        y = y[~np.isnan(y)]
        if len(y) > 0:
            y_all.append(y)
    if y_all:
        ymax = float(np.max(np.concatenate(y_all)))
        ax.set_ylim(0.0, ymax * 1.12)

    # clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, ncol=4, loc="upper right")

    save_fig(fig, out_png, out_pdf)


# ============================================================
# Plot: Fig2 delta RMSD (baseline - sampling) box + jitter
# ============================================================

def plot_fig2_delta_box_jitter(df: pd.DataFrame, out_png: Path, out_pdf: Path):
    rng = np.random.default_rng(RANDOM_SEED)

    # compute deltas
    deltas = []
    labels = []
    colors = []
    medians = []

    for base_col, label in DELTA_ORDER:
        sub = df[["sampling_rmsd", base_col]].dropna()
        if sub.empty:
            continue
        delta = sub[base_col].to_numpy(float) - sub["sampling_rmsd"].to_numpy(float)
        deltas.append(delta)
        labels.append(f"{label}\n(n={len(delta)})")
        colors.append(METHOD_COLORS.get(base_col, "#999999"))
        medians.append(float(np.median(delta)))

    if not deltas:
        print("[SKIP] Fig2: no paired deltas available.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    bp = ax.boxplot(
        deltas,
        labels=labels,
        showfliers=SHOW_OUTLIERS,
        patch_artist=True,
        widths=0.55,
    )

    # color boxes consistent with baseline method color
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.25)
        box.set_linewidth(1.2)
    for median in bp["medians"]:
        median.set_linewidth(1.6)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.2)
    for cap in bp["caps"]:
        cap.set_linewidth(1.2)

    # jitter points (paired per-case deltas)
    for i, (delta, c) in enumerate(zip(deltas, colors), start=1):
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(delta))
        ax.scatter(
            np.full(len(delta), i) + jitter,
            delta,
            s=28,
            alpha=0.85,
            color=c,
            edgecolors="none",
            zorder=3,
        )
        # median annotation
        ax.text(
            i,
            np.median(delta),
            f"{np.median(delta):.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT["annot"],
            color="black",
            zorder=4,
        )

    # reference line at 0 (no difference)
    ax.axhline(0.0, linewidth=1.2, linestyle="--", color="#333333")

    ax.set_ylabel("ΔRMSD (Å) = RMSD(baseline) − RMSD(sampling)")
    ax.set_title("Paired per-case improvement over Sampling (ΔRMSD)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_fig(fig, out_png, out_pdf)


# ============================================================
# Plot: FigS1 scatter triplet
# ============================================================

def plot_figs1_scatter_triplet(df: pd.DataFrame, out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=False, sharey=False)

    comparisons = [
        ("af2_rmsd", "AF2 (ColabFold)"),
        ("af3_rmsd", "AF3"),
        ("vqe_rmsd", "VQE"),
    ]

    for ax, (col, name) in zip(axes, comparisons):
        sub = df[["sampling_rmsd", col]].dropna()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["sampling_rmsd"].to_numpy(float)
        y = sub[col].to_numpy(float)
        r = pearson_r(x, y)
        rho = spearman_rho(x, y)

        # axis limits
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad

        ax.scatter(
            x, y,
            s=34,
            alpha=0.85,
            color=METHOD_COLORS.get(col, "#999999"),
            edgecolors="none",
        )
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        ax.set_xlabel("Sampling RMSD (Å)")
        ax.set_ylabel(f"{name} RMSD (Å)")
        title_r = "nan" if np.isnan(r) else f"{r:.3f}"
        title_rho = "nan" if np.isnan(rho) else f"{rho:.3f}"
        ax.set_title(f"{name}\n n={len(sub)}  r={title_r}  ρ={title_rho}")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Paired scatter comparisons (Sampling vs baselines)", fontsize=FONT["title"])
    save_fig(fig, out_png, out_pdf)


# ============================================================
# Main
# ============================================================

def main():
    apply_rcparams()

    root = project_root_from_tools_dir()
    rmsd_dir = root / "QDock_RMSD"

    af2_path = rmsd_dir / "af2_rmsd_summary.txt"
    af3_path = rmsd_dir / "af3_rmsd_summary.txt"
    vqe_path = rmsd_dir / "q_rmsd_summary.txt"
    sampling_csv = rmsd_dir / "backbone_rmsd_min.csv"

    if not rmsd_dir.exists():
        raise FileNotFoundError(f"Missing folder: {rmsd_dir}")
    if not sampling_csv.exists():
        raise FileNotFoundError(f"Missing sampling results csv: {sampling_csv}")

    # Load sampling results
    s_df = pd.read_csv(sampling_csv)
    if "pdb_id" not in s_df.columns or "min_rmsd" not in s_df.columns:
        raise ValueError("backbone_rmsd_min.csv must contain columns: pdb_id, min_rmsd")

    s_df["pdb_id"] = s_df["pdb_id"].astype(str).str.strip().str.lower()
    s_df["sampling_rmsd"] = pd.to_numeric(s_df["min_rmsd"], errors="coerce")
    s_df = s_df.dropna(subset=["sampling_rmsd"]).reset_index(drop=True)

    sampling_ids = sorted(set(s_df["pdb_id"].tolist()))
    print(f"[LOAD] sampling cases: {len(sampling_ids)}")

    # Load baselines
    af2 = read_rmsd_kv_txt(af2_path)
    af3 = read_rmsd_kv_txt(af3_path)
    vqe = read_rmsd_kv_txt(vqe_path)
    print(f"[LOAD] AF2={len(af2)} AF3={len(af3)} VQE={len(vqe)}")

    # Merge only on sampling ids
    rows = []
    for pid in sampling_ids:
        srows = s_df.loc[s_df["pdb_id"] == pid, "sampling_rmsd"]
        s_val = safe_float(srows.iloc[0]) if len(srows) > 0 else None
        rows.append(
            {
                "pdb_id": pid,
                "sampling_rmsd": s_val,
                "af2_rmsd": safe_float(af2.get(pid)),
                "af3_rmsd": safe_float(af3.get(pid)),
                "vqe_rmsd": safe_float(vqe.get(pid)),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["sampling_rmsd"]).reset_index(drop=True)

    # Save merged
    merged_out = rmsd_dir / "merged_rmsd_on_sampling_ids.csv"
    df.to_csv(merged_out, index=False)
    print(f"[SAVE] {merged_out} (n={len(df)})")

    # Output dir
    out_dir = rmsd_dir / "plots_topjournal"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig1: grouped bar
    plot_fig1_grouped_bar(
        df,
        out_png=out_dir / "fig1_grouped_bar.png",
        out_pdf=out_dir / "fig1_grouped_bar.pdf",
    )

    # Fig2: delta box + jitter
    plot_fig2_delta_box_jitter(
        df,
        out_png=out_dir / "fig2_delta_box_jitter.png",
        out_pdf=out_dir / "fig2_delta_box_jitter.pdf",
    )

    # FigS1: scatter triplet
    if ENABLE_SCATTER_TRIPLET:
        plot_figs1_scatter_triplet(
            df,
            out_png=out_dir / "figs1_scatter_triplet.png",
            out_pdf=out_dir / "figs1_scatter_triplet.pdf",
        )

    # Sanity: paired coverage
    print("\n[PAIRED COVERAGE]")
    n = len(df)
    print(f"  Sampling: n={n}")
    print(f"  AF2:      n={int(df['af2_rmsd'].notna().sum())}")
    print(f"  AF3:      n={int(df['af3_rmsd'].notna().sum())}")
    print(f"  VQE:      n={int(df['vqe_rmsd'].notna().sum())}")

    print(f"\n[DONE] Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
