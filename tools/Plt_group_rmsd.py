# --*-- conding:utf-8 --*--
# @time:1/9/26 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_group_rmsd.py

# ------------------------------------------------------------
# RMSD comparison plots (paired set, ~20 cases) in a clean journal style.
#
# Label convention:
#   - Sampling-based method: "our"
#   - AF2 (ColabFold):       "colabfold"
#   - AF3:                   "af3"
#   - VQE:                   "vqe"
#
# Figures:
#   Fig1. Grouped bar chart (per-case, 4 methods)
#   Fig2. Paired delta RMSD (baseline - our): box + jitter (journal style)
#   FigS1. (Optional) 3-panel paired scatter: our vs colabfold/af3/vqe with y=x and r, rho
#
# Inputs:
#   <project_root>/QDock_RMSD/
#     af2_rmsd_summary.txt
#     af3_rmsd_summary.txt
#     q_rmsd_summary.txt
#     backbone_rmsd_min.csv   (our min RMSD per pdb_id)
#
# Outputs:
#   <project_root>/QDock_RMSD/merged_rmsd_on_our_ids.csv
#   <project_root>/QDock_RMSD/plots_compare/
#     fig1_grouped_bar.png/.pdf
#     fig2_delta_box_jitter.png/.pdf
#     figS1_scatter_triplet.png/.pdf
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

# Display labels
METHODS: List[Tuple[str, str]] = [
    ("our_rmsd",        "our"),
    ("colabfold_rmsd",  "colabfold"),
    ("af3_rmsd",        "af3"),
    ("vqe_rmsd",        "vqe"),
]

# Your palette (requested)
METHOD_COLORS = {
    "our_rmsd":        "#fd968f",
    "colabfold_rmsd":  "#40c2a8",
    "af3_rmsd":        "#73dbda",
    "vqe_rmsd":        "#0e4577",
}

# Font sizes
FONT = {
    "title": 12,
    "axis": 12,
    "tick": 10,
    "legend": 10,
    "annot": 9,
}

# Export
DPI = 300

# Grid style (horizontal only)
GRID_COLOR = "#D9D9D9"
GRID_LINEWIDTH = 0.8
GRID_ALPHA = 1.0

# Global spine style
SPINE_LINEWIDTH = 1.0

# ----------------------------
# Fig1: Grouped bar parameters
# ----------------------------
SORT_BY = "our_rmsd"   # "our_rmsd" or "pdb_id"
BAR_WIDTH = 0.18
GROUP_GAP = 0.35
ROTATE_X = 45

# Bar edge styling
BAR_EDGE_COLOR = "#2E2E2E"
BAR_EDGE_LINEWIDTH = 0.6

# Optional: put legend outside
FIG1_LEGEND_OUTSIDE = False
FIG1_LEGEND_NCOL = 4

# ----------------------------
# Fig2: Delta box + jitter (journal-style recommended)
# ----------------------------
DELTA_ORDER: List[Tuple[str, str]] = [
    ("colabfold_rmsd", "colabfold − our"),
    ("af3_rmsd",       "af3 − our"),
    ("vqe_rmsd",       "vqe − our"),
]

# Since we overlay all points via jitter, boxplot fliers are redundant
SHOW_OUTLIERS = False

# Jitter controls
RANDOM_SEED = 0
JITTER_WIDTH = 0.08
JITTER_POINT_SIZE = 24
JITTER_POINT_ALPHA = 0.75

# Box geometry + line widths
BOX_WIDTH = 0.55
BOX_EDGE_LINEWIDTH = 1.0
WHISKER_LINEWIDTH = 1.0
CAP_LINEWIDTH = 1.0
MEDIAN_LINEWIDTH = 2.0
BOX_EDGE_COLOR = "#2E2E2E"
BOX_FACE_ALPHA = 0.18

# Reference line at Δ=0
ZERO_LINE_COLOR = "#333333"
ZERO_LINE_LINEWIDTH = 1.2
ZERO_LINE_STYLE = "--"

# Median numeric annotation (recommended off for cleaner journal figure)
ANNOTATE_MEDIAN = False
MEDIAN_TEXT_FORMAT = "{:.2f}"
MEDIAN_TEXT_DY = 0.08
MEDIAN_TEXT_VA = "bottom"
MEDIAN_TEXT_COLOR = "#111111"

# Shorter text (recommended)
FIG2_TITLE = "ΔRMSD relative to our method"
FIG2_YLABEL = "ΔRMSD (Å)"

# ----------------------------
# FigS1: Scatter triplet
# ----------------------------
ENABLE_SCATTER_TRIPLET = True
SCATTER_POINT_SIZE = 34
SCATTER_POINT_ALPHA = 0.85
DIAG_LINE_COLOR = "#333333"
DIAG_LINEWIDTH = 1.2


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
    plt.rcParams["axes.linewidth"] = SPINE_LINEWIDTH
    plt.rcParams["savefig.bbox"] = "tight"


def add_horizontal_grid(ax: plt.Axes):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    ax.xaxis.grid(False)


def journal_spines(ax: plt.Axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    return pearson_r(rx, ry)


def ax_boxplot_compat(ax: plt.Axes, data: List[np.ndarray], labels: List[str], **kwargs):
    """
    Matplotlib compatibility:
      - In Matplotlib >= 3.9, boxplot() uses 'tick_labels'
      - Older versions use 'labels'
    This wrapper avoids MatplotlibDeprecationWarning and keeps backward compatibility.
    """
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)  # Matplotlib >= 3.9
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)       # Older Matplotlib


# ============================================================
# Plot: Fig1 grouped bar (per-case)
# ============================================================

def plot_fig1_grouped_bar(df: pd.DataFrame, out_png: Path, out_pdf: Path):
    d = df.copy()

    if SORT_BY == "pdb_id":
        d = d.sort_values("pdb_id").reset_index(drop=True)
    else:
        d = d.sort_values("our_rmsd").reset_index(drop=True)

    n = len(d)
    case_labels = d["pdb_id"].tolist()

    x = np.arange(n, dtype=float) * (1.0 + GROUP_GAP)

    fig_w = max(10.0, 0.50 * n)
    fig, ax = plt.subplots(figsize=(fig_w, 4.9))

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
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            alpha=1.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=ROTATE_X, ha="right")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title(f"Per-case RMSD comparison (paired set, n={n})")

    # y range
    y_all = []
    for col, _ in METHODS:
        yv = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        yv = yv[~np.isnan(yv)]
        if len(yv) > 0:
            y_all.append(yv)
    if y_all:
        ymax = float(np.max(np.concatenate(y_all)))
        ax.set_ylim(0.0, ymax * 1.12)

    add_horizontal_grid(ax)
    journal_spines(ax)

    if FIG1_LEGEND_OUTSIDE:
        ax.legend(
            frameon=False,
            ncol=FIG1_LEGEND_NCOL,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.0,
            handlelength=1.2,
        )
    else:
        ax.legend(frameon=False, ncol=FIG1_LEGEND_NCOL, loc="upper right")

    save_fig(fig, out_png, out_pdf)


# ============================================================
# Plot: Fig2 delta RMSD (baseline - our) box + jitter
# ============================================================

def plot_fig2_delta_box_jitter(df: pd.DataFrame, out_png: Path, out_pdf: Path):
    rng = np.random.default_rng(RANDOM_SEED)

    deltas: List[np.ndarray] = []
    tick_labels: List[str] = []
    colors: List[str] = []

    for base_col, lab in DELTA_ORDER:
        sub = df[["our_rmsd", base_col]].dropna()
        if sub.empty:
            continue
        delta = sub[base_col].to_numpy(float) - sub["our_rmsd"].to_numpy(float)
        deltas.append(delta)
        tick_labels.append(f"{lab}\n(n={len(delta)})")
        colors.append(METHOD_COLORS.get(base_col, "#999999"))

    if not deltas:
        print("[SKIP] Fig2: no paired deltas available.")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.9))

    bp = ax_boxplot_compat(
        ax,
        deltas,
        labels=tick_labels,
        showfliers=SHOW_OUTLIERS,
        patch_artist=True,
        widths=BOX_WIDTH,
    )

    # style box elements
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(BOX_FACE_ALPHA)
        box.set_linewidth(BOX_EDGE_LINEWIDTH)
        box.set_edgecolor(BOX_EDGE_COLOR)

    for median in bp["medians"]:
        median.set_linewidth(MEDIAN_LINEWIDTH)
        median.set_color(BOX_EDGE_COLOR)

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(WHISKER_LINEWIDTH)
        whisker.set_color(BOX_EDGE_COLOR)

    for cap in bp["caps"]:
        cap.set_linewidth(CAP_LINEWIDTH)
        cap.set_color(BOX_EDGE_COLOR)

    # jitter points + optional median annotation
    for i, (delta, c) in enumerate(zip(deltas, colors), start=1):
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(delta))
        ax.scatter(
            np.full(len(delta), i) + jitter,
            delta,
            s=JITTER_POINT_SIZE,
            alpha=JITTER_POINT_ALPHA,
            color=c,
            edgecolors="none",
            zorder=3,
        )

        if ANNOTATE_MEDIAN:
            med = float(np.median(delta))
            ax.text(
                i,
                med + MEDIAN_TEXT_DY,
                MEDIAN_TEXT_FORMAT.format(med),
                ha="center",
                va=MEDIAN_TEXT_VA,
                fontsize=FONT["annot"],
                color=MEDIAN_TEXT_COLOR,
                zorder=4,
            )

    ax.axhline(
        0.0,
        linewidth=ZERO_LINE_LINEWIDTH,
        linestyle=ZERO_LINE_STYLE,
        color=ZERO_LINE_COLOR,
        zorder=2,
    )

    ax.set_ylabel(FIG2_YLABEL)
    ax.set_title(FIG2_TITLE)

    add_horizontal_grid(ax)
    journal_spines(ax)

    save_fig(fig, out_png, out_pdf)


# ============================================================
# Plot: FigS1 scatter triplet
# ============================================================

def plot_figs1_scatter_triplet(df: pd.DataFrame, out_png: Path, out_pdf: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharex=False, sharey=False)

    comps = [
        ("colabfold_rmsd", "colabfold"),
        ("af3_rmsd", "af3"),
        ("vqe_rmsd", "vqe"),
    ]

    for ax, (col, name) in zip(axes, comps):
        sub = df[["our_rmsd", col]].dropna()
        if sub.empty:
            ax.axis("off")
            continue

        x = sub["our_rmsd"].to_numpy(float)
        y = sub[col].to_numpy(float)
        r = pearson_r(x, y)
        rho = spearman_rho(x, y)

        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo + 1e-12)
        lo -= pad
        hi += pad

        ax.scatter(
            x, y,
            s=SCATTER_POINT_SIZE,
            alpha=SCATTER_POINT_ALPHA,
            color=METHOD_COLORS.get(col, "#999999"),
            edgecolors="none",
            zorder=3,
        )
        ax.plot([lo, hi], [lo, hi], color=DIAG_LINE_COLOR, linewidth=DIAG_LINEWIDTH, zorder=2)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("our RMSD (Å)")
        ax.set_ylabel(f"{name} RMSD (Å)")

        title_r = "nan" if np.isnan(r) else f"{r:.3f}"
        title_rho = "nan" if np.isnan(rho) else f"{rho:.3f}"
        ax.set_title(f"{name}\n n={len(sub)}  r={title_r}  ρ={title_rho}")

        add_horizontal_grid(ax)
        journal_spines(ax)

    fig.suptitle("Paired scatter comparisons (our vs baselines)", fontsize=FONT["title"])
    save_fig(fig, out_png, out_pdf)


# ============================================================
# Main
# ============================================================

def main():
    apply_rcparams()

    root = project_root_from_tools_dir()
    rmsd_dir = root / "QDock_RMSD"

    # input files
    colabfold_path = rmsd_dir / "af2_rmsd_summary.txt"
    af3_path = rmsd_dir / "af3_rmsd_summary.txt"
    vqe_path = rmsd_dir / "q_rmsd_summary.txt"
    our_csv = rmsd_dir / "backbone_rmsd_min.csv"

    if not rmsd_dir.exists():
        raise FileNotFoundError(f"Missing folder: {rmsd_dir}")
    if not our_csv.exists():
        raise FileNotFoundError(f"Missing our results csv: {our_csv}")

    # load our results
    s_df = pd.read_csv(our_csv)
    if "pdb_id" not in s_df.columns or "min_rmsd" not in s_df.columns:
        raise ValueError("backbone_rmsd_min.csv must contain columns: pdb_id, min_rmsd")

    s_df["pdb_id"] = s_df["pdb_id"].astype(str).str.strip().str.lower()
    s_df["our_rmsd"] = pd.to_numeric(s_df["min_rmsd"], errors="coerce")
    s_df = s_df.dropna(subset=["our_rmsd"]).reset_index(drop=True)

    our_ids = sorted(set(s_df["pdb_id"].tolist()))
    print(f"[LOAD] our cases: {len(our_ids)}")

    # load baselines
    colabfold = read_rmsd_kv_txt(colabfold_path)
    af3 = read_rmsd_kv_txt(af3_path)
    vqe = read_rmsd_kv_txt(vqe_path)
    print(f"[LOAD] colabfold={len(colabfold)} af3={len(af3)} vqe={len(vqe)}")

    # merge only on our ids
    rows = []
    for pid in our_ids:
        our_rows = s_df.loc[s_df["pdb_id"] == pid, "our_rmsd"]
        our_val = safe_float(our_rows.iloc[0]) if len(our_rows) > 0 else None

        rows.append(
            {
                "pdb_id": pid,
                "our_rmsd": our_val,
                "colabfold_rmsd": safe_float(colabfold.get(pid)),
                "af3_rmsd": safe_float(af3.get(pid)),
                "vqe_rmsd": safe_float(vqe.get(pid)),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["our_rmsd"]).reset_index(drop=True)

    # save merged
    merged_out = rmsd_dir / "merged_rmsd_on_our_ids.csv"
    df.to_csv(merged_out, index=False)
    print(f"[SAVE] {merged_out} (n={len(df)})")

    # output dir
    out_dir = rmsd_dir / "plots_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fig1_grouped_bar(
        df,
        out_png=out_dir / "fig1_grouped_bar.png",
        out_pdf=out_dir / "fig1_grouped_bar.pdf",
    )

    plot_fig2_delta_box_jitter(
        df,
        out_png=out_dir / "fig2_delta_box_jitter.png",
        out_pdf=out_dir / "fig2_delta_box_jitter.pdf",
    )

    if ENABLE_SCATTER_TRIPLET:
        plot_figs1_scatter_triplet(
            df,
            out_png=out_dir / "figS1_scatter_triplet.png",
            out_pdf=out_dir / "figS1_scatter_triplet.pdf",
        )

    # coverage check
    print("\n[PAIRED COVERAGE]")
    n = len(df)
    print(f"  our:        n={n}")
    print(f"  colabfold:  n={int(df['colabfold_rmsd'].notna().sum())}")
    print(f"  af3:        n={int(df['af3_rmsd'].notna().sum())}")
    print(f"  vqe:        n={int(df['vqe_rmsd'].notna().sum())}")

    print(f"\n[DONE] Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()


