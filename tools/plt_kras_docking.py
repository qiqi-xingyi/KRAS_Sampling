# --*-- conding:utf-8 --*--
# @time:1/21/26 19:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_kras_docking.py

# --*-- coding:utf-8 --*--
# @time:1/21/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_KRAS_docking_affinity.py
#
# Docking visualization (journal style)
# D1: Per-target affinity distribution (strip + box) + best marker from pipeline_summary
# D2: ΔAffinity vs WT (median-based) with simple error bar (IQR/2) for readability
#
# Input:
#   <project_root>/docking_result/40_pipeline/pipeline_summary.csv
#   plus either:
#     - the per-target runs_long.csv referenced by vina_runs_long_csv, OR
#     - fallback: parse Vina logs under docking_result/30_dock/<target>/runs/seed_*/log.txt
#
# Output:
#   <project_root>/docking_result/40_pipeline/figs_docking/
#     D1_affinity_distribution.png/.pdf   (dpi=600)
#     D2_delta_vs_WT.png/.pdf             (dpi=600)

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
DPI = 600  # REQUIRED
FIGSIZE_D1 = (6.8, 4.8)
FIGSIZE_D2 = (5.8, 4.2)

# Colors (keep consistent with your mutation style)
COLORS = {
    "WT":   "#666666",
    "G12C": "#1f77b4",
    "G12D": "#ff7f0e",
}

GRID_COLOR = "#D9D9D9"
SPINE_COLOR = "#777777"

JITTER_WIDTH = 0.10
POINT_SIZE = 42
POINT_ALPHA = 0.85

BOX_WIDTH = 0.50
BOX_FACE_ALPHA = 0.18
BOX_EDGE_COLOR = "#2E2E2E"
BOX_EDGE_LINEWIDTH = 1.0
MEDIAN_LINEWIDTH = 2.0

BEST_MARKER = "D"       # diamond
BEST_MARKER_SIZE = 78
BEST_MARKER_EDGE = "#111111"

TARGET_ORDER = ["WT", "G12C", "G12D"]


# -----------------------------
# Paths
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def out_dir(root: Path) -> Path:
    p = root / "docking_result" / "40_pipeline" / "figs_docking"
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Parsing helpers
# -----------------------------
def infer_label(target_group_key: str) -> str:
    s = str(target_group_key).upper()
    if "G12C" in s:
        return "G12C"
    if "G12D" in s:
        return "G12D"
    return "WT"


def add_horizontal_grid(ax: plt.Axes):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, alpha=0.85)
    ax.xaxis.grid(False)


def journal_spines(ax: plt.Axes):
    for sp in ax.spines.values():
        sp.set_color(SPINE_COLOR)
        sp.set_linewidth(0.9)


def ax_boxplot_compat(ax: plt.Axes, data: List[np.ndarray], labels: List[str], **kwargs):
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)  # Matplotlib >= 3.9
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)       # Older Matplotlib


def detect_affinity_column(df: pd.DataFrame) -> Optional[str]:
    # Try common column names in priority order
    candidates = [
        "affinity", "vina_affinity", "score", "binding_affinity", "best_affinity",
        "mode1_affinity", "affinity_kcal_mol",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]

    # Heuristic: any column containing 'affin' or 'kcal' or 'score'
    for c in df.columns:
        cl = c.lower()
        if ("affin" in cl) or ("kcal" in cl) or (cl == "score"):
            return c
    return None


def parse_vina_log_mode1_affinity(log_path: Path) -> Optional[float]:
    """
    Parse AutoDock Vina log.txt and return affinity of mode 1 (kcal/mol).
    We only need the first mode row like:
       1       -7.943          0          0
    """
    if not log_path.exists():
        return None

    try:
        txt = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Find table row for mode 1. Robust to spaces.
    m = re.search(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", txt, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def load_run_affinities_from_runs_long(csv_path: Path) -> List[float]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    col = detect_affinity_column(df)
    if col is None:
        return []
    a = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    # If runs_long contains multiple modes per run, prefer mode==1 if present
    mode_col = None
    for c in df.columns:
        if c.lower() in ["mode", "rank", "pose", "model"]:
            mode_col = c
            break
    if mode_col is not None:
        m = pd.to_numeric(df[mode_col], errors="coerce")
        mask = (m == 1)
        if mask.any():
            a = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna().to_numpy(float)
    return a.tolist()


def load_run_affinities_fallback_logs(root: Path, target_group_key: str) -> List[float]:
    """
    Fallback if runs_long.csv is missing/empty:
      docking_result/30_dock/<target_group_key>/runs/seed_*/log.txt
    """
    base = root / "docking_result" / "30_dock" / target_group_key / "runs"
    if not base.exists():
        return []
    vals: List[float] = []
    for log_path in sorted(base.glob("seed_*/log.txt")):
        v = parse_vina_log_mode1_affinity(log_path)
        if v is not None:
            vals.append(v)
    return vals


# -----------------------------
# Plotting
# -----------------------------
def plot_d1_distribution(
    groups: List[str],
    runs: Dict[str, List[float]],
    best: Dict[str, float],
    out_png: Path,
    out_pdf: Path,
):
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=FIGSIZE_D1)

    data: List[np.ndarray] = []
    labels: List[str] = []
    for g in groups:
        a = np.array(runs.get(g, []), dtype=float)
        a = a[~np.isnan(a)]
        data.append(a)
        labels.append(f"{g}\n(n={len(a)})")

    bp = ax_boxplot_compat(
        ax,
        data,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=BOX_WIDTH,
    )

    # Style boxes
    for i, (box, g) in enumerate(zip(bp["boxes"], groups)):
        c = COLORS.get(g, "#999999")
        box.set_facecolor(c)
        box.set_alpha(BOX_FACE_ALPHA)
        box.set_edgecolor(BOX_EDGE_COLOR)
        box.set_linewidth(BOX_EDGE_LINEWIDTH)

    for med in bp["medians"]:
        med.set_linewidth(MEDIAN_LINEWIDTH)
        med.set_color(BOX_EDGE_COLOR)

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.0)
        whisker.set_color(BOX_EDGE_COLOR)

    for cap in bp["caps"]:
        cap.set_linewidth(1.0)
        cap.set_color(BOX_EDGE_COLOR)

    # Scatter jitter points + best marker
    for i, g in enumerate(groups, start=1):
        a = np.array(runs.get(g, []), dtype=float)
        a = a[~np.isnan(a)]
        if len(a) > 0:
            jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(a))
            ax.scatter(
                np.full(len(a), i) + jitter,
                a,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                color=COLORS.get(g, "#999999"),
                edgecolors="none",
                zorder=3,
            )

            # median label (small, unobtrusive)
            med = float(np.median(a))
            ax.text(
                i,
                med,
                f"{med:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111111",
                zorder=5,
            )

        # pipeline best marker
        if g in best and best[g] is not None:
            ax.scatter(
                [i],
                [best[g]],
                s=BEST_MARKER_SIZE,
                marker=BEST_MARKER,
                facecolors="none",
                edgecolors=BEST_MARKER_EDGE,
                linewidths=1.2,
                zorder=6,
            )

    add_horizontal_grid(ax)
    journal_spines(ax)

    ax.set_ylabel("Affinity (kcal/mol)")
    ax.set_title("Docking affinity across seeds (mode 1)")

    # Make plot direction intuitive: more negative is better, keep natural axis (down is better)
    # If you prefer "up is better", uncomment next line:
    # ax.invert_yaxis()

    # Legend note
    ax.text(
        0.02, 0.02,
        "Points: per-seed mode 1\nBox: IQR, line: median\n◇: best from pipeline_summary",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_d2_delta_vs_wt(
    groups: List[str],
    runs: Dict[str, List[float]],
    out_png: Path,
    out_pdf: Path,
):
    # Use median as the central estimator (robust)
    if "WT" not in runs or len(runs["WT"]) == 0:
        return

    wt = np.array(runs["WT"], dtype=float)
    wt = wt[~np.isnan(wt)]
    if len(wt) == 0:
        return

    wt_med = float(np.median(wt))

    xs: List[int] = []
    ys: List[float] = []
    yerr: List[float] = []
    xt: List[str] = []

    for i, g in enumerate(groups):
        if g == "WT":
            continue
        a = np.array(runs.get(g, []), dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            continue
        med = float(np.median(a))
        # simple spread indicator: half IQR (readable even with n=5)
        q1, q3 = np.percentile(a, [25, 75])
        spread = float(0.5 * (q3 - q1))
        xs.append(len(xs) + 1)
        ys.append(med - wt_med)   # Δ = target - WT (more negative => better)
        yerr.append(spread)
        xt.append(g)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_D2)

    ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)

    for x, g, y, e in zip(xs, xt, ys, yerr):
        ax.errorbar(
            x, y, yerr=e,
            fmt="o",
            markersize=7,
            color=COLORS.get(g, "#999999"),
            ecolor=COLORS.get(g, "#999999"),
            elinewidth=1.2,
            capsize=4,
            zorder=3,
        )
        ax.text(
            x, y,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#111111",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(xt)
    ax.set_ylabel("ΔAffinity vs WT (kcal/mol)\n(median, mode 1)")
    ax.set_title("Mutation effect on docking affinity")

    add_horizontal_grid(ax)
    journal_spines(ax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    root = project_root_from_tools_dir()

    summary_csv = root / "docking_result" / "40_pipeline" / "pipeline_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing input: {summary_csv}")

    df = pd.read_csv(summary_csv)

    required = ["target_group_key", "best_affinity", "vina_runs_long_csv"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"pipeline_summary.csv missing column: {c}")

    # Collect per-target info
    runs: Dict[str, List[float]] = {}
    best: Dict[str, float] = {}

    for _, row in df.iterrows():
        tg = str(row["target_group_key"]).strip()
        lab = infer_label(tg)

        b = pd.to_numeric(pd.Series([row["best_affinity"]]), errors="coerce").iloc[0]
        best[lab] = float(b) if pd.notna(b) else None

        runs_long_rel = str(row["vina_runs_long_csv"]).strip()
        runs_long_path = (root / runs_long_rel) if runs_long_rel else None

        vals: List[float] = []
        if runs_long_path is not None and runs_long_path.exists():
            vals = load_run_affinities_from_runs_long(runs_long_path)

        # fallback to logs if needed
        if len(vals) == 0:
            vals = load_run_affinities_fallback_logs(root, tg)

        runs[lab] = vals

    groups = [g for g in TARGET_ORDER if g in runs]

    od = out_dir(root)
    out_png1 = od / "D1_affinity_distribution.png"
    out_pdf1 = od / "D1_affinity_distribution.pdf"
    out_png2 = od / "D2_delta_vs_WT.png"
    out_pdf2 = od / "D2_delta_vs_WT.pdf"

    plot_d1_distribution(groups, runs, best, out_png1, out_pdf1)
    plot_d2_delta_vs_wt(groups, runs, out_png2, out_pdf2)

    print("[DONE] docking figures saved to:", od)
    print("  -", out_png1.name)
    print("  -", out_pdf1.name)
    if out_png2.exists():
        print("  -", out_png2.name)
        print("  -", out_pdf2.name)

    # Quick console summary
    print("\n[SUMMARY]")
    for g in groups:
        a = np.array(runs.get(g, []), dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            print(f"  {g}: no runs parsed")
            continue
        print(
            f"  {g}: n={len(a)}  median={np.median(a):.3f}  mean={np.mean(a):.3f}  best={best.get(g)}"
        )


if __name__ == "__main__":
    main()
