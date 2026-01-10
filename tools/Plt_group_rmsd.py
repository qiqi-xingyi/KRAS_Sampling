# --*-- conding:utf-8 --*--
# @time:1/9/26 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_group_rmsd.py


# ------------------------------------------------------------
# QDock RMSD comparison plots (on sampling-available cases):
#   1) Grouped bar chart: per-case RMSD for ALL cases (Sampling defines the case set)
#   2) Boxplot: RMSD distribution comparison across methods
#
# Key requirements:
#   - Same method uses the SAME color in both plots
#   - Provide easy interfaces at the top to change:
#       (a) colors
#       (b) font sizes
#
# Inputs:
#   <project_root>/QDock_RMSD/
#     af2_rmsd_summary.txt
#     af3_rmsd_summary.txt
#     q_rmsd_summary.txt
#     backbone_rmsd_min.csv   (sampling-based min RMSD per pdb_id)
#
# Outputs:
#   <project_root>/QDock_RMSD/merged_rmsd_on_sampling_ids.csv
#   <project_root>/QDock_RMSD/plots_compare/
#     grouped_bar_all_cases.png/.pdf
#     boxplot_methods.png/.pdf
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

# 1) Method colors (keep consistent across all plots)
METHOD_STYLES = {
    # key must match columns in merged df
    "sampling_rmsd": {"label": "Our",        "color": "#6E6E6E"},  # gray
    "af2_rmsd":      {"label": "ColabFold", "color": "#2F6BFF"},  # blue
    "af3_rmsd":      {"label": "AF3",            "color": "#22A06B"},  # green
    "vqe_rmsd":      {"label": "VQE",            "color": "#E24A33"},  # red
}
# 2) Font sizes
FONT = {
    "title": 16,
    "axis_label": 14,
    "tick": 10,
    "legend": 11,
    "annot": 10,
}

# 3) Plot parameters
SORT_BY = "sampling_rmsd"   # "sampling_rmsd" or "pdb_id"
MAX_CASES = None           # None means plot all; set int for debugging
DPI = 300

# Grouped bar chart layout
BAR_WIDTH = 0.20
ROTATE_X = 90

# If you have many cases, you can tune these for readability
FIG_WIDTH_PER_CASE = 0.35   # width scale for bar chart
FIG_MIN_WIDTH = 12.0
FIG_BAR_HEIGHT = 6.0

# Boxplot options
SHOW_OUTLIERS = True


# ============================================================
# Internal helpers
# ============================================================

def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def apply_font_rcparams():
    # Global defaults; individual calls below also use FONT
    plt.rcParams["axes.titlesize"] = FONT["title"]
    plt.rcParams["axes.labelsize"] = FONT["axis_label"]
    plt.rcParams["xtick.labelsize"] = FONT["tick"]
    plt.rcParams["ytick.labelsize"] = FONT["tick"]
    plt.rcParams["legend.fontsize"] = FONT["legend"]


def read_rmsd_kv_txt(path: Path) -> Dict[str, float]:
    """
    Parse lines like:
      pdb_id <tab/spaces> rmsd
    Ignore empty lines and comment lines starting with '#'.
    """
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


def method_order() -> List[str]:
    # enforce consistent ordering across plots
    return ["sampling_rmsd", "af2_rmsd", "af3_rmsd", "vqe_rmsd"]


# ============================================================
# Plotting
# ============================================================

def plot_grouped_bar_all_cases(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
):
    """
    Grouped bar chart per pdb_id for all sampling-available cases.
    Missing values in other methods are skipped (no bar drawn for that method/case).
    """
    d = df.copy()

    if SORT_BY == "pdb_id":
        d = d.sort_values("pdb_id").reset_index(drop=True)
    else:
        d = d.sort_values("sampling_rmsd").reset_index(drop=True)

    if MAX_CASES is not None:
        d = d.iloc[: int(MAX_CASES)].reset_index(drop=True)

    n = len(d)
    x = np.arange(n, dtype=float)

    fig_w = max(FIG_MIN_WIDTH, FIG_WIDTH_PER_CASE * n)
    fig, ax = plt.subplots(figsize=(fig_w, FIG_BAR_HEIGHT))

    cols = method_order()
    offsets = np.linspace(-1.5 * BAR_WIDTH, 1.5 * BAR_WIDTH, num=len(cols))

    for col, off in zip(cols, offsets):
        if col not in d.columns:
            continue
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if not np.any(mask):
            continue

        label = METHOD_STYLES[col]["label"]
        color = METHOD_STYLES[col]["color"]
        ax.bar(
            x[mask] + off,
            y[mask],
            width=BAR_WIDTH,
            label=f"{label} (n={int(mask.sum())})",
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(d["pdb_id"].tolist(), rotation=ROTATE_X, ha="center")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Per-case RMSD comparison (only cases with Sampling results)", fontsize=FONT["title"])
    ax.legend(frameon=False, ncol=2, fontsize=FONT["legend"])

    # y-limit with padding
    y_all = []
    for col in cols:
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        y = y[~np.isnan(y)]
        if len(y) > 0:
            y_all.append(y)
    if y_all:
        ymax = float(np.max(np.concatenate(y_all)))
        ax.set_ylim(0.0, ymax * 1.10)

    # tick font size (in case rcParams not applied in some environments)
    ax.tick_params(axis="both", labelsize=FONT["tick"])

    save_fig(fig, out_png, out_pdf)


def plot_boxplot_methods(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
):
    """
    Boxplot comparing RMSD distributions across methods.
    Same method color as bar chart.
    Each method uses its available subset (still restricted to sampling-id set).
    """
    cols = method_order()

    data = []
    labels = []
    colors = []

    for col in cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(f"{METHOD_STYLES[col]['label']}\n(n={len(vals)})")
        colors.append(METHOD_STYLES[col]["color"])

    if not data:
        print("[SKIP] boxplot: no data")
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bp = ax.boxplot(
        data,
        labels=labels,
        showfliers=SHOW_OUTLIERS,
        patch_artist=True,   # allow box facecolor
    )

    # Color the boxes consistently with METHOD_STYLES
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.35)      # translucent fill for readability
        box.set_linewidth(1.2)

    # Color medians and whiskers for visibility
    for median in bp["medians"]:
        median.set_linewidth(1.6)
    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.2)
    for cap in bp["caps"]:
        cap.set_linewidth(1.2)

    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD distribution comparison (sampling-available cases)", fontsize=FONT["title"])
    ax.tick_params(axis="both", labelsize=FONT["tick"])

    save_fig(fig, out_png, out_pdf)


# ============================================================
# Main
# ============================================================

def main():
    apply_font_rcparams()

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

    # Load sampling-based mins
    s_df = pd.read_csv(sampling_csv)
    if "pdb_id" not in s_df.columns or "min_rmsd" not in s_df.columns:
        raise ValueError("backbone_rmsd_min.csv must contain columns: pdb_id, min_rmsd")

    s_df["pdb_id"] = s_df["pdb_id"].astype(str).str.strip().str.lower()
    s_df["sampling_rmsd"] = pd.to_numeric(s_df["min_rmsd"], errors="coerce")
    s_df = s_df.dropna(subset=["sampling_rmsd"]).reset_index(drop=True)

    sampling_ids = sorted(set(s_df["pdb_id"].tolist()))
    print(f"[LOAD] sampling cases: {len(sampling_ids)}")

    # Load other methods
    af2 = read_rmsd_kv_txt(af2_path)
    af3 = read_rmsd_kv_txt(af3_path)
    vqe = read_rmsd_kv_txt(vqe_path)
    print(f"[LOAD] AF2={len(af2)} AF3={len(af3)} VQE={len(vqe)}")

    # Merge only on sampling ids
    rows = []
    for pid in sampling_ids:
        # if duplicates exist, take first
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

    # Save merged table
    merged_out = rmsd_dir / "merged_rmsd_on_sampling_ids.csv"
    df.to_csv(merged_out, index=False)
    print(f"[SAVE] merged table: {merged_out} (n={len(df)})")

    # Output dir
    out_dir = rmsd_dir / "plots_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Grouped bar chart (all cases)
    plot_grouped_bar_all_cases(
        df,
        out_png=out_dir / "grouped_bar_all_cases.png",
        out_pdf=out_dir / "grouped_bar_all_cases.pdf",
    )

    # 2) Boxplot (same method colors)
    plot_boxplot_methods(
        df,
        out_png=out_dir / "boxplot_methods.png",
        out_pdf=out_dir / "boxplot_methods.pdf",
    )

    # Quick overlap stats
    print("\n[OVERLAP on sampling set]")
    print(f"  Sampling n = {len(df)}")
    print(f"  AF2 overlap = {int(df['af2_rmsd'].notna().sum())}")
    print(f"  AF3 overlap = {int(df['af3_rmsd'].notna().sum())}")
    print(f"  VQE overlap = {int(df['vqe_rmsd'].notna().sum())}")

    print(f"\n[DONE] Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()



