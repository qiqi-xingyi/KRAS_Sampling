# --*-- conding:utf-8 --*--
# @time:1/9/26 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_group_rmsd.py

# ------------------------------------------------------------
# QDock RMSD comparison plots (on sampling-available cases):
#   1) Grouped bar chart: per-case RMSD for ALL cases (sampling has)
#      - x axis: pdb_id (sorted by sampling RMSD)
#      - bars per case: Sampling / AF2 / AF3 / VQE (missing -> skip that bar)
#   2) Boxplot: RMSD distribution comparison across methods
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


# -----------------------------
# User parameters
# -----------------------------
SORT_BY = "sampling_rmsd"   # "sampling_rmsd" or "pdb_id"
MAX_CASES = None           # None means plot all. You can set e.g. 120 for debugging.

# Style tweaks
BAR_WIDTH = 0.20
ROTATE_X = 90
DPI = 300


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


# -----------------------------
# Parsing utilities
# -----------------------------
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


# -----------------------------
# Plotting
# -----------------------------
def plot_grouped_bar_all_cases(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
):
    """
    Grouped bar chart per pdb_id for all sampling-available cases.
    Missing values in other methods are skipped (no bar drawn for that method/case).
    """
    methods: List[Tuple[str, str]] = [
        ("sampling_rmsd", "Sampling"),
        ("af2_rmsd", "AF2 (ColabFold)"),
        ("af3_rmsd", "AF3"),
        ("vqe_rmsd", "VQE"),
    ]

    # fixed colors for readability (no seaborn)
    colors = {
        "sampling_rmsd": (0.45, 0.45, 0.45, 1.0),  # gray
        "af2_rmsd": (0.20, 0.40, 0.80, 1.0),       # blue-ish
        "af3_rmsd": (0.20, 0.70, 0.35, 1.0),       # green-ish
        "vqe_rmsd": (0.85, 0.35, 0.20, 1.0),       # red-ish
    }

    d = df.copy()

    # sort
    if SORT_BY == "pdb_id":
        d = d.sort_values("pdb_id").reset_index(drop=True)
    else:
        d = d.sort_values("sampling_rmsd").reset_index(drop=True)

    if MAX_CASES is not None:
        d = d.iloc[: int(MAX_CASES)].reset_index(drop=True)

    n = len(d)
    x = np.arange(n, dtype=float)

    # dynamic figure width: ensure labels are readable
    fig_w = max(12.0, 0.35 * n)  # increase with #cases
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    offsets = np.linspace(-1.5 * BAR_WIDTH, 1.5 * BAR_WIDTH, num=len(methods))

    for (col, label), off in zip(methods, offsets):
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if np.any(mask):
            ax.bar(
                x[mask] + off,
                y[mask],
                width=BAR_WIDTH,
                label=f"{label} (n={int(mask.sum())})",
                color=colors.get(col, None),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(d["pdb_id"].tolist(), rotation=ROTATE_X, ha="center")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("Per-case RMSD comparison (only cases with Sampling results)")
    ax.legend(frameon=False, ncol=2)

    # y limit a bit padded
    y_all = []
    for col, _ in methods:
        y = pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)
        y = y[~np.isnan(y)]
        if len(y) > 0:
            y_all.append(y)
    if y_all:
        ymax = float(np.max(np.concatenate(y_all)))
        ax.set_ylim(0.0, ymax * 1.10)

    save_fig(fig, out_png, out_pdf)


def plot_boxplot_methods(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
):
    """
    Boxplot comparing RMSD distributions across methods
    (still on the sampling-id set; each method uses its available subset).
    """
    methods: List[Tuple[str, str]] = [
        ("sampling_rmsd", "Sampling"),
        ("af2_rmsd", "AF2 (ColabFold)"),
        ("af3_rmsd", "AF3"),
        ("vqe_rmsd", "VQE"),
    ]

    data = []
    labels = []
    for col, name in methods:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(f"{name}\n(n={len(vals)})")

    if not data:
        print("[SKIP] boxplot: no data")
        return

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("RMSD distribution comparison (sampling-available cases)")
    save_fig(fig, out_png, out_pdf)


# -----------------------------
# Main
# -----------------------------
def main():
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
        # if duplicates exist, take the first sampling row
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

    # 1) Grouped bar (all cases)
    plot_grouped_bar_all_cases(
        df,
        out_png=out_dir / "grouped_bar_all_cases.png",
        out_pdf=out_dir / "grouped_bar_all_cases.pdf",
    )

    # 2) Boxplot
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


