# --*-- conding:utf-8 --*--
# @time:1/9/26 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_group_rmsd.py

# ------------------------------------------------------------
# Compare RMSD among four methods on the sampling-available set:
#   - Sampling-based (from backbone_rmsd_min.csv)
#   - AF2 (ColabFold) (from af2_rmsd_summary.txt)
#   - AF3 (AlphaFold3) (from af3_rmsd_summary.txt)
#   - VQE (from q_rmsd_summary.txt)
#
# Rule: ONLY plot pdb_ids that exist in sampling results.
# If other methods miss a pdb_id, skip those points.
#
# Inputs:
#   <project_root>/QDock_RMSD/
#     af2_rmsd_summary.txt
#     af3_rmsd_summary.txt
#     q_rmsd_summary.txt
#     backbone_rmsd_min.csv
#
# Outputs:
#   <project_root>/QDock_RMSD/merged_rmsd_on_sampling_ids.csv
#   <project_root>/QDock_RMSD/plots_compare/*.png/pdf
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
      pdb_id <tab or spaces> rmsd
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


# -----------------------------
# Plot helpers
# -----------------------------
def save_fig(out_png: Path, out_pdf: Path):
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def scatter_compare(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
):
    sub = df[[x_col, y_col]].dropna()
    if sub.empty:
        print(f"[SKIP] scatter {x_col} vs {y_col}: no overlapping points")
        return

    x = sub[x_col].to_numpy(float)
    y = sub[y_col].to_numpy(float)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(x, y, s=18, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linewidth=1.5)  # y=x reference
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{title} (n={len(sub)})")
    save_fig(out_png, out_pdf)


def boxplot_methods(
    df: pd.DataFrame,
    cols: Tuple[str, ...],
    labels: Tuple[str, ...],
    title: str,
    out_png: Path,
    out_pdf: Path,
):
    data = []
    used_labels = []
    counts = []
    for c, lab in zip(cols, labels):
        v = df[c].dropna().to_numpy(float)
        if len(v) == 0:
            continue
        data.append(v)
        used_labels.append(lab)
        counts.append(len(v))

    if not data:
        print("[SKIP] boxplot: no data")
        return

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot(data, labels=[f"{l}\n(n={n})" for l, n in zip(used_labels, counts)], showfliers=True)
    plt.ylabel("RMSD (Å)")
    plt.title(title)
    save_fig(out_png, out_pdf)


def per_case_sorted_lines(
    df: pd.DataFrame,
    sort_col: str,
    methods: Tuple[str, ...],
    labels: Tuple[str, ...],
    title: str,
    out_png: Path,
    out_pdf: Path,
):
    # Sort by sampling (or chosen sort_col)
    d = df.sort_values(sort_col).reset_index(drop=True).copy()
    x = np.arange(len(d), dtype=int)

    plt.figure(figsize=(10.5, 5.8))
    for col, lab in zip(methods, labels):
        y = d[col].to_numpy(float)
        # break line where NaN
        plt.plot(x, y, marker="o", markersize=3.2, linewidth=1.2, label=lab)

    plt.xlabel(f"Cases sorted by {sort_col}")
    plt.ylabel("RMSD (Å)")
    plt.title(title)
    plt.legend(frameon=False)
    save_fig(out_png, out_pdf)


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

    sampling_ids = set(s_df["pdb_id"].tolist())
    print(f"[LOAD] sampling cases: {len(sampling_ids)}")

    # Load other methods as dict
    af2 = read_rmsd_kv_txt(af2_path)
    af3 = read_rmsd_kv_txt(af3_path)
    vqe = read_rmsd_kv_txt(vqe_path)

    print(f"[LOAD] AF2 entries: {len(af2)} | AF3 entries: {len(af3)} | VQE entries: {len(vqe)}")

    # Build merged table only on sampling ids
    rows = []
    for pid in sorted(sampling_ids):
        rows.append(
            {
                "pdb_id": pid,
                "sampling_rmsd": safe_float(s_df.loc[s_df["pdb_id"] == pid, "sampling_rmsd"].iloc[0])
                if (s_df["pdb_id"] == pid).any()
                else None,
                "af2_rmsd": safe_float(af2.get(pid, None)),
                "af3_rmsd": safe_float(af3.get(pid, None)),
                "vqe_rmsd": safe_float(vqe.get(pid, None)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["sampling_rmsd"]).reset_index(drop=True)

    # Save merged table
    merged_out = rmsd_dir / "merged_rmsd_on_sampling_ids.csv"
    df.to_csv(merged_out, index=False)
    print(f"[SAVE] merged table: {merged_out} (n={len(df)})")

    # Output plots
    out_dir = rmsd_dir / "plots_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) boxplot distribution (on sampling-id set; each method uses available subset)
    boxplot_methods(
        df,
        cols=("sampling_rmsd", "af2_rmsd", "af3_rmsd", "vqe_rmsd"),
        labels=("Sampling", "AF2 (ColabFold)", "AF3", "VQE"),
        title="RMSD distribution on sampling-available cases",
        out_png=out_dir / "boxplot_methods.png",
        out_pdf=out_dir / "boxplot_methods.pdf",
    )

    # 2) scatter sampling vs others (paired where both exist)
    scatter_compare(
        df,
        x_col="sampling_rmsd",
        y_col="af2_rmsd",
        title="Sampling vs AF2 (ColabFold)",
        out_png=out_dir / "scatter_sampling_vs_af2.png",
        out_pdf=out_dir / "scatter_sampling_vs_af2.pdf",
    )
    scatter_compare(
        df,
        x_col="sampling_rmsd",
        y_col="af3_rmsd",
        title="Sampling vs AF3",
        out_png=out_dir / "scatter_sampling_vs_af3.png",
        out_pdf=out_dir / "scatter_sampling_vs_af3.pdf",
    )
    scatter_compare(
        df,
        x_col="sampling_rmsd",
        y_col="vqe_rmsd",
        title="Sampling vs VQE",
        out_png=out_dir / "scatter_sampling_vs_vqe.png",
        out_pdf=out_dir / "scatter_sampling_vs_vqe.pdf",
    )

    # 3) per-case curve (sorted by sampling)
    per_case_sorted_lines(
        df,
        sort_col="sampling_rmsd",
        methods=("sampling_rmsd", "af2_rmsd", "af3_rmsd", "vqe_rmsd"),
        labels=("Sampling", "AF2 (ColabFold)", "AF3", "VQE"),
        title="Per-case RMSD (sorted by Sampling RMSD; missing values are gaps)",
        out_png=out_dir / "per_case_sorted.png",
        out_pdf=out_dir / "per_case_sorted.pdf",
    )

    # Print quick counts (overlap sizes)
    n_sampling = len(df)
    n_af2 = int(df["af2_rmsd"].notna().sum())
    n_af3 = int(df["af3_rmsd"].notna().sum())
    n_vqe = int(df["vqe_rmsd"].notna().sum())
    print("\n[OVERLAP COUNTS] on sampling-id set")
    print(f"  Sampling: {n_sampling}")
    print(f"  AF2 available: {n_af2}")
    print(f"  AF3 available: {n_af3}")
    print(f"  VQE available: {n_vqe}")

    print(f"\n[DONE] Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
