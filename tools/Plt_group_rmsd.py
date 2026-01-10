# --*-- conding:utf-8 --*--
# @time:1/9/26 22:59
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt_group_rmsd.py


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
    """Parse lines: pdb_id <tab/spaces> rmsd ; ignore # comments."""
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
# Correlation + color mapping
# -----------------------------
def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    # remove constant vectors
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_to_color(r: float, cmap=plt.cm.coolwarm):
    """Map r in [-1, 1] to a RGBA color. If r is nan, return gray."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return (0.55, 0.55, 0.55, 1.0)
    r = max(-1.0, min(1.0, float(r)))
    t = (r + 1.0) / 2.0  # [-1,1] -> [0,1]
    return cmap(t)


def save_fig(out_png: Path, out_pdf: Path):
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Plotting
# -----------------------------
def plot_bar_means_with_corr_colors(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
):
    """
    Bar chart: mean RMSD per method (on sampling-id set),
    colors for AF2/AF3/VQE are based on correlation with sampling on paired subset.
    """
    methods = [
        ("sampling_rmsd", "Sampling"),
        ("af2_rmsd", "AF2 (ColabFold)"),
        ("af3_rmsd", "AF3"),
        ("vqe_rmsd", "VQE"),
    ]

    means = []
    sems = []
    ns = []
    colors = []

    # sampling baseline color
    sampling_color = (0.45, 0.45, 0.45, 1.0)

    # compute correlation for each other method vs sampling (paired subset)
    for col, label in methods:
        vals = df[col].dropna().to_numpy(float)
        n = len(vals)
        ns.append(n)
        if n == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(float(np.mean(vals)))
            sems.append(float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)

        if col == "sampling_rmsd":
            colors.append(sampling_color)
        else:
            paired = df[["sampling_rmsd", col]].dropna()
            r = pearson_r(paired["sampling_rmsd"].to_numpy(float), paired[col].to_numpy(float)) if not paired.empty else float("nan")
            colors.append(corr_to_color(r))

    x = np.arange(len(methods))

    plt.figure(figsize=(8.2, 5.3))
    bars = plt.bar(x, means, yerr=sems, capsize=4, color=colors)
    plt.xticks(x, [m[1] for m in methods], rotation=0)
    plt.ylabel("RMSD (Å)")
    plt.title("Mean RMSD on sampling-available cases\n(bar color encodes corr with Sampling for AF2/AF3/VQE)")

    # annotate n
    for i, b in enumerate(bars):
        h = b.get_height()
        if np.isnan(h):
            continue
        plt.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"n={ns[i]}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # add a small colorbar legend for correlation
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02)
    cbar.set_label("Pearson r vs Sampling")

    save_fig(out_png, out_pdf)


def plot_scatter_corrcolored(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    out_png: Path,
    out_pdf: Path,
):
    sub = df[["sampling_rmsd", y_col]].dropna()
    if sub.empty:
        print(f"[SKIP] scatter sampling vs {y_col}: no overlap")
        return

    x = sub["sampling_rmsd"].to_numpy(float)
    y = sub[y_col].to_numpy(float)
    r = pearson_r(x, y)
    color = corr_to_color(r)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.05 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad

    plt.figure(figsize=(6.6, 6.0))
    plt.scatter(x, y, s=20, alpha=0.75, color=color, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linewidth=1.4, color=(0.3, 0.3, 0.3, 1.0))
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("Sampling RMSD (Å)")
    plt.ylabel(f"{y_label} RMSD (Å)")
    title_r = "nan" if np.isnan(r) else f"{r:.3f}"
    plt.title(f"Sampling vs {y_label}   r={title_r}   n={len(sub)}")

    # colorbar for reference (optional, lightweight)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02)
    cbar.set_label("Pearson r (color meaning)")

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

    # Load other methods
    af2 = read_rmsd_kv_txt(af2_path)
    af3 = read_rmsd_kv_txt(af3_path)
    vqe = read_rmsd_kv_txt(vqe_path)
    print(f"[LOAD] AF2={len(af2)} AF3={len(af3)} VQE={len(vqe)}")

    # Merge only on sampling ids
    rows = []
    for pid in sorted(sampling_ids):
        # take the first sampling row if duplicates exist
        srows = s_df.loc[s_df["pdb_id"] == pid, "sampling_rmsd"]
        s_val = safe_float(srows.iloc[0]) if len(srows) > 0 else None
        rows.append(
            {
                "pdb_id": pid,
                "sampling_rmsd": s_val,
                "af2_rmsd": safe_float(af2.get(pid, None)),
                "af3_rmsd": safe_float(af3.get(pid, None)),
                "vqe_rmsd": safe_float(vqe.get(pid, None)),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["sampling_rmsd"]).reset_index(drop=True)

    merged_out = rmsd_dir / "merged_rmsd_on_sampling_ids.csv"
    df.to_csv(merged_out, index=False)
    print(f"[SAVE] {merged_out} (n={len(df)})")

    out_dir = rmsd_dir / "plots_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) one bar chart
    plot_bar_means_with_corr_colors(
        df,
        out_png=out_dir / "bar_mean_rmsd_corrcolor.png",
        out_pdf=out_dir / "bar_mean_rmsd_corrcolor.pdf",
    )

    # 2) three scatters (corr-colored)
    plot_scatter_corrcolored(
        df,
        y_col="af2_rmsd",
        y_label="AF2 (ColabFold)",
        out_png=out_dir / "scatter_sampling_vs_af2_corrcolor.png",
        out_pdf=out_dir / "scatter_sampling_vs_af2_corrcolor.pdf",
    )
    plot_scatter_corrcolored(
        df,
        y_col="af3_rmsd",
        y_label="AF3",
        out_png=out_dir / "scatter_sampling_vs_af3_corrcolor.png",
        out_pdf=out_dir / "scatter_sampling_vs_af3_corrcolor.pdf",
    )
    plot_scatter_corrcolored(
        df,
        y_col="vqe_rmsd",
        y_label="VQE",
        out_png=out_dir / "scatter_sampling_vs_vqe_corrcolor.png",
        out_pdf=out_dir / "scatter_sampling_vs_vqe_corrcolor.pdf",
    )

    # quick overlaps
    print("\n[OVERLAP on sampling set]")
    print(f"  sampling n = {len(df)}")
    print(f"  AF2 overlap = {int(df['af2_rmsd'].notna().sum())}")
    print(f"  AF3 overlap = {int(df['af3_rmsd'].notna().sum())}")
    print(f"  VQE overlap = {int(df['vqe_rmsd'].notna().sum())}")

    print(f"\n[DONE] Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

