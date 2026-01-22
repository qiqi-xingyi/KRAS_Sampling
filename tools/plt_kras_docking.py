# --*-- coding:utf-8 --*--
# @time:1/21/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_docking_waterfall_per_target.py
#
# Docking visualization (per target, no cross-target comparison)
# Panel A only: Energy waterfall / rank plot
# - For each target_group_key:
#     parse all docking_result/30_dock/<target>/runs/seed_*/log.txt
#     extract affinities for ALL modes (default: mode 1..20)
#     sort affinities ascending (more negative = better)
#     plot rank vs affinity with line + points
#     draw median + Q1/Q3 horizontal lines
#     annotate best affinity + its seed/mode
#
# Input:
#   <project_root>/docking_result/40_pipeline/pipeline_summary.csv
#   plus logs:
#   <project_root>/docking_result/30_dock/<target_group_key>/runs/seed_*/log.txt
#
# Output:
#   <project_root>/docking_result/figs_docking_per_target/
#     D_A_waterfall_<target_group_key>.png/.pdf   (dpi=600)
#   <project_root>/docking_result/figs_docking_per_target/waterfall_manifest.csv
#
# Notes:
# - This script intentionally does NOT compare different targets on the same axis.
# - If some seeds are missing logs, they are skipped.

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

FIGSIZE = (7.6, 4.8)

GRID_COLOR = "#D9D9D9"
SPINE_COLOR = "#777777"
LINE_COLOR = "#222222"
POINT_COLOR = "#222222"

POINT_SIZE = 22
POINT_ALPHA = 0.80
LINE_WIDTH = 1.2
LINE_ALPHA = 0.60

STAT_LINE_COLOR = "#666666"
STAT_LINE_WIDTH = 1.1
STAT_LINE_STYLE = "--"

TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 10
ANNOT_FONTSIZE = 10

# Limit to first K modes per seed; set None to parse all table rows
MAX_MODES_PER_SEED: Optional[int] = None  # e.g., 20

# Filename safety
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


# -----------------------------
# Paths
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def out_dir(root: Path) -> Path:
    p = root / "docking_result" / "figs_docking_per_target"
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Plot helpers
# -----------------------------
def add_horizontal_grid(ax: plt.Axes):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.9, alpha=0.85)
    ax.xaxis.grid(False)


def journal_spines(ax: plt.Axes):
    for sp in ax.spines.values():
        sp.set_color(SPINE_COLOR)
        sp.set_linewidth(0.9)


def safe_filename(name: str) -> str:
    s = SAFE_NAME_RE.sub("_", name.strip())
    return s[:180] if len(s) > 180 else s


# -----------------------------
# Vina log parsing
# -----------------------------
TABLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)

def parse_vina_log_table(log_path: Path) -> List[Dict[str, float]]:
    """
    Parse the Vina result table:
      mode | affinity | rmsd l.b. | rmsd u.b.
    Returns list of dicts with mode, affinity, rmsd_lb, rmsd_ub
    """
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    rows: List[Dict[str, float]] = []
    in_table = False
    for ln in lines:
        s = ln.rstrip("\n")
        # detect header line
        if "mode" in s.lower() and "affinity" in s.lower() and "rmsd" in s.lower():
            in_table = True
            continue
        if not in_table:
            continue

        m = TABLE_ROW_RE.match(s)
        if not m:
            # stop when leaving table
            # (after table, vina prints blank line or something else)
            if rows and s.strip() == "":
                break
            continue

        mode = int(m.group(1))
        aff = float(m.group(2))
        lb = float(m.group(3))
        ub = float(m.group(4))
        rows.append({"mode": mode, "affinity": aff, "rmsd_lb": lb, "rmsd_ub": ub})

        if MAX_MODES_PER_SEED is not None and len(rows) >= int(MAX_MODES_PER_SEED):
            break

    return rows


def collect_target_from_logs(root: Path, target_group_key: str) -> pd.DataFrame:
    """
    Collect all modes from all seeds for one target, from logs:
      docking_result/30_dock/<target>/runs/seed_*/log.txt
    Output columns: target, seed, mode, affinity, rmsd_lb, rmsd_ub, log_path
    """
    base = root / "docking_result" / "30_dock" / target_group_key / "runs"
    if not base.exists():
        return pd.DataFrame(columns=["target", "seed", "mode", "affinity", "rmsd_lb", "rmsd_ub", "log_path"])

    all_rows: List[Dict[str, object]] = []
    for seed_dir in sorted(base.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed = seed_dir.name.replace("seed_", "")
        log_path = seed_dir / "log.txt"
        rows = parse_vina_log_table(log_path)
        for r in rows:
            all_rows.append(
                {
                    "target": target_group_key,
                    "seed": seed,
                    "mode": int(r["mode"]),
                    "affinity": float(r["affinity"]),
                    "rmsd_lb": float(r["rmsd_lb"]),
                    "rmsd_ub": float(r["rmsd_ub"]),
                    "log_path": str(log_path),
                }
            )

    return pd.DataFrame(all_rows)


# -----------------------------
# Plot (Waterfall)
# -----------------------------
def plot_waterfall(df: pd.DataFrame, title: str, out_png: Path, out_pdf: Path) -> Dict[str, object]:
    """
    df must contain: affinity, seed, mode
    Returns summary stats for manifest.
    """
    if df.empty:
        return {"ok": False}

    d = df.copy()
    d = d.dropna(subset=["affinity"]).copy()
    if d.empty:
        return {"ok": False}

    # sort by affinity (more negative = better)
    d = d.sort_values("affinity", ascending=True).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1, dtype=int)

    a = d["affinity"].to_numpy(float)
    best_aff = float(a.min())
    best_row = d.iloc[int(np.argmin(a))]
    best_seed = str(best_row["seed"])
    best_mode = int(best_row["mode"])

    q1 = float(np.percentile(a, 25))
    med = float(np.median(a))
    q3 = float(np.percentile(a, 75))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        d["rank"].to_numpy(int),
        a,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        alpha=LINE_ALPHA,
        zorder=2,
    )
    ax.scatter(
        d["rank"].to_numpy(int),
        a,
        s=POINT_SIZE,
        color=POINT_COLOR,
        alpha=POINT_ALPHA,
        edgecolors="none",
        zorder=3,
    )

    # Stats lines
    ax.axhline(med, color=STAT_LINE_COLOR, linestyle=STAT_LINE_STYLE, linewidth=STAT_LINE_WIDTH, zorder=1)
    ax.axhline(q1,  color=STAT_LINE_COLOR, linestyle=STAT_LINE_STYLE, linewidth=STAT_LINE_WIDTH, alpha=0.75, zorder=1)
    ax.axhline(q3,  color=STAT_LINE_COLOR, linestyle=STAT_LINE_STYLE, linewidth=STAT_LINE_WIDTH, alpha=0.75, zorder=1)

    # Annotation for best
    ax.scatter([1], [best_aff], s=70, marker="D", facecolors="none", edgecolors="#111111", linewidths=1.2, zorder=4)
    ax.text(
        0.98,
        0.05,
        f"best = {best_aff:.3f} kcal/mol\n(seed {best_seed}, mode {best_mode})\nmedian = {med:.3f}\nIQR = [{q1:.3f}, {q3:.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOT_FONTSIZE,
        color="#111111",
    )

    add_horizontal_grid(ax)
    journal_spines(ax)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Rank (all poses, sorted by affinity)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Affinity (kcal/mol)", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {
        "ok": True,
        "n_poses": int(len(d)),
        "best_affinity": best_aff,
        "best_seed": best_seed,
        "best_mode": best_mode,
        "median": med,
        "q1": q1,
        "q3": q3,
        "out_png": str(out_png),
        "out_pdf": str(out_pdf),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    root = project_root_from_tools_dir()

    summary_csv = root / "docking_result" / "40_pipeline" / "pipeline_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing input: {summary_csv}")

    df_sum = pd.read_csv(summary_csv)
    if "target_group_key" not in df_sum.columns or "ligand_resname" not in df_sum.columns:
        raise ValueError("pipeline_summary.csv must contain: target_group_key, ligand_resname")

    od = out_dir(root)

    manifest_rows: List[Dict[str, object]] = []

    for _, row in df_sum.iterrows():
        target = str(row["target_group_key"]).strip()
        lig = str(row["ligand_resname"]).strip()

        df = collect_target_from_logs(root, target)
        if df.empty:
            print(f"[WARN] No logs parsed for target: {target}")
            manifest_rows.append({"target_group_key": target, "ok": False, "reason": "no_logs"})
            continue

        title = f"{target} ({lig}) docking energy waterfall"

        fname = safe_filename(target)
        out_png = od / f"D_A_waterfall_{fname}.png"
        out_pdf = od / f"D_A_waterfall_{fname}.pdf"

        info = plot_waterfall(df, title=title, out_png=out_png, out_pdf=out_pdf)
        info["target_group_key"] = target
        info["ligand_resname"] = lig

        manifest_rows.append(info)
        print(f"[SAVE] {out_png.name}")

    # Save manifest
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = od / "waterfall_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("\n[DONE] Saved figures to:", od)
    print("  -", manifest_path.name)


if __name__ == "__main__":
    main()
