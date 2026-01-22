# --*-- coding:utf-8 --*--
# @time:1/21/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_docking_waterfall_per_target.py
#
# Docking visualization (per target, no cross-target comparison)
# Panel A only: Energy waterfall / rank plot
#
# Input:
#   <project_root>/docking_result/40_pipeline/pipeline_summary.csv
#   plus logs:
#   <project_root>/docking_result/30_dock/<target_group_key>/runs/seed_*/log.txt
#
# Output:
#   <project_root>/docking_result/figs_docking_per_target/
#     D_A_waterfall_<target_group_key>.png/.pdf   (dpi=600)
#     waterfall_manifest.csv
#
# Why previous version failed:
# - Vina header splits "mode/affinity" and "rmsd" across two lines.
# - We now enter the table after the dashed separator line "-----+...".

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

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

SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# Matches numeric rows in the Vina table:
#    1       -7.943          0          0
TABLE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)


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
# Vina log parsing (FIXED)
# -----------------------------
def parse_vina_log_table(log_path: Path) -> List[Dict[str, float]]:
    """
    Parse AutoDock Vina log.txt and extract the result table:
      mode | affinity | rmsd l.b. | rmsd u.b.
    Robust strategy:
      - start parsing ONLY after the dashed separator line containing '-----+'
      - then collect any rows matching: mode affinity lb ub
    """
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    # Find the separator line that precedes numeric rows
    start_idx = None
    for i, ln in enumerate(lines):
        if "-----+" in ln:
            start_idx = i + 1
            break

    if start_idx is None:
        return []

    rows: List[Dict[str, float]] = []
    for ln in lines[start_idx:]:
        s = ln.rstrip("\n")
        m = TABLE_ROW_RE.match(s)
        if not m:
            # stop when we have started collecting and then hit a blank line
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
    Collect all modes from all seeds for one target:
      docking_result/30_dock/<target>/runs/seed_*/log.txt
    """
    base = root / "docking_result" / "30_dock" / target_group_key / "runs"
    if not base.exists():
        return pd.DataFrame(columns=["target", "seed", "mode", "affinity", "rmsd_lb", "rmsd_ub", "log_path"])

    all_rows: List[Dict[str, object]] = []
    seed_dirs = sorted([p for p in base.glob("seed_*") if p.is_dir()])

    for seed_dir in seed_dirs:
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
    if df.empty:
        return {"ok": False}

    d = df.dropna(subset=["affinity"]).copy()
    if d.empty:
        return {"ok": False}

    # sort by affinity (more negative = better)
    d = d.sort_values("affinity", ascending=True).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1, dtype=int)

    a = d["affinity"].to_numpy(float)
    best_idx = int(np.argmin(a))
    best_aff = float(a[best_idx])
    best_row = d.iloc[best_idx]
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

    # Best marker at rank 1 location (since sorted)
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
            manifest_rows.append({"target_group_key": target, "ok": False, "reason": "no_rows_after_parse"})
            continue

        title = f"{target} ({lig}) docking energy waterfall"
        fname = safe_filename(target)
        out_png = od / f"D_A_waterfall_{fname}.png"
        out_pdf = od / f"D_A_waterfall_{fname}.pdf"

        info = plot_waterfall(df, title=title, out_png=out_png, out_pdf=out_pdf)
        info["target_group_key"] = target
        info["ligand_resname"] = lig
        manifest_rows.append(info)

        print(f"[SAVE] {out_png.name}  (poses={info.get('n_poses')})")

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = od / "waterfall_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\n[DONE] Saved to:", od)
    print("  -", manifest_path.name)


if __name__ == "__main__":
    main()
