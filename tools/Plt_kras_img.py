#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# =========================
# Config
# =========================
BASINS = list(range(7))      # 0..6
DPI = 600

# Sector-filled pie geometry
R_OUT = 1.0
R_IN = 0.28                  # inner radius of the donut
GAP_DEG = 1.0                # small gap between sectors (degrees) for readability

# Output folders
SUB_PIES = "sector_pies"
SUB_SUMMARY = "summary"
SUB_WATERFALL = "waterfall"
SUB_QC = "qc_optional"
SUB_DENSITY_ARCHIVE = "density_existing"  # optional copy of old density figs

# Optional: copy your old density maps into KRAS_analysis/figs/
COPY_EXISTING_DENSITY = False


# =========================
# Path helpers
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_kras_analysis_dir(start: Path) -> Path:
    p = start.resolve()
    for _ in range(20):
        cand = p / "KRAS_analysis"
        if cand.exists() and cand.is_dir():
            return cand
        p = p.parent
    raise FileNotFoundError(f"Cannot find KRAS_analysis by searching upward from: {start}")


def pick_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    s = set(columns)
    for c in candidates:
        if c in s:
            return c
    return None


# =========================
# Data loading
# =========================
def normalize_label(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    u = s.upper()
    if u in {"WT", "WILD", "WILDTYPE", "WILD_TYPE"}:
        return "WT"
    if u in {"G12C", "KRAS_G12C"}:
        return "G12C"
    if u in {"G12D", "KRAS_G12D"}:
        return "G12D"
    return s


def coerce_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def load_points(points_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(points_csv)

    basin_col = pick_first_existing(df.columns.tolist(), ["basin_id", "basin", "cluster_id"])
    if basin_col is None:
        raise ValueError("points_enriched.csv: cannot find basin column (basin_id / basin / cluster_id).")

    case_col = pick_first_existing(df.columns.tolist(), ["case_id", "pdb_id", "example_id", "system", "target_group_key"])
    if case_col is None:
        raise ValueError("points_enriched.csv: cannot find case column (case_id / pdb_id / example_id / system / target_group_key).")

    label_col = pick_first_existing(df.columns.tolist(), ["label", "condition", "mut", "genotype", "state"])
    if label_col is None:
        raise ValueError("points_enriched.csv: cannot find label column (label / condition / mut / genotype / state).")

    w_col = pick_first_existing(df.columns.tolist(), ["weight", "p_mass", "prob", "mass", "count"])
    if w_col is None:
        raise ValueError("points_enriched.csv: cannot find weight column (weight/p_mass/prob/mass/count).")

    out = df[[case_col, label_col, basin_col, w_col]].copy()
    out.rename(columns={case_col: "case_id", label_col: "label", basin_col: "basin_id", w_col: "w"}, inplace=True)

    out["label"] = out["label"].map(normalize_label)
    out["basin_id"] = coerce_numeric(out["basin_id"], default=np.nan).astype("Int64")
    out["w"] = coerce_numeric(out["w"], default=0.0)

    out = out.dropna(subset=["basin_id"])
    out = out[out["w"] > 0]
    out["basin_id"] = out["basin_id"].astype(int)

    out = out[out["basin_id"].isin(BASINS)].copy()
    if out.empty:
        raise ValueError("After filtering to basin 0..6, points_enriched.csv has no rows.")

    return out


def load_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


# =========================
# Aggregations
# =========================
def dist_case_label_basin(points: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy table:
      case_id, label, basin_id, p
    where p is normalized within each (case_id, label).
    """
    g = points.groupby(["case_id", "label", "basin_id"], as_index=False)["w"].sum()
    tot = g.groupby(["case_id", "label"], as_index=False)["w"].sum().rename(columns={"w": "w_sum"})
    g = g.merge(tot, on=["case_id", "label"], how="left")
    g["p"] = g["w"] / (g["w_sum"] + 1e-12)
    return g[["case_id", "label", "basin_id", "p"]]


def dist_label_basin(points: pd.DataFrame) -> pd.DataFrame:
    """
    Returns label-level basin distribution:
      label, basin_id, p
    normalized within each label.
    """
    g = points.groupby(["label", "basin_id"], as_index=False)["w"].sum()
    tot = g.groupby(["label"], as_index=False)["w"].sum().rename(columns={"w": "w_sum"})
    g = g.merge(tot, on="label", how="left")
    g["p"] = g["w"] / (g["w_sum"] + 1e-12)
    return g[["label", "basin_id", "p"]]


def pivot_case_label(g: pd.DataFrame, case_id: str, label: str) -> np.ndarray:
    """
    g: tidy case-label-basin distribution
    return probs length 7 for basins 0..6
    """
    sub = g[(g["case_id"] == case_id) & (g["label"] == label)]
    p = np.zeros(len(BASINS), dtype=float)
    for _, r in sub.iterrows():
        b = int(r["basin_id"])
        if b in BASINS:
            p[b] = float(r["p"])
    return p


def pivot_label(g: pd.DataFrame, label: str) -> np.ndarray:
    sub = g[g["label"] == label]
    p = np.zeros(len(BASINS), dtype=float)
    for _, r in sub.iterrows():
        b = int(r["basin_id"])
        if b in BASINS:
            p[b] = float(r["p"])
    return p


# =========================
# Plotting: sector-filled pie (fixed 7 sectors)
# =========================
def basin_colors() -> List:
    # Use matplotlib default cycle but take first 7 distinct colors.
    # This keeps basin color consistent across WT/G12C/G12D.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle) >= 7:
        return cycle[:7]
    # fallback
    return ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def sector_fill_wedges(ax, probs: np.ndarray, title: str) -> None:
    """
    Draw a donut split into 7 equal sectors.
    In each sector, fill area fraction = probs[basin] with basin color.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs / (probs.sum() + 1e-12)

    cols = basin_colors()

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    # sector angles
    n = len(BASINS)
    sector = 360.0 / n

    # base ring sectors in light gray
    for i, b in enumerate(BASINS):
        theta1 = i * sector + GAP_DEG / 2.0
        theta2 = (i + 1) * sector - GAP_DEG / 2.0

        # background (full ring)
        bg = Wedge(
            center=(0, 0),
            r=R_OUT,
            theta1=theta1,
            theta2=theta2,
            width=(R_OUT - R_IN),
            facecolor=(0.92, 0.92, 0.92),
            edgecolor="white",
            linewidth=1.0,
        )
        ax.add_patch(bg)

        # filled fraction by area
        p = float(probs[b])
        p = max(0.0, min(1.0, p))

        # choose r_fill so that area fraction in the annular sector equals p
        # area ∝ (r^2 - R_IN^2), so solve:
        # (r_fill^2 - R_IN^2) / (R_OUT^2 - R_IN^2) = p
        r_fill = np.sqrt(R_IN**2 + p * (R_OUT**2 - R_IN**2))

        fg = Wedge(
            center=(0, 0),
            r=r_fill,
            theta1=theta1,
            theta2=theta2,
            width=(r_fill - R_IN),
            facecolor=cols[i],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.95,
        )
        ax.add_patch(fg)

        # label around the ring
        ang = np.deg2rad((theta1 + theta2) / 2.0)
        r_txt = 1.08
        ax.text(r_txt * np.cos(ang), r_txt * np.sin(ang), f"B{b}", ha="center", va="center", fontsize=10)

    # center text
    ax.text(0, 0, "basin\n0–6", ha="center", va="center", fontsize=10, alpha=0.9)


def save_fig(fig: plt.Figure, out_png: Path, out_pdf: Path) -> None:
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_case_triptych_sectorpies(
    g_case_label: pd.DataFrame,
    case_id: str,
    out_dir: Path
) -> None:
    labels = ["WT", "G12C", "G12D"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, lab in zip(axes, labels):
        p = pivot_case_label(g_case_label, case_id=case_id, label=lab)
        title = f"{case_id} | {lab}"
        sector_fill_wedges(ax, p, title=title)

    fig.suptitle(f"{case_id} basin contribution (fixed 7-sector, area-filled)", y=1.02, fontsize=13)
    fig.tight_layout()

    out_png = out_dir / f"case_{case_id}_sectorpie_triptych.png"
    out_pdf = out_dir / f"case_{case_id}_sectorpie_triptych.pdf"
    save_fig(fig, out_png, out_pdf)


# =========================
# Plotting: overall occupancy bars
# =========================
def plot_overall_basin_occupancy_groupedbar(g_label: pd.DataFrame, out_dir: Path) -> None:
    labs = ["WT", "G12C", "G12D"]
    x = np.arange(len(BASINS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    for i, lab in enumerate(labs):
        p = pivot_label(g_label, lab)
        ax.bar(x + (i - 1) * width, p, width=width, label=lab, alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}" for b in BASINS])
    ax.set_ylabel("Probability mass")
    ax.set_title("Overall basin occupancy (from points_enriched)")
    ax.legend(frameon=False)
    ax.axhline(0.0, linewidth=1.0)
    fig.tight_layout()

    save_fig(
        fig,
        out_dir / "overall_basin_occupancy_groupedbar.png",
        out_dir / "overall_basin_occupancy_groupedbar.pdf",
    )


# =========================
# Plotting: waterfall (delta + translucent cumulative envelope)
# =========================
def plot_waterfall_delta(delta: np.ndarray, title: str, out_base: Path) -> None:
    delta = np.asarray(delta, dtype=float).reshape(-1)
    if len(delta) != len(BASINS):
        raise ValueError("delta must have length 7 (basin 0..6).")

    x = np.arange(len(BASINS))
    cum = np.cumsum(delta)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(x, delta, width=0.72, alpha=0.95)
    ax.axhline(0.0, linewidth=1.0)

    ax.plot(x, cum, linewidth=2.0, alpha=0.85)
    ax.fill_between(x, 0.0, cum, alpha=0.18)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}" for b in BASINS])
    ax.set_ylabel("Δ probability mass")
    ax.set_title(title)

    ax.text(
        0.99, 0.95,
        f"sum Δ = {cum[-1]:.4f}",
        transform=ax.transAxes,
        ha="right", va="top"
    )

    fig.tight_layout()
    save_fig(fig, out_base.with_suffix(".png"), out_base.with_suffix(".pdf"))


def plot_waterfalls_from_label(g_label: pd.DataFrame, out_dir: Path) -> None:
    p_wt = pivot_label(g_label, "WT")
    for mut in ["G12C", "G12D"]:
        p_m = pivot_label(g_label, mut)
        delta = p_m - p_wt
        plot_waterfall_delta(
            delta=delta,
            title=f"Basin shift waterfall: {mut} − WT",
            out_base=out_dir / f"waterfall_{mut}_minus_WT",
        )


# =========================
# Optional plots: representatives RMSD
# =========================
def plot_rep_backbone_rmsd_by_label(rep: pd.DataFrame, out_dir: Path) -> None:
    label_col = pick_first_existing(rep.columns.tolist(), ["label", "condition", "mut", "genotype", "state"])
    rmsd_col = pick_first_existing(rep.columns.tolist(), ["backbone_rmsd", "backbone_rmsd_mean", "ca_rmsd", "rmsd"])
    if label_col is None or rmsd_col is None:
        return

    df = rep[[label_col, rmsd_col]].copy()
    df.rename(columns={label_col: "label", rmsd_col: "rmsd"}, inplace=True)
    df["label"] = df["label"].map(normalize_label)
    df["rmsd"] = coerce_numeric(df["rmsd"], default=np.nan)
    df = df.dropna(subset=["rmsd"])

    groups = []
    labs = []
    for lab in ["WT", "G12C", "G12D"]:
        v = df[df["label"] == lab]["rmsd"].to_numpy(dtype=float)
        if len(v) > 0:
            groups.append(v)
            labs.append(lab)

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.boxplot(groups, labels=labs, showfliers=False)
    ax.set_ylabel("Backbone RMSD")
    ax.set_title("Representative backbone RMSD by label")
    fig.tight_layout()

    save_fig(fig, out_dir / "rep_backbone_rmsd_by_label_box.png", out_dir / "rep_backbone_rmsd_by_label_box.pdf")


# =========================
# Optional plots: basin energy by label (if present)
# =========================
def plot_basin_energy_Etotal(basin_master: pd.DataFrame, out_dir: Path) -> None:
    # Expect columns like E_total_WT, E_total_G12C, E_total_G12D and basin_id
    basin_col = pick_first_existing(basin_master.columns.tolist(), ["basin_id", "basin"])
    if basin_col is None:
        return

    cols = {}
    for lab in ["WT", "G12C", "G12D"]:
        c = pick_first_existing(basin_master.columns.tolist(), [f"E_total_{lab}", f"e_total_{lab}", f"E_total_{lab.lower()}"])
        if c is not None:
            cols[lab] = c

    if len(cols) < 2:
        return

    df = basin_master[[basin_col] + list(cols.values())].copy()
    df.rename(columns={basin_col: "basin_id"}, inplace=True)
    df["basin_id"] = coerce_numeric(df["basin_id"], default=np.nan)
    df = df.dropna(subset=["basin_id"])
    df["basin_id"] = df["basin_id"].astype(int)
    df = df[df["basin_id"].isin(BASINS)].sort_values("basin_id")

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for i, lab in enumerate(["WT", "G12C", "G12D"]):
        if lab not in cols:
            continue
        v = coerce_numeric(df[cols[lab]], default=np.nan).to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, v, width=width, label=lab, alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels([f"B{b}" for b in df["basin_id"].tolist()])
    ax.set_ylabel("E_total")
    ax.set_title("Basin energy (E_total) by label")
    ax.legend(frameon=False)
    fig.tight_layout()

    save_fig(fig, out_dir / "basin_energy_E_total_by_label.png", out_dir / "basin_energy_E_total_by_label.pdf")


# =========================
# Optional: copy existing density plots into figs
# =========================
def maybe_copy_existing_density(project_root: Path, figs_dir: Path) -> None:
    if not COPY_EXISTING_DENSITY:
        return

    src = project_root / "KRAS_sampling_results" / "plots_A"
    if not src.exists():
        return

    dst = figs_dir / SUB_DENSITY_ARCHIVE
    ensure_dir(dst)

    for name in [
        "density_WT.png", "density_G12C.png", "density_G12D.png",
        "diff_G12C_minus_WT.png", "diff_G12D_minus_WT.png",
        "scatter_all.png",
        "density_WT.pdf", "density_G12C.pdf", "density_G12D.pdf",
        "diff_G12C_minus_WT.pdf", "diff_G12D_minus_WT.pdf",
        "scatter_all.pdf",
    ]:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


# =========================
# Main
# =========================
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    kras_dir = find_kras_analysis_dir(script_dir)
    project_root = kras_dir.parent

    merged_dir = kras_dir / "data_summary" / "merged"
    figs_dir = kras_dir / "figs"
    ensure_dir(figs_dir)

    # Output subfolders
    out_pies = figs_dir / SUB_PIES
    out_summary = figs_dir / SUB_SUMMARY
    out_wf = figs_dir / SUB_WATERFALL
    out_qc = figs_dir / SUB_QC
    for d in [out_pies, out_summary, out_wf, out_qc]:
        ensure_dir(d)

    # Required
    points_csv = merged_dir / "points_enriched.csv"
    if not points_csv.exists():
        raise FileNotFoundError(f"Missing required file: {points_csv}")

    points = load_points(points_csv)

    # Tidy distributions
    g_case_label = dist_case_label_basin(points)
    g_label = dist_label_basin(points)

    # 1) Per-case triptych sector-filled pies (WT/G12C/G12D)
    case_ids = sorted(points["case_id"].astype(str).unique().tolist())
    for cid in case_ids:
        plot_case_triptych_sectorpies(g_case_label, case_id=cid, out_dir=out_pies)

    # 2) Overall basin occupancy summary (grouped bar)
    plot_overall_basin_occupancy_groupedbar(g_label, out_dir=out_summary)

    # 3) Waterfalls (mut - WT)
    plot_waterfalls_from_label(g_label, out_dir=out_wf)

    # Optional inputs
    rep_csv = merged_dir / "representatives_enriched.csv"
    basin_master_csv = merged_dir / "basin_master.csv"
    rep = load_optional_csv(rep_csv)
    basin_master = load_optional_csv(basin_master_csv)

    # 4) Optional: representative RMSD QC
    if rep is not None:
        plot_rep_backbone_rmsd_by_label(rep, out_dir=out_qc)

    # 5) Optional: basin energy (E_total) if present
    if basin_master is not None:
        plot_basin_energy_Etotal(basin_master, out_dir=out_qc)

    # 6) Optional: archive existing density plots
    maybe_copy_existing_density(project_root, figs_dir)

    print(f"[OK] Done. All figures saved under: {figs_dir}")


if __name__ == "__main__":
    main()
