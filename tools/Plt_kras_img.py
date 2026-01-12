#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_kras_analysis_dir(start: Path) -> Path:
    """
    Search upward from 'start' for a folder named 'KRAS_analysis'.
    """
    p = start.resolve()
    for _ in range(15):
        cand = p / "KRAS_analysis"
        if cand.exists() and cand.is_dir():
            return cand
        p = p.parent
    raise FileNotFoundError(f"Cannot find KRAS_analysis by searching upward from: {start}")


def pick_first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def normalize_label_value(x: str) -> str:
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


def jsd_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(a * (np.log(a) - np.log(b)))

    jsd_nats = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(jsd_nats / math.log(2.0))


def tv_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    return float(0.5 * np.sum(np.abs(p - q)))


def save_fig(fig: plt.Figure, out_png: Path, out_pdf: Optional[Path] = None, dpi: int = 600) -> None:
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Density maps from points_enriched.csv (chunked)
# -----------------------------

def build_density_histograms(
    points_csv: Path,
    out_dir: Path,
    bins: int = 320,
    chunk_size: int = 250_000,
    eps: float = 1e-12,
) -> Dict[str, object]:
    df0 = pd.read_csv(points_csv, nrows=2000)
    cols = list(df0.columns)

    label_col = pick_first_existing(cols, ["label", "condition", "mut", "genotype", "state"])
    if label_col is None:
        raise RuntimeError(f"Cannot find label column in {points_csv.name}")

    z1_col = pick_first_existing(cols, ["z1", "tsne1", "embed_x", "x", "dim1"])
    z2_col = pick_first_existing(cols, ["z2", "tsne2", "embed_y", "y", "dim2"])
    if z1_col is None or z2_col is None:
        raise RuntimeError(f"Cannot find embedding columns in {points_csv.name}")

    w_col = pick_first_existing(cols, ["weight", "p_mass", "prob", "mass"])
    if w_col is None:
        w_col = "__unit_weight__"

    usecols = [label_col, z1_col, z2_col] + ([] if w_col == "__unit_weight__" else [w_col])

    # Pass 1: extent + label set
    z1_min, z1_max = np.inf, -np.inf
    z2_min, z2_max = np.inf, -np.inf
    labels_set = set()
    total_w = {}

    for chunk in pd.read_csv(points_csv, usecols=usecols, chunksize=chunk_size):
        chunk[label_col] = chunk[label_col].map(normalize_label_value)
        labs = chunk[label_col].dropna().unique().tolist()
        labels_set.update(labs)

        z1 = chunk[z1_col].to_numpy(dtype=np.float64, copy=False)
        z2 = chunk[z2_col].to_numpy(dtype=np.float64, copy=False)

        z1_min = min(z1_min, float(np.nanmin(z1)))
        z1_max = max(z1_max, float(np.nanmax(z1)))
        z2_min = min(z2_min, float(np.nanmin(z2)))
        z2_max = max(z2_max, float(np.nanmax(z2)))

        if w_col == "__unit_weight__":
            w = np.ones(len(chunk), dtype=np.float64)
        else:
            w = pd.to_numeric(chunk[w_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)

        for lab, idx in chunk.groupby(label_col).indices.items():
            total_w[lab] = total_w.get(lab, 0.0) + float(w[idx].sum())

    labels = sorted(list(labels_set))
    if not labels:
        raise RuntimeError("No labels found in points file.")

    pad1 = 0.02 * (z1_max - z1_min + eps)
    pad2 = 0.02 * (z2_max - z2_min + eps)
    z1_min, z1_max = z1_min - pad1, z1_max + pad1
    z2_min, z2_max = z2_min - pad2, z2_max + pad2

    xedges = np.linspace(z1_min, z1_max, bins + 1)
    yedges = np.linspace(z2_min, z2_max, bins + 1)

    hists = {lab: np.zeros((bins, bins), dtype=np.float64) for lab in labels}

    # Pass 2: hist accumulation
    for chunk in pd.read_csv(points_csv, usecols=usecols, chunksize=chunk_size):
        chunk[label_col] = chunk[label_col].map(normalize_label_value)

        z1 = chunk[z1_col].to_numpy(dtype=np.float64, copy=False)
        z2 = chunk[z2_col].to_numpy(dtype=np.float64, copy=False)

        if w_col == "__unit_weight__":
            w = np.ones(len(chunk), dtype=np.float64)
        else:
            w = pd.to_numeric(chunk[w_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)

        lab_arr = chunk[label_col].to_numpy()
        for lab in labels:
            mask = (lab_arr == lab)
            if not np.any(mask):
                continue
            H, _, _ = np.histogram2d(
                z1[mask], z2[mask],
                bins=[xedges, yedges],
                weights=w[mask],
            )
            hists[lab] += H

    probs = {}
    for lab in labels:
        s = hists[lab].sum()
        probs[lab] = (hists[lab] / s) if s > 0 else hists[lab].copy()

    metrics = {}
    if "WT" in probs:
        for mut in ["G12C", "G12D"]:
            if mut in probs:
                metrics[f"JSD_{mut}_vs_WT_bits"] = jsd_bits(probs["WT"].ravel(), probs[mut].ravel())
                metrics[f"TV_{mut}_vs_WT"] = tv_distance(probs["WT"].ravel(), probs[mut].ravel())

    ensure_dir(out_dir)
    extent = [z1_min, z1_max, z2_min, z2_max]
    (out_dir / "density_metrics.json").write_text(json.dumps({
        "points_csv": str(points_csv),
        "bins": bins,
        "labels": labels,
        "metrics": metrics,
        "total_weight_by_label": total_w,
        "z1_col": z1_col,
        "z2_col": z2_col,
        "label_col": label_col,
        "weight_col": (None if w_col == "__unit_weight__" else w_col),
        "extent": extent,
    }, indent=2), encoding="utf-8")

    all_vals = np.concatenate([probs[lab].ravel() for lab in labels])
    vmax = float(np.max(all_vals))
    vmin = float(np.min(all_vals[all_vals > 0])) if np.any(all_vals > 0) else eps
    vmin = max(vmin, eps)

    for lab in labels:
        title = f"KRAS structure-space sampling density ({lab})"
        if lab in ["G12C", "G12D"] and "WT" in probs:
            k = f"JSD_{lab}_vs_WT_bits"
            if k in metrics:
                title += f" | JSD vs WT = {metrics[k]:.3f} bits"

        fig, ax = plt.subplots(figsize=(8.2, 6.8))
        im = ax.imshow(
            probs[lab].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0)),
        )
        ax.set_xlabel("Structure axis 1 (embedding)")
        ax.set_ylabel("Structure axis 2 (embedding)")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Probability mass (log scale)")

        save_fig(fig, out_dir / f"density_{lab}.png", out_pdf=out_dir / f"density_{lab}.pdf", dpi=600)

    if "WT" in probs:
        for mut in ["G12C", "G12D"]:
            if mut not in probs:
                continue
            log_ratio = np.log10((probs[mut] + eps) / (probs["WT"] + eps))
            lim = float(np.quantile(np.abs(log_ratio.ravel()), 0.995))
            lim = max(lim, 1e-3)

            fig, ax = plt.subplots(figsize=(8.2, 6.8))
            im = ax.imshow(
                log_ratio.T,
                origin="lower",
                extent=extent,
                aspect="auto",
                vmin=-lim,
                vmax=lim,
            )
            ax.set_xlabel("Structure axis 1 (embedding)")
            ax.set_ylabel("Structure axis 2 (embedding)")
            title = f"KRAS density log-ratio: {mut} / WT"
            if f"JSD_{mut}_vs_WT_bits" in metrics:
                title += f" | JSD = {metrics[f'JSD_{mut}_vs_WT_bits']:.3f} bits"
            ax.set_title(title)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("log10((mut + eps)/(WT + eps))")

            save_fig(fig, out_dir / f"density_logratio_{mut}_vs_WT.png",
                     out_pdf=out_dir / f"density_logratio_{mut}_vs_WT.pdf", dpi=600)

    return {"labels": labels, "metrics": metrics}


# -----------------------------
# Basin plots
# -----------------------------

def plot_basin_overview(basin_csv: Path, out_dir: Path, top_n: int = 20) -> None:
    df = pd.read_csv(basin_csv)

    basin_id_col = pick_first_existing(df.columns.tolist(), ["basin_id", "basin", "cluster_id"])
    if basin_id_col is None:
        raise RuntimeError("Cannot find basin id column in basin_master.csv")

    wt_col = pick_first_existing(df.columns.tolist(), ["mass_WT", "WT", "wt", "p_WT"])
    g12c_col = pick_first_existing(df.columns.tolist(), ["mass_G12C", "G12C", "g12c", "p_G12C"])
    g12d_col = pick_first_existing(df.columns.tolist(), ["mass_G12D", "G12D", "g12d", "p_G12D"])

    if wt_col is None or (g12c_col is None and g12d_col is None):
        raise RuntimeError("Cannot find WT/G12C/G12D occupancy columns in basin_master.csv")

    for c in [wt_col, g12c_col, g12d_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    union = df[wt_col].copy()
    if g12c_col is not None:
        union += df[g12c_col]
    if g12d_col is not None:
        union += df[g12d_col]

    df_ranked = df.assign(_union=union).sort_values("_union", ascending=False).head(top_n)
    x = np.arange(len(df_ranked))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width, df_ranked[wt_col].to_numpy(), width, label="WT")
    if g12c_col is not None:
        ax.bar(x, df_ranked[g12c_col].to_numpy(), width, label="G12C")
    if g12d_col is not None:
        ax.bar(x + width, df_ranked[g12d_col].to_numpy(), width, label="G12D")
    ax.set_xticks(x)
    ax.set_xticklabels(df_ranked[basin_id_col].astype(str).tolist(), rotation=45, ha="right")
    ax.set_ylabel("Basin probability mass")
    ax.set_title(f"Top-{top_n} basins by union mass")
    ax.legend()
    save_fig(fig, out_dir / f"basin_top{top_n}_occupancy.png", out_pdf=out_dir / f"basin_top{top_n}_occupancy.pdf", dpi=600)

    if g12c_col is not None:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        delta = (df_ranked[g12c_col] - df_ranked[wt_col]).to_numpy()
        ax.bar(x, delta)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df_ranked[basin_id_col].astype(str).tolist(), rotation=45, ha="right")
        ax.set_ylabel("Delta mass (G12C - WT)")
        ax.set_title(f"Top-{top_n} basins: delta occupancy (G12C - WT)")
        save_fig(fig, out_dir / f"basin_top{top_n}_delta_G12C_minus_WT.png",
                 out_pdf=out_dir / f"basin_top{top_n}_delta_G12C_minus_WT.pdf", dpi=600)

    if g12d_col is not None:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        delta = (df_ranked[g12d_col] - df_ranked[wt_col]).to_numpy()
        ax.bar(x, delta)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df_ranked[basin_id_col].astype(str).tolist(), rotation=45, ha="right")
        ax.set_ylabel("Delta mass (G12D - WT)")
        ax.set_title(f"Top-{top_n} basins: delta occupancy (G12D - WT)")
        save_fig(fig, out_dir / f"basin_top{top_n}_delta_G12D_minus_WT.png",
                 out_pdf=out_dir / f"basin_top{top_n}_delta_G12D_minus_WT.pdf", dpi=600)


# -----------------------------
# Representatives plots
# -----------------------------

def plot_representatives(reps_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(reps_csv)

    label_col = pick_first_existing(df.columns.tolist(), ["label", "condition", "mut", "genotype", "state"])
    if label_col is None:
        raise RuntimeError("Cannot find label column in representatives_enriched.csv")
    df[label_col] = df[label_col].map(normalize_label_value)

    rmsd_col = pick_first_existing(df.columns.tolist(), ["backbone_rmsd", "ca_rmsd", "rmsd"])
    if rmsd_col is None:
        raise RuntimeError("Cannot find rmsd column in representatives_enriched.csv")
    df[rmsd_col] = pd.to_numeric(df[rmsd_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for lab in sorted(df[label_col].dropna().unique().tolist()):
        x = df.loc[df[label_col] == lab, rmsd_col].dropna().to_numpy()
        if len(x) == 0:
            continue
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=f"{lab} (n={len(x)})")
    ax.set_xlabel(rmsd_col)
    ax.set_ylabel("ECDF")
    ax.set_title("Representative RMSD ECDF")
    ax.legend()
    save_fig(fig, out_dir / "representatives_rmsd_ecdf.png", out_pdf=out_dir / "representatives_rmsd_ecdf.pdf", dpi=600)

    labs = sorted(df[label_col].dropna().unique().tolist())
    data = [df.loc[df[label_col] == lab, rmsd_col].dropna().to_numpy() for lab in labs]

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.boxplot(data, labels=labs, showfliers=False)
    ax.set_xlabel("Condition")
    ax.set_ylabel(rmsd_col)
    ax.set_title("Representative RMSD (boxplot)")
    save_fig(fig, out_dir / "representatives_rmsd_boxplot.png", out_pdf=out_dir / "representatives_rmsd_boxplot.pdf", dpi=600)


# -----------------------------
# Main
# -----------------------------

def main():
    script_dir = Path(__file__).resolve().parent
    kras_dir = find_kras_analysis_dir(script_dir)

    # Correct path:
    merged_dir = kras_dir / "data_summary" / "merged"
    figs_dir = kras_dir / "figs"
    ensure_dir(figs_dir)

    points_csv = merged_dir / "points_enriched.csv"
    basin_csv = merged_dir / "basin_master.csv"
    reps_csv = merged_dir / "representatives_enriched.csv"

    missing = [p for p in [points_csv, basin_csv, reps_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(str(p) for p in missing) +
            f"\n\n[Hint] Using merged_dir = {merged_dir}"
        )

    # 1) Density maps (+ log-ratio)
    build_density_histograms(points_csv=points_csv, out_dir=figs_dir, bins=320, chunk_size=250_000)

    # 2) Basin overview
    plot_basin_overview(basin_csv=basin_csv, out_dir=figs_dir, top_n=20)

    # 3) Representatives summary
    plot_representatives(reps_csv=reps_csv, out_dir=figs_dir)

    (figs_dir / "_plot_run_summary.json").write_text(json.dumps({
        "KRAS_analysis": str(kras_dir),
        "merged_dir": str(merged_dir),
        "figs_dir": str(figs_dir),
        "inputs": {
            "points_enriched": str(points_csv),
            "basin_master": str(basin_csv),
            "representatives_enriched": str(reps_csv),
        },
        "outputs_note": "All PNG saved at dpi=600; PDFs are vector.",
    }, indent=2), encoding="utf-8")

    print(f"[OK] All figures saved to: {figs_dir}")


if __name__ == "__main__":
    main()
