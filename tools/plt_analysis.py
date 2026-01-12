# --*-- conding:utf-8 --*--
# @time:1/11/26 21:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plt_analysis.py

# --*-- coding:utf-8 --*--
# @time: 2026-01-12
# @Author : Yuqi Zhang
# @File: plot_kras_final_figs_pub.py

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# -----------------------------
# Global config
# -----------------------------
DPI = 600
LABELS = ["WT", "G12C", "G12D"]
N_BASINS = 7
BASINS = list(range(N_BASINS))

MAX_SCATTER_POINTS = 60000

ENERGY_TERM_KEYS = [
    "E_steric", "E_geom", "E_bond", "E_mj", "E_dihedral", "E_hb",
    "E_hydroph", "E_cbeta", "E_rama", "E_total"
]

# Colormaps (publication-friendly defaults)
CMAP_OCC = mpl.cm.viridis      # sequential
CMAP_DELTA = mpl.cm.coolwarm   # diverging


# -----------------------------
# Style (publication)
# -----------------------------
def apply_pub_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "font.family": "DejaVu Sans",
        "font.size": 11,

        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        "legend.fontsize": 10,
        "legend.frameon": False,

        "lines.linewidth": 2.2,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.25,
    })


def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Robust column picking
# -----------------------------
def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(f"Cannot find any of columns: {candidates}\nAvailable columns: {list(df.columns)}")
    return None


def normalize_label_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(" ", "", regex=False)
    x_upper = x.str.upper()

    out = x_upper.copy()
    out = out.replace({"WILD": "WT", "WILDTYPE": "WT"})
    out = out.str.replace("KRAS_", "", regex=False)

    out = out.where(~out.str.contains("G12C"), "G12C")
    out = out.where(~out.str.contains("G12D"), "G12D")
    out = out.where(~out.str.contains("WT"), "WT")
    return out


def pick_weight_col(df: pd.DataFrame) -> str:
    return pick_col(df, ["weight", "p_mass", "mass", "prob", "w"], required=True)  # type: ignore


def pick_rmsd_col(df: pd.DataFrame) -> str:
    return pick_col(df, ["backbone_rmsd", "backbone_rmsd_mean", "ca_rmsd", "ca_rmsd_A"], required=True)  # type: ignore


def read_csv_maybe(p: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(p) if p.exists() else None


# -----------------------------
# IO loaders
# -----------------------------
def load_points(merged_dir: Path, analysis_dir: Path) -> pd.DataFrame:
    df = read_csv_maybe(merged_dir / "points_enriched.csv")
    if df is None:
        df = read_csv_maybe(merged_dir / "merged_points_with_basin.csv")
    if df is None:
        df = read_csv_maybe(analysis_dir / "embedding_points.csv")
    if df is None:
        raise FileNotFoundError(
            "Cannot find points table. Expected one of:\n"
            f"  {merged_dir/'points_enriched.csv'}\n"
            f"  {merged_dir/'merged_points_with_basin.csv'}\n"
            f"  {analysis_dir/'embedding_points.csv'}"
        )

    z1 = pick_col(df, ["z1", "x", "tsne1"], required=True)
    z2 = pick_col(df, ["z2", "y", "tsne2"], required=True)
    lab = pick_col(df, ["label", "system", "case", "case_id", "pdb_id", "example_id"], required=True)
    wcol = pick_weight_col(df)
    rcol = pick_rmsd_col(df)

    out = df[[z1, z2, lab, wcol, rcol]].copy()
    out.columns = ["z1", "z2", "label", "weight", "rmsd"]
    out["label"] = normalize_label_series(out["label"])
    out = out[out["label"].isin(LABELS)].copy()

    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["rmsd"] = pd.to_numeric(out["rmsd"], errors="coerce")
    out = out[(out["weight"] > 0) & np.isfinite(out["rmsd"])].copy()

    if out.empty:
        raise ValueError("Points table became empty after filtering for label/weight/rmsd validity.")
    return out


def load_representatives(merged_dir: Path, analysis_dir: Path) -> Optional[pd.DataFrame]:
    df = read_csv_maybe(merged_dir / "representatives_enriched.csv")
    if df is None:
        df = read_csv_maybe(analysis_dir / "representatives.csv")
    if df is None:
        return None

    lab = pick_col(df, ["label", "scope", "system"], required=True)
    bid = pick_col(df, ["basin_id", "basin"], required=False)
    z1 = pick_col(df, ["z1"], required=False)
    z2 = pick_col(df, ["z2"], required=False)
    rmsd = pick_col(df, ["backbone_rmsd", "ca_rmsd"], required=False)
    wcol = pick_col(df, ["weight", "p_mass", "mass", "prob", "w"], required=False)

    keep = [c for c in [lab, bid, z1, z2, rmsd, wcol] if c is not None]
    out = df[keep].copy()
    out.rename(columns={lab: "label"}, inplace=True)
    out["label"] = normalize_label_series(out["label"])
    out = out[out["label"].isin(LABELS)].copy()
    return out


def load_basin_master(merged_dir: Path, analysis_dir: Path) -> pd.DataFrame:
    df = read_csv_maybe(merged_dir / "basin_master.csv")
    if df is None:
        df = read_csv_maybe(analysis_dir / "basin_occupancy.csv")
    if df is None:
        raise FileNotFoundError(
            "Cannot find basin master/occupancy table. Expected one of:\n"
            f"  {merged_dir/'basin_master.csv'}\n"
            f"  {analysis_dir/'basin_occupancy.csv'}"
        )

    bid = pick_col(df, ["basin_id", "basin"], required=True)
    df = df.copy()
    df.rename(columns={bid: "basin_id"}, inplace=True)

    def get_mass_col(label: str) -> Optional[str]:
        return pick_col(df, [label, f"mass_{label}", f"{label}_mass", f"mass_{label.lower()}"], required=False)

    cols = {lab: get_mass_col(lab) for lab in LABELS}
    if any(v is None for v in cols.values()):
        raise ValueError(
            "Basin table does not contain per-label masses. Need columns like WT/G12C/G12D or mass_WT/mass_G12C/mass_G12D.\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[["basin_id", cols["WT"], cols["G12C"], cols["G12D"]]].copy()
    out.columns = ["basin_id", "WT", "G12C", "G12D"]
    for c in ["WT", "G12C", "G12D"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def load_key_basin_story(analysis_dir: Path) -> Optional[pd.DataFrame]:
    df = read_csv_maybe(analysis_dir / "addons" / "key_basin_story.csv")
    if df is None:
        df = read_csv_maybe(analysis_dir / "addons" / "key_basin_story_ranked.csv")
    return df


def load_per_residue_disp(analysis_dir: Path) -> Optional[pd.DataFrame]:
    return read_csv_maybe(analysis_dir / "addons" / "displacement" / "per_residue_displacement.csv")


# -----------------------------
# Save utils
# -----------------------------
def save_fig(fig: plt.Figure, out_png: Path, out_pdf: Path) -> None:
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)


def subsample_df(df: pd.DataFrame, max_n: int, seed: int = 0) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_n, replace=False)
    return df.iloc[idx].copy()


# -----------------------------
# A) Copy existing density/diff plots
# -----------------------------
def copy_existing_density_plots(project_root: Path, figs_dir: Path) -> None:
    candidates = [
        project_root / "KRAS_sampling_results" / "plots_A",
        project_root / "KRAS_sampling_results" / "analysis_closed_loop" / "plots_A",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        print("[WARN] Cannot find plots_A folder to copy density/diff plots.")
        return

    patterns = [
        "density_*.png", "density_*.pdf",
        "diff_*.png", "diff_*.pdf",
        "scatter_all.png", "scatter_all.pdf",
    ]
    copied = 0
    for pat in patterns:
        for f in src.glob(pat):
            shutil.copy2(f, figs_dir / f.name)
            copied += 1
    print(f"[OK] Copied {copied} existing density/diff/scatter plots from: {src}")


# -----------------------------
# B) RMSD overlay on embedding
# -----------------------------
def plot_rmsd_overlay(points: pd.DataFrame, figs_dir: Path) -> None:
    for lab in LABELS:
        df = points[points["label"] == lab].copy()
        df = subsample_df(df, MAX_SCATTER_POINTS, seed=0)

        w = df["weight"].to_numpy(dtype=float)
        w = w / (w.sum() + 1e-12)
        size = 6.0 + 240.0 * np.sqrt(w)  # stable perceptual scaling

        fig = plt.figure(figsize=(7.2, 6.0))
        ax = fig.add_subplot(111)

        sc = ax.scatter(
            df["z1"], df["z2"],
            c=df["rmsd"],
            s=size,
            alpha=0.78,
            linewidths=0,
            rasterized=True,  # keeps PDF size sane
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("Backbone RMSD")

        ax.set_title(f"Embedding RMSD overlay ({lab})")
        ax.set_xlabel("Structure axis 1 (embedding)")
        ax.set_ylabel("Structure axis 2 (embedding)")
        ax.grid(True)

        save_fig(fig, figs_dir / f"fig_B_rmsd_overlay_{lab}.png", figs_dir / f"fig_B_rmsd_overlay_{lab}.pdf")

    print("[OK] B: RMSD overlay plots saved.")


# -----------------------------
# C) Probability-mass CDF vs RMSD
# -----------------------------
def plot_mass_cdf(points: pd.DataFrame, figs_dir: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(111)

    for lab in LABELS:
        df = points[points["label"] == lab].copy().sort_values("rmsd")
        w = df["weight"].to_numpy(dtype=float)
        w = w / (w.sum() + 1e-12)
        cdf = np.cumsum(w)
        ax.plot(df["rmsd"].to_numpy(), cdf, label=lab)

    ax.set_title("Cumulative probability mass vs RMSD")
    ax.set_xlabel("Backbone RMSD threshold")
    ax.set_ylabel("Cumulative probability mass")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="lower right")

    save_fig(fig, figs_dir / "fig_C_mass_cdf_rmsd.png", figs_dir / "fig_C_mass_cdf_rmsd.pdf")
    print("[OK] C: Mass CDF plot saved.")


# -----------------------------
# D/E) 7-sector equal-angle rings (publication)
# -----------------------------
def _ring_base(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")


def draw_occupancy_ring(ax, values: List[float], title: str, cmap=CMAP_OCC) -> None:
    """
    Equal-angle 7 sectors, color encodes occupancy intensity.
    Each sector has a light-gray frame; fill is color-mapped.
    """
    inner_r = 0.58
    outer_r = 1.00

    vals = np.array(values, dtype=float)
    vals = np.clip(vals, 0.0, None)
    vmax = float(vals.max()) if float(vals.max()) > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    # background ring outline
    ax.add_patch(Circle((0, 0), outer_r, fill=False, linewidth=1.2, alpha=0.9))
    ax.add_patch(Circle((0, 0), inner_r, fill=False, linewidth=1.2, alpha=0.9))

    for i, val in enumerate(vals.tolist()):
        theta1 = 360.0 * (i / N_BASINS)
        theta2 = 360.0 * ((i + 1) / N_BASINS)

        # sector frame
        frame = Wedge(
            (0, 0), outer_r, theta1, theta2,
            width=outer_r - inner_r,
            facecolor=(0, 0, 0, 0.06),
            edgecolor=(1, 1, 1, 0.95),
            linewidth=1.2,
        )
        ax.add_patch(frame)

        # occupancy fill: map value to color and to radial proportion
        frac = float(val / (vmax + 1e-12))
        fill_outer = inner_r + frac * (outer_r - inner_r)
        if fill_outer > inner_r + 1e-6:
            color = cmap(norm(val))
            fill = Wedge(
                (0, 0), fill_outer, theta1, theta2,
                width=fill_outer - inner_r,
                facecolor=color,
                edgecolor=(1, 1, 1, 0.95),
                linewidth=1.0,
                alpha=0.95,
            )
            ax.add_patch(fill)

        # basin id label
        ang = math.radians((theta1 + theta2) / 2.0)
        rtxt = outer_r + 0.10
        ax.text(rtxt * math.cos(ang), rtxt * math.sin(ang), f"{i}",
                ha="center", va="center", fontsize=11)

    ax.set_title(title)
    _ring_base(ax)


def draw_delta_ring(ax, deltas: List[float], title: str, vmax: float, cmap=CMAP_DELTA) -> None:
    """
    Equal-angle 7 sectors, color encodes delta (mut - WT), centered at 0.
    """
    inner_r = 0.58
    outer_r = 1.00

    d = np.array(deltas, dtype=float)
    vmax = float(max(vmax, 1e-9))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    ax.add_patch(Circle((0, 0), outer_r, fill=False, linewidth=1.2, alpha=0.9))
    ax.add_patch(Circle((0, 0), inner_r, fill=False, linewidth=1.2, alpha=0.9))

    for i, val in enumerate(d.tolist()):
        theta1 = 360.0 * (i / N_BASINS)
        theta2 = 360.0 * ((i + 1) / N_BASINS)

        # frame
        frame = Wedge(
            (0, 0), outer_r, theta1, theta2,
            width=outer_r - inner_r,
            facecolor=(0, 0, 0, 0.06),
            edgecolor=(1, 1, 1, 0.95),
            linewidth=1.2,
        )
        ax.add_patch(frame)

        color = cmap(norm(val))
        fill = Wedge(
            (0, 0), outer_r, theta1, theta2,
            width=outer_r - inner_r,
            facecolor=color,
            edgecolor=(1, 1, 1, 0.95),
            linewidth=1.0,
            alpha=0.92,
        )
        ax.add_patch(fill)

        ang = math.radians((theta1 + theta2) / 2.0)
        rtxt = outer_r + 0.10
        ax.text(rtxt * math.cos(ang), rtxt * math.sin(ang), f"{i}",
                ha="center", va="center", fontsize=11)

    ax.set_title(title)
    _ring_base(ax)


def plot_basin_occupancy_rings(basin_master: pd.DataFrame, figs_dir: Path) -> None:
    bm = basin_master.copy()
    bm = bm[bm["basin_id"].isin(BASINS)].sort_values("basin_id")

    occ: Dict[str, List[float]] = {}
    for lab in LABELS:
        v = bm[lab].to_numpy(dtype=float)
        v = np.clip(v, 0.0, None)
        v = v / (v.sum() + 1e-12)
        occ[lab] = v.tolist()

    fig = plt.figure(figsize=(12.2, 4.7))
    for i, lab in enumerate(LABELS):
        ax = fig.add_subplot(1, 3, i + 1)
        draw_occupancy_ring(ax, occ[lab], title=f"Occupancy by basin ({lab})")

    # add a compact colorbar for occupancy
    # use global vmax=1.0 because values are normalized probabilities
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_OCC)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Occupancy (probability mass)")

    save_fig(fig, figs_dir / "fig_D_basin_occupancy_rings.png", figs_dir / "fig_D_basin_occupancy_rings.pdf")
    print("[OK] D: Basin occupancy rings saved.")


def plot_delta_rings(basin_master: pd.DataFrame, figs_dir: Path) -> None:
    bm = basin_master.copy()
    bm = bm[bm["basin_id"].isin(BASINS)].sort_values("basin_id")

    def normprob(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, None)
        return x / (x.sum() + 1e-12)

    wt = normprob(bm["WT"].to_numpy(float))
    c = normprob(bm["G12C"].to_numpy(float))
    d = normprob(bm["G12D"].to_numpy(float))

    dc = c - wt
    dd = d - wt
    vmax = float(max(np.max(np.abs(dc)), np.max(np.abs(dd)), 1e-9))

    fig = plt.figure(figsize=(9.2, 4.7))
    ax1 = fig.add_subplot(1, 2, 1)
    draw_delta_ring(ax1, dc.tolist(), title="Distribution shift (G12C − WT)", vmax=vmax)
    ax2 = fig.add_subplot(1, 2, 2)
    draw_delta_ring(ax2, dd.tolist(), title="Distribution shift (G12D − WT)", vmax=vmax)

    # colorbar centered at 0
    sm = ScalarMappable(norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), cmap=CMAP_DELTA)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Δ occupancy (mut − WT)")

    save_fig(fig, figs_dir / "fig_E_delta_rings.png", figs_dir / "fig_E_delta_rings.pdf")
    print("[OK] E: Delta rings saved.")


# -----------------------------
# F) Waterfall energy signature (publication)
# -----------------------------
def pick_key_basins_from_basin_master(basin_master: pd.DataFrame) -> Tuple[int, int]:
    bm = basin_master.copy()
    bm = bm[bm["basin_id"].isin(BASINS)].copy()

    def norm_col(col: str) -> pd.Series:
        x = bm[col].astype(float).clip(lower=0.0)
        return x / (x.sum() + 1e-12)

    wt = norm_col("WT")
    d = norm_col("G12D")
    delta = (d - wt).abs()
    order = bm.loc[delta.sort_values(ascending=False).index, "basin_id"].tolist()
    key = int(order[0]) if order else 0
    contrast = int(order[1]) if len(order) > 1 else key
    return key, contrast


def plot_waterfall_for_basin(key_story: pd.DataFrame, basin_id: int, mutant: str, figs_dir: Path) -> None:
    df = key_story.copy()
    bid_col = pick_col(df, ["basin_id", "basin"], required=True)
    df = df[df[bid_col] == basin_id].copy()
    if df.empty:
        print(f"[WARN] key_basin_story has no basin_id={basin_id}. Skip waterfall.")
        return

    row = df.iloc[0].to_dict()

    deltas = []
    terms = []

    def find_key(prefix: str, label: str) -> Optional[str]:
        for k in [f"{prefix}_{label}", f"{prefix.lower()}_{label.lower()}", f"{prefix}_{label.lower()}"]:
            if k in row:
                return k
        return None

    for t in ENERGY_TERM_KEYS:
        k_wt = find_key(t, "WT")
        k_mu = find_key(t, mutant)
        if k_wt is None or k_mu is None:
            continue
        v_wt = float(row[k_wt])
        v_mu = float(row[k_mu])
        deltas.append(v_mu - v_wt)
        terms.append(t.replace("E_", "").replace("_", " "))

    if not deltas:
        print(f"[WARN] No usable energy terms for waterfall (basin={basin_id}, mutant={mutant}).")
        return

    x = np.arange(len(deltas))
    dlt = np.array(deltas, dtype=float)
    cum = np.cumsum(dlt)

    fig = plt.figure(figsize=(9.6, 4.9))
    ax = fig.add_subplot(111)

    ax.axhline(0.0, linewidth=1.2)
    ax.grid(True, axis="y")

    ax.bar(x, dlt, alpha=0.88, width=0.75)
    ax.plot(x, cum, linewidth=2.4, alpha=0.35)
    ax.fill_between(x, 0.0, cum, alpha=0.10)

    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=28, ha="right")
    ax.set_title(f"Energy signature (basin {basin_id}): {mutant} − WT")
    ax.set_ylabel("ΔE (mut − WT)")
    ax.set_xlabel("Energy terms")

    save_fig(
        fig,
        figs_dir / f"fig_F_waterfall_energy_signature_basin{basin_id}_{mutant}.png",
        figs_dir / f"fig_F_waterfall_energy_signature_basin{basin_id}_{mutant}.pdf",
    )


# -----------------------------
# G) Per-residue displacement ruler (publication)
# -----------------------------
def plot_per_residue_displacement(per_res_disp: pd.DataFrame, basin_id: int, figs_dir: Path) -> None:
    df = per_res_disp.copy()
    bid = pick_col(df, ["basin_id", "basin"], required=True)
    pair = pick_col(df, ["pair"], required=True)
    disp = pick_col(df, ["disp", "displacement_A", "disp_A"], required=True)
    resseq = pick_col(df, ["resseq", "i"], required=True)

    df.rename(columns={bid: "basin_id", pair: "pair", disp: "disp", resseq: "pos"}, inplace=True)
    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce")
    df = df[df["basin_id"] == basin_id].copy()
    if df.empty:
        print(f"[WARN] No per-residue displacement rows for basin_id={basin_id}. Skip.")
        return

    fig = plt.figure(figsize=(10.8, 3.9))
    ax = fig.add_subplot(111)

    for p in sorted(df["pair"].unique()):
        sub = df[df["pair"] == p].copy()
        sub["pos"] = pd.to_numeric(sub["pos"], errors="coerce")
        sub["disp"] = pd.to_numeric(sub["disp"], errors="coerce")
        sub = sub[np.isfinite(sub["pos"]) & np.isfinite(sub["disp"])].sort_values("pos")
        if sub.empty:
            continue
        ax.plot(sub["pos"].to_numpy(), sub["disp"].to_numpy(), alpha=0.85, label=str(p))

    ax.set_title(f"Per-residue displacement (basin {basin_id})")
    ax.set_xlabel("Residue index")
    ax.set_ylabel("Displacement magnitude")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    save_fig(
        fig,
        figs_dir / f"fig_G_per_residue_displacement_basin{basin_id}.png",
        figs_dir / f"fig_G_per_residue_displacement_basin{basin_id}.pdf",
    )


# -----------------------------
# H) Representatives overlay (publication)
# -----------------------------
def plot_representatives_overlay(points: pd.DataFrame, reps: Optional[pd.DataFrame], figs_dir: Path) -> None:
    base = subsample_df(points, MAX_SCATTER_POINTS, seed=1)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111)

    for lab in LABELS:
        sub = base[base["label"] == lab]
        if sub.empty:
            continue
        ax.scatter(sub["z1"], sub["z2"], s=6, alpha=0.12, linewidths=0, rasterized=True, label=lab)

    if reps is not None and ("z1" in reps.columns) and ("z2" in reps.columns):
        ax.scatter(reps["z1"], reps["z2"], s=85, alpha=0.95, marker="x")
        if "basin_id" in reps.columns:
            for _, r in reps.iterrows():
                try:
                    ax.text(float(r["z1"]), float(r["z2"]), str(int(r["basin_id"])), fontsize=9)
                except Exception:
                    pass

    ax.set_title("Representatives on embedding")
    ax.set_xlabel("Structure axis 1 (embedding)")
    ax.set_ylabel("Structure axis 2 (embedding)")
    ax.grid(True)
    ax.legend(loc="best")

    save_fig(fig, figs_dir / "fig_H_representatives_overlay.png", figs_dir / "fig_H_representatives_overlay.pdf")
    print("[OK] H: Representatives overlay saved.")


# -----------------------------
# Main
# -----------------------------
def main():
    apply_pub_style()

    proj = project_root_from_tools_dir()
    kras_analysis_dir = proj / "KRAS_analysis"
    merged_dir = kras_analysis_dir / "data_summary" / "merged"
    figs_dir = kras_analysis_dir / "figs"
    ensure_dir(figs_dir)

    analysis_dir = proj / "KRAS_sampling_results" / "analysis_closed_loop"

    print("[INFO] PROJECT_ROOT:", proj)
    print("[INFO] MERGED_DIR:  ", merged_dir)
    print("[INFO] FIGS_DIR:    ", figs_dir)
    print("[INFO] ANALYSIS_DIR:", analysis_dir)

    # A) Copy existing density/diff plots (no redraw)
    copy_existing_density_plots(proj, figs_dir)

    # Core tables
    points = load_points(merged_dir, analysis_dir)
    basin_master = load_basin_master(merged_dir, analysis_dir)
    reps = load_representatives(merged_dir, analysis_dir)

    # B/C
    plot_rmsd_overlay(points, figs_dir)
    plot_mass_cdf(points, figs_dir)

    # D/E (your key story plots)
    plot_basin_occupancy_rings(basin_master, figs_dir)
    plot_delta_rings(basin_master, figs_dir)

    # Key basins + optional story plots
    key_basin, contrast_basin = pick_key_basins_from_basin_master(basin_master)
    print(f"[INFO] Key basins: key={key_basin}, contrast={contrast_basin}")

    key_story = load_key_basin_story(analysis_dir)
    if key_story is not None:
        plot_waterfall_for_basin(key_story, key_basin, "G12C", figs_dir)
        plot_waterfall_for_basin(key_story, key_basin, "G12D", figs_dir)
        if contrast_basin != key_basin:
            plot_waterfall_for_basin(key_story, contrast_basin, "G12C", figs_dir)
            plot_waterfall_for_basin(key_story, contrast_basin, "G12D", figs_dir)
        print("[OK] F: Waterfall energy plots saved.")
    else:
        print("[WARN] key_basin_story.csv not found; skip F (waterfall).")

    per_res = load_per_residue_disp(analysis_dir)
    if per_res is not None:
        plot_per_residue_displacement(per_res, key_basin, figs_dir)
        if contrast_basin != key_basin:
            plot_per_residue_displacement(per_res, contrast_basin, figs_dir)
        print("[OK] G: Per-residue displacement plots saved.")
    else:
        print("[WARN] per_residue_displacement.csv not found; skip G.")

    # H
    plot_representatives_overlay(points, reps, figs_dir)

    print("[DONE] All publication-style figures written to:", figs_dir)


if __name__ == "__main__":
    main()




