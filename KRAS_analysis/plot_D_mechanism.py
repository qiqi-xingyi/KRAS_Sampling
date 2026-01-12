# --*-- conding:utf-8 --*--
# @time:1/12/26 03:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_D_mechanism.py

# plot_D_mechanism.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
FIG_SUBDIR = "D_mechanism"  # outputs to KRAS_analysis/figs/D_mechanism/

GENOTYPES = ["WT", "G12C", "G12D"]

# Key panels (you can tweak these without touching plotting code)
KEY_ENERGY_METRICS = ["E_total", "E_steric"]
KEY_FEATURE_METRICS = ["feat_clash_count", "feat_Rg", "feat_contact_density", "feat_end_to_end"]

# For energy-term breakdown (from basin_energy_contrast.csv)
TERM_ORDER_PREFERRED = [
    "E_steric",
    "E_geom",
    "E_mj",
    "E_dihedral",
    "E_hydroph",
    "E_cbeta",
    "E_rama",
    "E_bond",
]


# -----------------------------
# Path helpers
# -----------------------------
def kras_root_from_script() -> Path:
    # This script lives in KRAS_analysis/
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, preferred_name: str) -> Path:
    """
    Prefer numbered filenames in data_used (e.g., 08_basin_energy_contrast.csv),
    but also support raw base name if present.
    """
    candidates = sorted(data_used_dir.glob(f"*_{preferred_name}"))
    if candidates:
        return candidates[0]
    p = data_used_dir / preferred_name
    if p.exists():
        return p
    candidates = sorted(data_used_dir.glob(f"*{preferred_name}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Cannot find '{preferred_name}' under: {data_used_dir}")


# -----------------------------
# IO + schema normalization
# -----------------------------
def require_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def read_story(data_used: Path) -> pd.DataFrame:
    """
    Prefer key_basin_story_ranked.csv (already in impact order),
    fallback to key_basin_story.csv.
    """
    try:
        p = pick_file(data_used, "key_basin_story_ranked.csv")
    except FileNotFoundError:
        p = pick_file(data_used, "key_basin_story.csv")

    df = pd.read_csv(p)
    require_cols(df, ["basin_id"], p.name)
    df = df.copy()
    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce")
    df = df.dropna(subset=["basin_id"]).copy()
    df["basin_id"] = df["basin_id"].astype(int)
    df.attrs["__source_path__"] = str(p)
    return df


def read_energy_contrast(data_used: Path) -> pd.DataFrame:
    p = pick_file(data_used, "basin_energy_contrast.csv")
    df = pd.read_csv(p)
    require_cols(df, ["basin_id", "term"], p.name)
    df = df.copy()
    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce")
    df = df.dropna(subset=["basin_id"]).copy()
    df["basin_id"] = df["basin_id"].astype(int)
    df["term"] = df["term"].astype(str)
    df.attrs["__source_path__"] = str(p)
    return df


def read_delta_summary(data_used: Path) -> pd.DataFrame:
    p = pick_file(data_used, "basin_delta_summary.csv")
    df = pd.read_csv(p)
    require_cols(df, ["basin_id"], p.name)
    df = df.copy()
    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce")
    df = df.dropna(subset=["basin_id"]).copy()
    df["basin_id"] = df["basin_id"].astype(int)
    df.attrs["__source_path__"] = str(p)
    return df


# -----------------------------
# Helpers
# -----------------------------
def metric_cols(metric: str) -> Dict[str, str]:
    """
    For a metric base name like 'E_total', return the expected per-genotype columns:
      E_total_WT, E_total_G12C, E_total_G12D
    """
    return {g: f"{metric}_{g}" for g in GENOTYPES}


def delta_col(metric: str, mutant: str, ref: str = "WT") -> str:
    return f"delta_{metric}_{mutant}_minus_{ref}"


def safe_get(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    return df[col] if col in df.columns else None


def get_key_basins(story: pd.DataFrame, max_n: int = 4) -> List[int]:
    """
    Use impact_G12D if present, else fall back to abs delta mass for G12D, else row order.
    """
    s = story.copy()
    if "impact_G12D" in s.columns:
        s["impact_G12D"] = pd.to_numeric(s["impact_G12D"], errors="coerce")
        s = s.sort_values("impact_G12D", ascending=False)
    elif "delta_mass_G12D_minus_WT" in s.columns:
        s["absd"] = np.abs(pd.to_numeric(s["delta_mass_G12D_minus_WT"], errors="coerce"))
        s = s.sort_values("absd", ascending=False)
    return s["basin_id"].astype(int).tolist()[:max_n]


def ordered_terms(terms: List[str]) -> List[str]:
    """
    Use preferred order when possible; append remaining terms.
    """
    tset = set(terms)
    out = [t for t in TERM_ORDER_PREFERRED if t in tset]
    out += [t for t in terms if t not in set(out)]
    return out


# -----------------------------
# Plotting primitives
# -----------------------------
def plot_grouped_bars_by_basin(
    story: pd.DataFrame,
    metric: str,
    basins: List[int],
    out_png: Path,
    out_pdf: Path,
    title: str,
    y_label: str,
):
    cols_map = metric_cols(metric)
    missing = [c for c in cols_map.values() if c not in story.columns]
    if missing:
        raise ValueError(f"Story table missing columns for metric '{metric}': {missing}")

    s = story.set_index("basin_id").loc[basins].reset_index()

    x = np.arange(len(basins), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, num=len(GENOTYPES))

    plt.figure(figsize=(7.8, 4.6))
    ax = plt.gca()

    for i, g in enumerate(GENOTYPES):
        y = pd.to_numeric(s[cols_map[g]], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + offsets[i], y, width=width, label=g)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in basins])
    ax.set_xlabel("Basin ID (key basins)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_features_2x2(
    story: pd.DataFrame,
    basins: List[int],
    out_png: Path,
    out_pdf: Path,
    title: str,
):
    """
    One figure, 2x2 panels, each panel is grouped bars for a feature metric across key basins.
    """
    # Collect available features
    feats = []
    for m in KEY_FEATURE_METRICS:
        cols_map = metric_cols(m)
        if all(c in story.columns for c in cols_map.values()):
            feats.append(m)

    if not feats:
        raise ValueError("None of the requested feature metrics are present in key_basin_story table.")

    # Ensure 4 panels layout; if fewer than 4, we still keep 2x2 and hide extras
    feats = feats[:4]

    s = story.set_index("basin_id").loc[basins].reset_index()
    x = np.arange(len(basins), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, num=len(GENOTYPES))

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    axes = axes.ravel()

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(feats):
            ax.axis("off")
            continue

        m = feats[ax_i]
        cols_map = metric_cols(m)

        for i, g in enumerate(GENOTYPES):
            y = pd.to_numeric(s[cols_map[g]], errors="coerce").to_numpy(dtype=float)
            ax.bar(x + offsets[i], y, width=width, label=g)

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in basins])
        ax.set_xlabel("Basin ID")

        # nicer y-label per feature
        if m == "feat_clash_count":
            ax.set_ylabel("Clash count")
            ax.set_title("Clash count")
        elif m == "feat_Rg":
            ax.set_ylabel("Rg")
            ax.set_title("Radius of gyration")
        elif m == "feat_contact_density":
            ax.set_ylabel("Contact density")
            ax.set_title("Contact density")
        elif m == "feat_end_to_end":
            ax.set_ylabel("End-to-end")
            ax.set_title("End-to-end distance")
        else:
            ax.set_ylabel(m)
            ax.set_title(m)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(title, y=1.06)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_bridge_scatter(
    story: pd.DataFrame,
    delta: pd.DataFrame,
    mutant: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
    energy_metric: str = "E_steric",
    highlight_basins: Optional[List[int]] = None,
):
    """
    Scatter: x = ΔE_metric (mutant - WT) from story table,
             y = Δoccupancy (mutant - WT) from basin_delta_summary.
    """
    if highlight_basins is None:
        highlight_basins = []

    # x: delta energy from story
    xcol = delta_col(energy_metric, mutant=mutant, ref="WT")
    if xcol not in story.columns:
        raise ValueError(f"Story table missing energy delta column: {xcol}")

    # y: delta occupancy from delta summary
    ycol = f"delta_{mutant}_minus_WT"
    if ycol not in delta.columns:
        # fallback naming
        ycol = f"delta_{mutant}_minus_WT".replace(mutant, mutant)  # no-op, keep for clarity
    if ycol not in delta.columns and f"delta_{mutant}_minus_WT" not in delta.columns:
        # in your basin_delta_summary.csv it is delta_G12D_minus_WT, delta_G12C_minus_WT
        ycol = f"delta_{mutant}_minus_WT"
    if ycol not in delta.columns:
        # final attempt: exact known pattern
        ycol = f"delta_{mutant}_minus_WT"
        if ycol not in delta.columns:
            ycol = f"delta_{mutant}_minus_WT".replace(mutant, mutant)

    # exact known columns from your file:
    if mutant == "G12C" and "delta_G12C_minus_WT" in delta.columns:
        ycol = "delta_G12C_minus_WT"
    if mutant == "G12D" and "delta_G12D_minus_WT" in delta.columns:
        ycol = "delta_G12D_minus_WT"

    require_cols(delta, ["basin_id", ycol], "basin_delta_summary.csv")

    # merge
    sx = story[["basin_id", xcol]].copy()
    sy = delta[["basin_id", ycol]].copy()
    sx[xcol] = pd.to_numeric(sx[xcol], errors="coerce")
    sy[ycol] = pd.to_numeric(sy[ycol], errors="coerce")

    m = sx.merge(sy, on="basin_id", how="inner").dropna()

    plt.figure(figsize=(6.6, 5.2))
    ax = plt.gca()

    ax.scatter(m[xcol], m[ycol], s=45, alpha=0.75)

    # highlight key basins with labels
    for b in highlight_basins:
        row = m[m["basin_id"] == b]
        if row.empty:
            continue
        x = float(row[xcol].iloc[0])
        y = float(row[ycol].iloc[0])
        ax.scatter([x], [y], s=90)
        ax.text(x, y, f"{b}", fontsize=11, ha="left", va="bottom")

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_xlabel(f"Δ {energy_metric} ({mutant} − WT)")
    ax.set_ylabel(f"Δ occupancy ({mutant} − WT)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_energy_term_breakdown_topbasins(
    energy: pd.DataFrame,
    basins: List[int],
    out_png: Path,
    out_pdf: Path,
    title: str,
    mutant: str = "G12D",
):
    """
    2x2 panel: for each top basin, bar chart of ΔE_term (mutant - WT) using basin_energy_contrast.csv
    """
    dcol = f"delta_{mutant}_minus_WT"
    # your file uses delta_G12D_minus_WT column names like delta_G12D_minus_WT
    if mutant == "G12D" and "delta_G12D_minus_WT" in energy.columns:
        dcol = "delta_G12D_minus_WT"
    if mutant == "G12C" and "delta_G12C_minus_WT" in energy.columns:
        dcol = "delta_G12C_minus_WT"

    require_cols(energy, ["basin_id", "term", dcol], "basin_energy_contrast.csv")

    basins = basins[:4]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    axes = axes.ravel()

    for ax_i, ax in enumerate(axes):
        if ax_i >= len(basins):
            ax.axis("off")
            continue

        b = basins[ax_i]
        sub = energy[energy["basin_id"] == b].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub[dcol] = pd.to_numeric(sub[dcol], errors="coerce")
        sub = sub.dropna(subset=[dcol])

        terms = sub["term"].astype(str).tolist()
        terms_ord = ordered_terms(terms)

        sub = sub.set_index("term").loc[terms_ord].reset_index()
        y = sub[dcol].to_numpy(dtype=float)
        x = np.arange(len(sub), dtype=float)

        ax.bar(x, y)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["term"].tolist(), rotation=45, ha="right")
        ax.set_ylabel(f"ΔE ({mutant} − WT)")
        ax.set_title(f"Basin {b}: energy-term breakdown")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    root = kras_root_from_script()
    data_used = root / "data_used"
    figs_root = root / "figs"
    out_dir = figs_root / FIG_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    story = read_story(data_used)
    energy = read_energy_contrast(data_used)
    delta = read_delta_summary(data_used)

    story_path = story.attrs.get("__source_path__", "")
    energy_path = energy.attrs.get("__source_path__", "")
    delta_path = delta.attrs.get("__source_path__", "")

    key_basins = get_key_basins(story, max_n=4)

    outputs: List[str] = []

    # D1/D2: key energy metrics across key basins
    for i, metric in enumerate(KEY_ENERGY_METRICS, start=1):
        cols_map = metric_cols(metric)
        if not all(c in story.columns for c in cols_map.values()):
            print(f"[WARN] Skip {metric}: missing columns in story table.")
            continue

        out_png = out_dir / f"D{i}_keybasins_{metric}.png"
        out_pdf = out_dir / f"D{i}_keybasins_{metric}.pdf"
        plot_grouped_bars_by_basin(
            story=story,
            metric=metric,
            basins=key_basins,
            out_png=out_png,
            out_pdf=out_pdf,
            title=f"Key basins: {metric} by genotype",
            y_label=metric,
        )
        outputs += [str(out_png), str(out_pdf)]

    # D3: features (2x2)
    out_png = out_dir / "D3_keybasins_features_2x2.png"
    out_pdf = out_dir / "D3_keybasins_features_2x2.pdf"
    plot_features_2x2(
        story=story,
        basins=key_basins,
        out_png=out_png,
        out_pdf=out_pdf,
        title="Key basins: structural features by genotype",
    )
    outputs += [str(out_png), str(out_pdf)]

    # D4/D5: bridge plot (ΔE vs Δoccupancy)
    out_png = out_dir / "D4_bridge_G12C_deltaEsteric_vs_deltaOcc.png"
    out_pdf = out_dir / "D4_bridge_G12C_deltaEsteric_vs_deltaOcc.pdf"
    plot_bridge_scatter(
        story=story,
        delta=delta,
        mutant="G12C",
        out_png=out_png,
        out_pdf=out_pdf,
        title="Bridge: ΔE_steric vs Δoccupancy (G12C − WT)",
        energy_metric="E_steric",
        highlight_basins=key_basins,
    )
    outputs += [str(out_png), str(out_pdf)]

    out_png = out_dir / "D5_bridge_G12D_deltaEsteric_vs_deltaOcc.png"
    out_pdf = out_dir / "D5_bridge_G12D_deltaEsteric_vs_deltaOcc.pdf"
    plot_bridge_scatter(
        story=story,
        delta=delta,
        mutant="G12D",
        out_png=out_png,
        out_pdf=out_pdf,
        title="Bridge: ΔE_steric vs Δoccupancy (G12D − WT)",
        energy_metric="E_steric",
        highlight_basins=key_basins,
    )
    outputs += [str(out_png), str(out_pdf)]

    # D6: energy-term breakdown for top basins (2x2)
    out_png = out_dir / "D6_energy_term_breakdown_topbasins_G12D.png"
    out_pdf = out_dir / "D6_energy_term_breakdown_topbasins_G12D.pdf"
    plot_energy_term_breakdown_topbasins(
        energy=energy,
        basins=key_basins,
        out_png=out_png,
        out_pdf=out_pdf,
        title="Top basins: energy-term breakdown (G12D − WT)",
        mutant="G12D",
    )
    outputs += [str(out_png), str(out_pdf)]

    # Manifest
    manifest = out_dir / "D_manifest.json"
    payload = {
        "inputs": {
            "key_basin_story": story_path,
            "basin_energy_contrast": energy_path,
            "basin_delta_summary": delta_path,
        },
        "key_basins": key_basins,
        "params": {
            "FIG_SUBDIR": FIG_SUBDIR,
            "GENOTYPES": GENOTYPES,
            "KEY_ENERGY_METRICS": KEY_ENERGY_METRICS,
            "KEY_FEATURE_METRICS": KEY_FEATURE_METRICS,
            "TERM_ORDER_PREFERRED": TERM_ORDER_PREFERRED,
        },
        "outputs": outputs,
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in outputs:
        print("  -", Path(p).name)
    print("  -", manifest.name)


if __name__ == "__main__":
    main()
