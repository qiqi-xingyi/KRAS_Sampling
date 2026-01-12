# --*-- conding:utf-8 --*--
# @time:1/12/26 03:44
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_C_occupancy.py

# plot_C_occupancy.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
FIG_SUBDIR = "C_occupancy"    # outputs to KRAS_analysis/figs/C_occupancy/
PAIR_G12C = ("G12C", "WT")
PAIR_G12D = ("G12D", "WT")


# -----------------------------
# Path helpers
# -----------------------------
def kras_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, preferred_name: str) -> Path:
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
# Utilities
# -----------------------------
def _col_present(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def normalize_basin_id(df: pd.DataFrame) -> pd.DataFrame:
    if "basin_id" not in df.columns:
        raise ValueError("Expected column 'basin_id' not found.")
    df = df.copy()
    df["basin_id"] = pd.to_numeric(df["basin_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["basin_id"]).copy()
    df["basin_id"] = df["basin_id"].astype(int)
    return df


def detect_label_columns(occ: pd.DataFrame) -> List[str]:
    """
    occupancy file may be in wide format (columns WT,G12C,G12D) or long format.
    Return label columns if wide, else [].
    """
    labels = []
    for lab in ["WT", "G12C", "G12D"]:
        if lab in occ.columns:
            labels.append(lab)
    return labels


def read_occupancy_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_basin_id(df)

    wide_labels = detect_label_columns(df)
    if wide_labels:
        # expected columns: basin_id, WT, G12C, G12D (any subset)
        keep = ["basin_id"] + wide_labels
        return df[keep].copy()

    # long format support: (basin_id, label, occupancy/weight/prob)
    for cand in ["occupancy", "mass", "prob", "p_mass", "value", "weight"]:
        if cand in df.columns:
            val_col = cand
            break
    else:
        raise ValueError(
            f"{path.name}: cannot detect occupancy value column. "
            "Expected wide columns WT/G12C/G12D or long with a value column "
            "(occupancy/mass/prob/p_mass/value/weight)."
        )

    if "label" not in df.columns:
        raise ValueError(f"{path.name}: long format requires a 'label' column.")

    pivot = df.pivot_table(index="basin_id", columns="label", values=val_col, aggfunc="sum").reset_index()
    # keep a consistent order if present
    cols = ["basin_id"] + [c for c in ["WT", "G12C", "G12D"] if c in pivot.columns]
    return pivot[cols].copy()


def read_delta_ci(path: Path) -> pd.DataFrame:
    """
    Support several likely schemas.

    We aim to produce a normalized table:
      basin_id, pair, delta, ci_low, ci_high

    Accepted patterns:
    - columns: basin_id, pair, delta, ci_low, ci_high
    - columns: basin_id, comparison, delta, low, high
    - columns: basin_id, mutant, reference, delta, ci_low, ci_high
    - columns: basin_id, label, delta, ci_low, ci_high (where label denotes mutant vs WT)
    """
    df = pd.read_csv(path)
    df = normalize_basin_id(df)

    # rename likely CI columns
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["ci_low", "lower", "low", "lcl", "ci_lower", "lower_ci"]:
            rename_map[c] = "ci_low"
        elif cl in ["ci_high", "upper", "high", "ucl", "ci_upper", "upper_ci"]:
            rename_map[c] = "ci_high"
        elif cl in ["delta", "diff", "difference", "delta_occ", "delta_occupancy"]:
            rename_map[c] = "delta"
        elif cl in ["pair", "comparison", "cmp", "contrast"]:
            rename_map[c] = "pair"
    df = df.rename(columns=rename_map)

    # detect delta
    if "delta" not in df.columns:
        raise ValueError(f"{path.name}: cannot find delta column (delta/diff/difference).")

    # detect ci
    if "ci_low" not in df.columns or "ci_high" not in df.columns:
        raise ValueError(f"{path.name}: cannot find CI columns (ci_low/ci_high).")

    out = df.copy()

    # detect pair
    if "pair" in out.columns:
        out["pair"] = out["pair"].astype(str)
    else:
        # try mutant/reference columns
        if "mutant" in out.columns and "reference" in out.columns:
            out["pair"] = out["mutant"].astype(str) + "-vs-" + out["reference"].astype(str)
        elif "label" in out.columns:
            # interpret as label-vs-WT if WT exists as reference
            out["pair"] = out["label"].astype(str) + "-vs-WT"
        else:
            # fall back: single pair, unknown naming
            out["pair"] = "unknown"

    keep = ["basin_id", "pair", "delta", "ci_low", "ci_high"]
    return out[keep].copy()


def find_pair(df_ci: pd.DataFrame, mutant: str, ref: str) -> pd.DataFrame:
    """
    Select rows matching mutant-vs-ref from a 'pair' column.
    Robust to different separators/casing.
    """
    m = mutant.lower()
    r = ref.lower()

    pairs = df_ci["pair"].astype(str).str.lower()

    # patterns
    ok = (
        pairs.eq(f"{m}-vs-{r}") |
        pairs.eq(f"{m} vs {r}") |
        pairs.eq(f"{m}_vs_{r}") |
        pairs.eq(f"{m}-{r}") |
        pairs.eq(f"{m}/{r}") |
        pairs.eq(f"{m}vs{r}") |
        pairs.eq(f"{m}-vs-{r}".replace("-", "_")) |
        pairs.eq(f"{m}-vs-{r}".replace("-", "")) |
        pairs.str.contains(m) & pairs.str.contains(r) & pairs.str.contains("vs")
    )

    sub = df_ci[ok].copy()
    if not sub.empty:
        return sub

    # second attempt: maybe stored as "{mutant}-vs-WT" without ref casing
    ok2 = pairs.eq(f"{m}-vs-wt") if r == "wt" else ok
    sub = df_ci[ok2].copy()
    return sub


def order_by_abs_delta(delta_df: pd.DataFrame) -> List[int]:
    tmp = delta_df.copy()
    tmp["absd"] = np.abs(tmp["delta"].to_numpy(dtype=float))
    tmp = tmp.sort_values("absd", ascending=False)
    return tmp["basin_id"].tolist()


# -----------------------------
# Plotting
# -----------------------------
def plot_occupancy_bars(occ: pd.DataFrame, out_png: Path, out_pdf: Path):
    """
    Grouped bars: WT/G12C/G12D for each basin_id.
    """
    labels = [c for c in ["WT", "G12C", "G12D"] if c in occ.columns]
    if not labels:
        raise ValueError("Occupancy table has no WT/G12C/G12D columns to plot.")

    basins = sorted(occ["basin_id"].unique().tolist())
    x = np.arange(len(basins), dtype=float)

    width = 0.22 if len(labels) == 3 else 0.28
    offsets = np.linspace(-width, width, num=len(labels))

    plt.figure(figsize=(7.6, 4.6))
    ax = plt.gca()

    for i, lab in enumerate(labels):
        y = occ.set_index("basin_id").loc[basins, lab].to_numpy(dtype=float)
        ax.bar(x + offsets[i], y, width=width, label=lab)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in basins])
    ax.set_xlabel("Basin ID")
    ax.set_ylabel("Occupancy (probability mass)")
    ax.set_title("Basin occupancy by genotype")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


def plot_delta_ci(
    df_ci: pd.DataFrame,
    mutant: str,
    ref: str,
    out_png: Path,
    out_pdf: Path,
    title: str,
):
    """
    Dot + CI errorbars. Sorted by |delta|.
    """
    sub = find_pair(df_ci, mutant=mutant, ref=ref)
    if sub.empty:
        raise ValueError(f"No matching pair found for {mutant} vs {ref} in occupancy_delta_ci.csv")

    # ensure numeric
    sub = sub.copy()
    sub["delta"] = pd.to_numeric(sub["delta"], errors="coerce")
    sub["ci_low"] = pd.to_numeric(sub["ci_low"], errors="coerce")
    sub["ci_high"] = pd.to_numeric(sub["ci_high"], errors="coerce")
    sub = sub.dropna(subset=["delta", "ci_low", "ci_high"])

    # sort by |delta|
    basin_order = order_by_abs_delta(sub)
    sub["basin_id"] = sub["basin_id"].astype(int)
    sub = sub.set_index("basin_id").loc[basin_order].reset_index()

    y = np.arange(len(sub), dtype=float)
    delta = sub["delta"].to_numpy(dtype=float)
    lo = sub["ci_low"].to_numpy(dtype=float)
    hi = sub["ci_high"].to_numpy(dtype=float)

    # convert to symmetric errorbar lengths
    xerr = np.vstack([delta - lo, hi - delta])

    plt.figure(figsize=(7.6, 4.8))
    ax = plt.gca()

    ax.errorbar(delta, y, xerr=xerr, fmt="o", capsize=3, elinewidth=1.2)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels([str(b) for b in sub["basin_id"].tolist()])
    ax.set_xlabel(f"Δ occupancy ({mutant} − {ref})")
    ax.set_ylabel("Basin ID (sorted by |Δ|)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    root = kras_root_from_script()
    data_used = root / "data_used"
    figs_root = root / "figs"
    out_dir = figs_root / FIG_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    occupancy_path = pick_file(data_used, "basin_occupancy.csv")
    delta_ci_path = pick_file(data_used, "occupancy_delta_ci.csv")
    delta_summary_path = pick_file(data_used, "basin_delta_summary.csv")

    occ = read_occupancy_table(occupancy_path)
    df_ci = read_delta_ci(delta_ci_path)

    # (Optional) Use basin_delta_summary.csv just for traceability / manifest;
    # ordering is derived from occupancy_delta_ci per pair.
    _ = pd.read_csv(delta_summary_path)

    # C1: grouped occupancy bars
    c1_png = out_dir / "C1_basin_occupancy_bars.png"
    c1_pdf = out_dir / "C1_basin_occupancy_bars.pdf"
    plot_occupancy_bars(occ, c1_png, c1_pdf)

    # C2: Δ occupancy + CI for G12C vs WT
    c2_png = out_dir / "C2_delta_occupancy_G12C_minus_WT.png"
    c2_pdf = out_dir / "C2_delta_occupancy_G12C_minus_WT.pdf"
    plot_delta_ci(
        df_ci=df_ci,
        mutant="G12C",
        ref="WT",
        out_png=c2_png,
        out_pdf=c2_pdf,
        title="Δ occupancy with 95% CI (G12C − WT)",
    )

    # C3: Δ occupancy + CI for G12D vs WT
    c3_png = out_dir / "C3_delta_occupancy_G12D_minus_WT.png"
    c3_pdf = out_dir / "C3_delta_occupancy_G12D_minus_WT.pdf"
    plot_delta_ci(
        df_ci=df_ci,
        mutant="G12D",
        ref="WT",
        out_png=c3_png,
        out_pdf=c3_pdf,
        title="Δ occupancy with 95% CI (G12D − WT)",
    )

    # Manifest
    manifest = out_dir / "C_manifest.json"
    payload = {
        "inputs": {
            "basin_occupancy": str(occupancy_path),
            "occupancy_delta_ci": str(delta_ci_path),
            "basin_delta_summary": str(delta_summary_path),
        },
        "outputs": [str(c1_png), str(c1_pdf), str(c2_png), str(c2_pdf), str(c3_png), str(c3_pdf)],
        "notes": {
            "C1": "Grouped bar chart of basin occupancy for WT/G12C/G12D.",
            "C2": "Dot+errorbar plot of Δ occupancy (G12C−WT) with 95% CI, basins sorted by |Δ|.",
            "C3": "Dot+errorbar plot of Δ occupancy (G12D−WT) with 95% CI, basins sorted by |Δ|.",
        },
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[DONE] Saved to:", out_dir)
    for p in payload["outputs"]:
        print("  -", Path(p).name)
    print("  -", manifest.name)


if __name__ == "__main__":
    main()
