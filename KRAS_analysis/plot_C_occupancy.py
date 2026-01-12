# plot_C_occupancy.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Parameters (IDE-friendly)
# -----------------------------
FIG_SUBDIR = "C_occupancy"  # outputs to KRAS_analysis/figs/C_occupancy/


# -----------------------------
# Path helpers
# -----------------------------
def kras_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def pick_file(data_used_dir: Path, preferred_name: str) -> Path:
    # Prefer numbered filenames like "07_occupancy_delta_ci.csv"
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
def normalize_basin_id(df: pd.DataFrame) -> pd.DataFrame:
    if "basin_id" not in df.columns:
        raise ValueError("Expected column 'basin_id' not found.")
    out = df.copy()
    out["basin_id"] = pd.to_numeric(out["basin_id"], errors="coerce")
    out = out.dropna(subset=["basin_id"]).copy()
    out["basin_id"] = out["basin_id"].astype(int)
    return out


def detect_label_columns_wide(df: pd.DataFrame) -> List[str]:
    labels = []
    for lab in ["WT", "G12C", "G12D"]:
        if lab in df.columns:
            labels.append(lab)
    return labels


def read_occupancy_table(path: Path) -> pd.DataFrame:
    """
    Supports:
    - wide: basin_id, WT, G12C, G12D
    - long: basin_id, label, value (occupancy/mass/prob/p_mass/value/weight)
    """
    df = pd.read_csv(path)
    df = normalize_basin_id(df)

    wide_labels = detect_label_columns_wide(df)
    if wide_labels:
        keep = ["basin_id"] + wide_labels
        return df[keep].copy()

    # long format
    if "label" not in df.columns:
        raise ValueError(
            f"{path.name}: cannot detect wide WT/G12C/G12D columns, and no 'label' column for long format."
        )

    val_col = None
    for cand in ["occupancy", "mass", "prob", "p_mass", "value", "weight"]:
        if cand in df.columns:
            val_col = cand
            break
    if val_col is None:
        raise ValueError(
            f"{path.name}: long format requires a value column among "
            "occupancy/mass/prob/p_mass/value/weight."
        )

    pivot = df.pivot_table(index="basin_id", columns="label", values=val_col, aggfunc="sum").reset_index()
    cols = ["basin_id"] + [c for c in ["WT", "G12C", "G12D"] if c in pivot.columns]
    return pivot[cols].copy()


def _find_ci_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Robust CI column detection. Your file uses ci95_lo/ci95_hi.
    This also supports: ci_low/ci_high, low/high, lower/upper, etc.
    """
    cols = list(df.columns)
    low_candidates = []
    high_candidates = []

    for c in cols:
        cl = c.lower()
        if ("ci" in cl and ("lo" in cl or "low" in cl or "lower" in cl)) or cl in ["low", "lower"]:
            low_candidates.append(c)
        if ("ci" in cl and ("hi" in cl or "high" in cl or "upper" in cl)) or cl in ["high", "upper"]:
            high_candidates.append(c)

    # Prefer ci95_lo/ci95_hi if present
    if "ci95_lo" in df.columns and "ci95_hi" in df.columns:
        return "ci95_lo", "ci95_hi"

    if "ci_low" in df.columns and "ci_high" in df.columns:
        return "ci_low", "ci_high"

    # fallback: pick first matched
    if low_candidates and high_candidates:
        return low_candidates[0], high_candidates[0]

    raise ValueError("cannot find CI columns (e.g., ci_low/ci_high or ci95_lo/ci95_hi).")


def read_delta_ci(path: Path) -> pd.DataFrame:
    """
    Normalizes to columns:
      basin_id, pair, delta, ci_low, ci_high
    """
    df = pd.read_csv(path)
    df = normalize_basin_id(df)

    if "delta" not in df.columns:
        # allow diff/difference variants
        for cand in ["diff", "difference", "delta_occ", "delta_occupancy"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "delta"})
                break
    if "delta" not in df.columns:
        raise ValueError(f"{path.name}: cannot find delta column (delta/diff/difference).")

    lo_col, hi_col = _find_ci_cols(df)

    if "pair" not in df.columns:
        # allow comparison/contrast variants
        for cand in ["comparison", "contrast", "cmp"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "pair"})
                break
    if "pair" not in df.columns:
        # last resort: create an unknown pair
        df["pair"] = "unknown"

    out = df.copy()
    out = out.rename(columns={lo_col: "ci_low", hi_col: "ci_high"})
    out["pair"] = out["pair"].astype(str)

    keep = ["basin_id", "pair", "delta", "ci_low", "ci_high"]
    return out[keep].copy()


def find_pair(df_ci: pd.DataFrame, mutant: str, ref: str) -> pd.DataFrame:
    """
    Match a pair from df_ci['pair'] robustly:
    supports: 'G12D-minus-WT', 'G12D_vs_WT', 'G12D-vs-WT', 'G12D WT', etc.
    """
    m = mutant.lower()
    r = ref.lower()
    pairs = df_ci["pair"].astype(str).str.lower().str.replace("_", "-").str.replace(" ", "")

    # normalize common separators/words
    pairs_norm = (
        pairs.str.replace("minus", "-minus-", regex=False)
             .str.replace("vs", "-vs-", regex=False)
    )

    def _ok(s: pd.Series) -> pd.Series:
        return (
            s.str.contains(m, regex=False)
            & s.str.contains(r, regex=False)
            & (s.str.contains("minus", regex=False) | s.str.contains("vs", regex=False) | s.str.contains("-", regex=False))
        )

    sub = df_ci[_ok(pairs_norm)].copy()
    if not sub.empty:
        return sub

    # fallback exact patterns
    exact = [
        f"{m}-minus-{r}",
        f"{m}-vs-{r}",
        f"{m}-{r}",
    ]
    sub = df_ci[pairs.isin(exact)].copy()
    return sub


def order_by_abs_delta(df: pd.DataFrame) -> List[int]:
    tmp = df.copy()
    tmp["delta"] = pd.to_numeric(tmp["delta"], errors="coerce")
    tmp = tmp.dropna(subset=["delta"])
    tmp["absd"] = np.abs(tmp["delta"].to_numpy(dtype=float))
    tmp = tmp.sort_values("absd", ascending=False)
    return tmp["basin_id"].astype(int).tolist()


# -----------------------------
# Plotting
# -----------------------------
def plot_occupancy_bars(occ: pd.DataFrame, out_png: Path, out_pdf: Path):
    labels = [c for c in ["WT", "G12C", "G12D"] if c in occ.columns]
    if not labels:
        raise ValueError("Occupancy table has no WT/G12C/G12D columns to plot.")

    basins = sorted(occ["basin_id"].unique().tolist())
    x = np.arange(len(basins), dtype=float)

    width = 0.22 if len(labels) == 3 else 0.28
    offsets = np.linspace(-width, width, num=len(labels))

    plt.figure(figsize=(7.6, 4.6))
    ax = plt.gca()

    idx = occ.set_index("basin_id")
    for i, lab in enumerate(labels):
        y = pd.to_numeric(idx.loc[basins, lab], errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
    sub = find_pair(df_ci, mutant=mutant, ref=ref)
    if sub.empty:
        raise ValueError(f"No matching pair found for {mutant} vs {ref} in occupancy_delta_ci.csv")

    sub = sub.copy()
    sub["delta"] = pd.to_numeric(sub["delta"], errors="coerce")
    sub["ci_low"] = pd.to_numeric(sub["ci_low"], errors="coerce")
    sub["ci_high"] = pd.to_numeric(sub["ci_high"], errors="coerce")
    sub = sub.dropna(subset=["delta", "ci_low", "ci_high"])

    basin_order = order_by_abs_delta(sub)
    sub = sub.set_index("basin_id").loc[basin_order].reset_index()

    y = np.arange(len(sub), dtype=float)
    delta = sub["delta"].to_numpy(dtype=float)
    lo = sub["ci_low"].to_numpy(dtype=float)
    hi = sub["ci_high"].to_numpy(dtype=float)

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
    delta_summary_path = pick_file(data_used, "basin_delta_summary.csv")  # just for manifest trace

    occ = read_occupancy_table(occupancy_path)
    df_ci = read_delta_ci(delta_ci_path)
    _ = pd.read_csv(delta_summary_path)

    # C1
    c1_png = out_dir / "C1_basin_occupancy_bars.png"
    c1_pdf = out_dir / "C1_basin_occupancy_bars.pdf"
    plot_occupancy_bars(occ, c1_png, c1_pdf)

    # C2
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

    # C3
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
            "schema_fix": "CI columns auto-detected (supports ci95_lo/ci95_hi). Pair matching supports '*-minus-*' format.",
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
