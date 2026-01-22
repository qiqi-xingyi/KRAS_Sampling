# --*-- coding:utf-8 --*--
# @time:1/21/26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:plot_KRAS_rmsd_track.py

# Scheme A1: Residue-track RMSD map (journal style)
# - x-axis: residue index (res_start..res_end)
# - y-axis: tracks by (Label, Chain)
# - each fragment: colored segment encodes min_rmsd (no midpoint circle)
#   optional: gray "rail" underlay for readability
#
# Input:
#   <project_root>/pp_result/result_summary.csv
#
# Output:
#   <project_root>/pp_result/figs_KRAS_RMSD/
#     KRAS_RMSD_track.png
#     KRAS_RMSD_track.pdf

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# -----------------------------
# Config
# -----------------------------
FIGSIZE = (10.8, 8.9)
DPI = 600  # REQUIRED

TRACK_ORDER = [("WT", "A"), ("WT", "B"), ("G12C", "A"), ("G12D", "A")]
LABEL_ORDER = ["WT", "G12C", "G12D"]

# Segment styling (A1)
RAIL_ENABLE = True
RAIL_COLOR = "#C7C7C7"
RAIL_ALPHA = 0.90
RAIL_LINEWIDTH_ADD = 2.5  # rail thickness = SEG_LINEWIDTH + this

SEG_LINEWIDTH = 6.0
SEG_ALPHA = 0.95

# Text
TEXT_COLOR = "#111111"
TEXT_FONTSIZE = 9
TEXT_DY = 0.16  # offset above the segment in track coordinates

# Axis / grid
GRID_COLOR = "#D9D9D9"
GRID_ALPHA = 0.85
SPINE_COLOR = "#777777"

# Colormap for RMSD
RMSD_COLOR_MAP = "viridis"
RMSD_TICKS = [1.5, 2.0, 2.5, 3.0]


# -----------------------------
# Paths
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_out_dir(root: Path) -> Path:
    out_dir = root / "pp_result" / "figs_KRAS_RMSD"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------
# Data helpers
# -----------------------------
def infer_label(fragment_id: str) -> Optional[str]:
    s = str(fragment_id).upper()
    if "G12C" in s:
        return "G12C"
    if "G12D" in s:
        return "G12D"
    if "WT" in s:
        return "WT"
    return None


def normalize_chain(chain: str) -> str:
    return str(chain).strip().upper()


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = ["fragment_id", "min_rmsd", "chain", "res_start", "res_end"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path.name}")

    d = df.copy()
    d["fragment_id"] = d["fragment_id"].astype(str)
    d["label"] = d["fragment_id"].apply(infer_label)
    d["chain"] = d["chain"].apply(normalize_chain)

    d["min_rmsd"] = pd.to_numeric(d["min_rmsd"], errors="coerce")
    d["res_start"] = pd.to_numeric(d["res_start"], errors="coerce")
    d["res_end"] = pd.to_numeric(d["res_end"], errors="coerce")

    d = d.dropna(subset=["label", "min_rmsd", "res_start", "res_end", "chain"]).copy()
    d["res_start"] = d["res_start"].astype(int)
    d["res_end"] = d["res_end"].astype(int)

    swap_mask = d["res_start"] > d["res_end"]
    if swap_mask.any():
        tmp = d.loc[swap_mask, "res_start"].copy()
        d.loc[swap_mask, "res_start"] = d.loc[swap_mask, "res_end"]
        d.loc[swap_mask, "res_end"] = tmp

    return d


def build_tracks(d: pd.DataFrame) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    present = set(zip(d["label"].tolist(), d["chain"].tolist()))
    tracks: List[Tuple[str, str]] = [t for t in TRACK_ORDER if t in present]

    remaining = sorted(
        present - set(tracks),
        key=lambda x: (LABEL_ORDER.index(x[0]) if x[0] in LABEL_ORDER else 99, x[1]),
    )
    tracks.extend(remaining)

    # Assign y positions (top -> bottom)
    # We'll map tracks_top_to_bottom to descending y (higher y at top)
    tracks_top_to_bottom = tracks
    y_map: Dict[Tuple[str, str], float] = {t: float(len(tracks_top_to_bottom) - 1 - i) for i, t in enumerate(tracks_top_to_bottom)}

    d2 = d.copy()
    d2["track_key"] = list(zip(d2["label"], d2["chain"]))
    d2["y"] = d2["track_key"].map(y_map).astype(float)
    return d2, tracks_top_to_bottom


# -----------------------------
# Plot
# -----------------------------
def plot_track_map(d: pd.DataFrame, tracks_top_to_bottom: List[Tuple[str, str]], out_png: Path, out_pdf: Path):
    if d.empty:
        raise ValueError("No valid rows to plot.")

    xmin = int(d["res_start"].min())
    xmax = int(d["res_end"].max())
    pad = max(2, int(0.04 * (xmax - xmin + 1)))
    xlim = (xmin - pad, xmax + pad)

    rmin = float(d["min_rmsd"].min())
    rmax = float(d["min_rmsd"].max())
    if abs(rmax - rmin) < 1e-9:
        rmin = max(0.0, rmin - 0.5)
        rmax = rmax + 0.5

    norm = Normalize(vmin=rmin, vmax=rmax)
    cmap = plt.get_cmap(RMSD_COLOR_MAP)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("white")

    # Track guide lines + labels
    yticks = []
    yticklabels = []
    for (lab, ch) in tracks_top_to_bottom:
        sub = d[(d["label"] == lab) & (d["chain"] == ch)]
        if sub.empty:
            continue
        y = float(sub["y"].iloc[0])

        ax.axhline(y, color=GRID_COLOR, linewidth=1.0, alpha=GRID_ALPHA, zorder=0)
        yticks.append(y)
        yticklabels.append(f"{lab}  |  Chain {ch}")

    # Draw segments (A1: colored segments encode RMSD; optional gray rail underlay)
    for _, row in d.iterrows():
        y = float(row["y"])
        x0 = int(row["res_start"])
        x1 = int(row["res_end"])
        xm = 0.5 * (x0 + x1)

        rmsd = float(row["min_rmsd"])
        color = cmap(norm(rmsd))

        if RAIL_ENABLE:
            ax.plot(
                [x0, x1], [y, y],
                color=RAIL_COLOR,
                alpha=RAIL_ALPHA,
                linewidth=SEG_LINEWIDTH + RAIL_LINEWIDTH_ADD,
                solid_capstyle="round",
                zorder=1,
            )

        ax.plot(
            [x0, x1], [y, y],
            color=color,
            alpha=SEG_ALPHA,
            linewidth=SEG_LINEWIDTH,
            solid_capstyle="round",
            zorder=2,
        )

        # Annotation: residue range + RMSD
        txt = f"{x0}-{x1}  ({rmsd:.2f} Å)"
        ax.text(
            xm, y + TEXT_DY,
            txt,
            ha="center",
            va="bottom",
            fontsize=TEXT_FONTSIZE,
            color=TEXT_COLOR,
            zorder=3,
        )

    # Axes styling
    ax.set_xlim(*xlim)
    if yticks:
        ax.set_ylim(min(yticks) - 0.8, max(yticks) + 0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Residue index")
    ax.set_title("KRAS fragment accuracy map (RMSD, Å)")

    for sp in ax.spines.values():
        sp.set_color(SPINE_COLOR)
        sp.set_linewidth(0.9)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Colorbar
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("min RMSD (Å)")
    ticks = [t for t in RMSD_TICKS if rmin <= t <= rmax]
    if ticks:
        cbar.set_ticks(ticks)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)  # vector; dpi not needed but harmless
    plt.close(fig)


def main():
    root = project_root_from_tools_dir()
    in_csv = root / "pp_result" / "result_summary.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input: {in_csv}")

    out_dir = ensure_out_dir(root)
    out_png = out_dir / "KRAS_RMSD_track.png"
    out_pdf = out_dir / "KRAS_RMSD_track.pdf"

    d = load_and_prepare(in_csv)
    d2, tracks = build_tracks(d)
    plot_track_map(d2, tracks, out_png, out_pdf)

    print("[DONE]")
    print("  input:", in_csv)
    print("  output:", out_png)
    print("  output:", out_pdf)


if __name__ == "__main__":
    main()
