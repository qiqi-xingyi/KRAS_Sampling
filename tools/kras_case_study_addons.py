# --*-- conding:utf-8 --*--
# @time:1/6/26 22:18
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_case_study_addons.py

# ------------------------------------------------------------
# Add-on script for KRAS case-study "missing pieces":
# 1) Basin mass shift tables (Δmass) and ranked summaries
# 2) Basin energy-term contrasts (WT vs G12C/G12D) for key basins
# 3) Representative CA-only PDB export for key basins + PyMOL loader
#
# Assumes you already ran the closed-loop analysis and have:
#   KRAS_sampling_results/analysis_closed_loop/
#     metrics.json
#     basin_occupancy.csv
#     basin_stats.csv
#     representatives.csv
# Optionally (best): a per-point merged table with basin_id column, e.g.:
#   merged_points.csv OR embedding_points.csv OR points_merged.csv
#
# It also uses pp_result/*/*/decoded.jsonl to export representative PDBs.
#
# Outputs:
#   KRAS_sampling_results/analysis_closed_loop/addons/
#     delta_basin_mass.csv
#     key_basins.json
#     basin_energy_contrast.csv
#     reps_manifest.csv
#     reps_pdbs/*.pdb
#     view_reps.pml
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# User-tunable parameters
# -----------------------------
TOPK_SHIFT_PER_MUT = 3          # choose top-K basins by |Δmass| per mutant
ENERGY_TERMS = [
    "E_total", "E_steric", "E_geom", "E_mj", "E_dihedral", "E_hydroph", "E_cbeta", "E_rama"
]
FEATURE_TERMS = [
    "backbone_rmsd",
    "feat_Rg",
    "feat_contact_density",
    "feat_clash_count",
    "feat_packing_rep_count",
    "feat_rama_allowed_ratio",
    "feat_end_to_end",
]
BOOTSTRAP_N = 400               # only used if per-point table exists
SEED = 0


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def find_analysis_dir(proj_root: Path) -> Path:
    d = proj_root / "KRAS_sampling_results" / "analysis_closed_loop"
    if not d.exists():
        raise FileNotFoundError(f"Cannot find analysis dir: {d}")
    return d


def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return pd.read_csv(p)


def find_optional_points_table(analysis_dir: Path) -> Optional[Path]:
    """
    Try to locate a per-point merged table with columns including:
      label, basin_id, weight, and energy terms.
    """
    candidates = [
        analysis_dir / "merged_points.csv",
        analysis_dir / "points_merged.csv",
        analysis_dir / "embedding_points.csv",
        analysis_dir / "all_points.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: search for any csv containing 'basin' and 'point'
    for c in analysis_dir.glob("*.csv"):
        if "basin" in c.name and "point" in c.name:
            return c
    return None


# -----------------------------
# Basin shift tables
# -----------------------------
def compute_delta_basin_mass(basin_occ: pd.DataFrame) -> pd.DataFrame:
    req = {"basin_id", "WT", "G12C", "G12D"}
    miss = req - set(basin_occ.columns)
    if miss:
        raise ValueError(f"basin_occupancy.csv missing columns: {miss}")

    df = basin_occ.copy()
    df["delta_G12C_minus_WT"] = df["G12C"] - df["WT"]
    df["delta_G12D_minus_WT"] = df["G12D"] - df["WT"]
    df["abs_delta_G12C"] = df["delta_G12C_minus_WT"].abs()
    df["abs_delta_G12D"] = df["delta_G12D_minus_WT"].abs()

    # ranks (1 = biggest shift)
    df["rank_abs_G12C"] = (-df["abs_delta_G12C"]).rank(method="dense").astype(int)
    df["rank_abs_G12D"] = (-df["abs_delta_G12D"]).rank(method="dense").astype(int)

    return df.sort_values(["abs_delta_G12D", "abs_delta_G12C"], ascending=False).reset_index(drop=True)


def pick_key_basins(delta_df: pd.DataFrame, topk: int) -> Dict[str, List[int]]:
    key_c = (
        delta_df.sort_values("abs_delta_G12C", ascending=False)
        .head(topk)["basin_id"].astype(int).tolist()
    )
    key_d = (
        delta_df.sort_values("abs_delta_G12D", ascending=False)
        .head(topk)["basin_id"].astype(int).tolist()
    )
    key_union = sorted(set(key_c) | set(key_d))
    return {"G12C_top": key_c, "G12D_top": key_d, "union": key_union}


# -----------------------------
# Weighted stats + bootstrap (if per-point table exists)
# -----------------------------
def wmean(x: np.ndarray, w: np.ndarray) -> float:
    s = float(np.sum(w))
    if s <= 0:
        return float("nan")
    return float(np.sum(w * x) / s)


def wvar(x: np.ndarray, w: np.ndarray) -> float:
    s = float(np.sum(w))
    if s <= 0:
        return float("nan")
    m = wmean(x, w)
    return float(np.sum(w * (x - m) ** 2) / s)


def weighted_bootstrap_mean(
    x: np.ndarray, w: np.ndarray, n: int, rng: np.random.Generator
) -> Tuple[float, float]:
    """
    Bootstrap CI for weighted mean by resampling indices with prob ~ w.
    Returns (lo, hi) 95% CI.
    """
    w = w.astype(float)
    w = w / (w.sum() + 1e-12)
    if len(x) == 0:
        return float("nan"), float("nan")

    stats = np.empty(n, dtype=float)
    idx = np.arange(len(x))
    for i in range(n):
        samp = rng.choice(idx, size=len(idx), replace=True, p=w)
        stats[i] = float(np.mean(x[samp]))  # resampled-unweighted mean over resampled set
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def compute_energy_contrast_from_points(
    points: pd.DataFrame,
    key_basins: List[int],
    labels: List[str],
    energy_terms: List[str],
    feature_terms: List[str],
    bootstrap_n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Requires points table to have at least: label, basin_id, weight.
    Preferably includes E_* and feature columns.
    """
    need = {"label", "basin_id", "weight"}
    miss = need - set(points.columns)
    if miss:
        raise ValueError(f"Points table missing required columns {miss}. Found: {points.columns.tolist()[:20]} ...")

    rng = np.random.default_rng(seed)
    rows = []

    terms = [t for t in (energy_terms + feature_terms) if t in points.columns]

    for bid in key_basins:
        for lab in labels:
            sub = points[(points["basin_id"] == bid) & (points["label"] == lab)]
            if sub.empty:
                continue
            w = sub["weight"].to_numpy(dtype=float)

            for term in terms:
                x = pd.to_numeric(sub[term], errors="coerce").to_numpy(dtype=float)
                m = wmean(x, w)
                sd = float(np.sqrt(wvar(x, w)))
                lo, hi = weighted_bootstrap_mean(x, w, bootstrap_n, rng)

                rows.append({
                    "basin_id": int(bid),
                    "label": lab,
                    "term": term,
                    "wmean": m,
                    "wstd": sd,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_points": int(len(sub)),
                })

    out = pd.DataFrame(rows)
    return out


def compute_energy_contrast_from_basin_stats(
    basin_stats: pd.DataFrame,
    key_basins: List[int],
    labels: List[str],
) -> pd.DataFrame:
    """
    Fallback mode: uses basin_stats.csv (means/stds already aggregated).
    Outputs long-form wmean/wstd without CI.
    """
    rows = []
    # basin_stats columns like: E_total_mean, E_total_std, backbone_rmsd_mean, feat_Rg_mean, ...
    for bid in key_basins:
        for lab in labels:
            sub = basin_stats[(basin_stats["label"] == lab) & (basin_stats["basin_id"] == bid)]
            if sub.empty:
                continue
            r = sub.iloc[0].to_dict()

            for k, v in r.items():
                if k.endswith("_mean"):
                    term = k[:-5]
                    stdk = term + "_std"
                    rows.append({
                        "basin_id": int(bid),
                        "label": lab,
                        "term": term,
                        "wmean": float(v) if pd.notna(v) else float("nan"),
                        "wstd": float(r.get(stdk, np.nan)) if pd.notna(r.get(stdk, np.nan)) else float("nan"),
                        "ci95_lo": np.nan,
                        "ci95_hi": np.nan,
                        "n_points": np.nan,
                    })
    return pd.DataFrame(rows)


def add_pairwise_deltas(long_df: pd.DataFrame, base_label: str = "WT") -> pd.DataFrame:
    """
    Convert (basin_id, label, term, wmean...) into deltas vs base_label.
    """
    pivot = long_df.pivot_table(
        index=["basin_id", "term"],
        columns="label",
        values="wmean",
        aggfunc="first"
    ).reset_index()

    for mut in ["G12C", "G12D"]:
        if mut in pivot.columns and base_label in pivot.columns:
            pivot[f"delta_{mut}_minus_{base_label}"] = pivot[mut] - pivot[base_label]
    return pivot


# -----------------------------
# Representative PDB export
# -----------------------------
AA1_TO_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

def locate_decoded_jsonl(proj_root: Path, fragment_id: str) -> Path:
    """
    pp_result/<fragment_id>/<inner>/decoded.jsonl
    where <inner> could be 4LPK_WT, 6OIM_G12C, 9C41_G12D, etc.
    We'll just find the first match.
    """
    base = proj_root / "pp_result" / fragment_id
    if not base.exists():
        raise FileNotFoundError(f"Missing fragment folder: {base}")

    hits = list(base.glob("*/*decoded.jsonl")) + list(base.glob("*/decoded.jsonl"))
    # normalize to exact decoded.jsonl
    hits = [h for h in hits if h.name == "decoded.jsonl"]
    if not hits:
        # try deeper
        hits = list(base.rglob("decoded.jsonl"))
    if not hits:
        raise FileNotFoundError(f"Cannot locate decoded.jsonl under: {base}")

    # Prefer pp_result/<fragment_id>/<inner>/decoded.jsonl
    hits = sorted(hits, key=lambda p: len(p.parts))
    return hits[0]


def load_selected_decoded_positions(decoded_path: Path, bitstrings_needed: set) -> Dict[str, Dict]:
    """
    Scan decoded.jsonl once and only keep entries whose bitstring is needed.
    Returns: bitstring -> parsed dict (sequence, main_positions, etc.)
    """
    out: Dict[str, Dict] = {}
    with open(decoded_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            bs = obj.get("bitstring", None)
            if bs in bitstrings_needed:
                out[bs] = obj
                if len(out) == len(bitstrings_needed):
                    break
    return out


def write_ca_pdb(
    out_pdb: Path,
    sequence: str,
    positions: List[List[float]],
    scale_factor: float,
    chain_id: str = "A",
):
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    n = min(len(sequence), len(positions))
    lines = []
    serial = 1
    for i in range(n):
        aa1 = sequence[i]
        resn = AA1_TO_AA3.get(aa1, "GLY")
        x, y, z = positions[i]
        x *= scale_factor
        y *= scale_factor
        z *= scale_factor
        lines.append(
            f"ATOM  {serial:5d}  CA  {resn:>3s} {chain_id}{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append(f"TER   {serial:5d}      CA  {AA1_TO_AA3.get(sequence[n-1], 'GLY'):>3s} {chain_id}{n:4d}")
    lines.append("END")

    out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_pymol_script(manifest: pd.DataFrame, out_pml: Path):
    """
    Creates a PyMOL script that loads all reps and groups by basin.
    """
    lines = [
        "reinitialize",
        "set ray_opaque_background, off",
        "bg_color white",
        "set cartoon_fancy_helices, on",
        "set cartoon_highlight_color, grey50",
    ]

    # Load all
    for _, r in manifest.iterrows():
        obj = r["pymol_obj"]
        pdb = r["pdb_path"]
        lines.append(f'load "{pdb}", {obj}')
        lines.append(f"show sticks, {obj}")
        lines.append(f"hide everything, {obj}")
        lines.append(f"show cartoon, {obj}")

    # Group by basin_id
    for bid in sorted(manifest["basin_id"].unique()):
        objs = manifest[manifest["basin_id"] == bid]["pymol_obj"].tolist()
        grp = f"basin_{int(bid)}"
        for o in objs:
            lines.append(f"group {grp}, {o}")

    out_pml.parent.mkdir(parents=True, exist_ok=True)
    out_pml.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main():
    proj_root = project_root_from_tools_dir()
    analysis_dir = find_analysis_dir(proj_root)

    metrics_path = analysis_dir / "metrics.json"
    basin_occ_path = analysis_dir / "basin_occupancy.csv"
    basin_stats_path = analysis_dir / "basin_stats.csv"
    reps_path = analysis_dir / "representatives.csv"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    analysis_set = metrics.get("analysis_set", {})
    # Expected: {"WT": "4LPK_WT_1", "G12C": "...", "G12D": "..."}
    labels = ["WT", "G12C", "G12D"]
    for lab in labels:
        if lab not in analysis_set:
            raise ValueError(f"metrics.json missing analysis_set[{lab}]")

    # Outputs
    out_dir = analysis_dir / "addons"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Δmass tables
    basin_occ = safe_read_csv(basin_occ_path)
    delta_df = compute_delta_basin_mass(basin_occ)
    delta_out = out_dir / "delta_basin_mass.csv"
    delta_df.to_csv(delta_out, index=False)

    key = pick_key_basins(delta_df, TOPK_SHIFT_PER_MUT)
    (out_dir / "key_basins.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    print("[OK] Δmass table:", delta_out)
    print("[OK] key basins:", key)

    key_basins = key["union"]

    # 2) Energy/feature contrasts
    basin_stats = safe_read_csv(basin_stats_path)
    reps = safe_read_csv(reps_path)

    points_path = find_optional_points_table(analysis_dir)
    long_df = None

    if points_path is not None:
        pts = pd.read_csv(points_path)
        # We require basin_id in per-point table; if missing, fallback.
        if "basin_id" in pts.columns and "label" in pts.columns and "weight" in pts.columns:
            print(f"[INFO] Using per-point table for bootstrap: {points_path.name}")
            long_df = compute_energy_contrast_from_points(
                points=pts,
                key_basins=key_basins,
                labels=labels,
                energy_terms=ENERGY_TERMS,
                feature_terms=FEATURE_TERMS,
                bootstrap_n=BOOTSTRAP_N,
                seed=SEED,
            )
        else:
            print(f"[WARN] Found points table but missing required cols; fallback to basin_stats: {points_path.name}")

    if long_df is None:
        print("[INFO] Fallback to basin_stats.csv (no bootstrap CI).")
        long_df = compute_energy_contrast_from_basin_stats(
            basin_stats=basin_stats,
            key_basins=key_basins,
            labels=labels,
        )

    contrast_pivot = add_pairwise_deltas(long_df, base_label="WT")
    contrast_out = out_dir / "basin_energy_contrast.csv"
    contrast_pivot.to_csv(contrast_out, index=False)
    print("[OK] basin energy/feature contrast:", contrast_out)

    # 3) Representative CA-only PDB export for key basins
    # Pick representatives: prefer within_label rows.
    reps_use = reps[reps["scope"].astype(str).str.contains("within_label", na=False)].copy()
    if reps_use.empty:
        reps_use = reps.copy()

    # We will export one representative per (label, basin_id) for key basins
    reps_keep = reps_use[reps_use["basin_id"].isin(key_basins)].copy()
    if reps_keep.empty:
        print("[WARN] No representatives matched key basins. Will export nothing.")
        return

    # Choose one row per (label, basin_id): max weight then arbitrary
    reps_keep["weight"] = pd.to_numeric(reps_keep.get("weight", 0.0), errors="coerce").fillna(0.0)
    reps_keep = reps_keep.sort_values(["label", "basin_id", "weight"], ascending=[True, True, False])
    reps_keep = reps_keep.drop_duplicates(subset=["label", "basin_id"], keep="first").reset_index(drop=True)

    # Locate decoded.jsonl per fragment id
    decoded_paths = {lab: locate_decoded_jsonl(proj_root, analysis_set[lab]) for lab in labels}
    for lab, p in decoded_paths.items():
        print(f"[INFO] decoded[{lab}] = {p}")

    # For each label, load only needed bitstrings
    manifests = []
    pdb_dir = out_dir / "reps_pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    for lab in labels:
        sub = reps_keep[reps_keep["label"] == lab]
        if sub.empty:
            continue
        needed = set(sub["bitstring"].astype(str).tolist())
        decoded = load_selected_decoded_positions(decoded_paths[lab], needed)

        for _, r in sub.iterrows():
            bs = str(r["bitstring"])
            bid = int(r["basin_id"])
            frag = str(r.get("fragment_id", analysis_set[lab]))
            scale = float(r.get("scale_factor", 3.8))

            obj = decoded.get(bs, None)
            if obj is None:
                print(f"[WARN] bitstring not found in decoded.jsonl: label={lab} basin={bid}")
                continue

            seq = str(obj.get("sequence", ""))
            pos = obj.get("main_positions", [])
            if not seq or not isinstance(pos, list) or len(pos) == 0:
                print(f"[WARN] missing sequence/positions in decoded entry: label={lab} basin={bid}")
                continue

            out_pdb = pdb_dir / f"{lab}_basin{bid}.pdb"
            write_ca_pdb(out_pdb, sequence=seq, positions=pos, scale_factor=scale, chain_id="A")

            manifests.append({
                "label": lab,
                "basin_id": bid,
                "fragment_id": frag,
                "bitstring": bs,
                "scale_factor": scale,
                "pdb_path": str(out_pdb),
                "pymol_obj": f"{lab}_b{bid}",
            })

    manifest_df = pd.DataFrame(manifests)
    manifest_out = out_dir / "reps_manifest.csv"
    manifest_df.to_csv(manifest_out, index=False)
    print("[OK] reps manifest:", manifest_out)

    # PyMOL script
    pml_out = out_dir / "view_reps.pml"
    if not manifest_df.empty:
        build_pymol_script(manifest_df, pml_out)
        print("[OK] PyMOL script:", pml_out)
    else:
        print("[WARN] No PDBs were exported (manifest empty).")

    print("\n[DONE] Add-ons generated under:", out_dir)


if __name__ == "__main__":
    main()
