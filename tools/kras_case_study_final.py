# --*-- conding:utf-8 --*--
# @time:1/6/26 22:22
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:kras_case_study_final.py

# ------------------------------------------------------------
# Produces "story-ready" tables for key basins:
#  - key_basin_story.csv: Δmass + Δenergy + Δfeatures (vs WT)
#  - key_basin_story_ranked.csv: ranked by impact
#  - rep_ca_rmsd.csv: CA-RMSD between representative structures (WT vs mutants) within same basin
#
# Inputs expected under:
#   KRAS_sampling_results/analysis_closed_loop/
#     basin_occupancy.csv
#     basin_stats.csv
#     representatives.csv
#     metrics.json
# And decoded.jsonl under:
#   pp_result/<fragment_id>/*/decoded.jsonl
# ------------------------------------------------------------

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

AA1_TO_AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

ENERGY_TERMS = ["E_total", "E_steric", "E_geom", "E_mj", "E_dihedral", "E_hydroph", "E_cbeta", "E_rama"]
FEATURE_TERMS = ["backbone_rmsd", "feat_clash_count", "feat_Rg", "feat_contact_density", "feat_end_to_end"]

# Choose key basins
KEY_BASINS = [1, 2, 5, 6]  # from your union

def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent

def find_analysis_dir(proj_root: Path) -> Path:
    d = proj_root / "KRAS_sampling_results" / "analysis_closed_loop"
    if not d.exists():
        raise FileNotFoundError(f"Cannot find analysis dir: {d}")
    return d

def locate_decoded_jsonl(proj_root: Path, fragment_id: str) -> Path:
    base = proj_root / "pp_result" / fragment_id
    if not base.exists():
        raise FileNotFoundError(f"Missing fragment folder: {base}")
    hits = list(base.rglob("decoded.jsonl"))
    if not hits:
        raise FileNotFoundError(f"Cannot locate decoded.jsonl under: {base}")
    hits = sorted(hits, key=lambda p: len(p.parts))
    return hits[0]

def load_decoded_by_bitstring(decoded_path: Path, bitstrings: set) -> Dict[str, Dict]:
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
            if bs in bitstrings:
                out[bs] = obj
                if len(out) == len(bitstrings):
                    break
    return out

def ca_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kabsch RMSD, no translation assumption (we center both).
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        return float("nan")
    P0 = P - P.mean(axis=0, keepdims=True)
    Q0 = Q - Q.mean(axis=0, keepdims=True)

    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_rot = P0 @ U
    diff = P_rot - Q0
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

def main():
    proj_root = project_root_from_tools_dir()
    analysis_dir = find_analysis_dir(proj_root)

    basin_occ = pd.read_csv(analysis_dir / "basin_occupancy.csv")
    basin_stats = pd.read_csv(analysis_dir / "basin_stats.csv")
    reps = pd.read_csv(analysis_dir / "representatives.csv")
    metrics = json.loads((analysis_dir / "metrics.json").read_text(encoding="utf-8"))
    aset = metrics["analysis_set"]
    labels = ["WT", "G12C", "G12D"]

    out_dir = analysis_dir / "addons"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Build story table (Δmass + Δterms)
    # -------------------------
    occ = basin_occ.set_index("basin_id")[["WT", "G12C", "G12D"]].copy()
    occ["delta_mass_G12C_minus_WT"] = occ["G12C"] - occ["WT"]
    occ["delta_mass_G12D_minus_WT"] = occ["G12D"] - occ["WT"]

    # basin_stats has rows: label, basin_id, mass, <term>_mean, <term>_std ...
    story_rows = []
    for bid in KEY_BASINS:
        row = {"basin_id": int(bid)}
        if bid in occ.index:
            row.update({
                "mass_WT": float(occ.loc[bid, "WT"]),
                "mass_G12C": float(occ.loc[bid, "G12C"]),
                "mass_G12D": float(occ.loc[bid, "G12D"]),
                "delta_mass_G12C_minus_WT": float(occ.loc[bid, "delta_mass_G12C_minus_WT"]),
                "delta_mass_G12D_minus_WT": float(occ.loc[bid, "delta_mass_G12D_minus_WT"]),
            })

        # Pull means from basin_stats
        def get_mean(lab: str, term: str) -> float:
            sub = basin_stats[(basin_stats["label"] == lab) & (basin_stats["basin_id"] == bid)]
            if sub.empty:
                return float("nan")
            col = f"{term}_mean"
            return float(sub.iloc[0][col]) if col in sub.columns else float("nan")

        for term in ENERGY_TERMS + FEATURE_TERMS:
            wt = get_mean("WT", term)
            c = get_mean("G12C", term)
            d = get_mean("G12D", term)
            row[f"{term}_WT"] = wt
            row[f"{term}_G12C"] = c
            row[f"{term}_G12D"] = d
            row[f"delta_{term}_G12C_minus_WT"] = c - wt
            row[f"delta_{term}_G12D_minus_WT"] = d - wt

        story_rows.append(row)

    story = pd.DataFrame(story_rows)

    # A simple "impact score" for ranking narrative priority:
    # combine |Δmass| and |ΔE_hydroph| for G12D vs WT (you can change)
    story["impact_G12D"] = (
        story["delta_mass_G12D_minus_WT"].abs()
        + 0.05 * story["delta_E_hydroph_G12D_minus_WT"].abs()
    )
    story_ranked = story.sort_values("impact_G12D", ascending=False)

    story_out = out_dir / "key_basin_story.csv"
    story_ranked_out = out_dir / "key_basin_story_ranked.csv"
    story.to_csv(story_out, index=False)
    story_ranked.to_csv(story_ranked_out, index=False)

    print("[OK] story table:", story_out)
    print("[OK] ranked story:", story_ranked_out)

    # -------------------------
    # 2) Representative CA-RMSD within same basin (WT vs mutants)
    # -------------------------
    reps_use = reps[reps["scope"].astype(str).str.contains("within_label", na=False)].copy()
    if reps_use.empty:
        reps_use = reps.copy()

    reps_keep = reps_use[reps_use["basin_id"].isin(KEY_BASINS)].copy()
    reps_keep = reps_keep.sort_values(["label", "basin_id", "weight"], ascending=[True, True, False])
    reps_keep = reps_keep.drop_duplicates(subset=["label", "basin_id"], keep="first").reset_index(drop=True)

    decoded_paths = {lab: locate_decoded_jsonl(proj_root, aset[lab]) for lab in labels}
    decoded_cache: Dict[str, Dict[str, Dict]] = {}

    # load needed bitstrings per label
    for lab in labels:
        sub = reps_keep[reps_keep["label"] == lab]
        bss = set(sub["bitstring"].astype(str).tolist())
        decoded_cache[lab] = load_decoded_by_bitstring(decoded_paths[lab], bss)

    # compute CA-RMSD for each basin between WT and mutants
    rmsd_rows = []
    for bid in KEY_BASINS:
        wt_row = reps_keep[(reps_keep["label"] == "WT") & (reps_keep["basin_id"] == bid)]
        if wt_row.empty:
            continue
        wt_bs = str(wt_row.iloc[0]["bitstring"])
        wt_scale = float(wt_row.iloc[0].get("scale_factor", 3.8))
        wt_obj = decoded_cache["WT"].get(wt_bs, None)
        if wt_obj is None:
            continue
        wt_pos = np.array(wt_obj.get("main_positions", []), dtype=float) * wt_scale

        for mut in ["G12C", "G12D"]:
            mut_row = reps_keep[(reps_keep["label"] == mut) & (reps_keep["basin_id"] == bid)]
            if mut_row.empty:
                continue
            mut_bs = str(mut_row.iloc[0]["bitstring"])
            mut_scale = float(mut_row.iloc[0].get("scale_factor", 3.8))
            mut_obj = decoded_cache[mut].get(mut_bs, None)
            if mut_obj is None:
                continue
            mut_pos = np.array(mut_obj.get("main_positions", []), dtype=float) * mut_scale

            n = min(len(wt_pos), len(mut_pos))
            if n < 3:
                r = float("nan")
            else:
                r = ca_rmsd(wt_pos[:n], mut_pos[:n])

            rmsd_rows.append({
                "basin_id": int(bid),
                "pair": f"{mut}-vs-WT",
                "WT_bitstring": wt_bs,
                f"{mut}_bitstring": mut_bs,
                "ca_rmsd": r,
            })

    rmsd_df = pd.DataFrame(rmsd_rows)
    rmsd_out = out_dir / "rep_ca_rmsd.csv"
    rmsd_df.to_csv(rmsd_out, index=False)
    print("[OK] representative CA-RMSD:", rmsd_out)

    print("\n[DONE] story + RMSD ready in:", out_dir)


if __name__ == "__main__":
    main()
