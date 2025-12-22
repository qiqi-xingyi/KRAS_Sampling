# --*-- conding:utf-8 --*--
# @time:12/19/25 01:26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rmsd.py
# tools/compute_backbone_rmsd.py
# ------------------------------------------------------------
# Compute backbone (C-alpha) RMSD between decoded samples and reference PDB fragment.
#
# Inputs (fixed layout):
#   <project_root>/pp_result/<fragment_id>/*/decoded.jsonl
#   <project_root>/dataset/benchmark_info.txt
#   <project_root>/RCSB_KRAS/<rcsb_pdb>
#
# Outputs:
#   For each decoded.jsonl:
#     <same_dir>/backbone_rmsd.jsonl   (one line per decoded sample, includes rmsd)
#   Global summaries:
#     <project_root>/pp_result/backbone_rmsd_summary.csv
#     <project_root>/pp_result/backbone_rmsd_min.csv
#
# Notes:
# - Uses optimal rigid alignment (Kabsch).
# - By default scales decoded lattice coordinates by 3.8 Å per step.
# ------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# User-tunable defaults
# -----------------------------
# Scaling for decoded coordinates:
#   "fixed": multiply decoded coords by FIXED_SCALE_ANGSTROM_PER_STEP
#   "auto":  scale decoded coords so mean consecutive distance matches reference mean
#   "none":  no scaling
SCALE_MODE = "fixed"
FIXED_SCALE_ANGSTROM_PER_STEP = 3.8

# Atom selection from reference PDB
REF_ATOM_NAME = "CA"  # backbone proxy

# If mismatch in length occurs, fail fast
STRICT_LENGTH_MATCH = True

# Numerical stability
EPS = 1e-12


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def load_benchmark_info(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"benchmark_info.txt not found: {path}")
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    required = {"pdb_id", "Residues", "Chain", "rcsb_pdb", "query_sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"benchmark_info.txt missing columns: {sorted(missing)}")
    return df


def parse_residue_range(r: str) -> Tuple[int, int]:
    # e.g., "5-16"
    s, e = r.strip().split("-")
    return int(s), int(e)


# -----------------------------
# PDB parsing (minimal, CA-only)
# -----------------------------
def extract_ca_coords_from_pdb(
    pdb_path: Path,
    chain_id: str,
    res_start: int,
    res_end: int,
    atom_name: str = "CA",
) -> np.ndarray:
    """
    Extract CA coordinates for residues [res_start, res_end] (inclusive) on a given chain.
    Uses PDB columns:
      - record: 1-6
      - atom name: 13-16
      - chain: 22
      - resseq: 23-26
      - x,y,z: 31-38, 39-46, 47-54
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"Reference PDB not found: {pdb_path}")

    coords: List[List[float]] = []
    seen_res = set()

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            if len(line) < 54:
                continue

            aname = line[12:16].strip()
            if aname != atom_name:
                continue

            ch = line[21].strip()
            if ch != chain_id.strip():
                continue

            # residue number (ignore insertion code for now)
            try:
                resseq = int(line[22:26].strip())
            except Exception:
                continue

            if resseq < res_start or resseq > res_end:
                continue

            # Avoid duplicates if altloc/multiple records; keep first occurrence
            # (If this becomes an issue, we can handle altLoc properly.)
            if resseq in seen_res:
                continue
            seen_res.add(resseq)

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except Exception:
                continue

            coords.append([x, y, z])

    coords = sorted(zip(sorted(seen_res), coords), key=lambda t: t[0])  # order by resseq
    out = np.array([c for _, c in coords], dtype=float)
    return out


# -----------------------------
# RMSD (Kabsch)
# -----------------------------
def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align P onto Q using Kabsch. Returns:
      P_aligned, R (3x3 rotation), t (translation)
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch P{P.shape} vs Q{Q.shape}")

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    P_rot = Pc @ R
    t = Q.mean(axis=0) - P_rot.mean(axis=0)
    P_aligned = P_rot + t
    return P_aligned, R, t


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    d2 = np.sum((P - Q) ** 2, axis=1)
    return float(np.sqrt(np.mean(d2)))


def mean_consecutive_distance(X: np.ndarray) -> float:
    if len(X) < 2:
        return 0.0
    diffs = X[1:] - X[:-1]
    return float(np.mean(np.linalg.norm(diffs, axis=1)))


def scale_decoded_coords(
    P: np.ndarray,
    Q_ref: np.ndarray,
    mode: str,
    fixed_scale: float,
) -> Tuple[np.ndarray, float]:
    """
    Returns scaled P and applied scale factor.
    """
    if mode == "none":
        return P, 1.0

    if mode == "fixed":
        return P * float(fixed_scale), float(fixed_scale)

    if mode == "auto":
        pd = mean_consecutive_distance(P)
        qd = mean_consecutive_distance(Q_ref)
        if pd <= EPS or qd <= EPS:
            return P, 1.0
        s = qd / pd
        return P * s, float(s)

    raise ValueError(f"Unknown SCALE_MODE: {mode}")


# -----------------------------
# JSONL decoding
# -----------------------------
def iter_decoded_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            yield i, obj


def get_main_positions(obj: dict) -> Optional[np.ndarray]:
    mp = obj.get("main_positions", None)
    if mp is None:
        return None
    try:
        arr = np.array(mp, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return None
        return arr
    except Exception:
        return None


# -----------------------------
# Main processing
# -----------------------------
def find_decoded_files(pp_root: Path, fragment_id: str) -> List[Path]:
    frag_dir = pp_root / fragment_id
    if not frag_dir.exists():
        return []
    return sorted(frag_dir.glob("*/decoded.jsonl"))


def compute_fragment_rmsd(
    fragment_id: str,
    decoded_files: List[Path],
    ref_coords: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute per-sample RMSD for all decoded.jsonl under a fragment.
    Writes backbone_rmsd.jsonl next to each decoded.jsonl.
    Returns:
      per_sample_df: rows with [fragment_id, decoded_path, idx, bitstring, sequence, rmsd]
      frag_min_info: dict with min rmsd and corresponding metadata
    """
    rows = []
    best = {
        "fragment_id": fragment_id,
        "min_rmsd": None,
        "best_bitstring": None,
        "best_sequence": None,
        "decoded_file": None,
        "line_index": None,
        "scale_factor": None,
    }

    for dec_path in decoded_files:
        out_path = dec_path.parent / "backbone_rmsd.jsonl"
        out_f = open(out_path, "w", encoding="utf-8")

        for idx, obj in iter_decoded_jsonl(dec_path):
            P = get_main_positions(obj)
            if P is None:
                continue

            if P.shape[0] != ref_coords.shape[0]:
                msg = f"[{fragment_id}] length mismatch decoded={P.shape[0]} ref={ref_coords.shape[0]} in {dec_path}"
                if STRICT_LENGTH_MATCH:
                    out_f.close()
                    raise ValueError(msg)
                else:
                    continue

            P_scaled, sf = scale_decoded_coords(P, ref_coords, SCALE_MODE, FIXED_SCALE_ANGSTROM_PER_STEP)
            P_aligned, _, _ = kabsch_align(P_scaled, ref_coords)
            r = rmsd(P_aligned, ref_coords)

            record = {
                "fragment_id": fragment_id,
                "decoded_file": str(dec_path),
                "line_index": idx,
                "sequence": obj.get("sequence", ""),
                "bitstring": obj.get("bitstring", ""),
                "rmsd": r,
                "scale_factor": sf,
            }
            out_f.write(json.dumps(record) + "\n")

            rows.append(record)

            if best["min_rmsd"] is None or r < float(best["min_rmsd"]):
                best.update(
                    {
                        "min_rmsd": r,
                        "best_bitstring": record["bitstring"],
                        "best_sequence": record["sequence"],
                        "decoded_file": record["decoded_file"],
                        "line_index": idx,
                        "scale_factor": sf,
                    }
                )

        out_f.close()

    df = pd.DataFrame(rows)
    return df, best


def summarize_fragment(df: pd.DataFrame, fragment_id: str) -> Dict[str, object]:
    if df.empty:
        return {
            "fragment_id": fragment_id,
            "n_samples": 0,
            "min_rmsd": np.nan,
            "mean_rmsd": np.nan,
            "median_rmsd": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }

    x = df["rmsd"].to_numpy(dtype=float)
    return {
        "fragment_id": fragment_id,
        "n_samples": int(len(x)),
        "min_rmsd": float(np.min(x)),
        "mean_rmsd": float(np.mean(x)),
        "median_rmsd": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main():
    root = project_root_from_tools_dir()
    pp_root = root / "pp_result"
    bench_path = root / "dataset" / "benchmark_info.txt"
    pdb_root = root / "RCSB_KRAS"

    if not pp_root.exists():
        raise FileNotFoundError(f"pp_result not found: {pp_root}")
    if not pdb_root.exists():
        raise FileNotFoundError(f"RCSB_KRAS not found: {pdb_root}")

    bench = load_benchmark_info(bench_path)

    all_samples = []
    frag_summaries = []
    frag_mins = []

    # Process each fragment listed in benchmark_info.txt
    for _, row in bench.iterrows():
        fragment_id = str(row["pdb_id"]).strip()
        residues = str(row["Residues"]).strip()
        chain = str(row["Chain"]).strip()
        ref_pdb = str(row["rcsb_pdb"]).strip()

        res_start, res_end = parse_residue_range(residues)
        pdb_path = pdb_root / ref_pdb

        decoded_files = find_decoded_files(pp_root, fragment_id)
        if not decoded_files:
            print(f"[SKIP] No decoded.jsonl found for fragment: {fragment_id}")
            continue

        ref_coords = extract_ca_coords_from_pdb(
            pdb_path=pdb_path,
            chain_id=chain,
            res_start=res_start,
            res_end=res_end,
            atom_name=REF_ATOM_NAME,
        )

        # Sanity: expected length by range
        expected_len = res_end - res_start + 1
        if ref_coords.shape[0] != expected_len:
            msg = (
                f"[{fragment_id}] reference CA count mismatch: got {ref_coords.shape[0]}, "
                f"expected {expected_len} (chain={chain}, residues={res_start}-{res_end}) in {pdb_path}"
            )
            if STRICT_LENGTH_MATCH:
                raise ValueError(msg)
            else:
                print("[WARN]", msg)

        df_samples, best = compute_fragment_rmsd(
            fragment_id=fragment_id,
            decoded_files=decoded_files,
            ref_coords=ref_coords,
        )

        # Add reference metadata columns to per-sample rows
        if not df_samples.empty:
            df_samples["ref_pdb"] = ref_pdb
            df_samples["chain"] = chain
            df_samples["res_start"] = res_start
            df_samples["res_end"] = res_end

        all_samples.append(df_samples)

        summ = summarize_fragment(df_samples, fragment_id)
        summ.update(
            {
                "ref_pdb": ref_pdb,
                "chain": chain,
                "res_start": res_start,
                "res_end": res_end,
                "scale_mode": SCALE_MODE,
                "fixed_scale": FIXED_SCALE_ANGSTROM_PER_STEP if SCALE_MODE == "fixed" else np.nan,
            }
        )
        frag_summaries.append(summ)

        best.update(
            {
                "ref_pdb": ref_pdb,
                "chain": chain,
                "res_start": res_start,
                "res_end": res_end,
                "scale_mode": SCALE_MODE,
            }
        )
        frag_mins.append(best)

        print(
            f"[OK] {fragment_id}: n={summ['n_samples']} min_rmsd={summ['min_rmsd']:.4f} "
            f"(decoded_files={len(decoded_files)})"
        )

    # Write global summary files
    out_summary = pp_root / "backbone_rmsd_summary.csv"
    out_min = pp_root / "backbone_rmsd_min.csv"
    out_all = pp_root / "backbone_rmsd_all_samples.csv"

    df_sum = pd.DataFrame(frag_summaries).sort_values("fragment_id")
    df_min = pd.DataFrame(frag_mins).sort_values("fragment_id")

    df_sum.to_csv(out_summary, index=False)
    df_min.to_csv(out_min, index=False)

    if all_samples:
        df_all = pd.concat(all_samples, ignore_index=True)
        df_all.to_csv(out_all, index=False)
    else:
        df_all = pd.DataFrame()

    print("\n[DONE] Wrote:")
    print(f"  - {out_summary}")
    print(f"  - {out_min}")
    print(f"  - {out_all}")
    print("\nPer-decoded outputs:")
    print("  - backbone_rmsd.jsonl next to each decoded.jsonl")

if __name__ == "__main__":
    main()
