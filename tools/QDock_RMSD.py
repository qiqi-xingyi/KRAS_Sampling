# --*-- conding:utf-8 --*--
# @time:1/9/26 20:02
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:QDock_RMSD.py

# ------------------------------------------------------------
# Compute backbone (C-alpha) RMSD between decoded samples and reference PDB fragment
# for QDockBank (PDBbind-style references).
#
# Inputs (fixed layout):
#   <project_root>/Q_Dock_pp_result/**/decoded.jsonl
#   <project_root>/dataset/QDockbank_info.txt
#   <project_root>/dataset/Pdbbind/<pdb_id>/<pdb_id>_protein.pdb  (preferred)
#                                     or <pdb_id>_pocket.pdb
#
# Outputs:
#   For each decoded.jsonl:
#     <same_dir>/backbone_rmsd.jsonl
#   Global summaries:
#     <project_root>/Q_Dock_pp_result/backbone_rmsd_summary.csv
#     <project_root>/Q_Dock_pp_result/backbone_rmsd_min.csv
#     <project_root>/Q_Dock_pp_result/backbone_rmsd_all_samples.csv
#
# Notes:
# - Optimal rigid alignment (Kabsch).
# - Default scales decoded lattice coordinates by 3.8 Å per step.
# - Reference CA coordinates are extracted by residue index range and best-matching chain.
# ------------------------------------------------------------

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

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

# Paths (fixed)
PP_RESULT_DIRNAME = "Q_Dock_pp_result"
QDOCK_INFO_PATH = Path("dataset") / "QDockbank_info.txt"
PDBBIND_ROOT = Path("dataset") / "Pdbbind"


# -----------------------------
# Path helpers
# -----------------------------
def project_root_from_tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent


# -----------------------------
# Load QDockbank_info.txt
# -----------------------------
def load_qdock_info(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"QDockbank_info.txt not found: {path}")

    # Accept both TSV and whitespace-separated
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")

    required = {"pdb_id", "Residues"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"QDockbank_info.txt missing columns: {sorted(missing)}")

    # Normalize
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
    df["Residues"] = df["Residues"].astype(str).str.strip()

    if "Sequence_length" in df.columns:
        df["Sequence_length"] = pd.to_numeric(df["Sequence_length"], errors="coerce")

    if "Residue_sequence" in df.columns:
        df["Residue_sequence"] = df["Residue_sequence"].astype(str).str.strip()

    return df


def parse_residue_range(r: str) -> Tuple[int, int]:
    # e.g., "47-59"
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(r))
    if not m:
        raise ValueError(f"Bad residue range format: {r!r} (expected like 47-59)")
    return int(m.group(1)), int(m.group(2))


# -----------------------------
# Reference PDB selection
# -----------------------------
def find_reference_pdb(pdbbind_root: Path, pdb_id: str) -> Path:
    """
    Prefer <pdb_id>_protein.pdb; fallback to <pdb_id>_pocket.pdb.
    """
    case_dir = pdbbind_root / pdb_id
    if not case_dir.exists():
        raise FileNotFoundError(f"PDBbind case folder not found: {case_dir}")

    p_protein = case_dir / f"{pdb_id}_protein.pdb"
    p_pocket = case_dir / f"{pdb_id}_pocket.pdb"

    if p_protein.exists():
        return p_protein
    if p_pocket.exists():
        return p_pocket

    raise FileNotFoundError(f"No reference PDB found for {pdb_id} under {case_dir}")


# -----------------------------
# PDB parsing (CA-only, best chain)
# -----------------------------
def _iter_pdb_ca_records(pdb_path: Path, atom_name: str) -> Iterable[Tuple[str, int, np.ndarray]]:
    """
    Yield (chain_id, resseq, xyz) for CA records.
    Ignores insertion code and altLoc complexities (keeps first resseq per chain).
    """
    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                continue

            aname = line[12:16].strip()
            if aname != atom_name:
                continue

            chain_id = line[21].strip() or "?"
            try:
                resseq = int(line[22:26].strip())
            except Exception:
                continue

            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except Exception:
                continue

            yield chain_id, resseq, np.array([x, y, z], dtype=float)


def extract_ca_coords_best_chain(
    pdb_path: Path,
    res_start: int,
    res_end: int,
    atom_name: str = "CA",
    expected_len: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """
    Extract CA coordinates for residues [res_start, res_end] (inclusive).
    Since QDockbank_info.txt doesn't give chain, we choose the chain that best matches:
      - Prefer chain with exact expected_len coverage in the range.
      - Else choose chain with maximal coverage.
    Returns: (coords, chosen_chain)
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"Reference PDB not found: {pdb_path}")

    # chain -> {resseq: xyz}
    chain_map: Dict[str, Dict[int, np.ndarray]] = {}
    for ch, resseq, xyz in _iter_pdb_ca_records(pdb_path, atom_name=atom_name):
        d = chain_map.setdefault(ch, {})
        # keep first occurrence
        if resseq not in d:
            d[resseq] = xyz

    if not chain_map:
        raise ValueError(f"No {atom_name} atoms found in reference PDB: {pdb_path}")

    # Evaluate each chain
    best_chain = None
    best_score = (-1, False)  # (coverage, exact_match)
    best_coords = None

    for ch, d in chain_map.items():
        coords = []
        missing = 0
        for r in range(res_start, res_end + 1):
            if r in d:
                coords.append(d[r])
            else:
                missing += 1

        coverage = len(coords)
        exact = False
        if expected_len is not None and coverage == expected_len and missing == 0:
            exact = True

        score = (coverage, exact)
        # Choose exact match first; if tie, choose higher coverage; if still tie, lexicographically smaller chain id
        if (exact and not best_score[1]) or (score > best_score) or (
            score == best_score and best_chain is not None and ch < best_chain
        ):
            best_chain = ch
            best_score = score
            best_coords = np.array(coords, dtype=float) if coords else np.zeros((0, 3), dtype=float)

    if best_chain is None or best_coords is None:
        raise RuntimeError(f"Failed to choose chain for {pdb_path}")

    return best_coords, best_chain


# -----------------------------
# RMSD (Kabsch)
# -----------------------------
def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch P{P.shape} vs Q{Q.shape}")

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

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
# Locate decoded.jsonl files under Q_Dock_pp_result
# -----------------------------
def infer_pdb_id_from_decoded_path(pp_root: Path, decoded_path: Path) -> Optional[str]:
    """
    Use the first-level folder name under Q_Dock_pp_result:
      Q_Dock_pp_result/<NAME>/.../decoded.jsonl
    Take token before '_' and require 4-char alnum.
    """
    try:
        rel0 = decoded_path.relative_to(pp_root).parts[0]
    except Exception:
        return None

    token = rel0.split("_")[0].strip().lower()
    if len(token) == 4 and token.isalnum():
        return token
    return None


def index_decoded_files(pp_root: Path) -> Dict[str, List[Path]]:
    buckets: Dict[str, List[Path]] = {}
    for p in pp_root.rglob("decoded.jsonl"):
        pid = infer_pdb_id_from_decoded_path(pp_root, p)
        if pid is None:
            continue
        buckets.setdefault(pid, []).append(p)
    for k in buckets:
        buckets[k] = sorted(buckets[k])
    return buckets


# -----------------------------
# Main processing
# -----------------------------
def compute_case_rmsd(
    pdb_id: str,
    decoded_files: List[Path],
    ref_coords: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    best = {
        "pdb_id": pdb_id,
        "min_rmsd": None,
        "best_bitstring": None,
        "best_sequence": None,
        "decoded_file": None,
        "line_index": None,
        "scale_factor": None,
    }

    for dec_path in decoded_files:
        out_path = dec_path.parent / "backbone_rmsd.jsonl"
        with open(out_path, "w", encoding="utf-8") as out_f:
            for idx, obj in iter_decoded_jsonl(dec_path):
                P = get_main_positions(obj)
                if P is None:
                    continue

                if P.shape[0] != ref_coords.shape[0]:
                    msg = (
                        f"[{pdb_id}] length mismatch decoded={P.shape[0]} "
                        f"ref={ref_coords.shape[0]} in {dec_path}"
                    )
                    if STRICT_LENGTH_MATCH:
                        raise ValueError(msg)
                    else:
                        continue

                P_scaled, sf = scale_decoded_coords(P, ref_coords, SCALE_MODE, FIXED_SCALE_ANGSTROM_PER_STEP)
                P_aligned, _, _ = kabsch_align(P_scaled, ref_coords)
                r = rmsd(P_aligned, ref_coords)

                record = {
                    "pdb_id": pdb_id,
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

    df = pd.DataFrame(rows)
    return df, best


def summarize_case(df: pd.DataFrame, pdb_id: str) -> Dict[str, object]:
    if df.empty:
        return {
            "pdb_id": pdb_id,
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
        "pdb_id": pdb_id,
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

    pp_root = root / PP_RESULT_DIRNAME
    info_path = root / QDOCK_INFO_PATH
    pdbbind_root = root / PDBBIND_ROOT

    if not pp_root.exists():
        raise FileNotFoundError(f"{PP_RESULT_DIRNAME} not found: {pp_root}")
    if not pdbbind_root.exists():
        raise FileNotFoundError(f"Pdbbind root not found: {pdbbind_root}")

    info = load_qdock_info(info_path)

    # index decoded files once
    buckets = index_decoded_files(pp_root)
    print(f"[INDEX] decoded.jsonl indexed for {len(buckets)} pdb_ids under {pp_root}")

    all_samples = []
    case_summaries = []
    case_mins = []

    for _, row in info.iterrows():
        pdb_id = str(row["pdb_id"]).strip().lower()
        residues = str(row["Residues"]).strip()
        res_start, res_end = parse_residue_range(residues)

        expected_len = (res_end - res_start + 1)
        if "Sequence_length" in row and pd.notna(row["Sequence_length"]):
            try:
                expected_len = int(row["Sequence_length"])
            except Exception:
                pass

        decoded_files = buckets.get(pdb_id, [])
        if not decoded_files:
            print(f"[SKIP] {pdb_id}: no decoded.jsonl found under {PP_RESULT_DIRNAME}")
            continue

        ref_pdb = find_reference_pdb(pdbbind_root, pdb_id)
        ref_coords, chosen_chain = extract_ca_coords_best_chain(
            pdb_path=ref_pdb,
            res_start=res_start,
            res_end=res_end,
            atom_name=REF_ATOM_NAME,
            expected_len=expected_len,
        )

        # sanity
        if ref_coords.shape[0] != expected_len:
            msg = (
                f"[{pdb_id}] reference CA count mismatch: got {ref_coords.shape[0]}, "
                f"expected {expected_len} (residues={res_start}-{res_end}, chain_chosen={chosen_chain}) "
                f"in {ref_pdb}"
            )
            if STRICT_LENGTH_MATCH:
                raise ValueError(msg)
            else:
                print("[WARN]", msg)

        df_samples, best = compute_case_rmsd(
            pdb_id=pdb_id,
            decoded_files=decoded_files,
            ref_coords=ref_coords,
        )

        if not df_samples.empty:
            df_samples["ref_pdb"] = str(ref_pdb)
            df_samples["ref_chain"] = chosen_chain
            df_samples["res_start"] = res_start
            df_samples["res_end"] = res_end

        all_samples.append(df_samples)

        summ = summarize_case(df_samples, pdb_id)
        summ.update(
            {
                "ref_pdb": str(ref_pdb),
                "ref_chain": chosen_chain,
                "res_start": res_start,
                "res_end": res_end,
                "expected_len": expected_len,
                "scale_mode": SCALE_MODE,
                "fixed_scale": FIXED_SCALE_ANGSTROM_PER_STEP if SCALE_MODE == "fixed" else np.nan,
                "n_decoded_files": len(decoded_files),
            }
        )
        case_summaries.append(summ)

        best.update(
            {
                "ref_pdb": str(ref_pdb),
                "ref_chain": chosen_chain,
                "res_start": res_start,
                "res_end": res_end,
                "expected_len": expected_len,
                "scale_mode": SCALE_MODE,
                "n_decoded_files": len(decoded_files),
            }
        )
        case_mins.append(best)

        if summ["n_samples"] > 0:
            print(
                f"[OK] {pdb_id}: files={len(decoded_files)} n={summ['n_samples']} "
                f"min_rmsd={summ['min_rmsd']:.4f} (chain={chosen_chain})"
            )
        else:
            print(f"[OK] {pdb_id}: files={len(decoded_files)} but no valid decoded samples")

    # Write global summary files
    out_summary = pp_root / "backbone_rmsd_summary.csv"
    out_min = pp_root / "backbone_rmsd_min.csv"
    out_all = pp_root / "backbone_rmsd_all_samples.csv"

    df_sum = pd.DataFrame(case_summaries).sort_values("pdb_id") if case_summaries else pd.DataFrame()
    df_min = pd.DataFrame(case_mins).sort_values("pdb_id") if case_mins else pd.DataFrame()

    df_sum.to_csv(out_summary, index=False)
    df_min.to_csv(out_min, index=False)

    if all_samples and any(not x.empty for x in all_samples):
        df_all = pd.concat([x for x in all_samples if not x.empty], ignore_index=True)
        df_all.to_csv(out_all, index=False)
    else:
        df_all = pd.DataFrame()
        df_all.to_csv(out_all, index=False)

    print("\n[DONE] Wrote:")
    print(f"  - {out_summary}")
    print(f"  - {out_min}")
    print(f"  - {out_all}")
    print("\nPer-decoded outputs:")
    print("  - backbone_rmsd.jsonl next to each decoded.jsonl")


if __name__ == "__main__":
    main()
