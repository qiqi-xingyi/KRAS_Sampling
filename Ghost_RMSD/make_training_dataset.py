# --*-- conding:utf-8 --*--
# @time:11/1/25 02:25
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:make_training_dataset.py

# Build unified training dataset:
# - Merge energies (JSONL) + features (JSONL) by (sequence, bitstring)
# - Compute CA-RMSD of candidate main_positions vs reference PDB fragment
# - Assign graded relevance labels by thresholds
# - Write one JSONL corpus: training_data/all_examples.jsonl

import os
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# -------------------- Paths --------------------
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "training_data"
ENERGIES_FILE = OUT_DIR / "all_energies.jsonl"
FEATURES_FILE = OUT_DIR / "all_features.jsonl"
INDEX_FILE = ROOT / "benchmark_info.txt"
PDB_ROOT = ROOT / "Pdbbind"  # expects Pdbbind/<pdb_id>/<pdb_id>_protein.pdb

OUTPUT_FILE = OUT_DIR / "all_examples.jsonl"

# -------------------- Grading thresholds --------------------
# rel = 3 if rmsd <= 2.5
# rel = 2 if 2.5 < rmsd <= 4.5
# rel = 1 if 4.5 < rmsd <= 6.5
# rel = 0 if rmsd > 6.5
THRESHOLDS = (2.5, 4.5, 6.5)

# -------------------- Utilities --------------------

def parse_residue_range(rr: str) -> Tuple[int, int]:
    # "47-59" -> (47, 59)
    a, b = rr.split("-")
    return int(a), int(b)

def load_benchmark_index(idx_path: Path) -> Dict[str, Dict]:
    """Map sequence -> row (containing pdb_id, Residues, etc.). If duplicates exist, later rows overwrite."""
    seq2row: Dict[str, Dict] = {}
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")
    with idx_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row = dict(zip(header, parts))
            seq = row["Residue_sequence"].strip()
            seq2row[seq] = row
    return seq2row

def load_ca_coords_from_pdb_fragment(pdb_path: Path, res_start: int, res_end: int) -> np.ndarray:
    """
    Return (L,3) CA coordinates for the first chain that has full coverage [res_start..res_end].
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")

    chains: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name != "CA":
                continue
            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq < res_start or resseq > res_end:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            chains.setdefault(chain, []).append((resseq, np.array([x, y, z], dtype=np.float64)))

    for ch, lst in chains.items():
        lst.sort(key=lambda t: t[0])
        seq_nums = [r for r, _ in lst]
        expected = list(range(res_start, res_end + 1))
        if seq_nums == expected:
            coords = np.stack([c for _, c in lst], axis=0)
            return coords

    raise ValueError(f"No chain in {pdb_path.name} has full CA coverage for {res_start}-{res_end}")

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kabsch alignment RMSD between two (L,3) arrays P (candidate) and Q (reference).
    """
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    U = V @ D @ Wt
    P_rot = Pc @ U
    diff2 = ((P_rot - Qc) ** 2).sum()
    return float(np.sqrt(diff2 / P.shape[0]))

def rel_from_rmsd(r: float, th=THRESHOLDS) -> int:
    t1, t2, t3 = th
    if r <= t1: return 3
    if r <= t2: return 2
    if r <= t3: return 1
    return 0

def load_features_map(features_path: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Build a lookup dict by (sequence, bitstring) -> feature dict.
    This fits in memory for ~275k rows comfortably.
    """
    fmap: Dict[Tuple[str, str], Dict] = {}
    if not features_path.exists():
        return fmap
    with features_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seq = str(obj.get("sequence", ""))  # may be missing; try to keep if present
            bit = str(obj.get("bitstring", ""))
            if not bit:
                continue
            key = (seq, bit)
            fmap[key] = obj
    return fmap

def pick_energy_scalars(obj: Dict) -> Dict:
    """
    Extract scalar energy/geometry fields from an energy record, excluding heavy arrays.
    """
    exclude = {"main_positions", "side_positions", "main_vectors", "side_vectors"}
    out = {}
    for k, v in obj.items():
        if k in exclude:
            continue
        # keep only scalars/basic types
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out

# -------------------- Main --------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load index: sequence -> (pdb_id, residues)
    seq_index = load_benchmark_index(INDEX_FILE)

    # 2) Load features map keyed by (sequence, bitstring)
    features_map = load_features_map(FEATURES_FILE)

    # 3) Cache for reference CA coords: (pdb_id, residues_str) -> ndarray
    ref_cache: Dict[Tuple[str, str], np.ndarray] = {}

    # 4) Iterate energies and build unified examples
    n_in = 0
    n_out = 0
    n_skip = 0

    with ENERGIES_FILE.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            n_in += 1
            try:
                obj = json.loads(s)
            except Exception:
                n_skip += 1
                continue

            seq = str(obj.get("sequence", ""))
            bit = str(obj.get("bitstring", ""))
            pos = obj.get("main_positions", None)

            if not seq or not bit or pos is None:
                # cannot compute RMSD without sequence/bitstring/positions
                n_skip += 1
                continue

            # lookup pdb_id and residues by sequence
            idx_row = seq_index.get(seq)
            if not idx_row:
                # sequence not found in index
                n_skip += 1
                continue
            pdb_id = idx_row["pdb_id"]
            residues_str = idx_row["Residues"]
            res_start, res_end = parse_residue_range(residues_str)

            # load cached reference CA coords
            cache_key = (pdb_id, residues_str)
            if cache_key not in ref_cache:
                pdb_path = PDB_ROOT / pdb_id / f"{pdb_id}_protein.pdb"
                ref_cache[cache_key] = load_ca_coords_from_pdb_fragment(pdb_path, res_start, res_end)
            ref = ref_cache[cache_key]

            # candidate coords
            P = np.array(pos, dtype=np.float64)
            if P.shape != ref.shape:
                # length mismatch; skip
                n_skip += 1
                continue

            # compute RMSD and rel
            rmsd = kabsch_rmsd(P, ref)
            rel = rel_from_rmsd(rmsd)

            # merge energies (scalars) + features
            rec = pick_energy_scalars(obj)
            # ensure core identifiers
            rec["protein"] = pdb_id
            rec["residues"] = residues_str
            rec["sequence"] = seq
            rec["bitstring"] = bit
            rec["rmsd"] = rmsd
            rec["rel"] = rel  # graded label

            # attach features (if available)
            fkey = (seq, bit)
            fdict = features_map.get(fkey)
            if fdict:
                # merge feature fields, avoiding overwriting core keys
                for k, v in fdict.items():
                    if k in ("sequence", "bitstring"):
                        continue
                    rec[k] = v

            fout.write(json.dumps(rec) + "\n")
            n_out += 1

    print(f"[DONE] Input energy lines: {n_in}")
    print(f"[DONE] Written examples  : {n_out}")
    print(f"[DONE] Skipped (errors)  : {n_skip}")
    print(f"[OUT ] {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
