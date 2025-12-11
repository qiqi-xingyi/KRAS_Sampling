# --*-- conding:utf-8 --*--
# @time:11/1/25 01:15
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:gen_data.py

import os
import csv
import numpy as np
from pathlib import Path

# ==============================================================
# Configuration
# ==============================================================

INDEX_FILE = "benchmark_info.txt"     # index file in the same directory
OUTPUT_DIR = "sampling_data"  # output directory
PER_GROUP = 5000                      # number of samples per protein
SUBTRACT_BITS = 5                     # bit length = qubits - SUBTRACT_BITS
SHOTS = 2000                          # prob = count / shots
BETA = 0.0
SEED = 42
BACKEND = "ibm"
IBM_BACKEND = "ibm_strasbourg"
LABEL_PREFIX = "qsad_ibm"

# ==============================================================


def read_info(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            rows.append(dict(zip(header, parts)))
    return rows


def gen_bitstrings(bit_len: int, n: int, rng: np.random.Generator):
    """Generate n random bitstrings of given length (no dedup)."""
    arr = rng.integers(0, 2, size=(n, bit_len), dtype=np.uint8)
    packed = np.packbits(arr, axis=1, bitorder='big')
    result = []
    for row in packed:
        s = ''.join(format(b, '08b') for b in row)[:bit_len]
        result.append(s)
    return result


def main():
    root = Path(__file__).resolve().parent
    index_path = root / INDEX_FILE
    outdir = root / OUTPUT_DIR
    outdir.mkdir(exist_ok=True)

    rows = read_info(index_path)
    print(f"[INFO] Loaded {len(rows)} benchmark entries from {index_path.name}")

    header = [
        "L","n_qubits","shots","beta","seed","label","backend",
        "ibm_backend","circuit_hash","protein","sequence",
        "group_id","bitstring","count","prob"
    ]

    base_rng = np.random.default_rng(SEED)

    for gi, r in enumerate(rows):
        pdb_id = r["pdb_id"].strip()
        seq = r["Residue_sequence"].strip()
        try:
            L = int(r["Sequence_length"])
        except Exception:
            L = len(seq)

        n_qubits_raw = int(r["Number_of_qubits"])
        bit_len = n_qubits_raw - SUBTRACT_BITS
        if bit_len <= 0:
            print(f"[WARN] Skip {pdb_id}: invalid bit length {bit_len}")
            continue

        rng = np.random.default_rng(SEED + gi * 9973)
        bitstrings = gen_bitstrings(bit_len, PER_GROUP, rng)

        prob = 1.0 / SHOTS
        count = 1
        label = f"{LABEL_PREFIX}_{pdb_id}_g{gi}"
        circuit_hash = ""

        out_path = outdir / f"samples_{pdb_id}_rand.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(header)
            for s in bitstrings:
                writer.writerow([
                    L, bit_len, SHOTS, BETA, SEED, label, BACKEND,
                    IBM_BACKEND, circuit_hash, pdb_id, seq, gi, s, count, prob
                ])

        print(f"[OK] {pdb_id}: {PER_GROUP} samples -> {out_path.name}")

    print("\n✅ All random bitstring files generated successfully!")


if __name__ == "__main__":
    main()
