# --*-- conding:utf-8 --*--
# @time:12/27/25 15:13
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pred_docking_decoded.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import linecache
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


AA1_TO_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def read_jsonl_line(decoded_path: str, line_index: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Try reading a JSONL record at line_index.
    We attempt both 0-based and 1-based interpretations.
    Returns (obj, mode) where mode is "0-based" or "1-based".
    """
    p = str(decoded_path)

    # linecache.getline is 1-based
    # Attempt 0-based: use line_index + 1
    s0 = linecache.getline(p, line_index + 1)
    if s0 and s0.strip():
        try:
            return json.loads(s0), "0-based"
        except Exception:
            pass

    # Attempt 1-based: use line_index
    if line_index >= 1:
        s1 = linecache.getline(p, line_index)
        if s1 and s1.strip():
            try:
                return json.loads(s1), "1-based"
            except Exception:
                pass

    return None, "not-found"


def scale_positions(main_positions: List[List[float]], scale_factor: float) -> List[List[float]]:
    out: List[List[float]] = []
    for xyz in main_positions:
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            continue
        x, y, z = safe_float(xyz[0]), safe_float(xyz[1]), safe_float(xyz[2])
        out.append([x * scale_factor, y * scale_factor, z * scale_factor])
    return out


def write_ca_json(out_json_path: str, payload: Dict[str, Any]) -> None:
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def format_pdb_atom_line(
    atom_serial: int,
    res_name: str,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
) -> str:
    # PDB fixed-width formatting (CA-only)
    # Columns: https://www.wwpdb.org/documentation/file-format
    return (
        f"ATOM  {atom_serial:5d}  CA  {res_name:>3s} {chain_id:1s}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.00:6.2f}{0.00:6.2f}          {'C':>2s}"
    )


def write_ca_pdb(
    out_pdb_path: str,
    sequence: str,
    chain_id: str,
    res_start: int,
    ca_positions_angstrom: List[List[float]],
) -> None:
    Path(out_pdb_path).parent.mkdir(parents=True, exist_ok=True)

    n = min(len(sequence), len(ca_positions_angstrom))
    lines: List[str] = []
    atom_serial = 1
    for i in range(n):
        aa1 = sequence[i].upper()
        res3 = AA1_TO_AA3.get(aa1, "UNK")
        res_seq = res_start + i
        x, y, z = ca_positions_angstrom[i]
        lines.append(format_pdb_atom_line(atom_serial, res3, chain_id, res_seq, x, y, z))
        atom_serial += 1
    lines.append("END")

    with open(out_pdb_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def is_empty_file(path: str) -> bool:
    try:
        p = Path(path)
        if not p.exists():
            return True
        return p.stat().st_size == 0
    except Exception:
        return True


def process_case(row: Dict[str, str], force: bool = False) -> Tuple[bool, str]:
    case_id = row.get("case_id", "").strip()
    decoded_file = row.get("decoded_file", "").strip()
    pred_ca_pdb = row.get("pred_ca_pdb", "").strip()
    pred_ca_json = row.get("pred_ca_json", "").strip()

    chain_id = (row.get("chain_id", "") or "A").strip()[:1]
    res_start = safe_int(row.get("start_resi", ""), 1)
    line_index = safe_int(row.get("line_index", ""), -1)
    scale_factor = safe_float(row.get("scale_factor", ""), 3.8)

    if not case_id:
        return False, "missing case_id"
    if not decoded_file or not Path(decoded_file).exists():
        return False, f"decoded_file not found: {decoded_file}"
    if line_index < 0:
        return False, f"invalid line_index: {line_index}"
    if not pred_ca_pdb or not pred_ca_json:
        return False, "missing pred_ca_pdb or pred_ca_json"

    # Skip if both outputs exist and are non-empty, unless forced
    if not force and (not is_empty_file(pred_ca_pdb)) and (not is_empty_file(pred_ca_json)):
        return True, "skipped (already filled)"

    obj, mode = read_jsonl_line(decoded_file, line_index)
    if obj is None:
        return False, f"failed to read JSON at line_index={line_index} ({mode})"

    seq = str(obj.get("sequence", "")).strip()
    main_positions = obj.get("main_positions", None)
    if not seq:
        # fallback to CSV best_sequence
        seq = (row.get("best_sequence", "") or "").strip()
    if not seq:
        return False, "missing sequence in decoded line and CSV"

    if not isinstance(main_positions, list) or not main_positions:
        return False, "decoded JSON missing main_positions"

    ca_positions_angstrom = scale_positions(main_positions, scale_factor)
    if not ca_positions_angstrom:
        return False, "failed to scale main_positions"

    payload = {
        "sequence": seq,
        "chain_id": chain_id,
        "res_start": res_start,
        "scale_factor": scale_factor,
        "ca_positions_angstrom": ca_positions_angstrom,
    }

    write_ca_json(pred_ca_json, payload)
    write_ca_pdb(pred_ca_pdb, seq, chain_id, res_start, ca_positions_angstrom)

    return True, f"filled (read {mode})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill pred_ca_json/pdb from decoded.jsonl using docking_data/cases.csv")
    ap.add_argument("--cases", default="docking_data/cases.csv", help="Path to cases.csv (default: docking_data/cases.csv)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing non-empty outputs")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N cases (0 = all)")
    args = ap.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"cases.csv not found: {cases_path}")

    ok_cnt = 0
    fail_cnt = 0
    skipped_cnt = 0

    with open(cases_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print(f"[INFO] cases: {cases_path} | total={len(rows)} | force={args.force}")

    for i, row in enumerate(rows, 1):
        case_id = (row.get("case_id", "") or "").strip()
        success, msg = process_case(row, force=args.force)

        if success and msg.startswith("skipped"):
            skipped_cnt += 1
            print(f"[{i:04d}] [SKIP] {case_id}: {msg}")
        elif success:
            ok_cnt += 1
            print(f"[{i:04d}] [OK]   {case_id}: {msg}")
        else:
            fail_cnt += 1
            print(f"[{i:04d}] [FAIL] {case_id}: {msg}")

    print("------------------------------------------------------------")
    print(f"[DONE] OK={ok_cnt} | SKIP={skipped_cnt} | FAIL={fail_cnt} | TOTAL={len(rows)}")


if __name__ == "__main__":
    main()
