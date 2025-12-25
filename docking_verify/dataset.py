# --*-- conding:utf-8 --*--
# @time:12/24/25 23:51
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:dataset.py

# docking_verify/dataset.py
# -*- coding: utf-8 -*-
"""
docking_verify.dataset

Purpose
-------
Normalize scattered inputs into a reusable docking dataset index.

Inputs
------
1) pp_result/result_summary.csv
   Columns (required):
     fragment_id,min_rmsd,best_bitstring,decoded_file,line_index,scale_factor,
     ref_pdb,chain,res_start,res_end,scale_mode,best_sequence

2) RCSB_KRAS/  (directory containing reference PDB files, e.g. 4lpk.pdb, 6OIM.pdb, 9C41.pdb)

3) decoded.jsonl files referenced by result_summary.csv
   Each line is a JSON object with at least:
     sequence (1-letter AA string)
     main_positions (list of [x,y,z] coordinates; length L or L+1)

Outputs (under --out directory)
------------------------------
- cases.csv                         (normalized index for later pipeline)
- fragments/{fragment_id}_ca.pdb     (CA-only PDB from predicted main_positions)
- fragments/{fragment_id}_ca.json    (optional; numeric coordinates, after scaling)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AA1_TO_AA3: Dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


@dataclass(frozen=True)
class SummaryRow:
    fragment_id: str
    min_rmsd: str
    best_bitstring: str
    decoded_file: str
    line_index: str
    scale_factor: str
    ref_pdb: str
    chain: str
    res_start: str
    res_end: str
    scale_mode: str
    best_sequence: str


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _read_summary_csv(path: Path) -> List[SummaryRow]:
    required = [
        "fragment_id", "min_rmsd", "best_bitstring", "decoded_file", "line_index",
        "scale_factor", "ref_pdb", "chain", "res_start", "res_end", "scale_mode", "best_sequence",
    ]
    rows: List[SummaryRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {path}")
        for k in required:
            if k not in reader.fieldnames:
                raise ValueError(f"Missing column '{k}' in {path}")

        for d in reader:
            rows.append(
                SummaryRow(
                    fragment_id=d["fragment_id"],
                    min_rmsd=d["min_rmsd"],
                    best_bitstring=d["best_bitstring"],
                    decoded_file=d["decoded_file"],
                    line_index=d["line_index"],
                    scale_factor=d["scale_factor"],
                    ref_pdb=d["ref_pdb"],
                    chain=d["chain"],
                    res_start=d["res_start"],
                    res_end=d["res_end"],
                    scale_mode=d["scale_mode"],
                    best_sequence=d["best_sequence"],
                )
            )
    return rows


def _find_pdb_case_insensitive(pdb_dir: Path, filename: str) -> Optional[Path]:
    # Direct hit
    direct = pdb_dir / filename
    if direct.exists():
        return direct.resolve()

    target = filename.lower()

    # Same directory
    for p in pdb_dir.glob("*.pdb"):
        if p.name.lower() == target:
            return p.resolve()

    # Recursive (in case nested)
    for p in pdb_dir.rglob("*.pdb"):
        if p.name.lower() == target:
            return p.resolve()

    return None


def _read_jsonl_line_with_fallback(path: Path, line_index: int) -> Optional[Dict[str, Any]]:
    """
    Try 0-based line_index first; if not found, try 1-based (line_index-1).
    """
    def read_n(n: int) -> Optional[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == n:
                    s = line.strip()
                    if not s:
                        return None
                    return json.loads(s)
        return None

    obj = read_n(line_index)
    if obj is not None:
        return obj
    if line_index > 0:
        return read_n(line_index - 1)
    return None


def _normalize_main_positions(sequence: str, main_positions: List[Any]) -> List[List[float]]:
    """
    main_positions may be length L or L+1.
    Return a list of length L, each is [x,y,z] floats.
    """
    L = len(sequence)

    if len(main_positions) == L:
        coords = main_positions
    elif len(main_positions) == L + 1:
        # Often there is an extra start point at index 0
        coords = main_positions[1:]
    else:
        raise ValueError(f"Unexpected main_positions length: {len(main_positions)} vs seq_len {L}")

    out: List[List[float]] = []
    for idx, c in enumerate(coords):
        if not (isinstance(c, list) or isinstance(c, tuple)) or len(c) != 3:
            raise ValueError(f"Bad coordinate at position {idx}: {c}")
        try:
            x = float(c[0])
            y = float(c[1])
            z = float(c[2])
        except Exception:
            raise ValueError(f"Non-numeric coordinate at position {idx}: {c}")
        out.append([x, y, z])
    return out


def _build_ca_only_pdb(
    sequence: str,
    ca_positions: List[List[float]],
    chain_id: str,
    res_start: int,
    scale_factor: float,
) -> str:
    """
    CA-only PDB. We treat predicted main_positions as CA positions.
    """
    if len(ca_positions) != len(sequence):
        raise ValueError(f"CA length {len(ca_positions)} != sequence length {len(sequence)}")

    lines: List[str] = []
    serial = 1
    for i, aa in enumerate(sequence):
        resi = res_start + i
        resname = AA1_TO_AA3.get(aa.upper(), "UNK")
        x, y, z = ca_positions[i]
        x *= scale_factor
        y *= scale_factor
        z *= scale_factor

        # Standard PDB ATOM formatting for CA
        line = (
            f"ATOM  {serial:5d}  CA  {resname} {chain_id}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
        serial += 1

    lines.append("END")
    return "\n".join(lines) + "\n"


def _extract_predicted_ca(
    decoded_jsonl: Path,
    line_index: int,
    out_pdb: Path,
    out_json: Optional[Path],
    chain_id: str,
    res_start: int,
    scale_factor: float,
) -> Tuple[Optional[Path], Optional[Path], str]:
    """
    Deterministically extract CA-only structure from decoded.jsonl line:
      - sequence
      - main_positions
    """
    obj = _read_jsonl_line_with_fallback(decoded_jsonl, line_index)
    if obj is None:
        return None, None, "decoded_line_not_found"

    seq = obj.get("sequence")
    main_positions = obj.get("main_positions")

    if not isinstance(seq, str):
        return None, None, "missing_or_bad_sequence"
    if not isinstance(main_positions, list):
        return None, None, "missing_or_bad_main_positions"

    try:
        ca = _normalize_main_positions(seq, main_positions)
    except Exception as e:
        return None, None, f"main_positions_parse_error:{e}"

    # Write CA-only PDB
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    try:
        pdb_str = _build_ca_only_pdb(
            sequence=seq,
            ca_positions=ca,
            chain_id=chain_id,
            res_start=res_start,
            scale_factor=scale_factor,
        )
        out_pdb.write_text(pdb_str, encoding="utf-8")
    except Exception as e:
        return None, None, f"pdb_write_error:{e}"

    # Optionally write JSON (after scaling, in Å)
    json_path: Optional[Path] = None
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        scaled = [[p[0] * scale_factor, p[1] * scale_factor, p[2] * scale_factor] for p in ca]
        payload = {
            "sequence": seq,
            "chain_id": chain_id,
            "res_start": res_start,
            "scale_factor": scale_factor,
            "ca_positions_angstrom": scaled,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        json_path = out_json.resolve()

    return out_pdb.resolve(), json_path, "extracted_ca_only"


def _default_ligand_resname(ref_pdb: str) -> str:
    # Your current KRAS set: 4LPK and 9C41 are GDP-bound; 6OIM has MOV.
    low = ref_pdb.lower()
    if "6oim" in low:
        return "MOV"
    return "GDP"


def _parse_kv_list(items: List[str]) -> Dict[str, str]:
    """
    Parse repeated --ligand-map entries like:
      --ligand-map 4lpk.pdb=GDP --ligand-map 6OIM.pdb=MOV
    """
    m: Dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad entry: {it} (expected key=value)")
        k, v = it.split("=", 1)
        m[k.strip()] = v.strip()
    return m


def build_cases(
    summary_csv: Path,
    pdb_dir: Path,
    out_dir: Path,
    extract_fragments: bool = True,
    write_ca_json: bool = True,
    ligand_map: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Build normalized cases.csv and (optionally) extract CA-only predicted fragments.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_summary_csv(summary_csv)
    ligand_map = ligand_map or {}

    fragments_dir = out_dir / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)

    cases_csv = out_dir / "cases.csv"

    fieldnames = [
        # Core fields for docking pipeline
        "case_id",
        "pdb_path",
        "ref_pdb",
        "chain_id",
        "start_resi",
        "end_resi",
        "pred_ca_pdb",
        "pred_ca_json",
        "ligand_resname",
        # Provenance/debug
        "decoded_file",
        "line_index",
        "min_rmsd",
        "best_sequence",
        "scale_factor",
        "scale_mode",
        "extract_status",
    ]

    with cases_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            pdb_path = _find_pdb_case_insensitive(pdb_dir, r.ref_pdb)
            if pdb_path is None:
                raise FileNotFoundError(f"Cannot find '{r.ref_pdb}' under {pdb_dir}")

            # Required numeric conversions
            start_resi = _safe_int(r.res_start)
            end_resi = _safe_int(r.res_end)
            line_idx = _safe_int(r.line_index)
            scale = _safe_float(r.scale_factor)

            if start_resi is None or end_resi is None:
                raise ValueError(f"Bad res_start/res_end for {r.fragment_id}: {r.res_start}, {r.res_end}")
            if scale is None:
                raise ValueError(f"Bad scale_factor for {r.fragment_id}: {r.scale_factor}")

            ligand_resname = (
                ligand_map.get(r.fragment_id)
                or ligand_map.get(r.ref_pdb)
                or _default_ligand_resname(r.ref_pdb)
            )

            pred_ca_pdb: str = ""
            pred_ca_json: str = ""
            status: str = "not_extracted"

            if extract_fragments:
                decoded = Path(r.decoded_file).expanduser()
                if decoded.exists() and line_idx is not None:
                    out_pdb = fragments_dir / f"{r.fragment_id}_ca.pdb"
                    out_js = fragments_dir / f"{r.fragment_id}_ca.json" if write_ca_json else None
                    pdb_out, json_out, status = _extract_predicted_ca(
                        decoded_jsonl=decoded,
                        line_index=line_idx,
                        out_pdb=out_pdb,
                        out_json=out_js,
                        chain_id=r.chain,
                        res_start=start_resi,
                        scale_factor=scale,
                    )
                    pred_ca_pdb = str(pdb_out) if pdb_out else ""
                    pred_ca_json = str(json_out) if json_out else ""
                else:
                    status = "decoded_missing_or_bad_line_index"

            writer.writerow(
                {
                    "case_id": r.fragment_id,
                    "pdb_path": str(pdb_path),
                    "ref_pdb": r.ref_pdb,
                    "chain_id": r.chain,
                    "start_resi": str(start_resi),
                    "end_resi": str(end_resi),
                    "pred_ca_pdb": pred_ca_pdb,
                    "pred_ca_json": pred_ca_json,
                    "ligand_resname": ligand_resname,
                    "decoded_file": r.decoded_file,
                    "line_index": r.line_index,
                    "min_rmsd": r.min_rmsd,
                    "best_sequence": r.best_sequence,
                    "scale_factor": r.scale_factor,
                    "scale_mode": r.scale_mode,
                    "extract_status": status,
                }
            )

    return cases_csv.resolve()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build docking dataset index (cases.csv) from pp_result summary + decoded.jsonl + RCSB PDBs."
    )
    ap.add_argument("--summary", type=str, required=True, help="Path to pp_result/result_summary.csv")
    ap.add_argument("--pdb-dir", type=str, required=True, help="Directory containing reference PDBs (RCSB_KRAS)")
    ap.add_argument("--out", type=str, required=True, help="Output directory (cases.csv + fragments/)")
    ap.add_argument("--no-extract", action="store_true", help="Do not extract CA-only predicted fragments")
    ap.add_argument("--no-ca-json", action="store_true", help="Do not write CA coordinate JSON files")
    ap.add_argument(
        "--ligand-map",
        type=str,
        action="append",
        default=[],
        help="Override ligand resname mapping: e.g. 4lpk.pdb=GDP 6OIM.pdb=MOV or fragment_id=MOV",
    )

    args = ap.parse_args()

    summary = Path(args.summary).expanduser().resolve()
    pdb_dir = Path(args.pdb_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not summary.exists():
        raise FileNotFoundError(f"summary not found: {summary}")
    if not pdb_dir.exists():
        raise FileNotFoundError(f"pdb-dir not found: {pdb_dir}")

    ligand_map = _parse_kv_list(args.ligand_map)

    cases_path = build_cases(
        summary_csv=summary,
        pdb_dir=pdb_dir,
        out_dir=out_dir,
        extract_fragments=(not args.no_extract),
        write_ca_json=(not args.no_ca_json),
        ligand_map=ligand_map if ligand_map else None,
    )

    print(f"[OK] Wrote: {cases_path}")
    print(f"[INFO] Extracted fragments under: {(out_dir / 'fragments').resolve()}")
    print("[INFO] Check column 'extract_status' in cases.csv for any extraction issues.")


if __name__ == "__main__":
    main()
