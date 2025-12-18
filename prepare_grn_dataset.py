# --*-- conding:utf-8 --*--
# @time:11/2/25 17:51
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prepare_grn_dataset.py

# Build per-PDB GRN inference inputs and compute reference RMSD labels.
# Inputs per PDB under: pp_result/<pdb_id>/<pdb_id>/{energies.jsonl, features.jsonl, decoded.jsonl}
# Reference structures under: dataset/Pdbbind/<pdb_id>/<pdb_id>_{pocket,protein}.pdb
# Benchmark index: dataset/benchmark_info.txt

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Optional

import numpy as np


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_benchmark_info(path: Path) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = [h.strip() for h in line.split("\t")]
                continue
            parts = [p.strip() for p in line.split("\t")]
            rec = dict(zip(header, parts))
            pdb = rec["pdb_id"]
            idx[pdb] = rec
    return idx


def parse_residue_span(span: str) -> Tuple[int, int]:
    s = span.strip()
    if "-" not in s:
        raise ValueError(f"Invalid residue span: {s}")
    a, b = s.split("-", 1)
    return int(a), int(b)


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("Input shapes must be (N,3) and equal.")
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    P_rot = P0 @ R
    P_aligned = P_rot + Qc
    diff = P_aligned - Q
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    return P_aligned, float(rmsd)


def load_reference_ca(pdb_path: Path, start_res: int, end_res: int) -> Optional[np.ndarray]:
    if not pdb_path.exists():
        return None
    ca = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq < start_res or resseq > end_res:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            ca.append([x, y, z])
    if not ca:
        return None
    return np.asarray(ca, dtype=np.float64)


def load_reference_ca_by_chain(pdb_path: Path, start_res: int, end_res: int) -> Dict[str, np.ndarray]:
    chains: Dict[str, np.ndarray] = {}
    if not pdb_path.exists():
        return chains
    tmp: Dict[str, List[List[float]]] = {}
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            if resseq < start_res or resseq > end_res:
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            tmp.setdefault(chain_id, []).append([x, y, z])
    for c, arr in tmp.items():
        if arr:
            chains[c] = np.asarray(arr, dtype=np.float64)
    return chains


def decode_main_positions(row: Dict[str, Any]) -> Optional[np.ndarray]:
    pos = row.get("main_positions")
    if pos is None:
        return None
    try:
        arr = np.asarray(pos, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr


def graded_label_from_rmsd(r: float) -> int:
    if r < 2.5:
        return 3
    if r < 4.5:
        return 2
    if r < 6.5:
        return 1
    return 0


def merge_inputs_per_pdb(pdb_id: str, pdb_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    energies_p = pdb_dir / "energies.jsonl"
    features_p = pdb_dir / "features.jsonl"
    decoded_p = pdb_dir / "decoded.jsonl"

    if not energies_p.exists() or not features_p.exists():
        raise FileNotFoundError(f"Missing energies/features for {pdb_id}: {pdb_dir}")

    E: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(energies_p):
        bs = row.get("bitstring")
        if not bs:
            bs = row.get("id") or row.get("bit_str")
        if not bs:
            continue
        E[bs] = row

    F: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(features_p):
        bs = row.get("bitstring")
        if not bs:
            bs = row.get("id") or row.get("bit_str")
        if not bs:
            continue
        F[bs] = row

    seq_by_bs: Dict[str, str] = {}
    pos_by_bs: Dict[str, np.ndarray] = {}
    if decoded_p.exists():
        for row in read_jsonl(decoded_p):
            bs = row.get("bitstring")
            if not bs:
                continue
            seq = row.get("sequence") if isinstance(row.get("sequence"), str) else ""
            P = decode_main_positions(row)
            if seq and P is not None:
                seq_by_bs[bs] = seq
                pos_by_bs[bs] = P

    merged_rows: List[Dict[str, Any]] = []
    pos_cache: Dict[str, Dict[str, Any]] = {}
    keys = list(E.keys())
    for bs in keys:
        row = {}
        row["group_id"] = 0
        row["pdb_id"] = pdb_id
        row["bitstring"] = bs
        seq = E.get(bs, {}).get("sequence") or F.get(bs, {}).get("sequence") or seq_by_bs.get(bs) or ""
        row["sequence"] = seq

        for src in (E.get(bs, {}), F.get(bs, {})):
            for k, v in src.items():
                if k in ("sequence", "bitstring", "main_positions", "side_positions", "fifth_bit"):
                    continue
                if isinstance(v, (int, float)) or (isinstance(v, (list, tuple)) is False and isinstance(v, (np.floating, np.integer))):
                    row[k] = float(v)
        merged_rows.append(row)

        if bs in pos_by_bs:
            pos_cache[bs] = {"sequence": seq, "positions": pos_by_bs[bs]}

    return merged_rows, pos_cache


def compute_rmsd_records(
    pdb_id: str,
    pos_cache: Dict[str, Dict[str, Any]],
    bench_rec: Dict[str, Any],
    pdbbind_root: Path,
) -> List[Dict[str, Any]]:
    span = bench_rec["Residues"]
    start_res, end_res = parse_residue_span(span)

    pocket_p = pdbbind_root / pdb_id / f"{pdb_id}_pocket.pdb"
    protein_p = pdbbind_root / pdb_id / f"{pdb_id}_protein.pdb"

    ref_chains = load_reference_ca_by_chain(pocket_p, start_res, end_res)
    if not ref_chains:
        ref_chains = load_reference_ca_by_chain(protein_p, start_res, end_res)
    if not ref_chains:
        raise FileNotFoundError(f"Cannot load reference CA for {pdb_id}")

    lengths = [rec["positions"].shape[0]
               for rec in pos_cache.values()
               if isinstance(rec.get("positions"), np.ndarray)
               and rec["positions"].ndim == 2 and rec["positions"].shape[1] == 3]
    if not lengths:
        return []

    from collections import Counter
    L_mode = Counter(lengths).most_common(1)[0][0]

    chain_choices = [(cid, arr.shape[0], arr) for cid, arr in ref_chains.items()]
    chain_choices.sort(key=lambda x: (abs(x[1] - L_mode), -x[1]))
    ref_chain_id, L_ref, ref_full = chain_choices[0]

    out: List[Dict[str, Any]] = []
    for bs, rec in pos_cache.items():
        P = rec.get("positions")
        if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[1] != 3:
            continue

        L_common = min(P.shape[0], L_ref)
        if L_common < 3:
            continue
        P_use = P[:L_common]
        Q_use = ref_full[:L_common]

        _, rmsd_val = kabsch_align(P_use, Q_use)
        rel = graded_label_from_rmsd(rmsd_val)
        out.append({
            "pdb_id": pdb_id,
            "group_id": 0,
            "bitstring": bs,
            "sequence": rec.get("sequence", ""),
            "rmsd": float(rmsd_val),
            "rel": int(rel),
        })
    return out


def save_rmsd_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pdb_id", "group_id", "bitstring", "sequence", "rmsd", "rel"])
        return
    fieldnames = ["pdb_id", "group_id", "bitstring", "sequence", "rmsd", "rel"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def join_input_with_rmsd(input_rows: List[Dict[str, Any]], rmsd_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_bs = {r["bitstring"]: r for r in rmsd_rows}
    merged: List[Dict[str, Any]] = []
    for r in input_rows:
        bs = r.get("bitstring")
        extra = by_bs.get(bs)
        if extra:
            m = dict(r)
            m["rmsd"] = extra["rmsd"]
            m["rel"] = extra["rel"]
            merged.append(m)
    return merged


def discover_pdb_ids(pp_root: Path) -> List[str]:
    pdbs = []
    for d in pp_root.iterdir():
        if not d.is_dir():
            continue
        pdb_id = d.name
        nested = d / pdb_id
        if (nested / "energies.jsonl").exists() and (nested / "features.jsonl").exists():
            pdbs.append(pdb_id)
    return sorted(pdbs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp_root", type=str, default="pp_result")
    ap.add_argument("--dataset_root", type=str, default="dataset")
    ap.add_argument("--bench_info", type=str, default="dataset/benchmark_info.txt")
    ap.add_argument("--out_root", type=str, default="prepared_dataset")
    ap.add_argument("--only", type=str, default="", help="Comma-separated pdb_ids to process, e.g. '4f5y,6czf'.")
    args = ap.parse_args()

    pp_root = Path(args.pp_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bench = read_benchmark_info(Path(args.bench_info))
    pdbbind_root = Path(args.dataset_root) / "Pdbbind"

    pdb_ids = discover_pdb_ids(pp_root)
    if not pdb_ids:
        print("[WARN] No pdb directories discovered under:", pp_root.resolve())
        return

    if args.only.strip():
        allow = {p.strip().lower() for p in args.only.split(",") if p.strip()}
        pdb_ids = [p for p in pdb_ids if p.lower() in allow]
        if not pdb_ids:
            print(f"[WARN] No pdb_ids matched --only={args.only}")
            return

    print(f"[INFO] Found {len(pdb_ids)} pdbs: {', '.join(pdb_ids[:8])}{' ...' if len(pdb_ids)>8 else ''}")

    for pdb_id in pdb_ids:
        print(f"[PROC] {pdb_id}")
        pdb_dir = pp_root / pdb_id / pdb_id
        try:
            input_rows, pos_cache = merge_inputs_per_pdb(pdb_id, pdb_dir)
        except Exception as e:
            print(f"[SKIP] {pdb_id} merge failed: {e}")
            continue

        input_path = out_root / f"{pdb_id}_grn_input.jsonl"
        write_jsonl(input_path, input_rows)
        print(f"  -> saved GRN input: {input_path}")

        if pdb_id not in bench:
            print(f"  -> benchmark_info missing entry for {pdb_id}, skip RMSD")
            continue

        try:
            rmsd_rows = compute_rmsd_records(pdb_id, pos_cache, bench[pdb_id], pdbbind_root)
        except Exception as e:
            print(f"  -> RMSD compute failed for {pdb_id}: {e}")
            continue

        rmsd_jsonl = out_root / f"{pdb_id}_rmsd.jsonl"
        write_jsonl(rmsd_jsonl, rmsd_rows)
        print(f"  -> saved RMSD jsonl: {rmsd_jsonl}")

        rmsd_csv = out_root / f"{pdb_id}_rmsd.csv"
        save_rmsd_csv(rmsd_csv, rmsd_rows)
        print(f"  -> saved RMSD csv:   {rmsd_csv}")

        merged_rows = join_input_with_rmsd(input_rows, rmsd_rows)
        if merged_rows:
            merged_path = out_root / f"{pdb_id}_grn_input_with_rmsd.jsonl"
            write_jsonl(merged_path, merged_rows)
            print(f"  -> saved merged:     {merged_path}")

    print("[DONE] All outputs are in:", out_root.resolve())


if __name__ == "__main__":
    main()
