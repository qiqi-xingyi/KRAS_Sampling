# --*-- conding:utf-8 --*--
# @time:12/18/25 21:58
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:build_grn_inputs_from_pp_result.py

# Build GRN inference inputs directly from pp_result outputs produced by QSAD post_process.py.
#
# Expected structure per fragment:
#   pp_result/<protein_id>/<pdbid>/{energies.jsonl, features.jsonl}
# Example:
#   pp_result/4LPK_WT_1/4LPK_WT/energies.jsonl
#   pp_result/4LPK_WT_1/4LPK_WT/features.jsonl
#
# Output:
#   - combined JSONL for all fragments: <out_root>/kras_all_grn_input.jsonl
#   - optional per-fragment JSONL:      <out_root>/<protein_id>_grn_input.jsonl
#
# IMPORTANT (your requirement):
#   WT_1/WT_2/WT_3 are different fragments, NOT replicates.
#   So we set:
#       pdb_id  = <protein_id>   (outer folder name, unique per fragment)
#       group_id = 0             (force all samples in the fragment to be ranked together)

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _to_float_if_number(x: Any) -> Any:
    if _is_number(x):
        return float(x)
    return x


def _infer_outer_and_inner(pp_root: Path, protein_id: str) -> Optional[Tuple[str, Path]]:
    """
    Returns (inner_pdbid, inner_dir) for a given protein_id folder.
    We pick the first child directory that contains energies.jsonl and features.jsonl.
    """
    outer = pp_root / protein_id
    if not outer.is_dir():
        return None

    # common QSAD output: pp_result/<protein_id>/<pdbid>/...
    for child in sorted(outer.iterdir()):
        if not child.is_dir():
            continue
        en = child / "energies.jsonl"
        ft = child / "features.jsonl"
        if en.exists() and ft.exists():
            return child.name, child

    return None


def merge_one_fragment(
    protein_id: str,
    inner_dir: Path,
    keep_non_numeric: bool = False,
) -> List[Dict[str, Any]]:
    """
    Merge energies.jsonl + features.jsonl by bitstring.
    Output records for GRN:
      - pdb_id = protein_id (outer dir name, unique per fragment)
      - group_id = 0
      - bitstring, sequence
      - numeric features from energies/features (float)
    """
    energies_p = inner_dir / "energies.jsonl"
    features_p = inner_dir / "features.jsonl"

    # Load energies by bitstring
    E: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(energies_p):
        bs = row.get("bitstring") or row.get("id") or row.get("bit_str")
        if not bs:
            continue
        E[str(bs)] = row

    # Load features by bitstring
    F: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(features_p):
        bs = row.get("bitstring") or row.get("id") or row.get("bit_str")
        if not bs:
            continue
        F[str(bs)] = row

    common = sorted(set(E.keys()) & set(F.keys()))
    out: List[Dict[str, Any]] = []

    # Keys we never want to copy (GRN doesn't need them; also avoids huge JSON)
    skip_keys = {
        "main_positions", "side_positions",
        "main_vectors", "side_vectors",
        "fifth_bit",
        # sometimes present
        "positions", "coords",
    }

    for bs in common:
        er = E[bs]
        fr = F[bs]

        seq = er.get("sequence") or fr.get("sequence") or ""
        rec: Dict[str, Any] = {
            "pdb_id": protein_id,   # critical: unique per fragment
            "group_id": 0,          # critical: rank all samples together within fragment
            "bitstring": bs,
            "sequence": str(seq),
        }

        # Copy numeric fields from energies and features
        for src in (er, fr):
            for k, v in src.items():
                if k in ("pdb_id", "group_id", "bitstring", "sequence"):
                    continue
                if k in skip_keys:
                    continue

                if _is_number(v):
                    rec[k] = float(v)
                else:
                    if keep_non_numeric:
                        # keep small non-numeric fields if requested (rarely useful)
                        if isinstance(v, (str, bool)) or v is None:
                            rec[k] = v

        out.append(rec)

    return out


def discover_protein_ids(pp_root: Path) -> List[str]:
    """
    Find pp_result/<protein_id>/... excluding _staging and obvious non-fragment files.
    """
    out: List[str] = []
    for p in sorted(pp_root.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("."):
            continue
        if name in {"_staging"}:
            continue
        out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp_root", type=str, default="pp_result", help="Root dir that contains per-fragment outputs.")
    ap.add_argument("--out_root", type=str, default="prepared_dataset", help="Where to write GRN input jsonl(s).")
    ap.add_argument("--out_name", type=str, default="kras_all_grn_input.jsonl", help="Combined output filename.")
    ap.add_argument("--per_fragment", action="store_true", help="Also write one jsonl per fragment.")
    ap.add_argument("--only", type=str, default="", help="Comma-separated protein_id filters (outer folder names).")
    ap.add_argument("--keep_non_numeric", action="store_true", help="Keep small non-numeric fields too (default: numeric only).")
    args = ap.parse_args()

    pp_root = Path(args.pp_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not pp_root.exists():
        raise FileNotFoundError(f"pp_root not found: {pp_root.resolve()}")

    protein_ids = discover_protein_ids(pp_root)

    if args.only.strip():
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        protein_ids = [pid for pid in protein_ids if pid in allow]
        if not protein_ids:
            print(f"[WARN] No protein_id matched --only={args.only}")
            return

    all_rows: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []

    for protein_id in protein_ids:
        found = _infer_outer_and_inner(pp_root, protein_id)
        if not found:
            skipped.append((protein_id, "missing <pdbid>/{energies.jsonl,features.jsonl}"))
            continue
        inner_pdbid, inner_dir = found

        try:
            rows = merge_one_fragment(
                protein_id=protein_id,
                inner_dir=inner_dir,
                keep_non_numeric=bool(args.keep_non_numeric),
            )
        except Exception as e:
            skipped.append((protein_id, f"merge failed: {e}"))
            continue

        if not rows:
            skipped.append((protein_id, "no common bitstrings between energies and features"))
            continue

        all_rows.extend(rows)

        if args.per_fragment:
            out_path = out_root / f"{protein_id}_grn_input.jsonl"
            write_jsonl(out_path, rows)
            print(f"[OK] {protein_id}: wrote {len(rows)} rows -> {out_path} (inner={inner_pdbid})")
        else:
            print(f"[OK] {protein_id}: collected {len(rows)} rows (inner={inner_pdbid})")

    combined_path = out_root / args.out_name
    write_jsonl(combined_path, all_rows)
    print(f"\n[DONE] Combined GRN input: {combined_path}  (rows={len(all_rows)})")

    if skipped:
        print("\n[SKIP] Some fragments were skipped:")
        for pid, reason in skipped:
            print(f"  - {pid}: {reason}")


if __name__ == "__main__":
    main()
