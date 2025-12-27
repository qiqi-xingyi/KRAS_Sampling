# --*-- conding:utf-8 --*--
# @time:12/27/25 16:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analyze_rcsb_kras_pockets.py

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------
# PDB fixed-column helpers
# ---------------------------
@dataclass(frozen=True)
class ResidueID:
    chain_id: str
    resseq: int
    icode: str = " "

    def key(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)


def _is_atom(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def _rec(line: str) -> str:
    return line[0:6].strip()


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def _resname(line: str) -> str:
    return line[17:20].strip()


def _chain(line: str) -> str:
    return (line[21].strip() if len(line) > 21 else "") or " "


def _resseq(line: str) -> int:
    return int(line[22:26].strip())


def _icode(line: str) -> str:
    c = line[26].strip() if len(line) > 26 else ""
    return c if c else " "


def _xyz(line: str) -> Tuple[float, float, float]:
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    return x, y, z


def load_atom_lines(pdb_path: Path) -> List[str]:
    text = Path(pdb_path).read_text(errors="replace")
    return [ln for ln in text.splitlines() if _is_atom(ln)]


# ---------------------------
# Ligand selection
# ---------------------------
_WATER = {"HOH", "WAT", "H2O", "DOD"}
_COMMON_IONS = {
    "NA", "CL", "K", "CA", "MG", "MN", "ZN", "FE", "CU", "CO", "NI",
    "BR", "I", "F",
    "SO4", "PO4", "NO3", "CO3",
    "EDO", "GOL", "PEG",  # common additives; not always, but often not your ligand
}

def group_ligand_instances(atom_lines: Iterable[str], ligand_resname: str) -> Dict[Tuple[str, int, str], List[str]]:
    groups: Dict[Tuple[str, int, str], List[str]] = {}
    for ln in atom_lines:
        if _rec(ln) != "HETATM":
            continue
        if _resname(ln) != ligand_resname:
            continue
        rid = (_chain(ln), _resseq(ln), _icode(ln))
        groups.setdefault(rid, []).append(ln)
    return groups


def pick_ligand_resname_auto(atom_lines: List[str]) -> Optional[str]:
    """
    Auto pick a ligand resname by:
      - scanning HETATM
      - grouping by (resname, chain, resseq, icode)
      - selecting the resname whose *largest instance* has the most atoms
    Exclude water and common ions/additives.
    """
    inst_counts: Dict[Tuple[str, str, int, str], int] = {}
    for ln in atom_lines:
        if _rec(ln) != "HETATM":
            continue
        rn = _resname(ln)
        if not rn or rn in _WATER or rn in _COMMON_IONS:
            continue
        key = (rn, _chain(ln), _resseq(ln), _icode(ln))
        inst_counts[key] = inst_counts.get(key, 0) + 1

    if not inst_counts:
        return None

    # choose instance with max atoms
    best_inst = max(inst_counts.items(), key=lambda kv: kv[1])[0]
    return best_inst[0]


def select_ligand_instance(
    ligand_groups: Dict[Tuple[str, int, str], List[str]],
    ligand_chain: Optional[str] = None,
    ligand_resseq: Optional[int] = None,
    ligand_icode: str = " ",
) -> Tuple[Tuple[str, int, str], List[str]]:
    if not ligand_groups:
        raise ValueError("No ligand instances found.")

    if ligand_chain is not None and ligand_resseq is not None:
        key = ((ligand_chain.strip() or " "), int(ligand_resseq), ligand_icode if ligand_icode else " ")
        if key not in ligand_groups:
            available = sorted(list(ligand_groups.keys()))
            raise ValueError(f"Specified ligand instance {key} not found. Available: {available}")
        return key, ligand_groups[key]

    # default: choose instance with most atoms
    key = max(ligand_groups.keys(), key=lambda k: len(ligand_groups[k]))
    return key, ligand_groups[key]


def atoms_to_coords(atom_lines: List[str]) -> np.ndarray:
    coords = np.array([_xyz(ln) for ln in atom_lines], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Invalid coordinate array.")
    return coords


# ---------------------------
# Receptor selection + pocket neighborhood
# ---------------------------
def build_receptor_atoms(
    atom_lines: Iterable[str],
    *,
    include_hetatm: bool,
    het_whitelist: Optional[Iterable[str]],
    skip_resnames: Optional[Iterable[str]],
) -> Tuple[np.ndarray, List[ResidueID]]:
    het_whitelist_set = set([x.strip() for x in (het_whitelist or []) if x.strip()])
    skip_set = set([x.strip() for x in (skip_resnames or []) if x.strip()])

    coords: List[Tuple[float, float, float]] = []
    res_ids: List[ResidueID] = []

    for ln in atom_lines:
        r = _rec(ln)
        rn = _resname(ln)

        if r == "ATOM":
            coords.append(_xyz(ln))
            res_ids.append(ResidueID(_chain(ln), _resseq(ln), _icode(ln)))
            continue

        if r == "HETATM":
            if rn in skip_set:
                continue
            if include_hetatm and (not het_whitelist_set or rn in het_whitelist_set):
                coords.append(_xyz(ln))
                res_ids.append(ResidueID(_chain(ln), _resseq(ln), _icode(ln)))

    if not coords:
        raise ValueError("No receptor atoms selected (check filters).")
    return np.array(coords, dtype=float), res_ids


def pocket_residues_within_radius(
    receptor_coords: np.ndarray,
    receptor_resids: List[ResidueID],
    ligand_coords: np.ndarray,
    radius: float,
) -> Dict[Tuple[str, int, str], float]:
    r2 = float(radius) * float(radius)
    pocket: Dict[Tuple[str, int, str], float] = {}

    dif = receptor_coords[:, None, :] - ligand_coords[None, :, :]
    d2 = np.sum(dif * dif, axis=2)
    min_d2 = np.min(d2, axis=1)

    for i, md2 in enumerate(min_d2):
        if md2 <= r2:
            rid = receptor_resids[i].key()
            d = float(np.sqrt(md2))
            prev = pocket.get(rid, None)
            if prev is None or d < prev:
                pocket[rid] = d

    return pocket


def summarize_by_chain(pocket_res: Dict[Tuple[str, int, str], float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for (ch, _, _), _d in pocket_res.items():
        out[ch] = out.get(ch, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def infer_is_multichain(
    chain_counts: Dict[str, int],
    *,
    min_second_chain_residues: int = 5,
    min_second_chain_fraction: float = 0.15,
) -> bool:
    if not chain_counts or len(chain_counts) < 2:
        return False
    total = sum(chain_counts.values())
    counts = sorted(chain_counts.values(), reverse=True)
    second = counts[1]
    return (second >= min_second_chain_residues) and (second / max(total, 1) >= min_second_chain_fraction)


# ---------------------------
# cases.csv mapping
# ---------------------------
def load_ligand_map_from_cases(cases_csv: Path) -> Dict[str, str]:
    """
    Returns mapping: ref_pdb (basename) -> ligand_resname
    """
    cases_csv = Path(cases_csv)
    if not cases_csv.exists():
        return {}

    m: Dict[str, str] = {}
    with cases_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = (row.get("ref_pdb") or "").strip()
            lig = (row.get("ligand_resname") or "").strip()
            if ref and lig:
                m[Path(ref).name] = lig
    return m


# ---------------------------
# Runner
# ---------------------------
def analyze_one_pdb(
    pdb_path: Path,
    *,
    ligand_resname: str,
    radius: float,
    include_hetatm_in_receptor: bool,
    het_whitelist: Optional[List[str]],
    skip_resnames: Optional[List[str]],
    min_second_chain_residues: int,
    min_second_chain_fraction: float,
) -> Dict:
    atom_lines = load_atom_lines(pdb_path)

    ligand_groups = group_ligand_instances(atom_lines, ligand_resname=ligand_resname)
    if not ligand_groups:
        raise ValueError(f"No ligand instances for resname={ligand_resname}")

    lig_key, lig_lines = select_ligand_instance(ligand_groups)
    ligand_coords = atoms_to_coords(lig_lines)
    ligand_centroid = ligand_coords.mean(axis=0)

    receptor_coords, receptor_resids = build_receptor_atoms(
        atom_lines,
        include_hetatm=include_hetatm_in_receptor,
        het_whitelist=het_whitelist,
        skip_resnames=skip_resnames,
    )

    pocket_res = pocket_residues_within_radius(receptor_coords, receptor_resids, ligand_coords, radius)
    chain_counts = summarize_by_chain(pocket_res)
    is_multi = infer_is_multichain(
        chain_counts,
        min_second_chain_residues=min_second_chain_residues,
        min_second_chain_fraction=min_second_chain_fraction,
    )

    closest_10 = sorted(pocket_res.items(), key=lambda kv: kv[1])[:10]

    return {
        "pdb": str(pdb_path),
        "ligand_resname": ligand_resname,
        "ligand_selected_instance": {"chain": lig_key[0], "resseq": lig_key[1], "icode": lig_key[2]},
        "radius_A": float(radius),
        "ligand_centroid": [float(x) for x in ligand_centroid.tolist()],
        "pocket_residue_count": int(len(pocket_res)),
        "pocket_chain_counts": chain_counts,
        "is_multichain_pocket": bool(is_multi),
        "closest_10_residues": [
            {"chain": ch, "resseq": rs, "icode": ic, "min_dist_A": float(d)}
            for (ch, rs, ic), d in closest_10
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Traverse a PDB folder and analyze ligand pocket chain composition.")
    ap.add_argument("--pdb_dir", type=str, default="RCSB_KRAS", help="Folder containing PDB files (default: RCSB_KRAS)")
    ap.add_argument("--cases_csv", type=str, default="docking_data/cases.csv",
                    help="Optional cases.csv to provide ligand_resname mapping (default: docking_data/cases.csv)")
    ap.add_argument("--radius", type=float, default=10.0, help="Radius in Angstrom (default: 6.0)")

    ap.add_argument("--include_hetatm_in_receptor", action="store_true", help="Include HETATM in receptor selection")
    ap.add_argument("--het_whitelist", type=str, default="", help="Comma-separated HETATM resnames to include (optional)")
    ap.add_argument("--skip_resnames", type=str, default="HOH,WAT,H2O,DOD",
                    help="Comma-separated resnames to skip (default: water set)")

    ap.add_argument("--min_second_chain_residues", type=int, default=5)
    ap.add_argument("--min_second_chain_fraction", type=float, default=0.15)

    ap.add_argument("--json_out", type=str, default="",
                    help="Optional output JSON path to save a full report (e.g., docking_result/pocket_chain_report.json)")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.exists():
        raise FileNotFoundError(f"pdb_dir not found: {pdb_dir}")

    ligand_map = load_ligand_map_from_cases(Path(args.cases_csv))

    het_whitelist = [x.strip() for x in args.het_whitelist.split(",") if x.strip()]
    skip_resnames = [x.strip() for x in args.skip_resnames.split(",") if x.strip()]

    pdb_files = sorted([p for p in pdb_dir.glob("*.pdb") if p.is_file()])

    results: List[Dict] = []

    print("=== Pocket Chain Analysis (Folder) ===")
    print("pdb_dir:", str(pdb_dir))
    print("radius_A:", float(args.radius))
    print("cases_csv:", str(Path(args.cases_csv)) if Path(args.cases_csv).exists() else "(not found; auto ligand pick)")
    print()

    for pdb_path in pdb_files:
        atom_lines = load_atom_lines(pdb_path)

        # ligand selection: cases.csv mapping first, else auto pick
        lig = ligand_map.get(pdb_path.name, "")
        if not lig:
            lig = pick_ligand_resname_auto(atom_lines) or ""

        if not lig:
            row = {
                "pdb": str(pdb_path),
                "ligand_resname": None,
                "error": "No suitable ligand found (HETATM filtered out).",
            }
            results.append(row)
            print(f"- {pdb_path.name}: [SKIP] no ligand found")
            continue

        try:
            r = analyze_one_pdb(
                pdb_path,
                ligand_resname=lig,
                radius=float(args.radius),
                include_hetatm_in_receptor=bool(args.include_hetatm_in_receptor),
                het_whitelist=het_whitelist if het_whitelist else None,
                skip_resnames=skip_resnames if skip_resnames else None,
                min_second_chain_residues=int(args.min_second_chain_residues),
                min_second_chain_fraction=float(args.min_second_chain_fraction),
            )
            results.append(r)

            cc = r["pocket_chain_counts"]
            multi = r["is_multichain_pocket"]
            nres = r["pocket_residue_count"]
            print(f"- {pdb_path.name}: ligand={lig}  residues={nres}  chains={cc}  multichain={multi}")

        except Exception as e:
            row = {
                "pdb": str(pdb_path),
                "ligand_resname": lig,
                "error": str(e),
            }
            results.append(row)
            print(f"- {pdb_path.name}: [ERROR] ligand={lig}  {e}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"results": results}, indent=2))
        print()
        print("JSON saved to:", str(out_path))


if __name__ == "__main__":
    main()
