# --*-- coding:utf-8 --*--
# @time:12/25/25
# @Author : Yuqi Zhang
# @File:pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import csv
import json
import time

from .vina import run_vina_multi_seed, parse_vina_scores_from_out_pdbqt
from .template_fit import (
    read_pdb_atoms, write_pdb, PDBAtom,
    build_hybrid_receptor_by_template_fit,
)

# unified API names (NO *_strict imports)
from .pdbqt import (
    prepare_ligand_pdbqt_obabel,
    prepare_receptor_pdbqt_obabel,
)


@dataclass
class DockBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_case_field(row: Dict[str, str], keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def load_cases_csv(cases_csv: Union[str, Path]) -> List[Dict[str, str]]:
    p = Path(cases_csv).expanduser().resolve()
    rows: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    if not rows:
        raise RuntimeError(f"No rows in cases_csv: {p}")
    return rows


# -------------------------
# Robust getters for PDBAtom
# -------------------------

def _atom_record(a: PDBAtom) -> str:
    return str(getattr(a, "record", "")).strip()

def _atom_resname(a: PDBAtom) -> str:
    return str(getattr(a, "resname", "")).strip()

def _atom_chain(a: PDBAtom) -> str:
    return str(getattr(a, "chain_id", getattr(a, "chain", "")) or "").strip()

def _atom_resseq(a: PDBAtom) -> int:
    for k in ("resseq", "res_seq", "resi", "resid"):
        if hasattr(a, k):
            try:
                return int(getattr(a, k))
            except Exception:
                pass
    return 0

def _atom_icode(a: PDBAtom) -> str:
    return str(getattr(a, "icode", getattr(a, "insertion_code", "")) or "").strip()


# -------------------------
# PDB helpers
# -------------------------

def _strip_receptor_atoms(
    atoms: List[PDBAtom],
    ligand_resname: str,
    remove_water: bool = True,
    keep_het_resnames: Optional[Sequence[str]] = None,
) -> List[PDBAtom]:
    keep_het = set([x.upper() for x in (keep_het_resnames or [])])
    lig = ligand_resname.upper()

    out: List[PDBAtom] = []
    for a in atoms:
        r = _atom_resname(a).upper()
        rec = _atom_record(a)

        if rec == "ATOM":
            out.append(a)
            continue

        if rec == "HETATM":
            if r == lig:
                continue
            if remove_water and r in ("HOH", "WAT", "H2O"):
                continue
            if keep_het and r in keep_het:
                out.append(a)
                continue
            continue

    return out


def _group_by_ligand_instance(
    atoms: List[PDBAtom],
    ligand_resname: str,
    chain_id: Optional[str] = None,
) -> Dict[Tuple[str, int, str], List[PDBAtom]]:
    lig = ligand_resname.upper()
    groups: Dict[Tuple[str, int, str], List[PDBAtom]] = {}

    want_chain = (chain_id or "").strip() if chain_id is not None else None

    for a in atoms:
        if _atom_record(a) != "HETATM":
            continue
        if _atom_resname(a).upper() != lig:
            continue

        ch = _atom_chain(a)
        if want_chain is not None and ch != want_chain:
            continue

        key = (ch, _atom_resseq(a), _atom_icode(a))
        groups.setdefault(key, []).append(a)

    return groups


def _pick_single_ligand_instance(
    atoms: List[PDBAtom],
    ligand_resname: str,
    chain_id: Optional[str],
) -> Tuple[List[PDBAtom], str]:
    groups = _group_by_ligand_instance(atoms, ligand_resname, chain_id=chain_id)
    if not groups and chain_id is not None:
        groups = _group_by_ligand_instance(atoms, ligand_resname, chain_id=None)

    if not groups:
        return [], ""

    items = list(groups.items())
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    key, lig_atoms = items[0]
    ch, resi, icode = key
    k_str = f"{ch}:{resi}{icode or ''}"
    return lig_atoms, k_str


def _centroid(atoms: List[PDBAtom]) -> Tuple[float, float, float]:
    xs = [float(a.x) for a in atoms]
    ys = [float(a.y) for a in atoms]
    zs = [float(a.z) for a in atoms]
    if not xs:
        raise RuntimeError("Empty atom list for centroid.")
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


# -------------------------
# Main pipeline
# -------------------------

def run_pipeline(
    cases_csv: Union[str, Path],
    out_dir: Union[str, Path],
    vina_exe: str,
    obabel_exe: str,
    receptor_mode: str = "hybrid_templatefit",  # "crystal" or "hybrid_templatefit"
    ligand_mode: str = "from_crystal",
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    box_size: Sequence[float] = (20.0, 20.0, 20.0),
    vina_params=None,  # keep it simple; your run_docking_verify passes dv.VinaParams from schema anyway
    keep_het_resnames_in_receptor: Optional[Sequence[str]] = None,
    remove_water: bool = True,
    resume: bool = True,
    strict: bool = False,
) -> Dict[str, str]:
    if ligand_mode != "from_crystal":
        raise RuntimeError("This pipeline only supports ligand_mode='from_crystal'.")

    out_dir = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    dirs = {
        "receptors": _ensure_dir(out_dir / "receptors"),
        "pdbqt_rec": _ensure_dir(out_dir / "pdbqt" / "receptors"),
        "pdbqt_lig": _ensure_dir(out_dir / "pdbqt" / "ligands"),
        "ligands": _ensure_dir(out_dir / "ligands"),
        "vina_out": _ensure_dir(out_dir / "vina_out"),
        "status": _ensure_dir(out_dir / "status"),
        "reports": _ensure_dir(out_dir / "reports"),
    }

    # VinaParams object can come from schema, keep flexible
    params = vina_params
    sx, sy, sz = float(box_size[0]), float(box_size[1]), float(box_size[2])

    rows = load_cases_csv(cases_csv)
    summary: List[Dict[str, Union[str, float, int]]] = []

    for row in rows:
        case_id = _get_case_field(row, ["case_id", "fragment_id"], None)
        if not case_id:
            raise RuntimeError("cases.csv missing case_id/fragment_id")

        pdb_path = _get_case_field(row, ["pdb_path"], None)
        if not pdb_path:
            raise RuntimeError(f"{case_id}: missing pdb_path in cases.csv")
        pdb_path = str(Path(pdb_path).expanduser().resolve())
        if not Path(pdb_path).exists():
            raise FileNotFoundError(f"{case_id}: pdb not found: {pdb_path}")

        chain_id = _get_case_field(row, ["chain_id", "chain"], "A") or "A"
        start_resi = int(_get_case_field(row, ["start_resi", "res_start", "start"], "0") or "0")
        end_resi = int(_get_case_field(row, ["end_resi", "res_end", "end"], "0") or "0")
        ligand_resname = (_get_case_field(row, ["ligand_resname"], "GDP") or "GDP").strip()

        decoded_file = _get_case_field(row, ["decoded_file"], None)
        line_index = int(_get_case_field(row, ["line_index"], "0") or "0")
        scale_factor = float(_get_case_field(row, ["scale_factor"], "1.0") or "1.0")

        status_path = dirs["status"] / f"{case_id}.json"

        case_status: Dict[str, object] = {
            "case_id": case_id,
            "pdb_path": pdb_path,
            "chain_id": chain_id,
            "start_resi": start_resi,
            "end_resi": end_resi,
            "ligand_resname": ligand_resname,
            "receptor_mode": receptor_mode,
            "ligand_mode": ligand_mode,
            "seeds": list(map(int, seeds)),
            "steps": {},
            "ok": False,
            "error": None,
        }

        t0_case = time.time()

        try:
            atoms_all, _other_all = read_pdb_atoms(pdb_path)

            # --- ligand: pick ONE instance from crystal ---
            lig_atoms, lig_key = _pick_single_ligand_instance(
                atoms_all, ligand_resname=ligand_resname, chain_id=chain_id
            )
            if not lig_atoms:
                raise RuntimeError(f"{case_id}: ligand {ligand_resname} not found in PDB (chain {chain_id}).")

            lig_pdb = dirs["ligands"] / f"{case_id}.{ligand_resname}.{lig_key or 'any'}.pdb"
            write_pdb(lig_pdb, lig_atoms, other_lines=None, drop_conect=True)

            lig_pdbqt = dirs["pdbqt_lig"] / f"{case_id}.{ligand_resname}.{lig_key or 'any'}.pdbqt"
            prepare_ligand_pdbqt_obabel(
                ligand_pdb=lig_pdb,
                out_pdbqt=lig_pdbqt,
                obabel_exe=obabel_exe,
                force=True,  # critical: no stale bad files
            )

            case_status["steps"]["ligand_pdb"] = {"path": str(lig_pdb), "instance": lig_key}
            case_status["steps"]["ligand_pdbqt"] = {"path": str(lig_pdbqt), "instance": lig_key}

            # --- box from ligand centroid ---
            cx, cy, cz = _centroid(lig_atoms)
            box = DockBox(center_x=cx, center_y=cy, center_z=cz, size_x=sx, size_y=sy, size_z=sz)
            case_status["steps"]["box"] = {"center": [cx, cy, cz], "size": [sx, sy, sz]}

            # --- receptor base (strip ligand/water) ---
            rec_atoms = _strip_receptor_atoms(
                atoms_all,
                ligand_resname=ligand_resname,
                remove_water=remove_water,
                keep_het_resnames=keep_het_resnames_in_receptor,
            )
            receptor_base_pdb = dirs["receptors"] / f"{case_id}.receptor_base.pdb"
            write_pdb(receptor_base_pdb, rec_atoms, other_lines=None, drop_conect=True)

            receptor_pdb = receptor_base_pdb

            # --- hybrid receptor (template_fit) ---
            if receptor_mode == "hybrid_templatefit":
                if not decoded_file:
                    raise RuntimeError(f"{case_id}: receptor_mode=hybrid_templatefit but decoded_file missing.")
                hybrid_pdb = dirs["receptors"] / f"{case_id}.hybrid.pdb"
                if (not resume) or (not hybrid_pdb.exists()) or hybrid_pdb.stat().st_size == 0:
                    build_hybrid_receptor_by_template_fit(
                        base_receptor_pdb=receptor_base_pdb,
                        decoded_jsonl=decoded_file,
                        line_index=line_index,
                        scale_factor=scale_factor,
                        chain_id=chain_id,
                        start_resi=start_resi,
                        end_resi=end_resi,
                        out_pdb=hybrid_pdb,
                    )
                receptor_pdb = hybrid_pdb
            elif receptor_mode == "crystal":
                receptor_pdb = receptor_base_pdb
            else:
                raise RuntimeError(f"{case_id}: unknown receptor_mode={receptor_mode}")

            case_status["steps"]["receptor_pdb"] = {"path": str(receptor_pdb), "mode": receptor_mode}

            receptor_pdbqt = dirs["pdbqt_rec"] / f"{case_id}.pdbqt"
            prepare_receptor_pdbqt_obabel(
                receptor_pdb=receptor_pdb,
                out_pdbqt=receptor_pdbqt,
                obabel_exe=obabel_exe,
                force=True,  # critical
            )
            case_status["steps"]["receptor_pdbqt"] = {"path": str(receptor_pdbqt)}

            # --- run vina ---
            vina_out_root = dirs["vina_out"] / case_id
            runs = run_vina_multi_seed(
                vina_exe=vina_exe,
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=lig_pdbqt,
                box=box,
                params=params,
                out_root=vina_out_root,
                seeds=seeds,
                case_id=case_id,
                receptor_type=receptor_mode,
            )
            case_status["steps"]["vina_out"] = {"path": str(vina_out_root)}
            case_status["steps"]["analysis"] = {"n_runs": len(runs)}

            all_scores: List[float] = []
            n_ok = 0
            for r in runs:
                outp = Path(r.out_pdbqt)
                if r.returncode == 0 and outp.exists() and outp.stat().st_size > 0:
                    scores = parse_vina_scores_from_out_pdbqt(outp)
                    if scores:
                        all_scores.append(min(scores))
                        n_ok += 1

            best = min(all_scores) if all_scores else None

            summary.append({
                "case_id": case_id,
                "receptor_mode": receptor_mode,
                "ligand_resname": ligand_resname,
                "n_runs": len(runs),
                "n_ok": n_ok,
                "best_score": (float(best) if best is not None else ""),
            })

            case_status["ok"] = True

        except Exception as e:
            case_status["ok"] = False
            case_status["error"] = {"type": type(e).__name__, "message": str(e)}
            if strict:
                status_path.write_text(json.dumps(case_status, indent=2), encoding="utf-8")
                raise
        finally:
            case_status["runtime_sec"] = float(time.time() - t0_case)
            status_path.write_text(json.dumps(case_status, indent=2), encoding="utf-8")

    # --- reports ---
    summary_csv = dirs["reports"] / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "receptor_mode", "ligand_resname", "n_runs", "n_ok", "best_score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary:
            w.writerow(r)

    aggregate_json = dirs["reports"] / "aggregate.json"
    aggregate_json.write_text(json.dumps({"n_cases": len(rows), "summary": summary}, indent=2), encoding="utf-8")

    return {
        "summary_csv": str(summary_csv),
        "aggregate_json": str(aggregate_json),
        "n_cases": str(len(rows)),
    }
