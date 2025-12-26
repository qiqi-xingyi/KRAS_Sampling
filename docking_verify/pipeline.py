# --*-- conding:utf-8 --*--
# @time:12/25/25 01:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

# docking_verify/pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import csv
import json
import subprocess
import time

from .vina import run_vina_multi_seed, parse_vina_scores_from_out_pdbqt
from .template_fit import (
    read_pdb_atoms, write_pdb, PDBAtom,
    build_hybrid_receptor_by_template_fit,
)


# -------------------------
# Data structures
# -------------------------

@dataclass
class VinaParams:
    exhaustiveness: int = 16
    num_modes: int = 20
    energy_range: int = 3
    cpu: int = 8
    seeds: Optional[List[int]] = None


@dataclass
class DockBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


# -------------------------
# Small utilities
# -------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]


def _run_cmd(cmd: List[str], timeout_sec: Optional[int] = None) -> Tuple[int, str, str, float]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    dt = time.time() - t0
    return int(proc.returncode), proc.stdout or "", proc.stderr or "", float(dt)


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
        r = a.resname.upper()
        rec = a.record.strip()

        if rec == "ATOM":
            out.append(a)
            continue

        if rec == "HETATM":
            # drop ligand
            if r == lig:
                continue
            # drop water
            if remove_water and r in ("HOH", "WAT", "H2O"):
                continue
            # keep selected ions/cofactors if requested
            if keep_het and r in keep_het:
                out.append(a)
                continue
            # default: drop other HETATM
            continue

    return out


def _extract_ligand_atoms(atoms: List[PDBAtom], ligand_resname: str) -> List[PDBAtom]:
    lig = ligand_resname.upper()
    return [a for a in atoms if (a.record.strip() == "HETATM" and a.resname.upper() == lig)]


def _centroid(atoms: List[PDBAtom]) -> Tuple[float, float, float]:
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    zs = [a.z for a in atoms]
    if not xs:
        raise RuntimeError("Empty atom list for centroid.")
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


# -------------------------
# PDBQT sanitize (CRITICAL)
# -------------------------

_BAD_PDBQT_PREFIX = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")

def _sanitize_pdbqt_drop_torsion_tree(pdbqt_path: Union[str, Path]) -> None:
    """
    Vina 1.2.5 (your build) rejects ROOT/BRANCH tags for receptor and may reject
    them for ligands as well (depending on parsing mode).
    We force-remove these torsion tree keywords from BOTH receptor and ligand PDBQT.

    This makes the ligand effectively rigid, which is acceptable for your goal
    (validation that predicted structure still forms a meaningful pocket / docking site).
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    kept = []
    for l in lines:
        if l.lstrip().startswith(_BAD_PDBQT_PREFIX):
            continue
        kept.append(l)
    p.write_text("\n".join(kept) + "\n", encoding="utf-8")


# -------------------------
# OpenBabel conversion
# -------------------------

def _obabel_to_pdbqt_receptor(
    obabel_exe: str,
    receptor_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
) -> Path:
    receptor_pdb = str(Path(receptor_pdb).expanduser().resolve())
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    # -xr is important: force rigid receptor output
    cmd = [
        str(obabel_exe),
        receptor_pdb,
        "-O", str(out_pdbqt),
        "-xr",
        "-h",
        "--partialcharge", "gasteiger",
    ]
    rc, so, se, _dt = _run_cmd(cmd)

    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            "OpenBabel receptor conversion failed (empty output).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    # Always sanitize (important even for resume workflows)
    _sanitize_pdbqt_drop_torsion_tree(out_pdbqt)
    return out_pdbqt


def _obabel_to_pdbqt_ligand(
    obabel_exe: str,
    ligand_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
) -> Path:
    ligand_pdb = str(Path(ligand_pdb).expanduser().resolve())
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    cmd = [
        str(obabel_exe),
        ligand_pdb,
        "-O", str(out_pdbqt),
        "-h",
        "--partialcharge", "gasteiger",
    ]
    rc, so, se, _dt = _run_cmd(cmd)

    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            "OpenBabel ligand conversion failed (empty output).\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    # Always sanitize ligand too (THIS is what fixes your ROOT error)
    _sanitize_pdbqt_drop_torsion_tree(out_pdbqt)
    return out_pdbqt


# -------------------------
# Main pipeline
# -------------------------

def run_pipeline(
    cases_csv: Union[str, Path],
    out_dir: Union[str, Path],
    vina_exe: str,
    obabel_exe: str,
    receptor_mode: str = "hybrid_templatefit",  # "crystal" or "hybrid_templatefit"
    ligand_mode: str = "from_crystal",          # only "from_crystal" in this minimal pipeline
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    box_size: Sequence[float] = (20.0, 20.0, 20.0),
    vina_params: Optional[VinaParams] = None,
    keep_het_resnames_in_receptor: Optional[Sequence[str]] = None,
    remove_water: bool = True,
    resume: bool = True,
    strict: bool = False,
) -> Dict[str, str]:
    if ligand_mode != "from_crystal":
        raise RuntimeError("This minimal pipeline only supports ligand_mode='from_crystal'.")

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

    params = vina_params or VinaParams()
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
            "vina_params": {
                "exhaustiveness": int(params.exhaustiveness),
                "num_modes": int(params.num_modes),
                "energy_range": int(params.energy_range),
                "cpu": int(params.cpu),
            },
            "steps": {},
            "ok": False,
            "error": None,
        }

        t0_case = time.time()

        try:
            # Read original pdb
            atoms_all, other_all = read_pdb_atoms(pdb_path)

            # ---------------------
            # Ligand: extract from crystal
            # ---------------------
            lig_atoms = _extract_ligand_atoms(atoms_all, ligand_resname=ligand_resname)
            if not lig_atoms:
                raise RuntimeError(f"{case_id}: ligand {ligand_resname} not found in crystal PDB.")

            lig_pdb = dirs["ligands"] / f"{case_id}.{ligand_resname}.pdb"
            # IMPORTANT: write minimal, avoid CONECT etc.
            write_pdb(lig_pdb, lig_atoms, other_lines=None, drop_conect=True)

            lig_pdbqt = dirs["pdbqt_lig"] / f"{case_id}.{ligand_resname}.pdbqt"

            # If resume and exists: still sanitize to remove ROOT
            if lig_pdbqt.exists() and lig_pdbqt.stat().st_size > 0:
                _sanitize_pdbqt_drop_torsion_tree(lig_pdbqt)
            else:
                _obabel_to_pdbqt_ligand(obabel_exe, lig_pdb, lig_pdbqt)

            case_status["steps"]["ligand_pdb"] = {"path": str(lig_pdb)}
            case_status["steps"]["ligand_pdbqt"] = {"path": str(lig_pdbqt)}

            # Box center from crystal ligand centroid
            cx, cy, cz = _centroid(lig_atoms)
            box = DockBox(center_x=cx, center_y=cy, center_z=cz, size_x=sx, size_y=sy, size_z=sz)
            case_status["steps"]["box"] = {"center": [cx, cy, cz], "size": [sx, sy, sz]}

            # ---------------------
            # Receptor base: crystal minus ligand/water/etc.
            # ---------------------
            rec_atoms = _strip_receptor_atoms(
                atoms_all,
                ligand_resname=ligand_resname,
                remove_water=remove_water,
                keep_het_resnames=keep_het_resnames_in_receptor,
            )
            receptor_base_pdb = dirs["receptors"] / f"{case_id}.receptor_base.pdb"

            # IMPORTANT: keep it minimal to avoid OpenBabel parse issues
            write_pdb(receptor_base_pdb, rec_atoms, other_lines=None, drop_conect=True)

            receptor_pdb = receptor_base_pdb

            # ---------------------
            # Hybrid (optional): template_fit replacement
            # ---------------------
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

            # ---------------------
            # Receptor PDBQT (rigid)
            # ---------------------
            receptor_pdbqt = dirs["pdbqt_rec"] / f"{case_id}.pdbqt"

            # If resume and exists: still sanitize to remove ROOT/BRANCH if any
            if receptor_pdbqt.exists() and receptor_pdbqt.stat().st_size > 0:
                _sanitize_pdbqt_drop_torsion_tree(receptor_pdbqt)
            else:
                _obabel_to_pdbqt_receptor(obabel_exe, receptor_pdb, receptor_pdbqt)

            case_status["steps"]["receptor_pdbqt"] = {"path": str(receptor_pdbqt)}

            # ---------------------
            # Run Vina
            # ---------------------
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

            # Parse best score among all seeds
            all_scores: List[float] = []
            n_ok = 0
            for r in runs:
                if r.returncode == 0 and Path(r.out_pdbqt).exists() and Path(r.out_pdbqt).stat().st_size > 0:
                    scores = parse_vina_scores_from_out_pdbqt(r.out_pdbqt)
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

    # ---------------------
    # Reports
    # ---------------------
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
