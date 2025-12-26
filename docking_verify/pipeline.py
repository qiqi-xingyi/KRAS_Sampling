# --*-- conding:utf-8 --*--
# @time:12/25/25 01:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

from __future__ import annotations

import csv
import json
import time
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .pdbqt import (
    parse_pdb_atoms,
    write_pdb,
    AtomRecord,
    kabsch,
    write_ca_trace_pdb,
    graft_fragment_allatom_into_crystal,
)
from .meeko_pdbqt import prepare_receptor_pdbqt_meeko, MeekoError
from .rebuild_allatom import rebuild_allatom_from_ca, RebuildConfig, RebuildError


# =========================
# Errors
# =========================
class PipelineError(RuntimeError):
    pass


class OpenBabelError(RuntimeError):
    pass


class VinaError(RuntimeError):
    pass


# =========================
# Data models
# =========================
@dataclass
class VinaParams:
    exhaustiveness: int = 16
    num_modes: int = 20
    energy_range: int = 3
    cpu: int = 8
    seeds: Optional[List[int]] = None  # if provided overrides global seeds


@dataclass
class DockCase:
    case_id: str
    pdb_path: str
    chain_id: str
    start_resi: int
    end_resi: int
    sequence: str

    # for hybrid_allatom
    decoded_file: Optional[str] = None
    line_index: Optional[int] = None
    scale_factor: Optional[float] = None

    # ligand
    ligand_resname: str = "GDP"


# =========================
# Helpers
# =========================
def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl_at_index(path: Union[str, Path], line_index: int) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == int(line_index):
                return json.loads(line)
    raise PipelineError(f"line_index={line_index} not found in jsonl: {path}")


def load_predicted_main_positions(decoded_jsonl: str, line_index: int) -> List[List[float]]:
    obj = _read_jsonl_at_index(decoded_jsonl, line_index)
    if "main_positions" not in obj:
        raise PipelineError("decoded.jsonl line does not contain 'main_positions'")
    return obj["main_positions"]


def extract_fragment_ca_from_crystal(
    pdb_path: Union[str, Path],
    chain_id: str,
    start_resi: int,
    end_resi: int,
) -> List[List[float]]:
    atoms, _ = parse_pdb_atoms(Path(pdb_path), keep_hetatm=True)
    ca: List[List[float]] = []
    for a in atoms:
        if a.record != "ATOM":
            continue
        if a.chain_id != chain_id:
            continue
        if not (start_resi <= int(a.resseq) <= end_resi):
            continue
        if a.name.strip() == "CA":
            ca.append([float(a.x), float(a.y), float(a.z)])
    return ca


def write_receptor_only_pdb(
    in_pdb: Union[str, Path],
    out_pdb: Union[str, Path],
    keep_metals: bool = False,
) -> Path:
    """
    Create receptor-only PDB for Meeko:
    - keep only protein ATOM lines
    - optionally keep metal ions (HETATM) like MG, ZN
    - drop everything else
    - drop other_lines to avoid CONECT issues
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    atoms, _ = parse_pdb_atoms(in_pdb, keep_hetatm=True)
    kept: List[AtomRecord] = []
    for a in atoms:
        if a.record == "ATOM":
            kept.append(a)
        elif keep_metals and a.record == "HETATM" and a.resname.strip() in {"MG", "MN", "ZN", "CA", "NA", "K"}:
            kept.append(a)

    for i, a in enumerate(kept, start=1):
        a.serial = i

    write_pdb(out_pdb, atoms=kept, other_lines=None)
    return out_pdb


def extract_ligand_pdb_from_crystal(
    crystal_pdb: Union[str, Path],
    ligand_resname: str,
    out_pdb: Union[str, Path],
) -> Tuple[Path, List[List[float]]]:
    """
    Extract ligand by resname from crystal PDB (HETATM). Returns ligand PDB path and coords.
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    atoms, _ = parse_pdb_atoms(crystal_pdb, keep_hetatm=True)
    lig: List[AtomRecord] = []
    coords: List[List[float]] = []

    for a in atoms:
        if a.record != "HETATM":
            continue
        if a.resname.strip() != ligand_resname.strip():
            continue
        lig.append(a)
        coords.append([float(a.x), float(a.y), float(a.z)])

    if not lig:
        raise PipelineError(f"Ligand {ligand_resname} not found in crystal PDB: {crystal_pdb}")

    # renumber
    for i, a in enumerate(lig, start=1):
        a.serial = i

    write_pdb(out_pdb, atoms=lig, other_lines=None)
    return out_pdb, coords


def centroid(xyz: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    arr = np.array(xyz, dtype=float)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def run_obabel_pdb_to_pdbqt(
    obabel_exe: str,
    in_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    log_file: Union[str, Path],
) -> Path:
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    log_file = Path(log_file).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(obabel_exe),
        str(in_pdb),
        "-O",
        str(out_pdbqt),
        "-h",
        "--partialcharge",
        "gasteiger",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with log_file.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n=== [stderr] ===\n")
            f.write(proc.stderr)

    if proc.returncode != 0:
        raise OpenBabelError(f"OpenBabel failed (rc={proc.returncode}). See log: {log_file}")

    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(f"OpenBabel produced no pdbqt: {out_pdbqt}. See log: {log_file}")

    return out_pdbqt


def parse_vina_affinities(stdout: str) -> List[float]:
    """
    Parse affinities from Vina stdout table.
    Lines look like:
      1       -7.5      0.000      0.000
    """
    aff: List[float] = []
    for line in (stdout or "").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                aff.append(float(parts[1]))
            except Exception:
                pass
    return aff


def run_vina_once(
    vina_exe: str,
    receptor_pdbqt: Union[str, Path],
    ligand_pdbqt: Union[str, Path],
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float],
    params: VinaParams,
    seed: int,
    out_pdbqt: Union[str, Path],
    log_file: Union[str, Path],
) -> Dict[str, Any]:
    receptor_pdbqt = str(Path(receptor_pdbqt).expanduser().resolve())
    ligand_pdbqt = str(Path(ligand_pdbqt).expanduser().resolve())
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    log_file = Path(log_file).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cx, cy, cz = center
    sx, sy, sz = box_size

    cmd = [
        str(vina_exe),
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", f"{cx:.6f}",
        "--center_y", f"{cy:.6f}",
        "--center_z", f"{cz:.6f}",
        "--size_x", f"{sx:.6f}",
        "--size_y", f"{sy:.6f}",
        "--size_z", f"{sz:.6f}",
        "--exhaustiveness", str(int(params.exhaustiveness)),
        "--num_modes", str(int(params.num_modes)),
        "--energy_range", str(int(params.energy_range)),
        "--cpu", str(int(params.cpu)),
        "--seed", str(int(seed)),
        "--out", str(out_pdbqt),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.time() - t0

    so, se = proc.stdout or "", proc.stderr or ""
    aff = parse_vina_affinities(so)

    with log_file.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        if so:
            f.write(so)
        if se:
            f.write("\n=== [stderr] ===\n")
            f.write(se)

    if proc.returncode != 0:
        raise VinaError(f"Vina failed (rc={proc.returncode}). See log: {log_file}")

    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise VinaError(f"Vina produced no output: {out_pdbqt}. See log: {log_file}")

    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "runtime_sec": float(dt),
        "out_pdbqt": str(out_pdbqt),
        "log_file": str(log_file),
        "affinities": aff,
        "best_affinity": min(aff) if aff else None,
        "stdout_tail": so[-1200:],
        "stderr_tail": se[-1200:],
    }


def read_cases_csv(cases_csv: Union[str, Path]) -> List[DockCase]:
    cases_csv = Path(cases_csv).expanduser().resolve()
    with cases_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cases: List[DockCase] = []
    for r in rows:
        # required
        case_id = r["case_id"]
        pdb_path = r["pdb_path"]
        chain_id = r.get("chain_id", "A")
        start_resi = int(r["start_resi"])
        end_resi = int(r["end_resi"])
        sequence = r["sequence"]

        # optional for hybrid_allatom
        decoded_file = r.get("decoded_file") or None
        line_index = int(r["line_index"]) if r.get("line_index") not in (None, "", "None") else None
        scale_factor = float(r["scale_factor"]) if r.get("scale_factor") not in (None, "", "None") else None

        ligand_resname = (r.get("ligand_resname") or "GDP").strip()

        cases.append(
            DockCase(
                case_id=case_id,
                pdb_path=pdb_path,
                chain_id=chain_id,
                start_resi=start_resi,
                end_resi=end_resi,
                sequence=sequence,
                decoded_file=decoded_file,
                line_index=line_index,
                scale_factor=scale_factor,
                ligand_resname=ligand_resname,
            )
        )
    return cases


# =========================
# Core pipeline
# =========================
def run_pipeline(
    case: DockCase,
    out_dir: Union[str, Path],
    receptor_mode: str,
    ligand_mode: str,
    vina_exe: str,
    obabel_exe: str,
    seeds: List[int],
    box_size: Tuple[float, float, float],
    vina_params: VinaParams,
    resume: bool = True,
    strict: bool = False,
    pulchra_exe: str = "pulchra",
    scwrl_exe: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single case. Writes:
      out_dir/status/<case_id>.json
      out_dir/receptors/...
      out_dir/pdbqt/...
      out_dir/vina_out/<case_id>/seed_x/...
    """
    out_dir = _ensure_dir(out_dir)
    status_dir = _
