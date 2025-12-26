# --*-- conding:utf-8 --*--
# @time:12/25/25 01:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

from __future__ import annotations

import csv
import json
import time
import traceback
import subprocess
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from .pdbqt import (
    AtomRecord,
    parse_pdb_atoms,
    write_pdb,
    kabsch,
    write_ca_trace_pdb,
    graft_fragment_allatom_into_crystal,
)

from .rebuild_allatom import rebuild_allatom_from_ca, RebuildConfig, RebuildError
from .meeko_pdbqt import prepare_receptor_pdbqt_meeko, MeekoError
from .vina import run_vina_once


# -----------------------------
# Errors
# -----------------------------
class PipelineError(RuntimeError):
    pass


class OpenBabelError(RuntimeError):
    pass


# -----------------------------
# Data models
# -----------------------------
@dataclass
class VinaParams:
    exhaustiveness: int = 16
    num_modes: int = 20
    energy_range: int = 3
    cpu: int = 8
    seeds: Optional[List[int]] = None


@dataclass
class Case:
    case_id: str
    pdb_path: str
    ref_pdb: Optional[str]
    chain_id: str
    start_resi: int
    end_resi: int
    ligand_resname: str
    receptor_mode: str
    ligand_mode: str
    seeds: List[int]
    vina_params: VinaParams

    # For hybrid_allatom reconstruction
    decoded_file: Optional[str] = None
    line_index: Optional[int] = None
    scale_factor: Optional[float] = None
    sequence: Optional[str] = None


# -----------------------------
# Small utils
# -----------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_jsonl_line(path: Union[str, Path], line_index: int) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == int(line_index):
                return json.loads(line)
    raise PipelineError(f"line_index not found: {p} @ {line_index}")


def _tail(s: str, n: int = 2000) -> str:
    return s[-n:] if s else ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _pick(row: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


# -----------------------------
# PDB ligand extraction
# -----------------------------
def extract_ligand_from_pdb(
    pdb_path: Union[str, Path],
    ligand_resname: str,
    out_pdb: Union[str, Path],
    allow_any_chain: bool = True,
) -> Path:
    """
    Extract HETATM records for given ligand_resname into a standalone PDB.
    """
    pdb_path = Path(pdb_path).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)

    atoms, _ = parse_pdb_atoms(pdb_path, keep_hetatm=True)

    lig = []
    for a in atoms:
        if a.record != "HETATM":
            continue
        if a.resname.strip().upper() != ligand_resname.strip().upper():
            continue
        lig.append(a)

    if not lig:
        raise PipelineError(f"Ligand {ligand_resname} not found in PDB: {pdb_path}")

    # Normalize: set chain to 'L' (optional but keeps things clean)
    for i, a in enumerate(lig, start=1):
        a.serial = i
        if allow_any_chain:
            a.chain_id = "L"

    write_pdb(out_pdb, atoms=lig, other_lines=None)
    return out_pdb


def ligand_center_from_pdb(pdb_path: Union[str, Path]) -> Tuple[float, float, float]:
    p = Path(pdb_path).expanduser().resolve()
    atoms, _ = parse_pdb_atoms(p, keep_hetatm=True)
    xyz = []
    for a in atoms:
        if a.record not in ("ATOM", "HETATM"):
            continue
        xyz.append([a.x, a.y, a.z])
    if not xyz:
        raise PipelineError(f"Empty ligand pdb: {p}")
    arr = np.array(xyz, dtype=float)
    c = arr.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


# -----------------------------
# OpenBabel conversions
# -----------------------------
def obabel_pdb_to_pdbqt(
    obabel_exe: str,
    in_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    add_h: bool = True,
    gasteiger: bool = True,
) -> Path:
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    cmd = [str(obabel_exe), str(in_pdb), "-O", str(out_pdbqt)]
    if add_h:
        cmd.append("-h")
    if gasteiger:
        cmd += ["--partialcharge", "gasteiger"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # OpenBabel sometimes returns 0 but outputs nothing.
    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel conversion produced empty output.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {proc.returncode}\n"
            f"STDOUT(tail): {_tail(proc.stdout)}\n"
            f"STDERR(tail): {_tail(proc.stderr)}\n"
        )

    return out_pdbqt


# -----------------------------
# Receptor-only PDB
# -----------------------------
def build_receptor_only_pdb(
    receptor_pdb: Union[str, Path],
    out_pdb: Union[str, Path],
    drop_ligand_resname: Optional[str] = None,
) -> Path:
    """
    Keep only ATOM lines (protein). Optionally keep selected HETATM (not used here).
    Also drops CONECT/REMARK by writing with other_lines=None.
    """
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)

    atoms, _ = parse_pdb_atoms(receptor_pdb, keep_hetatm=True)

    kept: List[AtomRecord] = []
    for a in atoms:
        if a.record == "ATOM":
            kept.append(a)
        elif a.record == "HETATM":
            if drop_ligand_resname and a.resname.strip().upper() == drop_ligand_resname.strip().upper():
                continue
            # default: drop all hetatm (kept empty)
            continue

    for i, a in enumerate(kept, start=1):
        a.serial = i

    write_pdb(out_pdb, atoms=kept, other_lines=None)
    return out_pdb


# -----------------------------
# Vina log parsing
# -----------------------------
_VINA_TABLE_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", re.MULTILINE)

def parse_vina_scores_from_log(log_path: Union[str, Path]) -> List[Dict[str, float]]:
    """
    Parse Vina stdout/stderr log saved by our runner. Returns list of poses with affinity/rmsd.
    """
    p = Path(log_path).expanduser().resolve()
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in _VINA_TABLE_RE.finditer(text):
        rows.append({
            "mode": float(m.group(1)),
            "affinity": float(m.group(2)),
            "rmsd_lb": float(m.group(3)),
            "rmsd_ub": float(m.group(4)),
        })
    return rows


# -----------------------------
# Cases CSV
# -----------------------------
def load_cases_csv(
    cases_csv: Union[str, Path],
    receptor_mode: str,
    ligand_mode: str,
    seeds: List[int],
    vina_params: VinaParams,
) -> List[Case]:
    cases_csv = Path(cases_csv).expanduser().resolve()
    with cases_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cases: List[Case] = []
    for row in rows:
        case_id = str(_pick(row, "case_id", "fragment_id")).strip()
        pdb_path = str(_pick(row, "pdb_path", "pdb", "ref_pdb_path")).strip()
        ref_pdb = _pick(row, "ref_pdb", default=None)

        chain_id = str(_pick(row, "chain_id", "chain", default="A")).strip() or "A"
        start_resi = _safe_int(_pick(row, "start_resi", "res_start", "start", default=1), 1)
        end_resi = _safe_int(_pick(row, "end_resi", "res_end", "end", default=start_resi), start_resi)

        ligand_resname = str(_pick(row, "ligand_resname", default="")).strip() or "GDP"

        decoded_file = _pick(row, "decoded_file", "decoded_jsonl", default=None)
        line_index = _pick(row, "line_index", default=None)
        scale_factor = _pick(row, "scale_factor", default=None)
        sequence = _pick(row, "sequence", "best_sequence", "main_chain_residue_seq", default=None)

        c = Case(
            case_id=case_id,
            pdb_path=pdb_path,
            ref_pdb=ref_pdb,
            chain_id=chain_id,
            start_resi=int(start_resi),
            end_resi=int(end_resi),
            ligand_resname=ligand_resname,
            receptor_mode=receptor_mode,
            ligand_mode=ligand_mode,
            seeds=list(seeds),
            vina_params=vina_params,
            decoded_file=str(decoded_file) if decoded_file else None,
            line_index=int(line_index) if line_index not in (None, "") else None,
            scale_factor=float(scale_factor) if scale_factor not in (None, "") else None,
            sequence=str(sequence) if sequence else None,
        )
        cases.append(c)

    return cases


# -----------------------------
# Core per-case runner
# -----------------------------
def run_case(
    c: Case,
    out_dir: Path,
    vina_exe: str,
    obabel_exe: str,
    mk_prepare_receptor_exe: str,
    pulchra_exe: str,
    scwrl_exe: Optional[str],
    box_size: Tuple[float, float, float],
    resume: bool,
    strict: bool,
) -> Dict[str, Any]:
    """
    Run one case end-to-end, returning case_status dict (JSON-serializable).
    """
    t0 = time.time()
    out_dir = out_dir.expanduser().resolve()

    case_status: Dict[str, Any] = {
        "case_id": c.case_id,
        "ref_pdb": c.ref_pdb,
        "pdb_path": str(Path(c.pdb_path).expanduser().resolve()),
        "chain_id": c.chain_id,
        "start_resi": c.start_resi,
        "end_resi": c.end_resi,
        "ligand_resname": c.ligand_resname,
        "receptor_mode": c.receptor_mode,
        "ligand_mode": c.ligand_mode,
        "seeds": c.seeds,
        "vina_params": asdict(c.vina_params),
        "steps": {},
        "ok": False,
        "error": None,
        "runtime_sec": None,
    }

    # Directories
    receptors_dir = _ensure_dir(out_dir / "receptors")
    rebuild_dir = _ensure_dir(out_dir / "rebuild")
    pdbqt_dir = _ensure_dir(out_dir / "pdbqt")
    vina_out_dir = _ensure_dir(out_dir / "vina_out")

    # -------------------------
    # Step A: Build receptor PDB
    # -------------------------
    if c.receptor_mode == "crystal":
        receptor_pdb = receptors_dir / "crystal" / f"{c.case_id}.pdb"
        receptor_pdb.parent.mkdir(parents=True, exist_ok=True)
        if (not resume) or (not receptor_pdb.exists()):
            receptor_pdb.write_text(
                Path(c.pdb_path).expanduser().resolve().read_text(encoding="utf-8", errors="ignore"),
                encoding="utf-8",
            )
        case_status["steps"]["receptor_pdb"] = {"path": str(receptor_pdb), "mode": "crystal"}

    elif c.receptor_mode == "hybrid_allatom":
        # Validate fields
        if not (c.decoded_file and c.line_index is not None and c.scale_factor is not None and c.sequence):
            raise PipelineError(
                f"hybrid_allatom requires decoded_file/line_index/scale_factor/sequence. case={c.case_id}"
            )

        # 1) Extract crystal fragment CA as alignment target
        cryst_atoms, _ = parse_pdb_atoms(Path(c.pdb_path).expanduser().resolve(), keep_hetatm=True)
        cryst_ca: List[List[float]] = []
        for a in cryst_atoms:
            if a.record == "ATOM" and a.chain_id == c.chain_id and c.start_resi <= a.resseq <= c.end_resi and a.name.strip() == "CA":
                cryst_ca.append([a.x, a.y, a.z])

        n = c.end_resi - c.start_resi + 1
        if len(cryst_ca) != n:
            raise PipelineError(f"Crystal CA mismatch: expected {n}, got {len(cryst_ca)} for {c.case_id}")

        # 2) Read predicted CA (lattice units) and scale to Å
        obj = _read_jsonl_line(c.decoded_file, int(c.line_index))
        pred_main_positions = obj.get("main_positions", None)
        if pred_main_positions is None:
            raise PipelineError(f"decoded.jsonl missing main_positions: {c.decoded_file}@{c.line_index}")
        P = np.array(pred_main_positions, dtype=float) * float(c.scale_factor)
        Q = np.array(cryst_ca, dtype=float)

        if P.shape != Q.shape:
            raise PipelineError(f"Pred/Crystal shape mismatch: {P.shape} vs {Q.shape} in {c.case_id}")

        # 3) Kabsch align predicted CA onto crystal CA
        R, t = kabsch(P, Q)
        aligned = (R @ P.T).T + t

        # 4) Write CA-only trace PDB
        ca_trace_dir = _ensure_dir(rebuild_dir / "ca_trace")
        ca_pdb = ca_trace_dir / f"{c.case_id}.ca.pdb"
        if (not resume) or (not ca_pdb.exists()):
            write_ca_trace_pdb(
                out_pdb=ca_pdb,
                chain_id=c.chain_id,
                res_start=c.start_resi,
                sequence=c.sequence,
                ca_xyz=aligned.tolist(),
            )

        # 5) Rebuild all-atom fragment via PULCHRA (+ optional SCWRL)
        frag_rebuild_dir = _ensure_dir(rebuild_dir / "allatom" / c.case_id)
        rebuilt_pdb = frag_rebuild_dir / "rebuilt_allatom.pdb"

        if (not resume) or (not rebuilt_pdb.exists()):
            cfg = RebuildConfig(
                pulchra_exe=pulchra_exe,
                scwrl_exe=scwrl_exe,
                strict=True if strict else False,
            )
            rebuilt = rebuild_allatom_from_ca(ca_pdb=ca_pdb, out_dir=frag_rebuild_dir, cfg=cfg)
            rebuilt_pdb = Path(rebuilt.allatom_pdb).expanduser().resolve()

        # 6) Graft fragment back into crystal to form full all-atom hybrid receptor
        receptor_pdb = receptors_dir / "hybrid_allatom" / f"{c.case_id}.pdb"
        receptor_pdb.parent.mkdir(parents=True, exist_ok=True)
        if (not resume) or (not receptor_pdb.exists()):
            graft_fragment_allatom_into_crystal(
                crystal_pdb=Path(c.pdb_path).expanduser().resolve(),
                fragment_allatom_pdb=rebuilt_pdb,
                chain_id=c.chain_id,
                res_start=c.start_resi,
                res_end=c.end_resi,
                out_pdb=receptor_pdb,
                keep_hetatm_outside=True,
            )

        case_status["steps"]["receptor_pdb"] = {
            "path": str(receptor_pdb),
            "mode": "hybrid_allatom",
            "ca_pdb": str(ca_pdb),
            "rebuilt_fragment": str(rebuilt_pdb),
        }

    else:
        raise PipelineError(f"Unknown receptor_mode: {c.receptor_mode}")

    # -------------------------
    # Step B: receptor-only PDB (protein only)
    # -------------------------
    receptor_only_dir = _ensure_dir(receptors_dir / "receptor_only")
    receptor_only_pdb = receptor_only_dir / f"{c.case_id}.receptor.pdb"
    if (not resume) or (not receptor_only_pdb.exists()):
        build_receptor_only_pdb(
            receptor_pdb=Path(case_status["steps"]["receptor_pdb"]["path"]),
            out_pdb=receptor_only_pdb,
            drop_ligand_resname=c.ligand_resname,
        )
    case_status["steps"]["receptor_only_pdb"] = {"path": str(receptor_only_pdb)}

    # -------------------------
    # Step C: receptor PDBQT (Meeko)
    # -------------------------
    pdbqt_receptor_dir = _ensure_dir(pdbqt_dir / "receptors")
    meeko_base = pdbqt_receptor_dir / c.case_id  # Meeko adds _rigid.pdbqt
    receptor_pdbqt = pdbqt_receptor_dir / f"{c.case_id}_rigid.pdbqt"

    if (not resume) or (not receptor_pdbqt.exists()):
        res = prepare_receptor_pdbqt_meeko(
            receptor_pdb=receptor_only_pdb,
            out_basename=meeko_base,
            mk_prepare_receptor_exe=mk_prepare_receptor_exe,
            default_altloc="A",
            write_json=False,
            write_gpf=False,
        )
        receptor_pdbqt = Path(res.rigid_pdbqt).expanduser().resolve()

    # sanity: rigid receptor must NOT contain ROOT
    txt = receptor_pdbqt.read_text(encoding="utf-8", errors="ignore")
    if "ROOT" in txt:
        raise PipelineError(f"Receptor pdbqt still contains ROOT: {receptor_pdbqt}")

    case_status["steps"]["receptor_pdbqt"] = {"path": str(receptor_pdbqt), "backend": "meeko"}

    # -------------------------
    # Step D: ligand PDB + PDBQT
    # -------------------------
    pdbqt_ligand_dir = _ensure_dir(pdbqt_dir / "ligands")
    lig_key = f"{Path(c.pdb_path).stem}_{c.ligand_resname}_any".lower()
    ligand_pdb = pdbqt_ligand_dir / f"{lig_key}.pdb"
    ligand_pdbqt = pdbqt_ligand_dir / f"{lig_key}.pdbqt"

    if c.ligand_mode == "from_crystal":
        if (not resume) or (not ligand_pdb.exists()):
            extract_ligand_from_pdb(
                pdb_path=Path(c.pdb_path).expanduser().resolve(),
                ligand_resname=c.ligand_resname,
                out_pdb=ligand_pdb,
                allow_any_chain=True,
            )
        if (not resume) or (not ligand_pdbqt.exists()):
            obabel_pdb_to_pdbqt(
                obabel_exe=obabel_exe,
                in_pdb=ligand_pdb,
                out_pdbqt=ligand_pdbqt,
                add_h=True,
                gasteiger=True,
            )
    else:
        raise PipelineError(f"Unknown ligand_mode: {c.ligand_mode}")

    # sanity: ligand pdbqt should contain ROOT (usually)
    lig_txt = ligand_pdbqt.read_text(encoding="utf-8", errors="ignore")
    if "ATOM" not in lig_txt:
        raise PipelineError(f"Ligand pdbqt looks empty/invalid: {ligand_pdbqt}")

    case_status["steps"]["ligand_pdbqt"] = {
        "path": str(ligand_pdbqt),
        "mode": c.ligand_mode,
        "key": lig_key,
        "ligand_resname": c.ligand_resname,
    }

    # -------------------------
    # Step E: docking box from ligand center
    # -------------------------
    cx, cy, cz = ligand_center_from_pdb(ligand_pdb)
    sx, sy, sz = float(box_size[0]), float(box_size[1]), float(box_size[2])
    case_status["steps"]["box"] = {"center": [cx, cy, cz], "size": [sx, sy, sz]}

    # -------------------------
    # Step F: run Vina multiple seeds
    # -------------------------
    case_vina_dir = _ensure_dir(vina_out_dir / c.case_id)
    case_status["steps"]["vina_out"] = {"path": str(case_vina_dir)}
    runs: List[Dict[str, Any]] = []

    for seed in c.seeds:
        seed_dir = _ensure_dir(case_vina_dir / f"seed_{seed}")
        out_pdbqt = seed_dir / "out.pdbqt"
        log_file = seed_dir / "log.txt"

        if resume and out_pdbqt.exists() and out_pdbqt.stat().st_size > 0 and log_file.exists():
            pass
        else:
            r = run_vina_once(
                vina_exe=vina_exe,
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                center=(cx, cy, cz),
                size=(sx, sy, sz),
                params=c.vina_params,
                out_pdbqt=out_pdbqt,
                log_file=log_file,
                seed=int(seed),
            )

            # hard checks
            if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
                raise PipelineError(
                    "Vina produced empty out.pdbqt.\n"
                    f"case={c.case_id} seed={seed}\n"
                    f"log={log_file}\n"
                )

            if r.get("returncode", 0) != 0:
                raise PipelineError(
                    "Vina returncode != 0.\n"
                    f"case={c.case_id} seed={seed}\n"
                    f"log={log_file}\n"
                    f"stderr_tail={r.get('stderr_tail','')}\n"
                )

        scores = parse_vina_scores_from_log(log_file)
        best_aff = scores[0]["affinity"] if scores else None
        runs.append({
            "seed": int(seed),
            "out_pdbqt": str(out_pdbqt),
            "log_file": str(log_file),
            "best_affinity": best_aff,
            "n_modes": len(scores),
        })

    case_status["steps"]["analysis"] = {"n_runs": len(runs), "runs": runs}

    case_status["ok"] = True
    case_status["runtime_sec"] = float(time.time() - t0)
    return case_status


# -----------------------------
# Public API
# -----------------------------
def run_pipeline(
    cases_csv: Union[str, Path],
    out_dir: Union[str, Path],
    receptor_mode: str = "hybrid_allatom",
    ligand_mode: str = "from_crystal",
    vina_exe: str = "vina",
    obabel_exe: str = "obabel",
    mk_prepare_receptor_exe: str = "mk_prepare_receptor.py",
    pulchra_exe: str = "pulchra",
    scwrl_exe: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    vina_params: Optional[VinaParams] = None,
    resume: bool = True,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end pipeline. Writes per-case status JSON under out_dir/status/.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if vina_params is None:
        vina_params = VinaParams()

    # Load cases
    cases = load_cases_csv(
        cases_csv=cases_csv,
        receptor_mode=receptor_mode,
        ligand_mode=ligand_mode,
        seeds=seeds,
        vina_params=vina_params,
    )

    status_dir = _ensure_dir(out_dir / "status")
    reports_dir = _ensure_dir(out_dir / "reports")

    all_status: List[Dict[str, Any]] = []
    t0 = time.time()

    for c in cases:
        st_path = status_dir / f"{c.case_id}.json"
        if resume and st_path.exists():
            try:
                old = json.loads(st_path.read_text(encoding="utf-8"))
                if old.get("ok", False):
                    all_status.append(old)
                    continue
            except Exception:
                pass

        try:
            st = run_case(
                c=c,
                out_dir=out_dir,
                vina_exe=vina_exe,
                obabel_exe=obabel_exe,
                mk_prepare_receptor_exe=mk_prepare_receptor_exe,
                pulchra_exe=pulchra_exe,
                scwrl_exe=scwrl_exe,
                box_size=box_size,
                resume=resume,
                strict=strict,
            )
        except Exception as e:
            st = {
                "case_id": c.case_id,
                "ref_pdb": c.ref_pdb,
                "pdb_path": c.pdb_path,
                "chain_id": c.chain_id,
                "start_resi": c.start_resi,
                "end_resi": c.end_resi,
                "ligand_resname": c.ligand_resname,
                "receptor_mode": receptor_mode,
                "ligand_mode": ligand_mode,
                "seeds": seeds,
                "vina_params": asdict(vina_params),
                "steps": {},
                "ok": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
                "runtime_sec": None,
            }

        st_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        all_status.append(st)

    # Write a simple summary report
    summary_rows = []
    for st in all_status:
        ok = bool(st.get("ok", False))
        best = None
        if ok:
            runs = st.get("steps", {}).get("analysis", {}).get("runs", [])
            affs = [r.get("best_affinity", None) for r in runs if r.get("best_affinity", None) is not None]
            best = min(affs) if affs else None
        summary_rows.append({
            "case_id": st.get("case_id"),
            "ok": ok,
            "best_affinity": best,
            "error_type": (st.get("error") or {}).get("type") if not ok else "",
        })

    # summary.csv
    summary_csv = reports_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "ok", "best_affinity", "error_type"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    aggregate_json = reports_dir / "aggregate.json"
    aggregate = {
        "cases_csv": str(Path(cases_csv).expanduser().resolve()),
        "out_dir": str(out_dir),
        "receptor_mode": receptor_mode,
        "ligand_mode": ligand_mode,
        "vina_exe": vina_exe,
        "obabel_exe": obabel_exe,
        "mk_prepare_receptor_exe": mk_prepare_receptor_exe,
        "pulchra_exe": pulchra_exe,
        "scwrl_exe": scwrl_exe,
        "seeds": seeds,
        "box_size": box_size,
        "vina_params": asdict(vina_params),
        "resume": resume,
        "strict": strict,
        "n_cases": len(all_status),
        "n_ok": sum(1 for x in all_status if x.get("ok")),
        "runtime_sec": float(time.time() - t0),
    }
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    return {"summary_csv": str(summary_csv), "aggregate_json": str(aggregate_json), "n_cases": len(all_status)}
