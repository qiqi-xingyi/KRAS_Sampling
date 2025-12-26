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
    """
    Group ligand atoms by (chain, resseq, icode). This avoids multi-ligand outputs that
    lead to multi-MODEL PDBQT from OpenBabel.
    """
    lig = ligand_resname.upper()
    groups: Dict[Tuple[str, int, str], List[PDBAtom]] = {}

    for a in atoms:
        if a.record.strip() != "HETATM":
            continue
        if a.resname.upper() != lig:
            continue
        if chain_id is not None and a.chain_id != chain_id:
            continue
        key = (a.chain_id, int(a.resseq), getattr(a, "icode", "") or "")
        groups.setdefault(key, []).append(a)

    return groups


def _pick_single_ligand_instance(
    atoms: List[PDBAtom],
    ligand_resname: str,
    chain_id: Optional[str],
) -> Tuple[List[PDBAtom], str]:
    """
    Pick ONE ligand instance to avoid OpenBabel producing multiple MODEL blocks.
    Strategy:
    - Prefer instances on requested chain_id
    - Pick the instance with the most atoms (more complete)
    - Fallback to first instance if tie
    Returns (atoms, instance_key_str).
    """
    # First try chain-filtered groups
    groups = _group_by_ligand_instance(atoms, ligand_resname, chain_id=chain_id)
    if not groups and chain_id is not None:
        # fallback: ignore chain_id
        groups = _group_by_ligand_instance(atoms, ligand_resname, chain_id=None)

    if not groups:
        return [], ""

    items = list(groups.items())
    # sort by atom count desc, then stable order
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    key, lig_atoms = items[0]
    k_str = f"{key[0]}:{key[1]}{key[2] or ''}"
    return lig_atoms, k_str


def _centroid(atoms: List[PDBAtom]) -> Tuple[float, float, float]:
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    zs = [a.z for a in atoms]
    if not xs:
        raise RuntimeError("Empty atom list for centroid.")
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


# -------------------------
# PDBQT sanitize (CRITICAL for Vina 1.2.5)
# -------------------------

_BAD_TORSION_PREFIX = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
_BAD_MODEL_PREFIX = ("MODEL", "ENDMDL")


def _keep_only_first_model_payload(lines: List[str]) -> List[str]:
    """
    If multi-MODEL exists, keep only payload of first MODEL.
    Remove MODEL/ENDMDL tags.
    """
    has_model = any(l.lstrip().startswith("MODEL") for l in lines)
    if not has_model:
        # also drop stray ENDMDL just in case
        return [l for l in lines if not l.lstrip().startswith(_BAD_MODEL_PREFIX)]

    out: List[str] = []
    model_idx = 0
    in_first = False
    for l in lines:
        s = l.lstrip()
        if s.startswith("MODEL"):
            model_idx += 1
            in_first = (model_idx == 1)
            continue
        if s.startswith("ENDMDL"):
            in_first = False
            continue
        if in_first:
            out.append(l)
    return out


def _fix_atom_name_field_pdbqt(name: str) -> str:
    """
    Atom name in PDBQT should be <=4 chars, alnum only for strict parsing.
    Replace non-alnum with 'P'.
    """
    fixed = "".join((c if c.isalnum() else "P") for c in name.strip())
    if not fixed:
        fixed = "X"
    return fixed[:4]


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _rewrite_atom_line_to_vina125(tokens: List[str]) -> Optional[str]:
    """
    Convert an AD4-style PDBQT ATOM/HETATM line into Vina 1.2.5-friendly format.

    Typical AD4 tokens (split):
      ATOM  1  PC4P  GDP  A  201  13.776  -8.601  -16.173  0.00  0.00  +0.178  C
    We rewrite to:
      ATOM      1 PC4P GDP A 201      13.776  -8.601 -16.173  +0.178 C

    We rely on:
      tokens[0]=ATOM/HETATM
      tokens[1]=serial
      tokens[2]=name
      tokens[3]=resname
      tokens[4]=chain
      tokens[5]=resseq
      tokens[6:9]=x,y,z
      charge=tokens[-2], type=tokens[-1]
    """
    if len(tokens) < 11:
        return None

    rec = tokens[0]
    if rec not in ("ATOM", "HETATM"):
        return None

    # minimal sanity
    if not (tokens[1].isdigit() or tokens[1].lstrip("-").isdigit()):
        return None
    if len(tokens) < 9:
        return None

    serial = int(tokens[1])
    name = _fix_atom_name_field_pdbqt(tokens[2])
    resname = tokens[3]
    chain = tokens[4]
    try:
        resseq = int(tokens[5])
    except Exception:
        return None

    # coords
    if not (_is_float(tokens[6]) and _is_float(tokens[7]) and _is_float(tokens[8])):
        return None
    x, y, z = float(tokens[6]), float(tokens[7]), float(tokens[8])

    # charge/type at end
    if len(tokens) < 2:
        return None
    charge_tok = tokens[-2]
    atype = tokens[-1]

    if not _is_float(charge_tok):
        # sometimes charge token may look like +0.178 (float ok), if not then fail
        return None
    charge = float(charge_tok)

    # Compose a strict-ish line (not perfect fixed columns, but Vina 1.2.5 accepts this style)
    # Keep it stable and clean.
    # Note: we intentionally omit vdW/Elec columns.
    line = (
        f"{rec:<6}"
        f"{serial:>5d} "
        f"{name:<4} "
        f"{resname:>3} "
        f"{chain:1}"
        f"{resseq:>4d}"
        f"    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f} "
        f"{charge:>7.3f} "
        f"{atype}"
    )
    return line


def _sanitize_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Strict sanitizer for BOTH receptor and ligand for Vina 1.2.5:
    - remove ROOT/BRANCH/TORSDOF
    - remove MODEL/ENDMDL and keep only first MODEL payload
    - keep only REMARK + ATOM/HETATM
    - rewrite ATOM/HETATM lines into Vina 1.2.5-friendly format:
        remove the AD4 vdW/Elec fields and keep only charge+type
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    # drop torsion-tree tags early
    tmp: List[str] = []
    for l in raw_lines:
        s = l.lstrip()
        if not s:
            continue
        if s.startswith(_BAD_TORSION_PREFIX):
            continue
        tmp.append(l.rstrip("\n"))

    # model handling
    tmp = _keep_only_first_model_payload(tmp)

    cleaned: List[str] = []
    for l in tmp:
        s = l.lstrip()
        if s.startswith("REMARK"):
            cleaned.append(l.strip("\n"))
            continue
        if s.startswith("ATOM") or s.startswith("HETATM"):
            toks = s.split()
            new_line = _rewrite_atom_line_to_vina125(toks)
            if new_line is not None:
                cleaned.append(new_line)
            # if cannot rewrite, drop line (better than keeping invalid)
            continue
        # drop everything else
        continue

    p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _validate_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Quick validation to catch the common fatal tags before calling Vina.
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"PDBQT missing/empty: {p}")

    bad_hits: List[str] = []
    for l in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = l.lstrip()
        if s.startswith(_BAD_TORSION_PREFIX):
            bad_hits.append(s.split()[0])
        if s.startswith(_BAD_MODEL_PREFIX):
            bad_hits.append(s.split()[0])
    if bad_hits:
        raise RuntimeError(f"PDBQT still contains disallowed tags for Vina 1.2.5: {sorted(set(bad_hits))} in {p}")


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

    cmd = [
        str(obabel_exe),
        receptor_pdb,
        "-O", str(out_pdbqt),
        "-xr",  # rigid receptor
        "-h",
        "--partialcharge", "gasteiger",
    ]
    rc, so, se, _dt = _run_cmd(cmd)

    # OpenBabel sometimes returns 0 but converts 0 molecules; catch it
    if ("0 molecules converted" in (so + se)) or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            "OpenBabel receptor conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    _sanitize_pdbqt_for_vina125(out_pdbqt)
    _validate_pdbqt_for_vina125(out_pdbqt)
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

    if ("0 molecules converted" in (so + se)) or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            "OpenBabel ligand conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    _sanitize_pdbqt_for_vina125(out_pdbqt)
    _validate_pdbqt_for_vina125(out_pdbqt)
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
            atoms_all, _other_all = read_pdb_atoms(pdb_path)

            # --- ligand (pick ONE instance from crystal) ---
            lig_atoms, lig_key = _pick_single_ligand_instance(
                atoms_all, ligand_resname=ligand_resname, chain_id=chain_id
            )
            if not lig_atoms:
                raise RuntimeError(f"{case_id}: ligand {ligand_resname} not found in crystal PDB (chain {chain_id}).")

            lig_pdb = dirs["ligands"] / f"{case_id}.{ligand_resname}.{lig_key or 'any'}.pdb"
            write_pdb(lig_pdb, lig_atoms, other_lines=None, drop_conect=True)

            lig_pdbqt = dirs["pdbqt_lig"] / f"{case_id}.{ligand_resname}.{lig_key or 'any'}.pdbqt"
            if resume and lig_pdbqt.exists() and lig_pdbqt.stat().st_size > 0:
                _sanitize_pdbqt_for_vina125(lig_pdbqt)
                _validate_pdbqt_for_vina125(lig_pdbqt)
            else:
                _obabel_to_pdbqt_ligand(obabel_exe, lig_pdb, lig_pdbqt)

            case_status["steps"]["ligand_pdb"] = {"path": str(lig_pdb), "instance": lig_key}
            case_status["steps"]["ligand_pdbqt"] = {"path": str(lig_pdbqt), "instance": lig_key}

            # box from ligand centroid
            cx, cy, cz = _centroid(lig_atoms)
            box = DockBox(center_x=cx, center_y=cy, center_z=cz, size_x=sx, size_y=sy, size_z=sz)
            case_status["steps"]["box"] = {"center": [cx, cy, cz], "size": [sx, sy, sz]}

            # --- receptor base ---
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

            # --- receptor pdbqt ---
            receptor_pdbqt = dirs["pdbqt_rec"] / f"{case_id}.pdbqt"
            if resume and receptor_pdbqt.exists() and receptor_pdbqt.stat().st_size > 0:
                _sanitize_pdbqt_for_vina125(receptor_pdbqt)
                _validate_pdbqt_for_vina125(receptor_pdbqt)
            else:
                _obabel_to_pdbqt_receptor(obabel_exe, receptor_pdb, receptor_pdbqt)

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
