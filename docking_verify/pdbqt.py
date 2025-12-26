# --*-- conding:utf-8 --*--
# @time:12/25/25 00:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdbqt.py

"""
docking_verify.pdbqt

OpenBabel-based receptor/ligand PDBQT preparation utilities.

Core features
------------
- Strip crystal/hybrid PDB to receptor-only PDB (ATOM only, optionally keep selected HET residues).
- Extract cognate ligand from crystal PDB by residue name (e.g., GDP, MOV).
- Convert receptor/ligand to PDBQT using OpenBabel (obabel).
- Sanitize OpenBabel PDBQT for AutoDock Vina v1.2.5 (remove ROOT/BRANCH/TORSDOF... tags).
- Batch helpers.
- Kabsch + graft helpers for hybrid construction (kept here for convenience).

Notes
-----
- We intentionally do NOT carry over other_lines from the original PDB in stripping/writing,
  because CONECT records can become inconsistent after hybrid rebuilding, which breaks OpenBabel parsing.
- For crystal ligands, gen3d is usually NOT needed (they already have 3D coordinates).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import shutil
import subprocess

import numpy as np

from .pdbio import AtomRecord, parse_pdb_atoms, write_pdb


class OpenBabelError(RuntimeError):
    pass


# =========================
# Executable + subprocess
# =========================

def _which_obabel(obabel_exe: str = "obabel") -> str:
    """
    Resolve OpenBabel executable:
    - If obabel_exe is an existing file path, use it directly.
    - Else search in PATH.
    """
    p = Path(obabel_exe)
    if p.exists() and p.is_file():
        return str(p.expanduser().resolve())

    found = shutil.which(obabel_exe)
    if not found:
        raise OpenBabelError(
            f"Cannot find OpenBabel executable '{obabel_exe}'. "
            f"Make sure you are in the correct conda environment, or pass an absolute path."
        )
    return found


def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out, err


# =========================
# PDBQT sanitize for Vina 1.2.5
# =========================

_BAD_TAGS_VINA125: Set[str] = {"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "CONECT"}


def sanitize_pdbqt_for_vina125(
    in_pdbqt: Union[str, Path],
    out_pdbqt: Union[str, Path],
    drop_tags: Optional[Set[str]] = None,
) -> Path:
    """
    Vina v1.2.5 may reject ROOT/BRANCH/TORSDOF tags. Some converters (including OpenBabel in
    some configs) emit these tags for ligands/receptors. This is a pure text filter.

    Removes any line whose first token is in drop_tags.
    """
    in_pdbqt = Path(in_pdbqt).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    tags = drop_tags or set(_BAD_TAGS_VINA125)

    lines_in = in_pdbqt.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    kept: List[str] = []
    for line in lines_in:
        s = line.strip()
        if not s:
            continue
        first = s.split()[0]
        if first in tags:
            continue
        kept.append(line)

    out_pdbqt.write_text("".join(kept), encoding="utf-8")
    return out_pdbqt


def _obabel_to_pdbqt_and_sanitize(
    in_file: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str,
    add_h: bool,
    partialcharge: Optional[str],
    ph: Optional[float],
    extra_args: Optional[Sequence[str]],
) -> Path:
    """
    Convert to a temporary raw PDBQT with OpenBabel, then sanitize to final PDBQT for Vina 1.2.5.
    """
    obabel = _which_obabel(obabel_exe)

    in_file = Path(in_file).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    raw_pdbqt = out_pdbqt.parent / f"{out_pdbqt.stem}.raw.pdbqt"

    cmd: List[str] = [obabel, str(in_file), "-O", str(raw_pdbqt)]

    if add_h:
        cmd.append("-h")
    if ph is not None:
        cmd.extend(["-p", str(float(ph))])
    if partialcharge:
        cmd.extend(["--partialcharge", str(partialcharge)])
    if extra_args:
        cmd.extend(list(extra_args))

    code, out, err = _run_cmd(cmd)
    if code != 0 or (not raw_pdbqt.exists()) or raw_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"Return code: {code}\n"
            f"STDOUT(tail): {out[-2000:]}\n"
            f"STDERR(tail): {err[-2000:]}\n"
        )

    sanitize_pdbqt_for_vina125(raw_pdbqt, out_pdbqt)
    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(f"sanitize_pdbqt_for_vina125 produced empty pdbqt: {out_pdbqt}")

    return out_pdbqt


# =========================
# PDB stripping + ligand extraction
# =========================

def strip_to_receptor_pdb(
    in_pdb: Union[str, Path],
    out_pdb: Union[str, Path],
    keep_hetatm: bool = False,
    keep_het_resnames: Optional[Set[str]] = None,
    remove_water: bool = True,
    water_resnames: Optional[Set[str]] = None,
) -> Path:
    """
    Strip a crystal/hybrid PDB to a receptor-only PDB.

    IMPORTANT:
    - We intentionally do NOT carry over other_lines from the original PDB,
      because CONECT records can become inconsistent after hybrid rebuilding,
      which breaks OpenBabel parsing.
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    keep_het_resnames = keep_het_resnames or set()
    water_resnames = water_resnames or {"HOH", "WAT", "H2O"}

    atoms, _other_lines = parse_pdb_atoms(in_pdb, keep_hetatm=True)
    kept_atoms: List[AtomRecord] = []

    for a in atoms:
        if a.record == "ATOM":
            kept_atoms.append(a)
            continue

        if a.record == "HETATM":
            if remove_water and a.resname in water_resnames:
                continue
            if keep_hetatm:
                kept_atoms.append(a)
                continue
            if a.resname in keep_het_resnames:
                kept_atoms.append(a)
                continue

    # Re-number serials to keep PDB clean
    for i, a in enumerate(kept_atoms, start=1):
        a.serial = i

    write_pdb(out_pdb, atoms=kept_atoms, other_lines=None)
    return out_pdb


def extract_ligand_pdb_from_crystal(
    crystal_pdb: Union[str, Path],
    out_ligand_pdb: Union[str, Path],
    ligand_resname: str,
    chain_id: Optional[str] = None,
    allow_atom_records: bool = True,
    exclude_resnames: Optional[Set[str]] = None,
) -> Path:
    """
    Extract a ligand from a crystal PDB by residue name (e.g., GDP, MOV).

    Rules:
    - Prefer HETATM with resname == ligand_resname
    - If none found and allow_atom_records=True, fall back to ATOM with same resname (rare)
    - Optionally filter by chain_id
    - Exclude common water resnames by default
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    out_ligand_pdb = Path(out_ligand_pdb).expanduser().resolve()
    out_ligand_pdb.parent.mkdir(parents=True, exist_ok=True)

    ligand_resname = ligand_resname.strip()
    exclude_resnames = exclude_resnames or {"HOH", "WAT", "H2O"}

    atoms, _other = parse_pdb_atoms(crystal_pdb, keep_hetatm=True)

    def match(a: AtomRecord) -> bool:
        if a.resname in exclude_resnames:
            return False
        if a.resname != ligand_resname:
            return False
        if chain_id is not None and a.chain_id != chain_id:
            return False
        return True

    lig_atoms: List[AtomRecord] = [a for a in atoms if a.record == "HETATM" and match(a)]
    if not lig_atoms and allow_atom_records:
        lig_atoms = [a for a in atoms if a.record == "ATOM" and match(a)]

    if not lig_atoms:
        raise ValueError(
            f"Cannot find ligand resname '{ligand_resname}' in {crystal_pdb} (chain_id={chain_id})."
        )

    write_pdb(out_ligand_pdb, atoms=lig_atoms, other_lines=None)
    return out_ligand_pdb


# =========================
# OpenBabel -> PDBQT (sanitized)
# =========================

def prepare_receptor_pdbqt_obabel(
    receptor_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Path:
    """
    Convert receptor PDB to PDBQT using OpenBabel, then sanitize for Vina v1.2.5.
    """
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    return _obabel_to_pdbqt_and_sanitize(
        in_file=receptor_pdb,
        out_pdbqt=out_pdbqt,
        obabel_exe=obabel_exe,
        add_h=add_h,
        partialcharge=partialcharge,
        ph=ph,
        extra_args=extra_args,
    )


def prepare_ligand_pdbqt_obabel(
    ligand_in: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    gen3d: bool = True,
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Path:
    """
    Convert ligand file (SDF/MOL2/PDB/...) to PDBQT using OpenBabel, then sanitize for Vina v1.2.5.
    """
    ligand_in = Path(ligand_in).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    xargs: List[str] = []
    if gen3d:
        xargs.append("--gen3d")
    if extra_args:
        xargs.extend(list(extra_args))

    return _obabel_to_pdbqt_and_sanitize(
        in_file=ligand_in,
        out_pdbqt=out_pdbqt,
        obabel_exe=obabel_exe,
        add_h=add_h,
        partialcharge=partialcharge,
        ph=ph,
        extra_args=xargs if xargs else None,
    )


def prepare_cognate_ligand_pdbqt_from_crystal(
    crystal_pdb: Union[str, Path],
    ligand_resname: str,
    out_ligand_pdb: Union[str, Path],
    out_ligand_pdbqt: Union[str, Path],
    chain_id: Optional[str] = None,
    obabel_exe: str = "obabel",
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[Path, Path]:
    """
    One-shot helper:
      crystal PDB -> extract ligand (PDB) -> convert to ligand PDBQT

    For crystal ligands, gen3d is forced to False by default, since coordinates are already 3D.
    Returns: (ligand_pdb_path, ligand_pdbqt_path)
    """
    lig_pdb = extract_ligand_pdb_from_crystal(
        crystal_pdb=crystal_pdb,
        out_ligand_pdb=out_ligand_pdb,
        ligand_resname=ligand_resname,
        chain_id=chain_id,
    )

    lig_pdbqt = prepare_ligand_pdbqt_obabel(
        ligand_in=lig_pdb,
        out_pdbqt=out_ligand_pdbqt,
        obabel_exe=obabel_exe,
        gen3d=False,
        add_h=add_h,
        partialcharge=partialcharge,
        ph=ph,
        extra_args=extra_args,
    )

    return lig_pdb, lig_pdbqt


def batch_prepare_receptors_from_pdb(
    receptor_pdbs: Iterable[Union[str, Path]],
    out_dir: Union[str, Path],
    obabel_exe: str = "obabel",
    strip_first: bool = True,
    keep_hetatm: bool = False,
    keep_het_resnames: Optional[Set[str]] = None,
    remove_water: bool = True,
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
) -> List[Path]:
    """
    Batch helper:
      - optionally strip crystal PDB -> receptor.pdb
      - convert receptor.pdb -> receptor.pdbqt (sanitized)
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []
    for pdb in receptor_pdbs:
        pdb = Path(pdb).expanduser().resolve()
        stem = pdb.stem

        clean_pdb = out_dir / f"{stem}.receptor.pdb" if strip_first else pdb
        pdbqt = out_dir / f"{stem}.pdbqt"

        if strip_first:
            strip_to_receptor_pdb(
                in_pdb=pdb,
                out_pdb=clean_pdb,
                keep_hetatm=keep_hetatm,
                keep_het_resnames=keep_het_resnames,
                remove_water=remove_water,
            )

        out_paths.append(
            prepare_receptor_pdbqt_obabel(
                receptor_pdb=clean_pdb,
                out_pdbqt=pdbqt,
                obabel_exe=obabel_exe,
                add_h=add_h,
                partialcharge=partialcharge,
                ph=ph,
            )
        )

    return out_paths


# =========================
# Geometry helpers (Kabsch + graft)
# =========================

def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find R,t s.t. R*P + t ~= Q
    P, Q: (N,3)
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc
    C = X.T @ Y
    V, _S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Qc - R @ Pc
    return R, t


def write_ca_trace_pdb(
    out_pdb: Union[str, Path],
    chain_id: str,
    res_start: int,
    sequence: str,
    ca_xyz: List[List[float]],
) -> Path:
    """
    Write a CA-only PDB using ATOM records with correct chain/resi/resname.
    """
    out_pdb = Path(out_pdb).expanduser().resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    aa3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    atoms: List[AtomRecord] = []
    serial = 1
    for i, (aa1, xyz) in enumerate(zip(sequence, ca_xyz)):
        resi = int(res_start + i)
        resname = aa3.get(aa1, "GLY")
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        a = AtomRecord(
            record="ATOM",
            serial=serial,
            name="CA",
            altloc="",
            resname=resname,
            chain_id=chain_id,
            resseq=resi,
            icode="",
            x=x, y=y, z=z,
            occupancy=1.00,
            tempfactor=0.00,
            element="C",
            charge="",
        )
        serial += 1
        atoms.append(a)

    write_pdb(out_pdb, atoms=atoms, other_lines=None)
    return out_pdb


def graft_fragment_allatom_into_crystal(
    crystal_pdb: Union[str, Path],
    fragment_allatom_pdb: Union[str, Path],
    chain_id: str,
    res_start: int,
    res_end: int,
    out_pdb: Union[str, Path],
) -> Path:
    """
    Replace residues [res_start, res_end] in crystal_pdb(chain_id) with atoms from fragment_allatom_pdb.
    Assumes fragment_allatom_pdb uses the same chain_id and residue numbering.
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    fragment_allatom_pdb = Path(fragment_allatom_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    crystal_atoms, _ = parse_pdb_atoms(crystal_pdb, keep_hetatm=True)
    frag_atoms, _ = parse_pdb_atoms(fragment_allatom_pdb, keep_hetatm=True)

    repl = set((chain_id, r) for r in range(int(res_start), int(res_end) + 1))

    kept: List[AtomRecord] = []
    for a in crystal_atoms:
        if (a.chain_id, a.resseq) in repl:
            continue
        kept.append(a)

    frag_keep: List[AtomRecord] = []
    for a in frag_atoms:
        if a.chain_id != chain_id:
            continue
        if int(res_start) <= int(a.resseq) <= int(res_end):
            frag_keep.append(a)

    merged = kept + frag_keep
    for i, a in enumerate(merged, start=1):
        a.serial = i

    write_pdb(out_pdb, atoms=merged, other_lines=None)
    return out_pdb
