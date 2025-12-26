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
- Strip crystal PDB to receptor-only PDB (ATOM only, optionally keep selected HET residues).
- Extract cognate ligand from crystal PDB by residue name (e.g., GDP, MOV).
- Convert receptor/ligand to PDBQT using OpenBabel (obabel).
- Batch helpers and a convenience function to prepare cognate ligand PDBQT from crystal.

Notes
-----
- For docking receptors, it is strongly recommended to strip away waters and co-crystal ligands first.
- For crystal ligands, gen3d is usually NOT needed (they already have 3D coordinates).
- This module is designed to be imported and called from external scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union
import shutil
import subprocess

from .pdbio import AtomRecord, parse_pdb_atoms, write_pdb


class OpenBabelError(RuntimeError):
    pass


def _which_obabel(obabel_exe: str = "obabel") -> str:
    p = shutil.which(obabel_exe)
    if not p:
        raise OpenBabelError(
            f"Cannot find OpenBabel executable '{obabel_exe}' in PATH. "
            f"Make sure you are in the correct conda environment."
        )
    return p


def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out, err


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

    # (Optional but recommended) re-number serials to keep PDB clean
    for i, a in enumerate(kept_atoms, start=1):
        try:
            a.serial = i
        except Exception:
            pass

    # Scheme A: do NOT write other_lines (avoid CONECT/END/Master issues)
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
        if chain_id is not None and a.chain != chain_id:
            return False
        return True

    lig_atoms: List[AtomRecord] = [a for a in atoms if a.record == "HETATM" and match(a)]
    if not lig_atoms and allow_atom_records:
        lig_atoms = [a for a in atoms if a.record == "ATOM" and match(a)]

    if not lig_atoms:
        raise ValueError(
            f"Cannot find ligand resname '{ligand_resname}' in {crystal_pdb} (chain_id={chain_id})."
        )

    # Ligand-only PDB: keep minimal, no need to preserve headers
    write_pdb(out_ligand_pdb, atoms=lig_atoms, other_lines=None)
    return out_ligand_pdb


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
    Convert receptor PDB to PDBQT using OpenBabel.

    Typical:
      obabel receptor.pdb -O receptor.pdbqt -h --partialcharge gasteiger
    """
    obabel = _which_obabel(obabel_exe)

    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [obabel, str(receptor_pdb), "-O", str(out_pdbqt)]

    if add_h:
        cmd.append("-h")
    if ph is not None:
        cmd.extend(["-p", str(float(ph))])
    if partialcharge:
        cmd.extend(["--partialcharge", str(partialcharge)])
    if extra_args:
        cmd.extend(list(extra_args))

    code, out, err = _run_cmd(cmd)
    if code != 0 or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel receptor conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"Return code: {code}\n"
            f"STDOUT(tail): {out[-2000:]}\n"
            f"STDERR(tail): {err[-2000:]}\n"
        )

    return out_pdbqt


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
    Convert ligand file (SDF/MOL2/PDB/...) to PDBQT using OpenBabel.

    Typical:
      obabel ligand.sdf -O ligand.pdbqt --gen3d -h --partialcharge gasteiger
      obabel ligand.pdb -O ligand.pdbqt -h --partialcharge gasteiger
    """
    obabel = _which_obabel(obabel_exe)

    ligand_in = Path(ligand_in).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [obabel, str(ligand_in), "-O", str(out_pdbqt)]

    if gen3d:
        cmd.append("--gen3d")
    if add_h:
        cmd.append("-h")
    if ph is not None:
        cmd.extend(["-p", str(float(ph))])
    if partialcharge:
        cmd.extend(["--partialcharge", str(partialcharge)])
    if extra_args:
        cmd.extend(list(extra_args))

    code, out, err = _run_cmd(cmd)
    if code != 0 or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel ligand conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"Return code: {code}\n"
            f"STDOUT(tail): {out[-2000:]}\n"
            f"STDERR(tail): {err[-2000:]}\n"
        )

    return out_pdbqt


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
      - convert receptor.pdb -> receptor.pdbqt
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
