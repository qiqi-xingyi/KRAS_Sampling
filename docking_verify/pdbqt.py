# --*-- conding:utf-8 --*--
# @time:12/25/25 00:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdbqt.py

"""
docking_verify.pdbqt

Use OpenBabel (obabel) to prepare receptor/ligand PDBQT files.

Design principles
-----------------
- Keep interfaces clean for external calls.
- Avoid hard dependency on any specific conda environment name.
- Always write provenance via deterministic file outputs (caller decides where).

Notes
-----
- For receptors, we recommend stripping crystal ligands/waters first, then convert.
- For ligands, we recommend generating 3D if input is 2D (SDF) and adding charges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union
import subprocess
import shutil

from .pdbio import AtomRecord, parse_pdb_atoms, write_pdb


class OpenBabelError(RuntimeError):
    pass


def _which_obabel(obabel_exe: str = "obabel") -> str:
    """
    Resolve obabel executable path. Raise if not found.
    """
    p = shutil.which(obabel_exe)
    if not p:
        raise OpenBabelError(
            f"Cannot find OpenBabel executable '{obabel_exe}' in PATH. "
            f"Please ensure OpenBabel is installed in your conda environment."
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
    Strip a crystal PDB to a receptor-only PDB.

    Default behavior:
      - Keep ATOM records (protein).
      - Drop all HETATM records (ligands, ions, waters).
      - Optionally keep selected HET resnames (e.g., {'MG'}).
      - Optionally remove waters by resname (default HOH/WAT/H2O).

    This creates a "clean receptor" that is safer for docking preparation.

    Returns: resolved output path
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    keep_het_resnames = keep_het_resnames or set()
    water_resnames = water_resnames or {"HOH", "WAT", "H2O"}

    atoms, other_lines = parse_pdb_atoms(in_pdb, keep_hetatm=True)
    kept_atoms: List[AtomRecord] = []
    kept_other: List[str] = []

    # Keep only essential header lines (optional). Here we keep all other lines as-is.
    kept_other = other_lines

    for a in atoms:
        if a.record == "ATOM":
            kept_atoms.append(a)
            continue

        # HETATM handling
        if a.record == "HETATM":
            if remove_water and a.resname in water_resnames:
                continue
            if keep_hetatm:
                # keep all non-water hetatm
                kept_atoms.append(a)
                continue
            # keep only specified HET resnames
            if a.resname in keep_het_resnames:
                kept_atoms.append(a)
                continue

    write_pdb(out_pdb, atoms=kept_atoms, other_lines=kept_other)
    return out_pdb


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

    Typical command:
      obabel receptor.pdb -O receptor.pdbqt -h --partialcharge gasteiger

    Args:
      add_h: add hydrogens
      partialcharge: 'gasteiger' recommended; set None to skip
      ph: optional pH for protonation (OpenBabel supports -p <pH>)
      extra_args: appended raw args if you need custom behavior

    Returns: resolved output path
    """
    obabel = _which_obabel(obabel_exe)

    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        obabel,
        str(receptor_pdb),
        "-O",
        str(out_pdbqt),
    ]

    if add_h:
        cmd.append("-h")

    if ph is not None:
        cmd.extend(["-p", str(float(ph))])

    if partialcharge:
        # OpenBabel: --partialcharge <method>
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

    Typical commands:
      obabel ligand.sdf -O ligand.pdbqt --gen3d -h --partialcharge gasteiger
      obabel ligand.mol2 -O ligand.pdbqt -h --partialcharge gasteiger

    Args:
      gen3d: add --gen3d for inputs that may lack 3D
      add_h: add hydrogens
      partialcharge: 'gasteiger' recommended; set None to skip
      ph: optional pH (OpenBabel -p)
      extra_args: appended raw args

    Returns: resolved output path
    """
    obabel = _which_obabel(obabel_exe)

    ligand_in = Path(ligand_in).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        obabel,
        str(ligand_in),
        "-O",
        str(out_pdbqt),
    ]

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
    Convenience batch helper:
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
