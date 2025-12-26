# --*-- conding:utf-8 --*--
# @time:12/25/25 00:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdbqt.py
# docking_verify/pdbqt.py
# -*- coding: utf-8 -*-

"""
docking_verify.pdbqt

OpenBabel-based receptor/ligand PDBQT preparation utilities,
with strict post-sanitization to be compatible with AutoDock Vina v1.2.5.

Key goals
---------
1) Always produce Vina-parseable PDBQT:
   - No ROOT/BRANCH/TORSDOF blocks (Vina 1.2.5 in your build rejects ROOT)
   - No multi-MODEL blocks (keep first MODEL only)
   - Ligand ATOM/HETATM lines are rewritten to Vina-friendly whitespace format:
        record serial name res chain resi x y z q type
     (drop vdW/Elec placeholders like "0.00 0.00", strip leading '+' in charge)
   - Atom names are normalized (e.g., C4' -> C4P) to avoid parser issues.

2) Keep dependencies minimal:
   - Uses OpenBabel CLI (obabel) only.
   - No Meeko, no Pulchra, no ProDy required for this module.

This module is designed to be imported and called by docking_verify.pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union
import shutil
import subprocess

from .pdbio import AtomRecord, parse_pdb_atoms, write_pdb


# =========================
# Errors
# =========================

class OpenBabelError(RuntimeError):
    pass


# =========================
# Helpers
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]


def _which(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise OpenBabelError(f"Cannot find executable '{exe}' in PATH.")
    return p


def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = proc.communicate()
    return int(proc.returncode), out or "", err or ""


def _sanitize_atom_name(name: str) -> str:
    """
    Vina parsers can be picky with non-alnum characters in atom names (e.g., C4').
    We map non-alnum -> 'P' and ensure length<=4 (pad to 4).
    """
    s = "".join(c if c.isalnum() else "P" for c in (name or "X"))
    s = s[:4]
    return s.ljust(4)


def _keep_first_model(lines: List[str]) -> List[str]:
    """
    If there are MODEL/ENDMDL blocks, keep only the first MODEL.
    If no MODEL tags, return unchanged.
    """
    has_model = any(l.lstrip().startswith("MODEL") for l in lines)
    if not has_model:
        return lines

    kept: List[str] = []
    model_idx = 0
    in_first = False

    for l in lines:
        s = l.lstrip()
        if s.startswith("MODEL"):
            model_idx += 1
            in_first = (model_idx == 1)
            continue
        if s.startswith("ENDMDL"):
            if model_idx == 1:
                in_first = False
            continue
        if in_first:
            kept.append(l)

    return kept


def sanitize_receptor_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Receptor must be rigid. Remove torsion-tree tags and multi-model.
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = _keep_first_model(lines)

    drop_prefix = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "MODEL", "ENDMDL")

    out_lines: List[str] = []
    for l in lines:
        s = l.lstrip()
        if not s:
            continue
        if s.startswith(drop_prefix):
            continue
        # keep ATOM/HETATM/REMARK, drop others if you want; Vina usually tolerates REMARK/TER/END
        out_lines.append(l)

    p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def sanitize_ligand_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Make ligand PDBQT compatible with your Vina v1.2.5 build.

    Fixes:
    - keep first MODEL only
    - remove ROOT/BRANCH/TORSDOF
    - rewrite ATOM/HETATM lines into:
        HETATM serial name res chain resi x y z q type
      dropping possible "vdW Elec" columns from OpenBabel:
        ... x y z 0.00 0.00 +0.178 C
      -> ... x y z 0.178 C
    - strip leading '+' in charge
    - normalize atom name (e.g., C4' -> C4P)
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = _keep_first_model(lines)

    drop_prefix = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF", "MODEL", "ENDMDL")

    out_lines: List[str] = []
    for l in lines:
        s = l.lstrip()
        if not s:
            continue

        if s.startswith(drop_prefix):
            continue

        if s.startswith("REMARK"):
            # keep remarks (optional)
            out_lines.append(l)
            continue

        if s.startswith("ATOM") or s.startswith("HETATM"):
            parts = s.split()

            # Two common shapes:
            # (A) "extended" OpenBabel-like:
            #     ATOM serial name res chain resi x y z vdw elec q type   (>=13 tokens)
            # (B) already vina-like:
            #     ATOM serial name res chain resi x y z q type           (>=11 tokens)
            serial = name = res = chain = resi = None
            x = y = z = None
            q = None
            atype = None

            if len(parts) >= 13:
                serial, name, res, chain, resi = parts[1], parts[2], parts[3], parts[4], parts[5]
                x, y, z = parts[6], parts[7], parts[8]
                q = parts[11]
                atype = parts[12]
            elif len(parts) >= 11:
                serial, name, res, chain, resi = parts[1], parts[2], parts[3], parts[4], parts[5]
                x, y, z = parts[6], parts[7], parts[8]
                q = parts[9]
                atype = parts[10]
            else:
                # malformed -> drop
                continue

            # charge cleanup
            q = q.strip()
            if q.startswith("+"):
                q = q[1:]

            # normalize atom name
            name_fixed = _sanitize_atom_name(name)

            # write a Vina-friendly whitespace format
            # record serial name res chain resi x y z q type
            try:
                out_lines.append(
                    f"HETATM{int(serial):>5} {name_fixed}{res:>4} {chain:1}{int(resi):>4}"
                    f"    {float(x):>8.3f}{float(y):>8.3f}{float(z):>8.3f}"
                    f"{float(q):>8.3f} {atype:<2}"
                )
            except Exception:
                # if any parse fails, drop the line
                continue

            continue

        # drop everything else to keep ligand clean

    p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


# =========================
# PDB preparation
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
    Strip a crystal/hybrid PDB to receptor-only PDB.

    IMPORTANT:
    - We do NOT preserve other_lines (CONECT can become inconsistent and breaks OpenBabel).
    - By default, remove waters and the co-crystal ligand should be removed elsewhere.
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)

    keep_het_resnames = set(x.upper() for x in (keep_het_resnames or set()))
    water_resnames = set(x.upper() for x in (water_resnames or {"HOH", "WAT", "H2O"}))

    atoms, _other_lines = parse_pdb_atoms(in_pdb, keep_hetatm=True)

    kept: List[AtomRecord] = []
    for a in atoms:
        rec = a.record.strip().upper()
        resn = (a.resname or "").strip().upper()

        if rec == "ATOM":
            kept.append(a)
            continue

        if rec == "HETATM":
            if remove_water and resn in water_resnames:
                continue
            if keep_hetatm:
                kept.append(a)
                continue
            if resn in keep_het_resnames:
                kept.append(a)
                continue

    # renumber serials
    for i, a in enumerate(kept, start=1):
        a.serial = i

    write_pdb(out_pdb, atoms=kept, other_lines=None)
    return out_pdb


def extract_ligand_pdb_from_crystal(
    crystal_pdb: Union[str, Path],
    out_ligand_pdb: Union[str, Path],
    ligand_resname: str,
    chain_id: Optional[str] = None,
    pick_first_instance: bool = True,
    exclude_resnames: Optional[Set[str]] = None,
) -> Tuple[Path, str]:
    """
    Extract ligand atoms from a crystal PDB by residue name (e.g., GDP).

    Returns:
      (ligand_pdb_path, instance_key)

    instance_key is like "A:201" for the chosen residue instance, which helps debugging.

    Behavior:
    - Collect all HETATM with resname == ligand_resname (optionally chain-filtered)
    - If pick_first_instance=True, choose only one residue instance (chain+resseq+icode)
      to avoid multi-molecule -> multi-MODEL output from OpenBabel.
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    out_ligand_pdb = Path(out_ligand_pdb).expanduser().resolve()
    _ensure_dir(out_ligand_pdb.parent)

    ligand_resname = ligand_resname.strip().upper()
    exclude_resnames = set(x.upper() for x in (exclude_resnames or {"HOH", "WAT", "H2O"}))

    atoms, _other = parse_pdb_atoms(crystal_pdb, keep_hetatm=True)

    lig: List[AtomRecord] = []
    for a in atoms:
        if a.record.strip().upper() != "HETATM":
            continue
        resn = (a.resname or "").strip().upper()
        if resn in exclude_resnames:
            continue
        if resn != ligand_resname:
            continue
        if chain_id is not None and (a.chain or "") != chain_id:
            continue
        lig.append(a)

    if not lig:
        raise ValueError(f"Cannot find ligand resname '{ligand_resname}' in {crystal_pdb} (chain_id={chain_id}).")

    # group by instance (chain, resseq, icode)
    def key(a: AtomRecord) -> Tuple[str, int, str]:
        return (a.chain or "", int(a.resseq), a.icode or "")

    if pick_first_instance:
        groups: dict[Tuple[str, int, str], List[AtomRecord]] = {}
        for a in lig:
            groups.setdefault(key(a), []).append(a)
        chosen_key = sorted(groups.keys(), key=lambda t: (t[0], t[1], t[2]))[0]
        lig_atoms = groups[chosen_key]
        instance_key = f"{chosen_key[0]}:{chosen_key[1]}{chosen_key[2]}"
    else:
        lig_atoms = lig
        # if multiple instances merged, instance_key is generic
        instance_key = "any"

    # renumber serials
    for i, a in enumerate(lig_atoms, start=1):
        a.serial = i

    write_pdb(out_ligand_pdb, atoms=lig_atoms, other_lines=None)
    return out_ligand_pdb, instance_key


# =========================
# OpenBabel conversion
# =========================

def prepare_receptor_pdbqt_obabel(
    receptor_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
    sanitize: bool = True,
) -> Path:
    """
    Convert receptor PDB to PDBQT using OpenBabel.

    Critical:
    - We pass -xr to force rigid receptor output (no torsion tree).
    - Then sanitize to remove any remaining torsion/multi-model tags.
    """
    obabel = _which(obabel_exe)
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    cmd: List[str] = [obabel, str(receptor_pdb), "-O", str(out_pdbqt), "-xr"]

    if add_h:
        cmd.append("-h")
    if ph is not None:
        cmd.extend(["-p", str(float(ph))])
    if partialcharge:
        cmd.extend(["--partialcharge", str(partialcharge)])
    if extra_args:
        cmd.extend(list(extra_args))

    rc, out, err = _run_cmd(cmd)
    if rc != 0 or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel receptor conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\n"
            f"STDOUT(tail): {_tail(out)}\n"
            f"STDERR(tail): {_tail(err)}\n"
        )

    if sanitize:
        sanitize_receptor_pdbqt_for_vina125(out_pdbqt)

    # final check
    if out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(f"Receptor PDBQT became empty after sanitization: {out_pdbqt}")

    return out_pdbqt


def prepare_ligand_pdbqt_obabel(
    ligand_in: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    gen3d: bool = False,
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
    sanitize: bool = True,
) -> Path:
    """
    Convert ligand file (PDB/SDF/MOL2/...) to PDBQT using OpenBabel,
    then rewrite into a strict Vina v1.2.5-friendly PDBQT.

    Notes:
    - For crystal ligands (PDB), gen3d should be False (keep original 3D).
    - Your Vina build rejects ROOT/BRANCH, so we sanitize them away.
    """
    obabel = _which(obabel_exe)

    ligand_in = Path(ligand_in).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

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

    rc, out, err = _run_cmd(cmd)
    if rc != 0 or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel ligand conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\n"
            f"STDOUT(tail): {_tail(out)}\n"
            f"STDERR(tail): {_tail(err)}\n"
        )

    if sanitize:
        sanitize_ligand_pdbqt_for_vina125(out_pdbqt)

    # final check
    if out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(f"Ligand PDBQT became empty after sanitization: {out_pdbqt}")

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
) -> Tuple[Path, Path, str]:
    """
    One-shot helper:
      crystal PDB -> extract ligand (PDB) -> convert ligand PDBQT -> sanitize for Vina 1.2.5

    Returns:
      (ligand_pdb_path, ligand_pdbqt_path, instance_key)
    """
    lig_pdb, instance_key = extract_ligand_pdb_from_crystal(
        crystal_pdb=crystal_pdb,
        out_ligand_pdb=out_ligand_pdb,
        ligand_resname=ligand_resname,
        chain_id=chain_id,
        pick_first_instance=True,
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
        sanitize=True,
    )

    return lig_pdb, lig_pdbqt, instance_key


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
      - optionally strip -> receptor.pdb
      - convert receptor.pdb -> receptor.pdbqt (rigid) + sanitize
    """
    out_dir = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

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
                sanitize=True,
            )
        )

    return out_paths


