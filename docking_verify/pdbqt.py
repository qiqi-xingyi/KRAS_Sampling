# --*-- conding:utf-8 --*--
# @time:12/25/25 00:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdbqt.py


"""
OpenBabel-based receptor/ligand PDBQT preparation utilities (Vina 1.2.5 friendly).

Key goals
---------
1) Crystal/hybrid PDB -> receptor-only PDB (remove CONECT and other lines).
2) Crystal PDB -> extract ligand by resname -> ligand-only PDB (clean).
3) OpenBabel convert to PDBQT:
   - receptor: use -xr (rigid), add H, gasteiger
   - ligand  : add H, gasteiger (gen3d optional; default False for crystal ligand)
4) Sanitize PDBQT for strict Vina 1.2.5:
   - remove ROOT/BRANCH/TORSDOF tags
   - remove MODEL/ENDMDL and keep only first model payload
   - fix atom name field (remove non-alnum, e.g., C4' -> C4P)
   - fix charge token like '+0.178' -> ' 0.178' (preserve width)
   - force ligand records to HETATM (more robust for some builds)

This file intentionally avoids extra dependencies. Only requires:
- OpenBabel executable (obabel)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union
import re
import shutil
import subprocess


# =========================
# Errors
# =========================

class OpenBabelError(RuntimeError):
    pass


# =========================
# Small helpers
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _which_exe(exe: str) -> str:
    p = shutil.which(exe) if not Path(exe).exists() else str(Path(exe).expanduser().resolve())
    if not p:
        raise OpenBabelError(f"Cannot find executable: {exe}")
    return p


def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]


# =========================
# PDB line parsing helpers (minimal, robust)
# =========================

def _pdb_resname(line: str) -> str:
    # PDB: columns 18-20 (1-based), python [17:20]
    if len(line) >= 20:
        return line[17:20].strip()
    # fallback: token
    toks = line.split()
    return toks[3] if len(toks) > 3 else ""


def _pdb_chain_id(line: str) -> str:
    # PDB: column 22 (1-based), python [21]
    if len(line) >= 22:
        return line[21].strip()
    toks = line.split()
    return toks[4] if len(toks) > 4 else ""


def _pdb_is_water_resname(res: str) -> bool:
    return res.upper() in {"HOH", "WAT", "H2O"}


def _write_pdb_lines(out_pdb: Union[str, Path], lines: List[str]) -> Path:
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)
    # Write minimal PDB; avoid CONECT/MASTER/END weirdness
    with out_pdb.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l.rstrip("\n") + "\n")
        f.write("END\n")
    return out_pdb


# =========================
# Step 1: Strip receptor PDB
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
    Strip a PDB to receptor-only PDB lines.

    - Keeps ATOM always
    - HETATM:
        - remove ligand/water by upstream logic; here we only apply:
            remove water if remove_water
            keep if keep_hetatm True OR resname in keep_het_resnames
        - otherwise drop
    - Drops all other records (including CONECT) to avoid OpenBabel parseConectRecord issues
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)

    keep_het_resnames = {x.upper() for x in (keep_het_resnames or set())}
    water_resnames = {x.upper() for x in (water_resnames or {"HOH", "WAT", "H2O"})}

    kept: List[str] = []
    for line in in_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        rec = line[:6].strip()
        if rec == "ATOM":
            kept.append(line)
            continue
        if rec == "HETATM":
            rn = _pdb_resname(line).upper()
            if remove_water and rn in water_resnames:
                continue
            if keep_hetatm or (rn in keep_het_resnames):
                kept.append(line)
            continue
        # drop all others

    return _write_pdb_lines(out_pdb, kept)


# =========================
# Step 2: Extract ligand PDB
# =========================

def extract_ligand_pdb_from_crystal(
    crystal_pdb: Union[str, Path],
    out_ligand_pdb: Union[str, Path],
    ligand_resname: str,
    chain_id: Optional[str] = None,
    allow_atom_records: bool = True,
    exclude_resnames: Optional[Set[str]] = None,
) -> Path:
    """
    Extract ligand lines by residue name (prefer HETATM).
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    out_ligand_pdb = Path(out_ligand_pdb).expanduser().resolve()
    _ensure_dir(out_ligand_pdb.parent)

    ligand_resname = ligand_resname.strip().upper()
    exclude_resnames = {x.upper() for x in (exclude_resnames or {"HOH", "WAT", "H2O"})}

    het: List[str] = []
    atm: List[str] = []

    for line in crystal_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        rec = line[:6].strip()
        if rec not in {"HETATM", "ATOM"}:
            continue
        rn = _pdb_resname(line).upper()
        if rn in exclude_resnames:
            continue
        if rn != ligand_resname:
            continue
        if chain_id is not None:
            ch = _pdb_chain_id(line)
            if ch != chain_id:
                continue
        if rec == "HETATM":
            het.append(line)
        else:
            atm.append(line)

    lig_lines = het if het else (atm if allow_atom_records else [])
    if not lig_lines:
        raise ValueError(f"Cannot find ligand resname '{ligand_resname}' in {crystal_pdb} (chain={chain_id}).")

    # Force ligand to HETATM for stability
    lig_out: List[str] = []
    for l in lig_lines:
        if l.startswith("ATOM"):
            lig_out.append("HETATM" + l[6:])
        else:
            lig_out.append(l)

    return _write_pdb_lines(out_ligand_pdb, lig_out)


# =========================
# Step 3: OpenBabel conversion
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
    Convert receptor PDB to rigid PDBQT using OpenBabel.

    IMPORTANT: -xr is critical to avoid ROOT/BRANCH in receptor PDBQT.
    """
    obabel = _which_exe(obabel_exe)

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

    rc, so, se = _run_cmd(cmd)
    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel receptor conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    sanitize_receptor_pdbqt_for_vina125(out_pdbqt)
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
) -> Path:
    """
    Convert ligand file to PDBQT using OpenBabel.
    For crystal ligands, gen3d should usually be False.
    """
    obabel = _which_exe(obabel_exe)

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

    rc, so, se = _run_cmd(cmd)
    if (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel ligand conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    sanitize_ligand_pdbqt_for_vina125(out_pdbqt)
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


# =========================
# Step 4: PDBQT sanitize for Vina 1.2.5
# =========================

_BAD_PREFIX = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")

# charge token appears near end; we preserve width by replacing "+x.y" -> " x.y"
_CHARGE_TAIL_RE = re.compile(r"(.*\s)([+\-]?\d+(?:\.\d+)?)(\s+[A-Za-z0-9]+)\s*$")


def _keep_only_first_model_payload(lines: List[str]) -> List[str]:
    has_model = any(l.lstrip().startswith("MODEL") for l in lines)
    if not has_model:
        return lines
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
            if model_idx == 1:
                in_first = False
            continue
        if in_first:
            out.append(l)
    return out


def _fix_atom_name_field_fixedwidth(line: str) -> str:
    """
    Fix PDB/PDBQT atom name field (columns 13-16, python [12:16]).
    Replace non-alnum with 'P'. Keep width 4.
    """
    if len(line) < 16:
        return line
    name = line[12:16]
    fixed = "".join((c if c.isalnum() else "P") for c in name)
    fixed = fixed[:4].ljust(4)
    return line[:12] + fixed + line[16:]


def _fix_charge_tail_preserve_width(line: str) -> str:
    """
    Convert charge like '+0.178' to ' 0.178' while preserving string length.
    Applies only if the tail matches '<spaces><charge><spaces><atomtype>'.
    """
    m = _CHARGE_TAIL_RE.match(line.rstrip("\n"))
    if not m:
        return line
    head, charge, tail = m.group(1), m.group(2), m.group(3)
    if charge.startswith("+"):
        # replace leading '+' with space to keep width
        charge2 = " " + charge[1:]
        return f"{head}{charge2}{tail}"
    return line


def sanitize_ligand_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Make ligand PDBQT compatible with strict Vina 1.2.5:
    - drop ROOT/BRANCH/TORSDOF lines
    - keep only first MODEL payload; remove MODEL/ENDMDL
    - keep only REMARK/ATOM/HETATM
    - force record to HETATM (ligand)
    - fix atom name field (no quotes)
    - fix '+0.xxx' charge format
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    # drop torsion tree and empty
    tmp: List[str] = []
    for l in lines:
        s = l.lstrip()
        if not s:
            continue
        if s.startswith(_BAD_PREFIX):
            continue
        tmp.append(l)

    tmp = _keep_only_first_model_payload(tmp)

    cleaned: List[str] = []
    for l in tmp:
        s = l.lstrip()
        if s.startswith("REMARK"):
            cleaned.append(l)
            continue
        if s.startswith("ATOM") or s.startswith("HETATM"):
            # force ligand record to HETATM
            if l.startswith("ATOM"):
                l = "HETATM" + l[6:]
            l = _fix_atom_name_field_fixedwidth(l)
            l = _fix_charge_tail_preserve_width(l)
            cleaned.append(l)
            continue
        # drop everything else

    p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def sanitize_receptor_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Receptor should be rigid. We still sanitize in case tools inject ROOT/BRANCH or MODEL tags.
    Keep only first model, drop torsion tags, keep REMARK/ATOM/HETATM.
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    tmp: List[str] = []
    for l in lines:
        s = l.lstrip()
        if not s:
            continue
        if s.startswith(_BAD_PREFIX):
            continue
        tmp.append(l)

    tmp = _keep_only_first_model_payload(tmp)

    cleaned: List[str] = []
    for l in tmp:
        s = l.lstrip()
        if s.startswith("REMARK"):
            cleaned.append(l)
            continue
        if s.startswith("ATOM") or s.startswith("HETATM"):
            l = _fix_atom_name_field_fixedwidth(l)
            l = _fix_charge_tail_preserve_width(l)
            cleaned.append(l)
            continue

    p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


# =========================
# Batch helper
# =========================

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
            )
        )

    return out_paths

