# docking_verify/pdbqt.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
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
# Utilities
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _which(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise FileNotFoundError(f"Cannot find executable in PATH: {exe}")
    return p


def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]


# =========================
# PDB parsing (minimal)
# =========================

@dataclass
class PDBLine:
    raw: str
    record: str
    resname: str
    chain: str
    resseq: int


def _pdb_iter_atom_lines(pdb_path: Path) -> Iterable[PDBLine]:
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        rec = line[0:6].strip()
        resname = line[17:20].strip()
        chain = line[21:22].strip()
        try:
            resseq = int(line[22:26])
        except Exception:
            resseq = 0
        yield PDBLine(raw=line, record=rec, resname=resname, chain=chain, resseq=resseq)


def strip_to_receptor_pdb(
    in_pdb: Union[str, Path],
    out_pdb: Union[str, Path],
    ligand_resname: Optional[str] = None,
    remove_water: bool = True,
    keep_het_resnames: Optional[Set[str]] = None,
) -> Path:
    """
    Create receptor-only PDB:
    - keep all ATOM
    - keep only selected HETATM (metals etc) if keep_het_resnames contains them
    - drop ligand_resname HETATM
    - drop water if remove_water
    IMPORTANT: we do NOT keep CONECT/etc lines.
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)

    keep_het_resnames = set([x.upper() for x in (keep_het_resnames or set())])
    lig = (ligand_resname or "").upper().strip()
    water = {"HOH", "WAT", "H2O"}

    out_lines: List[str] = []
    serial = 1

    for a in _pdb_iter_atom_lines(in_pdb):
        if a.record == "ATOM":
            line = a.raw
        else:
            rn = a.resname.upper()
            if lig and rn == lig:
                continue
            if remove_water and rn in water:
                continue
            if keep_het_resnames and rn in keep_het_resnames:
                line = a.raw
            else:
                continue

        # renumber atom serials cleanly (cols 7-11)
        if len(line) >= 11:
            line = f"{line[:6]}{serial:5d}{line[11:]}"
        serial += 1
        out_lines.append(line)

    out_pdb.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return out_pdb


def extract_ligand_pdb_from_crystal(
    crystal_pdb: Union[str, Path],
    out_ligand_pdb: Union[str, Path],
    ligand_resname: str,
    chain_id: Optional[str] = None,
) -> Path:
    """
    Extract ligand HETATM lines with given ligand_resname (e.g., GDP).
    Write a ligand-only PDB (no CONECT/header).
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    out_ligand_pdb = Path(out_ligand_pdb).expanduser().resolve()
    _ensure_dir(out_ligand_pdb.parent)

    lig = ligand_resname.upper().strip()
    chain_id = (chain_id or "").strip()

    hits: List[str] = []
    serial = 1

    for a in _pdb_iter_atom_lines(crystal_pdb):
        if a.record != "HETATM":
            continue
        if a.resname.upper() != lig:
            continue
        if chain_id and a.chain != chain_id:
            continue

        line = a.raw
        if len(line) >= 11:
            line = f"{line[:6]}{serial:5d}{line[11:]}"
        serial += 1
        hits.append(line)

    if not hits:
        raise RuntimeError(f"Cannot find ligand {ligand_resname} in {crystal_pdb} (chain={chain_id or 'ANY'})")

    out_ligand_pdb.write_text("\n".join(hits) + "\n", encoding="utf-8")
    return out_ligand_pdb


# =========================
# Strict PDBQT rewriting (Vina 1.2.5 friendly)
# =========================

_TORSION_TAGS = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
_MODEL_TAGS = ("MODEL", "ENDMDL")

_atom_line_re = re.compile(r"^(ATOM|HETATM)\s+")


def _sanitize_atom_name(name4: str) -> str:
    """
    Vina parser is picky. Keep only [A-Za-z0-9], replace others with 'P'.
    Ensure length 4, left-justified.
    """
    s = name4[:4]
    s = "".join(c if c.isalnum() else "P" for c in s)
    if not s.strip():
        s = "C"
    return s.ljust(4)


def _parse_charge_and_type_from_tokens(tokens: List[str]) -> Tuple[float, str]:
    """
    OpenBabel PDBQT often ends with: ... <q> <type>
    Sometimes it includes vdW/elec before q.
    We'll parse from the end robustly.
    """
    if len(tokens) < 2:
        return 0.0, "C"

    atype = tokens[-1]
    qtok = tokens[-2]
    try:
        q = float(qtok)
    except Exception:
        # sometimes q isn't second last if type is missing; fallback
        q = 0.0

    # normalize atom type to <=2 chars (AutoDock types are usually 1-2)
    atype = atype.strip()
    if len(atype) > 2:
        atype = atype[:2]
    if not atype:
        atype = "C"
    return q, atype


def _pdbqt_first_model_payload(lines: List[str]) -> List[str]:
    """
    If MODEL/ENDMDL exist, keep only lines inside first MODEL..ENDMDL.
    Otherwise return all lines.
    """
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
        if in_first and model_idx == 1:
            out.append(l)
    return out


def rewrite_pdbqt_strict_for_vina125(
    in_pdbqt: Union[str, Path],
    out_pdbqt: Union[str, Path],
    drop_torsion_tree: bool = True,
    keep_remark: bool = False,
    force_record: Optional[str] = "ATOM",  # "ATOM" or "HETATM" or None to keep
) -> Path:
    """
    Rewrite any PDBQT into a strict, fixed-width PDBQT that Vina 1.2.5 is more likely to accept.

    Key fixes:
    - keep only first MODEL payload (avoid multi-model error)
    - optionally drop torsion tree tags ROOT/BRANCH/TORSDOF
    - keep only ATOM/HETATM (and optionally REMARK)
    - sanitize atom name (remove apostrophes etc)
    - reformat into fixed columns with charge+type at end
    """
    in_pdbqt = Path(in_pdbqt).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    if not in_pdbqt.exists() or in_pdbqt.stat().st_size == 0:
        raise RuntimeError(f"Input PDBQT is missing/empty: {in_pdbqt}")

    raw_lines = in_pdbqt.read_text(encoding="utf-8", errors="ignore").splitlines()
    raw_lines = _pdbqt_first_model_payload(raw_lines)

    atoms_out: List[str] = []
    serial = 1

    for line in raw_lines:
        s = line.lstrip()
        if not s:
            continue

        if s.startswith(_MODEL_TAGS):
            continue
        if drop_torsion_tree and s.startswith(_TORSION_TAGS):
            continue

        if s.startswith("REMARK"):
            if keep_remark:
                atoms_out.append(line.rstrip("\n"))
            continue

        if not _atom_line_re.match(s):
            continue

        # Try fixed-column parse for coords/residue fields
        rec = line[0:6].strip()
        name4 = line[12:16] if len(line) >= 16 else "C"
        resname = line[17:20] if len(line) >= 20 else "LIG"
        chain = line[21:22] if len(line) >= 22 else "A"
        resseq_str = line[22:26] if len(line) >= 26 else "1"

        try:
            resseq = int(resseq_str)
        except Exception:
            resseq = 1

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except Exception:
            # fallback: parse by split
            toks = s.split()
            # expected: ATOM serial name resname chain resseq x y z ...
            if len(toks) < 9:
                continue
            try:
                x, y, z = float(toks[6]), float(toks[7]), float(toks[8])
            except Exception:
                continue

        toks = s.split()
        q, atype = _parse_charge_and_type_from_tokens(toks)

        name4 = _sanitize_atom_name(name4)
        resname = resname.strip().rjust(3)
        chain = (chain.strip() or "A")[:1]

        out_rec = force_record if force_record in ("ATOM", "HETATM") else rec

        # Fixed-width formatting:
        # cols: record(1-6), serial(7-11), name(13-16), resname(18-20), chain(22), resseq(23-26),
        # x(31-38) y(39-46) z(47-54), occ(55-60), temp(61-66), q(~71-76), type(78-79)
        out_line = (
            f"{out_rec:<6}{serial:5d} "
            f"{name4:<4}{resname:>4} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{0.00:6.2f}{0.00:6.2f}    "
            f"{q:7.3f} {atype:<2}"
        )
        atoms_out.append(out_line.rstrip())
        serial += 1

    if not atoms_out:
        raise RuntimeError(
            "Strict rewrite produced empty PDBQT. "
            f"Input may be malformed: {in_pdbqt}"
        )

    out_pdbqt.write_text("\n".join(atoms_out) + "\n", encoding="utf-8")
    return out_pdbqt


# =========================
# OpenBabel -> strict PDBQT
# =========================

def prepare_receptor_pdbqt_obabel_strict(
    receptor_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    add_h: bool = True,
    partialcharge: str = "gasteiger",
    force: bool = True,
) -> Path:
    """
    Receptor:
    - OpenBabel conversion with -xr (rigid)
    - rewrite to strict Vina-friendly PDBQT
    """
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    raw_pdbqt = out_pdbqt.with_suffix(".raw.pdbqt")

    if (not force) and out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        return out_pdbqt

    obabel = _which(obabel_exe)

    cmd = [obabel, str(receptor_pdb), "-O", str(raw_pdbqt), "-xr"]
    if add_h:
        cmd.append("-h")
    if partialcharge:
        cmd += ["--partialcharge", partialcharge]

    rc, so, se = _run_cmd(cmd)
    if rc != 0 or (not raw_pdbqt.exists()) or raw_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel receptor conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    return rewrite_pdbqt_strict_for_vina125(
        in_pdbqt=raw_pdbqt,
        out_pdbqt=out_pdbqt,
        drop_torsion_tree=True,
        keep_remark=False,
        force_record="ATOM",
    )


def prepare_ligand_pdbqt_obabel_strict(
    ligand_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    add_h: bool = True,
    partialcharge: str = "gasteiger",
    force: bool = True,
) -> Path:
    """
    Ligand (e.g., GDP from crystal):
    - OpenBabel conversion
    - rewrite to strict Vina-friendly PDBQT (NO multi-model, NO torsion tags)
    """
    ligand_pdb = Path(ligand_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)

    raw_pdbqt = out_pdbqt.with_suffix(".raw.pdbqt")

    if (not force) and out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        return out_pdbqt

    obabel = _which(obabel_exe)

    cmd = [obabel, str(ligand_pdb), "-O", str(raw_pdbqt)]
    if add_h:
        cmd.append("-h")
    if partialcharge:
        cmd += ["--partialcharge", partialcharge]

    rc, so, se = _run_cmd(cmd)
    if rc != 0 or (not raw_pdbqt.exists()) or raw_pdbqt.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel ligand conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    return rewrite_pdbqt_strict_for_vina125(
        in_pdbqt=raw_pdbqt,
        out_pdbqt=out_pdbqt,
        drop_torsion_tree=True,
        keep_remark=False,
        force_record="ATOM",
    )
