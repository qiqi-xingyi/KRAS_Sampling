# --*-- coding:utf-8 --*--
# @time:12/25/25
# @Author : Yuqi Zhang
# @File:pdbqt.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import shutil
import subprocess


class OpenBabelError(RuntimeError):
    pass


def _which(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise OpenBabelError(f"Cannot find executable in PATH: {exe}")
    return p


def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return int(p.returncode), out or "", err or ""


# -----------------------------
# Vina 1.2.5 PDBQT sanitizer
# -----------------------------

_BAD_TORSION_PREFIX = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
_BAD_MODEL_PREFIX = ("MODEL", "ENDMDL")


def _fix_atom_name_field(line: str) -> str:
    """
    PDBQT atom name field roughly occupies columns 13-16.
    Vina 1.2.5 can choke on names like C4' so we replace non-alnum with 'P'.
    """
    if len(line) < 16:
        return line
    name = line[12:16]
    fixed = "".join((c if c.isalnum() else "P") for c in name)
    fixed = fixed[:4].ljust(4)
    return line[:12] + fixed + line[16:]


def _keep_only_first_model_payload(lines: List[str]) -> List[str]:
    has_model = any(l.lstrip().startswith("MODEL") for l in lines)
    if not has_model:
        # also drop stray ENDMDL
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


def sanitize_pdbqt_for_vina125(pdbqt_path: Union[str, Path]) -> None:
    """
    Minimal, robust sanitizer for Vina 1.2.5:
    - remove ROOT/BRANCH/TORSDOF tags
    - remove MODEL/ENDMDL and keep only first model payload
    - keep REMARK + ATOM/HETATM lines only
    - fix atom name field (C4' -> C4P etc)
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return

    raw = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    tmp: List[str] = []
    for l in raw:
        s = l.lstrip()
        if not s:
            continue
        if s.startswith(_BAD_TORSION_PREFIX):
            continue
        tmp.append(l.rstrip("\n"))

    tmp = _keep_only_first_model_payload(tmp)

    cleaned: List[str] = []
    for l in tmp:
        s = l.lstrip()
        if s.startswith("REMARK"):
            cleaned.append(l)
        elif s.startswith("ATOM") or s.startswith("HETATM"):
            cleaned.append(_fix_atom_name_field(l))
        else:
            continue

    p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _run_cmd_simple(cmd: List[str], out_path: Path, what: str) -> None:
    rc, so, se = _run(cmd)
    # OpenBabel sometimes returns 0 but converts 0 molecules
    if ("0 molecules converted" in (so + se)) or (not out_path.exists()) or out_path.stat().st_size == 0:
        raise OpenBabelError(
            f"OpenBabel {what} conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )


# -----------------------------
# Public API (kept stable)
# -----------------------------

def prepare_receptor_pdbqt_obabel(
    receptor_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    add_h: bool = True,
    partialcharge: Optional[str] = "gasteiger",
    ph: Optional[float] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Path:
    obabel = _which(obabel_exe)
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [obabel, str(receptor_pdb), "-O", str(out_pdbqt), "-xr"]
    if add_h:
        cmd.append("-h")
    if ph is not None:
        cmd.extend(["-p", str(float(ph))])
    if partialcharge:
        cmd.extend(["--partialcharge", str(partialcharge)])
    if extra_args:
        cmd.extend(list(extra_args))

    _run_cmd_simple(cmd, out_pdbqt, what="receptor")
    sanitize_pdbqt_for_vina125(out_pdbqt)
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
    obabel = _which(obabel_exe)
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

    _run_cmd_simple(cmd, out_pdbqt, what="ligand")
    sanitize_pdbqt_for_vina125(out_pdbqt)
    return out_pdbqt
