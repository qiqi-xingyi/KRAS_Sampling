# --*-- coding:utf-8 --*--
# @time:12/26/25
# @Author : Yuqi Zhang
# @File:pdbqt.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import subprocess
import time


class OpenBabelError(RuntimeError):
    pass


_BAD_TORSION_PREFIX = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
_BAD_MODEL_PREFIX = ("MODEL", "ENDMDL")


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


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _keep_only_first_model_payload(lines: List[str]) -> List[str]:
    has_model = any(l.lstrip().startswith("MODEL") for l in lines)
    if not has_model:
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


def _fix_atom_name(name: str) -> str:
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


def _rewrite_atom_line(tokens: List[str]) -> Optional[str]:
    """
    Convert AD4-like PDBQT atom line to a strict Vina-friendly line
    by removing vdW/Elec columns and fixing atom name chars.

    Input tokens example:
      ATOM  1  C4'  GDP A 201  13.776 -8.601 -16.173  0.00 0.00 +0.178 C

    Output:
      ATOM      1 C4PP GDP A 201    13.776  -8.601 -16.173   0.178 C
    """
    if len(tokens) < 11:
        return None

    rec = tokens[0]
    if rec not in ("ATOM", "HETATM"):
        return None

    serial_tok = tokens[1]
    if not (serial_tok.isdigit() or serial_tok.lstrip("-").isdigit()):
        return None
    serial = int(serial_tok)

    name = _fix_atom_name(tokens[2])
    resname = tokens[3]
    chain = tokens[4]
    try:
        resseq = int(tokens[5])
    except Exception:
        return None

    if not (_is_float(tokens[6]) and _is_float(tokens[7]) and _is_float(tokens[8])):
        return None
    x, y, z = float(tokens[6]), float(tokens[7]), float(tokens[8])

    charge_tok = tokens[-2]
    atype = tokens[-1]
    if not _is_float(charge_tok):
        return None
    charge = float(charge_tok)

    return (
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


def _sanitize_pdbqt_inplace(pdbqt_path: Union[str, Path]) -> None:
    """
    Strict sanitizer for Vina 1.2.5:
    - remove ROOT/BRANCH/TORSDOF
    - remove MODEL/ENDMDL and keep only first MODEL payload
    - keep only REMARK + ATOM/HETATM
    - rewrite ATOM/HETATM lines: fix atom name + drop vdW/Elec cols
    """
    p = Path(pdbqt_path).expanduser().resolve()
    if (not p.exists()) or p.stat().st_size == 0:
        raise OpenBabelError(f"PDBQT missing/empty: {p}")

    raw_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    tmp: List[str] = []
    for l in raw_lines:
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
            cleaned.append(l.strip("\n"))
            continue
        if s.startswith("ATOM") or s.startswith("HETATM"):
            toks = s.split()
            new_line = _rewrite_atom_line(toks)
            if new_line is not None:
                cleaned.append(new_line)
            continue

    if not cleaned:
        raise OpenBabelError(f"Sanitizer produced empty PDBQT: {p}")

    p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _validate_pdbqt(pdbqt_path: Union[str, Path]) -> None:
    p = Path(pdbqt_path).expanduser().resolve()
    if (not p.exists()) or p.stat().st_size == 0:
        raise OpenBabelError(f"PDBQT missing/empty: {p}")

    for l in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = l.lstrip()
        if s.startswith(_BAD_TORSION_PREFIX) or s.startswith(_BAD_MODEL_PREFIX):
            raise OpenBabelError(f"PDBQT contains disallowed tag for Vina 1.2.5: {s.split()[0]} in {p}")


def prepare_receptor_pdbqt_obabel(
    receptor_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    force: bool = False,
    add_h: bool = True,
    partialcharge: str = "gasteiger",
    extra_args: Optional[Sequence[str]] = None,
) -> Path:
    in_p = Path(receptor_pdb).expanduser().resolve()
    out_p = Path(out_pdbqt).expanduser().resolve()
    _ensure_parent(out_p)

    if (not force) and out_p.exists() and out_p.stat().st_size > 0:
        _sanitize_pdbqt_inplace(out_p)
        _validate_pdbqt(out_p)
        return out_p

    cmd = [str(obabel_exe), str(in_p), "-O", str(out_p), "-xr"]
    if add_h:
        cmd += ["-h"]
    cmd += ["--partialcharge", str(partialcharge)]
    if extra_args:
        cmd += list(extra_args)

    rc, so, se, _dt = _run_cmd(cmd)

    if ("0 molecules converted" in (so + se)) or (not out_p.exists()) or out_p.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel receptor conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    _sanitize_pdbqt_inplace(out_p)
    _validate_pdbqt(out_p)
    return out_p


def prepare_ligand_pdbqt_obabel(
    ligand_pdb: Union[str, Path],
    out_pdbqt: Union[str, Path],
    obabel_exe: str = "obabel",
    force: bool = False,
    add_h: bool = True,
    partialcharge: str = "gasteiger",
    extra_args: Optional[Sequence[str]] = None,
) -> Path:
    in_p = Path(ligand_pdb).expanduser().resolve()
    out_p = Path(out_pdbqt).expanduser().resolve()
    _ensure_parent(out_p)

    if (not force) and out_p.exists() and out_p.stat().st_size > 0:
        _sanitize_pdbqt_inplace(out_p)
        _validate_pdbqt(out_p)
        return out_p

    cmd = [str(obabel_exe), str(in_p), "-O", str(out_p)]
    if add_h:
        cmd += ["-h"]
    cmd += ["--partialcharge", str(partialcharge)]
    if extra_args:
        cmd += list(extra_args)

    rc, so, se, _dt = _run_cmd(cmd)

    if ("0 molecules converted" in (so + se)) or (not out_p.exists()) or out_p.stat().st_size == 0:
        raise OpenBabelError(
            "OpenBabel ligand conversion failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {rc}\nSTDOUT(tail): {_tail(so)}\nSTDERR(tail): {_tail(se)}\n"
        )

    _sanitize_pdbqt_inplace(out_p)
    _validate_pdbqt(out_p)
    return out_p


# Backward-compatible aliases (keep old names from earlier iterations)
prepare_receptor_pdbqt_obabel_strict = prepare_receptor_pdbqt_obabel
prepare_ligand_pdbqt_obabel_strict = prepare_ligand_pdbqt_obabel
