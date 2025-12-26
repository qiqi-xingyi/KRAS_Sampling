# --*-- conding:utf-8 --*--
# @time:12/26/25 00:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:rebuild_allatom.py


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List, Tuple
import subprocess
import shutil


class RebuildError(RuntimeError):
    pass


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


@dataclass
class RebuildConfig:
    pulchra_exe: str = "pulchra"
    scwrl_exe: Optional[str] = None  # e.g. "Scwrl4"
    # If True, run pulchra in "rebuild all atoms" mode when supported
    # (pulchra variants differ; we keep the call simple and validate outputs).
    strict: bool = True


@dataclass
class RebuildResult:
    ca_pdb: Path
    allatom_pdb: Path
    pulchra_stdout: str
    pulchra_stderr: str
    scwrl_stdout: Optional[str] = None
    scwrl_stderr: Optional[str] = None


def run_pulchra(
    ca_pdb: Union[str, Path],
    out_dir: Union[str, Path],
    cfg: RebuildConfig,
) -> Tuple[Path, str, str]:
    """
    Run PULCHRA on a CA-only PDB and return the rebuilt all-atom PDB.
    PULCHRA output filename conventions vary by build; we search in out_dir.
    """
    ca_pdb = Path(ca_pdb).expanduser().resolve()
    out_dir = _ensure_dir(Path(out_dir).expanduser().resolve())

    pulchra = _which(cfg.pulchra_exe) or cfg.pulchra_exe
    cmd = [pulchra, str(ca_pdb)]

    proc = subprocess.run(cmd, cwd=str(out_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    so, se = proc.stdout or "", proc.stderr or ""

    if proc.returncode != 0 and cfg.strict:
        raise RebuildError(
            "PULCHRA failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {proc.returncode}\n"
            f"STDOUT(tail): {so[-2000:]}\n"
            f"STDERR(tail): {se[-2000:]}\n"
        )

    # Common pulchra outputs: *_rebuilt.pdb, *_pulchra.pdb, or same basename with suffix.
    cand = []
    for p in out_dir.glob("*.pdb"):
        cand.append(p)

    if not cand:
        raise RebuildError(
            "PULCHRA produced no PDB outputs in out_dir.\n"
            f"out_dir={out_dir}\n"
            f"STDOUT(tail): {so[-2000:]}\n"
            f"STDERR(tail): {se[-2000:]}\n"
        )

    # pick the newest pdb file
    cand.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    rebuilt = cand[0]
    if rebuilt.stat().st_size == 0:
        raise RebuildError(f"PULCHRA output is empty: {rebuilt}")

    return rebuilt, so, se


def run_scwrl4(
    in_pdb: Union[str, Path],
    out_pdb: Union[str, Path],
    scwrl_exe: str,
    strict: bool = True,
) -> Tuple[str, str]:
    """
    Run SCWRL4 to optimize side chains.
    Typical usage: Scwrl4 -i input.pdb -o output.pdb
    """
    in_pdb = Path(in_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()
    _ensure_dir(out_pdb.parent)

    scwrl = _which(scwrl_exe) or scwrl_exe
    cmd = [scwrl, "-i", str(in_pdb), "-o", str(out_pdb)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    so, se = proc.stdout or "", proc.stderr or ""

    if proc.returncode != 0 and strict:
        raise RebuildError(
            "SCWRL4 failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {proc.returncode}\n"
            f"STDOUT(tail): {so[-2000:]}\n"
            f"STDERR(tail): {se[-2000:]}\n"
        )

    if not out_pdb.exists() or out_pdb.stat().st_size == 0:
        raise RebuildError(f"SCWRL4 produced no output PDB: {out_pdb}")

    return so, se


def rebuild_allatom_from_ca(
    ca_pdb: Union[str, Path],
    out_dir: Union[str, Path],
    cfg: RebuildConfig,
) -> RebuildResult:
    """
    CA-only PDB -> all-atom PDB using PULCHRA (+ optional SCWRL4).
    """
    ca_pdb = Path(ca_pdb).expanduser().resolve()
    out_dir = _ensure_dir(Path(out_dir).expanduser().resolve())

    rebuilt0, so, se = run_pulchra(ca_pdb, out_dir, cfg)

    final_pdb = out_dir / "rebuilt_allatom.pdb"
    if cfg.scwrl_exe:
        so2, se2 = run_scwrl4(rebuilt0, final_pdb, cfg.scwrl_exe, strict=cfg.strict)
        return RebuildResult(
            ca_pdb=ca_pdb,
            allatom_pdb=final_pdb,
            pulchra_stdout=so,
            pulchra_stderr=se,
            scwrl_stdout=so2,
            scwrl_stderr=se2,
        )

    # no scwrl: just copy pulchra output to stable name
    final_pdb.write_text(rebuilt0.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    return RebuildResult(
        ca_pdb=ca_pdb,
        allatom_pdb=final_pdb,
        pulchra_stdout=so,
        pulchra_stderr=se,
    )
