# --*-- conding:utf-8 --*--
# @time:12/25/25 23:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:meeko_pdbqt.py


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import subprocess
import time


class MeekoError(RuntimeError):
    pass


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class MeekoReceptorResult:
    rigid_pdbqt: Path
    flex_pdbqt: Optional[Path]
    json_path: Optional[Path]
    gpf_path: Optional[Path]
    # debugging
    cmd: List[str]
    returncode: int
    runtime_sec: float
    stdout_tail: str
    stderr_tail: str
    log_file: Optional[Path]


def prepare_receptor_pdbqt_meeko(
    receptor_pdb: Union[str, Path],
    out_basename: Union[str, Path],
    mk_prepare_receptor_exe: str = "mk_prepare_receptor.py",
    default_altloc: Optional[str] = "A",
    write_json: bool = False,
    write_gpf: bool = False,
    allow_bad_res: bool = False,
    timeout_sec: Optional[int] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> MeekoReceptorResult:
    """
    Prepare receptor PDBQT with Meeko CLI.

    Output:
      <out_basename>_rigid.pdbqt  (required)
      <out_basename>_flex.pdbqt   (optional)
      <out_basename>.json         (optional, if write_json=True)
      <out_basename>.gpf          (optional, if write_gpf=True)

    Notes:
    - For batch processing with imperfect structures, allow_bad_res=True can help.
    """
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_basename = Path(out_basename).expanduser().resolve()
    _ensure_dir(out_basename.parent)

    cmd: List[str] = [
        str(mk_prepare_receptor_exe),
        "-i", str(receptor_pdb),
        "-o", str(out_basename),
    ]

    if default_altloc is not None:
        cmd += ["--default_altloc", str(default_altloc)]

    if allow_bad_res:
        # Meeko recommendation for batch: remove residues that cannot be templated
        cmd += ["--allow_bad_res"]

    if write_json:
        cmd += ["--write_json", str(out_basename.with_suffix(".json"))]
    if write_gpf:
        cmd += ["--write_gpf", str(out_basename.with_suffix(".gpf"))]

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as e:
        raise MeekoError(
            "mk_prepare_receptor.py timed out.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"timeout_sec: {timeout_sec}\n"
        ) from e
    dt = time.time() - t0

    rigid = out_basename.parent / (out_basename.name + "_rigid.pdbqt")
    flex = out_basename.parent / (out_basename.name + "_flex.pdbqt")
    json_path = out_basename.with_suffix(".json")
    gpf_path = out_basename.with_suffix(".gpf")

    stdout_tail = (proc.stdout or "")[-2000:]
    stderr_tail = (proc.stderr or "")[-2000:]

    # optional log file
    log_path: Optional[Path] = None
    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        _ensure_dir(log_path.parent)
        with log_path.open("w", encoding="utf-8") as f:
            f.write(" ".join(cmd) + "\n\n")
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write("\n=== [stderr] ===\n")
                f.write(proc.stderr)

    if proc.returncode != 0:
        raise MeekoError(
            "mk_prepare_receptor.py failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {proc.returncode}\n"
            f"STDOUT(tail): {stdout_tail}\n"
            f"STDERR(tail): {stderr_tail}\n"
        )

    if (not rigid.exists()) or rigid.stat().st_size == 0:
        raise MeekoError(
            "mk_prepare_receptor.py did not create rigid receptor pdbqt.\n"
            f"Expected: {rigid}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT(tail): {stdout_tail}\n"
            f"STDERR(tail): {stderr_tail}\n"
        )

    flex_pdbqt = flex if flex.exists() and flex.stat().st_size > 0 else None
    json_out = json_path if json_path.exists() and json_path.stat().st_size > 0 else None
    gpf_out = gpf_path if gpf_path.exists() and gpf_path.stat().st_size > 0 else None

    return MeekoReceptorResult(
        rigid_pdbqt=rigid,
        flex_pdbqt=flex_pdbqt,
        json_path=json_out,
        gpf_path=gpf_out,
        cmd=cmd,
        returncode=int(proc.returncode),
        runtime_sec=float(dt),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        log_file=log_path,
    )
