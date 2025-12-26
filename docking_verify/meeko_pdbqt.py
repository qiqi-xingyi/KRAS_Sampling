# --*-- conding:utf-8 --*--
# @time:12/25/25 23:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:meeko_pdbqt.py

# docking_verify/meeko_pdbqt.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union, Dict, Tuple, List
import subprocess
import json


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


def prepare_receptor_pdbqt_meeko(
    receptor_pdb: Union[str, Path],
    out_basename: Union[str, Path],
    mk_prepare_receptor_exe: str = "mk_prepare_receptor.py",
    default_altloc: Optional[str] = "A",
    write_json: bool = False,
    write_gpf: bool = False,
) -> MeekoReceptorResult:
    """
    Prepare receptor PDBQT with Meeko CLI.

    Meeko will write:
      <out_basename>_rigid.pdbqt
    and optionally:
      <out_basename>_flex.pdbqt   (if flexible residues requested; we won't use it)
      <out_basename>.json
      <out_basename>.gpf

    For rigid docking with Vina, you should pass the *_rigid.pdbqt file to --receptor.
    """
    receptor_pdb = Path(receptor_pdb).expanduser().resolve()
    out_basename = Path(out_basename).expanduser().resolve()
    _ensure_dir(out_basename.parent)

    cmd: List[str] = [
        mk_prepare_receptor_exe,
        "-i", str(receptor_pdb),
        "-o", str(out_basename),
    ]

    # default altloc is helpful for crystal structures with alternate locations
    if default_altloc is not None:
        cmd += ["--default_altloc", str(default_altloc)]

    if write_json:
        cmd += ["--write_json", str(out_basename.with_suffix(".json"))]
    if write_gpf:
        cmd += ["--write_gpf", str(out_basename.with_suffix(".gpf"))]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    rigid = out_basename.parent / (out_basename.name + "_rigid.pdbqt")
    flex = out_basename.parent / (out_basename.name + "_flex.pdbqt")
    json_path = out_basename.with_suffix(".json")
    gpf_path = out_basename.with_suffix(".gpf")

    if proc.returncode != 0:
        raise MeekoError(
            "mk_prepare_receptor.py failed.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"RC: {proc.returncode}\n"
            f"STDOUT(tail): {(proc.stdout or '')[-2000:]}\n"
            f"STDERR(tail): {(proc.stderr or '')[-2000:]}\n"
        )

    if (not rigid.exists()) or rigid.stat().st_size == 0:
        raise MeekoError(
            "mk_prepare_receptor.py did not create rigid receptor pdbqt.\n"
            f"Expected: {rigid}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT(tail): {(proc.stdout or '')[-2000:]}\n"
            f"STDERR(tail): {(proc.stderr or '')[-2000:]}\n"
        )

    # flex/json/gpf are optional
    flex_pdbqt = flex if flex.exists() and flex.stat().st_size > 0 else None
    json_out = json_path if json_path.exists() and json_path.stat().st_size > 0 else None
    gpf_out = gpf_path if gpf_path.exists() and gpf_path.stat().st_size > 0 else None

    return MeekoReceptorResult(
        rigid_pdbqt=rigid,
        flex_pdbqt=flex_pdbqt,
        json_path=json_out,
        gpf_path=gpf_out,
    )
