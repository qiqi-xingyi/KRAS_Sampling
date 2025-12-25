# --*-- conding:utf-8 --*--
# @time:12/25/25 00:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vina.py


from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import subprocess
import time

from .schema import Box, VinaParams


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate()
    return p.returncode, out, err


def parse_vina_scores_from_out_pdbqt(out_pdbqt: Path) -> List[float]:
    """
    Parse scores from Vina output PDBQT. Lines look like:
      REMARK VINA RESULT:    -7.2      0.000      0.000
    """
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    scores: List[float] = []
    if not out_pdbqt.exists():
        return scores
    with out_pdbqt.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.replace("REMARK VINA RESULT:", "").strip().split()
                if parts:
                    try:
                        scores.append(float(parts[0]))
                    except Exception:
                        pass
    return scores


def run_vina_once(
    vina_exe: str,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: Box,
    params: VinaParams,
    out_dir: Path,
    seed: int,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Run AutoDock Vina once (single seed). Writes:
      - out.pdbqt
      - log.txt
      - meta.json

    Returns a dict with paths and status info (for external orchestration).
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdbqt = out_dir / "out.pdbqt"
    log_txt = out_dir / "log.txt"
    meta_json = out_dir / "meta.json"

    cmd = [
        vina_exe,
        "--receptor", str(Path(receptor_pdbqt).expanduser().resolve()),
        "--ligand", str(Path(ligand_pdbqt).expanduser().resolve()),
        *box.as_vina_args(),
        "--exhaustiveness", str(int(params.exhaustiveness)),
        "--num_modes", str(int(params.num_modes)),
        "--energy_range", str(int(params.energy_range)),
        "--cpu", str(int(params.cpu)),
        "--seed", str(int(seed)),
        "--out", str(out_pdbqt),
        "--log", str(log_txt),
    ]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    code, stdout, stderr = _run_cmd(cmd)
    t1 = time.time()

    scores = parse_vina_scores_from_out_pdbqt(out_pdbqt)

    meta = {
        "cmd": cmd,
        "returncode": code,
        "runtime_sec": t1 - t0,
        "seed": seed,
        "params": asdict(params),
        "scores": scores,
        "stdout": stdout[-2000:],  # keep tail
        "stderr": stderr[-2000:],
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out_dir),
        "out_pdbqt": str(out_pdbqt),
        "log_txt": str(log_txt),
        "meta_json": str(meta_json),
        "returncode": str(code),
        "best_score": str(min(scores) if scores else ""),
    }


def run_vina_multi_seed(
    vina_exe: str,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: Box,
    params: VinaParams,
    out_root: Path,
    seeds: List[int],
    extra_args: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Run Vina multiple times with different seeds:
      out_root/seed_{seed}/...

    Returns list of run records.
    """
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, str]] = []
    for s in seeds:
        rec = run_vina_once(
            vina_exe=vina_exe,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            box=box,
            params=params,
            out_dir=out_root / f"seed_{s}",
            seed=s,
            extra_args=extra_args,
        )
        records.append(rec)
    return records
