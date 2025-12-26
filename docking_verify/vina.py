# --*-- conding:utf-8 --*--
# @time:12/25/25 00:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vina.py

# docking_verify/vina.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import json
import subprocess
import time


@dataclass
class VinaRunRecord:
    case_id: str
    receptor_type: str
    seed: int
    out_pdbqt: str
    log_file: str
    returncode: int
    runtime_sec: float
    cmd: List[str]


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_vina_once(
    vina_exe: str,
    receptor_pdbqt: Union[str, Path],
    ligand_pdbqt: Union[str, Path],
    center: Sequence[float],
    size: Sequence[float],
    params,
    out_pdbqt: Union[str, Path],
    log_file: Union[str, Path],
    seed: int,
) -> Dict:
    """
    Run AutoDock Vina once and write output + log to disk.

    This function ALWAYS:
    - writes --out to out_pdbqt
    - writes --log to log_file
    - captures stdout/stderr into log_file as well (append)
    """
    receptor_pdbqt = str(Path(receptor_pdbqt).expanduser().resolve())
    ligand_pdbqt = str(Path(ligand_pdbqt).expanduser().resolve())
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    log_file = Path(log_file).expanduser().resolve()
    _ensure_dir(out_pdbqt.parent)
    _ensure_dir(log_file.parent)

    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

    cmd = [
        str(vina_exe),
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", f"{cx:.6f}",
        "--center_y", f"{cy:.6f}",
        "--center_z", f"{cz:.6f}",
        "--size_x", f"{sx:.6f}",
        "--size_y", f"{sy:.6f}",
        "--size_z", f"{sz:.6f}",
        "--exhaustiveness", str(int(params.exhaustiveness)),
        "--num_modes", str(int(params.num_modes)),
        "--energy_range", str(int(params.energy_range)),
        "--cpu", str(int(params.cpu)),
        "--seed", str(int(seed)),
        "--out", str(out_pdbqt),
        "--log", str(log_file),
    ]

    t0 = time.time()
    # Capture stdout/stderr for debugging; append into log_file
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.time() - t0

    # Append captured stdout/stderr to log so你无需到处找
    with log_file.open("a", encoding="utf-8") as f:
        if proc.stdout:
            f.write("\n\n=== [captured stdout] ===\n")
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n\n=== [captured stderr] ===\n")
            f.write(proc.stderr)

    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "runtime_sec": float(dt),
        "out_pdbqt": str(out_pdbqt),
        "log_file": str(log_file),
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def run_vina_multi_seed(
    vina_exe: str,
    receptor_pdbqt: Union[str, Path],
    ligand_pdbqt: Union[str, Path],
    box,
    params,
    out_root: Union[str, Path],
    seeds: Sequence[int],
    case_id: Optional[str] = None,
    receptor_type: str = "unknown",
) -> List[VinaRunRecord]:
    """
    Run Vina for multiple seeds. Output layout:

    out_root/
      seed_0/
        out.pdbqt
        log.txt
        meta.json
      seed_1/
        ...
    """
    out_root = Path(out_root).expanduser().resolve()
    _ensure_dir(out_root)

    records: List[VinaRunRecord] = []
    cid = case_id or out_root.name

    for s in seeds:
        seed_dir = _ensure_dir(out_root / f"seed_{int(s)}")
        out_pdbqt = seed_dir / "out.pdbqt"
        log_file = seed_dir / "log.txt"
        meta_file = seed_dir / "meta.json"

        meta = run_vina_once(
            vina_exe=vina_exe,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            center=(box.center_x, box.center_y, box.center_z),
            size=(box.size_x, box.size_y, box.size_z),
            params=params,
            out_pdbqt=out_pdbqt,
            log_file=log_file,
            seed=int(s),
        )

        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        rec = VinaRunRecord(
            case_id=cid,
            receptor_type=receptor_type,
            seed=int(s),
            out_pdbqt=str(out_pdbqt),
            log_file=str(log_file),
            returncode=int(meta["returncode"]),
            runtime_sec=float(meta["runtime_sec"]),
            cmd=list(meta["cmd"]),
        )
        records.append(rec)

    return records


def parse_vina_scores_from_out_pdbqt(out_pdbqt: Union[str, Path]) -> List[float]:
    """
    Parse 'REMARK VINA RESULT:' lines from Vina output PDBQT.
    Returns list of affinities (kcal/mol) for each pose.
    """
    out_pdbqt = Path(out_pdbqt).expanduser().resolve()
    if not out_pdbqt.exists() or out_pdbqt.stat().st_size == 0:
        return []

    scores: List[float] = []
    for line in out_pdbqt.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("REMARK VINA RESULT:"):
            # format: REMARK VINA RESULT:     -7.6      0.000      0.000
            parts = line.split()
            if len(parts) >= 4:
                try:
                    scores.append(float(parts[3]))
                except Exception:
                    pass
    return scores
