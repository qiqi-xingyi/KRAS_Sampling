# --*-- conding:utf-8 --*--
# @time:12/26/25 20:12
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:vina_runner.py

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


@dataclass(frozen=True)
class VinaParams:
    exhaustiveness: int = 16
    num_modes: int = 20
    energy_range: int = 3
    cpu: int = 8


@dataclass(frozen=True)
class VinaPoseRow:
    mode: int
    affinity: float
    rmsd_lb: float
    rmsd_ub: float


# -----------------------------
# Parsing utilities
# -----------------------------
_VINA_TABLE_LINE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$"
)


def parse_vina_log_for_poses(log_text: str) -> List[VinaPoseRow]:
    """
    Parse vina stdout/stderr combined log for the result table:

      mode | affinity | dist from best mode
       1      -7.5      0.000      0.000
       2      -7.3      1.234      2.345
      ...

    Returns: list of VinaPoseRow sorted by mode.
    """
    rows: List[VinaPoseRow] = []
    for line in log_text.splitlines():
        m = _VINA_TABLE_LINE.match(line)
        if not m:
            continue
        mode = int(m.group(1))
        affinity = float(m.group(2))
        rmsd_lb = float(m.group(3))
        rmsd_ub = float(m.group(4))
        rows.append(VinaPoseRow(mode=mode, affinity=affinity, rmsd_lb=rmsd_lb, rmsd_ub=rmsd_ub))

    rows.sort(key=lambda r: r.mode)
    return rows


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _safe_write_json(path: Path, obj: object, indent: int = 2) -> None:
    _safe_write_text(path, json.dumps(obj, indent=indent))


# -----------------------------
# Runner
# -----------------------------
class VinaDockingRunner:
    """
    Component:
      - Run AutoDock Vina on a prepared receptor/ligand PDBQT pair
      - Support repeated runs with recorded random seeds for reproducibility
      - Capture full Vina logs
      - Parse results and write tidy tables for plotting / listing

    Output layout:
      docking_result/
        30_dock/
          <group_key>/
            input/
              receptor.pdbqt    (optional copy)
              ligand.pdbqt      (optional copy)
              box.json
              vina_params.json
            runs/
              seed_<seed>/
                out.pdbqt
                log.txt
                params.json
                parsed.json
                poses.csv
            summary/
              runs_long.csv
              runs_summary.csv
              summary.json
              best_pose.pdbqt
              best_run.json
    """

    def __init__(
        self,
        result_root: Path = Path("docking_result"),
        step_dirname: str = "30_dock",
        vina_bin: str = "vina",
        archive_inputs: bool = True,
    ) -> None:
        self.result_root = Path(result_root)
        self.step_dir = self.result_root / step_dirname
        self.vina_bin = str(vina_bin)
        self.archive_inputs = bool(archive_inputs)

    def run_group(
        self,
        *,
        target_group_key: str,
        receptor_pdbqt: Path,
        ligand_pdbqt: Path,
        box: DockingBox,
        vina_params: VinaParams = VinaParams(),
        n_repeats: int = 1,
        seed_list: Optional[Sequence[int]] = None,
        base_seed: int = 0,
        overwrite: bool = False,
        strict: bool = True,
    ) -> Dict[str, Path]:
        """
        Args:
          target_group_key: e.g. 4LPK_WT, 6OIM_G12C, 9C41_G12D
          receptor_pdbqt / ligand_pdbqt: Vina inputs
          box: docking box
          vina_params: exhaustiveness/num_modes/energy_range/cpu
          n_repeats: number of repeated dockings (ignored if seed_list provided)
          seed_list: explicit seeds to run (reproducible and preferred)
          base_seed: if seed_list not provided, seeds = base_seed + i
          overwrite: if False, will skip runs with existing out+log+parsed
          strict: if True, a failed run raises; else failures are recorded and summary continues

        Returns:
          dict of key outputs: out_dir, summary_json, runs_long_csv, runs_summary_csv, best_pose_pdbqt
        """
        receptor_pdbqt = Path(receptor_pdbqt)
        ligand_pdbqt = Path(ligand_pdbqt)
        if not receptor_pdbqt.exists():
            raise FileNotFoundError(f"receptor_pdbqt not found: {receptor_pdbqt}")
        if not ligand_pdbqt.exists():
            raise FileNotFoundError(f"ligand_pdbqt not found: {ligand_pdbqt}")

        out_dir = self.step_dir / target_group_key
        input_dir = out_dir / "input"
        runs_dir = out_dir / "runs"
        summary_dir = out_dir / "summary"
        for d in (input_dir, runs_dir, summary_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Archive key inputs for traceability
        _safe_write_json(input_dir / "box.json", asdict(box))
        _safe_write_json(input_dir / "vina_params.json", asdict(vina_params))

        if self.archive_inputs:
            shutil.copy2(receptor_pdbqt, input_dir / "receptor.pdbqt")
            shutil.copy2(ligand_pdbqt, input_dir / "ligand.pdbqt")

        seeds: List[int]
        if seed_list is not None:
            seeds = [int(s) for s in seed_list]
        else:
            if n_repeats <= 0:
                raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
            seeds = [int(base_seed) + i for i in range(int(n_repeats))]

        run_summaries: List[Dict[str, object]] = []
        long_rows: List[Dict[str, object]] = []
        failures: List[Dict[str, object]] = []

        best_global: Optional[Tuple[float, int, int, Path]] = None
        # (affinity, seed, mode, pose_file)

        for seed in seeds:
            run_dir = runs_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            out_pdbqt = run_dir / "out.pdbqt"
            log_txt = run_dir / "log.txt"
            parsed_json = run_dir / "parsed.json"
            poses_csv = run_dir / "poses.csv"
            params_json = run_dir / "params.json"

            already_done = out_pdbqt.exists() and log_txt.exists() and parsed_json.exists()
            if already_done and (not overwrite):
                # still load parsed for summaries
                try:
                    parsed = json.loads(parsed_json.read_text())
                    poses = [
                        VinaPoseRow(**row)  # type: ignore[arg-type]
                        for row in parsed.get("poses", [])
                    ]
                except Exception as e:
                    if strict:
                        raise RuntimeError(f"Failed to load existing parsed results for seed {seed}: {e}")
                    failures.append({"seed": seed, "stage": "load_existing", "error": str(e)})
                    continue
            else:
                # Run vina
                cmd = [
                    self.vina_bin,
                    "--receptor",
                    str(receptor_pdbqt),
                    "--ligand",
                    str(ligand_pdbqt),
                    "--center_x",
                    str(box.center_x),
                    "--center_y",
                    str(box.center_y),
                    "--center_z",
                    str(box.center_z),
                    "--size_x",
                    str(box.size_x),
                    "--size_y",
                    str(box.size_y),
                    "--size_z",
                    str(box.size_z),
                    "--exhaustiveness",
                    str(vina_params.exhaustiveness),
                    "--num_modes",
                    str(vina_params.num_modes),
                    "--energy_range",
                    str(vina_params.energy_range),
                    "--cpu",
                    str(vina_params.cpu),
                    "--seed",
                    str(seed),
                    "--out",
                    str(out_pdbqt),
                ]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                full_log = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
                _safe_write_text(log_txt, full_log)

                params_payload = {
                    "target_group_key": target_group_key,
                    "seed": seed,
                    "cmd": cmd,
                    "returncode": int(proc.returncode),
                    "inputs": {
                        "receptor_pdbqt": str(receptor_pdbqt),
                        "ligand_pdbqt": str(ligand_pdbqt),
                    },
                    "box": asdict(box),
                    "vina_params": asdict(vina_params),
                    "outputs": {
                        "out_pdbqt": str(out_pdbqt),
                        "log_txt": str(log_txt),
                    },
                }
                _safe_write_json(params_json, params_payload)

                if proc.returncode != 0 or (not out_pdbqt.exists()) or out_pdbqt.stat().st_size == 0:
                    msg = f"Vina failed for seed {seed} (returncode={proc.returncode}). See {log_txt}"
                    if strict:
                        raise RuntimeError(msg)
                    failures.append({"seed": seed, "stage": "vina", "error": msg})
                    continue

                poses = parse_vina_log_for_poses(full_log)
                parsed = {
                    "seed": seed,
                    "poses": [asdict(p) for p in poses],
                }
                _safe_write_json(parsed_json, parsed)

                # Write per-run poses CSV
                self._write_run_poses_csv(
                    poses_csv=poses_csv,
                    target_group_key=target_group_key,
                    seed=seed,
                    poses=poses,
                )

            # Build summaries (even if loaded from disk)
            best_aff = None
            best_mode = None
            if poses:
                best_pose = min(poses, key=lambda p: p.affinity)  # more negative is better
                best_aff = best_pose.affinity
                best_mode = best_pose.mode

                if best_global is None or best_aff < best_global[0]:
                    best_global = (best_aff, seed, best_mode, out_pdbqt)

            run_summaries.append(
                {
                    "target_group_key": target_group_key,
                    "seed": seed,
                    "n_poses": len(poses),
                    "best_affinity": best_aff,
                    "best_mode": best_mode,
                }
            )

            for p in poses:
                long_rows.append(
                    {
                        "target_group_key": target_group_key,
                        "seed": seed,
                        "mode": p.mode,
                        "affinity": p.affinity,
                        "rmsd_lb": p.rmsd_lb,
                        "rmsd_ub": p.rmsd_ub,
                    }
                )

        # Write summaries
        runs_long_csv = summary_dir / "runs_long.csv"
        runs_summary_csv = summary_dir / "runs_summary.csv"
        summary_json = summary_dir / "summary.json"

        self._write_long_csv(runs_long_csv, long_rows)
        self._write_summary_csv(runs_summary_csv, run_summaries)

        best_pose_pdbqt = summary_dir / "best_pose.pdbqt"
        best_run_json = summary_dir / "best_run.json"

        summary_payload: Dict[str, object] = {
            "target_group_key": target_group_key,
            "seeds": seeds,
            "vina_bin": self.vina_bin,
            "box": asdict(box),
            "vina_params": asdict(vina_params),
            "n_runs_requested": len(seeds),
            "n_runs_completed": len(run_summaries),
            "n_failures": len(failures),
            "failures": failures,
        }

        if best_global is not None:
            best_aff, best_seed, best_mode, best_out_pdbqt = best_global
            summary_payload["best_overall"] = {
                "best_affinity": best_aff,
                "seed": best_seed,
                "mode": best_mode,
                "out_pdbqt": str(best_out_pdbqt),
            }
            # Copy pose file for convenience
            try:
                shutil.copy2(best_out_pdbqt, best_pose_pdbqt)
                _safe_write_json(
                    best_run_json,
                    {
                        "target_group_key": target_group_key,
                        "best_affinity": best_aff,
                        "seed": best_seed,
                        "mode": best_mode,
                        "source_out_pdbqt": str(best_out_pdbqt),
                        "copied_best_pose_pdbqt": str(best_pose_pdbqt),
                    },
                )
            except Exception as e:
                # Don't fail the whole run if copying fails
                summary_payload["best_overall_copy_error"] = str(e)
        else:
            summary_payload["best_overall"] = None

        _safe_write_json(summary_json, summary_payload)

        return {
            "out_dir": out_dir,
            "runs_long_csv": runs_long_csv,
            "runs_summary_csv": runs_summary_csv,
            "summary_json": summary_json,
            "best_pose_pdbqt": best_pose_pdbqt,
            "best_run_json": best_run_json,
        }

    @staticmethod
    def _write_run_poses_csv(
        *,
        poses_csv: Path,
        target_group_key: str,
        seed: int,
        poses: Sequence[VinaPoseRow],
    ) -> None:
        poses_csv.parent.mkdir(parents=True, exist_ok=True)
        with poses_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["target_group_key", "seed", "mode", "affinity", "rmsd_lb", "rmsd_ub"],
            )
            w.writeheader()
            for p in poses:
                w.writerow(
                    {
                        "target_group_key": target_group_key,
                        "seed": seed,
                        "mode": p.mode,
                        "affinity": p.affinity,
                        "rmsd_lb": p.rmsd_lb,
                        "rmsd_ub": p.rmsd_ub,
                    }
                )

    @staticmethod
    def _write_long_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["target_group_key", "seed", "mode", "affinity", "rmsd_lb", "rmsd_ub"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})

    @staticmethod
    def _write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["target_group_key", "seed", "n_poses", "best_affinity", "best_mode"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
