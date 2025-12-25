# --*-- conding:utf-8 --*--
# @time:12/25/25 00:08
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:analyze.py


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import json
import statistics

from .schema import Box


from .pdbio import parse_pdb_atoms, centroid, distance


@dataclass
class RunMetric:
    case_id: str
    receptor_type: str
    seed: int
    best_score: Optional[float]
    pocket_hit: Optional[bool]
    out_pdbqt: Path
    meta_json: Path


@dataclass
class CaseAggregate:
    case_id: str
    receptor_type: str
    n_runs: int
    best_score_mean: Optional[float]
    best_score_std: Optional[float]
    pocket_hit_rate: Optional[float]


def _read_meta(meta_json: Path) -> Dict:
    meta_json = Path(meta_json).expanduser().resolve()
    return json.loads(meta_json.read_text(encoding="utf-8"))


def _extract_ligand_coords_from_pdbqt(pdbqt_path: Path) -> List[Tuple[float, float, float]]:
    """
    Vina out.pdbqt contains ligand coordinates as ATOM/HETATM.
    We parse all ATOM/HETATM coords.
    """
    atoms, _ = parse_pdb_atoms(pdbqt_path, keep_hetatm=True)
    return [a.coord() for a in atoms if a.record in ("ATOM", "HETATM")]


def compute_pocket_hit(coords: List[Tuple[float, float, float]], box: Box, margin: float = 0.0) -> bool:
    """
    Simple criterion: ligand centroid must fall inside box (with optional margin).
    """
    c = centroid(coords)
    hx, hy, hz = box.size_x / 2.0 + margin, box.size_y / 2.0 + margin, box.size_z / 2.0 + margin
    return (
        abs(c[0] - box.center_x) <= hx and
        abs(c[1] - box.center_y) <= hy and
        abs(c[2] - box.center_z) <= hz
    )


def analyze_vina_outputs(
    out_root: Path,
    case_id: str,
    receptor_type: str,
    box: Box,
    margin: float = 0.0,
) -> List[RunMetric]:
    """
    Analyze a directory like:
      out_root/seed_123/out.pdbqt
      out_root/seed_123/meta.json
    """
    out_root = Path(out_root).expanduser().resolve()
    metrics: List[RunMetric] = []

    for seed_dir in sorted(out_root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed_str = seed_dir.name.replace("seed_", "").strip()
        try:
            seed = int(seed_str)
        except Exception:
            continue

        out_pdbqt = seed_dir / "out.pdbqt"
        meta_json = seed_dir / "meta.json"
        if not meta_json.exists():
            continue

        meta = _read_meta(meta_json)
        scores = meta.get("scores", [])
        best_score = None
        if isinstance(scores, list) and scores:
            try:
                best_score = float(min(scores))
            except Exception:
                best_score = None

        pocket_hit = None
        if out_pdbqt.exists():
            coords = _extract_ligand_coords_from_pdbqt(out_pdbqt)
            if coords:
                pocket_hit = compute_pocket_hit(coords, box, margin=margin)

        metrics.append(
            RunMetric(
                case_id=case_id,
                receptor_type=receptor_type,
                seed=seed,
                best_score=best_score,
                pocket_hit=pocket_hit,
                out_pdbqt=out_pdbqt,
                meta_json=meta_json,
            )
        )

    return metrics


def aggregate_case_metrics(run_metrics: List[RunMetric]) -> List[CaseAggregate]:
    """
    Aggregate by (case_id, receptor_type).
    """
    groups: Dict[Tuple[str, str], List[RunMetric]] = {}
    for rm in run_metrics:
        groups.setdefault((rm.case_id, rm.receptor_type), []).append(rm)

    out: List[CaseAggregate] = []
    for (case_id, receptor_type), arr in groups.items():
        scores = [x.best_score for x in arr if x.best_score is not None]
        hits = [x.pocket_hit for x in arr if x.pocket_hit is not None]

        mean = statistics.mean(scores) if scores else None
        std = statistics.pstdev(scores) if len(scores) >= 2 else (0.0 if len(scores) == 1 else None)
        hit_rate = (sum(1 for h in hits if h) / len(hits)) if hits else None

        out.append(
            CaseAggregate(
                case_id=case_id,
                receptor_type=receptor_type,
                n_runs=len(arr),
                best_score_mean=mean,
                best_score_std=std,
                pocket_hit_rate=hit_rate,
            )
        )
    return out


def write_summary_csv(run_metrics: List[RunMetric], out_csv: Path) -> Path:
    out_csv = Path(out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id", "receptor_type", "seed",
                "best_score", "pocket_hit",
                "out_pdbqt", "meta_json",
            ],
        )
        w.writeheader()
        for rm in run_metrics:
            w.writerow(
                {
                    "case_id": rm.case_id,
                    "receptor_type": rm.receptor_type,
                    "seed": rm.seed,
                    "best_score": "" if rm.best_score is None else rm.best_score,
                    "pocket_hit": "" if rm.pocket_hit is None else int(bool(rm.pocket_hit)),
                    "out_pdbqt": str(rm.out_pdbqt),
                    "meta_json": str(rm.meta_json),
                }
            )
    return out_csv


def write_aggregate_json(aggs: List[CaseAggregate], out_json: Path) -> Path:
    out_json = Path(out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for a in aggs:
        payload.append(
            {
                "case_id": a.case_id,
                "receptor_type": a.receptor_type,
                "n_runs": a.n_runs,
                "best_score_mean": a.best_score_mean,
                "best_score_std": a.best_score_std,
                "pocket_hit_rate": a.pocket_hit_rate,
            }
        )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_json
