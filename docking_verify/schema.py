# --*-- conding:utf-8 --*--
# @time:12/25/25 00:03
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:schema.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json

# Optional YAML support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


@dataclass(frozen=True)
class Case:
    """
    A docking case produced by docking_verify.dataset (cases.csv).
    """
    case_id: str
    pdb_path: Path               # reference PDB path (crystal full protein)
    ref_pdb: str                 # original name in summary
    chain_id: str
    start_resi: int
    end_resi: int
    pred_ca_pdb: Optional[Path]  # CA-only predicted fragment PDB
    pred_ca_json: Optional[Path] # scaled CA coordinates json (optional)
    ligand_resname: str

    # provenance
    decoded_file: Optional[Path] = None
    line_index: Optional[int] = None
    min_rmsd: Optional[float] = None
    best_sequence: Optional[str] = None
    scale_factor: Optional[float] = None
    scale_mode: Optional[str] = None
    extract_status: Optional[str] = None


@dataclass(frozen=True)
class Box:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    def as_vina_args(self) -> List[str]:
        return [
            "--center_x", str(self.center_x),
            "--center_y", str(self.center_y),
            "--center_z", str(self.center_z),
            "--size_x", str(self.size_x),
            "--size_y", str(self.size_y),
            "--size_z", str(self.size_z),
        ]


@dataclass(frozen=True)
class VinaParams:
    exhaustiveness: int = 16
    num_modes: int = 20
    energy_range: int = 3
    cpu: int = 8
    seeds: Optional[List[int]] = None  # if None, caller provides one seed


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_cases_csv(path: Path) -> List[Case]:
    """
    Read docking_data/cases.csv produced by docking_verify.dataset.
    """
    path = Path(path).expanduser().resolve()
    cases: List[Case] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty cases.csv or missing header: {path}")

        required = [
            "case_id", "pdb_path", "ref_pdb", "chain_id",
            "start_resi", "end_resi", "ligand_resname",
        ]
        for k in required:
            if k not in reader.fieldnames:
                raise ValueError(f"Missing column '{k}' in {path}")

        for d in reader:
            pdb_path = Path(d["pdb_path"]).expanduser().resolve()

            pred_ca_pdb = d.get("pred_ca_pdb", "") or ""
            pred_ca_json = d.get("pred_ca_json", "") or ""

            cases.append(
                Case(
                    case_id=d["case_id"],
                    pdb_path=pdb_path,
                    ref_pdb=d.get("ref_pdb", ""),
                    chain_id=(d.get("chain_id", "") or "").strip() or "A",
                    start_resi=int(d["start_resi"]),
                    end_resi=int(d["end_resi"]),
                    pred_ca_pdb=Path(pred_ca_pdb).expanduser().resolve() if pred_ca_pdb else None,
                    pred_ca_json=Path(pred_ca_json).expanduser().resolve() if pred_ca_json else None,
                    ligand_resname=(d.get("ligand_resname", "") or "").strip(),
                    decoded_file=Path(d["decoded_file"]).expanduser().resolve() if d.get("decoded_file") else None,
                    line_index=_safe_int(d.get("line_index", "")),
                    min_rmsd=_safe_float(d.get("min_rmsd", "")),
                    best_sequence=d.get("best_sequence"),
                    scale_factor=_safe_float(d.get("scale_factor", "")),
                    scale_mode=d.get("scale_mode"),
                    extract_status=d.get("extract_status"),
                )
            )
    return cases


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load a config file (YAML preferred; JSON supported).
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    txt = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed but a YAML config was provided.")
        return yaml.safe_load(txt) or {}
    if suffix == ".json":
        return json.loads(txt)

    # auto-detect: try YAML then JSON
    if yaml is not None:
        try:
            return yaml.safe_load(txt) or {}
        except Exception:
            pass
    return json.loads(txt)
