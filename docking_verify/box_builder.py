# --*-- conding:utf-8 --*--
# @time:12/26/25 20:26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:box_builder.py.py


from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# We reuse the same DockingBox layout as vina_runner.py for seamless integration.
@dataclass(frozen=True)
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


# -----------------------------
# Lightweight PDB parsing
# -----------------------------
def _is_atom_record(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def _record_name(line: str) -> str:
    return line[0:6].strip()


def _resname(line: str) -> str:
    return line[17:20].strip()


def _chain_id(line: str) -> str:
    return (line[21].strip() if len(line) > 21 else "") or " "


def _resseq(line: str) -> int:
    return int(line[22:26].strip())


def _icode(line: str) -> str:
    return line[26].strip() if len(line) > 26 and line[26].strip() else " "


def _altloc(line: str) -> str:
    return line[16].strip() if len(line) > 16 else ""


def _occupancy(line: str) -> float:
    try:
        return float(line[54:60].strip())
    except Exception:
        return 1.0


def _xyz(line: str) -> Tuple[float, float, float]:
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    return (x, y, z)


def _atom_key(line: str) -> Tuple[str, str, int, str, str, str]:
    # (record, chain, resseq, icode, resname, atom_name)
    atom_name = line[12:16].strip()
    return (_record_name(line), _chain_id(line), _resseq(line), _icode(line), _resname(line), atom_name)


def _filter_altloc_keep_best(atom_lines: Sequence[str]) -> List[str]:
    """
    Keep best-occupancy altloc per atom identity, prefer altloc ' ' then 'A' if tie.
    """
    best: Dict[Tuple[str, str, int, str, str, str], Tuple[float, int, str]] = {}
    chosen: Dict[Tuple[str, str, int, str, str, str], str] = {}

    def rank(alt: str) -> int:
        alt = alt or " "
        if alt == " ":
            return 0
        if alt == "A":
            return 1
        return 2

    for ln in atom_lines:
        if not _is_atom_record(ln):
            continue
        key = _atom_key(ln)
        occ = _occupancy(ln)
        alt = _altloc(ln) or " "
        cand = (occ, rank(alt), alt)
        if key not in best:
            best[key] = cand
            chosen[key] = ln
        else:
            prev = best[key]
            if (cand[0] > prev[0]) or (cand[0] == prev[0] and cand[1] < prev[1]):
                best[key] = cand
                chosen[key] = ln

    out: List[str] = []
    for ln in atom_lines:
        if not _is_atom_record(ln):
            continue
        key = _atom_key(ln)
        if chosen.get(key) == ln:
            out.append(ln if ln.endswith("\n") else ln + "\n")
    return out


# -----------------------------
# Box builder
# -----------------------------
class LigandCenteredBoxBuilder:
    """
    Build a Vina docking box centered on the ligand (by resname) in a complex PDB.

    Integration points:
      - Feed `embedded.pdb` (from 10_embed) OR the original experimental complex PDB.
      - Use the same ligand_resname used by PDBQTPreparer.
      - Returns DockingBox compatible with VinaDockingRunner.

    Output (optional):
      docking_result/25_box/<group_key>/box.json
    """

    def __init__(
        self,
        result_root: Path = Path("docking_result"),
        step_dirname: str = "25_box",
        default_margin: float = 10.0,
        default_min_size: float = 20.0,
        default_max_size: Optional[float] = None,
        archive: bool = True,
    ) -> None:
        self.result_root = Path(result_root)
        self.step_dir = self.result_root / step_dirname
        self.default_margin = float(default_margin)
        self.default_min_size = float(default_min_size)
        self.default_max_size = float(default_max_size) if default_max_size is not None else None
        self.archive = bool(archive)

    def build(
        self,
        *,
        target_group_key: str,
        complex_pdb: Path,
        ligand_resname: str,
        margin: Optional[float] = None,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        fixed_size: Optional[Tuple[float, float, float]] = None,
        select_ligand_instance: str = "largest",  # "largest" or "first"
    ) -> DockingBox:
        """
        Args:
          target_group_key: for organizing outputs (no effect on computation)
          complex_pdb: PDB containing the ligand (embedded.pdb or original complex)
          ligand_resname: ligand residue name (e.g. GDP, MOV)
          margin: Å added around ligand bounding box on each axis
          min_size: minimum box size (Å) per axis
          max_size: maximum box size (Å) per axis (optional clamp)
          fixed_size: if provided, ignore ligand bbox sizing and use this (size_x,size_y,size_z);
                      center still comes from ligand centroid.
          select_ligand_instance:
            - "largest": choose the residue instance with the most atoms for that resname
            - "first": choose the first encountered

        Returns:
          DockingBox(center_x,center_y,center_z,size_x,size_y,size_z)
        """
        complex_pdb = Path(complex_pdb)
        if not complex_pdb.exists():
            raise FileNotFoundError(f"complex_pdb not found: {complex_pdb}")

        margin = float(self.default_margin if margin is None else margin)
        min_size = float(self.default_min_size if min_size is None else min_size)
        max_size_val = self.default_max_size if max_size is None else float(max_size)

        atom_lines = [ln for ln in complex_pdb.read_text().splitlines(keepends=True) if _is_atom_record(ln)]
        atom_lines = _filter_altloc_keep_best(atom_lines)

        # Collect ligand atoms by residue instance (chain, resseq, icode)
        instances: Dict[Tuple[str, int, str], List[Tuple[float, float, float]]] = {}
        for ln in atom_lines:
            if not ln.startswith("HETATM"):
                continue
            if _resname(ln) != ligand_resname:
                continue
            key = (_chain_id(ln), _resseq(ln), _icode(ln))
            instances.setdefault(key, []).append(_xyz(ln))

        if not instances:
            raise ValueError(f"No ligand HETATM found for resname={ligand_resname} in {complex_pdb}")

        # Choose instance
        if select_ligand_instance == "first":
            chosen_key = next(iter(instances.keys()))
        elif select_ligand_instance == "largest":
            chosen_key = max(instances.keys(), key=lambda k: len(instances[k]))
        else:
            raise ValueError(f"Unsupported select_ligand_instance: {select_ligand_instance}")

        coords = instances[chosen_key]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]

        # Center: ligand centroid
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)

        if fixed_size is not None:
            sx, sy, sz = float(fixed_size[0]), float(fixed_size[1]), float(fixed_size[2])
        else:
            # Size from bounding box + margin
            dx = (max(xs) - min(xs)) + 2.0 * margin
            dy = (max(ys) - min(ys)) + 2.0 * margin
            dz = (max(zs) - min(zs)) + 2.0 * margin
            sx, sy, sz = max(dx, min_size), max(dy, min_size), max(dz, min_size)

            if max_size_val is not None:
                sx, sy, sz = min(sx, max_size_val), min(sy, max_size_val), min(sz, max_size_val)

        box = DockingBox(center_x=cx, center_y=cy, center_z=cz, size_x=sx, size_y=sy, size_z=sz)

        if self.archive:
            out_dir = self.step_dir / target_group_key
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "target_group_key": target_group_key,
                "complex_pdb": str(complex_pdb),
                "ligand_resname": ligand_resname,
                "selected_ligand_instance": {"chain": chosen_key[0], "resseq": chosen_key[1], "icode": chosen_key[2]},
                "policy": {
                    "center": "ligand_centroid",
                    "size": "fixed" if fixed_size is not None else "ligand_bbox_plus_margin",
                    "margin": margin,
                    "min_size": min_size,
                    "max_size": max_size_val,
                    "fixed_size": list(fixed_size) if fixed_size is not None else None,
                },
                "box": asdict(box),
            }
            (out_dir / "box.json").write_text(json.dumps(payload, indent=2))

        return box
