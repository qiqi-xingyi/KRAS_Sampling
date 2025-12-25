# --*-- conding:utf-8 --*--
# @time:12/25/25 00:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdbio.py

# docking_verify/pdbio.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math


@dataclass
class AtomRecord:
    record: str            # "ATOM" or "HETATM"
    serial: int
    name: str              # atom name
    resname: str           # residue name
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    element: str
    line_raw: str          # original line (for non-coordinate fields preservation)

    def key_residue(self) -> Tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)

    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def parse_pdb_atoms(pdb_path: Path, keep_hetatm: bool = True) -> Tuple[List[AtomRecord], List[str]]:
    """
    Parse ATOM/HETATM lines.
    Return (atoms, other_lines) where other_lines are preserved verbatim.
    """
    pdb_path = Path(pdb_path).expanduser().resolve()
    atoms: List[AtomRecord] = []
    others: List[str] = []

    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = line[0:6].strip()
            if rec not in ("ATOM", "HETATM"):
                others.append(line.rstrip("\n"))
                continue
            if rec == "HETATM" and not keep_hetatm:
                others.append(line.rstrip("\n"))
                continue

            try:
                serial = int(line[6:11])
            except Exception:
                serial = 0
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = (line[21:22] or " ").strip() or "A"

            try:
                resseq = int(line[22:26])
            except Exception:
                resseq = 0
            icode = (line[26:27] or " ").strip()

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                # If coordinates missing, keep as 0
                x = y = z = 0.0

            element = (line[76:78].strip() if len(line) >= 78 else "").strip()
            if not element and name:
                # fallback
                element = name[0].upper()

            atoms.append(
                AtomRecord(
                    record=rec,
                    serial=serial,
                    name=name,
                    resname=resname,
                    chain=chain,
                    resseq=resseq,
                    icode=icode,
                    x=x, y=y, z=z,
                    element=element,
                    line_raw=line.rstrip("\n"),
                )
            )

    return atoms, others


def write_pdb(
    out_path: Path,
    atoms: List[AtomRecord],
    other_lines: Optional[List[str]] = None,
) -> None:
    """
    Write PDB, preserving non-ATOM lines optionally.
    ATOM/HETATM lines are re-rendered with updated coordinates.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        if other_lines:
            for l in other_lines:
                f.write(l.rstrip("\n") + "\n")

        serial_counter = 1
        for a in atoms:
            serial = a.serial if a.serial > 0 else serial_counter
            serial_counter += 1

            # Render ATOM/HETATM with standard columns
            line = (
                f"{a.record:<6}{serial:5d} "
                f"{a.name:<4}"
                f"{' ':1}"
                f"{a.resname:>3} "
                f"{a.chain:1}"
                f"{a.resseq:4d}"
                f"{(a.icode or ' '):1}"
                f"{' ':3}"
                f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}"
                f"{1.00:6.2f}{0.00:6.2f}"
                f"{' ':10}"
                f"{a.element:>2}"
            )
            f.write(line + "\n")

        f.write("END\n")


def centroid(coords: Iterable[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    xs, ys, zs = [], [], []
    for x, y, z in coords:
        xs.append(x); ys.append(y); zs.append(z)
    if not xs:
        return (0.0, 0.0, 0.0)
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
