# --*-- conding:utf-8 --*--
# @time:12/26/25 01:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:template_fit.py

# docking_verify/template_fit.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json

import numpy as np


# ----------------------------
# PDB atom parsing / writing
# ----------------------------

@dataclass
class PDBAtom:
    record: str           # "ATOM" or "HETATM"
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    icode: str
    x: float
    y: float
    z: float
    occ: float
    bfac: float
    element: str
    charge: str
    raw: str              # original line (for reference)


def _parse_pdb_atom_line(line: str) -> Optional[PDBAtom]:
    if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
        return None
    # Fixed-column parsing per PDB format
    record = line[0:6].strip() or "ATOM"
    serial = int(line[6:11].strip() or "0")
    name = line[12:16]
    altloc = line[16:17]
    resname = line[17:20].strip()
    chain = line[21:22].strip() or " "
    resseq = int(line[22:26].strip() or "0")
    icode = line[26:27]
    x = float(line[30:38].strip() or "0")
    y = float(line[38:46].strip() or "0")
    z = float(line[46:54].strip() or "0")
    occ = float(line[54:60].strip() or "1.00")
    bfac = float(line[60:66].strip() or "0.00")
    element = line[76:78].strip()
    charge = line[78:80].strip()
    return PDBAtom(
        record=record,
        serial=serial,
        name=name,
        altloc=altloc,
        resname=resname,
        chain=chain,
        resseq=resseq,
        icode=icode,
        x=x, y=y, z=z,
        occ=occ,
        bfac=bfac,
        element=element,
        charge=charge,
        raw=line.rstrip("\n"),
    )


def read_pdb_atoms(pdb_path: Union[str, Path]) -> Tuple[List[PDBAtom], List[str]]:
    """
    Returns (atoms, other_lines).
    We drop CONECT records in other_lines by default later when writing.
    """
    p = Path(pdb_path).expanduser().resolve()
    atoms: List[PDBAtom] = []
    other: List[str] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        a = _parse_pdb_atom_line(line)
        if a is not None:
            atoms.append(a)
        else:
            other.append(line)
    return atoms, other


def format_pdb_atom(a: PDBAtom, serial: Optional[int] = None) -> str:
    s = a.serial if serial is None else int(serial)
    # Keep atom name alignment: name field is cols 13-16
    # We'll preserve a.name exactly (it already includes spacing).
    line = (
        f"{a.record:<6}{s:>5} "
        f"{a.name:<4}{a.altloc:1}"
        f"{a.resname:>3} "
        f"{a.chain:1}"
        f"{a.resseq:>4}{a.icode:1}   "
        f"{a.x:>8.3f}{a.y:>8.3f}{a.z:>8.3f}"
        f"{a.occ:>6.2f}{a.bfac:>6.2f}          "
        f"{a.element:>2}{a.charge:>2}"
    )
    return line


def write_pdb(
    out_path: Union[str, Path],
    atoms: List[PDBAtom],
    other_lines: Optional[List[str]] = None,
    drop_conect: bool = True,
) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    if other_lines:
        for l in other_lines:
            if drop_conect and l.startswith("CONECT"):
                continue
            if l.startswith("END"):
                continue
            # keep headers/remarks optionally
            lines.append(l)

    # Renumber serials to avoid broken CONECT references
    for i, a in enumerate(atoms, start=1):
        lines.append(format_pdb_atom(a, serial=i))

    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ----------------------------
# Kabsch alignment
# ----------------------------

def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find R, t such that (P @ R.T) + t ~= Q, minimizing RMSD.
    P, Q: (N,3)
    Returns (R, t), where R is (3,3), t is (3,)
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Bad shapes: P={P.shape}, Q={Q.shape}")

    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P0 = P - cP
    Q0 = Q - cQ

    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cQ - (cP @ R.T)
    return R, t


def apply_rt_to_atoms(atoms: List[PDBAtom], R: np.ndarray, t: np.ndarray) -> List[PDBAtom]:
    out: List[PDBAtom] = []
    for a in atoms:
        v = np.array([a.x, a.y, a.z], dtype=float)
        v2 = (v @ R.T) + t
        out.append(PDBAtom(
            record=a.record,
            serial=a.serial,
            name=a.name,
            altloc=a.altloc,
            resname=a.resname,
            chain=a.chain,
            resseq=a.resseq,
            icode=a.icode,
            x=float(v2[0]), y=float(v2[1]), z=float(v2[2]),
            occ=a.occ, bfac=a.bfac,
            element=a.element,
            charge=a.charge,
            raw=a.raw,
        ))
    return out


# ----------------------------
# Template-fit graft
# ----------------------------

def _get_ca_by_residue(
    atoms: List[PDBAtom],
    chain: str,
    start_resi: int,
    end_resi: int,
) -> Tuple[List[int], np.ndarray]:
    """
    Return (resseq_list, coords Nx3) for CA atoms in [start,end] on chain.
    """
    coords: List[List[float]] = []
    resseqs: List[int] = []
    seen = set()
    for a in atoms:
        if a.chain != chain:
            continue
        if a.resseq < start_resi or a.resseq > end_resi:
            continue
        if a.record.strip() != "ATOM":
            continue
        if a.name.strip() != "CA":
            continue
        key = (a.chain, a.resseq, a.icode.strip())
        if key in seen:
            continue
        seen.add(key)
        resseqs.append(a.resseq)
        coords.append([a.x, a.y, a.z])
    if not coords:
        raise RuntimeError(f"No CA atoms found for chain {chain} {start_resi}-{end_resi}")
    # Ensure order by residue index
    order = np.argsort(np.array(resseqs))
    resseqs = [resseqs[i] for i in order]
    coords = [coords[i] for i in order]
    return resseqs, np.array(coords, dtype=float)


def _load_predicted_ca(decoded_jsonl: Union[str, Path], line_index: int, scale_factor: float) -> Tuple[str, np.ndarray]:
    """
    Read one line from decoded.jsonl and return (sequence, predicted_ca_coords).
    """
    p = Path(decoded_jsonl).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == int(line_index):
                obj = json.loads(line)
                seq = str(obj.get("sequence", ""))
                main_pos = obj.get("main_positions", [])
                if not seq:
                    raise RuntimeError("decoded.jsonl line has empty sequence")
                L = len(seq)
                if len(main_pos) < L:
                    raise RuntimeError(f"main_positions length {len(main_pos)} < sequence length {L}")
                pts = np.array(main_pos[:L], dtype=float) * float(scale_factor)
                return seq, pts
    raise RuntimeError(f"line_index {line_index} out of range in {p}")


def build_hybrid_receptor_by_template_fit(
    base_receptor_pdb: Union[str, Path],
    decoded_jsonl: Union[str, Path],
    line_index: int,
    scale_factor: float,
    chain_id: str,
    start_resi: int,
    end_resi: int,
    out_pdb: Union[str, Path],
) -> Path:
    """
    base_receptor_pdb: crystal receptor with ligand removed (but still has ATOMs for the template fragment)
    decoded_jsonl/line_index/scale_factor: to get predicted CA coords
    Replace residues [start,end] on chain with rigidly transformed crystal fragment atoms (aligned CA->predicted CA).
    """
    atoms, other = read_pdb_atoms(base_receptor_pdb)

    # Load predicted CA coords
    seq, pred_ca = _load_predicted_ca(decoded_jsonl, line_index=line_index, scale_factor=scale_factor)
    L = len(seq)
    expected_len = end_resi - start_resi + 1
    if expected_len != L:
        # not fatal but warn by raising clearer error (better to fail fast)
        raise RuntimeError(
            f"Sequence length {L} does not match residue span {expected_len} "
            f"({start_resi}-{end_resi})."
        )

    # Extract crystal CA coords for that fragment
    resseqs, cry_ca = _get_ca_by_residue(atoms, chain=chain_id, start_resi=start_resi, end_resi=end_resi)
    if len(resseqs) != L or cry_ca.shape[0] != L:
        raise RuntimeError(
            f"Crystal CA count {cry_ca.shape[0]} != sequence length {L}. "
            f"Found residues: {resseqs}"
        )

    # Compute rigid transform
    R, t = kabsch_align(cry_ca, pred_ca)

    # Extract all atoms of the crystal fragment (template), transform them
    frag_atoms = [a for a in atoms if (a.chain == chain_id and start_resi <= a.resseq <= end_resi)]
    if not frag_atoms:
        raise RuntimeError("No fragment atoms found in base receptor to transform (template missing).")
    frag_atoms_t = apply_rt_to_atoms(frag_atoms, R, t)

    # Remove old fragment atoms from base receptor and graft transformed ones
    kept_atoms: List[PDBAtom] = [a for a in atoms if not (a.chain == chain_id and start_resi <= a.resseq <= end_resi)]
    hybrid_atoms = kept_atoms + frag_atoms_t

    # Write hybrid PDB (drop CONECT to avoid OpenBabel 0 molecules converted)
    return write_pdb(out_pdb, hybrid_atoms, other_lines=other, drop_conect=True)
