# --*-- conding:utf-8 --*--
# @time:12/25/25 00:07
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prep.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from .schema import Box
from .pdbio import AtomRecord, parse_pdb_atoms, write_pdb, centroid


def _extract_ca_coords_from_atoms(
    atoms: List[AtomRecord],
    chain_id: str,
    start_resi: int,
    end_resi: int,
) -> Tuple[np.ndarray, List[Tuple[int, str]]]:
    """
    Extract CA coordinates from crystal atoms for residues [start_resi, end_resi] inclusive.
    Returns (coords Nx3, residue_keys list) where residue_keys track residue numbering.
    """
    coords: List[List[float]] = []
    res_keys: List[Tuple[int, str]] = []

    # residue order: increasing resseq, then icode
    # We rely on PDB having CA for each residue in this range.
    for resi in range(start_resi, end_resi + 1):
        # pick CA of this residue
        found = None
        for a in atoms:
            if a.chain == chain_id and a.resseq == resi and (a.name.strip() == "CA"):
                found = a
                break
        if found is None:
            raise ValueError(f"Missing CA for chain {chain_id} residue {resi} in crystal PDB.")
        coords.append([found.x, found.y, found.z])
        res_keys.append((resi, found.icode))

    return np.array(coords, dtype=float), res_keys


def _extract_ca_coords_from_pred_ca_pdb(pred_ca_pdb: Path) -> np.ndarray:
    """
    Read CA-only PDB (produced by dataset.py) and return CA coords in file order.
    """
    atoms, _ = parse_pdb_atoms(pred_ca_pdb, keep_hetatm=True)
    ca = [a for a in atoms if a.record in ("ATOM", "HETATM") and a.name.strip() == "CA"]
    if not ca:
        raise ValueError(f"No CA atoms found in predicted CA PDB: {pred_ca_pdb}")
    coords = np.array([[a.x, a.y, a.z] for a in ca], dtype=float)
    return coords


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rotation R and translation t such that:
      P @ R + t ≈ Q
    where P and Q are Nx3.
    Returns (R, t).
    """
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("Kabsch input shapes must match and be Nx3.")

    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Q.mean(axis=0) - (P.mean(axis=0) @ R)
    return R, t


def build_hybrid_receptor_by_ca_alignment(
    crystal_pdb: Path,
    chain_id: str,
    start_resi: int,
    end_resi: int,
    pred_ca_pdb: Path,
    out_pdb: Path,
    keep_hetatm: bool = True,
) -> Path:
    """
    Build a hybrid receptor:
      - extract CA coords from crystal residues [start_resi, end_resi]
      - extract CA coords from predicted CA-only PDB (should match length)
      - compute rigid transform mapping crystal fragment CA -> predicted CA
      - apply transform to ALL atoms of the fragment residues in crystal
      - write updated PDB (full-atom) to out_pdb

    This makes docking feasible even though prediction is coarse-grained.
    """
    crystal_pdb = Path(crystal_pdb).expanduser().resolve()
    pred_ca_pdb = Path(pred_ca_pdb).expanduser().resolve()
    out_pdb = Path(out_pdb).expanduser().resolve()

    atoms, others = parse_pdb_atoms(crystal_pdb, keep_hetatm=keep_hetatm)

    P, _ = _extract_ca_coords_from_atoms(atoms, chain_id, start_resi, end_resi)
    Q = _extract_ca_coords_from_pred_ca_pdb(pred_ca_pdb)

    if Q.shape[0] != P.shape[0]:
        raise ValueError(
            f"Predicted CA length {Q.shape[0]} != crystal fragment length {P.shape[0]} "
            f"for {crystal_pdb.name} {chain_id}:{start_resi}-{end_resi}"
        )

    R, t = _kabsch(P, Q)

    # Apply transform to fragment atoms
    for a in atoms:
        if a.chain == chain_id and (start_resi <= a.resseq <= end_resi):
            v = np.array([a.x, a.y, a.z], dtype=float)
            v2 = v @ R + t
            a.x, a.y, a.z = float(v2[0]), float(v2[1]), float(v2[2])

    write_pdb(out_pdb, atoms=atoms, other_lines=others)
    return out_pdb


def make_box_from_ligand_centroid(
    complex_pdb: Path,
    ligand_resname: str,
    size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    chain_id: Optional[str] = None,
) -> Box:
    """
    Make a Vina box by taking centroid of all atoms of the reference ligand in the crystal complex.

    Args:
      complex_pdb: crystal PDB containing ligand (HETATM)
      ligand_resname: e.g. GDP or MOV
      size: (sx, sy, sz)
      chain_id: optional chain filter for ligand

    Returns:
      Box(center, size)
    """
    complex_pdb = Path(complex_pdb).expanduser().resolve()
    atoms, _ = parse_pdb_atoms(complex_pdb, keep_hetatm=True)

    lig_atoms = []
    for a in atoms:
        if a.resname == ligand_resname and a.record == "HETATM":
            if chain_id is None or a.chain == chain_id:
                lig_atoms.append(a)

    if not lig_atoms:
        # Some PDB store ligand as ATOM (rare). Try both.
        for a in atoms:
            if a.resname == ligand_resname:
                if chain_id is None or a.chain == chain_id:
                    lig_atoms.append(a)

    if not lig_atoms:
        raise ValueError(f"Cannot find ligand resname '{ligand_resname}' in {complex_pdb}")

    c = centroid([a.coord() for a in lig_atoms])
    sx, sy, sz = size
    return Box(center_x=c[0], center_y=c[1], center_z=c[2], size_x=sx, size_y=sy, size_z=sz)
