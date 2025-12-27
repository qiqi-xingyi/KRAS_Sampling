# --*-- conding:utf-8 --*--
# @time:12/26/25 19:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:embedder.py

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class CaseRow:
    case_id: str
    pdb_path: str
    ref_pdb: str
    chain_id: str
    start_resi: int
    end_resi: int
    pred_ca_pdb: str
    pred_ca_json: str
    ligand_resname: str

    @property
    def group_key(self) -> str:
        # e.g. 4LPK_WT_1 -> 4LPK_WT
        parts = self.case_id.rsplit("_", 1)
        return parts[0] if len(parts) == 2 else self.case_id

    @property
    def fragment_index(self) -> Optional[int]:
        # e.g. *_1 -> 1
        parts = self.case_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return None


# -----------------------------
# Minimal PDB parser/writer
# -----------------------------
@dataclass
class PDBAtomLine:
    raw: str
    record: str  # ATOM/HETATM
    chain_id: str
    resseq: int
    icode: str
    atom_name: str
    x: float
    y: float
    z: float

    def coord(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def set_coord(self, xyz: np.ndarray) -> None:
        self.x, self.y, self.z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    def format_line(self) -> str:
        # PDB fixed columns: x[30:38], y[38:46], z[46:54]
        s = self.raw.rstrip("\n")
        if len(s) < 54:
            s = s.ljust(80)
        xyz = f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
        out = s[:30] + xyz + s[54:]
        return out + "\n"


def _parse_pdb_atom_line(line: str) -> Optional[PDBAtomLine]:
    rec = line[0:6].strip()
    if rec not in ("ATOM", "HETATM"):
        return None
    try:
        atom_name = line[12:16].strip()
        chain_id = line[21].strip() or " "
        resseq = int(line[22:26].strip())
        icode = line[26].strip() or " "
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except Exception:
        return None
    return PDBAtomLine(
        raw=line,
        record=rec,
        chain_id=chain_id,
        resseq=resseq,
        icode=icode,
        atom_name=atom_name,
        x=x,
        y=y,
        z=z,
    )


def read_pdb(pdb_path: Path) -> Tuple[List[str], List[PDBAtomLine]]:
    raw_lines = pdb_path.read_text().splitlines(keepends=True)
    atoms: List[PDBAtomLine] = []
    for ln in raw_lines:
        a = _parse_pdb_atom_line(ln)
        if a is not None:
            atoms.append(a)
    return raw_lines, atoms


def write_pdb(out_path: Path, raw_lines: List[str], atoms: List[PDBAtomLine]) -> None:
    atom_iter = iter(atoms)
    updated: List[str] = []
    for ln in raw_lines:
        if _parse_pdb_atom_line(ln) is None:
            updated.append(ln if ln.endswith("\n") else ln + "\n")
            continue
        a2 = next(atom_iter)
        updated.append(a2.format_line())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(updated))


def extract_scaffold_ca(
    atoms: List[PDBAtomLine],
    chain_id: str,
    start_resi: int,
    end_resi: int,
) -> np.ndarray:
    ca_by_res: Dict[int, np.ndarray] = {}
    for a in atoms:
        if a.chain_id != chain_id:
            continue
        if not (start_resi <= a.resseq <= end_resi):
            continue
        if a.atom_name == "CA":
            ca_by_res[a.resseq] = a.coord()

    coords: List[np.ndarray] = []
    missing: List[int] = []
    for r in range(start_resi, end_resi + 1):
        if r not in ca_by_res:
            missing.append(r)
        else:
            coords.append(ca_by_res[r])

    if missing:
        raise ValueError(f"Missing CA atoms in scaffold chain={chain_id} residues={missing}")
    return np.stack(coords, axis=0)


def select_atoms_in_segment(
    atoms: List[PDBAtomLine],
    chain_id: str,
    start_resi: int,
    end_resi: int,
) -> List[PDBAtomLine]:
    return [a for a in atoms if a.chain_id == chain_id and start_resi <= a.resseq <= end_resi]


# -----------------------------
# Kabsch alignment
# -----------------------------
def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) that maps P -> Q by rigid transform:
      x' = R @ x + t
    P, Q: (N, 3)
    """
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Invalid shapes: P{P.shape}, Q{Q.shape}")

    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    X = P - cP
    Y = Q - cQ

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cQ - (R @ cP)
    return R, t


def apply_rt(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ xyz) + t


# -----------------------------
# Embedder component
# -----------------------------
class KabschFragmentEmbedder:
    """
    Component: embed predicted CA fragments into an experimental scaffold PDB
    by segment-level Kabsch rigid alignment (template all-atom segment moved as rigid body).

    Supports mutation workflow:
      - Always embed fragments {1,2,3} into the TARGET scaffold.
      - If the target group lacks fragment idx (e.g. only _1 exists for G12C/G12D),
        the missing fragment predictions (and segment definitions) are taken from
        a fallback group (e.g. WT).

    NOTE (chain filtering):
      - Some mutant PDBs may not contain chain B (or any non-A chain fragments).
      - You can restrict embedding to allowed scaffold chains (default: ("A",)).
      - Fragments whose CaseRow.chain_id is not allowed are skipped (unless strict=True and you choose to enforce).

    Output layout (embedding stage only):
      docking_result/
        10_embed/
          <target_group_key>/
            scaffold_original.pdb
            embedded.pdb
            transforms.json
            embed_chain_policy.json
            fragments/
              <target_group_key>_frag1__pred_from_<case_id>.json
              <target_group_key>_frag1__pred_from_<case_id>.pdb
              ...
    """

    def __init__(
        self,
        cases_csv: Path,
        result_root: Path = Path("docking_result"),
        step_dirname: str = "10_embed",
        archive_inputs: bool = True,
    ) -> None:
        self.cases_csv = Path(cases_csv)
        self.result_root = Path(result_root)
        self.step_dir = self.result_root / step_dirname
        self.archive_inputs = bool(archive_inputs)

        # Pre-index for fast lookup
        self._cases: List[CaseRow] = self.load_cases()
        self._index: Dict[Tuple[str, int], CaseRow] = self._build_index(self._cases)

    # ---------- indexing ----------
    def load_cases(self) -> List[CaseRow]:
        rows: List[CaseRow] = []
        with self.cases_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    CaseRow(
                        case_id=r["case_id"],
                        pdb_path=r["pdb_path"],
                        ref_pdb=r.get("ref_pdb", ""),
                        chain_id=r["chain_id"],
                        start_resi=int(r["start_resi"]),
                        end_resi=int(r["end_resi"]),
                        pred_ca_pdb=r["pred_ca_pdb"],
                        pred_ca_json=r["pred_ca_json"],
                        ligand_resname=r.get("ligand_resname", ""),
                    )
                )
        return rows

    @staticmethod
    def _build_index(cases: Iterable[CaseRow]) -> Dict[Tuple[str, int], CaseRow]:
        idx: Dict[Tuple[str, int], CaseRow] = {}
        for c in cases:
            fi = c.fragment_index
            if fi is None:
                continue
            idx[(c.group_key, fi)] = c
        return idx

    def _get_case(self, group_key: str, frag_idx: int) -> Optional[CaseRow]:
        return self._index.get((group_key, frag_idx))

    # ---------- core ----------
    def embed_group(
        self,
        target_group_key: str,
        *,
        required_fragment_indices: Iterable[int] = (1, 2, 3),
        fallback_group_key: Optional[str] = None,
        strict: bool = True,
        allowed_scaffold_chains: Tuple[str, ...] = ("A",),
        verbose: bool = True,
    ) -> Path:
        """
        Embed required fragments into the TARGET scaffold.

        Resolution rule per fragment idx:
          - prefer case row (target_group_key, idx)
          - else if fallback_group_key provided, use (fallback_group_key, idx)
          - else error

        The scaffold PDB is always taken from the TARGET group (must exist at least one fragment in target).

        Chain filtering:
          - If src_case.chain_id not in allowed_scaffold_chains, that fragment will be skipped.
          - If all required fragments are skipped and strict=True -> raise ValueError
        """
        req = list(required_fragment_indices)

        allowed_set = {c.strip() for c in allowed_scaffold_chains if c and c.strip()}
        if not allowed_set:
            raise ValueError("allowed_scaffold_chains is empty")

        # Determine scaffold from any existing target fragment row
        scaffold_case: Optional[CaseRow] = None
        for i in req:
            c = self._get_case(target_group_key, i)
            if c is not None:
                scaffold_case = c
                break
        if scaffold_case is None:
            msg = f"Target group has no cases in CSV: {target_group_key}"
            raise KeyError(msg)

        scaffold_path = Path(scaffold_case.pdb_path)
        if not scaffold_path.exists():
            raise FileNotFoundError(f"Scaffold PDB not found: {scaffold_path}")

        out_dir = self.step_dir / target_group_key
        frag_dir = out_dir / "fragments"
        out_dir.mkdir(parents=True, exist_ok=True)
        frag_dir.mkdir(parents=True, exist_ok=True)

        # Keep a copy of the original scaffold for traceability
        scaffold_copy = out_dir / "scaffold_original.pdb"
        if not scaffold_copy.exists():
            shutil.copy2(scaffold_path, scaffold_copy)

        if verbose:
            print(
                f"[Embed] target_group={target_group_key} scaffold={scaffold_path} "
                f"fallback={fallback_group_key} allowed_scaffold_chains={sorted(list(allowed_set))} "
                f"required_fragments={req}"
            )

        raw_lines, atoms = read_pdb(scaffold_path)

        transforms: Dict[str, Dict[str, object]] = {}
        used_sources: Dict[int, str] = {}  # frag_idx -> case_id

        skipped_fragments: List[Dict[str, object]] = []
        embedded_fragments: List[Dict[str, object]] = []

        for frag_idx in req:
            src_case = self._get_case(target_group_key, frag_idx)
            if src_case is None and fallback_group_key is not None:
                src_case = self._get_case(fallback_group_key, frag_idx)

            if src_case is None:
                if strict:
                    raise KeyError(
                        f"Missing fragment {frag_idx} for target={target_group_key} "
                        f"(fallback={fallback_group_key})"
                    )
                else:
                    if verbose:
                        print(f"[Embed] SKIP frag={frag_idx} (missing in target and fallback)")
                    skipped_fragments.append(
                        {"frag_idx": frag_idx, "reason": "missing_in_target_and_fallback"}
                    )
                    continue

            # Segment definition comes from the selected src_case
            chain_id = src_case.chain_id.strip() or " "
            start_resi = src_case.start_resi
            end_resi = src_case.end_resi

            # Chain filter: skip non-allowed scaffold chains (e.g., chain B)
            if chain_id not in allowed_set:
                if verbose:
                    print(
                        f"[Embed] SKIP frag={frag_idx} source={src_case.case_id} "
                        f"(chain={chain_id} not in allowed_scaffold_chains)"
                    )
                skipped_fragments.append(
                    {
                        "frag_idx": frag_idx,
                        "case_id": src_case.case_id,
                        "chain_id": chain_id,
                        "start_resi": start_resi,
                        "end_resi": end_resi,
                        "reason": "chain_not_allowed",
                    }
                )
                continue

            pred_json_path = Path(src_case.pred_ca_json)
            pred_pdb_path = Path(src_case.pred_ca_pdb)

            if not pred_json_path.exists():
                raise FileNotFoundError(f"Pred CA json not found: {pred_json_path}")
            if not pred_pdb_path.exists():
                raise FileNotFoundError(f"Pred CA pdb not found: {pred_pdb_path}")

            used_sources[frag_idx] = src_case.case_id

            if self.archive_inputs:
                json_dst = frag_dir / f"{target_group_key}_frag{frag_idx}__pred_from_{src_case.case_id}.json"
                pdb_dst = frag_dir / f"{target_group_key}_frag{frag_idx}__pred_from_{src_case.case_id}.pdb"
                shutil.copy2(pred_json_path, json_dst)
                shutil.copy2(pred_pdb_path, pdb_dst)

            pred = json.loads(pred_json_path.read_text())
            Q = np.array(pred["ca_positions_angstrom"], dtype=np.float64)

            P = extract_scaffold_ca(
                atoms=atoms,
                chain_id=chain_id,
                start_resi=start_resi,
                end_resi=end_resi,
            )

            if P.shape[0] != Q.shape[0]:
                raise ValueError(
                    f"CA length mismatch for target={target_group_key} frag={frag_idx} "
                    f"(segment {chain_id}:{start_resi}-{end_resi}): "
                    f"scaffold={P.shape[0]} pred={Q.shape[0]} (pred source {src_case.case_id})"
                )

            R, t = kabsch(P, Q)

            seg_atoms = select_atoms_in_segment(atoms, chain_id, start_resi, end_resi)
            if not seg_atoms:
                raise ValueError(
                    f"No atoms found in scaffold for segment {chain_id}:{start_resi}-{end_resi} "
                    f"(target={target_group_key} frag={frag_idx})"
                )

            for a in seg_atoms:
                a.set_coord(apply_rt(a.coord(), R, t))

            transforms[f"frag_{frag_idx}"] = {
                "target_group": target_group_key,
                "pred_source_case_id": src_case.case_id,
                "pred_source_group": src_case.group_key,
                "chain_id": chain_id,
                "start_resi": start_resi,
                "end_resi": end_resi,
                "R": R.tolist(),
                "t": t.tolist(),
                "target_scaffold_pdb": str(scaffold_path),
                "pred_ca_json": str(pred_json_path),
            }

            embedded_fragments.append(
                {
                    "frag_idx": frag_idx,
                    "case_id": src_case.case_id,
                    "chain_id": chain_id,
                    "start_resi": start_resi,
                    "end_resi": end_resi,
                }
            )

            if verbose:
                print(
                    f"[Embed] OK  frag={frag_idx} source={src_case.case_id} "
                    f"segment={chain_id}:{start_resi}-{end_resi} "
                    f"n_atoms_in_segment={len(seg_atoms)} n_ca={P.shape[0]}"
                )

        if strict and len(embedded_fragments) == 0:
            raise ValueError(
                f"No fragments were embedded for target={target_group_key} "
                f"after chain filtering (allowed_scaffold_chains={sorted(list(allowed_set))})."
            )

        embedded_pdb = out_dir / "embedded.pdb"
        write_pdb(embedded_pdb, raw_lines, atoms)

        (out_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))

        policy_path = out_dir / "embed_chain_policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "target_group_key": target_group_key,
                    "scaffold_pdb": str(scaffold_path),
                    "fallback_group_key": fallback_group_key,
                    "required_fragment_indices": req,
                    "allowed_scaffold_chains": sorted(list(allowed_set)),
                    "embedded_fragments": embedded_fragments,
                    "skipped_fragments": skipped_fragments,
                },
                indent=2,
            )
        )

        if verbose:
            print(
                f"[Embed] DONE target_group={target_group_key} "
                f"embedded={len(embedded_fragments)} skipped={len(skipped_fragments)} "
                f"out={embedded_pdb}"
            )

        return embedded_pdb
