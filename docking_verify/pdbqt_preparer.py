# --*-- conding:utf-8 --*--
# @time:12/26/25 19:53
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pdbqt_preparer.py

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# PDB utilities (lightweight)
# -----------------------------
_WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD"}
_DEFAULT_SKIP_HETERO = _WATER_RESNAMES


@dataclass(frozen=True)
class ResidueID:
    chain_id: str
    resseq: int
    icode: str = " "

    def key(self) -> Tuple[str, int, str]:
        return (self.chain_id, self.resseq, self.icode)


def _is_atom_record(line: str) -> bool:
    return line.startswith("ATOM") or line.startswith("HETATM")


def _record_name(line: str) -> str:
    return line[0:6].strip()


def _altloc(line: str) -> str:
    return line[16].strip() if len(line) > 16 else ""


def _atom_name(line: str) -> str:
    return line[12:16].strip()


def _resname(line: str) -> str:
    return line[17:20].strip()


def _chain_id(line: str) -> str:
    return (line[21].strip() if len(line) > 21 else "") or " "


def _resseq(line: str) -> int:
    return int(line[22:26].strip())


def _icode(line: str) -> str:
    return line[26].strip() if len(line) > 26 and line[26].strip() else " "


def _occupancy(line: str) -> float:
    try:
        return float(line[54:60].strip())
    except Exception:
        return 1.0


def _residue_id(line: str) -> ResidueID:
    return ResidueID(chain_id=_chain_id(line), resseq=_resseq(line), icode=_icode(line))


def _ensure_trailing_newline(lines: List[str]) -> List[str]:
    return [ln if ln.endswith("\n") else ln + "\n" for ln in lines]


def _filter_altloc_keep_best(lines: Sequence[str]) -> List[str]:
    """
    Resolve alternate locations by keeping the best-occupancy altloc for each atom identity.
    Atom identity key: (record, chain, resseq, icode, resname, atom_name)
    If occupancy ties, prefer altloc=' ' then 'A' then others.
    """
    best: Dict[Tuple[str, str, int, str, str, str], Tuple[float, int, str]] = {}
    chosen_line: Dict[Tuple[str, str, int, str, str, str], str] = {}

    def pref_rank(a: str) -> int:
        if a == "":
            a = " "
        if a == " ":
            return 0
        if a == "A":
            return 1
        return 2

    for ln in lines:
        if not _is_atom_record(ln):
            continue
        rec = _record_name(ln)
        ch = _chain_id(ln)
        rs = _resseq(ln)
        ic = _icode(ln)
        rn = _resname(ln)
        an = _atom_name(ln)
        key = (rec, ch, rs, ic, rn, an)
        occ = _occupancy(ln)
        alt = _altloc(ln) or " "
        cand = (occ, pref_rank(alt), alt)

        if key not in best:
            best[key] = cand
            chosen_line[key] = ln
        else:
            prev = best[key]
            if (cand[0] > prev[0]) or (cand[0] == prev[0] and cand[1] < prev[1]):
                best[key] = cand
                chosen_line[key] = ln

    out: List[str] = []
    for ln in lines:
        if not _is_atom_record(ln):
            continue
        key = (
            _record_name(ln),
            _chain_id(ln),
            _resseq(ln),
            _icode(ln),
            _resname(ln),
            _atom_name(ln),
        )
        if chosen_line.get(key) == ln:
            out.append(ln)
    return _ensure_trailing_newline(out)


def _write_pdb(path: Path, atom_lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _ensure_trailing_newline(list(atom_lines))
    if not lines:
        path.write_text("END\n")
        return
    if not lines[-1].strip().startswith("END"):
        lines.append("END\n")
    path.write_text("".join(lines))


def _group_ligand_residues(atom_lines: Sequence[str], ligand_resname: str) -> Dict[Tuple[str, int, str], List[str]]:
    groups: Dict[Tuple[str, int, str], List[str]] = {}
    for ln in atom_lines:
        if not ln.startswith("HETATM"):
            continue
        if _resname(ln) != ligand_resname:
            continue
        rid = _residue_id(ln).key()
        groups.setdefault(rid, []).append(ln)
    return groups


# -----------------------------
# OpenBabel runner
# -----------------------------
@dataclass
class CommandResult:
    name: str
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str
    outputs: Dict[str, str]


def _run_cmd(name: str, cmd: List[str]) -> CommandResult:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return CommandResult(
        name=name,
        cmd=cmd,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        outputs={},
    )


# -----------------------------
# Main component
# -----------------------------
class PDBQTPreparer:
    """
    Component:
      - Split an embedded complex PDB into receptor and ligand
      - Receptor is restricted to specified chain(s) (default: chain A only)
      - Convert both to PDBQT via OpenBabel (obabel)
      - Save outputs under docking_result/<step>/<group_key>/...

    This class does NOT run docking (vina). It only prepares inputs for docking.
    """

    def __init__(
        self,
        result_root: Path = Path("docking_result"),
        step_dirname: str = "20_prepare_pdbqt",
        obabel_bin: str = "obabel",
        archive_inputs: bool = True,
        receptor_include_hetatm: bool = False,
        receptor_het_whitelist: Optional[Iterable[str]] = None,
        receptor_chain_whitelist: Optional[Iterable[str]] = ("A",),
    ) -> None:
        self.result_root = Path(result_root)
        self.step_dir = self.result_root / step_dirname
        self.obabel_bin = str(obabel_bin)
        self.archive_inputs = bool(archive_inputs)
        self.receptor_include_hetatm = bool(receptor_include_hetatm)
        self.receptor_het_whitelist = set(receptor_het_whitelist or [])
        self.receptor_chain_whitelist = set([c.strip() for c in (receptor_chain_whitelist or []) if c.strip()])

        if not self.receptor_chain_whitelist:
            raise ValueError("receptor_chain_whitelist must not be empty.")

    def prepare(
        self,
        *,
        target_group_key: str,
        embedded_pdb: Path,
        ligand_resname: str,
        ligand_residue_id: Optional[ResidueID] = None,
        keep_others_pdb: bool = True,
        ph: Optional[float] = None,
    ) -> Dict[str, Path]:
        """
        Args:
          target_group_key: e.g. 4LPK_WT, 6OIM_G12C, 9C41_G12D
          embedded_pdb: path to embedded complex PDB (protein + ligand)
          ligand_resname: resname to extract as ligand (e.g., GDP, MOV)
          ligand_residue_id: if multiple ligands share same resname, specify which one (chain, resseq, icode)
          keep_others_pdb: whether to save a debugging PDB for excluded atoms (other chains, non-selected HETATM, etc.)
          ph: if provided, use OpenBabel -p <ph> (typically for ligand hydrogenation)

        Returns:
          A dict of key paths:
            receptor_pdb, ligand_pdb, receptor_pdbqt, ligand_pdbqt, (optional others_pdb), out_dir
        """
        embedded_pdb = Path(embedded_pdb)
        if not embedded_pdb.exists():
            raise FileNotFoundError(f"embedded_pdb not found: {embedded_pdb}")

        out_dir = self.step_dir / target_group_key
        input_dir = out_dir / "input"
        split_dir = out_dir / "split"
        pdbqt_dir = out_dir / "pdbqt"
        logs_dir = out_dir / "logs"
        for d in (input_dir, split_dir, pdbqt_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        if self.archive_inputs:
            shutil.copy2(embedded_pdb, input_dir / "embedded.pdb")

        raw_lines = embedded_pdb.read_text().splitlines(keepends=True)
        atom_lines = [ln for ln in raw_lines if _is_atom_record(ln)]
        atom_lines = _filter_altloc_keep_best(atom_lines)

        # ---- Split receptor vs ligand vs others ----
        receptor_lines: List[str] = []
        others_lines: List[str] = []

        for ln in atom_lines:
            rec = _record_name(ln)
            rn = _resname(ln)
            ch = _chain_id(ln)

            # Chain filter for receptor selection
            in_receptor_chain = ch in self.receptor_chain_whitelist

            if rec == "ATOM":
                if in_receptor_chain:
                    receptor_lines.append(ln)
                else:
                    others_lines.append(ln)
                continue

            # HETATM
            if rn in _DEFAULT_SKIP_HETERO:
                others_lines.append(ln)
                continue

            if in_receptor_chain and self.receptor_include_hetatm and (rn in self.receptor_het_whitelist):
                receptor_lines.append(ln)
            else:
                others_lines.append(ln)

        # Ligand selection (search all atom_lines, independent of receptor chain filter)
        ligand_groups = _group_ligand_residues(atom_lines, ligand_resname=ligand_resname)
        if not ligand_groups:
            raise ValueError(
                f"No ligand HETATM with resname={ligand_resname} found in {embedded_pdb}"
            )

        if ligand_residue_id is not None:
            key = ligand_residue_id.key()
            if key not in ligand_groups:
                available = sorted(list(ligand_groups.keys()))
                raise ValueError(
                    f"Specified ligand_residue_id={key} not found for resname={ligand_resname}. "
                    f"Available residue ids: {available}"
                )
            ligand_key = key
        else:
            ligand_key = max(ligand_groups.keys(), key=lambda k: len(ligand_groups[k]))

        ligand_lines = ligand_groups[ligand_key]

        # Remove selected ligand atoms from others_lines (if present)
        ligand_set = set(ligand_lines)
        others_lines = [ln for ln in others_lines if ln not in ligand_set]

        # Save split artifacts
        receptor_pdb = split_dir / "receptor.pdb"
        ligand_pdb = split_dir / "ligand.pdb"
        _write_pdb(receptor_pdb, receptor_lines)
        _write_pdb(ligand_pdb, ligand_lines)

        others_pdb: Optional[Path] = None
        if keep_others_pdb:
            others_pdb = split_dir / "others.pdb"
            _write_pdb(others_pdb, others_lines)

        split_summary = {
            "embedded_pdb": str(embedded_pdb),
            "ligand_resname": ligand_resname,
            "ligand_selected_residue": {"chain": ligand_key[0], "resseq": ligand_key[1], "icode": ligand_key[2]},
            "counts": {
                "receptor_atom_lines": len(receptor_lines),
                "ligand_atom_lines": len(ligand_lines),
                "others_atom_lines": len(others_lines),
                "ligand_instances_found": len(ligand_groups),
            },
            "receptor_chain_whitelist": sorted(list(self.receptor_chain_whitelist)),
            "receptor_include_hetatm": self.receptor_include_hetatm,
            "receptor_het_whitelist": sorted(list(self.receptor_het_whitelist)),
        }
        (split_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2))

        # ---- Convert to PDBQT via OpenBabel ----
        receptor_pdbqt = pdbqt_dir / "receptor.pdbqt"
        ligand_pdbqt = pdbqt_dir / "ligand.pdbqt"

        # Receptor: rigid + charges; add -xr -xc to avoid ROOT / torsion tree
        receptor_cmd = [
            self.obabel_bin,
            "-ipdb", str(receptor_pdb),
            "-opdbqt",
            "-O", str(receptor_pdbqt),
            "-h",
            "-xr", "-xc",
        ]

        # Ligand: keep flexible (no -xr), add hydrogens + gasteiger charges
        ligand_cmd = [
            self.obabel_bin,
            "-ipdb", str(ligand_pdb),
            "-opdbqt",
            "-O", str(ligand_pdbqt),
            "-h",
            "--partialcharge", "gasteiger",
        ]
        if ph is not None:
            ligand_cmd.extend(["-p", str(ph)])

        r_res = _run_cmd("obabel_receptor", receptor_cmd)
        l_res = _run_cmd("obabel_ligand", ligand_cmd)

        (logs_dir / "obabel_receptor.stdout.txt").write_text(r_res.stdout)
        (logs_dir / "obabel_receptor.stderr.txt").write_text(r_res.stderr)
        (logs_dir / "obabel_ligand.stdout.txt").write_text(l_res.stdout)
        (logs_dir / "obabel_ligand.stderr.txt").write_text(l_res.stderr)

        commands_json = {
            "commands": [
                {"name": r_res.name, "cmd": r_res.cmd, "returncode": r_res.returncode},
                {"name": l_res.name, "cmd": l_res.cmd, "returncode": l_res.returncode},
            ]
        }
        (logs_dir / "commands.json").write_text(json.dumps(commands_json, indent=2))

        if r_res.returncode != 0 or not receptor_pdbqt.exists() or receptor_pdbqt.stat().st_size == 0:
            raise RuntimeError(
                f"OpenBabel receptor conversion failed (returncode={r_res.returncode}). "
                f"See logs in {logs_dir}"
            )
        if l_res.returncode != 0 or not ligand_pdbqt.exists() or ligand_pdbqt.stat().st_size == 0:
            raise RuntimeError(
                f"OpenBabel ligand conversion failed (returncode={l_res.returncode}). "
                f"See logs in {logs_dir}"
            )

        out: Dict[str, Path] = {
            "out_dir": out_dir,
            "receptor_pdb": receptor_pdb,
            "ligand_pdb": ligand_pdb,
            "receptor_pdbqt": receptor_pdbqt,
            "ligand_pdbqt": ligand_pdbqt,
        }
        if others_pdb is not None:
            out["others_pdb"] = others_pdb
        return out
