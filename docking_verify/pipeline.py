# --*-- conding:utf-8 --*--
# @time:12/26/25 20:37
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .embedder import KabschFragmentEmbedder
from .pdbqt_preparer import PDBQTPreparer, ResidueID
from .box_builder import LigandCenteredBoxBuilder, DockingBox
from .vina_runner import VinaDockingRunner, VinaParams


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class BoxConfig:
    """
    Box policy: centered on ligand centroid.
    If fixed_size is provided, size is fixed (still centered on ligand).
    Otherwise size = ligand bounding box + margin, clamped by min_size/max_size.
    """
    margin: float = 10.0
    min_size: float = 20.0
    max_size: Optional[float] = None
    fixed_size: Optional[Tuple[float, float, float]] = None
    select_ligand_instance: str = "largest"  # "largest" or "first"


@dataclass(frozen=True)
class PipelineConfig:
    # IO
    cases_csv: Path
    result_root: Path = Path("docking_result")

    # Tool binaries
    obabel_bin: str = "obabel"
    vina_bin: str = "vina"

    # Embed
    embed_step_dirname: str = "10_embed"
    # Prepare PDBQT
    prepare_step_dirname: str = "20_prepare_pdbqt"
    # Box
    box_step_dirname: str = "25_box"
    # Dock
    dock_step_dirname: str = "30_dock"
    # Pipeline summary
    pipeline_step_dirname: str = "40_pipeline"

    # Mutation fallback logic:
    # If target group is not WT, missing fragments (2,3) fall back to this group.
    wt_group_key: str = "4LPK_WT"

    # Vina
    vina_params: VinaParams = VinaParams()
    n_repeats: int = 5
    seed_list: Optional[Sequence[int]] = None
    base_seed: int = 0

    # Box
    box: BoxConfig = BoxConfig()

    # Behavior
    overwrite: bool = False
    strict: bool = True

    # Optional: specify which ligand instance when multiple same resname exist
    ligand_residue_id: Optional[ResidueID] = None
    # Optional pH adjustment for ligand hydrogenation in OpenBabel
    ligand_ph: Optional[float] = None

    # Logging
    verbose: bool = True


# -----------------------------
# Pipeline
# -----------------------------
class DockingPipeline:
    """
    End-to-end dispatcher for the docking_verify package.

    It orchestrates, per target_group_key:
      1) Embed fragments into scaffold -> embedded.pdb
      2) Split receptor/ligand and convert to PDBQT
      3) Build ligand-centered docking box
      4) Run Vina repeatedly (seeds) and parse results
      5) Write a pipeline-level summary table for plotting / reporting

    This class provides a simple external interface:
      - run_group(target_group_key)
      - run_all(target_groups=None)
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

        # Components
        self.embedder = KabschFragmentEmbedder(
            cases_csv=cfg.cases_csv,
            result_root=cfg.result_root,
            step_dirname=cfg.embed_step_dirname,
            archive_inputs=True,
        )
        self.preparer = PDBQTPreparer(
            result_root=cfg.result_root,
            step_dirname=cfg.prepare_step_dirname,
            obabel_bin=cfg.obabel_bin,
            archive_inputs=True,
            # Enforce receptor chain A only for docking comparability
            receptor_chain_whitelist=("A",),
        )
        self.box_builder = LigandCenteredBoxBuilder(
            result_root=cfg.result_root,
            step_dirname=cfg.box_step_dirname,
            default_margin=cfg.box.margin,
            default_min_size=cfg.box.min_size,
            default_max_size=cfg.box.max_size,
            archive=True,
        )
        self.runner = VinaDockingRunner(
            result_root=cfg.result_root,
            step_dirname=cfg.dock_step_dirname,
            vina_bin=cfg.vina_bin,
            archive_inputs=True,
        )

        # Index cases for metadata (ligand_resname per target group, etc.)
        self._cases = self._load_cases_csv(cfg.cases_csv)
        self._group_to_rows = self._group_cases(self._cases)

    # -----------------------------
    # Logging helpers
    # -----------------------------
    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)

    def _log_kv(self, title: str, kv: Dict[str, object], indent: int = 2) -> None:
        if not self.cfg.verbose:
            return
        pad = " " * indent
        self._log(title)
        for k, v in kv.items():
            self._log(f"{pad}- {k}: {v}")

    # -----------------------------
    # Public interface
    # -----------------------------
    def run_all(self, target_groups: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Path]]:
        """
        Run pipeline for all groups in cases.csv, or a specified subset.

        Returns:
          mapping: target_group_key -> outputs dict (paths)
        """
        groups = list(target_groups) if target_groups is not None else sorted(self._group_to_rows.keys())
        results: Dict[str, Dict[str, Path]] = {}
        pipeline_failures: List[Dict[str, object]] = []

        self._log("=== DockingPipeline: run_all ===")
        self._log_kv(
            "Config",
            {
                "cases_csv": str(self.cfg.cases_csv),
                "result_root": str(self.cfg.result_root),
                "obabel_bin": self.cfg.obabel_bin,
                "vina_bin": self.cfg.vina_bin,
                "wt_group_key": self.cfg.wt_group_key,
                "n_repeats": self.cfg.n_repeats,
                "seed_list": list(self.cfg.seed_list) if self.cfg.seed_list is not None else None,
                "base_seed": self.cfg.base_seed,
                "strict": self.cfg.strict,
                "overwrite": self.cfg.overwrite,
                "box.margin": self.cfg.box.margin,
                "box.min_size": self.cfg.box.min_size,
                "box.max_size": self.cfg.box.max_size,
                "box.fixed_size": self.cfg.box.fixed_size,
                "box.select_ligand_instance": self.cfg.box.select_ligand_instance,
                "receptor_chain_policy": "A only",
            },
        )
        self._log(f"Target groups ({len(groups)}): {groups}")

        for idx, g in enumerate(groups, start=1):
            self._log(f"\n--- [{idx}/{len(groups)}] Running group: {g} ---")
            try:
                results[g] = self.run_group(g)
                self._log(f"--- [{idx}/{len(groups)}] Completed group: {g} ---")
            except Exception as e:
                self._log(f"--- [{idx}/{len(groups)}] FAILED group: {g} | error: {e} ---")
                if self.cfg.strict:
                    raise
                pipeline_failures.append({"target_group_key": g, "error": str(e)})

        self._write_pipeline_summary(results, pipeline_failures)
        self._log("\n=== DockingPipeline: finished ===")
        self._log(f"Completed: {len(results)} | Failed: {len(pipeline_failures)}")
        return results

    def run_group(self, target_group_key: str) -> Dict[str, Path]:
        """
        Run pipeline for a single target group.

        Returns:
          dict of key output paths (embedded_pdb, receptor_pdbqt, ligand_pdbqt, box_json, vina_summary_json, ...)
        """
        if target_group_key not in self._group_to_rows:
            raise KeyError(f"target_group_key not found in cases.csv: {target_group_key}")

        ligand_resname = self._infer_group_ligand_resname(target_group_key)
        fallback = None if target_group_key == self.cfg.wt_group_key else self.cfg.wt_group_key

        self._log_kv(
            "Group settings",
            {
                "target_group_key": target_group_key,
                "ligand_resname": ligand_resname,
                "fallback_group_key": fallback,
                "n_repeats": self.cfg.n_repeats,
                "seed_list": list(self.cfg.seed_list) if self.cfg.seed_list is not None else None,
                "base_seed": self.cfg.base_seed,
                "strict": self.cfg.strict,
                "overwrite": self.cfg.overwrite,
                "receptor_chain_policy": "A only",
            },
        )

        # 1) Embed
        self._log("Step 10_embed: embedding fragments into scaffold ...")
        embedded_pdb = self.embedder.embed_group(
            target_group_key=target_group_key,
            fallback_group_key=fallback,
            strict=True,
            allowed_scaffold_chains=("A",),
            verbose=True,
        )

        self._log(f"  embedded_pdb: {embedded_pdb}")

        # 2) Prepare PDBQT
        self._log("Step 20_prepare_pdbqt: splitting + converting to PDBQT ...")
        prep_out = self.preparer.prepare(
            target_group_key=target_group_key,
            embedded_pdb=embedded_pdb,
            ligand_resname=ligand_resname,
            ligand_residue_id=self.cfg.ligand_residue_id,
            keep_others_pdb=True,
            ph=self.cfg.ligand_ph,
        )
        receptor_pdbqt = prep_out["receptor_pdbqt"]
        ligand_pdbqt = prep_out["ligand_pdbqt"]
        self._log(f"  receptor_pdbqt: {receptor_pdbqt}")
        self._log(f"  ligand_pdbqt  : {ligand_pdbqt}")

        # Print split summary (very helpful for debugging chain policy)
        try:
            split_summary = (
                self.cfg.result_root
                / self.cfg.prepare_step_dirname
                / target_group_key
                / "split"
                / "split_summary.json"
            )
            if split_summary.exists():
                ss = json.loads(split_summary.read_text())
                counts = ss.get("counts", {})
                chain_whitelist = ss.get("receptor_chain_whitelist")
                self._log_kv(
                    "  prepare summary",
                    {
                        "receptor_chain_whitelist": chain_whitelist,
                        "receptor_atom_lines": counts.get("receptor_atom_lines"),
                        "ligand_atom_lines": counts.get("ligand_atom_lines"),
                        "others_atom_lines": counts.get("others_atom_lines"),
                        "ligand_instances_found": counts.get("ligand_instances_found"),
                    },
                    indent=4,
                )
        except Exception:
            pass

        # 3) Build box
        self._log("Step 25_box: building ligand-centered docking box ...")
        box = self.box_builder.build(
            target_group_key=target_group_key,
            complex_pdb=embedded_pdb,
            ligand_resname=ligand_resname,
            margin=self.cfg.box.margin,
            min_size=self.cfg.box.min_size,
            max_size=self.cfg.box.max_size,
            fixed_size=self.cfg.box.fixed_size,
            select_ligand_instance=self.cfg.box.select_ligand_instance,
        )
        box_json = (self.cfg.result_root / self.cfg.box_step_dirname / target_group_key / "box.json")

        # Print box params (center + size)
        try:
            self._log_kv(
                "  box",
                {
                    "center": getattr(box, "center", None),
                    "size": getattr(box, "size", None),
                    "space": getattr(box, "space", None),
                    "box_json": str(box_json),
                },
                indent=4,
            )
        except Exception:
            self._log(f"  box_json: {box_json}")

        # 4) Dock with Vina
        self._log("Step 30_dock: running Vina ...")
        dock_out = self.runner.run_group(
            target_group_key=target_group_key,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=ligand_pdbqt,
            box=box,
            vina_params=self.cfg.vina_params,
            n_repeats=self.cfg.n_repeats,
            seed_list=self.cfg.seed_list,
            base_seed=self.cfg.base_seed,
            overwrite=self.cfg.overwrite,
            strict=self.cfg.strict,
        )

        # Print docking summary
        summary_json = Path(dock_out["summary_json"])
        self._log(f"  vina_summary_json: {summary_json}")
        try:
            if summary_json.exists():
                s = json.loads(summary_json.read_text())
                bo = s.get("best_overall") or {}
                self._log_kv(
                    "  vina summary",
                    {
                        "best_affinity": bo.get("best_affinity"),
                        "best_seed": bo.get("seed"),
                        "best_mode": bo.get("mode"),
                        "n_runs_completed": s.get("n_runs_completed"),
                        "n_failures": s.get("n_failures"),
                    },
                    indent=4,
                )
        except Exception:
            pass

        # 5) Return outputs for external caller
        return {
            "embedded_pdb": embedded_pdb,
            "receptor_pdbqt": receptor_pdbqt,
            "ligand_pdbqt": ligand_pdbqt,
            "box_json": box_json,
            "vina_summary_json": dock_out["summary_json"],
            "vina_runs_long_csv": dock_out["runs_long_csv"],
            "vina_runs_summary_csv": dock_out["runs_summary_csv"],
            "vina_best_pose_pdbqt": dock_out["best_pose_pdbqt"],
            "vina_best_run_json": dock_out["best_run_json"],
            "out_dir_embed": self.cfg.result_root / self.cfg.embed_step_dirname / target_group_key,
            "out_dir_prepare": self.cfg.result_root / self.cfg.prepare_step_dirname / target_group_key,
            "out_dir_box": self.cfg.result_root / self.cfg.box_step_dirname / target_group_key,
            "out_dir_dock": self.cfg.result_root / self.cfg.dock_step_dirname / target_group_key,
        }

    # -----------------------------
    # Internal: CSV indexing
    # -----------------------------
    @staticmethod
    def _load_cases_csv(cases_csv: Path) -> List[Dict[str, str]]:
        cases_csv = Path(cases_csv)
        if not cases_csv.exists():
            raise FileNotFoundError(f"cases.csv not found: {cases_csv}")
        with cases_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader]

    @staticmethod
    def _group_cases(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        g: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            case_id = r["case_id"]
            group_key = case_id.rsplit("_", 1)[0] if "_" in case_id else case_id
            g.setdefault(group_key, []).append(r)
        return g

    def _infer_group_ligand_resname(self, target_group_key: str) -> str:
        rows = self._group_to_rows[target_group_key]
        for r in rows:
            lr = (r.get("ligand_resname") or "").strip()
            if lr:
                return lr
        raise ValueError(f"ligand_resname missing for group {target_group_key} in cases.csv")

    # -----------------------------
    # Internal: pipeline summary
    # -----------------------------
    def _write_pipeline_summary(
        self,
        results: Dict[str, Dict[str, Path]],
        failures: List[Dict[str, object]],
    ) -> None:
        out_dir = self.cfg.result_root / self.cfg.pipeline_step_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        table_rows: List[Dict[str, object]] = []

        for g, outs in results.items():
            ligand_resname = self._infer_group_ligand_resname(g)
            summary_json = outs.get("vina_summary_json")
            best_aff = None
            best_seed = None
            best_mode = None
            n_completed = None
            n_failures = None

            if summary_json is not None and Path(summary_json).exists():
                try:
                    s = json.loads(Path(summary_json).read_text())
                    bo = s.get("best_overall")
                    if bo:
                        best_aff = bo.get("best_affinity")
                        best_seed = bo.get("seed")
                        best_mode = bo.get("mode")
                    n_completed = s.get("n_runs_completed")
                    n_failures = s.get("n_failures")
                except Exception:
                    pass

            table_rows.append(
                {
                    "target_group_key": g,
                    "ligand_resname": ligand_resname,
                    "best_affinity": best_aff,
                    "best_seed": best_seed,
                    "best_mode": best_mode,
                    "n_runs_completed": n_completed,
                    "n_failures": n_failures,
                    "embedded_pdb": str(outs.get("embedded_pdb", "")),
                    "receptor_pdbqt": str(outs.get("receptor_pdbqt", "")),
                    "ligand_pdbqt": str(outs.get("ligand_pdbqt", "")),
                    "box_json": str(outs.get("box_json", "")),
                    "vina_summary_json": str(outs.get("vina_summary_json", "")),
                    "vina_runs_long_csv": str(outs.get("vina_runs_long_csv", "")),
                    "vina_runs_summary_csv": str(outs.get("vina_runs_summary_csv", "")),
                }
            )

        csv_path = out_dir / "pipeline_summary.csv"
        fieldnames = [
            "target_group_key",
            "ligand_resname",
            "best_affinity",
            "best_seed",
            "best_mode",
            "n_runs_completed",
            "n_failures",
            "embedded_pdb",
            "receptor_pdbqt",
            "ligand_pdbqt",
            "box_json",
            "vina_summary_json",
            "vina_runs_long_csv",
            "vina_runs_summary_csv",
        ]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in table_rows:
                w.writerow({k: r.get(k) for k in fieldnames})

        json_path = out_dir / "pipeline_report.json"
        report = {
            "config": self._config_to_jsonable(self.cfg),
            "n_groups_completed": len(results),
            "n_groups_failed": len(failures),
            "failures": failures,
            "summary_csv": str(csv_path),
        }
        json_path.write_text(json.dumps(report, indent=2))

        self._log(f"\nPipeline summary written:")
        self._log(f"  - {csv_path}")
        self._log(f"  - {json_path}")

    @staticmethod
    def _config_to_jsonable(cfg: PipelineConfig) -> Dict[str, object]:
        d = asdict(cfg)
        d["cases_csv"] = str(cfg.cases_csv)
        d["result_root"] = str(cfg.result_root)
        return d
