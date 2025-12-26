# --*-- conding:utf-8 --*--
# @time:12/25/25 01:19
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pipeline.py

"""
docking_verify.pipeline

A reusable, external-call-friendly orchestration layer that runs the full docking
verification pipeline end-to-end.

Pipeline steps (per case)
-------------------------
1) Load cases from cases.csv
2) Build receptor PDB:
   - receptor_mode="crystal": use crystal PDB directly
   - receptor_mode="hybrid" : build hybrid receptor by CA alignment
3) Strip receptor PDB to receptor-only PDB (remove waters/ligands)
4) Convert receptor-only PDB -> receptor PDBQT via OpenBabel
5) Prepare ligand PDBQT:
   - ligand_mode="from_crystal": extract cognate ligand from crystal PDB and convert to PDBQT
   - ligand_mode="external": convert user-provided ligand file to PDBQT
6) Build Vina box from crystal ligand centroid
7) Run Vina for multiple seeds
8) Analyze outputs and write reports

Outputs
-------
out_dir/
  receptors/
    crystal/{case_id}.pdb
    hybrid/{case_id}.pdb
    receptor_only/{case_id}.receptor.pdb
  ligands/
    from_crystal/{ref_stem}_{ligand}_{chain}.pdb
  pdbqt/
    receptors/{case_id}.pdbqt
    ligands/{ligand_key}.pdbqt
  vina_out/{case_id}/seed_*/(out.pdbqt, log.txt, meta.json)
  status/{case_id}.json
  reports/
    summary.csv
    aggregate.json
    runs.csv           (per-seed run metrics)
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import time
import traceback

from .schema import Box, VinaParams, load_cases_csv
from .prep import build_hybrid_receptor_by_ca_alignment, make_box_from_ligand_centroid
from .pdbqt import (
    OpenBabelError,
    strip_to_receptor_pdb,
    prepare_receptor_pdbqt_obabel,
    prepare_ligand_pdbqt_obabel,
    prepare_cognate_ligand_pdbqt_from_crystal,
)
from .vina import run_vina_multi_seed
from .analyze import (
    analyze_vina_outputs,
    aggregate_case_metrics,
    write_summary_csv,
    write_aggregate_json,
)


class PipelineError(RuntimeError):
    pass


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _ligand_cache_key(ref_pdb: str, ligand_resname: str, chain_id: Optional[str]) -> str:
    stem = Path(ref_pdb).stem
    chain = chain_id.strip() if chain_id else "any"
    return f"{stem}_{ligand_resname}_{chain}"


def _file_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def run_pipeline(
    cases_csv: Union[str, Path],
    out_dir: Union[str, Path],
    vina_exe: str = "vina",
    obabel_exe: str = "obabel",
    receptor_mode: str = "hybrid",          # "hybrid" or "crystal"
    ligand_mode: str = "from_crystal",      # "from_crystal" or "external"
    external_ligand: Optional[Union[str, Path]] = None,  # used if ligand_mode="external"
    seeds: Optional[Sequence[int]] = None,
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    vina_params: Optional[VinaParams] = None,
    keep_het_resnames_in_receptor: Optional[Sequence[str]] = None,  # e.g., ("MG",)
    remove_water: bool = True,
    resume: bool = True,
    margin_for_pocket_hit: float = 0.0,
    strict: bool = False,
) -> Path:
    """
    Run the full pipeline and write reports under out_dir/reports.

    Args:
      cases_csv: docking_data/cases.csv
      out_dir: root output directory
      vina_exe: path/name of AutoDock Vina executable
      obabel_exe: path/name of OpenBabel obabel executable
      receptor_mode:
        - "crystal": docking on stripped crystal receptor
        - "hybrid" : docking on hybrid receptor built by CA alignment
      ligand_mode:
        - "from_crystal": extract cognate ligand from crystal PDB by resname and convert to PDBQT
        - "external": user provides external_ligand (sdf/mol2/pdb/...) converted to PDBQT once
      external_ligand: required if ligand_mode="external"
      seeds: list of seeds for repeated docking; default [0,1,2,3,4]
      box_size: Vina box sizes
      vina_params: VinaParams object; default is reasonable
      keep_het_resnames_in_receptor: optional list of HET residue names to keep in receptor stripping
      remove_water: remove water molecules during receptor stripping
      resume: if outputs exist, skip recomputation
      margin_for_pocket_hit: analysis margin (Å) for box hit
      strict: if True, raise on first failure; if False, skip failing cases and continue

    Returns:
      Path to reports directory.
    """
    cases_csv = Path(cases_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    _ensure_dir(out_dir)

    if receptor_mode not in ("hybrid", "crystal"):
        raise ValueError("receptor_mode must be 'hybrid' or 'crystal'")
    if ligand_mode not in ("from_crystal", "external"):
        raise ValueError("ligand_mode must be 'from_crystal' or 'external'")
    if ligand_mode == "external" and external_ligand is None:
        raise ValueError("external_ligand is required when ligand_mode='external'")

    seeds = list(seeds) if seeds is not None else [0, 1, 2, 3, 4]
    vina_params = vina_params or VinaParams(exhaustiveness=16, num_modes=20, energy_range=3, cpu=8)

    keep_het_set = set(keep_het_resnames_in_receptor or [])

    # Output structure
    receptors_crystal_dir = _ensure_dir(out_dir / "receptors" / "crystal")
    receptors_hybrid_dir = _ensure_dir(out_dir / "receptors" / "hybrid")
    receptors_only_dir = _ensure_dir(out_dir / "receptors" / "receptor_only")
    ligands_from_crystal_dir = _ensure_dir(out_dir / "ligands" / "from_crystal")
    pdbqt_receptors_dir = _ensure_dir(out_dir / "pdbqt" / "receptors")
    pdbqt_ligands_dir = _ensure_dir(out_dir / "pdbqt" / "ligands")
    vina_out_dir = _ensure_dir(out_dir / "vina_out")
    status_dir = _ensure_dir(out_dir / "status")
    reports_dir = _ensure_dir(out_dir / "reports")

    cases = load_cases_csv(cases_csv)

    # Prepare external ligand pdbqt once if needed
    external_ligand_pdbqt: Optional[Path] = None
    if ligand_mode == "external":
        ext = Path(external_ligand).expanduser().resolve()  # type: ignore[arg-type]
        ligand_key = f"external_{ext.stem}"
        external_ligand_pdbqt = pdbqt_ligands_dir / f"{ligand_key}.pdbqt"
        if resume and _file_nonempty(external_ligand_pdbqt):
            pass
        else:
            prepare_ligand_pdbqt_obabel(
                ligand_in=ext,
                out_pdbqt=external_ligand_pdbqt,
                obabel_exe=obabel_exe,
                gen3d=True,
                add_h=True,
                partialcharge="gasteiger",
            )

    # Cache for crystal-derived ligands: key -> pdbqt path
    ligand_cache: Dict[str, Path] = {}

    all_run_metrics = []  # collected for reporting

    for c in cases:
        case_t0 = time.time()
        status_path = status_dir / f"{c.case_id}.json"
        case_status: Dict = {
            "case_id": c.case_id,
            "ref_pdb": c.ref_pdb,
            "pdb_path": str(c.pdb_path),
            "chain_id": c.chain_id,
            "start_resi": c.start_resi,
            "end_resi": c.end_resi,
            "ligand_resname": c.ligand_resname,
            "receptor_mode": receptor_mode,
            "ligand_mode": ligand_mode,
            "seeds": list(seeds),
            "vina_params": asdict(vina_params),
            "steps": {},
            "ok": False,
            "error": None,
        }

        try:
            # ---- Step 1: Determine receptor PDB (crystal or hybrid)
            if receptor_mode == "crystal":
                receptor_pdb = receptors_crystal_dir / f"{c.case_id}.pdb"
                if (not resume) or (not _file_nonempty(receptor_pdb)):
                    receptor_pdb.write_text(Path(c.pdb_path).read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                case_status["steps"]["receptor_pdb"] = {"path": str(receptor_pdb), "mode": "crystal"}

            else:
                # hybrid
                if c.pred_ca_pdb is None or (not c.pred_ca_pdb.exists()):
                    raise PipelineError("Missing pred_ca_pdb for hybrid receptor build.")
                receptor_pdb = receptors_hybrid_dir / f"{c.case_id}.pdb"
                if (not resume) or (not _file_nonempty(receptor_pdb)):
                    build_hybrid_receptor_by_ca_alignment(
                        crystal_pdb=c.pdb_path,
                        chain_id=c.chain_id,
                        start_resi=c.start_resi,
                        end_resi=c.end_resi,
                        pred_ca_pdb=c.pred_ca_pdb,
                        out_pdb=receptor_pdb,
                    )
                case_status["steps"]["receptor_pdb"] = {"path": str(receptor_pdb), "mode": "hybrid"}

            # ---- Step 2: Strip receptor-only PDB
            receptor_only_pdb = receptors_only_dir / f"{c.case_id}.receptor.pdb"
            if (not resume) or (not _file_nonempty(receptor_only_pdb)):
                strip_to_receptor_pdb(
                    in_pdb=receptor_pdb,
                    out_pdb=receptor_only_pdb,
                    keep_hetatm=False,
                    keep_het_resnames=set(keep_het_set),
                    remove_water=remove_water,
                )
            case_status["steps"]["receptor_only_pdb"] = {"path": str(receptor_only_pdb)}

            # ---- Step 3: Receptor PDBQT
            receptor_pdbqt = pdbqt_receptors_dir / f"{c.case_id}.pdbqt"
            if (not resume) or (not _file_nonempty(receptor_pdbqt)):
                prepare_receptor_pdbqt_obabel(
                    receptor_pdb=receptor_only_pdb,
                    out_pdbqt=receptor_pdbqt,
                    obabel_exe=obabel_exe,
                    add_h=True,
                    partialcharge="gasteiger",
                )
            case_status["steps"]["receptor_pdbqt"] = {"path": str(receptor_pdbqt)}

            # ---- Step 4: Ligand PDBQT
            if ligand_mode == "external":
                assert external_ligand_pdbqt is not None
                ligand_pdbqt = external_ligand_pdbqt
                case_status["steps"]["ligand_pdbqt"] = {
                    "path": str(ligand_pdbqt),
                    "mode": "external",
                    "key": "external",
                }

            else:
                # from_crystal
                lig_resname = (c.ligand_resname or "").strip()
                if not lig_resname:
                    # Optional fallback (if you added detect_primary_ligand_resname)
                    # from .pdbqt import detect_primary_ligand_resname
                    # lig_resname = detect_primary_ligand_resname(c.pdb_path)
                    raise PipelineError("ligand_resname is empty for ligand_mode='from_crystal'.")

                key = _ligand_cache_key(c.ref_pdb, lig_resname, None)

                # Reuse cached ligand pdbqt if present
                cached = ligand_cache.get(key)
                if cached is not None and _file_nonempty(cached):
                    ligand_pdbqt = cached
                else:
                    lig_pdb = ligands_from_crystal_dir / f"{key}.pdb"
                    lig_pdbqt = pdbqt_ligands_dir / f"{key}.pdbqt"

                    if (not resume) or (not _file_nonempty(lig_pdbqt)):
                        prepare_cognate_ligand_pdbqt_from_crystal(
                            crystal_pdb=c.pdb_path,
                            ligand_resname=lig_resname,
                            out_ligand_pdb=lig_pdb,
                            out_ligand_pdbqt=lig_pdbqt,
                            chain_id=None,
                            obabel_exe=obabel_exe,
                            add_h=True,
                            partialcharge="gasteiger",
                        )

                    ligand_cache[key] = lig_pdbqt
                    ligand_pdbqt = lig_pdbqt

                case_status["steps"]["ligand_pdbqt"] = {
                    "path": str(ligand_pdbqt),
                    "mode": "from_crystal",
                    "key": key,
                    "ligand_resname": lig_resname,
                }

            # ---- Step 5: Box from crystal ligand centroid (always use crystal reference)
            box = make_box_from_ligand_centroid(
                complex_pdb=c.pdb_path,
                ligand_resname=c.ligand_resname,
                size=box_size,
            )
            case_status["steps"]["box"] = {
                "center": [box.center_x, box.center_y, box.center_z],
                "size": [box.size_x, box.size_y, box.size_z],
            }

            # ---- Step 6: Run Vina multi-seed
            case_vina_out = vina_out_dir / c.case_id
            if (not resume) or (not (case_vina_out.exists() and any((case_vina_out / f"seed_{s}").exists() for s in seeds))):
                # Ensure directory exists; run will create seed_* dirs
                _ensure_dir(case_vina_out)
                run_vina_multi_seed(
                    vina_exe=vina_exe,
                    receptor_pdbqt=receptor_pdbqt,
                    ligand_pdbqt=ligand_pdbqt,
                    box=box,
                    params=vina_params,
                    out_root=case_vina_out,
                    seeds=list(seeds),
                )
            case_status["steps"]["vina_out"] = {"path": str(case_vina_out)}

            # HARD CHECK: require real files
            missing = []
            for s in seeds:
                d = case_vina_out / f"seed_{int(s)}"
                outp = d / "out.pdbqt"
                logp = d / "log.txt"
                if (not outp.exists()) or outp.stat().st_size == 0:
                    missing.append(str(outp))
                if (not logp.exists()) or logp.stat().st_size == 0:
                    missing.append(str(logp))
            if missing:
                raise PipelineError("Vina produced no outputs: " + "; ".join(missing))

            case_status["steps"]["vina_out"] = {"path": str(case_vina_out)}

            # ---- Step 7: Analyze
            run_metrics = analyze_vina_outputs(
                out_root=case_vina_out,
                case_id=c.case_id,
                receptor_type=receptor_mode,
                box=box,
                margin=margin_for_pocket_hit,
            )
            # attach to global list
            all_run_metrics.extend(run_metrics)

            case_status["steps"]["analysis"] = {"n_runs": len(run_metrics)}
            case_status["ok"] = True

        except Exception as e:
            case_status["ok"] = False
            case_status["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(limit=10),
            }
            if strict:
                _write_json(status_path, case_status)
                raise
        finally:
            case_status["runtime_sec"] = time.time() - case_t0
            _write_json(status_path, case_status)

    # ---- Final reports
    runs_csv = reports_dir / "runs.csv"
    summary_csv = reports_dir / "summary.csv"
    agg_json = reports_dir / "aggregate.json"

    # runs.csv: per-seed records
    # We rely on analyze.write_summary_csv style, but that file writes only run-level summary.
    # Here we call write_summary_csv for run-level; name it runs.csv for clarity.
    write_summary_csv(all_run_metrics, runs_csv)

    # aggregate.json: per-case aggregates
    aggs = aggregate_case_metrics(all_run_metrics)
    write_aggregate_json(aggs, agg_json)

    # summary.csv: human-friendly table derived from aggregates (simple CSV)
    _write_summary_table_from_aggs(aggs, summary_csv)

    return reports_dir.resolve()


def _write_summary_table_from_aggs(aggs, out_csv: Path) -> None:
    """
    Write a simple CSV summary from CaseAggregate list.
    """
    import csv

    out_csv = Path(out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "receptor_type",
                "n_runs",
                "best_score_mean",
                "best_score_std",
                "pocket_hit_rate",
            ],
        )
        w.writeheader()
        for a in sorted(aggs, key=lambda x: (x.case_id, x.receptor_type)):
            w.writerow(
                {
                    "case_id": a.case_id,
                    "receptor_type": a.receptor_type,
                    "n_runs": a.n_runs,
                    "best_score_mean": "" if a.best_score_mean is None else a.best_score_mean,
                    "best_score_std": "" if a.best_score_std is None else a.best_score_std,
                    "pocket_hit_rate": "" if a.pocket_hit_rate is None else a.pocket_hit_rate,
                }
            )
