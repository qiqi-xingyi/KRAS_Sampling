# --*-- conding:utf-8 --*--
# @time:12/25/25 00:14
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cli.py


from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .schema import load_cases_csv, VinaParams
from .prep import build_hybrid_receptor_by_ca_alignment, make_box_from_ligand_centroid
from .vina import run_vina_multi_seed
from .analyze import analyze_vina_outputs, aggregate_case_metrics, write_summary_csv, write_aggregate_json


def main() -> None:
    ap = argparse.ArgumentParser("docking-verify")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # build hybrid receptors
    p_h = sub.add_parser("hybrid", help="Build hybrid receptors by CA alignment.")
    p_h.add_argument("--cases", required=True, type=str)
    p_h.add_argument("--out", required=True, type=str)

    # run vina
    p_r = sub.add_parser("run", help="Run vina for each case.")
    p_r.add_argument("--cases", required=True, type=str)
    p_r.add_argument("--vina", required=True, type=str, help="vina executable path/name")
    p_r.add_argument("--ligand", required=True, type=str, help="ligand pdbqt (single for now)")
    p_r.add_argument("--out", required=True, type=str)
    p_r.add_argument("--exhaustiveness", type=int, default=16)
    p_r.add_argument("--num_modes", type=int, default=20)
    p_r.add_argument("--energy_range", type=int, default=3)
    p_r.add_argument("--cpu", type=int, default=8)
    p_r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p_r.add_argument("--box_size", type=float, nargs=3, default=[20.0, 20.0, 20.0])

    # analyze
    p_a = sub.add_parser("analyze", help="Analyze vina outputs and write reports.")
    p_a.add_argument("--cases", required=True, type=str)
    p_a.add_argument("--out", required=True, type=str)
    p_a.add_argument("--margin", type=float, default=0.0)

    args = ap.parse_args()
    cases = load_cases_csv(Path(args.cases))

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cmd == "hybrid":
        hybrid_dir = out_root / "receptors" / "hybrid"
        hybrid_dir.mkdir(parents=True, exist_ok=True)

        for c in cases:
            if c.pred_ca_pdb is None:
                continue
            out_pdb = hybrid_dir / f"{c.case_id}.pdb"
            build_hybrid_receptor_by_ca_alignment(
                crystal_pdb=c.pdb_path,
                chain_id=c.chain_id,
                start_resi=c.start_resi,
                end_resi=c.end_resi,
                pred_ca_pdb=c.pred_ca_pdb,
                out_pdb=out_pdb,
            )
        print(f"[OK] Hybrid receptors written to {hybrid_dir}")
        return

    if args.cmd == "run":
        vina_exe = args.vina
        ligand_pdbqt = Path(args.ligand).expanduser().resolve()
        params = VinaParams(
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            energy_range=args.energy_range,
            cpu=args.cpu,
        )
        seeds: List[int] = list(args.seeds)
        sx, sy, sz = args.box_size

        for c in cases:
            # box from crystal ligand centroid
            box = make_box_from_ligand_centroid(
                complex_pdb=c.pdb_path,
                ligand_resname=c.ligand_resname,
                size=(sx, sy, sz),
            )

            # For now, assume receptor pdbqt is prepared externally.
            # Here we just point to receptor pdbqt paths you will generate later in prep step.
            # You can replace this with your own receptor pdbqt generation.
            receptor_pdbqt = out_root / "pdbqt" / "receptors" / f"{c.case_id}.pdbqt"
            if not receptor_pdbqt.exists():
                # fallback: skip
                print(f"[WARN] receptor pdbqt missing: {receptor_pdbqt} (skip {c.case_id})")
                continue

            out_case = out_root / "vina_out" / c.case_id
            run_vina_multi_seed(
                vina_exe=vina_exe,
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=ligand_pdbqt,
                box=box,
                params=params,
                out_root=out_case,
                seeds=seeds,
            )
        print("[OK] Vina runs completed.")
        return

    if args.cmd == "analyze":
        all_metrics = []
        for c in cases:
            # reconstruct out dir
            out_case = out_root / "vina_out" / c.case_id
            if not out_case.exists():
                continue
            box = make_box_from_ligand_centroid(
                complex_pdb=c.pdb_path,
                ligand_resname=c.ligand_resname,
                size=(20.0, 20.0, 20.0),
            )
            metrics = analyze_vina_outputs(
                out_root=out_case,
                case_id=c.case_id,
                receptor_type="unknown",
                box=box,
                margin=args.margin,
            )
            all_metrics.extend(metrics)

        summary_csv = write_summary_csv(all_metrics, out_root / "reports" / "summary.csv")
        aggs = aggregate_case_metrics(all_metrics)
        agg_json = write_aggregate_json(aggs, out_root / "reports" / "aggregate.json")

        print(f"[OK] Wrote {summary_csv}")
        print(f"[OK] Wrote {agg_json}")
        return


if __name__ == "__main__":
    main()
