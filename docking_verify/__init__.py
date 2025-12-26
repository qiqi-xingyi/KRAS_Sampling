# --*-- coding:utf-8 --*--
# @time:12/24/25
# @Author : Yuqi Zhang
# @File:__init__.py

from .schema import (
    Case,
    Box,
    VinaParams,
    load_cases_csv,
    load_config,
)

from .prep import (
    build_hybrid_receptor_by_ca_alignment,
    make_box_from_ligand_centroid,
)

from .vina import (
    run_vina_once,
    run_vina_multi_seed,
    parse_vina_scores_from_out_pdbqt,
)

from .analyze import (
    analyze_vina_outputs,
    aggregate_case_metrics,
    write_summary_csv,
    write_aggregate_json,
    compute_pocket_hit,
)

from .pdbqt import (
    OpenBabelError,
    prepare_receptor_pdbqt_obabel,
    prepare_ligand_pdbqt_obabel,
)

from .pipeline import run_pipeline

from .meeko_pdbqt import prepare_receptor_pdbqt_meeko, MeekoError


__all__ = [
    # schema
    "Case",
    "Box",
    "VinaParams",
    "load_cases_csv",
    "load_config",
    # prep
    "build_hybrid_receptor_by_ca_alignment",
    "make_box_from_ligand_centroid",
    # vina
    "run_vina_once",
    "run_vina_multi_seed",
    "parse_vina_scores_from_out_pdbqt",
    # analyze
    "analyze_vina_outputs",
    "aggregate_case_metrics",
    "write_summary_csv",
    "write_aggregate_json",
    "compute_pocket_hit",
    # pdbqt (OpenBabel)
    "OpenBabelError",
    "prepare_receptor_pdbqt_obabel",
    "prepare_ligand_pdbqt_obabel",
    # pipeline
    "run_pipeline",
    # meeko
    "prepare_receptor_pdbqt_meeko",
    "MeekoError",
]
