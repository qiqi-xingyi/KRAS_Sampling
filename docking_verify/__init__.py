# --*-- conding:utf-8 --*--
# @time:12/24/25 20:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py


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
    strip_to_receptor_pdb,
    extract_ligand_pdb_from_crystal,
    prepare_receptor_pdbqt_obabel,
    prepare_ligand_pdbqt_obabel,
    prepare_cognate_ligand_pdbqt_from_crystal,
    batch_prepare_receptors_from_pdb,
)

from .pipeline import run_pipeline, VinaParams

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
    "strip_to_receptor_pdb",
    "extract_ligand_pdb_from_crystal",
    "prepare_receptor_pdbqt_obabel",
    "prepare_ligand_pdbqt_obabel",
    "prepare_cognate_ligand_pdbqt_from_crystal",
    "batch_prepare_receptors_from_pdb",
    "run_pipeline",
    "prepare_receptor_pdbqt_meeko",
    "MeekoError"
]
