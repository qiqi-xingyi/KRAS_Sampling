# --*-- conding:utf-8 --*--
# @time:12/10/25 22:58
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:instance_sampling.py

import time
import random
from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict

import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from sampling import SamplingRunner, SamplingConfig, BackendConfig
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem


TASKS_FILE = "./tasks.csv"

# Penalty parameters for protein folding Hamiltonian
PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)

# Sampling hyperparameters
BETA_LIST: List[float] = [1.0, 2.0, 3.0, 4.0]
SEEDS: int = 3
REPS: int = 1

GROUP_COUNT = 10
SHOTS_PER_GROUP = 2000

# Use a KRAS-specific root directory for outputs
OUTPUT_ROOT = Path("KRAS_sampling_results")

# If you want to fix a backend, put its name here; otherwise set to None
IBM_BACKEND_NAME: str | None = "ibm_cleveland"  # or None to let SamplingRunner decide


def init_ibm_service() -> QiskitRuntimeService:
    """
    Initialize QiskitRuntimeService using ONLY the locally saved IBM account.

    Before running this script, make sure you have run once (in this env):

        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token="YOUR_TOKEN",
            instance="hub/group/project",
            overwrite=True,
        )

    After that, QiskitRuntimeService() will load credentials from local storage.
    """
    return QiskitRuntimeService()


def build_protein_hamiltonian(sequence: str, penalties: Tuple[int, int, int]) -> SparsePauliOp:
    """
    Build the protein folding Hamiltonian for a given main-chain sequence.
    """
    side_chain_residue_sequences = ['' for _ in range(len(sequence))]
    peptide = Peptide(sequence, side_chain_residue_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(*penalties)
    problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    H = problem.qubit_op()

    # Some versions return a tuple/list; extract the operator if so
    if isinstance(H, (list, tuple)) and len(H) > 0:
        H = H[0]
    if not isinstance(H, SparsePauliOp):
        H = SparsePauliOp(H)
    return H


def read_tasks(path: str) -> List[Dict[str, str]]:
    """
    Read KRAS tasks from CSV and generate a KRAS-specific label for each row.

    Expected columns (case-insensitive):
        - pdbid (preferred) or protein_name / pdb_id / protein
        - main_chain_residue_seq or sequence

    Example tasks.csv:

        pdbid,main_chain_residue_seq
        4LPK_WT,KLVVVGAGGVGK
        6OIM_G12C,KLVVVCAGGVGK
        9C41_G12D,KLVVVDAGGVGK
        4LPK_WT,ILDTAGQEEY
        4LPK_WT,EDIHHYREQIKR
    """
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]

    # Identify PDB/Protein name column
    if "protein_name" in cols:
        pn_col = df.columns[cols.index("protein_name")]
    elif "pdbid" in cols:
        pn_col = df.columns[cols.index("pdbid")]
    elif "pdb_id" in cols:
        pn_col = df.columns[cols.index("pdb_id")]
    elif "protein" in cols:
        pn_col = df.columns[cols.index("protein")]
    else:
        raise ValueError("Column protein_name / pdbid / pdb_id / protein is required")

    # Identify sequence column
    if "main_chain_residue_seq" in cols:
        seq_col = df.columns[cols.index("main_chain_residue_seq")]
    elif "sequence" in cols:
        seq_col = df.columns[cols.index("sequence")]
    else:
        raise ValueError("Column main_chain_residue_seq or sequence is required")

    tasks: List[Dict[str, str]] = []

    # Counter per PDB to distinguish multiple fragments for the same structure
    counters: Dict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        pdbid = str(row[pn_col]).strip()
        sequence = str(row[seq_col]).strip()
        if not pdbid or not sequence:
            continue

        counters[pdbid] += 1
        # KRAS-specific label, e.g. KRAS_4LPK_WT_1
        task_label = f"KRAS_{pdbid}_{counters[pdbid]}"

        tasks.append({
            "pdbid": pdbid,
            "main_chain_residue_seq": sequence,
            "task_label": task_label,
        })

    return tasks


def per_example_sampling(task_label: str, pdbid: str, sequence: str) -> str:
    """
    Run sampling for a single KRAS fragment.

    task_label is a KRAS-specific label (e.g., KRAS_4LPK_WT_1),
    pdbid is the original PDB identifier (e.g., 4LPK_WT),
    sequence is the main-chain residue sequence.
    """
    print(f"\n=== Running {task_label} | PDB: {pdbid} | Seq: {sequence} ===")

    # Each fragment gets its own directory under KRAS_sampling_results
    out_dir = OUTPUT_ROOT / task_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build Hamiltonian
    H = build_protein_hamiltonian(sequence, PENALTY_PARAMS)

    group_csvs: List[str] = []
    timing_rows: List[Dict[str, Any]] = []
    t0_total = time.perf_counter()

    for group_id in range(GROUP_COUNT):
        run_seed = random.randint(1, 2**31 - 1)

        cfg = SamplingConfig(
            L=len(sequence),
            betas=list(BETA_LIST),
            seeds=SEEDS,
            reps=REPS,
            entanglement="linear",
            label=f"qsad_ibm_{task_label}_g{group_id}",
            backend=BackendConfig(
                kind="ibm",
                shots=SHOTS_PER_GROUP,
                seed_sim=None,
                ibm_backend=IBM_BACKEND_NAME,
            ),
            out_csv=str(out_dir / f"samples_{task_label}_group{group_id}_ibm.csv"),
            extra_meta={
                "kras_task": task_label,
                "pdbid": pdbid,
                "sequence": sequence,
                "group_id": group_id,
                "run_seed": run_seed,
            },
        )

        runner = SamplingRunner(cfg, H)

        t0 = time.perf_counter()
        df = runner.run()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        print(f"[{task_label} | Group {group_id}] {len(df)} rows -> {cfg.out_csv} ({elapsed:.2f}s)")

        group_csvs.append(cfg.out_csv)
        timing_rows.append({
            "kras_task": task_label,
            "pdbid": pdbid,
            "group_id": group_id,
            "run_seed": run_seed,
            "rows": len(df),
            "seconds": round(elapsed, 6),
        })

    total_elapsed = time.perf_counter() - t0_total
    timing_df = pd.DataFrame(timing_rows)
    timing_df["total_seconds_for_task"] = round(total_elapsed, 6)
    timing_df.to_csv(out_dir / f"{task_label}_timing.csv", index=False)
    print(f"[Timing] {task_label} total: {total_elapsed:.2f}s")

    # Merge group CSVs for this fragment
    combined = []
    for fpath in group_csvs:
        try:
            combined.append(pd.read_csv(fpath))
        except Exception:
            pass

    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        out_path = out_dir / f"samples_{task_label}_all_ibm.csv"
        all_df.to_csv(out_path, index=False)
        print(f"[Merged] {task_label}: {len(all_df)} rows -> {out_path}")
        return str(out_path)

    return ""


if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # This line ensures local credentials are loaded; we don't pass `service` further
    service = init_ibm_service()

    # Read all KRAS tasks from CSV
    EXAMPLES: List[Dict[str, Any]] = read_tasks(TASKS_FILE)

    all_combined = []
    for ex in EXAMPLES:
        merged_path = per_example_sampling(
            ex["task_label"],
            ex["pdbid"],
            ex["main_chain_residue_seq"],
        )
        if merged_path:
            try:
                all_combined.append(pd.read_csv(merged_path))
            except Exception:
                pass

    # Global merged CSV across all KRAS fragments
    if all_combined:
        final_df = pd.concat(all_combined, ignore_index=True)
        final_path = OUTPUT_ROOT / "KRAS_samples_all_ibm.csv"
        final_df.to_csv(final_path, index=False)
        print(f"\n[Global merged] {len(final_df)} rows -> {final_path}")

    print("\nAll KRAS sampling runs completed.")
