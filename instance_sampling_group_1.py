# --*-- conding:utf-8 --*--
# @time:10/23/25 14:47
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:instance_sampling_group_1.py

import time
import random
from typing import Dict, Any, List, Tuple
from pathlib import Path
import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from sampling import SamplingRunner, SamplingConfig, BackendConfig
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem


IBM_CONFIG_FILE = "./ibm_config.txt"
TASKS_FILE = "./tasks.csv"

PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)
BETA_LIST: List[float] = [1.0, 2.0, 3.0, 4.0]
SEEDS: int = 3
REPS: int = 1

GROUP_COUNT = 10
SHOTS_PER_GROUP = 2000

OUTPUT_ROOT = Path("sampling_results")  # Root directory for results


def read_ibm_config(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                cfg[key.strip().upper()] = value.strip()
    except Exception as e:
        print(f"Failed to read IBM config: {e}")
    return cfg


cfg_data = read_ibm_config(IBM_CONFIG_FILE)
IBM_TOKEN = cfg_data.get("TOKEN", "")
IBM_INSTANCE = cfg_data.get("INSTANCE", None)
IBM_BACKEND_NAME = cfg_data.get("BACKEND", None)


def init_ibm_service() -> QiskitRuntimeService:
    if IBM_TOKEN:
        try:
            return QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=IBM_TOKEN,
                instance=IBM_INSTANCE
            )
        except Exception:
            return QiskitRuntimeService()
    return QiskitRuntimeService()


def build_protein_hamiltonian(sequence: str, penalties: Tuple[int, int, int]) -> SparsePauliOp:
    side_chain_residue_sequences = ['' for _ in range(len(sequence))]
    peptide = Peptide(sequence, side_chain_residue_sequences)
    mj_interaction = MiyazawaJerniganInteraction()
    penalty_terms = PenaltyParameters(*penalties)
    problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    H = problem.qubit_op()
    if isinstance(H, (list, tuple)) and len(H) > 0:
        H = H[0]
    if not isinstance(H, SparsePauliOp):
        H = SparsePauliOp(H)
    return H


def read_tasks(path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if "protein_name" in cols:
        pn_col = df.columns[cols.index("protein_name")]
    elif "pdbid" in cols:
        pn_col = df.columns[cols.index("pdbid")]
    elif "pdb_id" in cols:
        pn_col = df.columns[cols.index("pdb_id")]
    elif "protein" in cols:
        pn_col = df.columns[cols.index("protein")]
    else:
        raise ValueError("Input must contain column: protein_name / pdbid / pdb_id / protein")

    if "main_chain_residue_seq" in cols:
        seq_col = df.columns[cols.index("main_chain_residue_seq")]
    elif "sequence" in cols:
        seq_col = df.columns[cols.index("sequence")]
    else:
        raise ValueError("Input must contain column: main_chain_residue_seq or sequence")

    tasks: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        protein_name = str(row[pn_col]).strip()
        sequence = str(row[seq_col]).strip()
        if protein_name and sequence:
            tasks.append({"protein_name": protein_name, "main_chain_residue_seq": sequence})
    return tasks


def per_example_sampling(protein_name: str, sequence: str) -> str:
    print(f"\n=== Running {protein_name} ({sequence}) ===")

    out_dir = OUTPUT_ROOT / protein_name
    out_dir.mkdir(parents=True, exist_ok=True)

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
            label=f"qsad_ibm_{protein_name}_g{group_id}",
            backend=BackendConfig(
                kind="ibm",
                shots=SHOTS_PER_GROUP,
                seed_sim=None,  # Running on real backend, not simulator
                ibm_backend=IBM_BACKEND_NAME,
            ),
            out_csv=str(out_dir / f"samples_{protein_name}_group{group_id}_ibm.csv"),
            extra_meta={
                "protein": protein_name,
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

        print(f"[Group {group_id}] wrote {len(df)} rows -> {cfg.out_csv} (time: {elapsed:.2f}s)")
        group_csvs.append(cfg.out_csv)
        timing_rows.append({
            "protein_name": protein_name,
            "group_id": group_id,
            "run_seed": run_seed,
            "rows": int(len(df)),
            "seconds": round(elapsed, 6),
        })

    total_elapsed = time.perf_counter() - t0_total
    timing_df = pd.DataFrame(timing_rows)
    timing_df["total_seconds_for_protein"] = round(total_elapsed, 6)
    timing_csv = out_dir / f"{protein_name}_timing.csv"
    timing_df.to_csv(timing_csv, index=False)
    print(f"[Timing] -> {timing_csv} (total: {total_elapsed:.2f}s)")

    combined = []
    for fpath in group_csvs:
        try:
            df = pd.read_csv(fpath)
            combined.append(df)
        except Exception:
            pass
    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        merged_csv = out_dir / f"samples_{protein_name}_all_ibm.csv"
        all_df.to_csv(merged_csv, index=False)
        print(f"[Merged] {protein_name}: {len(all_df)} rows -> {merged_csv}")
        return str(merged_csv)
    return ""


if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    service = init_ibm_service()

    EXAMPLES: List[Dict[str, Any]] = read_tasks(TASKS_FILE)

    all_combined = []
    for ex in EXAMPLES:
        merged_path = per_example_sampling(ex["protein_name"], ex["main_chain_residue_seq"])
        if merged_path:
            try:
                df = pd.read_csv(merged_path)
                all_combined.append(df)
            except Exception:
                pass


    if all_combined:
        all_df = pd.concat(all_combined, ignore_index=True)
        out_all = OUTPUT_ROOT / "samples_all_ibm.csv"
        all_df.to_csv(out_all, index=False)
        print(f"\n[Global merged] all proteins -> {out_all} ({len(all_df)} rows)")

    print("\nAll sampling runs completed.")
