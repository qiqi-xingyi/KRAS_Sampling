# --*-- conding:utf-8 --*--
# @time:12/10/25 22:58
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:instance_sampling.py

import time
import random
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd

from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from sampling import SamplingRunner, SamplingConfig, BackendConfig
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem


TASKS_FILE = "./tasks2.csv"

# Penalty parameters for protein folding Hamiltonian
PENALTY_PARAMS: Tuple[int, int, int] = (10, 10, 10)

# Sampling hyperparameters
BETA_LIST: List[float] = [1.0, 2.0, 3.0, 4.0]
SEEDS: int = 3
REPS: int = 1

GROUP_COUNT = 10
SHOTS_PER_GROUP = 2000

# Output root
OUTPUT_ROOT = Path("QDock_sampling_results")

# If you want to fix a backend, put its name here; otherwise set to None
IBM_BACKEND_NAME: str | None = "ibm_cleveland"  # or None to let SamplingRunner decide


# -------------------------
# Checkpointing parameters
# -------------------------
TASK_PROGRESS_NAME = "progress.json"          # per-task, inside each task folder
GLOBAL_PROGRESS_NAME = "progress_global.json" # global, inside OUTPUT_ROOT
WRITE_PROGRESS_EVERY_GROUP = True             # always true for robustness


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # If corrupted, do not crash; treat as empty.
        return {}


def save_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def csv_is_valid(path: Path) -> bool:
    """
    Conservative validation:
    - file exists
    - can be read by pandas
    - has at least 1 row
    """
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        return len(df) > 0
    except Exception:
        return False


def init_ibm_service() -> QiskitRuntimeService:
    """
    Initialize QiskitRuntimeService using ONLY the locally saved IBM account.
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
    Read tasks from CSV and generate a label for each row.

    Expected columns (case-insensitive):
        - pdbid (preferred) or protein_name / pdb_id / protein
        - main_chain_residue_seq or sequence
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
    counters: Dict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        pdbid = str(row[pn_col]).strip()
        sequence = str(row[seq_col]).strip()
        if not pdbid or not sequence:
            continue

        counters[pdbid] += 1
        task_label = f"KRAS_{pdbid}_{counters[pdbid]}"

        tasks.append({
            "pdbid": pdbid,
            "main_chain_residue_seq": sequence,
            "task_label": task_label,
        })

    return tasks


def per_example_sampling(task_label: str, pdbid: str, sequence: str) -> str:
    """
    Run sampling for a single fragment, with resume support:
    - For each group_id, if its CSV already exists and is valid, skip it.
    - Progress is stored in out_dir/progress.json after each group.
    - If interrupted, re-run will continue from the first unfinished group.
    """
    print(f"\n=== Running {task_label} | PDB: {pdbid} | Seq: {sequence} ===")

    out_dir = OUTPUT_ROOT / task_label
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / TASK_PROGRESS_NAME
    progress = load_json(progress_path)

    # Initialize progress metadata (do not overwrite existing)
    progress.setdefault("task_label", task_label)
    progress.setdefault("pdbid", pdbid)
    progress.setdefault("sequence", sequence)
    progress.setdefault("created_at", progress.get("created_at") or utc_now_iso())
    progress.setdefault("settings", {})
    progress["settings"].setdefault("PENALTY_PARAMS", list(PENALTY_PARAMS))
    progress["settings"].setdefault("BETA_LIST", list(BETA_LIST))
    progress["settings"].setdefault("SEEDS", SEEDS)
    progress["settings"].setdefault("REPS", REPS)
    progress["settings"].setdefault("GROUP_COUNT", GROUP_COUNT)
    progress["settings"].setdefault("SHOTS_PER_GROUP", SHOTS_PER_GROUP)
    progress["settings"].setdefault("IBM_BACKEND_NAME", IBM_BACKEND_NAME)

    progress.setdefault("groups", {})  # group_id -> dict

    # Build Hamiltonian once per task (safe even for resume)
    H = build_protein_hamiltonian(sequence, PENALTY_PARAMS)

    group_csvs: List[str] = []
    timing_rows: List[Dict[str, Any]] = []

    t0_total = time.perf_counter()
    any_ran = False

    for group_id in range(GROUP_COUNT):
        group_csv_path = out_dir / f"samples_{task_label}_group{group_id}_ibm.csv"

        # Resume: skip if already done AND CSV is valid
        gkey = str(group_id)
        gstate = progress["groups"].get(gkey, {})
        if gstate.get("status") == "done" and csv_is_valid(group_csv_path):
            print(f"[{task_label} | Group {group_id}] Skip (already done): {group_csv_path}")
            group_csvs.append(str(group_csv_path))
            timing_rows.append({
                "kras_task": task_label,
                "pdbid": pdbid,
                "group_id": group_id,
                "run_seed": gstate.get("run_seed"),
                "rows": gstate.get("rows", 0),
                "seconds": gstate.get("seconds", None),
            })
            continue

        # If CSV exists but progress says not done, treat it as done if valid
        if csv_is_valid(group_csv_path):
            print(f"[{task_label} | Group {group_id}] Found existing valid CSV, marking done: {group_csv_path}")
            df_existing = pd.read_csv(group_csv_path)
            progress["groups"][gkey] = {
                "status": "done",
                "csv": str(group_csv_path),
                "rows": int(len(df_existing)),
                "seconds": gstate.get("seconds", None),
                "run_seed": gstate.get("run_seed", None),
                "finished_at": gstate.get("finished_at", utc_now_iso()),
            }
            if WRITE_PROGRESS_EVERY_GROUP:
                save_json_atomic(progress_path, progress)

            group_csvs.append(str(group_csv_path))
            timing_rows.append({
                "kras_task": task_label,
                "pdbid": pdbid,
                "group_id": group_id,
                "run_seed": progress["groups"][gkey].get("run_seed"),
                "rows": len(df_existing),
                "seconds": progress["groups"][gkey].get("seconds", None),
            })
            continue

        # Otherwise run this group
        any_ran = True
        run_seed = random.randint(1, 2**31 - 1)

        progress["groups"][gkey] = {
            "status": "running",
            "csv": str(group_csv_path),
            "run_seed": run_seed,
            "started_at": utc_now_iso(),
        }
        if WRITE_PROGRESS_EVERY_GROUP:
            save_json_atomic(progress_path, progress)

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
            out_csv=str(group_csv_path),
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
        try:
            df = runner.run()
        except KeyboardInterrupt:
            # Save state before re-raising to allow clean resume
            progress["groups"][gkey]["status"] = "interrupted"
            progress["groups"][gkey]["interrupted_at"] = utc_now_iso()
            save_json_atomic(progress_path, progress)
            print(f"\n[Interrupted] {task_label} group {group_id}. Progress saved to {progress_path}")
            raise
        except Exception as e:
            progress["groups"][gkey]["status"] = "failed"
            progress["groups"][gkey]["error"] = repr(e)
            progress["groups"][gkey]["failed_at"] = utc_now_iso()
            save_json_atomic(progress_path, progress)
            print(f"[Failed] {task_label} group {group_id}: {e!r}")
            # Continue to next group (or you can choose to raise)
            continue
        t1 = time.perf_counter()

        elapsed = t1 - t0
        n_rows = int(len(df))

        print(f"[{task_label} | Group {group_id}] {n_rows} rows -> {cfg.out_csv} ({elapsed:.2f}s)")

        progress["groups"][gkey] = {
            "status": "done",
            "csv": str(group_csv_path),
            "run_seed": run_seed,
            "rows": n_rows,
            "seconds": round(elapsed, 6),
            "finished_at": utc_now_iso(),
        }
        if WRITE_PROGRESS_EVERY_GROUP:
            save_json_atomic(progress_path, progress)

        group_csvs.append(cfg.out_csv)
        timing_rows.append({
            "kras_task": task_label,
            "pdbid": pdbid,
            "group_id": group_id,
            "run_seed": run_seed,
            "rows": n_rows,
            "seconds": round(elapsed, 6),
        })

    total_elapsed = time.perf_counter() - t0_total

    # Always (re)write timing.csv from current known states (including skipped groups)
    timing_df = pd.DataFrame(timing_rows)
    timing_df["total_seconds_for_task"] = round(total_elapsed, 6)
    timing_df.to_csv(out_dir / f"{task_label}_timing.csv", index=False)
    print(f"[Timing] {task_label} total (this run walltime): {total_elapsed:.2f}s")

    # Determine if all groups are done (for merge gating)
    all_done = True
    for group_id in range(GROUP_COUNT):
        gkey = str(group_id)
        gstate = progress.get("groups", {}).get(gkey, {})
        group_csv_path = out_dir / f"samples_{task_label}_group{group_id}_ibm.csv"
        if not (gstate.get("status") == "done" and csv_is_valid(group_csv_path)):
            all_done = False
            break

    # Merge only when fully complete
    if all_done:
        combined = []
        for group_id in range(GROUP_COUNT):
            fpath = out_dir / f"samples_{task_label}_group{group_id}_ibm.csv"
            try:
                combined.append(pd.read_csv(fpath))
            except Exception:
                # Shouldn't happen if validated, but keep robust
                pass

        if combined:
            all_df = pd.concat(combined, ignore_index=True)
            out_path = out_dir / f"samples_{task_label}_all_ibm.csv"
            all_df.to_csv(out_path, index=False)
            print(f"[Merged] {task_label}: {len(all_df)} rows -> {out_path}")

            progress["status"] = "done"
            progress["merged_csv"] = str(out_path)
            progress["updated_at"] = utc_now_iso()
            save_json_atomic(progress_path, progress)

            return str(out_path)

    # Not fully done yet; save progress and return empty (so global merge can be gated)
    progress["status"] = "partial"
    progress["updated_at"] = utc_now_iso()
    save_json_atomic(progress_path, progress)

    if not any_ran:
        print(f"[Info] {task_label}: no new groups ran in this execution (resume skip only).")

    return ""


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Ensure local credentials are loaded
    service = init_ibm_service()

    # Read tasks
    EXAMPLES: List[Dict[str, Any]] = read_tasks(TASKS_FILE)

    global_progress_path = OUTPUT_ROOT / GLOBAL_PROGRESS_NAME
    global_progress = load_json(global_progress_path)
    global_progress.setdefault("created_at", global_progress.get("created_at") or utc_now_iso())
    global_progress.setdefault("tasks", {})  # task_label -> {status, merged_csv, ...}

    all_combined = []

    try:
        for ex in EXAMPLES:
            task_label = ex["task_label"]
            pdbid = ex["pdbid"]
            seq = ex["main_chain_residue_seq"]

            # If global progress says done and merged exists, skip task
            gtask = global_progress["tasks"].get(task_label, {})
            merged_guess = (OUTPUT_ROOT / task_label / f"samples_{task_label}_all_ibm.csv")
            if gtask.get("status") == "done" and merged_guess.exists():
                print(f"\n=== Skip task (global done): {task_label} ===")
                try:
                    all_combined.append(pd.read_csv(merged_guess))
                except Exception:
                    pass
                continue

            merged_path = per_example_sampling(task_label, pdbid, seq)

            # Update global progress
            if merged_path:
                global_progress["tasks"][task_label] = {
                    "status": "done",
                    "merged_csv": merged_path,
                    "updated_at": utc_now_iso(),
                }
                save_json_atomic(global_progress_path, global_progress)

                try:
                    all_combined.append(pd.read_csv(merged_path))
                except Exception:
                    pass
            else:
                global_progress["tasks"].setdefault(task_label, {})
                global_progress["tasks"][task_label].update({
                    "status": "partial",
                    "updated_at": utc_now_iso(),
                })
                save_json_atomic(global_progress_path, global_progress)

    except KeyboardInterrupt:
        print("\n[Stopped] KeyboardInterrupt received. You can re-run to resume unfinished groups/tasks.")
        # Do not proceed to global merge
        raise

    # Global merged CSV only when ALL tasks have merged outputs
    all_tasks_done = True
    for ex in EXAMPLES:
        task_label = ex["task_label"]
        merged_guess = (OUTPUT_ROOT / task_label / f"samples_{task_label}_all_ibm.csv")
        if not merged_guess.exists():
            all_tasks_done = False
            break

    if all_tasks_done and all_combined:
        final_df = pd.concat(all_combined, ignore_index=True)
        final_path = OUTPUT_ROOT / "KRAS_samples_all_ibm.csv"
        final_df.to_csv(final_path, index=False)
        print(f"\n[Global merged] {len(final_df)} rows -> {final_path}")

        global_progress["status"] = "done"
        global_progress["global_merged_csv"] = str(final_path)
        global_progress["updated_at"] = utc_now_iso()
        save_json_atomic(global_progress_path, global_progress)
    else:
        print("\n[Global merged] Not produced (some tasks are not fully completed yet).")
        global_progress["status"] = "partial"
        global_progress["updated_at"] = utc_now_iso()
        save_json_atomic(global_progress_path, global_progress)

    print("\nAll sampling runs completed (or checkpointed).")
