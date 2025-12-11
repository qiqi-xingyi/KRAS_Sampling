# --*-- conding:utf-8 --*--
# @time:10/19/25 10:18
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:sampler.py

from __future__ import annotations
from typing import List, Optional, Dict, Any
from qiskit.quantum_info import SparsePauliOp

from .config import SamplingConfig
from .backends import make_backend
from .circuits import make_sampling_circuit, build_ansatz, random_params
from .utils import counts_to_rows, write_csv, circuit_hash


class SamplingRunner:
    """
    Orchestrates the full quantum sampling process.

    Parameters
    ----------
    cfg : SamplingConfig
        Configuration describing seeds, betas, backend, etc.
    H_sparse : SparsePauliOp
        The Hamiltonian defining the system.
    meta : Optional[dict]
        Optional metadata to be written into each CSV row.
    """

    def __init__(
        self,
        cfg: SamplingConfig,
        H_sparse: SparsePauliOp,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.H = H_sparse
        self.meta = meta or {}
        self.cfg.validate()

    def run(self):
        """Run all sampling jobs and return a pandas DataFrame."""
        be = make_backend(
            kind=self.cfg.backend.kind,
            shots=self.cfg.backend.shots,
            seed_sim=self.cfg.backend.seed_sim,
            ibm_backend=self.cfg.backend.ibm_backend,
        )

        rows: List[dict] = []
        n_qubits = getattr(self.H, "num_qubits", self.cfg.n_qubits_hint or 1)

        # Pre-build ansatz template once
        ans = build_ansatz(n_qubits, reps=self.cfg.reps, entanglement=self.cfg.entanglement)

        for seed in range(self.cfg.seeds):
            params = random_params(ans, seed)
            for beta in self.cfg.betas:
                circ = make_sampling_circuit(
                    n_qubits=n_qubits,
                    H=self.H,
                    beta=beta,
                    params=params,
                    reps=self.cfg.reps,
                    entanglement=self.cfg.entanglement,
                )

                counts = be.run_counts(circ)

                meta = dict(
                    L=self.cfg.L,
                    n_qubits=n_qubits,
                    shots=self.cfg.backend.shots,
                    beta=float(beta),
                    seed=int(seed),
                    label=self.cfg.label,
                    backend=self.cfg.backend.kind,
                    ibm_backend=self.cfg.backend.ibm_backend,
                    circuit_hash=circuit_hash(circ.qasm()) if hasattr(circ, "qasm") else None,
                    **self.cfg.extra_meta,
                    **self.meta,
                )


                rows.extend(counts_to_rows(counts, meta))

        df = write_csv(rows, self.cfg.out_csv, write_parquet=self.cfg.write_parquet)
        return df
