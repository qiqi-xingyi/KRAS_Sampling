# --*-- conding:utf-8 --*--
# @time:11/1/25 03:42
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:data.py

# GRN/data.py
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_ID_FIELDS = {"group_id", "protein", "sequence", "bitstring", "residues"}
_LABEL_FIELDS = {"rel", "rmsd"}

_AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
_AA_SET = set(_AA_LIST)
# simple AA groups
# _HYDROPHOBIC = set("AILMVFWYV")  # include V twice? ensure unique
_HYDROPHOBIC = set("AILMVFWYV".replace("V","")) | {"V"}  # ensure V once
_POLAR = set("STNQCY")
_POS = set("KRH")
_NEG = set("DE")
_AROM = set("FWY")


def _is_number(x) -> bool:
    try:
        if isinstance(x, bool):
            return False
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False


def _load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _seq_stats(seq: str) -> Tuple[List[float], List[str]]:
    """Return sequence-level statistical features and their names."""
    seq = (seq or "").strip().upper()
    L = max(1, len(seq))
    # 20 AA frequency (normalized)
    counts = {aa: 0 for aa in _AA_LIST}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1
    freq = [counts[aa] / L for aa in _AA_LIST]

    # physicochemical aggregates (ratios)
    n_hyd = sum(ch in _HYDROPHOBIC for ch in seq) / L
    n_pol = sum(ch in _POLAR for ch in seq) / L
    n_pos = sum(ch in _POS for ch in seq) / L
    n_neg = sum(ch in _NEG for ch in seq) / L
    n_arom = sum(ch in _AROM for ch in seq) / L
    n_pro = seq.count("P") / L
    n_gly = seq.count("G") / L
    charge_imbalance = (seq.count("K") + seq.count("R") + seq.count("H") - seq.count("D") - seq.count("E")) / L

    vec = freq + [n_hyd, n_pol, n_pos, n_neg, n_arom, n_pro, n_gly, charge_imbalance]
    names = [f"aa_freq_{aa}" for aa in _AA_LIST] + [
        "aa_ratio_hydrophobic", "aa_ratio_polar", "aa_ratio_pos",
        "aa_ratio_neg", "aa_ratio_arom", "aa_ratio_pro",
        "aa_ratio_gly", "aa_charge_imbalance"
    ]
    return vec, names


class GRNDataset(Dataset):
    """Tensor-ready dataset for GRN."""

    def __init__(
        self,
        rows: List[Dict],
        base_feature_names: List[str],
        scaler: Dict[str, Dict[str, float]],
        seq_feat_names: List[str],
    ):
        self.rows = rows
        self.base_feature_names = base_feature_names
        self.scaler = scaler
        self.seq_feat_names = seq_feat_names

        X, y, gid, rmsd = self._vectorize(rows)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.group_id = torch.from_numpy(gid).long()
        self.rmsd = torch.from_numpy(rmsd).float()

    def _vectorize(self, rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(rows)
        d_base = len(self.base_feature_names)
        d_seq = len(self.seq_feat_names)
        X = np.zeros((n, d_base + d_seq), dtype=np.float32)
        y = np.zeros((n,), dtype=np.int64)
        gid = np.zeros((n,), dtype=np.int64)
        rmsd = np.zeros((n,), dtype=np.float32)

        for i, r in enumerate(rows):
            # base numeric features (z-score)
            for j, col in enumerate(self.base_feature_names):
                val = r.get(col, 0.0)
                if not _is_number(val):
                    val = 0.0
                mu = self.scaler[col]["mean"]
                sd = self.scaler[col]["std"] or 1.0
                X[i, j] = (float(val) - mu) / (sd if sd != 0.0 else 1.0)

            # sequence stats (no scaling，保持在[0,1]范围；charge_imbalance在[-1,1])
            seq_vec, _ = _seq_stats(str(r.get("sequence", "")))
            X[i, d_base:d_base + d_seq] = np.asarray(seq_vec, dtype=np.float32)

            # labels/meta
            y[i] = int(r.get("rel", 0))
            gid[i] = int(r.get("group_id", -1))
            rmsd[i] = float(r.get("rmsd", float("nan")))
        return X, y, gid, rmsd

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "group_id": self.group_id[idx],
            "rmsd": self.rmsd[idx],
        }


class GRNDataModule:
    """
    - Auto-detect numeric base features from train.jsonl
    - Compute train-only mean/std for base features
    - Append per-sample sequence statistical features
    - Provide PyTorch DataLoaders
    """

    def __init__(
        self,
        data_dir: str = "training_dataset",
        train_file: str = "train.jsonl",
        valid_file: str = "valid.jsonl",
        test_file: str = "test.jsonl",
        batch_size: int = 1024,
        num_workers: int = 0,
        shuffle_train: bool = True,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / train_file
        self.valid_path = self.data_dir / valid_file
        self.test_path = self.data_dir / test_file

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.drop_last = drop_last

        self.base_feature_names: List[str] = []
        self.seq_feature_names: List[str] = []
        self.scaler: Dict[str, Dict[str, float]] = {}

        self.ds_train: Optional[GRNDataset] = None
        self.ds_valid: Optional[GRNDataset] = None
        self.ds_test: Optional[GRNDataset] = None

    def setup(self) -> None:
        rows_train = _load_jsonl(self.train_path)
        rows_valid = _load_jsonl(self.valid_path)
        rows_test = _load_jsonl(self.test_path)

        self.base_feature_names = self._infer_base_feature_columns(rows_train)
        self.scaler = self._compute_scaler(rows_train, self.base_feature_names)

        # derive names for seq features (fixed order)
        _, self.seq_feature_names = _seq_stats("ACDEFGHIKLMNPQRSTVWY")  # just to get names

        self.ds_train = GRNDataset(rows_train, self.base_feature_names, self.scaler, self.seq_feature_names)
        self.ds_valid = GRNDataset(rows_valid, self.base_feature_names, self.scaler, self.seq_feature_names)
        self.ds_test = GRNDataset(rows_test, self.base_feature_names, self.scaler, self.seq_feature_names)

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.ds_train, shuffle=self.shuffle_train)

    def valid_dataloader(self) -> DataLoader:
        return self._make_loader(self.ds_valid, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.ds_test, shuffle=False)

    def feature_dim(self) -> int:
        # base + seq-stats
        return len(self.base_feature_names) + len(self.seq_feature_names)

    # helpers
    def _make_loader(self, ds: GRNDataset, shuffle: bool) -> DataLoader:
        assert ds is not None, "Call setup() first."
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def _infer_base_feature_columns(self, rows: List[Dict]) -> List[str]:
        cand = set()
        for r in rows:
            for k, v in r.items():
                if k in _ID_FIELDS or k in _LABEL_FIELDS:
                    continue
                if _is_number(v):
                    cand.add(k)
        cols = sorted(cand)
        if not cols:
            raise ValueError("No numeric feature columns found in train set.")
        return cols

    def _compute_scaler(self, rows: List[Dict], cols: List[str]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for c in cols:
            vals: List[float] = []
            for r in rows:
                v = r.get(c, None)
                if _is_number(v):
                    vals.append(float(v))
            if len(vals) == 0:
                stats[c] = {"mean": 0.0, "std": 1.0}
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            stats[c] = {"mean": mean, "std": (std if std != 0.0 else 1.0)}
        return stats
