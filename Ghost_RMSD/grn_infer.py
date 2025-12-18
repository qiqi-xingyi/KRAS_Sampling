# --*-- conding:utf-8 --*--
# @time:12/17/25 20:41
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:grn_infer.py

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .GRN.model import GRNClassifier

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def _pick_device(arg_device: Optional[str]) -> torch.device:
    if arg_device and arg_device != "auto":
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_checkpoint_path(arg_path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    p = Path(arg_path)
    if p.exists():
        return p
    if base_dir is not None:
        alt = base_dir / p
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Cannot find checkpoint at '{p}' (base_dir={base_dir})")


def _load_jsonl_to_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _seq_features_from_sequence(seq: str, feature_names: List[str]) -> Dict[str, float]:
    seq = (seq or "").strip().upper()
    n = max(1, len(seq))
    counts = {aa: 0 for aa in AA_ORDER}
    for ch in seq:
        if ch in counts:
            counts[ch] += 1

    feats: Dict[str, float] = {}
    if "seq_len" in feature_names:
        feats["seq_len"] = float(n)
    for aa in AA_ORDER:
        k1 = f"aa_count_{aa}"
        if k1 in feature_names:
            feats[k1] = float(counts[aa])
    for aa in AA_ORDER:
        k2 = f"aa_frac_{aa}"
        if k2 in feature_names:
            feats[k2] = float(counts[aa]) / float(n)
    return feats


def _make_identity_scaler(expected_dim: int):
    class _Id:
        def __init__(self, dim): self.n_features_in_ = dim
        def transform(self, X): return X
    return _Id(expected_dim)


def _rebuild_scaler_if_needed(scaler_obj: Any, expected_dim: int) -> Any:
    if hasattr(scaler_obj, "transform"):
        return scaler_obj
    if scaler_obj is None or (isinstance(scaler_obj, str) and scaler_obj.lower() in {"identity", "none", "null"}):
        return _make_identity_scaler(expected_dim)

    if isinstance(scaler_obj, dict):
        d = scaler_obj
        if "state" in d and isinstance(d["state"], dict):
            d = d["state"]

        def _arr(key):
            v = d.get(key, None)
            if v is None:
                return None
            return np.asarray(v, dtype=float)

        mean = _arr("mean_") or _arr("mean") or _arr("center_") or _arr("center")
        scale = _arr("scale_") or _arr("scale") or _arr("std") or _arr("std_")

        if mean is not None and scale is not None:
            try:
                from sklearn.preprocessing import StandardScaler
                s = StandardScaler()
                s.mean_ = mean
                s.scale_ = scale
                s.var_ = (scale ** 2)
                s.n_features_in_ = int(mean.shape[0])
                return s
            except Exception:
                class _Lite:
                    def __init__(self, mean, scale):
                        self.mean_ = mean
                        self.scale_ = np.where(scale == 0, 1.0, scale)
                        self.n_features_in_ = int(mean.shape[0])
                    def transform(self, X):
                        return (X - self.mean_) / self.scale_
                return _Lite(mean, scale)

        feat_names = d.get("feature_names_", None)
        if isinstance(feat_names, (list, tuple)) and len(feat_names) == expected_dim:
            return _make_identity_scaler(expected_dim)

    print("[WARN] Unrecognized scaler in checkpoint; using identity transform.")
    return _make_identity_scaler(expected_dim)


def _ensure_base_features(df: pd.DataFrame, base_feature_names: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in base_feature_names:
        if col not in out.columns:
            out[col] = 0.0
    return out


def _build_design_matrix(
    df: pd.DataFrame,
    base_feature_names: List[str],
    seq_feature_names: List[str],
    scaler: Any,
) -> np.ndarray:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if "sequence" not in df.columns:
        raise ValueError("Missing required column: sequence")

    df2 = _ensure_base_features(df, base_feature_names)
    X_base = df2[base_feature_names].astype(float).to_numpy(copy=True)
    X_base = scaler.transform(X_base)

    seq_rows = [_seq_features_from_sequence(s, seq_feature_names) for s in df["sequence"].astype(str).tolist()]
    X_seq = pd.DataFrame(seq_rows, columns=seq_feature_names).fillna(0.0).to_numpy(copy=True)
    return np.concatenate([X_base, X_seq], axis=1)


@torch.no_grad()
def _infer_batches(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    score_mode: str = "expected_rel",
) -> np.ndarray:
    N = X.shape[0]
    scores_list: List[np.ndarray] = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device, non_blocking=True)
        out = model(xb)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=-1)

        if score_mode == "prob_rel3":
            score = prob[:, 3]
        elif score_mode == "logit_rel3":
            score = logits[:, 3]
        else:
            classes = torch.arange(logits.size(1), device=logits.device).float()
            score = (prob * classes[None, :]).sum(dim=-1)

        scores_list.append(score.detach().cpu().numpy())
    return np.concatenate(scores_list, axis=0)


def _rank_within_groups(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["score"] = scores
    if "group_id" not in out.columns:
        out["group_id"] = 0
    out["rank_in_group"] = (
        out.groupby(["pdb_id", "group_id"])["score"]
           .rank(method="first", ascending=False)
           .astype(int)
    )
    return out.sort_values(
        ["pdb_id", "group_id", "rank_in_group", "score"],
        ascending=[True, True, True, False]
    )


class GRNInferencer:
    """
    Library-style wrapper for GRN inference.

    Typical usage:
        inf = GRNInferencer(ckpt="checkpoints_full/grn_best.pt")
        df_ranked = inf.predict_jsonl("kras_combined.jsonl", topk=50)
    """

    def __init__(
        self,
        ckpt: Union[str, Path] = "checkpoints_full/grn_best.pt",
        device: str = "auto",
        batch_size: int = 4096,
        score_mode: str = "expected_rel",
        allow_omp_dup: bool = False,
    ):
        if allow_omp_dup and os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        self.base_dir = Path(__file__).resolve().parent  # Ghost_RMSD/
        self.device = _pick_device(device)
        self.batch_size = int(batch_size)
        self.score_mode = str(score_mode)

        ckpt_path = _resolve_checkpoint_path(ckpt, base_dir=self.base_dir)
        self.ckpt = torch.load(ckpt_path, map_location="cpu")

        self.base_feature_names: List[str] = self.ckpt["base_feature_names"]
        self.seq_feature_names: List[str] = self.ckpt["seq_feature_names"]
        self.scaler = _rebuild_scaler_if_needed(self.ckpt.get("scaler", None), expected_dim=len(self.base_feature_names))

        self.model = self._build_model(input_dim=len(self.base_feature_names) + len(self.seq_feature_names)).to(self.device)

    def _build_model(self, input_dim: int) -> nn.Module:
        model = GRNClassifier(
            in_dim=input_dim,
            hidden_dims=[512, 256, 128],
            dropout=self.ckpt.get("args", {}).get("dropout", 0.3),
            use_rank_head=True,
        )
        model.load_state_dict(self.ckpt["model_state"], strict=True)
        model.eval()
        return model

    def predict_df(self, df: pd.DataFrame, topk: Optional[int] = None) -> pd.DataFrame:
        for c in ["pdb_id", "sequence", "bitstring"]:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")
        if "group_id" not in df.columns:
            df = df.copy()
            df["group_id"] = 0

        X = _build_design_matrix(df, self.base_feature_names, self.seq_feature_names, self.scaler)
        scores = _infer_batches(self.model, X, self.device, batch_size=self.batch_size, score_mode=self.score_mode)
        ranked = _rank_within_groups(df, scores)

        if topk is not None and topk > 0:
            ranked = (
                ranked.sort_values(["pdb_id", "group_id", "score"], ascending=[True, True, False])
                      .groupby(["pdb_id", "group_id"])
                      .head(int(topk))
                      .reset_index(drop=True)
            )
        return ranked

    def predict_jsonl(self, input_jsonl: Union[str, Path], topk: Optional[int] = None) -> pd.DataFrame:
        df = _load_jsonl_to_df(input_jsonl)
        return self.predict_df(df, topk=topk)

    def predict_records(self, records: List[Dict[str, Any]], topk: Optional[int] = None) -> pd.DataFrame:
        return self.predict_df(pd.DataFrame(records), topk=topk)
