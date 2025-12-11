# --*-- coding:utf-8 --*--
# @time:11/2/25 17:26 (patched 11/3/25)
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py
#
# GRN inference over GRN input JSONL.
# - Robust checkpoint path resolution (script-dir fallback).
# - Rebuild sklearn StandardScaler from dict (mean_/scale_) if needed.
# - Device auto-pick (cuda/mps/cpu).
# - Per-(pdb_id, group_id) ranking.
# - Optional top-k export; zero-fill missing base features.

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from GRN.model import GRNClassifier

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard residues


def pick_device(arg_device: Optional[str]) -> torch.device:
    if arg_device and arg_device != "auto":
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint_path(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent / arg_path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Cannot find checkpoint at '{arg_path}' or '{alt}'")


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def seq_features_from_sequence(seq: str, feature_names: List[str]) -> Dict[str, float]:
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
        key = f"aa_count_{aa}"
        if key in feature_names:
            feats[key] = float(counts[aa])
    for aa in AA_ORDER:
        key = f"aa_frac_{aa}"
        if key in feature_names:
            feats[key] = float(counts[aa]) / float(n)
    return feats


def _make_identity_scaler(expected_dim: int):
    class _Id:
        def __init__(self, dim): self.n_features_in_ = dim
        def transform(self, X): return X
    return _Id(expected_dim)

def rebuild_scaler_if_needed(scaler_obj: Any, expected_dim: int) -> Any:
    """
    Accepts:
      - sklearn-like object with .transform
      - dict with any of:
          mean_/scale_, mean/std, state:{mean_, scale_}, center/scale
      - "identity"/None   -> identity scaler
    Fallback to identity scaler if nothing matches (with a warning).
    """
    # 1) already a transformer
    if hasattr(scaler_obj, "transform"):
        return scaler_obj

    # 2) explicit identity / none
    if scaler_obj is None or (isinstance(scaler_obj, str) and scaler_obj.lower() in {"identity", "none", "null"}):
        return _make_identity_scaler(expected_dim)

    # 3) dict-like formats
    if isinstance(scaler_obj, dict):
        d = scaler_obj

        # nested state
        if "state" in d and isinstance(d["state"], dict):
            d = d["state"]

        # try common key variants
        def _arr(key):
            v = d.get(key, None)
            if v is None: return None
            return np.asarray(v, dtype=float)

        mean = _arr("mean_") or _arr("mean") or _arr("center_") or _arr("center")
        scale = _arr("scale_") or _arr("scale") or _arr("std") or _arr("std_")

        if mean is not None and scale is not None:
            # rebuild StandardScaler if可用，否则构造轻量transformer
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

    # 4) ultimate fallback: identity
    print("[WARN] 'scaler' in checkpoint is unrecognized; using identity transform.")
    return _make_identity_scaler(expected_dim)


def ensure_base_features(df: pd.DataFrame, base_feature_names: List[str]) -> pd.DataFrame:
    """Ensure all base features exist; create missing columns filled with 0.0."""
    out = df.copy()
    for col in base_feature_names:
        if col not in out.columns:
            out[col] = 0.0
    return out


def build_design_matrix(
    df: pd.DataFrame,
    base_feature_names: List[str],
    seq_feature_names: List[str],
    scaler
) -> np.ndarray:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if "sequence" not in df.columns:
        raise ValueError("Missing required column: sequence")

    df2 = ensure_base_features(df, base_feature_names)
    X_base = df2[base_feature_names].astype(float).to_numpy(copy=True)
    X_base = scaler.transform(X_base)

    seq_rows: List[Dict[str, float]] = [
        seq_features_from_sequence(seq, seq_feature_names) for seq in df["sequence"].astype(str).tolist()
    ]
    X_seq = pd.DataFrame(seq_rows, columns=seq_feature_names).fillna(0.0).to_numpy(copy=True)

    X = np.concatenate([X_base, X_seq], axis=1)
    return X


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    input_dim: int,
    num_classes: int = 4,
    dropout: float = 0.3,
) -> nn.Module:
    model = GRNClassifier(
        in_dim=input_dim,
        hidden_dims=[512, 256, 128],
        dropout=dropout,
        use_rank_head=True,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_batches(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    score_mode: str = "expected_rel",
) -> Tuple[np.ndarray, np.ndarray]:
    N = X.shape[0]
    logits_list = []
    scores_list = []
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).float().to(device, non_blocking=True)
        out = model(xb)
        logits = out["logits"]
        prob = torch.softmax(logits, dim=-1)

        if score_mode == "prob_rel3":
            score = prob[:, 3]
        elif score_mode == "logit_rel3":
            score = logits[:, 3]
        else:  # expected_rel
            classes = torch.arange(logits.size(1), device=logits.device).float()
            score = (prob * classes[None, :]).sum(dim=-1)

        logits_list.append(logits.cpu().numpy())
        scores_list.append(score.cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    return logits, scores


def rank_within_groups(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["score"] = scores
    if "group_id" not in out.columns:
        out["group_id"] = 0
    # rank within (pdb_id, group_id)
    out["rank_in_group"] = (
        out.groupby(["pdb_id", "group_id"])["score"]
           .rank(method="first", ascending=False)
           .astype(int)
    )
    return out.sort_values(
        ["pdb_id", "group_id", "rank_in_group", "score"],
        ascending=[True, True, True, False]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints_full/grn_best.pt")
    ap.add_argument("--input_jsonl", type=str, required=True, help="New combined JSONL for inference")
    ap.add_argument("--out_csv", type=str, default="predictions.csv")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--score_mode", type=str, default="expected_rel",
                    choices=["prob_rel3", "logit_rel3", "expected_rel"])
    ap.add_argument("--topk", type=int, default=50, help="Optional per-group topk CSV export")
    ap.add_argument("--allow_omp_dup", action="store_true",
                    help="Set KMP_DUPLICATE_LIB_OK=TRUE for macOS OpenMP duplication issues")
    args = ap.parse_args()

    if args.allow_omp_dup and os.environ.get("KMP_DUPLICATE_LIB_OK") != "TRUE":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = pick_device(args.device)
    if device.type == "cuda":
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        print("[Device] Using Apple MPS backend")
    else:
        print("[Device] Using CPU")

    ckpt_path = resolve_checkpoint_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    base_feature_names: List[str] = ckpt["base_feature_names"]
    seq_feature_names: List[str] = ckpt["seq_feature_names"]
    scaler = rebuild_scaler_if_needed(ckpt.get("scaler", None), expected_dim=len(base_feature_names))

    df = load_jsonl(Path(args.input_jsonl))
    for c in ["pdb_id", "sequence", "bitstring"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "group_id" not in df.columns:
        df["group_id"] = 0

    X = build_design_matrix(df, base_feature_names, seq_feature_names, scaler)
    model = build_model_from_ckpt(
        ckpt, input_dim=X.shape[1], num_classes=4, dropout=ckpt.get("args", {}).get("dropout", 0.3)
    ).to(device)

    _, scores = infer_batches(model, X, device, batch_size=args.batch_size, score_mode=args.score_mode)
    df_ranked = rank_within_groups(df, scores)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_ranked.to_csv(out_csv, index=False)
    print(f"[SAVED] Full predictions: {out_csv.resolve()}")

    if args.topk and args.topk > 0:
        tops = (
            df_ranked.sort_values(["pdb_id", "group_id", "score"], ascending=[True, True, False])
                    .groupby(["pdb_id", "group_id"])
                    .head(args.topk)
        )
        out_top = out_csv.with_name(out_csv.stem + f"_top{args.topk}.csv")
        tops.to_csv(out_top, index=False)
        print(f"[SAVED] Per-group top-{args.topk}: {out_top.resolve()}")


if __name__ == "__main__":
    main()
