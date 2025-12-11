# --*-- conding:utf-8 --*--
# @time:11/1/25 03:52
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:metrics.py

# grn_simple/metrics.py
import math
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import spearmanr  # optional
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _group_indices(group_ids: np.ndarray) -> Dict[int, np.ndarray]:
    groups: Dict[int, List[int]] = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(int(g), []).append(i)
    return {g: np.asarray(idx, dtype=np.int64) for g, idx in groups.items()}


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.size == 0:
        return 0.0
    return float((y_pred == y_true).mean())


def _dcg_at_k(rels: np.ndarray, k: int) -> float:
    k = min(k, rels.shape[0])
    if k <= 0:
        return 0.0
    gains = (2.0 ** rels[:k] - 1.0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gains * discounts))


def ndcg_by_group(scores: np.ndarray, rels: np.ndarray, group_ids: np.ndarray, k: int) -> float:
    groups = _group_indices(group_ids)
    vals: List[float] = []
    for g, idx in groups.items():
        if idx.size == 0:
            continue
        s = scores[idx]
        r = rels[idx]
        order = np.argsort(-s, kind="mergesort")
        ideal = np.argsort(-r, kind="mergesort")
        dcg = _dcg_at_k(r[order], k)
        idcg = _dcg_at_k(r[ideal], k)
        if idcg > 0:
            vals.append(dcg / idcg)
        else:
            vals.append(0.0)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def spearman_by_group(scores: np.ndarray, rmsd: np.ndarray, group_ids: np.ndarray) -> float:
    """
    Compute Spearman correlation between predicted score and (-rmsd) per group, then average.
    Higher is better (since lower RMSD => higher quality).
    """
    groups = _group_indices(group_ids)
    vals: List[float] = []
    for g, idx in groups.items():
        if idx.size < 2:
            continue
        s = scores[idx]
        y = -rmsd[idx]  # invert RMSD so larger is better
        if _HAVE_SCIPY:
            rho = spearmanr(s, y, nan_policy="omit").correlation
        else:
            # Fallback: rank + Pearson on ranks
            s_rank = s.argsort().argsort().astype(np.float64)
            y_rank = y.argsort().argsort().astype(np.float64)
            s_rank -= s_rank.mean()
            y_rank -= y_rank.mean()
            denom = (np.linalg.norm(s_rank) * np.linalg.norm(y_rank))
            rho = float(np.sum(s_rank * y_rank) / denom) if denom > 0 else 0.0
        if not np.isnan(rho):
            vals.append(float(rho))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def summarize_classification(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    y_pred = logits.argmax(axis=1)
    acc = accuracy(y_pred, labels)
    return {"acc": acc}


def summarize_ranking(
    logits: np.ndarray,
    scores: np.ndarray,
    rels: np.ndarray,
    rmsd: np.ndarray,
    group_ids: np.ndarray,
    ks: Tuple[int, ...] = (5, 10, 20),
) -> Dict[str, float]:
    out = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_by_group(scores, rels, group_ids, k)
    out["spearman"] = spearman_by_group(scores, rmsd, group_ids)
    return out
