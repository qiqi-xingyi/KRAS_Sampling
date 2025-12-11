# --*-- conding:utf-8 --*--
# @time:11/1/25 16:43
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:losses.py

import torch
import torch.nn.functional as F
from typing import Tuple

def build_pairs(group_id: torch.Tensor, rel: torch.Tensor, max_pairs_per_group: int = 32) -> torch.Tensor:
    """
    Build pair indices (i, j) within each group where rel[i] > rel[j].
    Returns LongTensor of shape [P, 2]. Works on CPU tensors.
    """
    g = group_id.cpu().numpy()
    r = rel.cpu().numpy()
    pairs = []
    # group indices
    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, gid in enumerate(g):
        buckets[int(gid)].append(idx)
    for _, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        # simple random sampling of preference pairs
        import random
        tried = 0
        made = 0
        while made < max_pairs_per_group and tried < 10 * max_pairs_per_group:
            i, j = random.sample(idxs, 2)
            tried += 1
            if r[i] > r[j]:
                pairs.append((i, j))
                made += 1
            elif r[j] > r[i]:
                pairs.append((j, i))
                made += 1
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long)


def ranknet_loss(scores: torch.Tensor, pairs_ij: torch.Tensor) -> torch.Tensor:
    """
    RankNet pairwise loss: -log sigma(s_i - s_j)
    scores: [N]
    pairs_ij: [P, 2]
    """
    if pairs_ij.numel() == 0:
        return torch.zeros([], device=scores.device)
    s_i = scores[pairs_ij[:, 0]]
    s_j = scores[pairs_ij[:, 1]]
    return -F.logsigmoid(s_i - s_j).mean()
