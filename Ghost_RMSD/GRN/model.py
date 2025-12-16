# --*-- conding:utf-8 --*--
# @time:11/1/25 03:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:model.py

from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    """Linear -> (BatchNorm) -> ReLU -> (Dropout)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class GRNClassifier(nn.Module):
    """
    GRN with:
      - numeric backbone (MLP) on prepared features
      - classification head for 4-level labels (rel ∈ {0,1,2,3})
      - optional independent ranking head (scalar score for RankNet)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_bn: bool = True,
        score_mode: str = "expected_rel",  # used when rank_head is False
        use_rank_head: bool = True,        # if True, ranking uses an independent linear head
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: List[nn.Module] = []
        d_prev = in_dim
        for d in hidden_dims:
            layers.append(MLPBlock(d_prev, d, dropout=dropout, use_bn=use_bn))
            d_prev = d
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        # classification head
        self.classifier = nn.Linear(d_prev, 4)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # independent ranking head
        self.use_rank_head = use_rank_head
        if self.use_rank_head:
            self.rank_head = nn.Linear(d_prev, 1)
            nn.init.xavier_uniform_(self.rank_head.weight)
            if self.rank_head.bias is not None:
                nn.init.zeros_(self.rank_head.bias)
        else:
            self.rank_head = None

        self.score_mode = score_mode  # used only if rank_head is disabled

    @torch.no_grad()
    def _make_score_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Fallback scoring when rank_head is disabled."""
        if self.score_mode == "prob_rel3":
            probs = F.softmax(logits, dim=-1)
            return probs[..., 3]  # probability of best class
        if self.score_mode == "logit_rel3":
            return logits[..., 3]
        if self.score_mode == "expected_rel":
            probs = F.softmax(logits, dim=-1)
            levels = torch.arange(0, 4, device=logits.device, dtype=probs.dtype)
            return (probs * levels).sum(dim=-1)
        # default fallback
        probs = F.softmax(logits, dim=-1)
        return probs[..., 3]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)

        if self.use_rank_head:
            # independent scalar score for ranking loss
            rank_score = self.rank_head(h).squeeze(-1)
        else:
            # derive a scalar score from logits
            rank_score = self._make_score_from_logits(logits)

        return {"logits": logits, "probs": probs, "score": rank_score}


def build_grn_from_datamodule(
    dm,
    dropout: float = 0.3,
    use_bn: bool = True,
    score_mode: str = "expected_rel",
    use_rank_head: bool = True,
) -> GRNClassifier:
    """Convenience constructor when you already have a GRNDataModule."""
    in_dim = dm.feature_dim()
    return GRNClassifier(
        in_dim=in_dim,
        dropout=dropout,
        use_bn=use_bn,
        score_mode=score_mode,
        use_rank_head=use_rank_head,
    )
