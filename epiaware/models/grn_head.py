from __future__ import annotations

import torch
from torch import nn

from .backbone import Swish


class GRNPredictorHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, depth: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        in_dim = embed_dim * 2
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    Swish(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, tf_states: torch.Tensor, gene_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf_states: (N, dim)
            gene_states: (N, dim)
        Returns:
            logits: (N,)
        """
        h = torch.cat([tf_states, gene_states], dim=-1)
        features = self.trunk(h)
        return self.out_proj(features).squeeze(-1)


def hinge_loss(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - labels * logits)


def nn_pu_loss(
    pos_logits: torch.Tensor,
    unl_logits: torch.Tensor,
    positive_prior: float,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    pos_labels = torch.ones_like(pos_logits)
    unl_labels = -torch.ones_like(unl_logits)
    positive_risk = positive_prior * hinge_loss(pos_labels, pos_logits).mean()
    negative_risk = hinge_loss(unl_labels, unl_logits).mean() - positive_prior * hinge_loss(
        -pos_labels, pos_logits
    ).mean()
    negative_risk = torch.clamp(negative_risk, min=0.0)
    return positive_risk + negative_weight * negative_risk


__all__ = ["GRNPredictorHead", "nn_pu_loss"]
