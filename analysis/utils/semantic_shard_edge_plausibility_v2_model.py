"""
Small MLP edge plausibility scorer (Method 1 V2). Not a GNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgePlausibilityMLP(nn.Module):
    """Two hidden layers, sigmoid output in [0, 1]."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        hidden_dim2: int = 32,
        activation: str = "gelu",
    ):
        super().__init__()
        act: nn.Module
        if activation.lower() == "relu":
            act = nn.ReLU()
        else:
            act = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim2),
            act,
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
