"""Small feed-forward encoder for four soil features."""

from __future__ import annotations

import torch
from torch import nn


class SoilMLP(nn.Module):
    """Encode [B, 4] soil features into a compact embedding."""

    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 4:
            raise ValueError(f"Expected soil [B,4], got {tuple(inputs.shape)}")
        return self.network(inputs)
