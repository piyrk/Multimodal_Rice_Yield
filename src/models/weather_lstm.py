"""Compact masked LSTM encoder for daily weather sequences."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class WeatherLSTM(nn.Module):
    """Encode [B, T, 3] weather data into a fixed embedding."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != 3:
            raise ValueError(f"Expected weather [B,T,3], got {tuple(inputs.shape)}")
        if mask is None:
            lengths = torch.full(
                (inputs.shape[0],), inputs.shape[1], dtype=torch.long, device=inputs.device
            )
        else:
            if mask.shape != inputs.shape[:2]:
                raise ValueError(f"Weather mask must be [B,T], got {tuple(mask.shape)}")
            lengths = mask.to(dtype=torch.long).sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.projection(hidden[-1])
