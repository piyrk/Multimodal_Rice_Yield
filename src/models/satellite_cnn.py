"""Compact convolutional encoder for seven-band satellite patches."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SatelliteCNN(nn.Module):
    """Encode [B, 7, 128, 128] into a fixed [B, embedding_dim] vector."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(7, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1:] != (7, 128, 128):
            raise ValueError(f"Expected satellite [B,7,128,128], got {tuple(inputs.shape)}")
        return self.projection(self.pool(self.features(inputs)))
