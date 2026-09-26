"""Attention-gated multimodal yield regression model."""

from __future__ import annotations

import torch
from torch import nn

from .satellite_cnn import SatelliteCNN
from .soil_mlp import SoilMLP
from .weather_lstm import WeatherLSTM


class ModalityAttention(nn.Module):
    """Learn normalized sample-specific weights over modality embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, modalities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # modalities: [B, 3, D]
        logits = self.scorer(modalities).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        fused = (modalities * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class MultimodalYieldModel(nn.Module):
    """CNN + masked LSTM + MLP with learnable attention fusion."""

    def __init__(
        self,
        satellite_dim: int = 128,
        weather_dim: int = 128,
        soil_dim: int = 64,
        fusion_dim: int = 128,
    ) -> None:
        super().__init__()
        self.satellite_encoder = SatelliteCNN(satellite_dim)
        self.weather_encoder = WeatherLSTM(embedding_dim=weather_dim)
        self.soil_encoder = SoilMLP(soil_dim)
        self.satellite_projection = nn.Linear(satellite_dim, fusion_dim)
        self.weather_projection = nn.Linear(weather_dim, fusion_dim)
        self.soil_projection = nn.Linear(soil_dim, fusion_dim)
        self.attention = ModalityAttention(fusion_dim)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        soil: torch.Tensor,
        weather_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        satellite_embedding = self.satellite_encoder(satellite)
        weather_embedding = self.weather_encoder(weather, weather_mask)
        soil_embedding = self.soil_encoder(soil)
        modalities = torch.stack(
            (
                self.satellite_projection(satellite_embedding),
                self.weather_projection(weather_embedding),
                self.soil_projection(soil_embedding),
            ),
            dim=1,
        )
        fused, attention_weights = self.attention(modalities)
        prediction = self.regression_head(fused).squeeze(-1)
        if return_details:
            return {
                "satellite_embedding": satellite_embedding,
                "weather_embedding": weather_embedding,
                "soil_embedding": soil_embedding,
                "fused": fused,
                "attention_weights": attention_weights,
                "prediction": prediction,
            }
        return prediction
