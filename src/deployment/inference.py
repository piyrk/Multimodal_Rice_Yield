"""Input validation and inference helpers for the locked final model."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.satellite_cnn import SatelliteCNN
from models.soil_mlp import SoilMLP
from models.weather_lstm import WeatherLSTM


EXPECTED_WEATHER_COLUMNS = ["rainfall_mm", "temperature_c", "humidity_pct"]
EXPECTED_SATELLITE_SHAPE = (7, 128, 128)
SEQUENCE_LENGTH = 183


class ConcatenationFusionInference(nn.Module):
    """The locked concatenation-fusion architecture for inference only."""

    def __init__(self) -> None:
        super().__init__()
        self.satellite_encoder = SatelliteCNN(128)
        self.weather_encoder = WeatherLSTM(embedding_dim=128)
        self.soil_encoder = SoilMLP(64)
        self.fusion = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        soil: torch.Tensor,
        weather_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = (
            self.satellite_encoder(satellite),
            self.weather_encoder(weather, weather_mask),
            self.soil_encoder(soil),
        )
        return self.regression_head(self.fusion(torch.cat(embeddings, dim=1))).squeeze(-1)


def validate_satellite(uploaded_file: object) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Upload a satellite .npy file before predicting.")
    name = str(getattr(uploaded_file, "name", "")).lower()
    if not name.endswith(".npy"):
        raise ValueError("Satellite input must be a .npy file.")
    try:
        array = np.load(io.BytesIO(uploaded_file.getvalue()), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not read satellite .npy file: {exc}") from exc
    if array.shape == (128, 128, 7):
        array = np.transpose(array, (2, 0, 1))
    if array.shape != EXPECTED_SATELLITE_SHAPE:
        raise ValueError(
            "Satellite shape must be [7,128,128] or [128,128,7]; "
            f"received {tuple(array.shape)}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Satellite input contains NaN or Inf values.")
    return np.ascontiguousarray(array, dtype=np.float32)


def validate_weather(uploaded_file: object) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Upload a weather CSV file before predicting.")
    name = str(getattr(uploaded_file, "name", "")).lower()
    if not name.endswith(".csv"):
        raise ValueError("Weather input must be a .csv file.")
    try:
        frame = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        raise ValueError(f"Could not read weather CSV: {exc}") from exc
    missing = [column for column in EXPECTED_WEATHER_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            "Weather CSV must contain columns in this feature order: "
            "rainfall_mm, temperature_c, humidity_pct. "
            f"Missing: {', '.join(missing)}."
        )
    values = frame[EXPECTED_WEATHER_COLUMNS]
    if len(values) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Weather CSV must contain exactly {SEQUENCE_LENGTH} observations; "
            f"received {len(values)}."
        )
    array = values.to_numpy(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Weather input contains NaN or Inf values.")
    return np.ascontiguousarray(array, dtype=np.float32)


def validate_soil(nitrogen: float, ph: float, soc: float, clay: float) -> np.ndarray:
    values = np.asarray([nitrogen, ph, soc, clay], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("All soil values must be finite numbers.")
    if nitrogen < 0 or ph <= 0 or soc < 0 or clay < 0 or clay > 100:
        raise ValueError(
            "Invalid soil values. Nitrogen, SOC, and clay must be non-negative; "
            "pH must be positive; clay must not exceed 100%."
        )
    return values


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConcatenationFusionInference().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict(
    model: torch.nn.Module,
    satellite: np.ndarray,
    weather: np.ndarray,
    soil: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
) -> float:
    satellite_tensor = torch.from_numpy(satellite).unsqueeze(0).to(device)
    weather_tensor = torch.from_numpy(weather).unsqueeze(0).to(device)
    soil_tensor = torch.from_numpy(soil).unsqueeze(0).to(device)
    mask = torch.ones((1, weather.shape[0]), dtype=torch.bool, device=device)
    with torch.no_grad():
        normalized = model(
            satellite=satellite_tensor,
            weather=weather_tensor,
            soil=soil_tensor,
            weather_mask=mask,
        )
        prediction = normalized * float(target_std) + float(target_mean)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Model returned a NaN or Inf prediction.")
    return float(prediction.squeeze().cpu())
