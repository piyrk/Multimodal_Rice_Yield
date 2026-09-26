"""PyTorch dataset and collation utilities for the validated multimodal data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REQUIRED_WEATHER_COLUMNS = ("date", "rainfall_mm", "temperature_c", "humidity_pct")


class RiceMultimodalDataset(Dataset):
    """Load one validated satellite, weather, soil, and yield sample."""

    def __init__(
        self,
        samples_csv: str | Path = PROCESSED / "final_training_samples.csv",
        manifest_csv: str | Path = PROCESSED / "multimodal_manifest.csv",
        soil_csv: str | Path = PROCESSED / "soil_features.csv",
        master_csv: str | Path = PROCESSED / "master_samples.csv",
        validate: bool = True,
    ) -> None:
        self.samples = pd.read_csv(samples_csv)
        manifest = pd.read_csv(manifest_csv)
        soil = pd.read_csv(soil_csv)
        master = pd.read_csv(master_csv)
        if self.samples["sample_id"].duplicated().any():
            raise ValueError("samples CSV contains duplicate sample_id values")
        if manifest["sample_id"].duplicated().any():
            raise ValueError("multimodal manifest contains duplicate sample_id values")
        if not set(self.samples["sample_id"]).issubset(manifest["sample_id"]):
            raise ValueError("samples CSV contains IDs absent from multimodal manifest")
        self.manifest = manifest.set_index("sample_id")
        self.soil = soil.set_index("district")
        self.masked_satellite_samples: set[str] = set()
        date_columns = ["sample_id", "weather_start", "weather_end"]
        self.dates = master[date_columns].set_index("sample_id")
        missing_dates = pd.Index(self.samples["sample_id"]).difference(self.dates.index)
        if len(missing_dates):
            raise ValueError(f"Missing weather dates for sample IDs: {missing_dates.tolist()}")
        if validate:
            for index in range(len(self.samples)):
                self._load(index, validate=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_satellite(self, path: Path, sample_id: str) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Satellite file not found: {path}")
        with rasterio.open(path) as source:
            array = source.read()
            dtypes = set(source.dtypes)
        if dtypes != {"float32"}:
            raise ValueError(f"Satellite dtype must be float32, found {dtypes}: {path}")
        if array.shape == (128, 128, 7):
            array = np.transpose(array, (2, 0, 1))
        if array.shape != (7, 128, 128):
            raise ValueError(f"Satellite shape must be [7,128,128], found {array.shape}: {path}")
        if not np.isfinite(array).all():
            # NaN pixels are nodata produced by QA cloud masking. Keep raw files
            # unchanged and represent those masked pixels as zero in memory.
            self.masked_satellite_samples.add(sample_id)
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float32)

    def _load_weather(self, path: Path, sample_id: str) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Weather file not found: {path}")
        frame = pd.read_csv(path)
        if not set(REQUIRED_WEATHER_COLUMNS).issubset(frame.columns):
            raise ValueError(f"Weather columns missing for {sample_id}")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        expected_start = pd.Timestamp(self.dates.loc[sample_id, "weather_start"])
        expected_end = pd.Timestamp(self.dates.loc[sample_id, "weather_end"])
        expected = pd.date_range(expected_start, expected_end, freq="D")
        actual = dates.dt.normalize()
        if (
            dates.isna().any()
            or len(actual) != len(expected)
            or not actual.is_monotonic_increasing
            or actual.duplicated().any()
            or not actual.reset_index(drop=True).equals(pd.Series(expected, name="date"))
        ):
            raise ValueError(f"Weather dates are incomplete or unordered for {sample_id}")
        values = frame[["rainfall_mm", "temperature_c", "humidity_pct"]].to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Weather contains NaN/Inf for {sample_id}")
        return torch.from_numpy(values)

    def _load_soil(self, district: str) -> torch.Tensor:
        if district not in self.soil.index:
            raise ValueError(f"Soil record missing for district: {district}")
        values = self.soil.loc[district, ["nitrogen_g_kg", "ph", "soc_g_kg", "clay_pct"]]
        array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
        if array.shape != (4,) or not np.isfinite(array).all():
            raise ValueError(f"Soil values invalid for district: {district}")
        return torch.from_numpy(array)

    def _load(self, index: int, validate: bool = True) -> dict[str, Any]:
        row = self.samples.iloc[index]
        sample_id = str(row["sample_id"])
        metadata = self.manifest.loc[sample_id]
        satellite = self._load_satellite(
            ROOT / Path(str(metadata["satellite_path"])), sample_id
        )
        weather = self._load_weather(ROOT / Path(str(metadata["weather_path"])), sample_id)
        soil = self._load_soil(str(row["district"]))
        target = pd.to_numeric(pd.Series([row["yield_kg_ha"]]), errors="coerce").iloc[0]
        if not np.isfinite(target):
            raise ValueError(f"Yield target is missing or invalid for {sample_id}")
        item = {
            "sample_id": sample_id,
            "satellite": satellite,
            "weather": weather,
            "soil": soil,
            "target": torch.tensor(float(target), dtype=torch.float32),
        }
        if validate and any(
            not torch.isfinite(value).all()
            for key, value in item.items()
            if key != "sample_id"
        ):
            raise ValueError(f"NaN/Inf detected in sample {sample_id}")
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._load(index, validate=True)


def multimodal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length daily weather sequences by right-padding with zeros."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(item["weather"].shape[0] for item in batch)
    weather = torch.zeros(
        (len(batch), max_length, 3), dtype=torch.float32
    )
    weather_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    for index, item in enumerate(batch):
        length = item["weather"].shape[0]
        weather[index, :length] = item["weather"]
        weather_mask[index, :length] = True
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "satellite": torch.stack([item["satellite"] for item in batch]),
        "weather": weather,
        "weather_mask": weather_mask,
        "soil": torch.stack([item["soil"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }
