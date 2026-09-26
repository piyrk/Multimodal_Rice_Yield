"""Unified reproducibility runner for the controlled fusion comparison."""

from __future__ import annotations

import copy
import gc
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import RiceMultimodalDataset, multimodal_collate
from models.fusion_model import MultimodalYieldModel
from models.satellite_cnn import SatelliteCNN
from models.soil_mlp import SoilMLP
from models.weather_lstm import WeatherLSTM
from run_baselines import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    PROCESSED,
    ROOT,
    epoch,
    regression_metrics,
)


RESULTS = ROOT / "results"
SEEDS = (42, 123, 2026)
MODEL_NAMES = ("satellite_only", "attention_fusion", "concatenation_fusion")


class ConcatenationFusionModel(nn.Module):
    """The controlled non-attention fusion variant."""

    def __init__(self) -> None:
        super().__init__()
        self.satellite_encoder = SatelliteCNN(128)
        self.weather_encoder = WeatherLSTM(embedding_dim=128)
        self.soil_encoder = SoilMLP(64)
        self.fusion = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets() -> dict[str, RiceMultimodalDataset]:
    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    datasets: dict[str, RiceMultimodalDataset] = {}
    for split in ("train", "validation", "test"):
        path = PROCESSED / f".unified_{split}.csv"
        subset = splits.loc[splits["split"] == split].drop(columns=["split"])
        subset.to_csv(path, index=False)
        datasets[split] = RiceMultimodalDataset(path)
    return datasets


def cleanup() -> None:
    for split in ("train", "validation", "test"):
        path = PROCESSED / f".unified_{split}.csv"
        if path.exists():
            path.unlink()


def make_loaders(
    datasets: dict[str, RiceMultimodalDataset], seed: int
) -> dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=multimodal_collate,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate,
        ),
    }


def create_model(name: str) -> nn.Module:
    if name == "satellite_only":
        return SatelliteOnlyModel()
    if name == "attention_fusion":
        return MultimodalYieldModel()
    if name == "concatenation_fusion":
        return ConcatenationFusionModel()
    raise ValueError(f"Unknown model: {name}")


class SatelliteOnlyModel(nn.Module):
    """Use the existing satellite encoder with the existing ablation head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SatelliteCNN(128)
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        soil: torch.Tensor,
        weather_mask: torch.Tensor,
    ) -> torch.Tensor:
        del weather, soil, weather_mask
        return self.head(self.encoder(satellite)).squeeze(-1)


def forward_model(
    model: nn.Module, batch: dict[str, object], device: torch.device
) -> torch.Tensor:
    inputs = {
        "satellite": batch["satellite"].to(device),
        "weather": batch["weather"].to(device),
        "soil": batch["soil"].to(device),
        "weather_mask": batch["weather_mask"].to(device),
    }
    if isinstance(model, MultimodalYieldModel):
        return model(**inputs)
    return model(**inputs)


def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, float], list[str]]:
    model.eval()
    actual: list[float] = []
    predicted: list[float] = []
    test_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            normalized = forward_model(model, batch, device)
            prediction = normalized.cpu() * target_std + target_mean
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Test predictions contain NaN/Inf")
            actual.extend(batch["target"].tolist())
            predicted.extend(prediction.tolist())
            test_ids.extend(batch["sample_id"])
    metrics = regression_metrics(
        np.asarray(actual, dtype=np.float64),
        np.asarray(predicted, dtype=np.float64),
    )
    return metrics, test_ids


def run_one(
    model_name: str,
    seed: int,
    datasets: dict[str, RiceMultimodalDataset],
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict[str, float | int | str]:
    seed_everything(seed)
    loaders = make_loaders(datasets, seed)
    model = create_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    stale = 0
    for current_epoch in range(1, DEFAULT_EPOCHS + 1):
        model.train()
        epoch(
            model, loaders["train"], device, target_mean, target_std, optimizer
        )
        with torch.no_grad():
            validation = epoch(
                model, loaders["validation"], device, target_mean, target_std, None
            )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_epoch = current_epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= DEFAULT_PATIENCE:
                break
    model.load_state_dict(best_state)
    metrics, test_ids = evaluate_test(
        model, loaders["test"], device, target_mean, target_std
    )
    ids_digest = hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()
    result = {
        "seed": seed,
        "model": model_name,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_sample_count": len(test_ids),
        "test_ids_sha256": ids_digest,
        **metrics,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_report(frame: pd.DataFrame, device_name: str) -> None:
    lines = [
        "# Unified Fusion Reproducibility Rerun",
        "",
        "All nine runs use one implementation path for model construction, "
        "seeding, DataLoader creation, training, validation checkpoint "
        "selection, and test evaluation.",
        "",
        f"Device: `{device_name}`",
        f"Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        "Split: exact existing chronological split, 172 train / 26 validation / 41 test",
        "Training: AdamW, MSELoss, batch size 16, learning rate 0.001, "
        "maximum 8 epochs, patience 3",
        "Normalization: target mean and standard deviation from train samples only",
        "",
        "## Per-run results",
        "",
        "| Model | Seed | Best epoch | Best validation loss | Test count | MAE | RMSE | R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['best_epoch']} | "
            f"{row['best_validation_loss']:.6f} | {row['test_sample_count']} | "
            f"{row['mae_kg_ha']:.3f} | {row['rmse_kg_ha']:.3f} | {row['r2']:.6f} |"
        )
    lines += [
        "",
        "## Verification",
        "",
        f"- Exactly nine runs: `{len(frame) == 9}`",
        f"- Exactly three models and three seeds: "
        f"`{set(frame['model']) == set(MODEL_NAMES) and set(frame['seed']) == set(SEEDS)}`",
        f"- Every run has 41 test predictions: `{(frame['test_sample_count'] == 41).all()}`",
        f"- All metrics are finite: "
        f"`{np.isfinite(frame[['mae_kg_ha', 'rmse_kg_ha', 'r2', 'best_validation_loss']].to_numpy()).all()}`",
        f"- Same test IDs in every run: `{frame['test_ids_sha256'].nunique() == 1}`",
        "",
        "## Cross-run audit conclusion",
        "",
        "The earlier discrepancy is not resolved by comparing the old outputs "
        "alone because those runners duplicated execution logic. This unified "
        "rerun is the controlled comparison to use going forward. Its results "
        "should be compared by the per-run rows above, not mixed with the "
        "previous runner outputs.",
    ]
    (RESULTS / "final_fusion_ablation_reproducibility.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Unified rerun requires the RTX 3050 CUDA configuration")
    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    datasets = make_datasets()
    try:
        targets = datasets["train"].samples["yield_kg_ha"].to_numpy(dtype=np.float64)
        target_mean = float(targets.mean())
        target_std = float(targets.std(ddof=0))
        if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid train-only target normalization statistics")
        rows: list[dict[str, float | int | str]] = []
        for model_name in MODEL_NAMES:
            for seed in SEEDS:
                result = run_one(
                    model_name, seed, datasets, device, target_mean, target_std
                )
                rows.append(result)
                print(
                    f"{model_name} seed={seed} MAE={result['mae_kg_ha']:.3f} "
                    f"RMSE={result['rmse_kg_ha']:.3f} R2={result['r2']:.6f}"
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(
            RESULTS / "final_fusion_ablation_reproducibility.csv", index=False
        )
        if len(frame) != 9:
            raise ValueError(f"Expected 9 runs, found {len(frame)}")
        if (frame["test_sample_count"] != 41).any():
            raise ValueError("At least one run did not evaluate 41 test samples")
        if frame["test_ids_sha256"].nunique() != 1:
            raise ValueError("Test IDs differ between unified runs")
        metric_columns = [
            "best_validation_loss",
            "mae_kg_ha",
            "rmse_kg_ha",
            "r2",
        ]
        if not np.isfinite(frame[metric_columns].to_numpy()).all():
            raise ValueError("Non-finite unified metrics")
        write_report(frame, torch.cuda.get_device_name(0))
    finally:
        cleanup()


if __name__ == "__main__":
    main()
