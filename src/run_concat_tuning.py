"""Validation-only hyperparameter tuning for concatenation fusion."""

from __future__ import annotations

import copy
import gc
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import RiceMultimodalDataset, multimodal_collate
from models.satellite_cnn import SatelliteCNN
from models.soil_mlp import SoilMLP
from models.weather_lstm import WeatherLSTM
from run_baselines import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PATIENCE,
    PROCESSED,
    ROOT,
    epoch,
)


RESULTS = ROOT / "results"
SEEDS = (42, 123, 2026)
LEARNING_RATES = (0.001, 0.0005)
FUSION_DROPOUTS = (0.0, 0.1)
MAX_EPOCHS = 12


class TunableConcatenationFusion(nn.Module):
    """Existing concatenation architecture with only fusion dropout exposed."""

    def __init__(self, fusion_dropout: float) -> None:
        super().__init__()
        self.satellite_encoder = SatelliteCNN(128)
        self.weather_encoder = WeatherLSTM(embedding_dim=128)
        self.soil_encoder = SoilMLP(64)
        self.fusion = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
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
        combined = torch.cat(embeddings, dim=1)
        return self.regression_head(self.fusion(combined)).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_validation_datasets() -> dict[str, RiceMultimodalDataset]:
    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    datasets: dict[str, RiceMultimodalDataset] = {}
    for split in ("train", "validation"):
        path = PROCESSED / f".tuning_{split}.csv"
        splits.loc[splits["split"] == split].drop(columns=["split"]).to_csv(
            path, index=False
        )
        datasets[split] = RiceMultimodalDataset(path)
    return datasets


def cleanup() -> None:
    for split in ("train", "validation"):
        path = PROCESSED / f".tuning_{split}.csv"
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
    }


def model_output(
    model: nn.Module, batch: dict[str, object], device: torch.device
) -> torch.Tensor:
    return model(
        satellite=batch["satellite"].to(device),
        weather=batch["weather"].to(device),
        soil=batch["soil"].to(device),
        weather_mask=batch["weather_mask"].to(device),
    )


def validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    actual: list[float] = []
    predicted: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device)
            normalized_prediction = model_output(model, batch, device)
            normalized_target = (target - target_mean) / target_std
            loss = nn.functional.mse_loss(normalized_prediction, normalized_target)
            if not torch.isfinite(loss):
                raise FloatingPointError("NaN/Inf validation loss encountered")
            prediction = normalized_prediction * target_std + target_mean
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("NaN/Inf validation prediction encountered")
            losses.append(float(loss.cpu()))
            actual.extend(target.cpu().tolist())
            predicted.extend(prediction.cpu().tolist())
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    error = predicted_array - actual_array
    total = np.sum((actual_array - actual_array.mean()) ** 2)
    return {
        "validation_loss": float(np.mean(losses)),
        "validation_mae_kg_ha": float(np.mean(np.abs(error))),
        "validation_rmse_kg_ha": float(np.sqrt(np.mean(error**2))),
        "validation_r2": float(1.0 - np.sum(error**2) / total) if total else 0.0,
    }


def run_one(
    learning_rate: float,
    fusion_dropout: float,
    seed: int,
    datasets: dict[str, RiceMultimodalDataset],
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict[str, float | int | str]:
    seed_everything(seed)
    loaders = make_loaders(datasets, seed)
    model = TunableConcatenationFusion(fusion_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    for current_epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_result = epoch(
            model,
            loaders["train"],
            device,
            target_mean,
            target_std,
            optimizer,
        )
        validation = validation_epoch(
            model, loaders["validation"], device, target_mean, target_std
        )
        print(
            f"lr={learning_rate} dropout={fusion_dropout} seed={seed} "
            f"epoch={current_epoch}/{MAX_EPOCHS} "
            f"train_loss={train_result['loss']:.4f} "
            f"validation_loss={validation['validation_loss']:.4f}"
        )
        if validation["validation_loss"] < best_validation_loss:
            best_validation_loss = validation["validation_loss"]
            best_metrics = validation
            best_epoch = current_epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= DEFAULT_PATIENCE:
                break
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "learning_rate": learning_rate,
        "fusion_dropout": fusion_dropout,
        "configuration": f"lr_{learning_rate:g}_dropout_{fusion_dropout:g}",
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_mae_kg_ha": best_metrics["validation_mae_kg_ha"],
        "validation_rmse_kg_ha": best_metrics["validation_rmse_kg_ha"],
        "validation_r2": best_metrics["validation_r2"],
        "training_time_seconds": elapsed,
        "device": str(device),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    datasets = make_train_validation_datasets()
    try:
        train_targets = datasets["train"].samples["yield_kg_ha"].to_numpy(
            dtype=np.float64
        )
        target_mean = float(train_targets.mean())
        target_std = float(train_targets.std(ddof=0))
        if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid train-only target normalization statistics")
        rows: list[dict[str, float | int | str]] = []
        for learning_rate in LEARNING_RATES:
            for fusion_dropout in FUSION_DROPOUTS:
                for seed in SEEDS:
                    rows.append(
                        run_one(
                            learning_rate,
                            fusion_dropout,
                            seed,
                            datasets,
                            device,
                            target_mean,
                            target_std,
                        )
                    )
        results = pd.DataFrame(rows)
        results.to_csv(RESULTS / "tuning_results.csv", index=False)
        summary = (
            results.groupby(
                ["configuration", "learning_rate", "fusion_dropout"], as_index=False
            )
            .agg(
                runs=("seed", "count"),
                mean_validation_loss=("best_validation_loss", "mean"),
                std_validation_loss=("best_validation_loss", "std"),
                mean_validation_mae_kg_ha=("validation_mae_kg_ha", "mean"),
                std_validation_mae_kg_ha=("validation_mae_kg_ha", "std"),
                mean_validation_rmse_kg_ha=("validation_rmse_kg_ha", "mean"),
                std_validation_rmse_kg_ha=("validation_rmse_kg_ha", "std"),
                mean_validation_r2=("validation_r2", "mean"),
                std_validation_r2=("validation_r2", "std"),
            )
            .sort_values(["mean_validation_loss", "mean_validation_mae_kg_ha"])
        )
        summary.to_csv(RESULTS / "tuning_summary.csv", index=False)
        selected = summary.iloc[0]
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.bar(summary["configuration"], summary["mean_validation_mae_kg_ha"])
        axis.set_ylabel("Mean validation MAE (kg/ha)")
        axis.set_xlabel("Configuration")
        axis.set_title("Concatenation-fusion validation MAE")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(RESULTS / "tuning_validation_mae.png", dpi=160)
        plt.close(figure)
        lines = [
            "# Concatenation Fusion Hyperparameter Tuning",
            "",
            "Validation-only tuning was performed for exactly four configurations "
            "and three seeds per configuration. The test loader was never "
            "created or evaluated.",
            "",
            f"Device: `{device}`",
            "Split: 172 train / 26 validation / 41 held-out test (test unused)",
            "Seeds: 42, 123, 2026",
            "Optimizer/loss: AdamW / MSELoss",
            "Batch size: 16",
            "Maximum epochs: 12",
            "Early-stopping patience: 3",
            "Target normalization: train targets only",
            "",
            "## Per-run validation results",
            "",
            "| Configuration | Seed | Best epoch | Best validation loss | Validation MAE | Validation RMSE | Validation R2 | Time (s) | Device |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in results.to_dict("records"):
            lines.append(
                f"| {row['configuration']} | {row['seed']} | {row['best_epoch']} | "
                f"{row['best_validation_loss']:.6f} | {row['validation_mae_kg_ha']:.3f} | "
                f"{row['validation_rmse_kg_ha']:.3f} | {row['validation_r2']:.6f} | "
                f"{row['training_time_seconds']:.1f} | {row['device']} |"
            )
        lines += [
            "",
            "## Aggregate validation results",
            "",
            "| Configuration | Mean loss +/- std | Mean MAE +/- std | Mean RMSE +/- std | Mean R2 +/- std |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in summary.to_dict("records"):
            lines.append(
                f"| {row['configuration']} | {row['mean_validation_loss']:.6f} +/- "
                f"{row['std_validation_loss']:.6f} | {row['mean_validation_mae_kg_ha']:.3f} +/- "
                f"{row['std_validation_mae_kg_ha']:.3f} | "
                f"{row['mean_validation_rmse_kg_ha']:.3f} +/- "
                f"{row['std_validation_rmse_kg_ha']:.3f} | "
                f"{row['mean_validation_r2']:.6f} +/- {row['std_validation_r2']:.6f} |"
            )
        lines += [
            "",
            "## Selection",
            "",
            f"Selected configuration: `{selected['configuration']}` "
            f"(learning rate {selected['learning_rate']:g}, fusion dropout "
            f"{selected['fusion_dropout']:g}).",
            f"It was selected because it had the lowest mean validation loss "
            f"({selected['mean_validation_loss']:.6f}) across the three seeds. "
            "Mean validation MAE is reported as a secondary descriptive metric; "
            "it was not the primary selection criterion.",
            "",
            "No test metrics were calculated, inspected, or used for selection. "
            "Final retraining and held-out test evaluation were intentionally "
            "not performed.",
        ]
        (RESULTS / "tuning_report.md").write_text("\n".join(lines), encoding="utf-8")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
