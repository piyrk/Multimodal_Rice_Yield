"""Controlled comparison of attention and concatenation fusion."""

from __future__ import annotations

import copy
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    create_model,
    epoch,
    regression_metrics,
)
from run_repeatability import (
    SEEDS,
    evaluate_test,
    make_datasets,
    make_loaders,
    seed_everything,
)


RESULTS = ROOT / "results"
MODEL_NAMES = ("satellite_only", "attention_fusion", "concatenation_fusion")


class ConcatenationFusionModel(nn.Module):
    """Identical branch encoders followed by concatenation and an MLP."""

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
        satellite_embedding = self.satellite_encoder(satellite)
        weather_embedding = self.weather_encoder(weather, weather_mask)
        soil_embedding = self.soil_encoder(soil)
        combined = torch.cat(
            (satellite_embedding, weather_embedding, soil_embedding), dim=1
        )
        return self.regression_head(self.fusion(combined)).squeeze(-1)


def model_for(name: str) -> nn.Module:
    if name == "satellite_only":
        return create_model("satellite_only")
    if name == "attention_fusion":
        return MultimodalYieldModel()
    if name == "concatenation_fusion":
        return ConcatenationFusionModel()
    raise ValueError(f"Unknown fusion ablation model: {name}")


def train_one(
    name: str,
    seed: int,
    datasets: dict[str, object],
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict[str, float | int | str]:
    seed_everything(seed)
    loaders = make_loaders(datasets, seed)
    model = model_for(name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    stale = 0
    for current_epoch in range(1, DEFAULT_EPOCHS + 1):
        epoch(model, loaders["train"], device, target_mean, target_std, optimizer)
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
    metrics, _ = evaluate_test(
        model, loaders["test"], device, target_mean, target_std
    )
    metrics.update(
        {
            "model": name,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        }
    )
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Fusion ablation requires the configured GPU")
    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = make_datasets()
    try:
        targets = datasets["train"].samples["yield_kg_ha"].to_numpy(dtype=np.float64)
        target_mean = float(targets.mean())
        target_std = float(targets.std(ddof=0))
        rows: list[dict[str, float | int | str]] = []
        for name in MODEL_NAMES:
            for seed in SEEDS:
                result = train_one(
                    name, seed, datasets, device, target_mean, target_std
                )
                rows.append(result)
                print(
                    f"{name} seed={seed} MAE={result['mae_kg_ha']:.3f} "
                    f"RMSE={result['rmse_kg_ha']:.3f} R2={result['r2']:.6f}"
                )
        per_seed = pd.DataFrame(rows)
        summary_rows: list[dict[str, float | int | str]] = []
        for name in MODEL_NAMES:
            frame = per_seed.loc[per_seed["model"] == name]
            summary_rows.append(
                {
                    "result_type": "summary",
                    "model": name,
                    "seed": "",
                    "runs": len(frame),
                    "mean_mae_kg_ha": frame["mae_kg_ha"].mean(),
                    "std_mae_kg_ha": frame["mae_kg_ha"].std(ddof=0),
                    "mean_rmse_kg_ha": frame["rmse_kg_ha"].mean(),
                    "std_rmse_kg_ha": frame["rmse_kg_ha"].std(ddof=0),
                    "mean_r2": frame["r2"].mean(),
                    "std_r2": frame["r2"].std(ddof=0),
                }
            )
        per_seed.insert(0, "result_type", "per_seed")
        output = pd.concat([per_seed, pd.DataFrame(summary_rows)], ignore_index=True)
        output.to_csv(RESULTS / "fusion_ablation_results.csv", index=False)

        summary = pd.DataFrame(summary_rows)
        lines = [
            "# Controlled Fusion Ablation",
            "",
            "This experiment compares the existing Satellite-only baseline, "
            "the unchanged attention fusion model, and a new concatenation "
            "fusion model.",
            "",
            "## Controlled protocol",
            "",
            "- Exact chronological split: 172 train, 26 validation, 41 test",
            "- Seeds: 42, 123, 2026",
            "- Train-only target normalization",
            "- AdamW, learning rate 0.001, MSELoss",
            "- Batch size 16, maximum 8 epochs, early-stopping patience 3",
            f"- Device: `{torch.cuda.get_device_name(0)}`",
            "",
            "The satellite CNN, weather LSTM, and soil MLP branch architectures "
            "are unchanged. The concatenation variant concatenates the 128-, "
            "128-, and 64-dimensional branch embeddings (320 total), then "
            "applies an MLP fusion block and regression head. It has no "
            "attention or learned modality weighting.",
            "",
            "## Per-seed results",
            "",
            "| Model | Seed | Best epoch | Validation loss | MAE | RMSE | R2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in per_seed.to_dict("records"):
            lines.append(
                f"| {row['model']} | {row['seed']} | {row['best_epoch']} | "
                f"{row['best_validation_loss']:.6f} | {row['mae_kg_ha']:.3f} | "
                f"{row['rmse_kg_ha']:.3f} | {row['r2']:.6f} |"
            )
        lines += [
            "",
            "## Three-seed mean +/- standard deviation",
            "",
            "| Model | MAE (kg/ha) | RMSE (kg/ha) | R2 |",
            "|---|---:|---:|---:|",
        ]
        for row in summary.to_dict("records"):
            lines.append(
                f"| {row['model']} | {row['mean_mae_kg_ha']:.3f} +/- "
                f"{row['std_mae_kg_ha']:.3f} | {row['mean_rmse_kg_ha']:.3f} +/- "
                f"{row['std_rmse_kg_ha']:.3f} | {row['mean_r2']:.6f} +/- "
                f"{row['std_r2']:.6f} |"
            )
        concat = summary.loc[summary["model"] == "concatenation_fusion"].iloc[0]
        attention = summary.loc[summary["model"] == "attention_fusion"].iloc[0]
        improves = (
            concat["mean_mae_kg_ha"] < attention["mean_mae_kg_ha"]
            and concat["mean_rmse_kg_ha"] < attention["mean_rmse_kg_ha"]
            and concat["mean_r2"] > attention["mean_r2"]
        )
        lines += [
            "",
            "## Conclusion",
            "",
            f"Under this three-seed controlled experiment, concatenation fusion "
            f"{'improves' if improves else 'does not improve'} the full "
            "attention architecture across all three reported mean metrics. "
            "This conclusion is limited to the fixed split and configuration; "
            "no general hyperparameter tuning was performed.",
        ]
        (RESULTS / "fusion_ablation_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    finally:
        for split in ("train", "validation", "test"):
            path = PROCESSED / f".repeatability_{split}.csv"
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
