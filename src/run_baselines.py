"""Run lightweight modality-ablation experiments on the fixed data split."""

from __future__ import annotations

import copy
import platform
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
from models.fusion_model import MultimodalYieldModel
from models.satellite_cnn import SatelliteCNN
from models.soil_mlp import SoilMLP
from models.weather_lstm import WeatherLSTM


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "baseline_predictions"
SEED = 20260926
DEFAULT_EPOCHS = 8
DEFAULT_PATIENCE = 3
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3


def make_dataset(split: str) -> RiceMultimodalDataset:
    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    subset = splits.loc[splits["split"] == split].drop(columns=["split"])
    path = PROCESSED / f".baseline_{split}.csv"
    subset.to_csv(path, index=False)
    return RiceMultimodalDataset(path)


def cleanup() -> None:
    for split in ("train", "validation", "test"):
        path = PROCESSED / f".baseline_{split}.csv"
        if path.exists():
            path.unlink()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    total = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae_kg_ha": float(np.mean(np.abs(error))),
        "rmse_kg_ha": float(np.sqrt(np.mean(error**2))),
        "r2": float(1 - np.sum(error**2) / total) if total else 0.0,
    }


class AblationModel(nn.Module):
    """Use selected existing encoders and a compact concatenation head."""

    def __init__(self, modalities: tuple[str, ...]) -> None:
        super().__init__()
        self.modalities = modalities
        self.satellite = SatelliteCNN(128) if "satellite" in modalities else None
        self.weather = WeatherLSTM(embedding_dim=128) if "weather" in modalities else None
        self.soil = SoilMLP(64) if "soil" in modalities else None
        dimensions = {
            "satellite": 128,
            "weather": 128,
            "soil": 64,
        }
        total = sum(dimensions[name] for name in modalities)
        hidden = max(32, min(128, total // 2))
        self.head = nn.Sequential(
            nn.LayerNorm(total),
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        satellite: torch.Tensor,
        weather: torch.Tensor,
        soil: torch.Tensor,
        weather_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = []
        if self.satellite is not None:
            embeddings.append(self.satellite(satellite))
        if self.weather is not None:
            embeddings.append(self.weather(weather, weather_mask))
        if self.soil is not None:
            embeddings.append(self.soil(soil))
        return self.head(torch.cat(embeddings, dim=1)).squeeze(-1)


def create_model(name: str) -> nn.Module:
    if name == "satellite_weather_soil_attention":
        return MultimodalYieldModel()
    definitions = {
        "weather_only": ("weather",),
        "satellite_only": ("satellite",),
        "soil_only": ("soil",),
        "satellite_weather": ("satellite", "weather"),
        "weather_soil": ("weather", "soil"),
        "satellite_soil": ("satellite", "soil"),
    }
    return AblationModel(definitions[name])


def model_output(
    model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    inputs = {
        "satellite": batch["satellite"].to(device),
        "weather": batch["weather"].to(device),
        "soil": batch["soil"].to(device),
        "weather_mask": batch["weather_mask"].to(device),
    }
    if isinstance(model, MultimodalYieldModel):
        details = model(**inputs, return_details=True)
        return details["prediction"], details["attention_weights"]
    return model(**inputs), None


def epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.MSELoss()
    losses: list[float] = []
    actual_values: list[float] = []
    predicted_values: list[float] = []
    for batch in loader:
        target = batch["target"].to(device)
        normalized_target = (target - target_mean) / target_std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            normalized_prediction, _ = model_output(model, batch, device)
            loss = criterion(normalized_prediction, normalized_target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN/Inf loss encountered")
        prediction = normalized_prediction.detach() * target_std + target_mean
        losses.append(float(loss.detach().cpu()))
        actual_values.extend(target.detach().cpu().tolist())
        predicted_values.extend(prediction.cpu().tolist())
    result = regression_metrics(np.asarray(actual_values), np.asarray(predicted_values))
    result["loss"] = float(np.mean(losses))
    return result


def test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for batch in loader:
            normalized_prediction, attention = model_output(model, batch, device)
            prediction = normalized_prediction.cpu() * target_std + target_mean
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Test predictions contain NaN/Inf")
            for index, sample_id in enumerate(batch["sample_id"]):
                row: dict[str, float | str] = {
                    "sample_id": sample_id,
                    "actual_yield_kg_ha": float(batch["target"][index]),
                    "predicted_yield_kg_ha": float(prediction[index]),
                }
                if attention is not None:
                    weights = attention.cpu()[index]
                    row.update(
                        {
                            "satellite_attention": float(weights[0]),
                            "weather_attention": float(weights[1]),
                            "soil_attention": float(weights[2]),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: unavailable; running lightweight baseline suite on CPU")

    RESULTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    try:
        datasets = {split: make_dataset(split) for split in ("train", "validation", "test")}
        loaders = {
            "train": DataLoader(
                datasets["train"], batch_size=DEFAULT_BATCH_SIZE, shuffle=True,
                num_workers=0, collate_fn=multimodal_collate,
            ),
            "validation": DataLoader(
                datasets["validation"], batch_size=DEFAULT_BATCH_SIZE, shuffle=False,
                num_workers=0, collate_fn=multimodal_collate,
            ),
            "test": DataLoader(
                datasets["test"], batch_size=DEFAULT_BATCH_SIZE, shuffle=False,
                num_workers=0, collate_fn=multimodal_collate,
            ),
        }
        targets = datasets["train"].samples["yield_kg_ha"].to_numpy(dtype=np.float64)
        target_mean = float(targets.mean())
        target_std = float(targets.std(ddof=0))
        if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid train-only target normalization statistics")

        names = (
            "weather_only",
            "satellite_only",
            "soil_only",
            "satellite_weather",
            "weather_soil",
            "satellite_soil",
            "satellite_weather_soil_attention",
        )
        comparison: list[dict[str, float | int | str]] = []
        for index, name in enumerate(names):
            torch.manual_seed(SEED + index)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model = create_model(name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
            best_state = copy.deepcopy(model.state_dict())
            best_validation_loss = float("inf")
            best_epoch = 0
            stale = 0
            start = time.perf_counter()
            for current_epoch in range(1, DEFAULT_EPOCHS + 1):
                train_result = epoch(
                    model, loaders["train"], device, target_mean, target_std, optimizer
                )
                with torch.no_grad():
                    validation_result = epoch(
                        model, loaders["validation"], device, target_mean, target_std, None
                    )
                print(
                    f"{name} epoch {current_epoch}/{DEFAULT_EPOCHS} "
                    f"train_loss={train_result['loss']:.4f} "
                    f"validation_loss={validation_result['loss']:.4f}"
                )
                if validation_result["loss"] < best_validation_loss:
                    best_validation_loss = validation_result["loss"]
                    best_epoch = current_epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= DEFAULT_PATIENCE:
                        break
            model.load_state_dict(best_state)
            elapsed = time.perf_counter() - start
            predictions = test_predictions(
                model, loaders["test"], device, target_mean, target_std
            )
            prediction_path = PREDICTIONS / f"{name}.csv"
            predictions.to_csv(prediction_path, index=False)
            actual = predictions["actual_yield_kg_ha"].to_numpy()
            predicted = predictions["predicted_yield_kg_ha"].to_numpy()
            result = regression_metrics(actual, predicted)
            comparison.append(
                {
                    "model": name,
                    "train_samples": len(datasets["train"]),
                    "validation_samples": len(datasets["validation"]),
                    "test_samples": len(datasets["test"]),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "training_time_seconds": elapsed,
                    **result,
                    "prediction_file": str(prediction_path.relative_to(ROOT)),
                }
            )
            print(
                f"{name} test MAE={result['mae_kg_ha']:.3f} "
                f"RMSE={result['rmse_kg_ha']:.3f} R2={result['r2']:.6f} "
                f"time={elapsed:.1f}s"
            )
            del model, optimizer

        comparison_frame = pd.DataFrame(comparison).sort_values("mae_kg_ha")
        comparison_frame.to_csv(RESULTS / "baseline_comparison.csv", index=False)
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        labels = comparison_frame["model"].str.replace("_", "\n")
        axes[0].bar(labels, comparison_frame["mae_kg_ha"])
        axes[0].set_title("Test MAE")
        axes[0].set_ylabel("kg/ha")
        axes[1].bar(labels, comparison_frame["rmse_kg_ha"])
        axes[1].set_title("Test RMSE")
        axes[1].set_ylabel("kg/ha")
        figure.tight_layout()
        figure.savefig(RESULTS / "baseline_comparison.png", dpi=160)
        plt.close(figure)
        lines = [
            "# Baseline Comparison",
            "",
            "All experiments used the exact existing chronological split: "
            "172 train, 26 validation, 41 test samples. Target normalization "
            "mean and standard deviation were calculated from training targets only.",
            "",
            f"Device used: `{device}`",
            f"PyTorch: `{torch.__version__}`",
            f"Configuration: AdamW, MSELoss, batch size {DEFAULT_BATCH_SIZE}, "
            f"learning rate {DEFAULT_LEARNING_RATE}, maximum epochs {DEFAULT_EPOCHS}, "
            f"early-stopping patience {DEFAULT_PATIENCE}.",
            "",
            "## Test Comparison",
            "",
            "| Model | Best epoch | Validation loss | MAE (kg/ha) | RMSE (kg/ha) | R2 | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in comparison_frame.to_dict("records"):
            lines.append(
                f"| {row['model']} | {row['best_epoch']} | {row['best_validation_loss']:.6f} | "
                f"{row['mae_kg_ha']:.3f} | {row['rmse_kg_ha']:.3f} | {row['r2']:.6f} | "
                f"{row['training_time_seconds']:.1f} |"
            )
        lines += [
            "",
            "## Models",
            "",
            "- Weather-only: existing masked WeatherLSTM plus regression head.",
            "- Satellite-only: existing SatelliteCNN plus regression head.",
            "- Soil-only: existing SoilMLP plus regression head.",
            "- Pair models: selected existing encoders concatenated into a compact regression head.",
            "- Full model: existing `MultimodalYieldModel` with learned attention fusion; "
            "its architecture was not changed.",
            "",
            "Predictions for every model are saved under "
            "`results/baseline_predictions/`. The comparison plot is "
            "`results/baseline_comparison.png`.",
            "",
            "No test data was used for normalization, training, or model selection.",
        ]
        (RESULTS / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
