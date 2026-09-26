"""Repeat selected baseline experiments with controlled random seeds."""

from __future__ import annotations

import copy
import gc
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import RiceMultimodalDataset, multimodal_collate
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
from models.fusion_model import MultimodalYieldModel


RESULTS = ROOT / "results"
SEEDS = (42, 123, 2026)
MODELS = ("satellite_only", "satellite_weather", "satellite_weather_soil_attention")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and all visible CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets() -> dict[str, RiceMultimodalDataset]:
    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    datasets: dict[str, RiceMultimodalDataset] = {}
    for split in ("train", "validation", "test"):
        path = PROCESSED / f".repeatability_{split}.csv"
        subset = splits.loc[splits["split"] == split].drop(columns=["split"])
        subset.to_csv(path, index=False)
        datasets[split] = RiceMultimodalDataset(path)
    return datasets


def cleanup() -> None:
    for split in ("train", "validation", "test"):
        path = PROCESSED / f".repeatability_{split}.csv"
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


def model_output(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device
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


def evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    actual_values: list[float] = []
    predicted_values: list[float] = []
    attention_rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for batch in loader:
            normalized_prediction, attention = model_output(model, batch, device)
            prediction = normalized_prediction.cpu() * target_std + target_mean
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Test predictions contain NaN/Inf")
            actual_values.extend(batch["target"].tolist())
            predicted_values.extend(prediction.tolist())
            if attention is not None:
                weights = attention.cpu()
                for index, sample_id in enumerate(batch["sample_id"]):
                    attention_rows.append(
                        {
                            "sample_id": sample_id,
                            "satellite_attention": float(weights[index, 0]),
                            "weather_attention": float(weights[index, 1]),
                            "soil_attention": float(weights[index, 2]),
                        }
                    )
    actual = np.asarray(actual_values, dtype=np.float64)
    predicted = np.asarray(predicted_values, dtype=np.float64)
    return regression_metrics(actual, predicted), pd.DataFrame(attention_rows)


def train_one(
    model_name: str,
    seed: int,
    datasets: dict[str, RiceMultimodalDataset],
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    seed_everything(seed)
    loaders = make_loaders(datasets, seed)
    model = create_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    best_epoch = 0
    stale = 0
    for current_epoch in range(1, DEFAULT_EPOCHS + 1):
        train_result = epoch(
            model, loaders["train"], device, target_mean, target_std, optimizer
        )
        with torch.no_grad():
            validation_result = epoch(
                model, loaders["validation"], device, target_mean, target_std, None
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
    metrics, attention = evaluate_test(
        model, loaders["test"], device, target_mean, target_std
    )
    metrics.update(
        {
            "seed": seed,
            "model": model_name,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        }
    )
    return metrics, attention


def write_attention_summary(attention_runs: list[pd.DataFrame]) -> None:
    rows: list[dict[str, float | int | str]] = []
    columns = ("satellite_attention", "weather_attention", "soil_attention")
    labels = {
        "satellite_attention": "satellite",
        "weather_attention": "weather",
        "soil_attention": "soil",
    }
    for seed, frame in zip(SEEDS, attention_runs):
        for column in columns:
            values = frame[column].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "scope": f"seed_{seed}",
                    "seed": seed,
                    "sample_count": len(values),
                    "modality": labels[column],
                    "mean_attention": float(values.mean()),
                    "std_attention": float(values.std(ddof=0)),
                    "min_attention": float(values.min()),
                    "max_attention": float(values.max()),
                }
            )
    combined = pd.concat(attention_runs, ignore_index=True)
    for column in columns:
        values = combined[column].to_numpy(dtype=np.float64)
        rows.append(
            {
                "scope": "all_seeds",
                "seed": "",
                "sample_count": len(values),
                "modality": labels[column],
                "mean_attention": float(values.mean()),
                "std_attention": float(values.std(ddof=0)),
                "min_attention": float(values.min()),
                "max_attention": float(values.max()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "attention_summary.csv", index=False)
    lines = [
        "# Attention Weight Summary",
        "",
        "These are descriptive learned attention weights from the full "
        "Satellite + Weather + Soil + Attention model. They are not causal "
        "importance measures.",
        "",
        "Each seed row summarizes the 41 held-out test samples. The "
        "`all_seeds` rows summarize 123 attention vectors across the three "
        "repeatability runs.",
        "",
        "| Scope | Modality | N | Mean | Std | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scope']} | {row['modality']} | {row['sample_count']} | "
            f"{row['mean_attention']:.6f} | {row['std_attention']:.6f} | "
            f"{row['min_attention']:.6f} | {row['max_attention']:.6f} |"
        )
    lines += [
        "",
        "Attention weights are normalized across the three modalities for each "
        "sample by the model's softmax attention layer.",
    ]
    (RESULTS / "attention_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Repeatability audit requires the GPU configuration")
    print(f"device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    datasets = make_datasets()
    try:
        targets = datasets["train"].samples["yield_kg_ha"].to_numpy(dtype=np.float64)
        target_mean = float(targets.mean())
        target_std = float(targets.std(ddof=0))
        if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid train-only target normalization statistics")
        result_rows: list[dict[str, float | int | str]] = []
        attention_runs: list[pd.DataFrame] = []
        for model_name in MODELS:
            for seed in SEEDS:
                metrics, attention = train_one(
                    model_name,
                    seed,
                    datasets,
                    device,
                    target_mean,
                    target_std,
                )
                if (
                    model_name == "satellite_weather_soil_attention"
                    and len(attention) != 41
                ):
                    raise ValueError(
                        f"{model_name} seed {seed} produced {len(attention)} attention rows"
                    )
                result_rows.append(metrics)
                if model_name == "satellite_weather_soil_attention":
                    attention_runs.append(attention.assign(seed=seed))
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"{model_name} seed={seed} MAE={metrics['mae_kg_ha']:.3f} "
                    f"RMSE={metrics['rmse_kg_ha']:.3f} R2={metrics['r2']:.6f}"
                )
        raw = pd.DataFrame(result_rows)
        raw.to_csv(RESULTS / "repeatability_runs.csv", index=False)
        summary_rows: list[dict[str, float | str]] = []
        for model_name in MODELS:
            frame = raw.loc[raw["model"] == model_name]
            summary_rows.append(
                {
                    "model": model_name,
                    "runs": len(frame),
                    "mean_mae_kg_ha": frame["mae_kg_ha"].mean(),
                    "std_mae_kg_ha": frame["mae_kg_ha"].std(ddof=0),
                    "mean_rmse_kg_ha": frame["rmse_kg_ha"].mean(),
                    "std_rmse_kg_ha": frame["rmse_kg_ha"].std(ddof=0),
                    "mean_r2": frame["r2"].mean(),
                    "std_r2": frame["r2"].std(ddof=0),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(RESULTS / "repeatability_results.csv", index=False)
        write_attention_summary(attention_runs)
        lines = [
            "# GPU Repeatability Report",
            "",
            "Three GPU runs were completed for each selected model with seeds "
            "42, 123, and 2026.",
            "",
            f"Device: `{torch.cuda.get_device_name(0)}`",
            f"PyTorch: `{torch.__version__}`; CUDA runtime: `{torch.version.cuda}`",
            "",
            "The exact existing split (172/26/41), train-only target "
            "normalization, AdamW, MSELoss, batch size 16, learning rate "
            "0.001, maximum 8 epochs, and patience 3 were retained.",
            "",
            "## Per-seed results",
            "",
            "| Model | Seed | Best epoch | Validation loss | MAE | RMSE | R2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in raw.to_dict("records"):
            lines.append(
                f"| {row['model']} | {row['seed']} | {row['best_epoch']} | "
                f"{row['best_validation_loss']:.6f} | {row['mae_kg_ha']:.3f} | "
                f"{row['rmse_kg_ha']:.3f} | {row['r2']:.6f} |"
            )
        lines += [
            "",
            "## Mean and standard deviation",
            "",
            "| Model | Mean MAE | Std MAE | Mean RMSE | Std RMSE | Mean R2 | Std R2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary.to_dict("records"):
            lines.append(
                f"| {row['model']} | {row['mean_mae_kg_ha']:.3f} | "
                f"{row['std_mae_kg_ha']:.3f} | {row['mean_rmse_kg_ha']:.3f} | "
                f"{row['std_rmse_kg_ha']:.3f} | {row['mean_r2']:.6f} | "
                f"{row['std_r2']:.6f} |"
            )
        lines += [
            "",
            "Attention statistics are in `attention_summary.csv` and "
            "`attention_summary.md`. They are descriptive model outputs, not "
            "causal modality importance.",
        ]
        (RESULTS / "repeatability_report.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
