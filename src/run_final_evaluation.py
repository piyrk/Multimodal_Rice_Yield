"""Train the locked concatenation model and evaluate the untouched test split."""

from __future__ import annotations

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
from run_concat_tuning import TunableConcatenationFusion
from run_baselines import regression_metrics


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SEEDS = (42, 123, 2026)
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
FUSION_DROPOUT = 0.0
FINAL_EPOCHS = 9


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_subset(splits: pd.DataFrame, split: str) -> Path:
    path = PROCESSED / f".final_{split}.csv"
    splits.loc[splits["split"] == split].drop(columns=["split"]).to_csv(
        path, index=False
    )
    return path


def cleanup() -> None:
    for split in ("train", "validation", "test", "train_validation"):
        path = PROCESSED / f".final_{split}.csv"
        if path.exists():
            path.unlink()


def make_loaders(
    train_validation: RiceMultimodalDataset,
    test: RiceMultimodalDataset,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_validation,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=multimodal_collate,
    )
    test_loader = DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate,
    )
    return train_loader, test_loader


def model_output(
    model: nn.Module, batch: dict[str, object], device: torch.device
) -> torch.Tensor:
    return model(
        satellite=batch["satellite"].to(device),
        weather=batch["weather"].to(device),
        soil=batch["soil"].to(device),
        weather_mask=batch["weather_mask"].to(device),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    losses: list[float] = []
    for batch in loader:
        target = batch["target"].to(device)
        normalized_target = (target - target_mean) / target_std
        optimizer.zero_grad(set_to_none=True)
        normalized_prediction = model_output(model, batch, device)
        loss = criterion(normalized_prediction, normalized_target)
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN/Inf training loss encountered")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evaluate_test(
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
            normalized_prediction = model_output(model, batch, device)
            prediction = normalized_prediction * target_std + target_mean
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("NaN/Inf test prediction encountered")
            actual = batch["target"]
            for sample_id, actual_value, predicted_value in zip(
                batch["sample_id"],
                actual.tolist(),
                prediction.cpu().tolist(),
            ):
                actual_float = float(actual_value)
                predicted_float = float(predicted_value)
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "actual_yield_kg_ha": actual_float,
                        "predicted_yield_kg_ha": predicted_float,
                        "error_kg_ha": predicted_float - actual_float,
                        "absolute_error_kg_ha": abs(predicted_float - actual_float),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    train_count = int((splits["split"].isin(["train", "validation"])).sum())
    test_count = int((splits["split"] == "test").sum())
    if train_count != 198 or test_count != 41:
        raise ValueError(f"Unexpected canonical split sizes: {train_count}/{test_count}")

    train_validation_path = PROCESSED / ".final_train_validation.csv"
    splits.loc[splits["split"].isin(["train", "validation"])].drop(
        columns=["split"]
    ).to_csv(train_validation_path, index=False)
    test_path = write_subset(splits, "test")
    train_validation = RiceMultimodalDataset(train_validation_path)
    test_dataset = RiceMultimodalDataset(test_path)
    target_values = train_validation.samples["yield_kg_ha"].to_numpy(dtype=np.float64)
    target_mean = float(target_values.mean())
    target_std = float(target_values.std(ddof=0))
    if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("Invalid train+validation normalization statistics")

    checkpoints: dict[int, Path] = {}
    training_times: dict[int, float] = {}
    try:
        print(
            f"training samples: {len(train_validation)}; test samples: {len(test_dataset)}"
        )
        print(f"target normalization mean={target_mean:.6f} std={target_std:.6f}")
        for seed in SEEDS:
            seed_everything(seed)
            train_loader, _ = make_loaders(train_validation, test_dataset, seed)
            model = TunableConcatenationFusion(FUSION_DROPOUT).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            start = time.perf_counter()
            for epoch in range(1, FINAL_EPOCHS + 1):
                loss = train_one_epoch(
                    model, train_loader, optimizer, device, target_mean, target_std
                )
                print(f"seed={seed} epoch={epoch}/{FINAL_EPOCHS} loss={loss:.6f}")
            elapsed = time.perf_counter() - start
            checkpoint_path = MODELS / f"final_model_seed{seed}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": FINAL_EPOCHS,
                    "best_validation_loss": None,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "model_configuration": {
                        "model": "Concatenation Fusion",
                        "satellite_embedding_dim": 128,
                        "weather_embedding_dim": 128,
                        "soil_embedding_dim": 64,
                        "fusion_dropout": FUSION_DROPOUT,
                        "learning_rate": LEARNING_RATE,
                        "batch_size": BATCH_SIZE,
                        "epochs": FINAL_EPOCHS,
                        "optimizer": "AdamW",
                        "loss": "MSELoss",
                        "seed": seed,
                        "training_samples": train_count,
                    },
                },
                checkpoint_path,
            )
            checkpoints[seed] = checkpoint_path
            training_times[seed] = elapsed
            del model, optimizer, train_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # The test loader is consumed only after every final checkpoint exists.
        all_predictions: list[pd.DataFrame] = []
        test_ids_by_seed: dict[int, list[str]] = {}
        for seed in SEEDS:
            _, test_loader = make_loaders(train_validation, test_dataset, seed)
            model = TunableConcatenationFusion(FUSION_DROPOUT).to(device)
            checkpoint = torch.load(checkpoints[seed], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            predictions = evaluate_test(
                model, test_loader, device, target_mean, target_std
            )
            if len(predictions) != test_count:
                raise ValueError(f"Seed {seed} produced {len(predictions)} test rows")
            if predictions["sample_id"].duplicated().any():
                raise ValueError(f"Duplicate test IDs for seed {seed}")
            if not np.isfinite(
                predictions[
                    ["actual_yield_kg_ha", "predicted_yield_kg_ha", "error_kg_ha"]
                ].to_numpy()
            ).all():
                raise FloatingPointError(f"Non-finite test values for seed {seed}")
            predictions.insert(0, "seed", seed)
            metrics = regression_metrics(
                predictions["actual_yield_kg_ha"].to_numpy(),
                predictions["predicted_yield_kg_ha"].to_numpy(),
            )
            test_ids_by_seed[seed] = predictions["sample_id"].tolist()
            all_predictions.append(predictions)
            print(f"evaluated seed={seed}: {metrics}")
            del model, test_loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        reference_ids = test_ids_by_seed[SEEDS[0]]
        if any(ids != reference_ids for ids in test_ids_by_seed.values()):
            raise ValueError("Test IDs differ across final seed evaluations")
        predictions_all = pd.concat(all_predictions, ignore_index=True)
        predictions_all.to_csv(RESULTS / "final_test_predictions.csv", index=False)

        result_rows: list[dict[str, float | int | str]] = []
        for seed in SEEDS:
            frame = predictions_all.loc[predictions_all["seed"] == seed]
            metrics = regression_metrics(
                frame["actual_yield_kg_ha"].to_numpy(),
                frame["predicted_yield_kg_ha"].to_numpy(),
            )
            result_rows.append(
                {
                    "seed": seed,
                    **metrics,
                    "minimum_prediction_kg_ha": float(frame["predicted_yield_kg_ha"].min()),
                    "maximum_prediction_kg_ha": float(frame["predicted_yield_kg_ha"].max()),
                    "mean_prediction_kg_ha": float(frame["predicted_yield_kg_ha"].mean()),
                    "test_sample_count": len(frame),
                    "training_time_seconds": training_times[seed],
                    "device": str(device),
                }
            )
        results = pd.DataFrame(result_rows)
        results.to_csv(RESULTS / "final_test_results.csv", index=False)

        figure, axis = plt.subplots(figsize=(8, 6))
        actual = predictions_all.drop_duplicates("sample_id")["actual_yield_kg_ha"]
        for seed in SEEDS:
            frame = predictions_all.loc[predictions_all["seed"] == seed]
            axis.scatter(
                actual,
                frame["predicted_yield_kg_ha"],
                s=24,
                alpha=0.7,
                label=f"seed {seed}",
            )
        limits = [float(actual.min()), float(actual.max())]
        axis.plot(limits, limits, "k--", linewidth=1, label="ideal")
        axis.set_xlabel("Actual yield (kg/ha)")
        axis.set_ylabel("Predicted yield (kg/ha)")
        axis.set_title("Final concatenation-fusion test evaluation")
        axis.legend()
        figure.tight_layout()
        figure.savefig(RESULTS / "final_actual_vs_predicted.png", dpi=160)
        plt.close(figure)

        mean_metrics = results[["mae_kg_ha", "rmse_kg_ha", "r2"]].mean()
        std_metrics = results[["mae_kg_ha", "rmse_kg_ha", "r2"]].std(ddof=1)
        report = [
            "# Final Held-Out Evaluation",
            "",
            "The locked concatenation-fusion configuration was retrained from scratch "
            "for three seeds using the combined train and validation samples. All "
            "three checkpoints were saved before the test loader was evaluated.",
            "",
            f"- Device: `{device}`",
            f"- Training samples: **{train_count}**",
            f"- Test samples: **{test_count}**",
            f"- Fixed epochs: **{FINAL_EPOCHS}** (median of selected-configuration "
            "best epochs 1, 9, and 9)",
            f"- Learning rate: **{LEARNING_RATE}**",
            f"- Fusion dropout: **{FUSION_DROPOUT}**",
            "- Optimizer/loss: **AdamW / MSELoss**",
            f"- Target normalization: train+validation targets only "
            f"(mean={target_mean:.6f}, std={target_std:.6f})",
            "",
            "## Per-seed results",
            "",
            "| Seed | MAE (kg/ha) | RMSE (kg/ha) | R² | Min prediction | Max prediction | Mean prediction |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in result_rows:
            report.append(
                f"| {row['seed']} | {row['mae_kg_ha']:.3f} | {row['rmse_kg_ha']:.3f} | "
                f"{row['r2']:.6f} | {row['minimum_prediction_kg_ha']:.3f} | "
                f"{row['maximum_prediction_kg_ha']:.3f} | {row['mean_prediction_kg_ha']:.3f} |"
            )
        report += [
            "",
            "## Three-seed aggregate",
            "",
            f"- Test MAE: **{mean_metrics['mae_kg_ha']:.3f} +/- {std_metrics['mae_kg_ha']:.3f} kg/ha**",
            f"- Test RMSE: **{mean_metrics['rmse_kg_ha']:.3f} +/- {std_metrics['rmse_kg_ha']:.3f} kg/ha**",
            f"- Test R²: **{mean_metrics['r2']:.6f} +/- {std_metrics['r2']:.6f}**",
            "",
            "Test predictions and per-sample errors are stored in "
            "`results/final_test_predictions.csv`. The test set was not used for "
            "configuration selection, epoch selection, or training.",
        ]
        (RESULTS / "final_evaluation.md").write_text("\n".join(report), encoding="utf-8")

        experiment_report = [
            "# Final Experiment Report",
            "",
            "## Architecture",
            "",
            "The final model is the locked concatenation-fusion architecture: the "
            "unchanged SatelliteCNN (128-dimensional embedding), WeatherLSTM "
            "(128-dimensional embedding), and SoilMLP (64-dimensional embedding) "
            "are concatenated into a 320-dimensional representation. The fusion "
            "MLP uses LayerNorm, Linear(320,128), ReLU, fusion dropout 0.0, "
            "Linear(128,64), and ReLU, followed by the existing regression head.",
            "",
            "## Locked configuration",
            "",
            f"- Learning rate: {LEARNING_RATE}",
            f"- Fusion dropout: {FUSION_DROPOUT}",
            "- Optimizer: AdamW",
            "- Loss: MSELoss",
            f"- Batch size: {BATCH_SIZE}",
            f"- Fixed final epochs: {FINAL_EPOCHS}",
            "- Seeds: 42, 123, 2026",
            "",
            "## Data and protocol",
            "",
            f"- Training samples: {train_count} (canonical train + validation)",
            f"- Held-out test samples: {test_count}",
            "- Normalization statistics were calculated from the 198 training samples "
            "only and stored in each checkpoint.",
            "- The test set was evaluated exactly once per final checkpoint, after all "
            "three final models had been trained and saved.",
            "- No test metric was used for model or epoch selection.",
            "",
            "## Results",
            "",
            f"Mean test MAE: {mean_metrics['mae_kg_ha']:.3f} +/- {std_metrics['mae_kg_ha']:.3f} kg/ha",
            f"Mean test RMSE: {mean_metrics['rmse_kg_ha']:.3f} +/- {std_metrics['rmse_kg_ha']:.3f} kg/ha",
            f"Mean test R²: {mean_metrics['r2']:.6f} +/- {std_metrics['r2']:.6f}",
            "",
            "## Limitations",
            "",
            "The final estimates are based on 41 held-out district-season-year "
            "samples and three random seeds. The test set is a historical holdout "
            "for final reporting, not a basis for further tuning. No uncertainty "
            "intervals or external validation set were calculated.",
        ]
        (RESULTS / "final_experiment_report.md").write_text(
            "\n".join(experiment_report), encoding="utf-8"
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
