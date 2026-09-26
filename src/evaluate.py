"""Evaluate the best checkpoint on the held-out test split only."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import RiceMultimodalDataset, multimodal_collate
from models.fusion_model import MultimodalYieldModel


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CHECKPOINT = ROOT / "models" / "best_multimodal_model.pt"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    test = splits.loc[splits["split"] == "test"].drop(columns=["split"])
    runtime_path = PROCESSED / ".runtime_test.csv"
    test.to_csv(runtime_path, index=False)
    try:
        dataset = RiceMultimodalDataset(runtime_path)
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=multimodal_collate
        )
        checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model = MultimodalYieldModel(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        rows: list[dict[str, float | str]] = []
        with torch.inference_mode():
            for batch in loader:
                details = model(
                    batch["satellite"].to(device),
                    batch["weather"].to(device),
                    batch["soil"].to(device),
                    batch["weather_mask"].to(device),
                    return_details=True,
                )
                predictions = details["prediction"].cpu() * checkpoint["target_std"] + checkpoint["target_mean"]
                weights = details["attention_weights"].cpu()
                if not torch.isfinite(predictions).all():
                    raise FloatingPointError("Test predictions contain NaN/Inf")
                for index, sample_id in enumerate(batch["sample_id"]):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "actual_yield_kg_ha": float(batch["target"][index]),
                            "predicted_yield_kg_ha": float(predictions[index]),
                            "satellite_attention": float(weights[index, 0]),
                            "weather_attention": float(weights[index, 1]),
                            "soil_attention": float(weights[index, 2]),
                        }
                    )
        predictions = pd.DataFrame(rows)
        actual = predictions["actual_yield_kg_ha"].to_numpy()
        predicted = predictions["predicted_yield_kg_ha"].to_numpy()
        error = predicted - actual
        ss_total = np.sum((actual - actual.mean()) ** 2)
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        r2 = float(1 - np.sum(error**2) / ss_total) if ss_total else 0.0
        predictions.to_csv(RESULTS / "test_predictions.csv", index=False)
        plt.figure(figsize=(7, 6))
        plt.scatter(actual, predicted, alpha=0.8)
        bounds = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        plt.plot(bounds, bounds, linestyle="--", color="black", label="ideal")
        plt.xlabel("Actual yield (kg/ha)")
        plt.ylabel("Predicted yield (kg/ha)")
        plt.title("Test-set actual vs predicted rice yield")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "actual_vs_predicted_yield.png", dpi=160)
        plt.close()
        report = f"""# Test-Set Model Evaluation

Evaluation used only the held-out chronological test split: **{len(predictions)} samples**.
No test target was used for training, normalization, or checkpoint selection.

## Metrics

- MAE: **{mae:.3f} kg/ha**
- RMSE: **{rmse:.3f} kg/ha**
- R2: **{r2:.6f}**
- Minimum prediction: **{predicted.min():.3f} kg/ha**
- Maximum prediction: **{predicted.max():.3f} kg/ha**
- Mean prediction: **{predicted.mean():.3f} kg/ha**
- Device: `{device}`

## Outputs

- Per-sample predictions and attention: `results/test_predictions.csv`
- Actual-vs-predicted plot: `results/actual_vs_predicted_yield.png`
- Training history: `results/training_history.csv`
"""
        (RESULTS / "model_evaluation.md").write_text(report, encoding="utf-8")
        print(f"test samples = {len(predictions)}")
        print(f"test MAE = {mae:.3f} kg/ha")
        print(f"test RMSE = {rmse:.3f} kg/ha")
        print(f"test R2 = {r2:.6f}")
        print(f"minimum prediction = {predicted.min():.3f} kg/ha")
        print(f"maximum prediction = {predicted.max():.3f} kg/ha")
        print(f"mean prediction = {predicted.mean():.3f} kg/ha")
        print("test evaluation = PASSED")
    finally:
        if runtime_path.exists():
            runtime_path.unlink()


if __name__ == "__main__":
    main()
