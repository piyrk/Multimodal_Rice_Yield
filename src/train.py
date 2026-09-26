"""Train the multimodal yield model with train-only target normalization."""

from __future__ import annotations

import argparse
import json
import platform
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


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CHECKPOINT = MODELS / "best_multimodal_model.pt"


def subset_dataset(split: str) -> RiceMultimodalDataset:
    splits = pd.read_csv(PROCESSED / "dataset_splits.csv")
    subset = splits.loc[splits["split"] == split].drop(columns=["split"])
    path = PROCESSED / f".runtime_{split}.csv"
    subset.to_csv(path, index=False)
    return RiceMultimodalDataset(path)


def cleanup_runtime_files() -> None:
    for split in ("train", "validation", "test"):
        path = PROCESSED / f".runtime_{split}.csv"
        if path.exists():
            path.unlink()


def metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    pred = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    error = pred - target
    ss_total = float(np.sum((target - target.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / ss_total) if ss_total else 0.0,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[float] = []
    targets: list[float] = []
    for batch in loader:
        satellite = batch["satellite"].to(device)
        weather = batch["weather"].to(device)
        soil = batch["soil"].to(device)
        mask = batch["weather_mask"].to(device)
        target_kg = batch["target"].to(device)
        target_normalized = (target_kg - mean) / std
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction_normalized = model(satellite, weather, soil, mask)
            loss = criterion(prediction_normalized, target_normalized)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN/Inf loss encountered")
        prediction_kg = prediction_normalized.detach() * std + mean
        losses.append(float(loss.detach().cpu()))
        predictions.extend(prediction_kg.cpu().tolist())
        targets.extend(target_kg.cpu().tolist())
    result = metrics(predictions, targets)
    result["loss"] = float(np.mean(losses))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("GPU name: unavailable")
        print("Training device: CPU (lightweight configuration)")
        device = torch.device("cpu")

    try:
        train_dataset = subset_dataset("train")
        validation_dataset = subset_dataset("validation")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=multimodal_collate,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate,
        )
        train_targets = train_dataset.samples["yield_kg_ha"].to_numpy(dtype=np.float64)
        target_mean = float(train_targets.mean())
        target_std = float(train_targets.std(ddof=0))
        if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
            raise ValueError("Invalid train-only target normalization statistics")
        model_config = {
            "satellite_dim": 128,
            "weather_dim": 128,
            "soil_dim": 64,
            "fusion_dim": 128,
        }
        model = MultimodalYieldModel(**model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        history: list[dict[str, float | int]] = []
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, args.epochs + 1):
            train_result = run_epoch(
                model, train_loader, device, target_mean, target_std, criterion, optimizer
            )
            with torch.no_grad():
                validation_result = run_epoch(
                    model, validation_loader, device, target_mean, target_std, criterion, None
                )
            record = {
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "validation_loss": validation_result["loss"],
                "train_mae": train_result["mae"],
                "validation_mae": validation_result["mae"],
                "train_rmse": train_result["rmse"],
                "validation_rmse": validation_result["rmse"],
                "validation_r2": validation_result["r2"],
            }
            history.append(record)
            print(
                f"epoch {epoch}/{args.epochs} "
                f"train_loss={record['train_loss']:.5f} "
                f"validation_loss={record['validation_loss']:.5f} "
                f"validation_mae={record['validation_mae']:.2f}"
            )
            if validation_result["loss"] < best_loss:
                best_loss = validation_result["loss"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_validation_loss": best_loss,
                        "target_mean": target_mean,
                        "target_std": target_std,
                        "model_config": model_config,
                        "training_config": vars(args),
                        "device": str(device),
                    },
                    args.checkpoint,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"early stopping at epoch {epoch}")
                    break
        pd.DataFrame(history).to_csv(RESULTS / "training_history.csv", index=False)
        print(f"epochs completed: {len(history)}")
        print(f"best validation loss: {best_loss:.6f}")
        print(f"best epoch: {best_epoch}")
        print(f"checkpoint: {args.checkpoint}")
        print(f"normalization mean/std: {target_mean:.6f}/{target_std:.6f}")
    finally:
        cleanup_runtime_files()


if __name__ == "__main__":
    main()
