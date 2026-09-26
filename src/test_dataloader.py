"""Smoke-test the PyTorch dataset, chronological splits, and DataLoaders."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from dataset import RiceMultimodalDataset, multimodal_collate


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "dataset_splits.csv"


def make_dataset(split: str) -> RiceMultimodalDataset:
    import pandas as pd

    split_frame = pd.read_csv(SPLITS)
    split_frame = split_frame.loc[split_frame["split"] == split]
    path = ROOT / "data" / "processed" / f".pipeline_{split}.csv"
    split_frame.drop(columns=["split"]).to_csv(path, index=False)
    return RiceMultimodalDataset(path)


def main() -> int:
    datasets = {name: make_dataset(name) for name in ("train", "validation", "test")}
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=8,
            shuffle=(name == "train"),
            num_workers=0,
            collate_fn=multimodal_collate,
        )
        for name, dataset in datasets.items()
    }
    try:
        for name, loader in loaders.items():
            batches = 0
            for batch in loader:
                batches += 1
                assert batch["satellite"].shape[1:] == (7, 128, 128)
                assert batch["weather"].shape[2:] == (3,)
                assert batch["soil"].shape[1:] == (4,)
                assert batch["target"].shape[0] == len(batch["sample_id"])
                for key in ("satellite", "weather", "weather_mask", "soil", "target"):
                    assert torch.isfinite(batch[key].float()).all(), f"{name}: {key} has NaN/Inf"
            if batches == 0:
                raise AssertionError(f"{name} loader produced no batches")
        print(f"train count = {len(datasets['train'])}")
        print(f"validation count = {len(datasets['validation'])}")
        print(f"test count = {len(datasets['test'])}")
        first = next(iter(loaders["train"]))
        print(f"satellite shape = {tuple(first['satellite'].shape)}")
        print(f"weather shape = {tuple(first['weather'].shape)}")
        print(f"soil shape = {tuple(first['soil'].shape)}")
        print(f"target shape = {tuple(first['target'].shape)}")
        print(f"dtypes = satellite:{first['satellite'].dtype}, weather:{first['weather'].dtype}, soil:{first['soil'].dtype}, target:{first['target'].dtype}")
        print("NaN/Inf check = PASSED")
        print("every batch load = PASSED")
        return 0
    finally:
        for split in ("train", "validation", "test"):
            path = ROOT / "data" / "processed" / f".pipeline_{split}.csv"
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    sys.exit(main())
