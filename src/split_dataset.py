"""Create a chronological, year-grouped split for the validated samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "final_training_samples.csv"
OUTPUT = ROOT / "data" / "processed" / "dataset_splits.csv"


def main() -> None:
    frame = pd.read_csv(INPUT)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs found")
    years = sorted(frame["year"].unique(), key=lambda value: int(str(value)[:4]))
    if len(years) < 3:
        raise ValueError("At least three year groups are required")
    test_years = set(years[-2:])
    validation_years = {years[-3]}
    train_years = set(years) - test_years - validation_years
    frame["split"] = frame["year"].map(
        lambda year: (
            "test"
            if year in test_years
            else "validation"
            if year in validation_years
            else "train"
        )
    )
    if frame["split"].isna().any():
        raise ValueError("Some samples were not assigned a split")
    frame.to_csv(OUTPUT, index=False)
    print(f"train count = {(frame['split'] == 'train').sum()}")
    print(f"validation count = {(frame['split'] == 'validation').sum()}")
    print(f"test count = {(frame['split'] == 'test').sum()}")
    print(f"train years = {sorted(train_years)}")
    print(f"validation years = {sorted(validation_years)}")
    print(f"test years = {sorted(test_years)}")


if __name__ == "__main__":
    main()
