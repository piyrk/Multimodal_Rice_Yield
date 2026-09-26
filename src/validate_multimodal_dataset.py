"""Validate the repaired multimodal manifests without changing project data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reconcile_multimodal_dataset import PROCESSED, evaluate_sample, ROOT


def main() -> int:
    index = pd.read_csv(PROCESSED / "final_multimodal_index.csv")
    master = pd.read_csv(PROCESSED / "master_samples.csv")
    candidates = index.loc[index["final_sample_available"] == True].merge(
        master[["sample_id", "satellite_start", "satellite_end", "weather_start", "weather_end"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    satellite = pd.read_csv(PROCESSED / "satellite_manifest.csv")
    weather = pd.read_csv(PROCESSED / "weather_manifest.csv")
    soil = pd.read_csv(PROCESSED / "soil_features.csv")
    evaluations = [
        evaluate_sample(row, satellite, weather, soil)
        for _, row in candidates.iterrows()
    ]
    checks = pd.DataFrame(evaluations)
    manifest = pd.read_csv(PROCESSED / "multimodal_manifest.csv")
    training = pd.read_csv(PROCESSED / "final_training_samples.csv")
    expected_ids = set(checks.loc[checks["multimodal_valid"], "sample_id"])
    actual_ids = set(training["sample_id"])
    manifest_ids = set(manifest.loc[manifest["multimodal_valid"], "sample_id"])
    issues: list[str] = []
    if actual_ids != expected_ids:
        issues.append("final_training_samples does not equal the independently computed intersection")
    if manifest_ids != expected_ids:
        issues.append("multimodal_manifest multimodal_valid flags do not equal the independent intersection")
    if manifest["sample_id"].duplicated().any() or training["sample_id"].duplicated().any():
        issues.append("duplicate sample IDs found")
    print(f"candidate samples = {len(candidates)}")
    print(f"satellite complete count = {int(checks['satellite_valid'].sum())}")
    print(f"weather complete count = {int(checks['weather_valid'].sum())}")
    print(f"soil complete count = {int(checks['soil_valid'].sum())}")
    print(f"yield complete count = {int(checks['yield_valid'].sum())}")
    print(f"complete all-four-modalities count = {len(expected_ids)}")
    print(f"final_training_samples rows = {len(training)}")
    if issues:
        print("VALIDATION FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
