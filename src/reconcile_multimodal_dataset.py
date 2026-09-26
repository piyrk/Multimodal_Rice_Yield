"""Reconcile extracted modalities and repair derived manifests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MIN_VALID_PIXEL_FRACTION = 0.05
REQUIRED_WEATHER_COLUMNS = {
    "date",
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
}


def _bool(value: bool) -> bool:
    return bool(value)


def evaluate_sample(
    row: pd.Series,
    satellite_by_id: pd.DataFrame,
    weather_by_id: pd.DataFrame,
    soil_by_district: pd.DataFrame,
) -> dict:
    sample_id = row["sample_id"]
    district = row["district"]
    reasons: list[str] = []

    sat_rows = satellite_by_id.loc[satellite_by_id["sample_id"] == sample_id]
    sat_row = sat_rows.iloc[0] if not sat_rows.empty else None
    satellite_path = "" if sat_row is None else str(sat_row.get("satellite_path", ""))
    satellite_complete = True
    if sat_row is None or not satellite_path or satellite_path == "nan":
        satellite_complete = False
        reasons.append("satellite_missing")
    else:
        sat_file = ROOT / Path(satellite_path)
        if not sat_file.exists():
            satellite_complete = False
            reasons.append("satellite_missing")
        else:
            try:
                with rasterio.open(sat_file) as source:
                    if source.count != 7:
                        satellite_complete = False
                        reasons.append("satellite_bands")
                    if source.height != 128 or source.width != 128:
                        satellite_complete = False
                        reasons.append("satellite_shape")
                    if any(dtype != "float32" for dtype in source.dtypes):
                        satellite_complete = False
                        reasons.append("satellite_dtype")
            except Exception:
                satellite_complete = False
                reasons.append("satellite_read_error")
        fraction = pd.to_numeric(
            sat_row.get("valid_pixel_fraction"), errors="coerce"
        )
        if not np.isfinite(fraction) or fraction < MIN_VALID_PIXEL_FRACTION:
            satellite_complete = False
            reasons.append("satellite_valid_fraction")

    weather_rows = weather_by_id.loc[weather_by_id["sample_id"] == sample_id]
    weather_row = weather_rows.iloc[0] if not weather_rows.empty else None
    weather_path = "" if weather_row is None else str(weather_row.get("weather_path", ""))
    weather_complete = True
    if weather_row is None or not weather_path or weather_path == "nan":
        weather_complete = False
        reasons.append("weather_missing")
    else:
        weather_file = ROOT / Path(weather_path)
        if not weather_file.exists():
            weather_complete = False
            reasons.append("weather_missing")
        else:
            try:
                weather = pd.read_csv(weather_file)
                if not REQUIRED_WEATHER_COLUMNS.issubset(weather.columns):
                    weather_complete = False
                    reasons.append("weather_columns")
                else:
                    dates = pd.to_datetime(weather["date"], errors="coerce")
                    expected_start = pd.Timestamp(row["weather_start"])
                    expected_end = pd.Timestamp(row["weather_end"])
                    expected_dates = pd.date_range(expected_start, expected_end, freq="D")
                    actual_dates = dates.dt.normalize()
                    if (
                        dates.isna().any()
                        or len(actual_dates) != len(expected_dates)
                        or not actual_dates.is_monotonic_increasing
                        or actual_dates.duplicated().any()
                        or not actual_dates.reset_index(drop=True).equals(
                            pd.Series(expected_dates, name="date")
                        )
                        or weather[
                            ["rainfall_mm", "temperature_c", "humidity_pct"]
                        ].isna().any().any()
                    ):
                        weather_complete = False
                        reasons.append("weather_incomplete")
            except Exception:
                weather_complete = False
                reasons.append("weather_read_error")

    soil_rows = soil_by_district.loc[soil_by_district["district"] == district]
    soil_complete = not soil_rows.empty
    if soil_complete:
        soil_values = soil_rows.iloc[0][
            ["nitrogen_g_kg", "ph", "soc_g_kg", "clay_pct"]
        ]
        soil_complete = bool(soil_values.notna().all())
    if not soil_complete:
        reasons.append("soil_missing")

    yield_value = pd.to_numeric(pd.Series([row["yield_kg_ha"]]), errors="coerce").iloc[0]
    yield_complete = bool(np.isfinite(yield_value))
    if not yield_complete:
        reasons.append("yield_invalid")

    return {
        "sample_id": sample_id,
        "satellite_path": satellite_path if satellite_path != "nan" else "",
        "weather_path": weather_path if weather_path != "nan" else "",
        "satellite_valid": _bool(satellite_complete),
        "weather_valid": _bool(weather_complete),
        "soil_valid": _bool(soil_complete),
        "yield_valid": _bool(yield_complete),
        "multimodal_valid": _bool(
            satellite_complete and weather_complete and soil_complete and yield_complete
        ),
        "reasons": reasons,
    }


def main() -> None:
    index = pd.read_csv(PROCESSED / "final_multimodal_index.csv")
    master = pd.read_csv(PROCESSED / "master_samples.csv")
    candidates = index.loc[index["final_sample_available"] == True].copy()
    candidates = candidates.merge(
        master[["sample_id", "satellite_start", "satellite_end", "weather_start", "weather_end"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if len(candidates) != 273:
        raise ValueError(f"Expected 273 candidates, found {len(candidates)}")

    satellite = pd.read_csv(PROCESSED / "satellite_manifest.csv")
    weather = pd.read_csv(PROCESSED / "weather_manifest.csv")
    soil = pd.read_csv(PROCESSED / "soil_features.csv")
    if satellite["sample_id"].duplicated().any() or weather["sample_id"].duplicated().any():
        raise ValueError("Satellite or weather manifest contains duplicate sample IDs")

    evaluations = [
        evaluate_sample(row, satellite, weather, soil)
        for _, row in candidates.iterrows()
    ]
    checks = pd.DataFrame(evaluations).set_index("sample_id")

    manifest = candidates[
        ["sample_id", "district", "year", "season", "yield_kg_ha"]
    ].copy()
    manifest["soil_district"] = manifest["district"]
    manifest = manifest.set_index("sample_id")
    for column in [
        "satellite_path",
        "satellite_valid",
        "weather_path",
        "weather_valid",
        "soil_valid",
        "yield_valid",
        "multimodal_valid",
    ]:
        manifest[column] = checks[column]
    manifest = manifest.reset_index()[
        [
            "sample_id",
            "soil_district",
            "year",
            "season",
            "yield_kg_ha",
            "satellite_path",
            "satellite_valid",
            "weather_path",
            "weather_valid",
            "soil_valid",
            "yield_valid",
            "multimodal_valid",
        ]
    ]
    manifest.to_csv(PROCESSED / "multimodal_manifest.csv", index=False)

    training = manifest.loc[manifest["multimodal_valid"]].copy()
    training[
        ["sample_id", "soil_district", "year", "season", "yield_kg_ha"]
    ].rename(columns={"soil_district": "district"}).to_csv(
        PROCESSED / "final_training_samples.csv", index=False
    )

    reason_counts = Counter(
        reason for result in evaluations for reason in result["reasons"]
    )
    complete_count = int(checks["multimodal_valid"].sum())
    report_lines = [
        "# Final Dataset Integrity Report",
        "",
        "This report was generated from the 273 rows marked `final_sample_available=True` "
        "in `final_multimodal_index.csv`. Counts are row-level intersections after "
        "checking the on-disk files and modality contents.",
        "",
        "## Counts",
        "",
        f"- Candidate samples: {len(candidates)}",
        f"- Satellite complete count: {int(checks['satellite_valid'].sum())}",
        f"- Weather complete count: {int(checks['weather_valid'].sum())}",
        f"- Soil complete count: {int(checks['soil_valid'].sum())}",
        f"- Yield complete count: {int(checks['yield_valid'].sum())}",
        f"- Complete all-four-modalities count: {complete_count}",
        f"- Excluded sample count: {len(candidates) - complete_count}",
        f"- Duplicate sample IDs: {int(candidates['sample_id'].duplicated().sum())}",
        "",
        "## Exclusion Reason Counts",
        "",
    ]
    for reason, count in sorted(reason_counts.items()):
        report_lines.append(f"- {reason}: {count}")
    report_lines += [
        "",
        "## Integrity Checks",
        "",
        "- Satellite shape consistency: every complete satellite is 7 bands, 128x128, float32.",
        f"- Satellite valid-pixel threshold: {MIN_VALID_PIXEL_FRACTION:.2f}; "
        f"{int(checks['satellite_valid'].sum())} samples passed.",
        "- Weather completeness: expected daily date range, chronological order, required "
        "columns, and non-missing values checked for every candidate.",
        f"- Weather complete: {int(checks['weather_valid'].sum())}/{len(candidates)}.",
        f"- Soil coverage: {int(checks['soil_valid'].sum())}/{len(candidates)} samples; "
        f"{soil['district'].nunique()} district records available.",
        f"- Yield coverage: {int(checks['yield_valid'].sum())}/{len(candidates)} numeric finite values.",
        f"- Missing modality rows: {int((~checks[['satellite_valid', 'weather_valid', 'soil_valid']]).any(axis=1).sum())}.",
        "",
        "## Repaired Outputs",
        "",
        "- `data/processed/final_training_samples.csv` now contains only the complete intersection.",
        "- `data/processed/multimodal_manifest.csv` flags are based on the independent checks above.",
        "",
    ]
    (ROOT / "results" / "final_dataset_integrity_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"candidate samples = {len(candidates)}")
    print(f"satellite complete count = {int(checks['satellite_valid'].sum())}")
    print(f"weather complete count = {int(checks['weather_valid'].sum())}")
    print(f"soil complete count = {int(checks['soil_valid'].sum())}")
    print(f"yield complete count = {int(checks['yield_valid'].sum())}")
    print(f"complete all-four-modalities count = {complete_count}")
    print(f"excluded sample count = {len(candidates) - complete_count}")
    print(f"final_training_samples rows = {len(training)}")
    print("exclusion reasons:", dict(sorted(reason_counts.items())))


if __name__ == "__main__":
    main()
