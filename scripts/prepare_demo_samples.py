"""Copy frozen final samples into a faculty-friendly demo directory.

This script preserves the existing manual-upload Streamlit workflow. It copies
prepared satellite tensors and existing weather CSVs; it never moves or edits
the source files, frozen datasets, models, or result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_CSV = ROOT / "data" / "processed" / "final_training_samples.csv"
SATELLITE_DIR = ROOT / "data" / "processed" / "streamlit_satellite_samples"
WEATHER_DIR = ROOT / "data" / "raw" / "weather"
DEMO_DIR = ROOT / "data" / "demo_samples"
MANIFEST = DEMO_DIR / "demo_manifest.csv"
MANIFEST_COLUMNS = [
    "sample_id",
    "district",
    "year",
    "season",
    "satellite_path",
    "weather_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES_CSV)
    parser.add_argument("--satellite-dir", type=Path, default=SATELLITE_DIR)
    parser.add_argument("--weather-dir", type=Path, default=WEATHER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEMO_DIR)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_if_needed(source: Path, destination: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        if sha256(destination) == sha256(source):
            return "skipped"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return "copied"


def main() -> int:
    args = parse_args()
    samples_path = project_path(args.samples)
    satellite_dir = project_path(args.satellite_dir)
    weather_dir = project_path(args.weather_dir)
    output_dir = project_path(args.output_dir)
    samples = pd.read_csv(samples_path, dtype=str)
    required = {"sample_id", "district", "year", "season"}
    missing_columns = required.difference(samples.columns)
    if missing_columns:
        raise ValueError(f"Final sample list is missing: {sorted(missing_columns)}")
    if samples["sample_id"].duplicated().any():
        raise ValueError("Final sample list contains duplicate sample IDs")

    manifest_rows: list[dict[str, str]] = []
    copied = 0
    skipped = 0
    for row in samples.itertuples(index=False):
        sample_id = str(row.sample_id)
        district = str(row.district)
        year_label = str(row.year).replace("-", "_")
        season = str(row.season)
        folder = output_dir / district / f"{year_label}_{season}"
        source_satellite = satellite_dir / f"{sample_id}_satellite.npy"
        source_weather = weather_dir / f"{sample_id}.csv"
        destination_satellite = folder / "satellite.npy"
        destination_weather = folder / "weather.csv"

        satellite_status = copy_if_needed(source_satellite, destination_satellite)
        weather_status = copy_if_needed(source_weather, destination_weather)
        if satellite_status == "copied" or weather_status == "copied":
            copied += 1
        else:
            skipped += 1
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "district": district,
                "year": str(row.year),
                "season": season,
                "satellite_path": destination_satellite.relative_to(ROOT).as_posix(),
                "weather_path": destination_weather.relative_to(ROOT).as_posix(),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Final samples: {len(samples)}")
    print(f"Sample folders prepared: {len(manifest_rows)}")
    print(f"Folders/files copied or refreshed: {copied}")
    print(f"Already-correct sample folders skipped: {skipped}")
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
