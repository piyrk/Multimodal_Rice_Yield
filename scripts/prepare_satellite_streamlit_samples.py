"""Prepare lossless Streamlit satellite tensors from existing training GeoTIFFs.

The source GeoTIFFs are the outputs of ``src/extract_multimodal_dataset.py``.
That pipeline applies the frozen Landsat Collection 2 scaling, QA_PIXEL mask,
seasonal date window, district boundary, and seven-channel ordering before
writing each 128x128 float32 patch. This utility only converts those existing
patches to the channel-first NumPy format expected by the model and never
reprocesses, resizes, crops, or overwrites source data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "data" / "processed" / "final_training_samples.csv"
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "satellite_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "streamlit_satellite_samples"
DEFAULT_LOG = DEFAULT_OUTPUT / "satellite_tensor_generation_log.csv"
EXPECTED_SHAPE = (7, 128, 128)
EXPECTED_DTYPE = np.dtype("float32")
CHANNEL_ORDER = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "NDVI")


@dataclass(frozen=True)
class Result:
    sample_id: str
    status: str
    source_path: str
    output_path: str
    shape: str
    dtype: str
    finite: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--satellite-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-id", help="Process only this final sample ID.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many missing samples after filtering.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_final_samples(path: Path) -> pd.DataFrame:
    samples = pd.read_csv(path)
    required = {"sample_id", "district", "year", "season"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Final sample list is missing columns: {sorted(missing)}")
    if samples["sample_id"].duplicated().any():
        raise ValueError("Final sample list contains duplicate sample_id values")
    return samples


def load_satellite_sources(path: Path) -> dict[str, Path]:
    manifest = pd.read_csv(path)
    required = {"sample_id", "satellite_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Satellite manifest is missing columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Satellite manifest contains duplicate sample_id values")
    return {
        str(row.sample_id): resolve_project_path(str(row.satellite_path))
        for row in manifest.itertuples(index=False)
    }


def validate_tensor(array: np.ndarray, sample_id: str) -> None:
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{sample_id}: expected shape {EXPECTED_SHAPE}, found {array.shape}"
        )
    if array.dtype != EXPECTED_DTYPE:
        raise ValueError(
            f"{sample_id}: expected dtype {EXPECTED_DTYPE}, found {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{sample_id}: tensor contains NaN or Inf")


def source_to_tensor(source_path: Path, sample_id: str) -> np.ndarray:
    if not source_path.exists():
        raise FileNotFoundError(f"{sample_id}: source GeoTIFF not found: {source_path}")
    with rasterio.open(source_path) as source:
        if source.count != 7:
            raise ValueError(f"{sample_id}: source must contain seven bands")
        if tuple(source.dtypes) != ("float32",) * 7:
            raise ValueError(
                f"{sample_id}: source dtype must be float32, found {source.dtypes}"
            )
        array = source.read()
    if array.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"{sample_id}: expected shape {EXPECTED_SHAPE}, found {array.shape}"
        )
    # Match RiceMultimodalDataset: QA-masked nodata pixels are represented as
    # zero for model input while the extraction GeoTIFF remains unchanged.
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    validate_tensor(array, sample_id)
    return np.ascontiguousarray(array, dtype=np.float32)


def is_valid_output(path: Path, sample_id: str) -> tuple[bool, str]:
    if not path.exists():
        return False, "output does not exist"
    try:
        array = np.load(path, allow_pickle=False)
        validate_tensor(array, sample_id)
    except (OSError, ValueError) as error:
        return False, str(error)
    return True, "valid existing tensor"


def result_for(
    sample_id: str,
    status: str,
    source_path: Path,
    output_path: Path,
    shape: tuple[int, ...] | None,
    dtype: str,
    finite: bool,
    message: str,
) -> Result:
    return Result(
        sample_id=sample_id,
        status=status,
        source_path=str(source_path.relative_to(ROOT)),
        output_path=str(output_path.relative_to(ROOT)),
        shape=str(list(shape)) if shape is not None else "",
        dtype=dtype,
        finite=finite,
        message=message,
    )


def write_log(path: Path, rows: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=Result.__annotations__.keys())
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)


def write_manifest(path: Path, rows: list[Result]) -> None:
    manifest_rows = [
        row.__dict__
        for row in rows
        if row.status in {"generated", "skipped"} and row.finite
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    samples_path = resolve_project_path(args.samples)
    satellite_manifest_path = resolve_project_path(args.satellite_manifest)
    output_dir = resolve_project_path(args.output_dir)
    log_path = resolve_project_path(args.log_file)
    generated_manifest_path = output_dir / "streamlit_satellite_manifest.csv"

    samples = load_final_samples(samples_path)
    sources = load_satellite_sources(satellite_manifest_path)
    missing_rows: list[tuple[str, Path, Path]] = []
    results: list[Result] = []

    for row in samples.itertuples(index=False):
        sample_id = str(row.sample_id)
        if args.sample_id and sample_id != args.sample_id:
            continue
        source_path = sources.get(sample_id)
        if source_path is None:
            output_path = output_dir / f"{sample_id}_satellite.npy"
            missing_rows.append((sample_id, Path(""), output_path))
            continue
        output_path = output_dir / f"{sample_id}_satellite.npy"
        valid, message = is_valid_output(output_path, sample_id)
        if valid:
            results.append(
                result_for(
                    sample_id,
                    "skipped",
                    source_path,
                    output_path,
                    EXPECTED_SHAPE,
                    str(EXPECTED_DTYPE),
                    True,
                    message,
                )
            )
        else:
            missing_rows.append((sample_id, source_path, output_path))

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        missing_rows = missing_rows[: args.limit]

    if args.dry_run:
        print(f"Final samples: {len(samples)}")
        print(f"Already valid and skipped: {len(results)}")
        print(f"Would generate: {len(missing_rows)}")
        for sample_id, source_path, output_path in missing_rows:
            source_status = "source-present" if source_path.exists() else "source-missing"
            print(f"DRY-RUN {sample_id}: {source_status} -> {output_path}")
        return 0

    for sample_id, source_path, output_path in missing_rows:
        try:
            array = source_to_tensor(source_path, sample_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_suffix(".tmp.npy")
            np.save(temporary_path, array, allow_pickle=False)
            temporary_path.replace(output_path)
            results.append(
                result_for(
                    sample_id,
                    "generated",
                    source_path,
                    output_path,
                    array.shape,
                    str(array.dtype),
                    bool(np.isfinite(array).all()),
                    "generated from existing extraction GeoTIFF",
                )
            )
            print(f"GENERATED {sample_id}: {array.shape} {array.dtype}")
        except (OSError, ValueError) as error:
            results.append(
                result_for(
                    sample_id,
                    "failed",
                    source_path,
                    output_path,
                    None,
                    "",
                    False,
                    str(error),
                )
            )
            print(f"FAILED {sample_id}: {error}", file=sys.stderr)

    write_log(log_path, results)
    write_manifest(generated_manifest_path, results)
    counts = pd.Series([row.status for row in results]).value_counts().to_dict()
    print(f"Final samples: {len(samples)}")
    print(f"Generated: {counts.get('generated', 0)}")
    print(f"Skipped: {counts.get('skipped', 0)}")
    print(f"Failed: {counts.get('failed', 0)}")
    print(f"Output directory: {output_dir}")
    print(f"Generation log: {log_path}")
    print(f"Generated manifest: {generated_manifest_path}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
