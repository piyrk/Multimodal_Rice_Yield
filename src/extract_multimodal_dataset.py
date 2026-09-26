from __future__ import annotations

import io
import math
import traceback
import zipfile
from pathlib import Path
from urllib.request import urlopen

import ee
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile


PROJECT_ID = "poetic-world-441912-n4"
GAUL_DATASET = "FAO/GAUL/2015/level2"
LANDSAT_DATASET = "LANDSAT/LT05/C02/T1_L2"
ERA5_DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
SOIL_LAYERS = {
    "nitrogen_g_kg": ("projects/soilgrids-isric/nitrogen_mean", "nitrogen_0-5cm_mean", 100.0),
    "ph": ("projects/soilgrids-isric/phh2o_mean", "phh2o_0-5cm_mean", 10.0),
    "soc_g_kg": ("projects/soilgrids-isric/soc_mean", "soc_0-5cm_mean", 10.0),
    "clay_pct": ("projects/soilgrids-isric/clay_mean", "clay_0-5cm_mean", 10.0),
}
PATCH_SIZE = 128
PATCH_METERS = PATCH_SIZE * 30
MIN_VALID_PIXEL_FRACTION = 0.05

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/processed/final_multimodal_index.csv"
MASTER = ROOT / "data/processed/master_samples.csv"
TRAINING = ROOT / "data/processed/final_training_samples.csv"
SATELLITE_DIR = ROOT / "data/raw/satellite"
WEATHER_DIR = ROOT / "data/raw/weather"
SATELLITE_MANIFEST = ROOT / "data/processed/satellite_manifest.csv"
WEATHER_MANIFEST = ROOT / "data/processed/weather_manifest.csv"
SOIL_FEATURES = ROOT / "data/processed/soil_features.csv"
MULTIMODAL_MANIFEST = ROOT / "data/processed/multimodal_manifest.csv"
LOG_FILE = ROOT / "results/extraction_log.csv"

GAUL_NAME_MAP = {
    "ANANTPUR": "Anantapur",
    "CHITTOOR": "Chittoor",
    "CUDDAPAH": "Cuddapah",
    "EAST GODAVARI": "East Godavari",
    "GUNTUR": "Guntur",
    "KRISHNA": "Krishna",
    "KURNOOL": "Kurnool",
    "NELLORE": "Nellore",
    "PRAKASAM": "Prakasam",
    "SRIKAKULAM": "Srikakulam",
    "VISAKHAPATNAM": "Vishakhapatnam",
    "VIZIANAGARM": "Vizianagaram",
    "WEST GODAVARI": "West Godavari",
}

SATELLITE_COLUMNS = [
    "SR_B1",
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B7",
    "NDVI",
]
WEATHER_COLUMNS = ["date", "rainfall_mm", "temperature_c", "humidity_pct"]


def end_exclusive(value: str) -> str:
    return (pd.Timestamp(value) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def districts_collection() -> ee.FeatureCollection:
    return (
        ee.FeatureCollection(GAUL_DATASET)
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.eq("ADM1_NAME", "Andhra Pradesh"))
    )


def district_region(collection: ee.FeatureCollection, district: str) -> ee.FeatureCollection:
    name = GAUL_NAME_MAP.get(district.upper())
    if name is None:
        raise ValueError(f"No GAUL mapping found for district: {district}")
    return collection.filter(ee.Filter.eq("ADM2_NAME", name))


def mask_landsat(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(int("11111", 2)).eq(0)
    scaled = image.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])
    scaled = scaled.multiply(0.0000275).add(-0.2)
    ndvi = scaled.normalizedDifference(["SR_B4", "SR_B3"]).rename("NDVI")
    return scaled.addBands(ndvi).updateMask(mask).copyProperties(image, image.propertyNames())


def patch_region(region: ee.FeatureCollection) -> ee.Geometry:
    return region.geometry().centroid(maxError=100).buffer(PATCH_METERS / 2).bounds(maxError=100)


def download_patch(image: ee.Image, region: ee.FeatureCollection, destination: Path) -> float:
    clipped = image.clip(region).toFloat()
    params = {
        "region": patch_region(region),
        "dimensions": f"{PATCH_SIZE}x{PATCH_SIZE}",
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    payload = urlopen(clipped.getDownloadURL(params), timeout=180).read()
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            tif_names = [name for name in archive.namelist() if name.endswith(".tif")]
            if len(tif_names) != 1:
                raise ValueError(f"Expected one GeoTIFF, found {tif_names}")
            payload = archive.read(tif_names[0])

    with MemoryFile(payload) as memory:
        with memory.open() as source:
            data = source.read()
            profile = source.profile.copy()
            valid = np.all(np.isfinite(data), axis=0)
            valid_fraction = float(valid.mean())
            profile.update(count=len(SATELLITE_COLUMNS), dtype="float32", nodata=np.nan)
            output = np.full((len(SATELLITE_COLUMNS), PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
            output[:, : data.shape[1], : data.shape[2]] = data.astype(np.float32)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(destination, "w", **profile) as target:
                target.write(output)
    return valid_fraction


def extract_weather(
    collection: ee.ImageCollection,
    region: ee.FeatureCollection,
    start: str,
    end: str,
) -> pd.DataFrame:
    filtered = collection.filterBounds(region).filterDate(start, end_exclusive(end))

    def add_means(image: ee.Image) -> ee.Image:
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region.geometry(),
            scale=10000,
            bestEffort=True,
            maxPixels=1000000,
        )
        return image.set(values).set("sample_date", image.date().format("YYYY-MM-dd"))

    images = filtered.map(add_means)
    values = images.aggregate_array("total_precipitation_sum").getInfo()
    temperatures = images.aggregate_array("temperature_2m").getInfo()
    dewpoints = images.aggregate_array("dewpoint_temperature_2m").getInfo()
    dates = images.aggregate_array("sample_date").getInfo()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "rainfall_mm": np.asarray(values, dtype=float) * 1000,
            "temperature_c": np.asarray(temperatures, dtype=float) - 273.15,
            "dewpoint_c": np.asarray(dewpoints, dtype=float) - 273.15,
        }
    ).sort_values("date")
    # Magnus formula using daily mean air temperature and dew point.
    frame["humidity_pct"] = (
        100
        * np.exp(
            (17.625 * frame["dewpoint_c"]) / (243.04 + frame["dewpoint_c"])
            - (17.625 * frame["temperature_c"]) / (243.04 + frame["temperature_c"])
        )
    ).clip(0, 100)
    return frame[WEATHER_COLUMNS].reset_index(drop=True)


def extract_soil(region: ee.FeatureCollection) -> dict[str, float]:
    images = {
        output: ee.Image(asset).select(band)
        for output, (asset, band, _) in SOIL_LAYERS.items()
    }
    combined = ee.Image.cat(list(images.values()))
    values = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region.geometry(),
        scale=250,
        bestEffort=True,
        maxPixels=100000000,
    ).getInfo()
    result = {}
    for output, (_, band, divisor) in SOIL_LAYERS.items():
        value = values.get(band)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Missing SoilGrids value for {band}")
        result[output] = float(value) / divisor
    return result


def record_failure(rows: list[dict], sample_id: str, stage: str, error: Exception) -> None:
    rows.append(
        {
            "sample_id": sample_id,
            "stage": stage,
            "status": "failed",
            "error": str(error),
        }
    )
    print(f"ERROR [{stage}] {sample_id}: {error}")


def main() -> None:
    index = pd.read_csv(INDEX)
    master = pd.read_csv(MASTER)
    index = index.merge(
        master[
            [
                "sample_id",
                "satellite_start",
                "satellite_end",
                "weather_start",
                "weather_end",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    candidates = index[index["final_sample_available"] == True].copy()
    if len(candidates) != 273:
        raise ValueError(f"Expected 273 final candidates, found {len(candidates)}")
    candidates[["sample_id", "district", "year", "season", "yield_kg_ha"]].to_csv(
        TRAINING, index=False
    )

    ee.Initialize(project=PROJECT_ID)
    districts = districts_collection()
    regions = {
        district: district_region(districts, district)
        for district in candidates["district"].astype(str).str.strip().unique()
    }
    landsat = ee.ImageCollection(LANDSAT_DATASET)
    era5 = ee.ImageCollection(ERA5_DATASET)
    failures: list[dict] = []
    satellite_rows: list[dict] = []
    weather_rows: list[dict] = []
    soil_rows: list[dict] = []

    for row in candidates.itertuples(index=False):
        region = regions[str(row.district).strip()]
        try:
            composite = (
                landsat.filterBounds(region)
                .filterDate(row.satellite_start, end_exclusive(row.satellite_end))
                .map(mask_landsat)
                .median()
            )
            path = SATELLITE_DIR / f"{row.sample_id}.tif"
            fraction = download_patch(composite, region, path)
            valid = fraction >= MIN_VALID_PIXEL_FRACTION
            satellite_rows.append(
                {
                    "sample_id": row.sample_id,
                    "district": row.district,
                    "year": row.year,
                    "season": row.season,
                    "satellite_path": str(path.relative_to(ROOT)),
                    "height": PATCH_SIZE,
                    "width": PATCH_SIZE,
                    "bands": len(SATELLITE_COLUMNS),
                    "valid_pixel_fraction": fraction,
                    "satellite_patch_valid": valid,
                }
            )
            if not valid:
                record_failure(failures, row.sample_id, "satellite_patch_invalid", ValueError(f"valid fraction {fraction:.4f}"))
        except Exception as error:
            record_failure(failures, row.sample_id, "satellite", error)

        try:
            frame = extract_weather(era5, region, row.weather_start, row.weather_end)
            expected = (pd.Timestamp(row.weather_end) - pd.Timestamp(row.weather_start)).days + 1
            valid = (
                len(frame) == expected
                and frame["date"].is_monotonic_increasing
                and frame[["rainfall_mm", "temperature_c", "humidity_pct"]].notna().all().all()
                and frame["date"].dt.strftime("%Y-%m-%d").nunique() == expected
            )
            path = WEATHER_DIR / f"{row.sample_id}.csv"
            if valid:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False, date_format="%Y-%m-%d")
            weather_rows.append(
                {
                    "sample_id": row.sample_id,
                    "weather_path": str(path.relative_to(ROOT)),
                    "record_count": len(frame),
                    "expected_record_count": expected,
                    "weather_valid": valid,
                }
            )
            if not valid:
                record_failure(failures, row.sample_id, "weather", ValueError("weather completeness validation failed"))
        except Exception as error:
            record_failure(failures, row.sample_id, "weather", error)

    for district, region in regions.items():
        try:
            values = extract_soil(region)
            soil_rows.append({"district": district, **values, "source": "SoilGrids250m v2.0, 0-5 cm"})
        except Exception as error:
            record_failure(failures, district, "soil", error)

    satellite = pd.DataFrame(satellite_rows)
    weather = pd.DataFrame(weather_rows)
    soil = pd.DataFrame(soil_rows)
    satellite.to_csv(SATELLITE_MANIFEST, index=False)
    weather.to_csv(WEATHER_MANIFEST, index=False)
    soil.to_csv(SOIL_FEATURES, index=False)

    multimodal = (
        candidates[["sample_id", "district", "year", "season", "yield_kg_ha"]]
        .merge(satellite[["sample_id", "satellite_path", "satellite_patch_valid"]], on="sample_id", how="left")
        .merge(weather[["sample_id", "weather_path", "weather_valid"]], on="sample_id", how="left")
        .merge(soil[["district"]], on="district", how="left")
        .rename(columns={"district": "soil_district", "satellite_patch_valid": "satellite_valid"})
    )
    multimodal["soil_valid"] = multimodal["soil_district"].isin(soil["district"])
    multimodal["multimodal_valid"] = multimodal[["satellite_valid", "weather_valid", "soil_valid"]].all(axis=1)
    multimodal.to_csv(MULTIMODAL_MANIFEST, index=False)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(failures, columns=["sample_id", "stage", "status", "error"])
    log.to_csv(LOG_FILE, index=False)
    print(f"Final candidates: {len(candidates)}")
    print(f"Satellite files: {len(satellite)}")
    print(f"Weather files: {int(weather['weather_valid'].sum()) if not weather.empty else 0}")
    print(f"Soil districts: {len(soil)}")
    print(f"Final multimodal: {int(multimodal['multimodal_valid'].sum())}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
