from pathlib import Path

import ee
import pandas as pd


PROJECT_ID = "poetic-world-441912-n4"
GAUL_DATASET = "FAO/GAUL/2015/level2"
ERA5_LAND_DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
SOILGRIDS_DATASET = "projects/soilgrids-isric/clay_mean"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_FILE = PROJECT_ROOT / "data/processed/master_samples.csv"
SATELLITE_FILE = PROJECT_ROOT / "data/processed/satellite_availability.csv"
WEATHER_OUTPUT = PROJECT_ROOT / "data/processed/weather_availability.csv"
SOIL_OUTPUT = PROJECT_ROOT / "data/processed/soil_availability.csv"
INDEX_OUTPUT = PROJECT_ROOT / "data/processed/final_multimodal_index.csv"

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


def load_districts() -> ee.FeatureCollection:
    return (
        ee.FeatureCollection(GAUL_DATASET)
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filter(ee.Filter.eq("ADM1_NAME", "Andhra Pradesh"))
    )


def district_region(
    districts: ee.FeatureCollection,
    district: str,
) -> ee.FeatureCollection:
    gaul_name = GAUL_NAME_MAP.get(district.upper())
    if gaul_name is None:
        raise ValueError(f"No GAUL mapping found for district: {district}")
    return districts.filter(ee.Filter.eq("ADM2_NAME", gaul_name))


def inclusive_end_exclusive(end_date: str) -> str:
    return (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")


def expected_daily_records(start_date: str, end_date: str) -> int:
    return int(
        (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    )


def get_weather_count(
    collection: ee.ImageCollection,
    region: ee.FeatureCollection,
    start_date: str,
    end_date: str,
) -> int:
    return int(
        collection
        .filterBounds(region)
        .filterDate(start_date, inclusive_end_exclusive(end_date))
        .size()
        .getInfo()
    )


def check_soil(
    soil_image: ee.Image,
    region: ee.FeatureCollection,
) -> bool:
    result = (
        soil_image
        .select("clay_0-5cm_mean")
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region.geometry(),
            scale=250,
            bestEffort=True,
            maxPixels=100000000,
        )
        .getInfo()
    )
    return int(result.get("clay_0-5cm_mean", 0) or 0) > 0


def main() -> None:
    for path in (MASTER_FILE, SATELLITE_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path}")

    master = pd.read_csv(MASTER_FILE)
    satellite = pd.read_csv(SATELLITE_FILE)
    required_master = {
        "sample_id",
        "district",
        "year",
        "season",
        "yield_kg_ha",
        "weather_start",
        "weather_end",
    }
    missing_master = required_master.difference(master.columns)
    if missing_master:
        raise ValueError(f"Missing master columns: {sorted(missing_master)}")

    required_satellite = {"sample_id", "satellite_available"}
    missing_satellite = required_satellite.difference(satellite.columns)
    if missing_satellite:
        raise ValueError(
            f"Missing satellite availability columns: {sorted(missing_satellite)}"
        )
    if len(master) != len(satellite) or set(master["sample_id"]) != set(
        satellite["sample_id"]
    ):
        raise ValueError("Master and satellite indexes do not contain the same samples.")

    ee.Initialize(project=PROJECT_ID)
    districts = load_districts()
    weather_collection = ee.ImageCollection(ERA5_LAND_DATASET)
    soil_image = ee.Image(SOILGRIDS_DATASET)

    region_cache = {}
    for district in master["district"].astype(str).str.strip().unique():
        region_cache[district] = district_region(districts, district)

    weather_rows = []
    print("=" * 70)
    print("ERA5-LAND WEATHER AVAILABILITY CHECK")
    print("=" * 70)
    print(f"Checking {len(master)} candidate samples...")

    for position, row in enumerate(master.itertuples(index=False), start=1):
        start_date = str(row.weather_start)
        end_date = str(row.weather_end)
        expected = expected_daily_records(start_date, end_date)
        try:
            count = get_weather_count(
                weather_collection,
                region_cache[str(row.district).strip()],
                start_date,
                end_date,
            )
            available = count == expected
        except Exception as error:
            print(f"ERROR: {row.sample_id}: {error}")
            count = 0
            available = False

        weather_rows.append(
            {
                "sample_id": row.sample_id,
                "district": row.district,
                "year": row.year,
                "season": row.season,
                "weather_start": start_date,
                "weather_end": end_date,
                "weather_record_count": count,
                "weather_available": available,
            }
        )
        print(
            f"[{position}/{len(master)}] {row.sample_id} -> "
            f"{count}/{expected} daily records"
        )

    weather = pd.DataFrame(weather_rows)
    WEATHER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_OUTPUT, index=False)

    print("\n" + "=" * 70)
    print("SOILGRIDS DISTRICT AVAILABILITY CHECK")
    print("=" * 70)
    soil_rows = []
    districts_in_master = sorted(master["district"].astype(str).str.strip().unique())
    for district in districts_in_master:
        try:
            available = check_soil(soil_image, region_cache[district])
        except Exception as error:
            print(f"ERROR: {district}: {error}")
            available = False
        soil_rows.append(
            {
                "district": district,
                "soil_available": available,
                "soil_source": SOILGRIDS_DATASET,
            }
        )
        print(f"{district} -> soil_available={available}")

    soil = pd.DataFrame(soil_rows)
    soil.to_csv(SOIL_OUTPUT, index=False)

    final_index = (
        master[
            [
                "sample_id",
                "district",
                "year",
                "season",
                "yield_kg_ha",
            ]
        ]
        .merge(
            satellite[["sample_id", "satellite_available"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            weather[["sample_id", "weather_available"]],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            soil[["district", "soil_available"]],
            on="district",
            how="left",
            validate="many_to_one",
        )
    )
    availability_columns = [
        "satellite_available",
        "weather_available",
        "soil_available",
    ]
    if final_index[availability_columns].isna().any().any():
        raise ValueError("Final index contains missing availability values.")
    final_index["final_sample_available"] = final_index[
        availability_columns
    ].all(axis=1)
    final_index.to_csv(INDEX_OUTPUT, index=False)

    satellite_count = int(final_index["satellite_available"].sum())
    weather_count = int(final_index["weather_available"].sum())
    soil_district_count = int(soil["soil_available"].sum())
    final_count = int(final_index["final_sample_available"].sum())

    print("\n" + "=" * 70)
    print("FINAL MULTIMODAL SAMPLE INDEX SUMMARY")
    print("=" * 70)
    print(f"Total candidate samples: {len(final_index)}")
    print(f"Satellite-available samples: {satellite_count}")
    print(f"Weather-available samples: {weather_count}")
    print(f"Soil-available districts: {soil_district_count}/{len(soil)}")
    print(f"Final multimodal samples: {final_count}")
    print(f"Excluded after satellite stage: {len(final_index) - satellite_count}")
    print(f"Excluded after weather stage: {len(final_index) - weather_count}")
    print(f"Excluded after soil stage: {len(final_index) - final_index['soil_available'].sum()}")
    print(f"Final percentage: {final_count / len(final_index) * 100:.2f}%")
    print("\nSaved:")
    print(WEATHER_OUTPUT)
    print(SOIL_OUTPUT)
    print(INDEX_OUTPUT)


if __name__ == "__main__":
    main()
