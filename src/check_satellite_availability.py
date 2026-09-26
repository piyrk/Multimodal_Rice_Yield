import ee
import pandas as pd
from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ID = "poetic-world-441912-n4"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = PROJECT_ROOT / "data/processed/master_samples.csv"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/satellite_availability.csv"


# ==========================================
# DISTRICT NAME MAPPING
# Our CSV -> GAUL Earth Engine
# ==========================================

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


# ==========================================
# GET LANDSAT IMAGE COUNT
# ==========================================

def get_landsat_count(
    district: str,
    start_date: str,
    end_date: str
) -> int:

    # Convert our dataset district name
    # to the Earth Engine/GAUL name
    gaul_name = GAUL_NAME_MAP.get(
        district.upper()
    )

    if gaul_name is None:
        raise ValueError(
            f"No GAUL mapping found for district: {district}"
        )

    # Load district boundaries
    districts = ee.FeatureCollection(
        "FAO/GAUL/2015/level2"
    )

    # Find the correct district
    region = (
        districts
        .filter(
            ee.Filter.eq(
                "ADM0_NAME",
                "India"
            )
        )
        .filter(
            ee.Filter.eq(
                "ADM1_NAME",
                "Andhra Pradesh"
            )
        )
        .filter(
            ee.Filter.eq(
                "ADM2_NAME",
                gaul_name
            )
        )
    )

    # Get Landsat 5 scenes
    # Earth Engine's filterDate end is exclusive, while the master table
    # stores inclusive season end dates.
    end_date_exclusive = (
        pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection(
            "LANDSAT/LT05/C02/T1_L2"
        )
        .filterBounds(region)
        .filterDate(
            start_date,
            end_date_exclusive
        )
        .filter(
            ee.Filter.lt(
                "CLOUD_COVER",
                50
            )
        )
    )

    return collection.size().getInfo()


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 70)
    print("LANDSAT AVAILABILITY CHECK")
    print("=" * 70)

    # Initialize Earth Engine
    ee.Initialize(
        project=PROJECT_ID
    )

    # Load master dataset
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Master dataset not found: {MASTER_FILE}"
        )

    df = pd.read_csv(
        MASTER_FILE
    )

    results = []

    total = len(df)

    print(
        f"\nChecking {total} samples...\n"
    )

    # --------------------------------------
    # Check every sample
    # --------------------------------------

    for i, row in df.iterrows():

        district = str(
            row["district"]
        ).strip()

        start_date = str(
            row["satellite_start"]
        )

        end_date = str(
            row["satellite_end"]
        )

        try:

            count = get_landsat_count(
                district,
                start_date,
                end_date
            )

            available = count > 0

        except Exception as e:

            print(
                f"\nERROR: {row['sample_id']}"
            )

            print(e)

            count = 0
            available = False

        results.append({
            "sample_id":
                row["sample_id"],

            "district":
                district,

            "year":
                row["year"],

            "season":
                row["season"],

            "yield_kg_ha":
                row["yield_kg_ha"],

            "satellite_start":
                start_date,

            "satellite_end":
                end_date,

            "landsat_image_count":
                count,

            "satellite_available":
                available
        })

        print(
            f"[{i + 1}/{total}] "
            f"{row['sample_id']} -> "
            f"{count} images"
        )

    # --------------------------------------
    # Save results
    # --------------------------------------

    result_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------
    # Summary
    # --------------------------------------

    available_count = int(
        result_df[
            "satellite_available"
        ].sum()
    )

    unavailable_count = (
        len(result_df)
        - available_count
    )

    print("\n" + "=" * 70)
    print("SATELLITE AVAILABILITY SUMMARY")
    print("=" * 70)

    print(
        "Total samples:",
        len(result_df)
    )

    print(
        "Available:",
        available_count
    )

    print(
        "Unavailable:",
        unavailable_count
    )

    print(
        "Availability percentage:",
        f"{available_count / len(result_df) * 100:.2f}%"
    )

    print("\nAvailability by district:")
    district_summary = (
        result_df
        .groupby("district", sort=True)
        .agg(
            total_samples=("sample_id", "size"),
            available_samples=("satellite_available", "sum"),
        )
    )
    district_summary["unavailable_samples"] = (
        district_summary["total_samples"]
        - district_summary["available_samples"]
    )
    district_summary["availability_percentage"] = (
        district_summary["available_samples"]
        / district_summary["total_samples"]
        * 100
    ).round(2)
    print(district_summary.to_string())

    print("\nAvailability by season:")
    season_summary = (
        result_df
        .groupby("season", sort=True)
        .agg(
            total_samples=("sample_id", "size"),
            available_samples=("satellite_available", "sum"),
        )
    )
    season_summary["unavailable_samples"] = (
        season_summary["total_samples"]
        - season_summary["available_samples"]
    )
    season_summary["availability_percentage"] = (
        season_summary["available_samples"]
        / season_summary["total_samples"]
        * 100
    ).round(2)
    print(season_summary.to_string())

    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()