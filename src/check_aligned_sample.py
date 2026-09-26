from pathlib import Path
import pandas as pd
import numpy as np


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

YIELD_FILE = PROJECT_ROOT / Path(
    "data/processed/final_yield_ap_rice.csv"
)

SATELLITE_FILE = PROJECT_ROOT / Path(
    "data/processed/Krishna_2010_Kharif_satellite.npy"
)

WEATHER_FILE = PROJECT_ROOT / Path(
    "data/raw/weather/Krishna_2010_Kharif_weather.csv"
)

SOIL_FILE = PROJECT_ROOT / Path(
    "data/raw/soil/Krishna_soil.csv"
)


DISTRICT = "KRISHNA"
YEAR = "2010-11"
SEASON = "Kharif"


def main():

    print("=" * 70)
    print("ALIGNED MULTIMODAL SAMPLE CHECK")
    print("=" * 70)

    # ==========================================
    # 1. LOAD YIELD
    # ==========================================

    yield_df = pd.read_csv(YIELD_FILE)

    yield_row = yield_df[
        (yield_df["district"].str.upper() == DISTRICT)
        & (yield_df["year"] == YEAR)
        & (yield_df["season"].str.lower() == SEASON.lower())
    ]

    if len(yield_row) != 1:
        raise ValueError(
            f"Expected exactly 1 yield record, "
            f"found {len(yield_row)}"
        )

    target_yield = float(
        yield_row.iloc[0]["yield_kg_ha"]
    )

    print("\nYIELD")
    print("District:", DISTRICT)
    print("Year:", YEAR)
    print("Season:", SEASON)
    print("Yield:", target_yield, "kg/ha")


    # ==========================================
    # 2. LOAD SATELLITE
    # ==========================================

    if not SATELLITE_FILE.exists():
        raise FileNotFoundError(
            f"Satellite file not found: {SATELLITE_FILE}"
        )

    satellite = np.load(SATELLITE_FILE)

    print("\nSATELLITE")
    print("Shape:", satellite.shape)
    print("Datatype:", satellite.dtype)

    if satellite.ndim != 3:
        raise ValueError(
            "Satellite tensor should be 3-dimensional."
        )

    if satellite.shape[-1] != 7:
        raise ValueError(
            "Expected 7 satellite bands."
        )


    # ==========================================
    # 3. LOAD WEATHER
    # ==========================================

    weather = pd.read_csv(WEATHER_FILE)

    weather["date"] = pd.to_datetime(
        weather["date"]
    )

    print("\nWEATHER")
    print("Rows:", len(weather))
    print(
        "Date:",
        weather["date"].min().date(),
        "to",
        weather["date"].max().date()
    )

    weather_features = [
        "rainfall_mm",
        "temperature_c",
        "humidity_pct"
    ]

    missing_weather = [
        col for col in weather_features
        if col not in weather.columns
    ]

    if missing_weather:
        raise ValueError(
            f"Missing weather columns: {missing_weather}"
        )

    print(
        "Weather feature matrix:",
        weather[weather_features].shape
    )


    # ==========================================
    # 4. LOAD SOIL
    # ==========================================

    soil = pd.read_csv(SOIL_FILE)

    print("\nSOIL")
    print(
        soil[
            [
                "district",
                "soil_depth",
                "nitrogen",
                "ph",
                "soc",
                "clay"
            ]
        ].to_string(index=False)
    )

    soil_features = [
        "nitrogen",
        "ph",
        "soc",
        "clay"
    ]

    if soil[soil_features].isna().any().any():
        raise ValueError(
            "Missing soil values."
        )


    # ==========================================
    # 5. FINAL CHECK
    # ==========================================

    print("\n" + "=" * 70)
    print("ALIGNMENT RESULT")
    print("=" * 70)

    print("Yield found")
    print("Satellite found")
    print("Weather found")
    print("Soil found")

    print("\nObservation:")

    print(
        f"{DISTRICT} | {YEAR} | {SEASON}"
    )

    print(
        "\nSatellite:",
        satellite.shape
    )

    print(
        "Weather:",
        weather[weather_features].shape
    )

    print(
        "Soil:",
        soil[soil_features].shape
    )

    print(
        "Target:",
        target_yield,
        "kg/ha"
    )

    print(
        "\nFIRST MULTIMODAL SAMPLE IS ALIGNED!"
    )


if __name__ == "__main__":
    main()