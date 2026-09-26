import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / Path(
    "data/raw/yield/India_Agri_Data/agridata.csv"
)

OUTPUT = PROJECT_ROOT / Path(
    "data/processed/final_yield_ap_rice.csv"
)

# Districts from the old combined AP dataset
# that belong to Andhra Pradesh, using the historical 13-district boundaries.
AP_DISTRICTS = {
    "ANANTPUR",
    "CHITTOOR",
    "CUDDAPAH",
    "EAST GODAVARI",
    "GUNTUR",
    "KRISHNA",
    "KURNOOL",
    "NELLORE",
    "PRAKASAM",
    "SRIKAKULAM",
    "VISAKHAPATNAM",
    "VIZIANAGARM",
    "WEST GODAVARI",
}


def main():
    df = pd.read_csv(INPUT)

    # Clean text
    for col in ["state", "district", "crop", "season", "year"]:
        df[col] = df[col].astype(str).str.strip()

    # Rice records from the old Andhra Pradesh dataset
    df = df[
        (df["state"].str.upper() == "ANDHRA PRADESH")
        & (df["crop"].str.lower() == "rice")
        & (df["district"].isin(AP_DISTRICTS))
    ].copy()

    # Numeric conversion
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    df["production"] = pd.to_numeric(
        df["production"],
        errors="coerce"
    )

    # Calculate kg/ha ourselves
    df["yield_kg_ha"] = (
        df["production"] * 1000 / df["area"]
    )

    # Keep useful columns
    df = df[
        [
            "state",
            "district",
            "year",
            "season",
            "area",
            "production",
            "yield_kg_ha",
        ]
    ].sort_values(
        ["year", "district", "season"]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("=" * 60)
    print("FINAL AP RICE YIELD DATASET")
    print("=" * 60)

    print("Rows:", len(df))
    print("Districts:", df["district"].nunique())

    print("\nDistricts:")
    print(sorted(df["district"].unique()))

    print("\nYears:")
    print(sorted(df["year"].unique()))

    print("\nSeasons:")
    print(sorted(df["season"].unique()))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSaved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()