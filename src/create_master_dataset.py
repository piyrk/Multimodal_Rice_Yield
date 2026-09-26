from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / Path(
    "data/processed/final_yield_ap_rice.csv"
)

OUTPUT = PROJECT_ROOT / Path(
    "data/processed/master_samples.csv"
)


def get_year_start(year_string: str) -> int:
    """
    Example:
        2010-11 -> 2010
    """
    return int(year_string.split("-")[0])


def add_season_dates(row):
    # End dates are inclusive; Earth Engine queries must add one day.
    year_start = get_year_start(row["year"])

    if row["season"].strip().lower() == "kharif":
        satellite_start = f"{year_start}-06-01"
        satellite_end = f"{year_start}-11-30"

        weather_start = satellite_start
        weather_end = satellite_end

    elif row["season"].strip().lower() == "rabi":
        satellite_start = f"{year_start}-11-01"
        satellite_end = f"{year_start + 1}-04-30"

        weather_start = satellite_start
        weather_end = satellite_end

    else:
        raise ValueError(
            f"Unknown season: {row['season']}"
        )

    return pd.Series(
        {
            "satellite_start": satellite_start,
            "satellite_end": satellite_end,
            "weather_start": weather_start,
            "weather_end": weather_end,
        }
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    # Clean text
    for col in [
        "district",
        "year",
        "season"
    ]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    # Remove invalid yield records
    df["yield_kg_ha"] = pd.to_numeric(
        df["yield_kg_ha"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["yield_kg_ha"]
    )

    df = df[
        df["yield_kg_ha"] > 0
    ].copy()

    # Create unique ID
    df["sample_id"] = (
        "AP_"
        + df["district"].str.replace(
            " ",
            "_",
            regex=False
        )
        + "_"
        + df["year"].str.replace(
            "-",
            "_",
            regex=False
        )
        + "_"
        + df["season"].str.capitalize()
    )

    # Add season-specific time windows
    date_info = df.apply(
        add_season_dates,
        axis=1
    )

    df = pd.concat(
        [df, date_info],
        axis=1
    )

    # Keep only what we need for the master table
    master = df[
        [
            "sample_id",
            "district",
            "year",
            "season",
            "yield_kg_ha",
            "satellite_start",
            "satellite_end",
            "weather_start",
            "weather_end",
        ]
    ].copy()

    master = master.sort_values(
        [
            "year",
            "district",
            "season"
        ]
    )

    # Check duplicate sample IDs
    duplicates = master[
        master["sample_id"].duplicated(
            keep=False
        )
    ]

    if len(duplicates) > 0:
        raise ValueError(
            "Duplicate sample IDs found:\n"
            + duplicates.to_string()
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    master.to_csv(
        OUTPUT,
        index=False
    )

    print("=" * 70)
    print("MASTER DATASET")
    print("=" * 70)

    print("\nTotal samples:", len(master))

    print(
        "Unique districts:",
        master["district"].nunique()
    )

    print(
        "Unique years:",
        master["year"].nunique()
    )

    print("\nSamples by season:")
    print(
        master["season"].value_counts()
    )

    print("\nSamples by district:")
    print(
        master["district"]
        .value_counts()
        .sort_index()
    )

    print("\nDate range:")
    print(
        master["satellite_start"].min(),
        "to",
        master["satellite_end"].max()
    )

    print("\nFirst 10 samples:")
    print(
        master.head(10)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()