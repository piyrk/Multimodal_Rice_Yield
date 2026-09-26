from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT = PROJECT_ROOT / Path(
    "data/raw/weather/Krishna_2010_Kharif_weather.csv"
)

OUTPUT = PROJECT_ROOT / Path(
    "data/processed/Krishna_2010_Kharif_weather.npy"
)


def main():
    df = pd.read_csv(INPUT)

    print("=" * 60)
    print("WEATHER → LSTM PREPARATION")
    print("=" * 60)

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Required features
    features = [
        "rainfall_mm",
        "temperature_c",
        "humidity_pct"
    ]

    # Check columns
    missing_columns = [
        col for col in features
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check missing values
    print("\nMissing values:")
    print(df[features].isna().sum())

    # Keep only model features
    weather = df[features].copy()

    # Convert to float32
    sequence = weather.to_numpy(
        dtype=np.float32
    )

    print("\nDate range:")
    print(df["date"].min(), "to", df["date"].max())

    print("\nNumber of days:", len(df))

    print("\nLSTM sequence shape:")
    print(sequence.shape)

    # Save raw sequence
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(OUTPUT, sequence)

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()