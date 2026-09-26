from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE = PROJECT_ROOT / Path(
    "data/raw/soil/Krishna_soil.csv"
)


def main():
    if not FILE.exists():
        raise FileNotFoundError(
            f"File not found: {FILE}"
        )

    df = pd.read_csv(FILE)

    print("=" * 60)
    print("SOIL DATA INSPECTION")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData:")
    print(df.to_string(index=False))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nStatistics:")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()