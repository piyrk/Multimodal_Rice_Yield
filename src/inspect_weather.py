from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE = PROJECT_ROOT / Path(
    "data/raw/weather/Krishna_2010_Kharif_weather.csv"
)

df = pd.read_csv(FILE)

print("=" * 60)
print("WEATHER DATA INSPECTION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())