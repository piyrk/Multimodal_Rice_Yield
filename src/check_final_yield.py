import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE = PROJECT_ROOT / "data/processed/final_yield_ap_rice.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("FINAL YIELD DATASET CHECK")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDistrict count:")
print(df["district"].nunique())

print("\nDistricts:")
print(sorted(df["district"].unique()))

print("\nYears:")
print(sorted(df["year"].unique()))

print("\nSeasons:")
print(sorted(df["season"].unique()))

print("\nRows per season:")
print(df["season"].value_counts())

print("\nRows per district:")
print(df["district"].value_counts().sort_index())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nYield statistics (kg/ha):")
print(df["yield_kg_ha"].describe())

print("\nSample rows:")
print(df.head(15).to_string(index=False))