import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILE = PROJECT_ROOT / "data/processed/master_samples.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("MASTER DATASET ANALYSIS")
print("=" * 70)

print("\nTotal samples:", len(df))

print("\nUnique districts:", df["district"].nunique())

print("\nUnique years:", df["year"].nunique())

print("\nSamples by season:")
print(df["season"].value_counts())

print("\nSamples by district:")
print(
    df["district"]
    .value_counts()
    .sort_index()
)

print("\nSamples by year:")
print(
    df["year"]
    .value_counts()
    .sort_index()
)

print("\nFirst 20 samples:")
print(
    df.head(20).to_string(index=False)
)