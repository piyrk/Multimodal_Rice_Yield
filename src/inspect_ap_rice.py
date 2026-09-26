import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data/raw/yield/India_Agri_Data/agridata.csv"

df = pd.read_csv(INPUT)

# Clean text columns
df["state"] = df["state"].astype(str).str.strip()
df["district"] = df["district"].astype(str).str.strip()
df["crop"] = df["crop"].astype(str).str.strip()
df["season"] = df["season"].astype(str).str.strip()
df["year"] = df["year"].astype(str).str.strip()

# Andhra Pradesh + Rice
ap_rice = df[
    (df["state"].str.upper() == "ANDHRA PRADESH")
    & (df["crop"].str.lower() == "rice")
].copy()

# Numeric conversion
ap_rice["area"] = pd.to_numeric(ap_rice["area"], errors="coerce")
ap_rice["production"] = pd.to_numeric(
    ap_rice["production"],
    errors="coerce"
)

# Calculate yield in kg/ha
ap_rice["yield_kg_ha"] = (
    ap_rice["production"] * 1000 / ap_rice["area"]
)

print("=" * 60)
print("ANDHRA PRADESH - RICE DATA")
print("=" * 60)

print("\nRows:", len(ap_rice))

print("\nYears:")
print(sorted(ap_rice["year"].unique()))

print("\nSeasons:")
print(sorted(ap_rice["season"].unique()))

print("\nNumber of districts:")
print(ap_rice["district"].nunique())

print("\nDistricts:")
for district in sorted(ap_rice["district"].unique()):
    count = (ap_rice["district"] == district).sum()
    print(f"{district:25} {count} rows")

print("\nMissing values:")
print(ap_rice.isna().sum())

print("\nYield statistics (kg/ha):")
print(
    ap_rice["yield_kg_ha"].describe()
)

print("\nFirst 20 rows:")
print(
    ap_rice[
        [
            "district",
            "year",
            "season",
            "area",
            "production",
            "yield_kg_ha",
        ]
    ].head(20).to_string(index=False)
)

# Save candidate dataset
output = PROJECT_ROOT / "data/processed/ap_rice_candidate.csv"
output.parent.mkdir(parents=True, exist_ok=True)

ap_rice[
    [
        "state",
        "district",
        "year",
        "season",
        "area",
        "production",
        "yield_kg_ha",
    ]
].to_csv(output, index=False)

print("\nSaved:")
print(output)