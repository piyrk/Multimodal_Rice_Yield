import ee

PROJECT_ID = "poetic-world-441912-n4"

ee.Initialize(project=PROJECT_ID)

districts = (
    ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq("ADM0_NAME", "India"))
    .filter(ee.Filter.eq("ADM1_NAME", "Andhra Pradesh"))
)

names = (
    districts
    .aggregate_array("ADM2_NAME")
    .getInfo()
)

print("=" * 60)
print("GAUL DISTRICT NAMES")
print("=" * 60)

for name in sorted(names):
    print(name)