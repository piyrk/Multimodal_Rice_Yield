import ee

PROJECT_ID = "poetic-world-441912-n4"

def main():
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialized successfully.")

    count = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .limit(1)
        .size()
        .getInfo()
    )
    if count != 1:
        raise RuntimeError("No Landsat 5 scenes returned by the access check.")

    print("Earth Engine dataset access is working (metadata query only).")


if __name__ == "__main__":
    main()