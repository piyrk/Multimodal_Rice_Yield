from pathlib import Path
import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SATELLITE_DIR = PROJECT_ROOT / Path(
    "data/raw/satellite/Krishna_2010_Kharif"
)

BANDS = [
    "download.SR_B1.tif",
    "download.SR_B2.tif",
    "download.SR_B3.tif",
    "download.SR_B4.tif",
    "download.SR_B5.tif",
    "download.SR_B7.tif",
    "download.NDVI.tif",
]


def main():
    arrays = []

    print("=" * 60)
    print("SATELLITE DATA INSPECTION")
    print("=" * 60)

    for band_name in BANDS:
        path = SATELLITE_DIR / band_name

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with rasterio.open(path) as src:
            data = src.read(1)

            print(f"\n{band_name}")
            print("  Width :", src.width)
            print("  Height:", src.height)
            print("  CRS   :", src.crs)
            print("  Dtype :", data.dtype)
            print("  Min   :", np.nanmin(data))
            print("  Max   :", np.nanmax(data))
            print("  Mean  :", np.nanmean(data))

            arrays.append(data.astype(np.float32))

    # Verify all bands have the same dimensions
    shapes = [arr.shape for arr in arrays]

    if len(set(shapes)) != 1:
        raise ValueError(f"Band dimensions do not match: {shapes}")

    # Stack:
    # (height, width, 7)
    image_tensor = np.stack(arrays, axis=-1)

    print("\n" + "=" * 60)
    print("FINAL ML TENSOR")
    print("=" * 60)

    print("Shape:", image_tensor.shape)
    print("Dtype:", image_tensor.dtype)

    output = PROJECT_ROOT / Path(
        "data/processed/"
        "Krishna_2010_Kharif_satellite.npy"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(output, image_tensor)

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()