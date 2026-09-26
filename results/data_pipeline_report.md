# PyTorch Data Pipeline Report

## Dataset

- Total validated training samples: **239**
- Source: `data/processed/final_training_samples.csv`
- Each sample is linked through `multimodal_manifest.csv` to one satellite patch,
  one daily weather sequence, one district-level soil record, and one yield target.
- Target: rice yield in **kg/ha**.

## Split Methodology

The split is chronological and year-grouped to avoid temporal leakage. All
samples from the same agricultural year label remain in one split:

- **Train: 172 samples** — 1998-99 through 2008-09
- **Validation: 26 samples** — 2009-10
- **Test: 41 samples** — 2010-11 and 2011-12

The split is not a random row split. This is preferable for this dataset because
the target is district-season-year yield and adjacent observations can share
temporal and district-level conditions. Holding out the latest year groups
provides a forward-time evaluation. The split assignments are stored in
`data/processed/dataset_splits.csv`.

## Tensor Shapes

- Satellite: `FloatTensor [7, 128, 128]` per sample
- Weather: `FloatTensor [T, 3]` per sample, where `T` is the number of daily
  records in that sample's season window
- Soil: `FloatTensor [4]` per sample
- Target: scalar `FloatTensor`

The DataLoader collate function right-pads weather sequences with zeros so
variable-length daily sequences can be batched. It also returns
`weather_mask`, which identifies real observations versus padding.

Satellite GeoTIFFs are loaded with rasterio. If a file is represented as
`[128, 128, 7]`, it is transposed in memory to `[7, 128, 128]`. The current
files are validated for seven bands, 128x128 spatial dimensions, and float32
storage.

## Normalization Status

No statistical normalization or standardization has been applied. Values are
returned in their extracted physical units:

- Satellite: scaled Landsat surface-reflectance bands and NDVI
- Weather: rainfall in mm, temperature in degrees Celsius, humidity in percent
- Soil: nitrogen in g/kg, pH, SOC in g/kg, clay in percent
- Target: yield in kg/ha

Normalization should be fitted on the training split only in a later modeling
step. No model or training code was created here.

## Validation Checks

The pipeline checks:

- Missing satellite and weather files
- Satellite dimensions, band count, and float32 dtype
- Satellite finite values after loading
- Weather required columns and complete chronological daily date range
- Weather finite values
- Soil record presence and four finite soil features
- Numeric finite yield target
- Duplicate sample IDs and manifest membership
- NaN/Inf absence in every DataLoader batch
- Successful loading of every batch in train, validation, and test loaders

Some accepted satellite patches contain NaN nodata pixels from QA cloud
masking. These are detected during loading and converted to zero **in memory
only**; the raw GeoTIFFs are unchanged. The manifest's valid-pixel threshold
remains the authority for whether a patch has enough usable pixels. This
handling prevents masked pixels from entering PyTorch tensors as NaN/Inf while
avoiding fabricated changes to the source files.

## Smoke-Test Result

`src/test_dataloader.py` passed:

```text
train count = 172
validation count = 26
test count = 41
satellite shape = (8, 7, 128, 128)
weather shape = (8, 183, 3)
soil shape = (8, 4)
target shape = (8,)
dtypes = satellite:torch.float32, weather:torch.float32, soil:torch.float32, target:torch.float32
NaN/Inf check = PASSED
every batch load = PASSED
```

## Remaining Limitations

- Weather sequences have different seasonal lengths; batching uses zero
  padding plus a mask.
- Soil is static at district level and is reused across samples from the same
  district.
- Satellite patches are district-centroid spatial proxies, not field-level
  observations.
- The chronological validation and test sets contain fewer samples than the
  training set because the source covers only 13 agricultural year labels.
- No normalization, model architecture, or model training has been performed.
