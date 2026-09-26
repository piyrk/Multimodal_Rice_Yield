# Final Dataset Integrity Report

This report was generated from the 273 rows marked `final_sample_available=True` in `final_multimodal_index.csv`. Counts are row-level intersections after checking the on-disk files and modality contents.

## Counts

- Candidate samples: 273
- Satellite complete count: 240
- Weather complete count: 270
- Soil complete count: 273
- Yield complete count: 273
- Complete all-four-modalities count: 239
- Excluded sample count: 34
- Duplicate sample IDs: 0

## Exclusion Reason Counts

- satellite_missing: 4
- satellite_valid_fraction: 29
- weather_missing: 3

## Integrity Checks

- Satellite shape consistency: every complete satellite is 7 bands, 128x128, float32.
- Satellite valid-pixel threshold: 0.05; 240 samples passed.
- Weather completeness: expected daily date range, chronological order, required columns, and non-missing values checked for every candidate.
- Weather complete: 270/273.
- Soil coverage: 273/273 samples; 13 district records available.
- Yield coverage: 273/273 numeric finite values.
- Missing modality rows: 34.

## Repaired Outputs

- `data/processed/final_training_samples.csv` now contains only the complete intersection.
- `data/processed/multimodal_manifest.csv` flags are based on the independent checks above.
