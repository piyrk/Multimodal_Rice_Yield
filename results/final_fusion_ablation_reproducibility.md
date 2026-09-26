# Unified Fusion Reproducibility Rerun

All nine runs use one implementation path for model construction, seeding, DataLoader creation, training, validation checkpoint selection, and test evaluation.

Device: `NVIDIA GeForce RTX 3050 Laptop GPU`
Seeds: 42, 123, 2026
Split: exact existing chronological split, 172 train / 26 validation / 41 test
Training: AdamW, MSELoss, batch size 16, learning rate 0.001, maximum 8 epochs, patience 3
Normalization: target mean and standard deviation from train samples only

## Per-run results

| Model | Seed | Best epoch | Best validation loss | Test count | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| satellite_only | 42 | 1 | 1.337022 | 41 | 797.583 | 1036.436 | -0.244110 |
| satellite_only | 123 | 1 | 1.154927 | 41 | 797.091 | 1012.981 | -0.188437 |
| satellite_only | 2026 | 7 | 0.498589 | 41 | 629.228 | 809.657 | 0.240764 |
| attention_fusion | 42 | 1 | 1.013343 | 41 | 734.901 | 923.799 | 0.011608 |
| attention_fusion | 123 | 2 | 0.991878 | 41 | 728.721 | 928.041 | 0.002510 |
| attention_fusion | 2026 | 4 | 0.683679 | 41 | 655.115 | 827.355 | 0.207211 |
| concatenation_fusion | 42 | 1 | 1.004141 | 41 | 738.414 | 933.351 | -0.008938 |
| concatenation_fusion | 123 | 8 | 0.572357 | 41 | 592.980 | 759.616 | 0.331715 |
| concatenation_fusion | 2026 | 7 | 0.667908 | 41 | 611.412 | 786.259 | 0.284012 |

## Verification

- Exactly nine runs: `True`
- Exactly three models and three seeds: `True`
- Every run has 41 test predictions: `True`
- All metrics are finite: `True`
- Same test IDs in every run: `True`

## Cross-run audit conclusion

The earlier discrepancy is not resolved by comparing the old outputs alone because those runners duplicated execution logic. This unified rerun is the controlled comparison to use going forward. Its results should be compared by the per-run rows above, not mixed with the previous runner outputs.