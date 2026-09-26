# Concatenation Fusion Hyperparameter Tuning

Validation-only tuning was performed for exactly four configurations and three seeds per configuration. The test loader was never created or evaluated.

Device: `cuda`
Split: 172 train / 26 validation / 41 held-out test (test unused)
Seeds: 42, 123, 2026
Optimizer/loss: AdamW / MSELoss
Batch size: 16
Maximum epochs: 12
Early-stopping patience: 3
Target normalization: train targets only

## Per-run validation results

| Configuration | Seed | Best epoch | Best validation loss | Validation MAE | Validation RMSE | Validation R2 | Time (s) | Device |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| lr_0.001_dropout_0 | 42 | 1 | 0.989828 | 610.123 | 755.123 | 0.025356 | 25.3 | cuda |
| lr_0.001_dropout_0 | 123 | 6 | 0.654309 | 448.991 | 614.755 | 0.354027 | 47.9 | cuda |
| lr_0.001_dropout_0 | 2026 | 7 | 0.609161 | 454.744 | 590.185 | 0.404630 | 51.4 | cuda |
| lr_0.001_dropout_0.1 | 42 | 1 | 1.019779 | 637.335 | 774.058 | -0.024136 | 22.3 | cuda |
| lr_0.001_dropout_0.1 | 123 | 9 | 0.501863 | 418.004 | 537.273 | 0.506598 | 66.5 | cuda |
| lr_0.001_dropout_0.1 | 2026 | 6 | 0.699121 | 533.812 | 647.324 | 0.283769 | 45.4 | cuda |
| lr_0.0005_dropout_0 | 42 | 1 | 1.063317 | 594.256 | 771.237 | -0.016685 | 15.3 | cuda |
| lr_0.0005_dropout_0 | 123 | 9 | 0.468042 | 408.737 | 550.611 | 0.481797 | 46.6 | cuda |
| lr_0.0005_dropout_0 | 2026 | 9 | 0.436015 | 406.318 | 519.533 | 0.538644 | 50.1 | cuda |
| lr_0.0005_dropout_0.1 | 42 | 1 | 1.151096 | 600.128 | 792.411 | -0.073276 | 19.6 | cuda |
| lr_0.0005_dropout_0.1 | 123 | 8 | 0.476498 | 401.787 | 533.830 | 0.512901 | 48.3 | cuda |
| lr_0.0005_dropout_0.1 | 2026 | 5 | 0.657168 | 467.960 | 603.268 | 0.377943 | 37.0 | cuda |

## Aggregate validation results

| Configuration | Mean loss +/- std | Mean MAE +/- std | Mean RMSE +/- std | Mean R2 +/- std |
|---|---:|---:|---:|---:|
| lr_0.0005_dropout_0 | 0.655791 +/- 0.353290 | 469.770 +/- 107.814 | 613.794 +/- 137.233 | 0.334585 +/- 0.305534 |
| lr_0.001_dropout_0.1 | 0.740254 +/- 0.261396 | 529.717 +/- 109.723 | 652.885 +/- 118.490 | 0.255410 +/- 0.266501 |
| lr_0.001_dropout_0 | 0.751099 +/- 0.207974 | 504.619 +/- 91.414 | 653.355 +/- 88.986 | 0.261338 +/- 0.205926 |
| lr_0.0005_dropout_0.1 | 0.761587 +/- 0.349211 | 489.958 +/- 100.983 | 643.170 +/- 133.829 | 0.272523 +/- 0.306979 |

## Selection

Selected configuration: `lr_0.0005_dropout_0` (learning rate 0.0005, fusion dropout 0).
It was selected because it had the lowest mean validation loss (0.655791) across the three seeds. Mean validation MAE is reported as a secondary descriptive metric; it was not the primary selection criterion.

No test metrics were calculated, inspected, or used for selection. Final retraining and held-out test evaluation were intentionally not performed.