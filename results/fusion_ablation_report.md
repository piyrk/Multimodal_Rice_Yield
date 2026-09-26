# Controlled Fusion Ablation

This experiment compares the existing Satellite-only baseline, the unchanged attention fusion model, and a new concatenation fusion model.

## Controlled protocol

- Exact chronological split: 172 train, 26 validation, 41 test
- Seeds: 42, 123, 2026
- Train-only target normalization
- AdamW, learning rate 0.001, MSELoss
- Batch size 16, maximum 8 epochs, early-stopping patience 3
- Device: `NVIDIA GeForce RTX 3050 Laptop GPU`

The satellite CNN, weather LSTM, and soil MLP branch architectures are unchanged. The concatenation variant concatenates the 128-, 128-, and 64-dimensional branch embeddings (320 total), then applies an MLP fusion block and regression head. It has no attention or learned modality weighting.

## Per-seed results

| Model | Seed | Best epoch | Validation loss | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| satellite_only | 42 | 1 | 1.266338 | 749.029 | 982.673 | -0.118387 |
| satellite_only | 123 | 1 | 1.170963 | 803.020 | 1020.017 | -0.205006 |
| satellite_only | 2026 | 5 | 0.521552 | 696.568 | 861.472 | 0.140480 |
| attention_fusion | 42 | 1 | 1.015385 | 735.332 | 924.011 | 0.011154 |
| attention_fusion | 123 | 4 | 0.877181 | 580.012 | 726.745 | 0.388301 |
| attention_fusion | 2026 | 6 | 0.602399 | 533.202 | 668.722 | 0.482077 |
| concatenation_fusion | 42 | 1 | 1.005203 | 737.401 | 932.439 | -0.006965 |
| concatenation_fusion | 123 | 4 | 0.673978 | 631.996 | 813.446 | 0.233642 |
| concatenation_fusion | 2026 | 8 | 0.650367 | 635.713 | 788.836 | 0.279311 |

## Three-seed mean +/- standard deviation

| Model | MAE (kg/ha) | RMSE (kg/ha) | R2 |
|---|---:|---:|---:|
| satellite_only | 749.539 +/- 43.460 | 954.721 +/- 67.676 | -0.060971 +/- 0.146771 |
| attention_fusion | 616.182 +/- 86.392 | 773.159 +/- 109.267 | 0.293844 +/- 0.203525 |
| concatenation_fusion | 668.370 +/- 48.836 | 844.907 +/- 62.704 | 0.168663 +/- 0.125580 |

## Conclusion

Under this three-seed controlled experiment, concatenation fusion does not improve the full attention architecture across all three reported mean metrics. This conclusion is limited to the fixed split and configuration; no general hyperparameter tuning was performed.