# Final Held-Out Evaluation

The locked concatenation-fusion configuration was retrained from scratch for three seeds using the combined train and validation samples. All three checkpoints were saved before the test loader was evaluated.

- Device: `cuda`
- Training samples: **198**
- Test samples: **41**
- Fixed epochs: **9** (median of selected-configuration best epochs 1, 9, and 9)
- Learning rate: **0.0005**
- Fusion dropout: **0.0**
- Optimizer/loss: **AdamW / MSELoss**
- Target normalization: train+validation targets only (mean=2962.227140, std=800.780155)

## Per-seed results

| Seed | MAE (kg/ha) | RMSE (kg/ha) | R² | Min prediction | Max prediction | Mean prediction |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 512.187 | 660.994 | 0.493979 | 1964.696 | 3996.095 | 3033.837 |
| 123 | 622.524 | 806.461 | 0.246747 | 2284.581 | 3900.123 | 2816.684 |
| 2026 | 638.815 | 807.604 | 0.244611 | 2318.327 | 4104.426 | 2930.785 |

## Three-seed aggregate

- Test MAE: **591.175 +/- 68.889 kg/ha**
- Test RMSE: **758.353 +/- 84.317 kg/ha**
- Test R²: **0.328446 +/- 0.143360**

Test predictions and per-sample errors are stored in `results/final_test_predictions.csv`. The test set was not used for configuration selection, epoch selection, or training.