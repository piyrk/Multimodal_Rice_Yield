# Test-Set Model Evaluation

Evaluation used only the held-out chronological test split: **41 samples**.
No test target was used for training, normalization, or checkpoint selection.

## Metrics

- MAE: **614.144 kg/ha**
- RMSE: **789.553 kg/ha**
- R2: **0.278000**
- Minimum prediction: **2561.509 kg/ha**
- Maximum prediction: **3739.081 kg/ha**
- Mean prediction: **3065.228 kg/ha**
- Device: `cuda`

## Outputs

- Per-sample predictions and attention: `results/test_predictions.csv`
- Actual-vs-predicted plot: `results/actual_vs_predicted_yield.png`
- Training history: `results/training_history.csv`
