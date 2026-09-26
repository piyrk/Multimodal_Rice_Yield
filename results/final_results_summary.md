# Final Results Summary

## Locked final model

- Concatenation Fusion
- Learning rate: 0.0005
- Fusion dropout: 0.0
- Optimizer: AdamW
- Loss: MSELoss
- Batch size: 16
- Epochs: 9
- Training data: 198 train+validation samples

## Final three-seed test metrics

- MAE: **591.175 +/- 68.889 kg/ha**
- RMSE: **758.353 +/- 84.317 kg/ha**
- R²: **0.328446 +/- 0.143360**

## Key error-analysis findings

- Mean actual yield: **2852.582 kg/ha**; mean predicted yield: **2927.102 kg/ha**.
- Mean absolute error across mean-per-sample predictions: **556.416 kg/ha**.
- Mean signed error: **74.520 kg/ha**, indicating average overprediction.
- Pearson observed-predicted correlation: **0.6655**.
- District and season differences are descriptive observations from the holdout, not causal findings.

## Limitations

The analysis covers 41 test samples and aggregates three seed predictions per sample. Small group sizes limit the stability of district and season comparisons. The results do not establish why individual predictions are wrong and should not be generalized beyond the evaluated holdout.