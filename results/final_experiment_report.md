# Final Experiment Report

## Architecture

The final model is the locked concatenation-fusion architecture: the unchanged SatelliteCNN (128-dimensional embedding), WeatherLSTM (128-dimensional embedding), and SoilMLP (64-dimensional embedding) are concatenated into a 320-dimensional representation. The fusion MLP uses LayerNorm, Linear(320,128), ReLU, fusion dropout 0.0, Linear(128,64), and ReLU, followed by the existing regression head.

## Locked configuration

- Learning rate: 0.0005
- Fusion dropout: 0.0
- Optimizer: AdamW
- Loss: MSELoss
- Batch size: 16
- Fixed final epochs: 9
- Seeds: 42, 123, 2026

## Data and protocol

- Training samples: 198 (canonical train + validation)
- Held-out test samples: 41
- Normalization statistics were calculated from the 198 training samples only and stored in each checkpoint.
- The test set was evaluated exactly once per final checkpoint, after all three final models had been trained and saved.
- No test metric was used for model or epoch selection.

## Results

Mean test MAE: 591.175 +/- 68.889 kg/ha
Mean test RMSE: 758.353 +/- 84.317 kg/ha
Mean test R²: 0.328446 +/- 0.143360

## Limitations

The final estimates are based on 41 held-out district-season-year samples and three random seeds. The test set is a historical holdout for final reporting, not a basis for further tuning. No uncertainty intervals or external validation set were calculated.