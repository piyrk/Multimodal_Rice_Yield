# Baseline Comparison

All experiments used the exact existing chronological split: 172 train, 26 validation, 41 test samples. Target normalization mean and standard deviation were calculated from training targets only.

Device used: `cuda`
PyTorch: `2.14.0+cu130`
Configuration: AdamW, MSELoss, batch size 16, learning rate 0.001, maximum epochs 8, early-stopping patience 3.

## Test Comparison

| Model | Best epoch | Validation loss | MAE (kg/ha) | RMSE (kg/ha) | R2 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| satellite_soil | 8 | 0.470450 | 609.077 | 830.203 | 0.201742 | 31.6 |
| satellite_weather | 5 | 0.538964 | 621.958 | 788.047 | 0.280752 | 33.3 |
| weather_soil | 6 | 0.820775 | 640.461 | 813.089 | 0.234314 | 27.8 |
| satellite_weather_soil_attention | 2 | 0.848304 | 642.199 | 839.091 | 0.184559 | 24.4 |
| weather_only | 2 | 0.877475 | 673.363 | 851.108 | 0.161036 | 20.6 |
| soil_only | 1 | 1.026053 | 742.728 | 934.700 | -0.011856 | 15.8 |
| satellite_only | 1 | 1.468478 | 894.185 | 1130.157 | -0.479285 | 17.1 |

## Models

- Weather-only: existing masked WeatherLSTM plus regression head.
- Satellite-only: existing SatelliteCNN plus regression head.
- Soil-only: existing SoilMLP plus regression head.
- Pair models: selected existing encoders concatenated into a compact regression head.
- Full model: existing `MultimodalYieldModel` with learned attention fusion; its architecture was not changed.

Predictions for every model are saved under `results/baseline_predictions/`. The comparison plot is `results/baseline_comparison.png`.

No test data was used for normalization, training, or model selection.