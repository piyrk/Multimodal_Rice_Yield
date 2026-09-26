# Baseline Comparison

All experiments used the exact existing chronological split: 172 train, 26 validation, 41 test samples. Target normalization mean and standard deviation were calculated from training targets only.

Device used: `cpu`
PyTorch: `2.14.0+cu130`
Configuration: AdamW, MSELoss, batch size 16, learning rate 0.001, maximum epochs 8, early-stopping patience 3.

## Test Comparison

| Model | Best epoch | Validation loss | MAE (kg/ha) | RMSE (kg/ha) | R2 | Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| satellite_only | 6 | 0.645864 | 608.146 | 785.235 | 0.285877 | 88.5 |
| weather_only | 5 | 0.784297 | 617.102 | 802.332 | 0.254441 | 56.5 |
| weather_soil | 7 | 0.823568 | 650.026 | 824.268 | 0.213115 | 54.0 |
| satellite_weather_soil_attention | 4 | 0.772616 | 654.480 | 798.738 | 0.261104 | 119.6 |
| satellite_weather | 8 | 0.635924 | 681.999 | 859.761 | 0.143889 | 117.6 |
| satellite_soil | 7 | 0.667589 | 701.850 | 880.391 | 0.102312 | 94.9 |
| soil_only | 7 | 1.011659 | 739.024 | 934.612 | -0.011665 | 25.0 |

## Models

- Weather-only: existing masked WeatherLSTM plus regression head.
- Satellite-only: existing SatelliteCNN plus regression head.
- Soil-only: existing SoilMLP plus regression head.
- Pair models: selected existing encoders concatenated into a compact regression head.
- Full model: existing `MultimodalYieldModel` with learned attention fusion; its architecture was not changed.

Predictions for every model are saved under `results/baseline_predictions/`. The comparison plot is `results/baseline_comparison.png`.

No test data was used for normalization, training, or model selection.