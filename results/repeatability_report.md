# GPU Repeatability Report

Three GPU runs were completed for each selected model with seeds 42, 123, and 2026.

Device: `NVIDIA GeForce RTX 3050 Laptop GPU`
PyTorch: `2.14.0+cu130`; CUDA runtime: `13.0`

The exact existing split (172/26/41), train-only target normalization, AdamW, MSELoss, batch size 16, learning rate 0.001, maximum 8 epochs, and patience 3 were retained.

## Per-seed results

| Model | Seed | Best epoch | Validation loss | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|---:|
| satellite_only | 42 | 6 | 0.661330 | 600.912 | 787.467 | 0.281812 |
| satellite_only | 123 | 6 | 0.657554 | 653.910 | 816.448 | 0.227975 |
| satellite_only | 2026 | 5 | 0.531642 | 703.819 | 903.639 | 0.054276 |
| satellite_weather | 42 | 1 | 1.021337 | 733.905 | 917.127 | 0.025835 |
| satellite_weather | 123 | 2 | 1.297952 | 780.026 | 1015.223 | -0.193704 |
| satellite_weather | 2026 | 6 | 0.592683 | 549.211 | 708.159 | 0.419188 |
| satellite_weather_soil_attention | 42 | 1 | 1.014659 | 735.406 | 924.118 | 0.010925 |
| satellite_weather_soil_attention | 123 | 2 | 0.886555 | 714.084 | 904.864 | 0.051711 |
| satellite_weather_soil_attention | 2026 | 8 | 0.718027 | 729.830 | 912.648 | 0.035327 |

## Mean and standard deviation

| Model | Mean MAE | Std MAE | Mean RMSE | Std RMSE | Mean R2 | Std R2 |
|---|---:|---:|---:|---:|---:|---:|
| satellite_only | 652.880 | 42.018 | 835.851 | 49.372 | 0.188021 | 0.097092 |
| satellite_weather | 687.714 | 99.730 | 880.169 | 128.053 | 0.083773 | 0.253544 |
| satellite_weather_soil_attention | 726.440 | 9.029 | 913.877 | 7.908 | 0.032654 | 0.016758 |

Attention statistics are in `attention_summary.csv` and `attention_summary.md`. They are descriptive model outputs, not causal modality importance.