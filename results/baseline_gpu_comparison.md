# CPU and GPU Baseline Comparison

The seven baseline experiments were rerun on the same project data using the
same canonical chronological split, target normalization, model definitions,
and fixed training configuration. The only execution difference was the
compute device.

## Fixed experimental protocol

- Train/validation/test split: 172/26/41 samples
- Split definition: `data/processed/dataset_splits.csv`
- Target normalization: mean and standard deviation fitted from the 172
  training targets only
- Optimizer: AdamW
- Loss: MSELoss
- Batch size: 16
- Learning rate: 0.001
- Maximum epochs: 8
- Early-stopping patience: 3
- Metrics: MAE, RMSE, and R2 on the same held-out 41-sample test set
- Models: weather-only, satellite-only, soil-only, satellite+weather,
  weather+soil, satellite+soil, and the unchanged full attention model

## CUDA diagnosis

The project `.venv` contains CUDA-enabled PyTorch `2.14.0+cu130`, and the
machine NVIDIA driver detects an NVIDIA GeForce RTX 3050 Laptop GPU. The
agent terminal inherited `CUDA_VISIBLE_DEVICES=-1`, which masked all devices
from PyTorch and caused the earlier CPU run. The GPU rerun removed that
masking variable for the run process only; no packages, datasets, splits, or
model code were changed.

The successful project `.venv` checks were:

```text
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
torch.cuda.get_device_name(0): NVIDIA GeForce RTX 3050 Laptop GPU
```

## CPU results

The original CPU artifacts are preserved under
`results/baseline_cpu/`, including its comparison CSV, report, plot, and
per-model predictions.

| Model | Best epoch | MAE (kg/ha) | RMSE (kg/ha) | R2 | Time (s) |
|---|---:|---:|---:|---:|---:|
| satellite_only | 6 | 608.146 | 785.235 | 0.285877 | 88.5 |
| weather_only | 5 | 617.102 | 802.332 | 0.254441 | 56.5 |
| weather_soil | 7 | 650.026 | 824.268 | 0.213115 | 54.0 |
| satellite_weather_soil_attention | 4 | 654.480 | 798.738 | 0.261104 | 119.6 |
| satellite_weather | 8 | 681.999 | 859.761 | 0.143889 | 117.6 |
| satellite_soil | 7 | 701.850 | 880.391 | 0.102312 | 94.9 |
| soil_only | 7 | 739.024 | 934.612 | -0.011665 | 25.0 |

## GPU results

The current files in `results/baseline_comparison.csv`,
`results/baseline_comparison.md`, `results/baseline_comparison.png`, and
`results/baseline_predictions/` are the GPU run outputs.

| Model | Best epoch | MAE (kg/ha) | RMSE (kg/ha) | R2 | Time (s) |
|---|---:|---:|---:|---:|---:|
| satellite_soil | 8 | 609.077 | 830.203 | 0.201742 | 31.6 |
| satellite_weather | 5 | 621.958 | 788.047 | 0.280752 | 33.3 |
| weather_soil | 6 | 640.461 | 813.089 | 0.234314 | 27.8 |
| satellite_weather_soil_attention | 2 | 642.199 | 839.091 | 0.184559 | 24.4 |
| weather_only | 2 | 673.363 | 851.108 | 0.161036 | 20.6 |
| soil_only | 1 | 742.728 | 934.700 | -0.011856 | 15.8 |
| satellite_only | 1 | 894.185 | 1130.157 | -0.479285 | 17.1 |

The metrics differ between CPU and GPU runs because the runner uses the same
seed and configuration but GPU and CPU kernels can produce different floating
point optimization trajectories; early stopping then selects different
epochs. These are fixed-protocol device comparisons, not hyperparameter
tuning results.

## Prediction-file validation

All seven GPU prediction files were checked after the run:

- exactly 41 rows per model
- exactly 41 unique sample IDs per model
- 41 finite actual yields and 41 finite predictions per model
- no NaN or Inf prediction values

No test samples were used for target normalization, training, or checkpoint
selection. No additional data was downloaded.
