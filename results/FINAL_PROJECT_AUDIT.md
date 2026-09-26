# Final Project Audit and Results Freeze

Audit status: **complete**. This document freezes the authoritative final experiment without modifying raw data, canonical splits, checkpoints, or prior experiment artifacts.

## A. Final project definition

The project predicts rice yield in Andhra Pradesh, India, in **kg/ha** from satellite imagery, weather, and soil inputs. The final application is an inference/demo system using supplied inputs.

## B. Dataset summary

- Validated multimodal samples: **239**
- Historical range: **1998-99 through 2011-12**
- Satellite: Landsat Collection 2-derived six reflectance bands plus NDVI; stored for modeling as float32 `[7,128,128]`.
- Weather: 183 daily observations with rainfall, temperature, and humidity.
- Soil: district-level 0–5 cm features nitrogen, pH, SOC, and clay.
- Target: district-season-year rice yield in kg/ha.

## C. Data split

The canonical chronological split is unchanged:

| Split | Count | Years |
|---|---:|---|
| Train | 172 | 1998-99 through 2008-09 |
| Validation | 26 | 2009-10 |
| Test | 41 | 2010-11 and 2011-12 |

There is no year overlap between splits. The final run used train + validation (**198 samples**) for fitting and retained the 41 test samples for final evaluation only.

## D. Final architecture

The final model is **Concatenation Fusion**:

1. SatelliteCNN → 128-dimensional embedding
2. WeatherLSTM → 128-dimensional embedding
3. SoilMLP → 64-dimensional embedding
4. Concatenate → 320 dimensions
5. Fusion MLP: LayerNorm → Linear(320,128) → ReLU → Dropout(0.0) → Linear(128,64) → ReLU
6. Existing regression head: LayerNorm → Linear(64,32) → ReLU → Dropout(0.1) → Linear(32,1)

The deployed inference helper reproduces this architecture and loads the saved final checkpoint state dictionaries. The final model has **948,225 trainable parameters**. The earlier 954,242 count was for the attention model and is not a final-model count.

## E. Final hyperparameters

- Learning rate: **0.0005**
- Fusion dropout: **0.0**
- Optimizer: **AdamW**
- Loss: **MSELoss**
- Batch size: **16**
- Epochs: **9**
- Seeds: **42, 123, 2026**
- Final normalization: target mean and standard deviation calculated from the 198 train+validation samples only; saved in each checkpoint.

The nine-epoch choice is the median of selected-configuration validation best epochs 1, 9, and 9. No test metric was used for this choice.

## F. Primary final test results

Primary results are the per-seed metrics in `final_test_results.csv` and their arithmetic mean ± sample standard deviation:

| Seed | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---:|---:|---:|---:|
| 42 | 512.187 | 660.994 | 0.493979 |
| 123 | 622.524 | 806.461 | 0.246747 |
| 2026 | 638.815 | 807.604 | 0.244611 |
| **Mean ± std** | **591.175 ± 68.889** | **758.353 ± 84.317** | **0.328446 ± 0.143360** |

Definitions:

- MAE: mean absolute prediction error in kg/ha.
- RMSE: square root of mean squared prediction error in kg/ha.
- R²: `1 - SSE/SST` on the 41-sample test set.
- ± values: sample standard deviation across the three final seeds, not confidence intervals.

These are the only metrics to quote as the primary final results.

## G. Error-analysis methodology

`error_analysis.csv` and `error_analysis.md` average the three seed predictions per `sample_id` and then compute sample-level errors for 41 samples. This is a different aggregation from the primary results:

- Primary: compute metrics separately for each seed, then average the three metric values.
- Secondary error analysis: average predictions across seeds first, then compute sample-level metrics.

Secondary findings include mean absolute error **556.416 kg/ha**, mean signed error **+74.520 kg/ha**, mean actual yield **2852.582 kg/ha**, mean predicted yield **2927.102 kg/ha**, and Pearson correlation **0.6655**. These are descriptive analyses, not causal findings.

## H. Baseline comparison summary

`baseline_comparison.csv` is an exploratory GPU baseline run using the earlier 172/26/41 training protocol. Its best single run reported satellite+soil MAE **609.077 kg/ha**, satellite+weather MAE **621.958**, weather+soil **640.461**, full attention **642.199**, weather-only **673.363**, soil-only **742.728**, and satellite-only **894.185**.

These are secondary historical comparisons only. They used the earlier training/validation fitting protocol and are not the locked final model result. They must not be substituted for Section F.

## I. Fusion-ablation summary

The unified three-seed ablation used one controlled execution path:

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| Satellite-only | 741.301 | 953.025 | -0.063928 |
| Attention Fusion | 706.246 | 893.065 | 0.073776 |
| Concatenation Fusion | 647.602 | 826.409 | 0.202263 |

This is an exploratory architecture comparison on the historical fixed protocol. It supported locking concatenation fusion, but it is not the final retrained result in Section F.

## J. Repeatability findings

The earlier repeatability study reported seed variability for older controlled runs:

- Satellite-only: MAE **652.880 ± 42.018**, RMSE **835.851 ± 49.372**, R² **0.188021 ± 0.097092**
- Satellite+weather: MAE **687.714 ± 99.730**, RMSE **880.169 ± 128.053**, R² **0.083773 ± 0.253544**
- Attention Fusion: MAE **726.440 ± 9.029**, RMSE **913.877 ± 7.908**, R² **0.032654 ± 0.016758**

These are secondary exploratory repeatability results. They must not be quoted as the final model metrics.

## K. Streamlit deployment status

Deployment is verified locally. The app:

- loads one of `final_model_seed42.pt`, `final_model_seed123.pt`, or `final_model_seed2026.pt`;
- reproduces the locked concatenation architecture;
- converts satellite arrays to `[7,128,128]`, weather to `[183,3]`, and soil to `[4]` float32 tensors;
- uses the checkpoint's saved train+validation target mean and standard deviation;
- calls `model.eval()` and `torch.no_grad()`;
- uses CUDA when available and CPU otherwise;
- reads uploaded files in memory and does not modify them;
- performs no training, tuning, or live API access.

The local health endpoint returned `200 ok`, and a valid supplied-input probe produced a finite prediction. An invalid satellite shape produced a controlled validation error.

## L. Limitations

- The test set contains only 41 historical samples.
- The ± values are seed variability, not confidence intervals.
- District and season group sizes are small.
- The model has not been externally validated outside this historical holdout.
- The application requires supplied future inputs; it does not obtain live or future weather/satellite observations.
- Error analysis is descriptive and cannot identify causes.

## M. Claims supported

- The final locked model achieved the Section F metrics on the 41-sample held-out test set.
- The canonical split is chronological with no year overlap.
- The final model combines satellite, weather, and soil branches through concatenation fusion.
- The Streamlit app performs inference using supplied inputs and saved checkpoints.
- The reported baselines, ablations, and repeatability studies are historical exploratory analyses with their own stated protocols.

## N. Claims that must NOT be made

- Do not claim the model is scientifically validated for 2027 forecasting.
- Do not claim the model has been externally or operationally validated.
- Do not present secondary error-analysis metrics as the primary final test metrics.
- Do not compare exploratory baseline values to the final values as if they came from identical final-training protocols.
- Do not claim causal district, seasonal, weather, soil, or satellite effects from the grouped error summaries.
- Do not quote the earlier 954,242 attention-model parameter count as the final concatenation-model parameter count.
- Do not claim that the Streamlit demo automatically retrieves live satellite or weather data.
