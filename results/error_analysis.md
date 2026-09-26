# Final Error Analysis

This analysis uses only `final_test_predictions.csv`. Because that file contains one prediction for each of three final seeds, sample-level statistics below use the mean prediction across the three seeds for each of the 41 test samples. Seed-specific metrics are retained separately below. No model was loaded or retrained.

## Absolute error statistics

- Samples analyzed: **41**
- Mean absolute error: **556.416 kg/ha**
- Median absolute error: **458.759 kg/ha**
- Standard deviation: **466.955 kg/ha**
- Minimum absolute error: **2.248 kg/ha**
- Maximum absolute error: **1812.565 kg/ha**
- Interquartile range: **217.923–838.802 kg/ha**

## Signed error and bias

- Mean signed error (prediction − observation): **74.520 kg/ha**
- Median signed error: **20.571 kg/ha**
- Overpredictions: **22** samples
- Underpredictions: **19** samples
- Exact-zero errors: **0** samples
A positive mean signed error indicates average overprediction; a negative value indicates average underprediction.

## Actual versus predicted relationship

- Mean actual yield: **2852.582 kg/ha**
- Median actual yield: **2916.736 kg/ha**
- Mean predicted yield: **2927.102 kg/ha**
- Median predicted yield: **2836.020 kg/ha**
- Pearson correlation: **0.6655**
The correlation describes association in this holdout; it does not establish causation or imply that any input modality caused an error.

## Largest overpredictions

| Sample | District | Season | Actual | Predicted | Signed error |
|---|---|---|---:|---:|---:|
| AP_SRIKAKULAM_2010_11_Kharif | SRIKAKULAM | Kharif | 813.6 | 2626.2 | +1812.6 |
| AP_VISAKHAPATNAM_2011_12_Rabi | VISAKHAPATNAM | Rabi | 1136.0 | 2568.6 | +1432.6 |
| AP_CUDDAPAH_2010_11_Kharif | CUDDAPAH | Kharif | 1512.0 | 2714.2 | +1202.2 |
| AP_EAST_GODAVARI_2010_11_Kharif | EAST GODAVARI | Kharif | 1926.0 | 2992.3 | +1066.3 |
| AP_VISAKHAPATNAM_2010_11_Rabi | VISAKHAPATNAM | Rabi | 1292.1 | 2194.9 | +902.8 |

## Largest underpredictions

| Sample | District | Season | Actual | Predicted | Signed error |
|---|---|---|---:|---:|---:|
| AP_EAST_GODAVARI_2010_11_Rabi | EAST GODAVARI | Rabi | 4916.0 | 3324.8 | -1591.2 |
| AP_WEST_GODAVARI_2010_11_Rabi | WEST GODAVARI | Rabi | 4649.0 | 3068.2 | -1580.8 |
| AP_GUNTUR_2011_12_Kharif | GUNTUR | Kharif | 3835.9 | 2858.5 | -977.5 |
| AP_KRISHNA_2011_12_Kharif | KRISHNA | Kharif | 3729.1 | 2841.2 | -887.9 |
| AP_KRISHNA_2010_11_Rabi | KRISHNA | Rabi | 3871.0 | 2986.9 | -884.1 |

## Error by district

| district | samples | mean_absolute_error_kg_ha | median_absolute_error_kg_ha | mean_signed_error_kg_ha |
|---|---|---|---|---|
| EAST GODAVARI | 2.000 | 1328.753 | 1328.753 | -262.454 |
| VISAKHAPATNAM | 4.000 | 932.617 | 804.203 | 932.617 |
| SRIKAKULAM | 4.000 | 770.473 | 497.446 | 770.473 |
| WEST GODAVARI | 4.000 | 725.519 | 536.403 | -318.080 |
| CUDDAPAH | 2.000 | 692.759 | 692.759 | 509.486 |
| GUNTUR | 3.000 | 633.199 | 574.297 | -18.439 |
| KRISHNA | 4.000 | 461.179 | 468.121 | -450.893 |
| NELLORE | 3.000 | 388.509 | 231.504 | -388.509 |
| VIZIANAGARM | 4.000 | 380.091 | 450.551 | 83.040 |
| PRAKASAM | 3.000 | 328.546 | 354.998 | 91.881 |
| KURNOOL | 3.000 | 223.262 | 24.070 | -221.763 |
| CHITTOOR | 2.000 | 222.312 | 222.312 | 94.031 |
| ANANTPUR | 3.000 | 175.121 | 217.923 | -28.315 |

## Error by season

| season | samples | mean_absolute_error_kg_ha | median_absolute_error_kg_ha | mean_signed_error_kg_ha |
|---|---|---|---|---|
| Rabi | 18.000 | 618.758 | 486.936 | -84.926 |
| Kharif | 23.000 | 507.627 | 438.922 | 199.304 |

## Error distribution and visible patterns

The plots show the observed error distribution and grouped error summaries. The mean bias is 74.5 kg/ha, so the aggregate sample-level errors are slightly positive on average. Group differences are descriptive patterns in this 41-sample holdout only. They should not be interpreted as causal district or seasonal effects, especially because the sample size per group is limited.

## Seed-specific final metrics

| seed | mae_kg_ha | rmse_kg_ha | r2 |
|---|---|---|---|
| 42.000 | 512.187 | 660.994 | 0.494 |
| 123.000 | 622.524 | 806.461 | 0.247 |
| 2026.000 | 638.815 | 807.604 | 0.245 |

## Limitations

This is a small held-out test set with three stochastic final models. Sample-level values use the mean across seeds, which summarizes prediction variability but is not a separately trained ensemble. Error groupings are associational and do not identify causes. No additional tuning or retraining was performed.