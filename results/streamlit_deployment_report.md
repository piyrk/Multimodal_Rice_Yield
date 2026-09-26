# Streamlit Inference Deployment Report

## Run locally

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application is inference-only. It loads one of the locked checkpoints:

- `models/final_model_seed42.pt`
- `models/final_model_seed123.pt`
- `models/final_model_seed2026.pt`

## Expected inputs

- Satellite: `.npy`, shape `[7,128,128]` or `[128,128,7]`; the latter is transposed automatically. Channel 7 is treated as NDVI for the preview.
- Weather: `.csv` with exactly 183 rows and `rainfall_mm`, `temperature_c`, `humidity_pct` columns. The feature order is preserved. An optional `date` column is ignored.
- Soil: nitrogen (g/kg), pH, SOC (g/kg), and clay (%).

Inputs are checked for shape, row count, required columns, finite values, and valid soil ranges. Weather padding follows the dataset convention: the submitted 183 observations are represented with a 183-element all-true mask.

## Preprocessing and inference

The app converts inputs to float32 tensors and satellite data to `[1,7,128,128]`, weather data to `[1,183,3]`, and soil data to `[1,4]`. Soil values use the same raw feature representation as training; no additional scaling is applied. The selected checkpoint's train+validation target mean and standard deviation are used to convert the normalized output back to kg/ha.

Inference calls `model.eval()` and `torch.no_grad()`. Uploaded files are read in memory only and are never modified.

## CPU/GPU behavior

CUDA is used when available; otherwise inference runs on CPU. The selected device is shown in the interface.

## Limitations

This is a demo system using supplied inputs, not a live satellite or weather service. It does not claim to produce a scientifically validated 2027 forecast from unknown future weather. The locked model was evaluated on a 41-sample historical holdout with mean test performance of MAE 591.175 +/- 68.889 kg/ha, RMSE 758.353 +/- 84.317 kg/ha, and R2 0.328446 +/- 0.143360 across three seeds. Historical holdout evaluation and future-season forecasting are distinct: future forecasting requires valid future weather and satellite inputs and additional validation.
