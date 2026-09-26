# Multimodal Rice Yield Prediction

## Project Overview

This project develops a multimodal deep-learning system for estimating rice yield in Andhra Pradesh, India. The system combines satellite imagery, daily weather observations, and district-level soil features to predict yield in **kg/ha**.

The project includes:

- A validated multimodal dataset and chronological evaluation split.
- Satellite, weather, and soil preprocessing.
- CNN, LSTM, and MLP modality encoders.
- A locked concatenation-fusion regression model.
- Reproducibility, baseline, ablation, tuning, final evaluation, and error-analysis artifacts.
- An inference-only Streamlit application that accepts supplied inputs.

The authoritative frozen audit is [`results/FINAL_PROJECT_AUDIT.md`](results/FINAL_PROJECT_AUDIT.md). Reusable paper-safe facts are in [`results/PAPER_FACTS.md`](results/PAPER_FACTS.md).

## Problem Statement

Rice yield depends on interacting environmental and soil conditions. A model that uses only one information source may miss relevant spatial, temporal, or soil context. This project investigates whether satellite, weather, and soil modalities can be combined for district-season-year rice-yield estimation.

## Motivation

The goal is to build a reproducible multimodal research pipeline that:

- represents crop conditions using satellite observations;
- represents seasonal weather using a daily sequence;
- represents soil properties using structured features; and
- evaluates generalization using a chronological held-out test set.

The current application is a research and demonstration system, not an operational forecasting service.

## Objectives

1. Prepare and validate aligned satellite, weather, soil, and yield observations.
2. Preserve a chronological train/validation/test protocol.
3. Train separate modality encoders.
4. Combine the learned embeddings using concatenation fusion.
5. Evaluate the locked final model across three random seeds.
6. Provide reproducible inference through Streamlit using supplied inputs.

## Multimodal Architecture

```text
Satellite image [7, 128, 128] ──> Satellite CNN ──┐
Weather sequence [183, 3] ──────> Weather LSTM ────┼─> Concatenation Fusion
Soil features [4] ──────────────> Soil MLP ────────┘       └─> Regression head
                                                               └─> Yield (kg/ha)
```

The final model uses concatenation fusion. The earlier attention-fusion experiments remain historical exploratory analyses and are not the final architecture.

## Dataset

The validated multimodal dataset contains **239 samples** covering Andhra Pradesh district-season-year observations. The historical range is **1998-99 through 2011-12**.

### Modalities

- **Satellite:** Landsat Collection 2-derived reflectance bands B1, B2, B3, B4, B5, and B7, plus NDVI derived from B4/B3. The model representation is a float32 tensor with shape **7 × 128 × 128**.
- **Weather:** 183 daily observations with the feature order `rainfall_mm`, `temperature_c`, `humidity_pct`. The model representation is **183 × 3**.
- **Soil:** Four 0–5 cm soil features: nitrogen, pH, soil organic carbon (SOC), and clay.
- **Target:** District-season-year rice yield in **kg/ha**.

The canonical processed files are [`data/processed/final_training_samples.csv`](data/processed/final_training_samples.csv) and [`data/processed/dataset_splits.csv`](data/processed/dataset_splits.csv). They are frozen and must not be changed for the reported results.

## Data Split

The canonical split is chronological with no year overlap:

| Split | Samples | Years |
|---|---:|---|
| Train | 172 | 1998-99 through 2008-09 |
| Validation | 26 | 2009-10 |
| Test | 41 | 2010-11 and 2011-12 |

Final training used the combined **198 train+validation samples**. The 41 test samples were retained for final held-out evaluation.

## Preprocessing

- Satellite arrays are validated for finite values and converted to the model layout **[7, 128, 128]**.
- Weather inputs are validated for the required columns, 183 observations, and finite values. Dataset batching supports the established weather mask/padding convention.
- Soil inputs are validated for finite, physically bounded values and represented as the same four raw features used during training.
- The target is normalized using statistics from the applicable fitting data only:
  - tuning used the training split;
  - final training used the 198 train+validation samples.
- The final checkpoint stores the train+validation target mean and standard deviation used to convert normalized predictions back to kg/ha.

## Model Architecture

The locked final model is:

- SatelliteCNN embedding: 128 dimensions.
- WeatherLSTM embedding: 128 dimensions.
- SoilMLP embedding: 64 dimensions.
- Concatenated embedding: 320 dimensions.
- Fusion MLP: LayerNorm → Linear(320, 128) → ReLU → Dropout(0.0) → Linear(128, 64) → ReLU.
- Regression head: LayerNorm → Linear(64, 32) → ReLU → Dropout(0.1) → Linear(32, 1).
- Trainable parameters: **948,225**.

The final checkpoints are:

- [`models/final_model_seed42.pt`](models/final_model_seed42.pt)
- [`models/final_model_seed123.pt`](models/final_model_seed123.pt)
- [`models/final_model_seed2026.pt`](models/final_model_seed2026.pt)

## Training

The locked final configuration is:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Loss | MSELoss |
| Learning rate | 0.0005 |
| Fusion dropout | 0.0 |
| Batch size | 16 |
| Epochs | 9 |
| Seeds | 42, 123, 2026 |
| Fitting samples | 198 train+validation samples |

The nine-epoch count was fixed before final test evaluation as the median of the selected validation configuration's best epochs across seeds: 1, 9, and 9. The test set was not used for this choice.

## Baseline/Ablation Experiments

Baseline, repeatability, and fusion-ablation files are preserved as historical experiment artifacts:

- [`results/baseline_comparison.csv`](results/baseline_comparison.csv)
- [`results/repeatability_results.csv`](results/repeatability_results.csv)
- [`results/final_fusion_ablation_reproducibility.csv`](results/final_fusion_ablation_reproducibility.csv)

The unified ablation means were:

| Historical model | Mean MAE (kg/ha) | Mean RMSE (kg/ha) | Mean R² |
|---|---:|---:|---:|
| Satellite-only | 741.301 | 953.025 | -0.063928 |
| Attention Fusion | 706.246 | 893.065 | 0.073776 |
| Concatenation Fusion | 647.602 | 826.409 | 0.202263 |

These values are **exploratory historical comparisons**, not the locked final test results. The earlier baseline protocol differs from final training and must not be presented as a directly equivalent final-model comparison.

## Final Results

The primary final results are computed separately for each final seed on the same 41-sample test set, followed by the arithmetic mean and sample standard deviation across seeds. The complete table is [`results/FINAL_RESULTS_TABLE.csv`](results/FINAL_RESULTS_TABLE.csv).

| Seed | MAE (kg/ha) | RMSE (kg/ha) | R² |
|---:|---:|---:|---:|
| 42 | 512.187 | 660.994 | 0.493979 |
| 123 | 622.524 | 806.461 | 0.246747 |
| 2026 | 638.815 | 807.604 | 0.244611 |
| **Mean ± sample std** | **591.175 ± 68.889** | **758.353 ± 84.317** | **0.328446 ± 0.143360** |

Metric definitions:

- **MAE:** mean absolute prediction error in kg/ha.
- **RMSE:** square root of mean squared prediction error in kg/ha.
- **R²:** `1 - SSE/SST` on the 41-sample test set.
- **±:** sample standard deviation across the three final seeds, not a confidence interval.

These are the only primary final metrics. The per-seed predictions are in [`results/final_test_predictions.csv`](results/final_test_predictions.csv), and the human-readable evaluation is in [`results/final_evaluation.md`](results/final_evaluation.md).

## Error Analysis

The secondary analysis is documented in [`results/error_analysis.md`](results/error_analysis.md) and [`results/error_analysis.csv`](results/error_analysis.csv). It first averages the three seed predictions for each test sample and then computes sample-level errors, so it is not the same aggregation as the primary results.

Secondary descriptive findings:

- Mean absolute error: **556.416 kg/ha**.
- Mean signed error: **+74.520 kg/ha**.
- Mean actual yield: **2852.582 kg/ha**.
- Mean predicted yield: **2927.102 kg/ha**.
- Pearson correlation: **0.6655**.

These findings describe observed errors in this holdout. They do not establish causal district, season, weather, soil, or satellite effects.

## Streamlit Deployment

The deployment is an **inference-only demo system using supplied satellite, weather, and soil inputs**. It:

- loads one of the three saved final checkpoints;
- reconstructs the final concatenation-fusion architecture;
- uses the checkpoint's saved target-normalization parameters;
- calls `model.eval()` and `torch.no_grad()`;
- uses CUDA when available and CPU otherwise;
- reads uploaded files in memory without modifying them;
- does not train, tune, download data, or call live satellite/weather APIs.

Deployment details are in [`results/streamlit_deployment_report.md`](results/streamlit_deployment_report.md). The app does **not** provide a scientifically validated 2027 forecast from unknown future weather or satellite observations.

## Input Format

### Satellite

- File type: `.npy`.
- Accepted shape: **7 × 128 × 128** or **128 × 128 × 7**.
- The latter layout is converted automatically to **7 × 128 × 128**.
- Values must be finite.
- Channel 7 is used for the NDVI preview.

### Weather

- File type: `.csv`.
- Exactly **183 rows**.
- Required feature columns, in this order:

  ```text
  rainfall_mm
  temperature_c
  humidity_pct
  ```

- An optional `date` column may be present and is ignored.
- Values must be finite.

### Soil

Provide four numeric values:

1. nitrogen (g/kg)
2. pH
3. SOC (g/kg)
4. clay (%)

The app validates the values and passes the same raw four-feature representation used by the trained model.

## How to Run

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The project virtual environment uses Python 3.12. The app opens locally in a browser, normally at `http://localhost:8501`.

If activation is not desired, run the project interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app requires the three final checkpoint files under `models/`. Supply valid satellite and weather uploads and the four soil values, select a checkpoint seed, and press **PREDICT YIELD**.

## Project Structure

```text
Multimodal_Rice_Yield/
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── final_model_seed42.pt
│   ├── final_model_seed123.pt
│   └── final_model_seed2026.pt
├── results/
│   ├── FINAL_PROJECT_AUDIT.md
│   ├── FINAL_RESULTS_TABLE.csv
│   ├── PAPER_FACTS.md
│   ├── final_test_results.csv
│   ├── final_test_predictions.csv
│   ├── error_analysis.md
│   └── streamlit_deployment_report.md
├── src/
│   ├── dataset.py
│   ├── split_dataset.py
│   ├── models/
│   └── deployment/
└── figures/
```

The `results/` directory contains the reproducibility, baseline, ablation, tuning, evaluation, error-analysis, audit, and deployment artifacts. Existing artifacts are preserved as part of the experimental record.

### Version-control note

The project repository excludes the local `.venv/`, Python caches, editor metadata, credential-like files, and the embedded raw yield-data repository `data/raw/yield/India_Agri_Data/`, which contains its own `.git/` metadata. The canonical processed datasets and required result files are included; the excluded raw source can be restored separately if needed. No project file was excluded because of GitHub's individual-file size limit; the final checkpoints and required result files are intentionally included. Recreate the virtual environment from [`requirements.txt`](requirements.txt) rather than committing it.

## Limitations

- The final test set contains only 41 historical samples.
- Seed variability is reported as sample standard deviation, not a confidence interval.
- District and season subgroup sizes are small.
- The model has not been externally validated outside this historical holdout.
- The application requires supplied inputs and does not retrieve live or future observations.
- Error analysis is descriptive and cannot identify causes.
- Historical holdout evaluation is distinct from future-season forecasting.

## Future Work

Potential future work includes external validation on additional years or regions, larger and more diverse multimodal datasets, uncertainty estimation, improved input-quality monitoring, and deployment evaluation under realistic data-availability conditions. Any future experiment should be separately documented and must not silently replace the frozen results reported here.

This submission README documents the frozen experiment; it does not constitute the research paper.
