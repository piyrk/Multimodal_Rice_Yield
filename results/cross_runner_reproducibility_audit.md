# Cross-Runner Reproducibility Audit

## Scope

This audit compares:

- `src/run_repeatability.py`
- `src/run_fusion_ablation.py`
- `src/dataset.py`
- `src/split_dataset.py`
- `src/models/satellite_cnn.py`
- `src/models/weather_lstm.py`
- `src/models/soil_mlp.py`
- `src/models/fusion_model.py`

The audit was read-only with respect to raw data, final samples, the canonical
split, and previous result files.

## Shared data and model implementation

Both runners load the exact rows from `data/processed/dataset_splits.csv` and
construct temporary split CSVs from those rows. Both use
`RiceMultimodalDataset` and `multimodal_collate`, so satellite conversion,
in-memory nodata handling, weather date validation, weather masks, soil
features, target values, and batch shapes are the same.

Both use the same existing `SatelliteCNN`, `WeatherLSTM`, `SoilMLP`, and
`MultimodalYieldModel`. The attention implementation is identical:
projection of each branch to 128 dimensions, scoring with the same
`ModalityAttention` module, softmax across three modalities, weighted sum, and
the existing regression head.

The concatenation runner's new model uses the same branch constructor
dimensions (128, 128, 64) and does not alter any existing encoder or attention
code.

## Training and normalization comparison

The two runners use the same:

- train-only target mean and population standard deviation
- normalized-target MSELoss
- AdamW optimizer
- learning rate `0.001`
- batch size `16`
- maximum eight epochs
- validation-loss early stopping with patience three
- best-state checkpoint copied on strictly lower validation loss
- restored best state before test evaluation
- gradient clipping through the shared `epoch` function
- evaluation conversion back to kg/ha
- validation/test `model.eval()` behavior through the shared epoch/evaluation
  paths

Both use training `shuffle=True`, validation/test `shuffle=False`, and
`num_workers=0`. Both explicitly seed a DataLoader generator with the run
seed. Both call Python `random.seed`, NumPy `np.random.seed`, Torch
`manual_seed`, and CUDA `manual_seed_all` before loader/model construction.

Neither old runner enables `torch.use_deterministic_algorithms(True)`,
`cudnn.deterministic=True`, or `cudnn.benchmark=False`. These settings can
affect bitwise CUDA repeatability.

## Differences found

### Model dispatch

`run_repeatability.py` dispatches:

- Satellite-only through `run_baselines.create_model("satellite_only")`
- Attention through `create_model("satellite_weather_soil_attention")`

`run_fusion_ablation.py` dispatches:

- Satellite-only through its own `model_for` wrapper, which then calls the same
  `create_model("satellite_only")`
- Attention directly through `MultimodalYieldModel()`

The attention constructors resolve to the same class and default dimensions,
but the duplicated dispatch paths were a reproducibility risk.

### Model/run order

The repeatability runner executes:

1. Satellite-only
2. Satellite + Weather
3. Attention Fusion

The fusion runner executes:

1. Satellite-only
2. Attention Fusion
3. Concatenation Fusion

Each run reseeds before model construction, so ordering should not matter in
principle. However, duplicated code and device allocator state made the old
outputs difficult to establish as directly comparable.

### Attention handling

The repeatability runner requests attention details for the attention model
through its `model_output` helper and stores attention rows. The fusion runner
only requests predictions for all three models. This does not change the
attention model's forward path or prediction because the same
`return_details=True` branch computes the same prediction, but it is an
observable code-path difference.

### Cleanup and process state

The fusion runner explicitly calls garbage collection and
`torch.cuda.empty_cache()` after each run. The repeatability runner also does
this in its completed implementation, but its earlier failed attempt did not
perform cleanup before the full-model run and encountered a Rasterio
allocation error. That failure was operational, not a feature-preprocessing
difference.

### Unified implementation

To remove these differences, `src/run_unified_fusion_reproducibility.py`
contains one model factory, one seed initializer, one loader builder, one
training/early-stopping loop, and one test evaluator for:

- Satellite-only
- Attention Fusion
- Concatenation Fusion

It records a SHA-256 digest of the ordered 41 test IDs for each run and
rejects any mismatch.

## Interpretation

The earlier attention metric discrepancy cannot be attributed to a changed
encoder, attention formula, split, target normalization, optimizer, or stated
hyperparameter. The old runners did contain duplicated dispatch and
evaluation paths and different model lists/order. Because their outputs were
not produced by one canonical execution path, they should not be mixed as if
they were one experiment. The unified rerun is the authoritative controlled
comparison for this question.
