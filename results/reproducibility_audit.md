# Reproducibility Audit

## Scope

This audit covers the baseline runner and the data/model code used by the
seven experiments:

- `src/run_baselines.py`
- `src/dataset.py`
- `src/models/satellite_cnn.py`
- `src/models/weather_lstm.py`
- `src/models/soil_mlp.py`
- `src/models/fusion_model.py`

No dataset, split file, raw file, or model architecture was modified.

## Configuration found

The baseline runner uses the following fixed settings:

- canonical split from `data/processed/dataset_splits.csv`
- 172 train, 26 validation, and 41 test samples
- target mean and population standard deviation calculated from train targets
  only
- AdamW with learning rate `0.001`
- MSELoss on normalized targets
- batch size `16`
- maximum `8` epochs
- validation-loss early stopping with patience `3`
- gradient norm clipping at `1.0`
- `num_workers=0`
- train DataLoader `shuffle=True`
- validation and test DataLoaders `shuffle=False`

The model initialization and optimizer construction are performed after the
per-model Torch seed is set. The same Python model/data code is selected on
CPU and GPU; only the Torch device changes.

## Random-state findings

### Seeds that are controlled

- `torch.manual_seed(...)` is called at process startup.
- `np.random.seed(...)` is called at process startup.
- Torch is reseeded before each model with `SEED + model_index`.
- The dataset, collation, validation, and test paths do not intentionally
  sample random values.
- `num_workers=0` avoids worker-process random-state divergence.

### Seeds and deterministic controls that are missing

- Python's `random.seed(...)` is never called. The current code does not use
  Python `random`, but this remains an uncontrolled RNG source if future code
  adds it.
- `torch.cuda.manual_seed_all(...)` is not called explicitly. On CUDA,
  `torch.manual_seed(...)` seeds the default CUDA generators in current
  PyTorch, but the explicit all-device call is absent.
- `torch.use_deterministic_algorithms(True)` is not enabled.
- `torch.backends.cudnn.deterministic` is not enabled.
- `torch.backends.cudnn.benchmark` is not disabled.
- No deterministic-operation error policy is configured.
- The training DataLoader does not receive an explicit `torch.Generator`.
  Its shuffle order therefore uses the global Torch generator at iterator
  creation, after model initialization and other Torch operations have
  consumed random numbers.

These omissions mean that the original runner does not guarantee bitwise
repeatability, particularly on CUDA.

## DataLoader and model behavior

- The training loader shuffles samples each epoch; validation and test order
  is fixed.
- The training loader has no workers and no persistent worker state.
- Satellite loading, weather loading, soil lookup, padding, and validation are
  deterministic for the same file contents and row order.
- BatchNorm layers in the satellite CNN remain in training mode during training
  and evaluation mode during validation/test, as expected.
- Dropout layers are active during training and disabled for validation/test.
- The optimizer is freshly initialized for each model and receives the same
  learning rate and parameter groups on CPU and GPU.
- Early stopping uses the validation loss only; test data is not used for
  selection.

## CPU/GPU environment differences

The baseline code and hyperparameters are otherwise the same, but execution
differs in these ways:

1. CPU uses host implementations; GPU uses CUDA/cuDNN kernels.
2. Floating-point reduction order and kernel implementations can differ.
3. CUDA kernels may be nondeterministic when deterministic settings are not
   enabled.
4. Early stopping can select different epochs after small validation-loss
   differences, amplifying metric differences.
5. The first CPU run used PyTorch `2.14.0+cu130` in the project `.venv` on
   CPU; the GPU run used the same build on the RTX 3050.
6. The agent terminal initially inherited `CUDA_VISIBLE_DEVICES=-1`, which
   masked the GPU and caused the earlier CPU run. The GPU rerun removed that
   variable only for its process. This is an environment difference, not a
   package or code difference.

The default system Python is a separate installation without PyTorch; it was
not used for either experiment.

## Controlled repeatability test

The requested GPU repeatability runner explicitly seeds Python, NumPy, CPU
Torch, and all visible CUDA generators, and supplies an explicit seeded
DataLoader generator. It keeps the existing model/configuration and uses
seeds `42`, `123`, and `2026`.

Results are in:

- `results/repeatability_runs.csv` (per-seed records)
- `results/repeatability_results.csv` (mean/std summary)
- `results/repeatability_report.md`

The three-seed standard deviations show meaningful run-to-run sensitivity:

- Satellite-only: MAE std `42.018` kg/ha; RMSE std `49.372` kg/ha; R2 std
  `0.097092`
- Satellite + Weather: MAE std `99.730` kg/ha; RMSE std `128.053` kg/ha; R2
  std `0.253544`
- Full Attention: MAE std `9.029` kg/ha; RMSE std `7.908` kg/ha; R2 std
  `0.016758`

These are seed-sensitivity measurements, not a same-seed duplicate test.
Therefore, the GPU experiments should not yet be described as strictly
reproducible or as having a single definitive metric. The full attention
model is comparatively stable across these three seeds, while the two
satellite-containing ablations show larger variability, especially
Satellite + Weather. Exact cross-device equality should not be expected
without a deliberate deterministic-kernel configuration.
