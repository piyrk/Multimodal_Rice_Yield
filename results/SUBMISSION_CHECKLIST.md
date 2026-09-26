# Final Submission Checklist

This checklist records the final-submission readiness checks for the frozen project. It does not authorize retraining, tuning, architecture changes, dataset changes, checkpoint changes, or split changes.

## Code and execution

- [x] Project-local `app.py` exists.
- [x] `requirements.txt` exists and lists the runtime dependencies.
- [x] The project virtual environment is `.venv` using Python 3.12.
- [x] The Streamlit application starts locally.
- [x] A valid supplied-input inference probe produced a finite prediction.
- [x] Invalid satellite input produced a controlled validation error.
- [x] The deployment path is inference-only and contains no training loop, optimizer step, or backward pass.

## Model checkpoints

- [x] `models/final_model_seed42.pt` is present.
- [x] `models/final_model_seed123.pt` is present.
- [x] `models/final_model_seed2026.pt` is present.
- [x] All three checkpoints load with the deployed concatenation-fusion architecture.
- [x] The deployed model has 948,225 trainable parameters.
- [x] The checkpoints contain the final train+validation target-normalization parameters.

## Results and reports

- [x] `results/final_test_results.csv` is present.
- [x] `results/final_test_predictions.csv` is present.
- [x] `results/final_evaluation.md` is present.
- [x] `results/final_experiment_report.md` is present.
- [x] `results/error_analysis.csv` is present.
- [x] `results/error_analysis.md` is present.
- [x] `results/final_results_summary.md` is present.
- [x] `results/tuning_summary.csv` is present.
- [x] `results/final_fusion_ablation_reproducibility.csv` is present.
- [x] `results/baseline_comparison.csv` is present.
- [x] `results/repeatability_results.csv` is present.
- [x] `results/streamlit_deployment_report.md` is present.
- [x] `results/FINAL_PROJECT_AUDIT.md` is present.
- [x] `results/FINAL_RESULTS_TABLE.csv` is present.
- [x] `results/PAPER_FACTS.md` is present.

## Scientific integrity

- [x] The canonical split remains chronological: 172 train, 26 validation, and 41 test samples.
- [x] Final fitting used 198 train+validation samples.
- [x] The test set contains 41 unique samples with no duplicate IDs.
- [x] Test IDs are identical across all three final seeds.
- [x] Tuning used training-only target normalization and did not use test metrics for model selection.
- [x] Final training used train+validation normalization before the one-time held-out evaluation.
- [x] Final metrics are finite and internally consistent with the authoritative results table.
- [x] Primary final metrics are clearly separated from secondary error-analysis metrics.
- [x] Baseline, repeatability, and ablation results are labeled exploratory and are not presented as final results.
- [x] No unsupported scientifically validated 2027 forecasting claim is made.

## README and paper-facing facts

- [x] Project-local `README.md` includes the requested project, data, model, training, results, deployment, and limitations sections.
- [x] README documents satellite shape `7 × 128 × 128`.
- [x] README documents weather shape `183 × 3`.
- [x] README documents four soil features.
- [x] README documents the exact local `.venv` activation, dependency-installation, and Streamlit commands.
- [x] README documents the final model and 948,225 parameters.
- [x] README links to important result files.
- [x] `results/PAPER_FACTS.md` is treated as the concise source for paper, presentation, viva, and README facts.
- [x] `results/FINAL_PROJECT_AUDIT.md` and `results/FINAL_RESULTS_TABLE.csv` agree with the README's primary final numbers.

## Repository hygiene

- [x] The project root was reviewed for obvious temporary files.
- [x] No experiment artifacts were deleted.
- [x] Preserved log-like files under `results/` were treated as part of the project record.
- [x] Raw data, canonical processed data, checkpoints, and split files were not modified for this submission-preparation task.

## Submission status

**Ready for final project submission preparation.** The research paper has not been written as part of this task. The frozen primary results remain:

- MAE: **591.175 ± 68.889 kg/ha**
- RMSE: **758.353 ± 84.317 kg/ha**
- R²: **0.328446 ± 0.143360**

The ± values are sample standard deviations across seeds 42, 123, and 2026, not confidence intervals.
