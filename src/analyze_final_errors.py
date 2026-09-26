"""Error analysis for the locked final test predictions.

This script reads predictions only. It does not load a model or retrain anything.
The three seed predictions are averaged per test sample for sample-level analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"
PREDICTIONS_PATH = RESULTS / "final_test_predictions.csv"
SAMPLES_PATH = PROCESSED / "final_training_samples.csv"


def require_finite(frame: pd.DataFrame, columns: list[str]) -> None:
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Final predictions contain NaN or Inf values")


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional dependencies."""
    include_index = frame.index.name is not None
    columns = ([frame.index.name] if include_index else []) + list(frame.columns)
    lines = [
        "| " + " | ".join(str(column) for column in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for index, row in frame.iterrows():
        values = [index, *row.tolist()] if include_index else row.tolist()
        lines.append(
            "| "
            + " | ".join(
                f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in values
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    samples = pd.read_csv(SAMPLES_PATH)[
        ["sample_id", "district", "year", "season"]
    ].drop_duplicates("sample_id")
    required_prediction_columns = [
        "seed",
        "sample_id",
        "actual_yield_kg_ha",
        "predicted_yield_kg_ha",
    ]
    missing = set(required_prediction_columns) - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    require_finite(
        predictions,
        ["actual_yield_kg_ha", "predicted_yield_kg_ha"],
    )
    if predictions["sample_id"].duplicated().sum() != 82:
        raise ValueError("Expected exactly three prediction rows per 41 test samples")
    counts = predictions.groupby("sample_id")["seed"].nunique()
    if len(counts) != 41 or not (counts == 3).all():
        raise ValueError("Expected exactly three seeds for each of 41 test samples")

    predictions = predictions.merge(samples, on="sample_id", how="left", validate="many_to_one")
    if predictions[["district", "year", "season"]].isna().any().any():
        raise ValueError("Could not resolve district/year/season for every prediction")

    seed_metrics = []
    for seed, group in predictions.groupby("seed", sort=True):
        actual = group["actual_yield_kg_ha"].to_numpy(dtype=np.float64)
        predicted = group["predicted_yield_kg_ha"].to_numpy(dtype=np.float64)
        error = predicted - actual
        seed_metrics.append(
            {
                "seed": int(seed),
                "mae_kg_ha": float(np.mean(np.abs(error))),
                "rmse_kg_ha": float(np.sqrt(np.mean(error**2))),
                "r2": float(
                    1.0
                    - np.sum(error**2)
                    / np.sum((actual - actual.mean()) ** 2)
                ),
            }
        )

    analysis = (
        predictions.groupby(
            ["sample_id", "district", "year", "season"], as_index=False
        )
        .agg(
            actual_yield_kg_ha=("actual_yield_kg_ha", "first"),
            predicted_yield_kg_ha=("predicted_yield_kg_ha", "mean"),
            prediction_std_kg_ha=("predicted_yield_kg_ha", "std"),
            seed_count=("seed", "nunique"),
        )
    )
    analysis["signed_error_kg_ha"] = (
        analysis["predicted_yield_kg_ha"] - analysis["actual_yield_kg_ha"]
    )
    analysis["absolute_error_kg_ha"] = analysis["signed_error_kg_ha"].abs()
    analysis["squared_error_kg_ha2"] = analysis["signed_error_kg_ha"] ** 2
    analysis = analysis.sort_values("absolute_error_kg_ha", ascending=False)
    analysis.to_csv(RESULTS / "error_analysis.csv", index=False)

    actual = analysis["actual_yield_kg_ha"].to_numpy(dtype=np.float64)
    predicted = analysis["predicted_yield_kg_ha"].to_numpy(dtype=np.float64)
    signed_error = analysis["signed_error_kg_ha"].to_numpy(dtype=np.float64)
    absolute_error = analysis["absolute_error_kg_ha"].to_numpy(dtype=np.float64)
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    bias = float(signed_error.mean())
    q1, median, q3 = np.percentile(absolute_error, [25, 50, 75])

    district_summary = (
        analysis.groupby("district")
        .agg(
            samples=("sample_id", "count"),
            mean_absolute_error_kg_ha=("absolute_error_kg_ha", "mean"),
            median_absolute_error_kg_ha=("absolute_error_kg_ha", "median"),
            mean_signed_error_kg_ha=("signed_error_kg_ha", "mean"),
        )
        .sort_values("mean_absolute_error_kg_ha", ascending=False)
    )
    season_summary = (
        analysis.groupby("season")
        .agg(
            samples=("sample_id", "count"),
            mean_absolute_error_kg_ha=("absolute_error_kg_ha", "mean"),
            median_absolute_error_kg_ha=("absolute_error_kg_ha", "median"),
            mean_signed_error_kg_ha=("signed_error_kg_ha", "mean"),
        )
        .sort_values("mean_absolute_error_kg_ha", ascending=False)
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(actual, predicted, s=42, alpha=0.78, edgecolor="white", linewidth=0.4)
    limits = [float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))]
    ax.plot(limits, limits, linestyle="--", color="#333333", linewidth=1.4, label="Ideal")
    ax.set_xlabel("Observed yield (kg/ha)")
    ax.set_ylabel("Mean predicted yield (kg/ha)")
    ax.set_title("Final model: observed vs predicted yield")
    ax.legend(frameon=True)
    save_plot(fig, RESULTS / "actual_vs_predicted_final.png")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(signed_error, bins=12, color="#4C78A8", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.3)
    ax.axvline(bias, color="#E45756", linewidth=1.5, label=f"Mean bias = {bias:.1f}")
    ax.set_xlabel("Signed prediction error (kg/ha)")
    ax.set_ylabel("Number of test samples")
    ax.set_title("Distribution of final prediction errors")
    ax.legend(frameon=True)
    save_plot(fig, RESULTS / "prediction_error_distribution.png")

    district_plot = district_summary.sort_values("mean_absolute_error_kg_ha")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        district_plot.index,
        district_plot["mean_absolute_error_kg_ha"],
        color="#72B7B2",
    )
    ax.set_xlabel("Mean absolute error (kg/ha)")
    ax.set_ylabel("District")
    ax.set_title("Final prediction error by district")
    save_plot(fig, RESULTS / "error_by_district.png")

    season_plot = season_summary.sort_values("mean_absolute_error_kg_ha")
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(
        season_plot.index,
        season_plot["mean_absolute_error_kg_ha"],
        color=["#F58518" if s == "Kharif" else "#54A24B" for s in season_plot.index],
    )
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean absolute error (kg/ha)")
    ax.set_title("Final prediction error by season")
    save_plot(fig, RESULTS / "error_by_season.png")

    largest_over = analysis.nlargest(5, "signed_error_kg_ha")
    largest_under = analysis.nsmallest(5, "signed_error_kg_ha")
    report: list[str] = [
        "# Final Error Analysis",
        "",
        "This analysis uses only `final_test_predictions.csv`. Because that file "
        "contains one prediction for each of three final seeds, sample-level "
        "statistics below use the mean prediction across the three seeds for each "
        "of the 41 test samples. Seed-specific metrics are retained separately "
        "below. No model was loaded or retrained.",
        "",
        "## Absolute error statistics",
        "",
        f"- Samples analyzed: **{len(analysis)}**",
        f"- Mean absolute error: **{absolute_error.mean():.3f} kg/ha**",
        f"- Median absolute error: **{median:.3f} kg/ha**",
        f"- Standard deviation: **{absolute_error.std(ddof=1):.3f} kg/ha**",
        f"- Minimum absolute error: **{absolute_error.min():.3f} kg/ha**",
        f"- Maximum absolute error: **{absolute_error.max():.3f} kg/ha**",
        f"- Interquartile range: **{q1:.3f}–{q3:.3f} kg/ha**",
        "",
        "## Signed error and bias",
        "",
        f"- Mean signed error (prediction − observation): **{bias:.3f} kg/ha**",
        f"- Median signed error: **{np.median(signed_error):.3f} kg/ha**",
        f"- Overpredictions: **{int((signed_error > 0).sum())}** samples",
        f"- Underpredictions: **{int((signed_error < 0).sum())}** samples",
        f"- Exact-zero errors: **{int((signed_error == 0).sum())}** samples",
        "A positive mean signed error indicates average overprediction; a negative "
        "value indicates average underprediction.",
        "",
        "## Actual versus predicted relationship",
        "",
        f"- Mean actual yield: **{actual.mean():.3f} kg/ha**",
        f"- Median actual yield: **{np.median(actual):.3f} kg/ha**",
        f"- Mean predicted yield: **{predicted.mean():.3f} kg/ha**",
        f"- Median predicted yield: **{np.median(predicted):.3f} kg/ha**",
        f"- Pearson correlation: **{correlation:.4f}**",
        "The correlation describes association in this holdout; it does not establish "
        "causation or imply that any input modality caused an error.",
        "",
        "## Largest overpredictions",
        "",
        "| Sample | District | Season | Actual | Predicted | Signed error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in largest_over.itertuples():
        report.append(
            f"| {row.sample_id} | {row.district} | {row.season} | "
            f"{row.actual_yield_kg_ha:.1f} | {row.predicted_yield_kg_ha:.1f} | "
            f"+{row.signed_error_kg_ha:.1f} |"
        )
    report += [
        "",
        "## Largest underpredictions",
        "",
        "| Sample | District | Season | Actual | Predicted | Signed error |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in largest_under.itertuples():
        report.append(
            f"| {row.sample_id} | {row.district} | {row.season} | "
            f"{row.actual_yield_kg_ha:.1f} | {row.predicted_yield_kg_ha:.1f} | "
            f"{row.signed_error_kg_ha:.1f} |"
        )
    report += [
        "",
        "## Error by district",
        "",
        markdown_table(district_summary),
        "",
        "## Error by season",
        "",
        markdown_table(season_summary),
        "",
        "## Error distribution and visible patterns",
        "",
        "The plots show the observed error distribution and grouped error summaries. "
        f"The mean bias is {bias:.1f} kg/ha, so the aggregate sample-level errors "
        f"are {'slightly positive' if bias > 0 else 'slightly negative' if bias < 0 else 'centered near zero'} "
        "on average. Group differences are descriptive patterns in this 41-sample "
        "holdout only. They should not be interpreted as causal district or seasonal "
        "effects, especially because the sample size per group is limited.",
        "",
        "## Seed-specific final metrics",
        "",
        markdown_table(pd.DataFrame(seed_metrics)),
        "",
        "## Limitations",
        "",
        "This is a small held-out test set with three stochastic final models. "
        "Sample-level values use the mean across seeds, which summarizes prediction "
        "variability but is not a separately trained ensemble. Error groupings are "
        "associational and do not identify causes. No additional tuning or retraining "
        "was performed.",
    ]
    (RESULTS / "error_analysis.md").write_text("\n".join(report), encoding="utf-8")

    final_summary = [
        "# Final Results Summary",
        "",
        "## Locked final model",
        "",
        "- Concatenation Fusion",
        "- Learning rate: 0.0005",
        "- Fusion dropout: 0.0",
        "- Optimizer: AdamW",
        "- Loss: MSELoss",
        "- Batch size: 16",
        "- Epochs: 9",
        "- Training data: 198 train+validation samples",
        "",
        "## Final three-seed test metrics",
        "",
        "- MAE: **591.175 +/- 68.889 kg/ha**",
        "- RMSE: **758.353 +/- 84.317 kg/ha**",
        "- R²: **0.328446 +/- 0.143360**",
        "",
        "## Key error-analysis findings",
        "",
        f"- Mean actual yield: **{actual.mean():.3f} kg/ha**; mean predicted yield: "
        f"**{predicted.mean():.3f} kg/ha**.",
        f"- Mean absolute error across mean-per-sample predictions: **{absolute_error.mean():.3f} kg/ha**.",
        f"- Mean signed error: **{bias:.3f} kg/ha**, indicating "
        f"{'average overprediction' if bias > 0 else 'average underprediction' if bias < 0 else 'no average signed bias'}.",
        f"- Pearson observed-predicted correlation: **{correlation:.4f}**.",
        "- District and season differences are descriptive observations from the "
        "holdout, not causal findings.",
        "",
        "## Limitations",
        "",
        "The analysis covers 41 test samples and aggregates three seed predictions "
        "per sample. Small group sizes limit the stability of district and season "
        "comparisons. The results do not establish why individual predictions are "
        "wrong and should not be generalized beyond the evaluated holdout.",
    ]
    (RESULTS / "final_results_summary.md").write_text(
        "\n".join(final_summary), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
