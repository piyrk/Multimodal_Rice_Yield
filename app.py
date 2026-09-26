"""Inference-only Streamlit application for rice yield estimation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from deployment.inference import (  # noqa: E402
    EXPECTED_WEATHER_COLUMNS,
    load_model,
    predict,
    validate_satellite,
    validate_soil,
    validate_weather,
)


CHECKPOINTS = {
    "Seed 42": ROOT / "models" / "final_model_seed42.pt",
    "Seed 123": ROOT / "models" / "final_model_seed123.pt",
    "Seed 2026": ROOT / "models" / "final_model_seed2026.pt",
}

st.set_page_config(page_title="Rice Yield Prediction System", page_icon="🌾", layout="wide")


@st.cache_resource
def cached_model(seed_label: str, device_name: str):
    return load_model(CHECKPOINTS[seed_label], torch.device(device_name))


def main() -> None:
    st.title("Rice Yield Prediction System")
    st.subheader("Multimodal Deep Learning for Rice Yield Estimation")
    st.info(
        "Inference/demo system using supplied satellite, weather, and soil inputs. "
        "It does not connect to live APIs and does not produce a scientifically "
        "validated forecast from unknown future weather."
    )

    metadata_col, model_col = st.columns(2)
    with metadata_col:
        district = st.text_input("District", value="KRISHNA").strip()
        season = st.selectbox("Season", ["Kharif", "Rabi"])
        year = st.text_input("Year", value="2010-11").strip()
    with model_col:
        seed_label = st.selectbox("Final checkpoint", list(CHECKPOINTS))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.caption(f"Device available for inference: `{device}`")

    st.markdown("### Satellite input")
    satellite_file = st.file_uploader("Upload satellite .npy", type=["npy"])
    st.caption("Expected shape: [7,128,128] or [128,128,7]. Channel 7 is NDVI.")

    st.markdown("### Weather input")
    weather_file = st.file_uploader("Upload weather CSV", type=["csv"])
    st.caption(
        "Expected 183 rows and these feature columns in this order: "
        + ", ".join(EXPECTED_WEATHER_COLUMNS)
        + ". An optional date column is ignored."
    )

    st.markdown("### Soil input")
    soil_col1, soil_col2, soil_col3, soil_col4 = st.columns(4)
    nitrogen = soil_col1.number_input("Nitrogen (g/kg)", min_value=0.0, value=18.0)
    ph = soil_col2.number_input("pH", min_value=0.1, value=6.8)
    soc = soil_col3.number_input("SOC (g/kg)", min_value=0.0, value=27.0)
    clay = soil_col4.number_input("Clay (%)", min_value=0.0, max_value=100.0, value=30.0)

    satellite = weather = soil = None
    if satellite_file is not None:
        try:
            satellite = validate_satellite(satellite_file)
            st.success(f"Satellite validated: {list(satellite.shape)}")
            preview = satellite[6]
            fig, ax = plt.subplots(figsize=(5, 3))
            image = ax.imshow(preview, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_title("NDVI preview")
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.046)
            st.pyplot(fig, clear_figure=True)
        except ValueError as exc:
            st.error(str(exc))
    if weather_file is not None:
        try:
            weather = validate_weather(weather_file)
            st.success(f"Weather validated: {weather.shape[0]} observations × 3 features")
        except ValueError as exc:
            st.error(str(exc))
    try:
        soil = validate_soil(nitrogen, ph, soc, clay)
    except ValueError as exc:
        st.error(str(exc))

    if st.button("PREDICT YIELD", type="primary", use_container_width=True):
        if not district or not season or not year:
            st.error("District, season, and year are required.")
            return
        if satellite is None or weather is None or soil is None:
            st.error("Provide valid satellite, weather, and soil inputs before predicting.")
            return
        try:
            model, checkpoint = cached_model(seed_label, str(device))
            prediction = predict(
                model,
                satellite,
                weather,
                soil,
                checkpoint["target_mean"],
                checkpoint["target_std"],
                device,
            )
        except (FileNotFoundError, KeyError, RuntimeError, ValueError, FloatingPointError) as exc:
            st.error(f"Inference failed: {exc}")
            return

        st.success("Prediction completed.")
        result_col, detail_col = st.columns(2)
        with result_col:
            st.metric("Predicted rice yield", f"{prediction:,.2f} kg/ha")
            st.metric("Predicted yield", f"{prediction / 1000:,.3f} tonnes/ha")
            st.write(f"**Location:** {district} · {season} · {year}")
            st.write(f"**Model:** {seed_label}")
            st.write(f"**Device:** `{device}`")
        with detail_col:
            st.markdown("### Input Analysis")
            st.write(f"Satellite dimensions: `{list(satellite.shape)}`")
            st.write(f"Weather observations: `{weather.shape[0]} × {weather.shape[1]}`")
            st.write(
                "Soil features: "
                f"nitrogen={soil[0]:.3f}, pH={soil[1]:.3f}, "
                f"SOC={soil[2]:.3f}, clay={soil[3]:.3f}"
            )
            st.write(
                f"NDVI summary: mean={satellite[6].mean():.4f}, "
                f"min={satellite[6].min():.4f}, max={satellite[6].max():.4f}"
            )

    with st.expander("Model Information"):
        st.write("Satellite encoder: CNN")
        st.write("Weather encoder: LSTM")
        st.write("Soil encoder: MLP")
        st.write("Fusion: Concatenation Fusion")
        st.write("Output: rice yield in kg/ha")
        parameter_count = sum(
            parameter.numel()
            for parameter in cached_model(seed_label, str(device))[0].parameters()
        )
        st.write(f"Trainable parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
