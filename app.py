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

DISTRICTS = [
    "ANANTPUR",
    "CHITTOOR",
    "CUDDAPPAH",
    "EAST GODAVARI",
    "GUNTUR",
    "KRISHNA",
    "KURNOOL",
    "NELLORE",
    "PRAKASAM",
    "SRIKAKULAM",
    "VISAKHAPATNAM",
    "VIZIANAGARM",
    "WEST GODAVARI",
]

st.set_page_config(page_title="Rice Yield Prediction System", page_icon="🌾", layout="wide")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 1.5rem 1.75rem; border-radius: 18px; background: linear-gradient(135deg, #12372a 0%, #1f6f4a 100%); color: white; margin-bottom: 1.2rem; }
        .hero h1 { margin: 0; color: white; font-size: 2.35rem; letter-spacing: -0.03em; }
        .hero p { margin: 0.35rem 0 0; color: #d9f4e5; font-size: 1.08rem; }
        .section-title { color: #12372a; font-size: 1.25rem; font-weight: 700; margin: 0.2rem 0 0.8rem; }
        .section-caption { color: #5c6b63; font-size: 0.88rem; margin: -0.35rem 0 0.85rem; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; border-color: #d7e5dc; background: #fbfdfb; }
        div.stButton > button[kind="primary"] { display: block; margin: 0.5rem auto 0; min-width: 280px; border-radius: 10px; font-weight: 700; letter-spacing: 0.04em; }
        .result-card { padding: 1.25rem 1.5rem; border-radius: 16px; background: #eef8f1; border: 1px solid #b9ddc4; margin: 0.8rem 0 1rem; }
        .result-label { color: #356247; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.2rem; }
        .result-value { color: #12372a; font-size: 2.3rem; font-weight: 800; line-height: 1.1; }
        .result-unit { color: #567461; font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, caption: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-caption">{caption}</div>', unsafe_allow_html=True)


@st.cache_resource
def cached_model(seed_label: str, device_name: str):
    return load_model(CHECKPOINTS[seed_label], torch.device(device_name))


def main() -> None:
    apply_styles()
    st.markdown(
        """
        <div class="hero">
            <h1>🌾 Rice Yield Prediction System</h1>
            <p>Multi-Modal Deep Learning for Rice Yield Estimation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "This demonstration combines **satellite imagery**, **weather time-series**, "
        "and **soil properties** to estimate rice yield from supplied inputs."
    )
    st.info(
        "Inference/demo system using supplied satellite, weather, and soil inputs. "
        "It does not connect to live APIs and does not produce a scientifically "
        "validated forecast from unknown future weather."
    )

    with st.container(border=True):
        section_heading(
            "📍 Location & Crop Information",
            "Metadata identifies the prediction context; it is not passed into the model.",
        )
        metadata_col, model_col = st.columns(2)
        with metadata_col:
            district = st.selectbox("District", DISTRICTS)
            season = st.selectbox("Season", ["Kharif", "Rabi"])
            year = st.text_input(
                "Year",
                value="",
                placeholder="Enter a historical year or year label, e.g. 2010-11",
            ).strip()
        with model_col:
            seed_label = st.selectbox("Final checkpoint", list(CHECKPOINTS))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.caption(f"Device available for inference: `{device}`")
        st.caption(
            "Demo samples are available in "
            "`data/demo_samples/<DISTRICT>/<YEAR_SEASON>/`."
        )

    with st.container(border=True):
        section_heading(
            "🛰️ Satellite Data",
            "Upload a prepared satellite tensor in the model-compatible NumPy format.",
        )
        satellite_file = st.file_uploader("Upload satellite .npy", type=["npy"])
        st.caption("Expected shape: [7,128,128] or [128,128,7]. Channel 7 is NDVI.")

    with st.container(border=True):
        section_heading(
            "🌦️ Weather Data",
            "Upload the 183-day weather sequence used for inference.",
        )
        weather_file = st.file_uploader("Upload weather CSV", type=["csv"])
        st.caption(
            "Expected 183 rows and these feature columns in this order: "
            + ", ".join(EXPECTED_WEATHER_COLUMNS)
            + ". An optional date column is ignored."
        )

    with st.container(border=True):
        section_heading(
            "🌱 Soil Properties",
            "Provide the four soil features in the same representation used during training.",
        )
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

    action_left, action_center, action_right = st.columns([1, 2, 1])
    with action_center:
        predict_clicked = st.button("PREDICT YIELD", type="primary", use_container_width=True)
    if predict_clicked:
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
        st.markdown("## Predicted Rice Yield")
        result_col, detail_col = st.columns([1, 1.35])
        with result_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated production</div>
                    <div class="result-value">{prediction:,.2f}</div>
                    <div class="result-unit">kg/ha</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Equivalent yield", f"{prediction / 1000:,.3f} tonnes/ha")
        with detail_col:
            with st.container(border=True):
                st.markdown("### Prediction Context")
                context_col1, context_col2 = st.columns(2)
                context_col1.metric("District", district)
                context_col1.metric("Season", season)
                context_col2.metric("Year", year)
                context_col2.metric("Model seed", seed_label.replace("Seed ", ""))
                st.caption(f"Inference device: `{device}`")

        with st.container(border=True):
            st.markdown("### Input Analysis")
            analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
            analysis_col1.metric("Satellite", f"{satellite.shape[1]} × {satellite.shape[2]}")
            analysis_col2.metric("Weather", f"{weather.shape[0]} × {weather.shape[1]}")
            analysis_col3.metric("Soil features", "4")
            analysis_col4.metric("NDVI mean", f"{satellite[6].mean():.4f}")
            st.caption(
                f"NDVI range: {satellite[6].min():.4f} to {satellite[6].max():.4f} · "
                f"Soil: nitrogen={soil[0]:.3f}, pH={soil[1]:.3f}, "
                f"SOC={soil[2]:.3f}, clay={soil[3]:.3f}"
            )

    with st.expander("Model Information"):
        info_col1, info_col2 = st.columns(2)
        info_col1.write("**Satellite encoder:** CNN")
        info_col1.write("**Weather encoder:** LSTM")
        info_col1.write("**Soil encoder:** MLP")
        info_col2.write("**Fusion:** Concatenation Fusion")
        info_col2.write("**Output:** rice yield in kg/ha")
        parameter_count = sum(
            parameter.numel()
            for parameter in cached_model(seed_label, str(device))[0].parameters()
        )
        st.metric("Trainable parameters", f"{parameter_count:,}")


if __name__ == "__main__":
    main()
