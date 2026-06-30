import streamlit as st
import requests
import pandas as pd

# ==================================================
# Page Config
# ==================================================

st.set_page_config(
    page_title="Cyclone Track Forecasting System",
    page_icon="🌪️",
    layout="wide"
)

# ==================================================
# Header
# ==================================================

st.title("🌪️ Cyclone Track Forecasting System")

st.markdown(
    """
    Predict the next cyclone position using a
    Random Forest model trained on IBTrACS data.
    """
)

# ==================================================
# Inputs
# ==================================================

col1, col2 = st.columns(2)

with col1:

    latitude = st.number_input(
        "Latitude",
        value=15.0,
        format="%.4f"
    )

    longitude = st.number_input(
        "Longitude",
        value=85.0,
        format="%.4f"
    )

with col2:

    wind_speed = st.number_input(
        "Wind Speed (knots)",
        value=60.0
    )

    pressure = st.number_input(
        "Pressure (hPa)",
        value=980.0
    )

# ==================================================
# Predict Button
# ==================================================

if st.button("Predict Cyclone Position"):

    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "wind_speed": wind_speed,
        "pressure": pressure,
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        # ==========================================
        # Success Message
        # ==========================================

        st.success(
            "Prediction Generated Successfully!"
        )

        # ==========================================
        # Prediction Results
        # ==========================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Latitude",
                f"{result['predicted_lat']:.4f}"
            )

        with col2:

            st.metric(
                "Predicted Longitude",
                f"{result['predicted_lon']:.4f}"
            )

        # ==========================================
        # Map Visualization
        # ==========================================

        st.subheader("🗺️ Cyclone Location Map")

        map_df = pd.DataFrame(
            {
                "lat": [
                    latitude,
                    result["predicted_lat"],
                ],
                "lon": [
                    longitude,
                    result["predicted_lon"],
                ],
            }
        )

        st.map(map_df)

        # ==========================================
        # Model Information
        # ==========================================

        st.subheader("📊 Model Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Model",
                result["model"]
            )

        with c2:
            st.metric(
                "Forecast Horizon",
                result["forecast_horizon"]
            )

        with c3:
            st.metric(
                "Track Error (km)",
                result["track_error_km"]
            )

    except Exception as e:

        st.error(
            f"API Error: {e}"
        )