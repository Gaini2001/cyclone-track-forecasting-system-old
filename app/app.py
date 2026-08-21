"""
app.py

Streamlit dashboard for the Cyclone Track Forecast API.

This is a thin client: it holds no model and does no feature engineering of
its own. Every forecast is a POST to the API's /predict endpoint, over HTTP,
exactly like the curl command Swagger shows you. That is what makes it a
second *deployment* rather than a second copy of the model -- see
DEPLOYMENT.md and docker-compose.yml, where this runs as its own container
that talks to the api container over the network.

Track construction
-------------------
The API answers one horizon at a time, from a fixed observation history --
it does not know about "the other four" horizons in a single call. This page
calls /predict once per configured horizon (6, 12, 24, 48, 72h), all from the
same input history, and strings the results together into a forecast track.
That mirrors the "direct" strategy in src.s6_inference.predict: each lead
time is its own independent forecast from the same starting state, not a
chain of forecasts feeding each other.

Usage
-----
    streamlit run app/app.py
    API_URL=http://api:8000 streamlit run app/app.py   # pointed at a container
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 10

# Restricted to what the deploy image actually ships (see Dockerfile.deploy):
# XGBoost only, at 6/12/24h. Random Forest and the 48/72h horizons exist in
# the full local model set but are deliberately not offered here, so the
# dashboard never promises a forecast the deployed API can't serve.
MODEL = "xgboost"
DASHBOARD_HORIZONS = (6, 12, 24)

DEFAULT_OBSERVATIONS = [
    {"latitude": 13.8, "longitude": 83.4, "wind_speed": 50.0, "pressure": 990.0},
    {"latitude": 14.0, "longitude": 83.7, "wind_speed": 52.0, "pressure": 988.0},
    {"latitude": 14.2, "longitude": 84.0, "wind_speed": 55.0, "pressure": 985.0},
    {"latitude": 14.5, "longitude": 84.6, "wind_speed": 58.0, "pressure": 982.0},
    {"latitude": 15.0, "longitude": 85.2, "wind_speed": 60.0, "pressure": 980.0},
]

st.set_page_config(page_title="Cyclone Track Forecast", layout="wide")


# ==========================================================
# API client
# ==========================================================

def fetch_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def fetch_prediction(observations: list[dict], horizon: int, model: str) -> dict:
    """
    One /predict call. Raises for anything other than a clean 200 -- the
    caller decides what a missing horizon or an unreachable API means for the
    track as a whole; this function just reports what happened.
    """

    response = requests.post(
        f"{API_URL}/predict",
        json={"observations": observations, "forecast_horizon": horizon, "model": model},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(f"{response.status_code}: {detail}")

    return response.json()


# ==========================================================
# Map
# ==========================================================

def build_track_figure(observations: list[dict], forecasts: list[dict]) -> go.Figure:
    """
    Observed history and forecast track on a world map.

    Plotly's Scattergeo draws its own coastlines rather than fetching map
    tiles, so this works with no internet access from inside a container --
    unlike st.map/pydeck, which need a live tile server.
    """

    fig = go.Figure()

    obs_lat = [o["latitude"] for o in observations]
    obs_lon = [o["longitude"] for o in observations]

    fig.add_trace(go.Scattergeo(
        lat=obs_lat,
        lon=obs_lon,
        mode="lines+markers",
        name="Observed",
        line=dict(color="#2b6cb0", width=2),
        marker=dict(size=7, color="#2b6cb0"),
        hovertext=[f"Observed #{i + 1}" for i in range(len(observations))],
    ))

    if forecasts:
        # The forecast track starts at the last observed position, so the two
        # lines visibly connect instead of leaving a gap.
        fc_lat = [obs_lat[-1]] + [f["predicted_lat"] for f in forecasts]
        fc_lon = [obs_lon[-1]] + [f["predicted_lon"] for f in forecasts]

        fig.add_trace(go.Scattergeo(
            lat=fc_lat,
            lon=fc_lon,
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#e53e3e", width=2, dash="dash"),
            marker=dict(size=8, color="#e53e3e", symbol="diamond"),
            hovertext=["Last observed"] + [f"+{f['forecast_horizon']}h" for f in forecasts],
        ))

    fig.update_geos(
        showland=True, landcolor="#e8e4d8",
        showocean=True, oceancolor="#cfe3ee",
        showcountries=True, countrycolor="#b8b2a0",
        showcoastlines=True, coastlinecolor="#8a8570",
        fitbounds="locations",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=560,
    )

    return fig


# ==========================================================
# Layout
# ==========================================================

st.title("Cyclone Track Forecast")
st.caption(f"API: {API_URL}")

health = fetch_health()

with st.sidebar:
    st.header("Service")

    if health is None:
        st.error("API unreachable. Is it running?")
        st.code(f"uvicorn app.main:app --host 0.0.0.0 --port 8000", language="bash")
        models_available: dict[str, list[int]] = {}
    else:
        status = health.get("status", "unknown")
        (st.success if status == "healthy" else st.warning)(f"Status: {status}")
        models_available = health.get("models_available", {})

        if not models_available:
            st.warning("No trained models found on the API. Train one first.")

    st.header("Forecast settings")

    model = MODEL
    st.caption(f"Model: **{model}**")

    if models_available and model not in models_available:
        st.error(f"'{model}' is not on the API. Train or mount it first.")

    served_horizons = models_available.get(model, [])
    horizon_options = [h for h in DASHBOARD_HORIZONS if not served_horizons or h in served_horizons]

    horizons = st.multiselect(
        "Horizons (hours)",
        options=horizon_options,
        default=horizon_options,
        help="Each horizon is a separate forecast from the same observation "
             "history, not a chain -- see the module docstring.",
    )

st.subheader("Observation history")
st.caption(
    "Consecutive 6-hourly positions, oldest first. Five gives the model its "
    "full lag and rolling-window features; fewer still works, with a warning."
)

observations_df = st.data_editor(
    DEFAULT_OBSERVATIONS,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "latitude": st.column_config.NumberColumn(min_value=-90.0, max_value=90.0, format="%.2f"),
        "longitude": st.column_config.NumberColumn(min_value=-180.0, max_value=180.0, format="%.2f"),
        "wind_speed": st.column_config.NumberColumn(min_value=0.0, max_value=250.0, format="%.1f kt"),
        "pressure": st.column_config.NumberColumn(min_value=800.0, max_value=1050.0, format="%.1f hPa"),
    },
)

run = st.button("Forecast", type="primary", disabled=health is None)

if run:
    observations = [dict(row) for row in observations_df]

    if not observations:
        st.error("Add at least one observation.")
    elif not horizons:
        st.error("Select at least one horizon.")
    else:
        forecasts, errors = [], []

        for horizon in sorted(horizons):
            try:
                forecasts.append(fetch_prediction(observations, horizon, model))
            except (requests.RequestException, RuntimeError) as exc:
                errors.append(f"{horizon}h: {exc}")

        for message in errors:
            st.warning(message)

        if any("warning" in f for f in forecasts):
            st.info(forecasts[0]["warning"])

        if forecasts:
            st.plotly_chart(
                build_track_figure(observations, forecasts), use_container_width=True
            )

            st.dataframe(
                [
                    {
                        "Horizon": f"{f['forecast_horizon']}h",
                        "Predicted lat": round(f["predicted_lat"], 3),
                        "Predicted lon": round(f["predicted_lon"], 3),
                        "Model": f["model"],
                        "Mean track error (km)": f.get("track_error_km"),
                    }
                    for f in forecasts
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.error("No forecast could be produced. See the messages above.")
