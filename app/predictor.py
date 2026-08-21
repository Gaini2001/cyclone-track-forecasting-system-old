"""
predictor.py

Ties a request to a loaded model and a geographic forecast.

Models are loaded once per (name, horizon) and cached for the life of the
process -- the alternative is unpickling a multi-megabyte artifact on every
request, which is most of the latency a demo API would otherwise have.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from src.s5_training.save_model import load_model
from src.utils.config import (
    FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_HORIZON,
    TARGET_MODE,
    EVALUATION_RESULTS_PATH,
    model_path,
)
from src.utils.metrics import offset_position

from app.feature_builder import build_feature_row

# Display names used in reports/evaluation_results.json -- see
# src.s5_training.evaluate, which writes one row per model there.
_MODEL_DISPLAY_NAMES = {"xgboost": "XGBoost", "random_forest": "Random Forest"}

_FULL_HISTORY = 5  # three lags + a 24h (4-step) rolling window


@lru_cache(maxsize=None)
def _load(model_name: str, horizon: int):
    return load_model(model_path(model_name, horizon, TARGET_MODE))


@lru_cache(maxsize=1)
def _evaluation_results() -> list[dict]:
    if not EVALUATION_RESULTS_PATH.exists():
        return []
    return json.loads(EVALUATION_RESULTS_PATH.read_text())


def _track_error_km(model_name: str, horizon: int) -> float | None:
    """
    Measured mean track error for this model and horizon, from the last
    evaluation run -- reported rather than hardcoded, so it can't drift from
    what the model actually does.
    """

    display_name = _MODEL_DISPLAY_NAMES.get(model_name)

    for entry in _evaluation_results():
        if entry.get("model") == display_name and entry.get("horizon") == horizon:
            return round(entry["mean_km"], 1)

    return None


def available_models() -> dict[str, list[int]]:
    """
    Horizons actually present on disk, per model. What /health reports and
    what /predict is allowed to serve.
    """

    available = {}

    for name in _MODEL_DISPLAY_NAMES:
        horizons = sorted(
            h for h in FORECAST_HORIZON_HOURS
            if model_path(name, h, TARGET_MODE).exists()
        )
        if horizons:
            available[name] = horizons

    return available


def predict_location(payload: dict) -> dict:
    """
    Forecast one position from an observation history.

    Parameters
    ----------
    payload : dict
        `CycloneInput.model_dump()`: observations, forecast_horizon, model.

    Raises
    ------
    FileNotFoundError
        No trained artifact for the requested model/horizon. The caller (see
        app.main) turns this into a 503 -- the request was well-formed, the
        service just isn't equipped to answer it right now.
    """

    observations = payload["observations"]
    horizon = payload.get("forecast_horizon", DEFAULT_FORECAST_HORIZON)
    model_name = payload.get("model", "xgboost")

    model = _load(model_name, horizon)
    features = build_feature_row(observations)

    prediction = np.asarray(model.predict(features), dtype=float)
    delta_lat, delta_lon = float(prediction[0, 0]), float(prediction[0, 1])

    last = observations[-1]
    pred_lat, pred_lon = offset_position(
        last["latitude"], last["longitude"], delta_lat, delta_lon
    )

    result = {
        "predicted_lat": float(pred_lat),
        "predicted_lon": float(pred_lon),
        "model": model_name,
        "forecast_horizon": horizon,
        "track_error_km": _track_error_km(model_name, horizon),
        "n_observations": len(observations),
    }

    if len(observations) < _FULL_HISTORY:
        result["warning"] = (
            f"Only {len(observations)} observation(s) supplied; the model "
            f"uses up to {_FULL_HISTORY} for its lag and rolling-window "
            "features. The missing ones were left unset rather than "
            "fabricated, so this forecast is less informed than a full-"
            "history one."
        )

    return result
