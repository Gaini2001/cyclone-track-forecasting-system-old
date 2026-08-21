"""
schemas.py

Request and response models for the prediction API.

Bounds mirror src.utils.config: MIN/MAX_WIND_SPEED, MIN/MAX_PRESSURE and the
geographic ranges. Duplicating them as Field constraints (rather than
importing and wrapping in a validator) is deliberate here -- these are the
values a client sees in the OpenAPI schema and in a 422 response, so they need
to be visible in this file, not one hop away in config.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from src.utils.config import (
    FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_HORIZON,
    MIN_WIND_SPEED,
    MAX_WIND_SPEED,
    MIN_PRESSURE,
    MAX_PRESSURE,
)


class Observation(BaseModel):
    """
    A single storm position report.

    The model needs three lags plus a 24h rolling window, i.e. up to five
    consecutive 6-hourly observations (see Observation history below). Fewer
    are accepted -- the missing lag and motion features are left unset rather
    than fabricated -- but the forecast is correspondingly less informed, and
    the response says so.
    """

    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    wind_speed: float = Field(..., ge=MIN_WIND_SPEED, le=MAX_WIND_SPEED,
                               description="1-minute sustained wind, knots.")
    pressure: float = Field(..., ge=MIN_PRESSURE, le=MAX_PRESSURE,
                             description="Central pressure, hPa.")


class CycloneInput(BaseModel):
    """
    A forecast request: an observation history, oldest first, ending at the
    position to forecast from.
    """

    observations: list[Observation] = Field(
        ..., min_length=1,
        description="Consecutive 6-hourly observations, oldest first. Five "
                     "gives the model its full feature set.",
    )
    forecast_horizon: int = Field(
        DEFAULT_FORECAST_HORIZON,
        description=f"Lead time in hours. One of {list(FORECAST_HORIZON_HOURS)}.",
    )
    model: str = Field(
        "xgboost",
        description="Which trained model to serve the forecast from.",
    )

    @field_validator("forecast_horizon")
    @classmethod
    def _horizon_is_configured(cls, value: int) -> int:
        if value not in FORECAST_HORIZON_HOURS:
            raise ValueError(
                f"forecast_horizon must be one of {list(FORECAST_HORIZON_HOURS)}, got {value}"
            )
        return value

    @field_validator("model")
    @classmethod
    def _model_is_known(cls, value: str) -> str:
        if value not in ("xgboost", "random_forest"):
            raise ValueError("model must be 'xgboost' or 'random_forest'")
        return value
