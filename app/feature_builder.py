"""
feature_builder.py

Turns a short observation history from an API request into the single feature
row a trained model expects.

This calls the real functions from src.s3_features.feature_engineering, in the
same order main() there does, rather than re-implementing the arithmetic.
Two implementations of "what a lag feature is" can only drift apart; one
implementation used from both places cannot.

What a request does not supply
-------------------------------
The request carries only position, wind and pressure -- STORM_SPEED,
STORM_DIR, DIST2LAND and BASIN are IBTrACS-reported fields nothing at request
time can measure. STORM_SPEED and STORM_DIR are approximated from consecutive
positions (which is, after all, most of what they mean); DIST2LAND and BASIN
have no such proxy and are left unset, which src.s3_features.feature_engineering
already treats as "column absent" rather than "value zero". A model asked to
predict from these features sees them as missing, not wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.s3_features.feature_engineering import (
    add_temporal_features,
    add_lag_features,
    add_motion_features,
    add_velocity_features,
    add_turning_features,
    add_rolling_features,
    add_acceleration_features,
    add_storm_context_features,
)
from src.utils.config import FEATURE_COLUMNS, OBSERVATION_INTERVAL
from src.utils.metrics import haversine_distance, compute_bearing

KM_PER_HOUR_TO_KNOTS = 1.0 / 1.852

# A placeholder SID/SEGMENT_ID: STORM_AGE ends up measured from the start of
# the supplied history rather than true genesis, since genesis is not part of
# the request. Documented, not hidden -- see the module docstring.
_LIVE_SID = "LIVE"


def _synthesize_raw_frame(observations: list[dict]) -> pd.DataFrame:
    """
    Build the raw-column frame the feature pipeline expects from a list of
    {latitude, longitude, wind_speed, pressure} dicts, oldest first.
    """

    n = len(observations)
    now = pd.Timestamp.now("UTC").tz_localize(None)
    times = pd.date_range(end=now, periods=n, freq=f"{OBSERVATION_INTERVAL}h")

    lat = np.array([o["latitude"] for o in observations], dtype=float)
    lon = np.array([o["longitude"] for o in observations], dtype=float)

    speed = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    if n > 1:
        distance = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
        bearing = compute_bearing(
            lat[:-1], lon[:-1], lat[1:], lon[1:], undefined_as_nan=True
        )
        speed[1:] = distance / OBSERVATION_INTERVAL * KM_PER_HOUR_TO_KNOTS
        direction[1:] = bearing

    return pd.DataFrame({
        "SID": _LIVE_SID,
        "SEGMENT_ID": f"{_LIVE_SID}_S01",
        "ISO_TIME": times,
        "LAT": lat,
        "LON": lon,
        "WMO_WIND": [o["wind_speed"] for o in observations],
        "WMO_PRES": [o["pressure"] for o in observations],
        "STORM_SPEED": speed,
        "STORM_DIR": direction,
    })


def build_feature_row(observations: list[dict]) -> pd.DataFrame:
    """
    Run the observation history through the real feature pipeline and return
    the single row -- the most recent observation -- a model predicts from.

    Returns
    -------
    pd.DataFrame
        One row, columns in FEATURE_COLUMNS order. Features the request could
        not supply enough history for come back as NaN; XGBoost's default
        missing-value handling routes those splits rather than erroring.
    """

    df = _synthesize_raw_frame(observations)

    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_motion_features(df)
    df = add_velocity_features(df)
    df = add_turning_features(df)
    df = add_rolling_features(df)
    df = add_acceleration_features(df)
    df = add_storm_context_features(df)

    row = df.iloc[[-1]].copy()

    for column in FEATURE_COLUMNS:
        if column not in row.columns:
            row[column] = np.nan

    return row[FEATURE_COLUMNS].reset_index(drop=True)
