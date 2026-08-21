"""
feature_engineering.py

Feature construction for cyclone track forecasting.

Four things distinguish this from a naive implementation:

1. Everything time-relative is grouped by SEGMENT_ID, not SID. Cleaning removes
   rows from the middle of storms; grouping by SID after that produces "6-hour"
   lags that are silently 24 hours old.

2. Targets are displacements by default. With absolute targets, LAT is both an
   input and essentially the answer, so R^2 saturates near 1.0 and reports
   nothing about forecast skill. Predicting (dlat, dlon) forces the model to
   learn the part that is actually hard. Absolute targets are emitted too, for
   evaluation and verification.

3. Longitude differences are wrapped rather than storms discarded. The original
   dropped every storm crossing the antimeridian, which removes most of the
   West Pacific -- the busiest basin there is. Wrapping the difference
   ((a - b + 180) % 360 - 180) handles the seam exactly and keeps the data.

4. Direction is encoded circularly. BEARING and STORM_DIR are compass
   quantities: 359 and 1 are adjacent, but as raw numbers they sit at opposite
   ends of the range.

Feature families
----------------
temporal      calendar components and cyclical encodings
lag           position, wind, pressure at t-1..t-n
motion        first differences, great-circle displacement, bearing
velocity      translation speed and u/v components in km/h
turning       rate of change of heading -- the recurvature signal
rolling       smoothed motion over the previous 24h
acceleration  second differences of position
context       storm age, intensity category, basin, latitude terms
targets       absolute and displacement, at every configured horizon
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils import get_logger, Timer
from src.utils.metrics import haversine_distance, compute_bearing, angular_difference
from src.utils.config import (
    CLEAN_DATA_PATH,
    CLEAN_PARQUET_PATH,
    FEATURE_DATA_PATH,
    FEATURE_PARQUET_PATH,
    FORECAST_HORIZONS,
    OBSERVATION_INTERVAL,
    INTENSITY_BINS,
    INTENSITY_LABELS,
    CSV_NA_VALUES,
)

logger = get_logger(__name__)

# ==========================================================
# Configuration
# ==========================================================

N_LAGS = 3

# Rolling windows in observations. 4 x 6h = 24h of recent motion.
ROLLING_WINDOWS = (4,)

GROUP_KEY = "SEGMENT_ID"

KM_PER_DEGREE_LAT = 111.19

# Stable basin encoding. Deriving codes from the categories present would make
# the mapping depend on which storms happen to be in the file.
BASIN_CODES = {
    "NA": 0,   # North Atlantic
    "EP": 1,   # Eastern North Pacific
    "WP": 2,   # Western North Pacific
    "NI": 3,   # North Indian
    "SI": 4,   # South Indian
    "SP": 5,   # South Pacific
    "SA": 6,   # South Atlantic
}


# ==========================================================
# Helpers
# ==========================================================

def wrapped_degree_diff(a, b):
    """
    Signed difference a - b for angular degrees, wrapped to (-180, 180].

    The reason the dateline filter is unnecessary. A storm moving from 179E to
    179W has travelled +2 degrees, not -358.
    """

    return ((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0) - 180.0


def _group(df: pd.DataFrame):
    """
    Grouper for all time-relative operations.
    """

    if GROUP_KEY not in df.columns:
        raise KeyError(
            f"{GROUP_KEY} missing. Run the updated clean_data.py first -- "
            "grouping by SID would let lags and targets span time gaps."
        )

    return df.groupby(GROUP_KEY, sort=False)


def _attach(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    Attach many columns at once.

    Assigning ~50 columns individually fragments the block manager and emits
    PerformanceWarning on pandas 2.x.
    """

    return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)


# ==========================================================
# Load
# ==========================================================

def load_clean_dataset() -> pd.DataFrame:
    """
    Load the cleaned dataset, preferring Parquet.
    """

    logger.info("Loading cleaned dataset...")

    if CLEAN_PARQUET_PATH.exists():
        df = pd.read_parquet(CLEAN_PARQUET_PATH)
    else:
        df = pd.read_csv(
            CLEAN_DATA_PATH,
            parse_dates=["ISO_TIME"],
            keep_default_na=False,
            na_values=CSV_NA_VALUES,
        )

    logger.info(f"Dataset loaded. Shape: {df.shape}")

    if GROUP_KEY not in df.columns:
        raise KeyError(
            f"{GROUP_KEY} not found in the cleaned dataset. "
            "Re-run clean_data.py with the segmentation stage."
        )

    return df.sort_values([GROUP_KEY, "ISO_TIME"], kind="mergesort").reset_index(drop=True)


# ==========================================================
# Temporal
# ==========================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar components plus cyclical encodings.

    Month and hour are periodic: December and January are adjacent, and 23:00
    is adjacent to 00:00. sin/cos encoding preserves that adjacency.
    """

    logger.info("Creating temporal features...")

    time = df["ISO_TIME"].dt

    month = time.month
    hour = time.hour
    day_of_year = time.dayofyear

    return _attach(df, {
        "MONTH": month,
        "DAY": time.day,
        "HOUR": hour,
        "DAY_OF_YEAR": day_of_year,
        "MONTH_SIN": np.sin(2 * np.pi * month / 12),
        "MONTH_COS": np.cos(2 * np.pi * month / 12),
        "HOUR_SIN": np.sin(2 * np.pi * hour / 24),
        "HOUR_COS": np.cos(2 * np.pi * hour / 24),
        # Seasonal position matters more than calendar month for basins whose
        # season straddles the new year.
        "DOY_SIN": np.sin(2 * np.pi * day_of_year / 365.25),
        "DOY_COS": np.cos(2 * np.pi * day_of_year / 365.25),
    })


# ==========================================================
# Lags
# ==========================================================

def add_lag_features(df: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """
    Past values of position and intensity, within the segment.
    """

    logger.info(f"Creating lag features (n_lags={n_lags}, grouped by {GROUP_KEY})...")

    sources = {
        "LAT": "LAT",
        "LON": "LON",
        "WMO_WIND": "WIND",
        "WMO_PRES": "PRESSURE",
    }

    grouped = _group(df)
    columns = {}

    for source, prefix in sources.items():
        for lag in range(1, n_lags + 1):
            columns[f"{prefix}_LAG_{lag}"] = grouped[source].shift(lag)

    return _attach(df, columns)


# ==========================================================
# Motion
# ==========================================================

def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    First differences, great-circle displacement, and heading.

    Longitude differences are wrapped, so no storm needs to be discarded for
    crossing the antimeridian.

    Where the previous step is unknown (the first row of a segment) every
    motion quantity is NaN, not zero. A zero there would assert that the storm
    was stationary and heading due north, and those fabricated rows would
    survive into training.
    """

    logger.info("Creating motion features...")

    lat = df["LAT"].to_numpy(dtype=float)
    lon = df["LON"].to_numpy(dtype=float)
    lat_prev = df["LAT_LAG_1"].to_numpy(dtype=float)
    lon_prev = df["LON_LAG_1"].to_numpy(dtype=float)

    known = np.isfinite(lat_prev) & np.isfinite(lon_prev)

    delta_lat = np.where(known, lat - lat_prev, np.nan)
    delta_lon = np.where(known, wrapped_degree_diff(lon, lon_prev), np.nan)

    distance = np.full(len(df), np.nan)
    bearing = np.full(len(df), np.nan)

    if known.any():
        distance[known] = haversine_distance(
            lat_prev[known], lon_prev[known], lat[known], lon[known],
        )
        bearing[known] = compute_bearing(
            lat_prev[known], lon_prev[known], lat[known], lon[known],
            undefined_as_nan=True,
        )

    bearing_radians = np.radians(bearing)

    return _attach(df, {
        "DELTA_LAT": delta_lat,
        "DELTA_LON": delta_lon,
        "DELTA_WIND": df["WMO_WIND"] - df["WIND_LAG_1"],
        "DELTA_PRESSURE": df["WMO_PRES"] - df["PRESSURE_LAG_1"],
        "MOVEMENT_DISTANCE": distance,
        "BEARING": bearing,
        # Circular encoding: a heading of 359 deg and one of 1 deg are nearly
        # the same direction but maximally far apart as raw numbers.
        "BEARING_SIN": np.sin(bearing_radians),
        "BEARING_COS": np.cos(bearing_radians),
    })


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translation speed and zonal/meridional components in physical units.

    A degree of longitude is 111 km at the equator and 55 km at 60 degrees, so
    DELTA_LON alone is not comparable across latitudes. u and v are.
    """

    logger.info("Creating velocity features...")

    interval = float(OBSERVATION_INTERVAL)

    latitude_radians = np.radians(df["LAT"].to_numpy(dtype=float))

    v_km_h = df["DELTA_LAT"].to_numpy(dtype=float) * KM_PER_DEGREE_LAT / interval
    u_km_h = (
        df["DELTA_LON"].to_numpy(dtype=float)
        * KM_PER_DEGREE_LAT
        * np.cos(latitude_radians)
        / interval
    )

    columns = {
        "VELOCITY_U": u_km_h,          # eastward, km/h
        "VELOCITY_V": v_km_h,          # northward, km/h
        "TRANSLATION_SPEED": df["MOVEMENT_DISTANCE"].to_numpy(dtype=float) / interval,
    }

    # IBTrACS reports its own storm direction; encode it circularly too.
    if "STORM_DIR" in df.columns:
        storm_dir_radians = np.radians(df["STORM_DIR"].to_numpy(dtype=float))
        columns["STORM_DIR_SIN"] = np.sin(storm_dir_radians)
        columns["STORM_DIR_COS"] = np.cos(storm_dir_radians)

    return _attach(df, columns)


def add_turning_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rate of change of heading.

    Straight-line tracks are easy; the error budget is dominated by
    recurvature. A storm about to recurve is usually already turning, so the
    signed change in bearing over recent steps is among the most informative
    features available -- and the original feature set had nothing like it.

    The difference is wrapped, so a turn from 350 deg to 10 deg reads as +20,
    not -340.
    """

    logger.info("Creating turning-rate features...")

    grouped = _group(df)
    bearing = df["BEARING"]

    turn_1 = angular_difference(bearing, grouped["BEARING"].shift(1))
    turn_2 = angular_difference(bearing, grouped["BEARING"].shift(2))

    columns = {
        "BEARING_CHANGE_1": turn_1,
        "BEARING_CHANGE_2": turn_2,
        # Magnitude only: how sharply the track is bending, regardless of side.
        "TURN_MAGNITUDE": np.abs(turn_1),
    }

    return _attach(df, columns)


def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Smoothed recent motion.

    A single 6-hour difference is noisy relative to best-track positional
    precision. Averaging the previous 24 hours gives a steadier estimate of the
    steering flow the storm is embedded in.
    """

    logger.info(f"Creating rolling features (windows={windows})...")

    grouped = _group(df)
    columns = {}

    for window in windows:
        hours = window * OBSERVATION_INTERVAL

        for source in ("DELTA_LAT", "DELTA_LON", "DELTA_WIND"):
            columns[f"{source}_MEAN_{hours}H"] = grouped[source].transform(
                lambda s, w=window: s.rolling(w, min_periods=w).mean()
            )

        columns[f"SPEED_MEAN_{hours}H"] = grouped["TRANSLATION_SPEED"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w).mean()
        )

    return _attach(df, columns)


def add_acceleration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second differences of position, and intensity trend.
    """

    logger.info("Creating acceleration features...")

    grouped = _group(df)

    return _attach(df, {
        "ACCEL_LAT": df["DELTA_LAT"] - grouped["DELTA_LAT"].shift(1),
        "ACCEL_LON": df["DELTA_LON"] - grouped["DELTA_LON"].shift(1),
        # Rapid intensification and weakening both correlate with track changes.
        "WIND_TREND": df["DELTA_WIND"] - grouped["DELTA_WIND"].shift(1),
    })


# ==========================================================
# Context
# ==========================================================

def add_storm_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Storm age, intensity class, basin, and latitude terms.

    Storm age is derived from timestamps rather than `cumcount()`. A row index
    undercounts age wherever cleaning removed rows, and it is measured per
    segment rather than per storm.
    """

    logger.info("Creating storm context features...")

    genesis = df.groupby("SID")["ISO_TIME"].transform("min")
    age_hours = (df["ISO_TIME"] - genesis).dt.total_seconds() / 3600.0

    intensity = pd.cut(
        df["WMO_WIND"],
        bins=INTENSITY_BINS,
        labels=INTENSITY_LABELS,
        right=False,
    ).astype(float)

    latitude = df["LAT"].to_numpy(dtype=float)

    columns = {
        "STORM_AGE": age_hours,
        "INTENSITY_CATEGORY": intensity,
        # Distance from the equator drives the poleward beta drift and the
        # likelihood of recurvature; the sign of LAT alone does not express it.
        "ABS_LAT": np.abs(latitude),
        "CORIOLIS_SIN_LAT": np.sin(np.radians(latitude)),
    }

    if "BASIN" in df.columns:
        # Basin was carried through the whole pipeline and never used. Motion
        # climatology differs sharply between basins.
        columns["BASIN_CODE"] = (
            df["BASIN"].astype(str).str.strip().str.upper().map(BASIN_CODES)
        ).astype("float")

        unmapped = columns["BASIN_CODE"].isna().sum()

        if unmapped:
            logger.warning(f"  {unmapped:,} rows have an unrecognised BASIN value")

    if "DIST2LAND" in df.columns:
        # Approaching land (negative) behaves differently from receding.
        columns["DELTA_DIST2LAND"] = df["DIST2LAND"] - _group(df)["DIST2LAND"].shift(1)

    return _attach(df, columns)


# ==========================================================
# Targets
# ==========================================================

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Absolute and displacement targets at every configured horizon.

    Both are emitted: displacement is what the model learns, absolute is what
    verification and reconstruction need. Longitude displacement is wrapped, so
    targets are valid across the antimeridian.
    """

    logger.info("Creating multi-horizon targets...")

    grouped = _group(df)
    lat = df["LAT"]
    lon = df["LON"]

    columns = {}

    for horizon, steps in FORECAST_HORIZONS.items():
        future_lat = grouped["LAT"].shift(-steps)
        future_lon = grouped["LON"].shift(-steps)

        columns[f"TARGET_LAT_{horizon}H"] = future_lat
        columns[f"TARGET_LON_{horizon}H"] = future_lon
        columns[f"TARGET_DLAT_{horizon}H"] = future_lat - lat
        columns[f"TARGET_DLON_{horizon}H"] = pd.Series(
            wrapped_degree_diff(future_lon, lon), index=df.index
        ).where(future_lon.notna())

        logger.info(
            f"  {horizon:>3}h  shift={steps:>2} steps  "
            f"({columns[f'TARGET_LAT_{horizon}H'].notna().sum():,} non-null)"
        )

    return _attach(df, columns)


# ==========================================================
# Save & Summary
# ==========================================================

def save_dataset(df: pd.DataFrame, write_csv: bool = True) -> None:
    """
    Persist the feature dataset. Parquet preserves dtypes and loads far faster.
    """

    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(FEATURE_PARQUET_PATH, index=False)
        logger.info(f"Saved: {FEATURE_PARQUET_PATH}")
    except Exception as exc:
        logger.warning(f"Parquet write failed ({exc}); relying on CSV.")
        write_csv = True

    if write_csv:
        df.to_csv(FEATURE_DATA_PATH, index=False)
        logger.info(f"Saved: {FEATURE_DATA_PATH}")


def print_summary(df: pd.DataFrame) -> None:
    """
    Report the shape of the engineered dataset.
    """

    lag_columns = [c for c in df.columns if "_LAG_" in c]
    target_columns = [c for c in df.columns if c.startswith("TARGET_")]

    logger.info("=" * 68)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 68)
    logger.info(f"  Rows             : {len(df):,}")
    logger.info(f"  Columns          : {df.shape[1]}")
    logger.info(f"  Storms           : {df['SID'].nunique():,}")
    logger.info(f"  Segments         : {df[GROUP_KEY].nunique():,}")
    logger.info(f"  Lag columns      : {len(lag_columns)}")
    logger.info(f"  Target columns   : {len(target_columns)}")

    if "BASIN" in df.columns:
        basins = df["BASIN"].value_counts().to_dict()
        logger.info(f"  Basins           : {basins}")

    logger.info("=" * 68)


# ==========================================================
# Main
# ==========================================================

def main(write_csv: bool = True) -> pd.DataFrame:

    logger.info("=" * 68)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 68)

    with Timer("Feature Engineering Pipeline"):
        df = load_clean_dataset()

        # Order matters: lags feed motion, motion feeds velocity and turning,
        # velocity feeds the rolling means.
        df = add_temporal_features(df)
        df = add_lag_features(df)
        df = add_motion_features(df)
        df = add_velocity_features(df)
        df = add_turning_features(df)
        df = add_rolling_features(df)
        df = add_acceleration_features(df)
        df = add_storm_context_features(df)
        df = create_target_variables(df)

        print_summary(df)
        save_dataset(df, write_csv=write_csv)

    logger.info("Feature engineering complete.")

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the feature dataset.")
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Write Parquet only (faster).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(write_csv=not args.no_csv)