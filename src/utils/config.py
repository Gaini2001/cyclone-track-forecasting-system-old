"""
config.py

Central configuration for the Cyclone Track Forecasting project.

Principles
----------
* This module is inert. Importing it has no side effects -- it does not touch
  the filesystem. Call `ensure_directories()` explicitly from a CLI entry point
  if you want the tree pre-created; every writer in the codebase already calls
  `mkdir(parents=True, exist_ok=True)` on its own output path.
* Derived values are computed, never duplicated. Row shifts come from
  OBSERVATION_INTERVAL; model paths come from a single template.
* Anything that changes the meaning of a saved artifact is part of that
  artifact's filename (see `split_paths`), so two configurations cannot
  silently overwrite each other's cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(
    os.environ.get("CYCLONE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
)

DATA_DIR = Path(os.environ.get("CYCLONE_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORT_DIR / "figures"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

ALL_DIRECTORIES = (
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURES_DIR,
)


def ensure_directories() -> None:
    """
    Create the project directory tree.

    Call this from a CLI entry point -- NOT at import time. Import-time
    filesystem writes break tests, read-only containers, and any tooling that
    merely wants to inspect configuration.
    """

    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Dataset Paths
# ==========================================================

RAW_DATA_PATH = RAW_DATA_DIR / "ibtracs.csv"
RAW_CACHE_PATH = RAW_DATA_DIR / "ibtracs_raw.parquet"

CLEAN_DATA_PATH = INTERIM_DATA_DIR / "ibtracs_clean.csv"
CLEAN_PARQUET_PATH = INTERIM_DATA_DIR / "ibtracs_clean.parquet"

FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "ibtracs_features.csv"
FEATURE_PARQUET_PATH = PROCESSED_DATA_DIR / "ibtracs_features.parquet"

PREDICTION_PATH = PREDICTIONS_DIR / "predictions.csv"

EVALUATION_RESULTS_PATH = REPORT_DIR / "evaluation_results.json"
MODEL_COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
TUNED_PARAMS_PATH = REPORT_DIR / "tuned_params.json"
OPTUNA_STUDY_PATH = REPORT_DIR / "optuna_study.db"

# ==========================================================
# Logging & Reproducibility
# ==========================================================

LOG_LEVEL = os.environ.get("CYCLONE_LOG_LEVEL", "INFO")
RANDOM_STATE = 42

# ==========================================================
# Ingestion / Cleaning Configuration
# ==========================================================

START_YEAR = 1980
OBSERVATION_INTERVAL = 6  # hours between consecutive synoptic observations

# Synoptic hours derived from the interval so the two cannot drift apart.
SYNOPTIC_HOURS = tuple(range(0, 24, OBSERVATION_INTERVAL))

# IBTrACS track_type: "spur" rows are secondary fragments of the same system.
VALID_TRACK_TYPES = ("main",)

# IBTrACS nature: DS=disturbance, TS=tropical, ET=extratropical,
# SS=subtropical, NR=not reported, MX=mixture.
VALID_NATURES = ("TS", "SS")

# Position is never imputed. Only these may be forward-filled.
IMPUTABLE_COLUMNS = ("WMO_WIND", "WMO_PRES", "STORM_SPEED", "STORM_DIR", "DIST2LAND")
MAX_FFILL_STEPS = 2

# Deepest lag (3) + longest horizon in steps (12) + 1 = 16, plus headroom.
MIN_SEGMENT_OBSERVATIONS = 20

# ==========================================================
# Forecast Horizons
# ==========================================================
# Derived from OBSERVATION_INTERVAL rather than hardcoded, so a change to the
# observation cadence cannot silently invalidate the target construction.

FORECAST_HORIZON_HOURS = (6, 12, 24, 48, 72)

FORECAST_HORIZONS = {
    hours: hours // OBSERVATION_INTERVAL
    for hours in FORECAST_HORIZON_HOURS
}

DEFAULT_FORECAST_HORIZON = 24

# ==========================================================
# Target Configuration
# ==========================================================
# "displacement" predicts (delta_lat, delta_lon) from the current position and
# reconstructs the absolute position at evaluation time. "absolute" predicts
# the future coordinate directly.
#
# Prefer displacement. With absolute targets, LAT is both an input and almost
# the answer, so R^2 saturates near 1.0 and reports nothing about forecast
# skill. Displacement forces the model to learn the part that is actually hard.

TARGET_MODE = "displacement"  # "displacement" | "absolute"

TARGET_COLUMNS = {
    hours: [f"TARGET_LAT_{hours}H", f"TARGET_LON_{hours}H"]
    for hours in FORECAST_HORIZON_HOURS
}

DELTA_TARGET_COLUMNS = {
    hours: [f"TARGET_DLAT_{hours}H", f"TARGET_DLON_{hours}H"]
    for hours in FORECAST_HORIZON_HOURS
}


def target_columns(horizon: int, mode: str = None) -> list[str]:
    """
    Return the target column names for a horizon under the active target mode.
    """

    mode = mode or TARGET_MODE

    if mode == "displacement":
        return DELTA_TARGET_COLUMNS[horizon]

    if mode == "absolute":
        return TARGET_COLUMNS[horizon]

    raise ValueError(f"Unknown TARGET_MODE: {mode!r}")


# ==========================================================
# Schema
# ==========================================================

REQUIRED_COLUMNS = [
    "SID",
    "SEASON",
    "BASIN",
    "SUBBASIN",
    "ISO_TIME",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "STORM_SPEED",
    "STORM_DIR",
    "NATURE",
    "DIST2LAND",
    "TRACK_TYPE",
]

NUMERIC_COLUMNS = [
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "STORM_SPEED",
    "STORM_DIR",
    "DIST2LAND",
]

DATETIME_COLUMNS = ["ISO_TIME"]

# ==========================================================
# CSV Missing-Value Handling
# ==========================================================
# pandas treats the literal string "NA" as missing by default. In IBTrACS,
# "NA" is the basin code for the North Atlantic, so the default behaviour
# silently erases the basin label from every Atlantic storm. These are the
# pandas defaults with "NA" and "N/A" removed.

CSV_NA_VALUES = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "NULL", "NaN", "None", "n/a", "nan", "null",
]

# Identifier columns carried through the pipeline but never used as features.
# SEGMENT_ID marks runs of contiguous 6-hourly observations: group by it for
# anything time-relative, and by SID for train/test splitting.
ID_COLUMNS = ["SID", "SEGMENT_ID", "ISO_TIME", "SEASON", "BASIN", "SUBBASIN"]

# ==========================================================
# Feature Groups
# ==========================================================
# Grouped rather than flat so you can run ablations: drop a group, retrain,
# report the delta in track error. That table is worth more in an interview
# than another model.

FEATURE_GROUPS = {
    "position": [
        "LAT",
        "LON",
    ],
    "intensity": [
        "WMO_WIND",
        "WMO_PRES",
        "STORM_SPEED",
        "DIST2LAND",
    ],
    "temporal": [
        "MONTH",
        "DAY",
        "HOUR",
        "DAY_OF_YEAR",
    ],
    "cyclic": [
        "MONTH_SIN",
        "MONTH_COS",
        "HOUR_SIN",
        "HOUR_COS",
        "DOY_SIN",
        "DOY_COS",
    ],
    "lag_position": [
        "LAT_LAG_1", "LAT_LAG_2", "LAT_LAG_3",
        "LON_LAG_1", "LON_LAG_2", "LON_LAG_3",
    ],
    "lag_intensity": [
        "WIND_LAG_1", "WIND_LAG_2", "WIND_LAG_3",
        "PRESSURE_LAG_1", "PRESSURE_LAG_2", "PRESSURE_LAG_3",
    ],
    "motion": [
        "DELTA_LAT",
        "DELTA_LON",
        "DELTA_WIND",
        "DELTA_PRESSURE",
        "MOVEMENT_DISTANCE",
        # Raw BEARING and STORM_DIR are deliberately excluded: they are
        # compass angles with a discontinuity at north. The sin/cos pair
        # carries the same information without the seam.
        "BEARING_SIN",
        "BEARING_COS",
    ],
    "velocity": [
        "VELOCITY_U",
        "VELOCITY_V",
        "TRANSLATION_SPEED",
        "STORM_DIR_SIN",
        "STORM_DIR_COS",
    ],
    "turning": [
        "BEARING_CHANGE_1",
        "BEARING_CHANGE_2",
        "TURN_MAGNITUDE",
    ],
    "rolling": [
        "DELTA_LAT_MEAN_24H",
        "DELTA_LON_MEAN_24H",
        "DELTA_WIND_MEAN_24H",
        "SPEED_MEAN_24H",
    ],
    "acceleration": [
        "ACCEL_LAT",
        "ACCEL_LON",
        "WIND_TREND",
    ],
    "context": [
        "STORM_AGE",
        "INTENSITY_CATEGORY",
        "ABS_LAT",
        "CORIOLIS_SIN_LAT",
        "BASIN_CODE",
        "DELTA_DIST2LAND",
    ],
}

FEATURE_COLUMNS = [col for group in FEATURE_GROUPS.values() for col in group]


def features_excluding(*groups: str) -> list[str]:
    """
    Feature list with the named groups removed. For ablation studies.

    Example
    -------
    >>> features_excluding("acceleration", "cyclic")
    """

    unknown = [g for g in groups if g not in FEATURE_GROUPS]

    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")

    return [
        col
        for name, group in FEATURE_GROUPS.items()
        if name not in groups
        for col in group
    ]


# ==========================================================
# Train / Validation / Test Split
# ==========================================================
# "season" holds out later seasons -- the only split that reflects how a
# forecast system is actually deployed, and the one to report in the README.
# "storm" randomly partitions storm IDs; keep it available for comparison, but
# expect it to look optimistic because same-season storms share synoptic state.

SPLIT_STRATEGY = "season"  # "season" | "storm"

# Season split boundaries (inclusive).
TRAIN_END_SEASON = 2014
VAL_END_SEASON = 2018
# Test = every season after VAL_END_SEASON.

# Storm split ratios, used only when SPLIT_STRATEGY == "storm".
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


def split_config() -> dict:
    """
    The full set of parameters that determine a split.
    """

    if SPLIT_STRATEGY == "season":
        return {
            "strategy": "season",
            "train_end": TRAIN_END_SEASON,
            "val_end": VAL_END_SEASON,
            "start_year": START_YEAR,
        }

    return {
        "strategy": "storm",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "seed": RANDOM_STATE,
    }


def split_fingerprint(config: dict = None) -> str:
    """
    Short deterministic hash of a split configuration.

    This is what stops a 70/10/20 tuning run from silently overwriting an
    80/20 training split. The parameters live in the filename, so a mismatched
    cache simply is not found rather than being loaded as if it were correct.
    """

    config = config if config is not None else split_config()
    payload = json.dumps(config, sort_keys=True).encode()

    return hashlib.md5(payload).hexdigest()[:8]


def split_paths(config: dict = None) -> dict[str, Path]:
    """
    Paths to the persisted storm-ID lists for a given split configuration.
    """

    tag = split_fingerprint(config)

    return {
        "train": PROCESSED_DATA_DIR / f"split_{tag}_train.csv",
        "val": PROCESSED_DATA_DIR / f"split_{tag}_val.csv",
        "test": PROCESSED_DATA_DIR / f"split_{tag}_test.csv",
        "meta": PROCESSED_DATA_DIR / f"split_{tag}_meta.json",
    }


# Backwards-compatible aliases for the default configuration.
TRAIN_STORMS_PATH = split_paths()["train"]
VAL_STORMS_PATH = split_paths()["val"]
TEST_STORMS_PATH = split_paths()["test"]

# ==========================================================
# Model Artifacts
# ==========================================================

MODEL_FILENAME_TEMPLATE = "{model}_{horizon}h_{target_mode}.pkl"


def model_path(model: str, horizon: int = None, target_mode: str = None) -> Path:
    """
    Canonical path for a trained model artifact.

    The target mode is part of the filename because a displacement model and an
    absolute-position model are not interchangeable, and loading the wrong one
    fails in a way that is very hard to notice.

    Parameters
    ----------
    model : str
        Short model key, e.g. "random_forest" or "xgboost".
    horizon : int
        Forecast horizon in hours.
    """

    horizon = horizon or DEFAULT_FORECAST_HORIZON
    target_mode = target_mode or TARGET_MODE

    return MODEL_DIR / MODEL_FILENAME_TEMPLATE.format(
        model=model, horizon=horizon, target_mode=target_mode,
    )


# Backwards-compatible aliases.
RANDOM_FOREST_MODEL_PATH = model_path("random_forest")
XGB_MODEL_PATH = model_path("xgboost")

# ==========================================================
# Hyperparameters
# ==========================================================
# Defaults below are starting points, not tuned values. Tuned parameters are
# written to TUNED_PARAMS_PATH by tune_hyperparameters.py and loaded here, so
# the provenance of every number is recorded instead of hand-copied.

RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 25,
    "min_samples_split": 9,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.01,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

DEFAULT_PARAMS = {
    "random_forest": RF_PARAMS,
    "xgboost": XGB_PARAMS,
}


def get_model_params(model: str, horizon: int = None) -> dict:
    """
    Hyperparameters for a model at a horizon.

    Resolution order: tuned params for this exact (model, horizon) ->
    tuned params for the model at any horizon -> hardcoded defaults.

    Longer horizons generally want different capacity than short ones, so
    per-horizon tuning is worth recording separately.
    """

    horizon = horizon or DEFAULT_FORECAST_HORIZON
    params = dict(DEFAULT_PARAMS[model])

    if TUNED_PARAMS_PATH.exists():
        try:
            tuned = json.loads(TUNED_PARAMS_PATH.read_text())
            entry = tuned.get(model, {}).get(str(horizon))

            if entry:
                params.update(entry.get("params", {}))
        except (json.JSONDecodeError, OSError):
            pass  # fall back to defaults rather than fail a training run

    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("n_jobs", -1)

    return params


# ==========================================================
# Saffir-Simpson Wind Scale (knots, 1-minute sustained)
# ==========================================================
# Seven edges -> seven categories. Note the indexing: category 0 is a tropical
# depression and category 1 is a tropical storm, so Cat 1 hurricane is label 2.
# The previous label map was off by one, which mislabeled the intensity plots.
#
# The upper edge is infinite. A finite cap (the old value was 300) sends any
# out-of-range wind to NaN, which then gets filled as category 0 -- labelling
# the strongest storms as the weakest.

INTENSITY_BINS = (0, 34, 64, 83, 96, 113, 137, float("inf"))
INTENSITY_LABELS = (0, 1, 2, 3, 4, 5, 6)

INTENSITY_LABEL_NAMES = {
    0: "TD",       # < 34 kt   tropical depression
    1: "TS",       # 34-63 kt  tropical storm
    2: "Cat 1",    # 64-82 kt
    3: "Cat 2",    # 83-95 kt
    4: "Cat 3",    # 96-112 kt
    5: "Cat 4",    # 113-136 kt
    6: "Cat 5",    # >= 137 kt
}

# ==========================================================
# Physical Constants & Validation Bounds
# ==========================================================

EARTH_RADIUS_KM = 6371.0

LATITUDE_RANGE = (-90.0, 90.0)
LONGITUDE_RANGE = (-180.0, 180.0)

# Tropical cyclones essentially never occur outside this band. A tighter bound
# than the geometric one catches corrupt records that -90/90 would pass.
TROPICAL_LATITUDE_RANGE = (-70.0, 70.0)

MIN_WIND_SPEED = 0.0
MAX_WIND_SPEED = 250.0   # kt; the record is ~190 kt
MIN_PRESSURE = 800.0     # mb; the record low is ~870 mb
MAX_PRESSURE = 1050.0

# ==========================================================
# Self-Check
# ==========================================================
# Cheap invariants. A misconfigured constant should fail loudly at import
# rather than produce quietly wrong metrics.

assert len(INTENSITY_BINS) - 1 == len(INTENSITY_LABELS), (
    "INTENSITY_BINS must have exactly one more edge than there are labels."
)

assert set(INTENSITY_LABELS) == set(INTENSITY_LABEL_NAMES), (
    "Every intensity label needs a display name."
)

assert all(
    hours % OBSERVATION_INTERVAL == 0 for hours in FORECAST_HORIZON_HOURS
), "Every forecast horizon must be a whole multiple of OBSERVATION_INTERVAL."

assert DEFAULT_FORECAST_HORIZON in FORECAST_HORIZONS, (
    "DEFAULT_FORECAST_HORIZON must be one of the configured horizons."
)

assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)), (
    "Duplicate entries in FEATURE_GROUPS."
)

assert TARGET_MODE in ("displacement", "absolute")
assert SPLIT_STRATEGY in ("season", "storm")