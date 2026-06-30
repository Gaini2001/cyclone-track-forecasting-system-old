"""
config.py

Central configuration file for the Cyclone Track Forecasting project.

All project paths, filenames, constants, and hyperparameters should
be defined here to avoid hardcoding values throughout the project.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# ==========================================================
# Dataset Filenames
# ==========================================================

RAW_DATA_FILENAME = "ibtracs.csv"

CLEAN_DATA_FILENAME = "ibtracs_clean.csv"

FEATURE_DATA_FILENAME = "ibtracs_features.csv"

PREDICTION_FILENAME = "predictions.csv"

# ==========================================================
# Dataset Paths
# ==========================================================

RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

CLEAN_DATA_PATH = INTERIM_DATA_DIR / CLEAN_DATA_FILENAME

FEATURE_DATA_PATH = PROCESSED_DATA_DIR / FEATURE_DATA_FILENAME

PREDICTION_PATH = PREDICTIONS_DIR / PREDICTION_FILENAME

# ==========================================================
# Model Filenames
# ==========================================================

RF_MODEL_FILENAME = "random_forest.pkl"

XGB_MODEL_FILENAME = "xgboost.pkl"

LSTM_MODEL_FILENAME = "lstm.keras"

SCALER_FILENAME = "scaler.pkl"

# ==========================================================
# Model Paths
# ==========================================================

RF_MODEL_PATH = MODEL_DIR / RF_MODEL_FILENAME

XGB_MODEL_PATH = MODEL_DIR / XGB_MODEL_FILENAME

LSTM_MODEL_PATH = MODEL_DIR / LSTM_MODEL_FILENAME

SCALER_PATH = MODEL_DIR / SCALER_FILENAME

# ==========================================================
# Dataset Configuration
# ==========================================================

START_YEAR = 1980

END_YEAR = None          # Use latest available year

OBSERVATION_INTERVAL = 6  # hours

# ==========================================================
# Machine Learning Configuration
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

# ==========================================================
# Forecast Horizons (hours)
# ==========================================================

# Forecast horizon (hours : number of observations ahead)

FORECAST_HORIZONS = {
    6: 1,
    12: 2,
    24: 4,
    48: 8,
    72: 12
}
# ==========================================================
# Sequence Model Configuration
# ==========================================================

PAST_STEPS = 12

FUTURE_STEPS = 4

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = "INFO"

# ==========================================================
# Required Data Columns
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
    "TRACK_TYPE"
]


# ==========================================================
# Numeric Columns
# ==========================================================

NUMERIC_COLUMNS = [
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "STORM_SPEED",
    "STORM_DIR",
    "DIST2LAND",
]

# ==========================================================
#Datetime Columns
# ==========================================================

DATETIME_COLUMNS = [
    "ISO_TIME"
]

#Engineered Dataset

FEATURE_DATA_FILENAME = "ibtracs_features.csv"

FEATURE_DATA_PATH = PROCESSED_DATA_DIR / FEATURE_DATA_FILENAME

# ==========================================================
# Input Features
# ==========================================================

FEATURE_COLUMNS = [

    # Position
    "LAT",
    "LON",

    # Storm Characteristics
    "WMO_WIND",
    "WMO_PRES",

    # Temporal
    "MONTH",
    "DAY",
    "HOUR",
    "DAY_OF_YEAR",

    # Cyclic
    "MONTH_SIN",
    "MONTH_COS",
    "HOUR_SIN",
    "HOUR_COS",

    # Lag
    "LAT_LAG_1",
    "LAT_LAG_2",
    "LAT_LAG_3",

    "LON_LAG_1",
    "LON_LAG_2",
    "LON_LAG_3",

    "WIND_LAG_1",
    "WIND_LAG_2",
    "WIND_LAG_3",

    "PRESSURE_LAG_1",
    "PRESSURE_LAG_2",
    "PRESSURE_LAG_3",

    # Motion
    "DELTA_LAT",
    "DELTA_LON",
    "DELTA_WIND",
    "DELTA_PRESSURE",
    "MOVEMENT_DISTANCE"
]

# target columns

TARGET_COLUMNS = {
    6: ["TARGET_LAT_6H", "TARGET_LON_6H"],
    12: ["TARGET_LAT_12H", "TARGET_LON_12H"],
    24: ["TARGET_LAT_24H", "TARGET_LON_24H"],
    48: ["TARGET_LAT_48H", "TARGET_LON_48H"],
    72: ["TARGET_LAT_72H", "TARGET_LON_72H"],
}

# ==========================================================
# Geographic Limits
# ==========================================================

LATITUDE_RANGE = (-90, 90)

LONGITUDE_RANGE = (-180, 180)

MIN_WIND_SPEED = 0

MIN_PRESSURE = 0

# ==========================================================
# Train/Test Storm IDs
# ==========================================================

TRAIN_STORMS_PATH = PROCESSED_DATA_DIR / "train_storms.csv"

TEST_STORMS_PATH = PROCESSED_DATA_DIR / "test_storms.csv"

# ==========================================================
# Saved Models
# ==========================================================

RANDOM_FOREST_MODEL_PATH = (
    MODEL_DIR / "random_forest_6h.pkl"
)

