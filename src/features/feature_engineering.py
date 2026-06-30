"""
feature_engineering.py

Feature Engineering Module

Current Features:
1. Temporal Features

Future Features:
2. Lag Features
3. Motion Features
4. Trend Features
5. Target Variables
"""

import logging
import numpy as np
import pandas as pd

from src.utils.config import (
    CLEAN_DATA_PATH,
    FEATURE_DATA_PATH,
    FORECAST_HORIZONS,
)

# ==========================================================
# Configure Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Load Dataset
# ==========================================================

def load_clean_dataset() -> pd.DataFrame:
    """
    Load the cleaned IBTrACS dataset.
    """

    logger.info("Loading cleaned dataset...")

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=["ISO_TIME"]
    )

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df


# ==========================================================
# Temporal Feature Engineering
# ==========================================================

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal and cyclical features from ISO_TIME.
    """

    logger.info("Creating temporal features...")

    # ------------------------------------------------------
    # Basic Date Features
    # ------------------------------------------------------

    df["YEAR"] = df["ISO_TIME"].dt.year

    df["MONTH"] = df["ISO_TIME"].dt.month

    df["DAY"] = df["ISO_TIME"].dt.day

    df["HOUR"] = df["ISO_TIME"].dt.hour

    df["DAY_OF_YEAR"] = df["ISO_TIME"].dt.dayofyear

    df["DAY_OF_WEEK"] = df["ISO_TIME"].dt.dayofweek

    df["QUARTER"] = df["ISO_TIME"].dt.quarter

    # ------------------------------------------------------
    # Season
    # ------------------------------------------------------

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn"
    }

    df["SEASON_NAME"] = df["MONTH"].map(season_map)

    # ------------------------------------------------------
    # Cyclical Encoding
    # ------------------------------------------------------

    logger.info("Creating cyclical time features...")

    # Month

    df["MONTH_SIN"] = np.sin(
        2 * np.pi * df["MONTH"] / 12
    )

    df["MONTH_COS"] = np.cos(
        2 * np.pi * df["MONTH"] / 12
    )

    # Hour

    df["HOUR_SIN"] = np.sin(
        2 * np.pi * df["HOUR"] / 24
    )

    df["HOUR_COS"] = np.cos(
        2 * np.pi * df["HOUR"] / 24
    )

    # Day of Year

    df["DAY_OF_YEAR_SIN"] = np.sin(
        2 * np.pi * df["DAY_OF_YEAR"] / 365
    )

    df["DAY_OF_YEAR_COS"] = np.cos(
        2 * np.pi * df["DAY_OF_YEAR"] / 365
    )

    logger.info("Temporal features created successfully.")

    return df


# ==========================================================
# Lag Features
# ==========================================================

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag features for each cyclone.
    """

    logger.info("Creating lag features...")

    lag_columns = {
        "LAT": "LAT",
        "LON": "LON",
        "WMO_WIND": "WIND",
        "WMO_PRES": "PRESSURE"
    }

    for column, prefix in lag_columns.items():

        for lag in range(1, 4):

            df[f"{prefix}_LAG_{lag}"] = (
                df.groupby("SID")[column]
                .shift(lag)
            )

    logger.info("Lag features created successfully.")

    return df

# ==========================================================
# Motion Features
# ==========================================================

def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclone motion features.
    """

    logger.info("Creating motion features...")

    # ------------------------------------------------------
    # Position Change
    # ------------------------------------------------------

    df["DELTA_LAT"] = df["LAT"] - df["LAT_LAG_1"]

    df["DELTA_LON"] = df["LON"] - df["LON_LAG_1"]

    # ------------------------------------------------------
    # Intensity Change
    # ------------------------------------------------------

    df["DELTA_WIND"] = (
        df["WMO_WIND"] -
        df["WIND_LAG_1"]
    )

    df["DELTA_PRESSURE"] = (
        df["WMO_PRES"] -
        df["PRESSURE_LAG_1"]
    )

    # ------------------------------------------------------
    # Movement Speed
    # ------------------------------------------------------

    df["MOVEMENT_DISTANCE"] = np.sqrt(
        df["DELTA_LAT"] ** 2 +
        df["DELTA_LON"] ** 2
    )

    logger.info("Motion features created successfully.")

    return df

# ==========================================================
# Target Generation
# ==========================================================

# ==========================================================
# Target Variable Generation
# ==========================================================

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables for multiple forecast horizons.

    For each forecast horizon, future latitude and longitude
    are created using cyclone-wise shifting.
    """

    logger.info("Creating multi-horizon target variables...")

    for horizon, shift_steps in FORECAST_HORIZONS.items():

        logger.info(f"Creating {horizon}-hour targets...")

        df[f"TARGET_LAT_{horizon}H"] = (
            df.groupby("SID")["LAT"]
            .shift(-shift_steps)
        )

        df[f"TARGET_LON_{horizon}H"] = (
            df.groupby("SID")["LON"]
            .shift(-shift_steps)
        )

    logger.info("Target variables created successfully.")

    return df

# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(df: pd.DataFrame) -> None:
    """
    Save feature engineered dataset.
    """

    logger.info("Saving feature engineered dataset...")

    FEATURE_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FEATURE_DATA_PATH,
        index=False
    )

    logger.info(f"Dataset saved to:\n{FEATURE_DATA_PATH}")


# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    df = load_clean_dataset()

    df = add_temporal_features(df)

    df = add_lag_features(df)

    df = add_motion_features(df)

    df = create_target_variables(df)

    save_dataset(df)

    logger.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()