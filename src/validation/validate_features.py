"""
validate_features.py

Validate the feature engineered dataset before model training.

Checks:
1. Feature columns exist
2. Target columns exist
3. Missing values
4. Infinite values
5. Latitude range
6. Longitude range
7. Wind speed
8. Pressure
"""

import logging
import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURE_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    MIN_WIND_SPEED,
    MIN_PRESSURE,
    NUMERIC_COLUMNS,
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

def load_feature_dataset() -> pd.DataFrame:
    """
    Load the feature engineered dataset.
    """

    logger.info("Loading feature dataset...")

    df = pd.read_csv(FEATURE_DATA_PATH)

    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

    return df


# ==========================================================
# Validate Feature Columns
# ==========================================================

def validate_feature_columns(df: pd.DataFrame) -> None:
    """
    Ensure all feature columns exist.
    """

    logger.info("Validating feature columns...")

    missing = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns:\n{missing}"
        )

    logger.info("Feature columns validated.")


# ==========================================================
# Validate Target Columns
# ==========================================================

def validate_target_columns(df: pd.DataFrame) -> None:
    """
    Ensure all target columns exist.
    """

    logger.info("Validating target columns...")

    required_targets = []

    for cols in TARGET_COLUMNS.values():
        required_targets.extend(cols)

    missing = [
        col
        for col in required_targets
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing target columns:\n{missing}"
        )

    logger.info("Target columns validated.")



# ==========================================================
# Validate Missing Values
# ==========================================================

def validate_missing_values(df: pd.DataFrame) -> None:
    """
    Report missing values in the feature engineered dataset.
    """

    logger.info("Checking missing values...")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        logger.info("No missing values found.")
    else:
        logger.warning("Missing values detected:")
        print(missing)


# ==========================================================
# Validate Infinite Values
# ==========================================================

def validate_infinite_values(df: pd.DataFrame) -> None:
    """
    Check for infinite values.
    """

    logger.info("Checking infinite values...")

    numeric_df = df.select_dtypes(include=np.number)

    has_inf = np.isinf(numeric_df).sum()

    has_inf = has_inf[has_inf > 0]

    if not has_inf.empty:

        raise ValueError(
            f"Infinite values detected:\n{has_inf}"
        )

    logger.info("No infinite values found.")


# ==========================================================
# Validate Coordinates
# ==========================================================

def validate_coordinates(df: pd.DataFrame) -> None:
    """
    Validate latitude and longitude ranges.
    """

    logger.info("Validating coordinates...")

    invalid_lat = df[
        ~df["LAT"].between(*LATITUDE_RANGE)
    ]

    invalid_lon = df[
        ~df["LON"].between(*LONGITUDE_RANGE)
    ]

    if not invalid_lat.empty:

        print("\nInvalid Latitude Values:")
        print(invalid_lat[["SID", "ISO_TIME", "LAT"]].head())

        raise ValueError(
            f"{len(invalid_lat)} invalid latitude values found."
        )

    if not invalid_lon.empty:

        print("\nInvalid Longitude Values:")
        print(invalid_lon[["SID", "ISO_TIME", "LON"]].head())

        raise ValueError(
            f"{len(invalid_lon)} invalid longitude values found."
        )

    logger.info("Coordinate validation passed.")


# ==========================================================
# Validate Wind Speed
# ==========================================================

def validate_wind_speed(df: pd.DataFrame) -> None:
    """
    Validate wind speed values.
    """

    logger.info("Validating wind speed...")

    if (df["WMO_WIND"] < MIN_WIND_SPEED).any():

        raise ValueError(
            "Negative wind speed detected."
        )

    logger.info("Wind speed validation passed.")


# ==========================================================
# Validate Pressure
# ==========================================================

def validate_pressure(df: pd.DataFrame) -> None:
    """
    Validate pressure values.
    """

    logger.info("Validating pressure...")

    if (df["WMO_PRES"] < MIN_PRESSURE).any():

        raise ValueError(
            "Negative pressure detected."
        )

    logger.info("Pressure validation passed.")


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("FEATURE VALIDATION")
    logger.info("=" * 60)

    df = load_feature_dataset()

    validate_feature_columns(df)

    validate_target_columns(df)

    validate_missing_values(df)

    validate_infinite_values(df)

    validate_coordinates(df)

    validate_wind_speed(df)

    validate_pressure(df)

    logger.info("=" * 60)
    logger.info("All feature validations passed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()