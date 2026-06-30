"""
clean_data.py

Performs data cleaning on the raw IBTrACS dataset.

Pipeline:
1. Load raw dataset
2. Filter seasons
3. Select required columns
4. Convert data types
5. Sort storms chronologically
6. Handle missing values
7. Save cleaned dataset
"""

import logging

import pandas as pd

from src.ingestion.dataset_loader import load_dataset
from src.utils.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    START_YEAR,
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    DATETIME_COLUMNS,
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
# Filter Dataset
# ==========================================================

def filter_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep cyclone records from START_YEAR onwards.
    """

    logger.info(f"Filtering records from {START_YEAR} onwards...")

    df = df[df["SEASON"] >= START_YEAR].copy()

    logger.info(f"Remaining records: {len(df):,}")

    return df


# ==========================================================
# Select Required Columns
# ==========================================================

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the required columns.
    """

    logger.info("Selecting required columns...")

    return df[REQUIRED_COLUMNS].copy()


# ==========================================================
# Convert Data Types
# ==========================================================

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    """

    logger.info("Converting data types...")

    # ----------------------------
    # Datetime Columns
    # ----------------------------

    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    # ----------------------------
    # Numeric Columns
    # ----------------------------

    for col in NUMERIC_COLUMNS:

        missing_before = df[col].isna().sum()

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        missing_after = df[col].isna().sum()

        if missing_after > missing_before:

            logger.info(
                f"{col}: "
                f"{missing_after - missing_before:,} values "
                f"converted to NaN during numeric conversion."
            )

    return df

# ==========================================================
# Normalize Longitude
# ==========================================================

def normalize_longitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert longitude values from 0–360° to -180–180°.
    """

    logger.info("Normalizing longitude values...")

    df["LON"] = df["LON"].apply(
        lambda x: x - 360 if pd.notna(x) and x > 180 else x
    )

    logger.info("Longitude normalization completed.")

    return df


# ==========================================================
# Sort Storms
# ==========================================================

def sort_storms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort storm observations chronologically.
    """

    logger.info("Sorting storms chronologically...")

    df = df.sort_values(
        by=["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    return df


# ==========================================================
# Remove Duplicate Observations
# ==========================================================

def remove_duplicate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate observations based on storm ID and timestamp.
    """

    logger.info("Removing duplicate observations...")

    rows_before = len(df)

    df = df.drop_duplicates(
        subset=["SID", "ISO_TIME"]
    ).reset_index(drop=True)

    rows_after = len(df)

    logger.info(
        f"Removed {rows_before - rows_after:,} duplicate rows."
    )

    return df



# ==========================================================
# Handle Missing Values
# ==========================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the IBTrACS dataset.

    Strategy:
    1. Forward-fill and backward-fill numeric values within each cyclone.
    2. Remove rows that still have missing WMO_WIND or WMO_PRES.
    """

    logger.info("Handling missing values...")

    # --------------------------------------------------
    # Missing values before cleaning
    # --------------------------------------------------

    wind_before = df["WMO_WIND"].isna().sum()
    pressure_before = df["WMO_PRES"].isna().sum()

    rows_before = len(df)

    logger.info(f"WMO_WIND missing before : {wind_before:,}")
    logger.info(f"WMO_PRES missing before : {pressure_before:,}")

    # --------------------------------------------------
    # Fill numeric values within each cyclone
    # --------------------------------------------------

    logger.info("Forward/Backward filling numeric columns within each storm...")

    print("\nNumeric Columns Used for Filling:")
    print(NUMERIC_COLUMNS)

    df[NUMERIC_COLUMNS] = (
        df.groupby("SID")[NUMERIC_COLUMNS]
        .transform(lambda x: x.ffill().bfill())
    )

    # --------------------------------------------------
    # Remove rows still missing wind or pressure
    # --------------------------------------------------

    df = (
        df.dropna(
            subset=["WMO_WIND", "WMO_PRES"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Statistics after cleaning
    # --------------------------------------------------

    rows_after = len(df)

    wind_after = df["WMO_WIND"].isna().sum()
    pressure_after = df["WMO_PRES"].isna().sum()

    logger.info(f"Remaining missing WMO_WIND after cleaning : {wind_after:,}")
    logger.info(f"Remaining missing WMO_PRES after cleaning  : {pressure_after:,}")

    logger.info(f"Rows removed            : {rows_before - rows_after:,}")
    logger.info(f"Remaining rows          : {rows_after:,}")

    return df


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(df: pd.DataFrame) -> None:
    """
    Save cleaned dataset.
    """

    logger.info("Saving cleaned dataset...")

    CLEAN_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CLEAN_DATA_PATH,
        index=False
    )

    logger.info(f"Dataset saved to:\n{CLEAN_DATA_PATH}")


# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    df = load_dataset(RAW_DATA_PATH)
    
    # Diagnostic
    print("\nBefore Cleaning")
    print("WMO_WIND missing :", df["WMO_WIND"].isna().sum())
    print("WMO_PRES missing :", df["WMO_PRES"].isna().sum())

    df = filter_by_year(df)

    df = select_columns(df)

    df = convert_data_types(df)

    df = normalize_longitude(df)

    df = sort_storms(df)

    df = remove_duplicate_observations(df)

    df = handle_missing_values(df)

    save_dataset(df)

    logger.info("Data cleaning completed successfully.")


if __name__ == "__main__":
    main()