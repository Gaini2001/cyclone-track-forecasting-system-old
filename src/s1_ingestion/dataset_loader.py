"""
dataset_loader.py

Loads the raw IBTrACS dataset and performs basic validation.

Author: Om Prakash Gaini
Project: End-to-End Tropical Cyclone Track Forecasting System
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import RAW_DATA_PATH, REQUIRED_COLUMNS, CSV_NA_VALUES

# --------------------------------------------------
# Projects Paths
# --------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ibtracs.csv"

# --------------------------------------------------
# Configure Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Required Columns
# --------------------------------------------------




# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the IBTrACS dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    file_path = Path(filepath)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Loading dataset...")

    # pandas treats the literal string "NA" as missing by default. In IBTrACS,
    # "NA" is the basin code for the North Atlantic, so the default behaviour
    # silently erases the basin label from every Atlantic storm. This guard
    # only stops the corruption from happening again on this read -- see
    # below for recovering the labels this file already lost.
    df = pd.read_csv(
        file_path,
        low_memory=False,
        keep_default_na=False,
        na_values=CSV_NA_VALUES,
    )

    # data/raw/ibtracs.csv was itself produced by an earlier, unguarded
    # `pd.read_csv` of the official NOAA source, so the "NA" -> null bug is
    # already baked into this file: BASIN is blank, not "NA", for every North
    # Atlantic row. Recovered by cross-referencing NOAA's public v04r00 file
    # on (SID, ISO_TIME): 39,441 of 40,784 blank rows matched "NA" exactly and
    # zero matched anything else. The remaining 1,343 belong to 27 storms
    # whose SEASON predates the reference file's 1980 cutoff -- same tracks,
    # continuous with their matched rows, and geographically confined to the
    # Atlantic (7-71N, 108W-14E), so by elimination they are "NA" too: IBTrACS
    # defines exactly seven basins and the other six are already labeled here.
    if "BASIN" in df.columns:
        blank_basin = df["BASIN"].isna()
        recovered = int(blank_basin.sum())

        if recovered:
            df.loc[blank_basin, "BASIN"] = "NA"
            logger.info(
                f"Recovered {recovered:,} North Atlantic BASIN labels "
                "(blank -> 'NA')"
            )

    logger.info("Dataset loaded successfully.")

    return df


# --------------------------------------------------
# Validate Required Columns
# --------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    """
    Check whether all required columns exist.
    """

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info("All required columns are present.")


# --------------------------------------------------
# Display Dataset Information
# --------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> None:
    """
    Display basic dataset information.
    """

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print("\n" + "=" * 60)
    print("IBTrACS DATASET SUMMARY")
    print("=" * 60)

    print(f"Shape            : {df.shape}")
    print(f"Features         : {df.shape[1]}")
    print(f"Records          : {len(df):,}")

    if "SID" in df.columns:
        print(f"Unique Storms    : {df['SID'].nunique():,}")

    if "SEASON" in df.columns:
        print(
            f"Season Range     : "
            f"{df['SEASON'].min()} - {df['SEASON'].max()}"
        )

    if "BASIN" in df.columns:
        print(f"Ocean Basins     : {df['BASIN'].nunique()}")

    print(f"Memory Usage     : {memory_usage:.2f} MB")

    print("=" * 60)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # dataset_path = "data/raw/ibtracs.csv"

    df = load_dataset(RAW_DATA_PATH)

    validate_columns(df)

    dataset_summary(df)


if __name__ == "__main__":
    main()