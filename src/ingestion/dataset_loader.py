"""
dataset_loader.py

Loads the raw IBTrACS dataset and performs basic validation.

Author: Om Prakash
Project: End-to-End Tropical Cyclone Track Forecasting System
"""

from pathlib import Path
import logging
import pandas as pd

# --------------------------------------------------
# Projects Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ibtracs.csv"

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

    df = pd.read_csv(file_path, low_memory=False)

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

    df = load_dataset(DATA_PATH)

    validate_columns(df)

    dataset_summary(df)


if __name__ == "__main__":
    main()