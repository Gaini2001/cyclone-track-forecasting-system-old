"""
train.py

Prepare the feature engineered dataset for model training.
"""

import logging
import pandas as pd
import numpy as np

from src.utils.config import (
    FEATURE_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    RANDOM_STATE,
    TRAIN_STORMS_PATH,
    TEST_STORMS_PATH,
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

    df = pd.read_csv(
        FEATURE_DATA_PATH,
        parse_dates=["ISO_TIME"]
    )

    logger.info(f"Dataset Shape: {df.shape}")

    return df


# ==========================================================
# Prepare Dataset
# ==========================================================

def prepare_training_data(
    df: pd.DataFrame,
    forecast_horizon: int = 6,
):
    """
    Prepare features and targets for training.
    """

    logger.info(
        f"Preparing {forecast_horizon}-hour prediction dataset..."
    )

    target_lat, target_lon = TARGET_COLUMNS[
        forecast_horizon
    ]

    # Remove rows without targets

    df = df.dropna(
        subset=[
            target_lat,
            target_lon,
        ]
    )

    # Remove rows with missing features

    df = df.dropna(
        subset=FEATURE_COLUMNS
    )

    logger.info(
        f"Training samples: {len(df):,}"
    )

    return df



# ==========================================================
# Save Storm IDs
# ==========================================================

def save_storm_ids(
    train_storms,
    test_storms,
):
    """
    Save train and test Storm IDs.

    These files are reused in future runs so that
    every model uses the exact same train/test split.
    """

    logger.info("Saving train/test storm IDs...")

    pd.DataFrame(
        {"SID": train_storms}
    ).to_csv(
        TRAIN_STORMS_PATH,
        index=False,
    )

    pd.DataFrame(
        {"SID": test_storms}
    ).to_csv(
        TEST_STORMS_PATH,
        index=False,
    )

    logger.info(
        f"Training Storm IDs saved: {TRAIN_STORMS_PATH}"
    )

    logger.info(
        f"Testing Storm IDs saved : {TEST_STORMS_PATH}"
    )

# ==========================================================
# Load Storm IDs
# ==========================================================

# ==========================================================
# Load Storm IDs
# ==========================================================

def load_storm_ids():
    """
    Load previously saved Storm IDs.
    """

    logger.info("Loading existing train/test storm IDs...")

    train_storms = (
        pd.read_csv(TRAIN_STORMS_PATH)["SID"]
        .tolist()
    )

    test_storms = (
        pd.read_csv(TEST_STORMS_PATH)["SID"]
        .tolist()
    )

    logger.info("Storm IDs loaded successfully.")

    return train_storms, test_storms



# ==========================================================
# Storm Based Train/Test Split
# ==========================================================

def storm_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    regenerate: bool = False,
):
    """
    Split dataset by Storm ID (SID).

    If train/test storm files already exist, reuse them.
    Otherwise create a new split and save it.

    Parameters
    ----------
    df : pd.DataFrame
        Feature engineered dataframe.

    train_ratio : float
        Fraction of storms used for training.

    regenerate : bool
        If True, ignore existing split and create a new one.
    """

    logger.info("Creating storm-based train/test split...")

    train_storms = None
    test_storms = None

    # --------------------------------------------------
    # Load Existing Split
    # --------------------------------------------------

    if (
        not regenerate
        and TRAIN_STORMS_PATH.exists()
        and TEST_STORMS_PATH.exists()
        and TRAIN_STORMS_PATH.stat().st_size > 0
        and TEST_STORMS_PATH.stat().st_size > 0
    ):

        logger.info("Existing train/test split found.")

        try:

            train_storms, test_storms = load_storm_ids()

        except (
            pd.errors.EmptyDataError,
            FileNotFoundError,
            KeyError,
        ):

            logger.warning(
                "Existing split is invalid. A new split will be generated."
            )

    # --------------------------------------------------
    # Generate New Split
    # --------------------------------------------------

    if train_storms is None or test_storms is None:

        logger.info("Generating new train/test split...")

        storm_ids = df["SID"].unique()

        np.random.seed(RANDOM_STATE)

        storm_ids = np.random.permutation(storm_ids)

        split_index = int(
            len(storm_ids) * train_ratio
        )

        train_storms = storm_ids[:split_index]

        test_storms = storm_ids[split_index:]

        save_storm_ids(
            train_storms,
            test_storms,
        )

    # --------------------------------------------------
    # Create Train/Test DataFrames
    # --------------------------------------------------

    train_df = (
        df[
            df["SID"].isin(train_storms)
        ]
        .copy()
    )

    test_df = (
        df[
            df["SID"].isin(test_storms)
        ]
        .copy()
    )

    # --------------------------------------------------
    # Verify No Storm Leakage
    # --------------------------------------------------

    common_storms = set(train_df["SID"]).intersection(
        set(test_df["SID"])
    )

    if common_storms:
        raise ValueError(
            "Storm leakage detected between train and test sets."
        )

    logger.info(
        "Storm split validation passed. No storm leakage detected."
    )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    logger.info("=" * 60)
    logger.info("STORM SPLIT SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"Total Storms     : {len(train_storms) + len(test_storms):,}"
    )

    logger.info(
        f"Training Storms  : {len(train_storms):,}"
    )

    logger.info(
        f"Testing Storms   : {len(test_storms):,}"
    )

    logger.info(
        f"Training Samples : {len(train_df):,}"
    )

    logger.info(
        f"Testing Samples  : {len(test_df):,}"
    )

    logger.info("=" * 60)

    return train_df, test_df


# ==========================================================
# Create Features and Targets
# ==========================================================

def create_features_targets(
    df: pd.DataFrame,
    forecast_horizon: int = 6,
):
    """
    Create feature matrix X and target matrix y.
    """

    target_lat, target_lon = TARGET_COLUMNS[
        forecast_horizon
    ]

    X = df[FEATURE_COLUMNS]

    y = df[
        [
            target_lat,
            target_lon,
        ]
    ]

    return X, y


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    df = load_feature_dataset()

    df = prepare_training_data(
        df,
        forecast_horizon=6,
    )

    train_df, test_df = storm_based_split(
        df
        # regenerate=True,
    )

    X_train, y_train = create_features_targets(
        train_df,
        forecast_horizon=6,
    )

    X_test, y_test = create_features_targets(
        test_df,
        forecast_horizon=6,
    )

    logger.info("=" * 60)
    logger.info("TRAINING DATA SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Features        : {X_train.shape[1]}")
    logger.info(f"Training Samples: {len(X_train):,}")
    logger.info(f"Testing Samples : {len(X_test):,}")

    logger.info("=" * 60)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()