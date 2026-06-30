"""
create_sequences.py

Create sequential data for LSTM training.
"""

import logging
import numpy as np
import pandas as pd

# ==========================================================
# Configure Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# LSTM Features
# ==========================================================

LSTM_FEATURES = [
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "STORM_SPEED",
    "STORM_DIR",
]

TARGET_COLUMNS = [
    "TARGET_LAT_6H",
    "TARGET_LON_6H",
]

SEQUENCE_LENGTH = 8

# ==========================================================
# Create Sequences
# ==========================================================

def create_sequences(df: pd.DataFrame):
    """
    Create LSTM sequences.

    Returns
    -------
    X : np.ndarray
        Shape = (samples, timesteps, features)

    y : np.ndarray
        Shape = (samples, 2)

    sequence_sids : np.ndarray
        Storm ID corresponding to each sequence
    """

    logger.info(
        f"Creating sequences (length={SEQUENCE_LENGTH})..."
    )

    X_sequences = []
    y_sequences = []
    sequence_sids = []

    # ------------------------------------------------------
    # Process cyclone by cyclone
    # ------------------------------------------------------

    for sid, storm_df in df.groupby("SID"):

        storm_df = storm_df.sort_values(
            "ISO_TIME"
        )

        storm_df = storm_df.dropna(
            subset=LSTM_FEATURES + TARGET_COLUMNS
        )

        values = storm_df[
            LSTM_FEATURES
        ].values

        targets = storm_df[
            TARGET_COLUMNS
        ].values

        # --------------------------------------------------
        # Sliding Window
        # --------------------------------------------------

        for i in range(
            len(storm_df) - SEQUENCE_LENGTH
        ):

            X_sequences.append(
                values[
                    i : i + SEQUENCE_LENGTH
                ]
            )

            y_sequences.append(
                targets[
                    i + SEQUENCE_LENGTH - 1
                ]
            )

            sequence_sids.append(sid)

    X = np.array(X_sequences)

    y = np.array(y_sequences)

    sequence_sids = np.array(sequence_sids)

    logger.info(
        f"Sequence dataset shape: {X.shape}"
    )

    logger.info(
        f"Target dataset shape: {y.shape}"
    )

    logger.info(
        f"Sequence SID shape: {sequence_sids.shape}"
    )

    return (
        X,
        y,
        sequence_sids
    )