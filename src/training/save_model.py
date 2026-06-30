"""
save_model.py

Save and load trained machine learning models.
"""

import logging
import joblib

from src.utils.config import (
    RANDOM_FOREST_MODEL_PATH,
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
# Save Model
# ==========================================================

def save_model(
    model,
    model_path=RANDOM_FOREST_MODEL_PATH,
):
    """
    Save trained model to disk.
    """

    logger.info(
        f"Saving model to:\n{model_path}"
    )

    joblib.dump(
        model,
        model_path,
    )

    logger.info(
        "Model saved successfully."
    )


# ==========================================================
# Load Model
# ==========================================================

def load_model(
    model_path=RANDOM_FOREST_MODEL_PATH,
):
    """
    Load trained model from disk.
    """

    logger.info(
        f"Loading model from:\n{model_path}"
    )

    model = joblib.load(
        model_path
    )

    logger.info(
        "Model loaded successfully."
    )

    return model