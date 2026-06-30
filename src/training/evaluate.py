"""
evaluate.py

Evaluate the trained Random Forest model.
"""

import logging
import numpy as np
from src.training.save_model import save_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.training.train_random_forest import (
    main as train_random_forest,
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
# Mean Haversine Distance
# ==========================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Compute Haversine distance in kilometers.
    """

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate model performance.
    """

    logger.info("Generating predictions...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    haversine = haversine_distance(
        y_test.iloc[:, 0].values,
        y_test.iloc[:, 1].values,
        predictions[:, 0],
        predictions[:, 1],
    ).mean()

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    logger.info(f"MAE                 : {mae:.4f}")
    logger.info(f"RMSE                : {rmse:.4f}")
    logger.info(f"R² Score            : {r2:.4f}")
    logger.info(f"Mean Track Error    : {haversine:.2f} km")

    logger.info("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    (
        model,
        X_test,
        y_test,
    ) = train_random_forest()

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_model(model)


if __name__ == "__main__":
    main()