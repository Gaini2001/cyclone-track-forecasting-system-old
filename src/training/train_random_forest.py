"""
train_random_forest.py

Train a Random Forest model for cyclone track prediction.
"""

import logging

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.training.train import main as prepare_training_data


# ==========================================================
# Configure Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Build Random Forest Pipeline
# ==========================================================

def build_pipeline():
    """
    Create Random Forest training pipeline.
    """

    logger.info("Building Random Forest pipeline...")

    pipeline = Pipeline(
        steps=[
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        ]
    )

    logger.info("Pipeline created successfully.")

    return pipeline


# ==========================================================
# Train Model
# ==========================================================

def train_model(
    pipeline,
    X_train,
    y_train,
):
    """
    Train Random Forest model.
    """

    logger.info("Training Random Forest model...")

    pipeline.fit(
        X_train,
        y_train,
    )

    logger.info("Model training completed.")

    return pipeline


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("RANDOM FOREST TRAINING")
    logger.info("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_training_data()

    pipeline = build_pipeline()

    model = train_model(
        pipeline,
        X_train,
        y_train,
    )

    logger.info("=" * 60)
    logger.info("Random Forest training completed successfully.")
    logger.info("=" * 60)

    return (
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()