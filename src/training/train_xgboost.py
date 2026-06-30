"""
train_xgboost.py

Train an XGBoost model for cyclone track prediction.
"""

import logging

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

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
# Build XGBoost Pipeline
# ==========================================================

def build_pipeline():
    """
    Create XGBoost training pipeline.
    """

    logger.info("Building XGBoost pipeline...")

    pipeline = Pipeline(
        steps=[
            (
                "model",
                MultiOutputRegressor(
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                    )
                )
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
    Train XGBoost model.
    """

    logger.info("Training XGBoost model...")

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
    logger.info("XGBOOST TRAINING")
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
    logger.info("XGBoost training completed successfully.")
    logger.info("=" * 60)

    return (
        model,
        X_test,
        y_test,
    )


if __name__ == "__main__":
    main()