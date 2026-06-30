"""
predict.py

Load saved model and make predictions on test samples.
"""

import logging

from src.training.save_model import load_model
from src.training.train import main as prepare_data

# ==========================================================
# Configure Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def predict_sample():

    logger.info("=" * 60)
    logger.info("CYCLONE TRACK PREDICTION")
    logger.info("=" * 60)

    # Load model
    model = load_model()

    # Get test data
    _, X_test, _, y_test = prepare_data()

    # Debug longitude features for first sample
    print("\nFIRST TEST SAMPLE LONGITUDE FEATURES")
    print(
        X_test.iloc[0][
            [
                "LON",
                "LON_LAG_1",
                "LON_LAG_2",
                "LON_LAG_3",
                "DELTA_LON"
            ]
        ]
    )

    # First 10 samples
    X_samples = X_test.iloc[:10]

    predictions = model.predict(X_samples)

    for i in range(10):

        actual_lat = y_test.iloc[i, 0]
        actual_lon = y_test.iloc[i, 1]

        pred_lat = predictions[i][0]
        pred_lon = predictions[i][1]

        logger.info("-" * 60)

        logger.info(f"Sample {i+1}")

        logger.info(
            f"Actual Latitude     : {actual_lat:.4f}"
        )

        logger.info(
            f"Predicted Latitude  : {pred_lat:.4f}"
        )

        logger.info(
            f"Actual Longitude    : {actual_lon:.4f}"
        )

        logger.info(
            f"Predicted Longitude : {pred_lon:.4f}"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    predict_sample()