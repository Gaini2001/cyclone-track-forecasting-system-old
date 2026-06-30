"""
prediction_visualization.py

Visualize Random Forest Predictions.
"""

import matplotlib.pyplot as plt

from src.training.save_model import load_model
from src.training.train import (
    load_feature_dataset,
    prepare_training_data,
    storm_based_split,
    create_features_targets,
)


def load_test_data():

    df = load_feature_dataset()

    df = prepare_training_data(
        df,
        forecast_horizon=6,
    )

    _, test_df = storm_based_split(df)

    return test_df


def main():

    print("Loading model...")

    model = load_model()

    print("Loading test data...")

    test_df = load_test_data()

    X_test, y_test = create_features_targets(
        test_df,
        forecast_horizon=6,
    )

    print("Generating predictions...")

    predictions = model.predict(X_test)

    # ======================================================
    # Top 50 Samples
    # ======================================================

    n = 50

    actual_lat = y_test.iloc[:n, 0].values
    actual_lon = y_test.iloc[:n, 1].values

    pred_lat = predictions[:n, 0]
    pred_lon = predictions[:n, 1]

    samples = range(n)

    # ======================================================
    # Latitude Plot
    # ======================================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        samples,
        actual_lat,
        marker="o",
        label="Actual Latitude",
    )

    plt.plot(
        samples,
        pred_lat,
        marker="x",
        linestyle="--",
        label="Predicted Latitude",
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Latitude")

    plt.title(
        "Actual vs Predicted Latitude (First 50 Samples)"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "reports/actual_vs_predicted_latitude.png",
        dpi=300,
    )

    plt.show()

    # ======================================================
    # Longitude Plot
    # ======================================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        samples,
        actual_lon,
        marker="o",
        label="Actual Longitude",
    )

    plt.plot(
        samples,
        pred_lon,
        marker="x",
        linestyle="--",
        label="Predicted Longitude",
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Longitude")

    plt.title(
        "Actual vs Predicted Longitude (First 50 Samples)"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "reports/actual_vs_predicted_longitude.png",
        dpi=300,
    )

    plt.show()

    # ======================================================
    # Combined Track Plot
    # ======================================================

    plt.figure(figsize=(10, 8))

    plt.plot(
        actual_lon,
        actual_lat,
        marker="o",
        label="Actual",
    )

    plt.plot(
        pred_lon,
        pred_lat,
        marker="x",
        linestyle="--",
        label="Predicted",
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title(
        "Top 50 Cyclone Track Predictions"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "reports/top50_track_predictions.png",
        dpi=300,
    )

    plt.show()

    print("\nPlots saved successfully.")


if __name__ == "__main__":
    main()