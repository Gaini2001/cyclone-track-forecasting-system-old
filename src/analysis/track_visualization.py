"""
track_visualization.py

Visualize Actual vs Predicted Cyclone Track
using Random Forest.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.training.save_model import load_model
from src.training.train import (
    load_feature_dataset,
    prepare_training_data,
    storm_based_split,
    create_features_targets,
)

# ==========================================================
# Load Test Dataset
# ==========================================================

def load_test_data():

    df = load_feature_dataset()

    df = prepare_training_data(
        df,
        forecast_horizon=6,
    )

    _, test_df = storm_based_split(
        df
    )

    return test_df


# ==========================================================
# Main
# ==========================================================

def main():

    print("Loading model...")

    model = load_model()

    print("Loading test dataset...")

    test_df = load_test_data()

    # ------------------------------------------------------
    # Select One Cyclone
    # ------------------------------------------------------

    storm_id = test_df["SID"].iloc[0]

    storm_df = (
        test_df[
            test_df["SID"] == storm_id
        ]
        .copy()
    )

    print(f"Selected Storm: {storm_id}")

    # ------------------------------------------------------
    # Create Features
    # ------------------------------------------------------

    X_storm, y_storm = create_features_targets(
        storm_df,
        forecast_horizon=6,
    )

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------

    predictions = model.predict(
        X_storm
    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    # Actual future positions

    plt.plot(
        y_storm.iloc[:, 1],     # longitude
        y_storm.iloc[:, 0],     # latitude
        marker="o",
        label="Actual Track",
    )

    # Predicted future positions

    plt.plot(
        predictions[:, 1],
        predictions[:, 0],
        marker="x",
        linestyle="--",
        label="Predicted Track",
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title(
        f"Cyclone Track Prediction\nStorm ID: {storm_id}"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "reports/sample_track_prediction.png",
        dpi=300
    )

    plt.show()

    print(
        "\nSaved to reports/sample_track_prediction.png"
    )


if __name__ == "__main__":
    main()