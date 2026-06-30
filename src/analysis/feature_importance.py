"""
feature_importance.py

Analyze Random Forest feature importance.
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.training.save_model import load_model
from src.utils.config import FEATURE_COLUMNS


def main():

    model = load_model()

    rf = model.named_steps["model"]

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": rf.feature_importances_,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("\nTop 15 Features\n")
    print(
        importance_df.iloc[1:16]
    )

    plt.figure(figsize=(10,6))

    plt.barh(
        importance_df["Feature"][:15][::-1],
        importance_df["Importance"][:15][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png",
        dpi=300
    )

    plt.show()


if __name__ == "__main__":
    main()