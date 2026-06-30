"""
analyze_missing.py

Analyze missing values in the cleaned IBTrACS dataset.

This script helps determine the best strategy for handling
missing meteorological variables before model training.
"""

import pandas as pd

from src.utils.config import CLEAN_DATA_PATH


def main():

    print("=" * 70)
    print("IBTrACS Missing Data Analysis")
    print("=" * 70)

    df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"\nDataset Shape : {df.shape}")

    # ======================================================
    # Missing Values
    # ======================================================

    print("\n" + "=" * 70)
    print("Missing Values")
    print("=" * 70)

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    print(missing)

    # ======================================================
    # Storms with ALL Wind Missing
    # ======================================================

    print("\n" + "=" * 70)
    print("Storms with ALL WMO_WIND Missing")
    print("=" * 70)

    wind_missing = (
        df.groupby("SID")["WMO_WIND"]
        .apply(lambda x: x.isna().all())
    )

    print(f"Storms : {wind_missing.sum()}")

    # ======================================================
    # Storms with ALL Pressure Missing
    # ======================================================

    print("\n" + "=" * 70)
    print("Storms with ALL WMO_PRES Missing")
    print("=" * 70)

    pressure_missing = (
        df.groupby("SID")["WMO_PRES"]
        .apply(lambda x: x.isna().all())
    )

    print(f"Storms : {pressure_missing.sum()}")

    # ======================================================
    # Missing Before Fill
    # ======================================================

    print("\n" + "=" * 70)
    print("Before Forward/Backward Fill")
    print("=" * 70)

    print(f"WMO_WIND : {df['WMO_WIND'].isna().sum():,}")

    print(f"WMO_PRES : {df['WMO_PRES'].isna().sum():,}")

    # ======================================================
    # Simulate Filling
    # ======================================================

    filled = df.copy()

    filled[["WMO_WIND", "WMO_PRES"]] = (
        filled
        .groupby("SID")[["WMO_WIND", "WMO_PRES"]]
        .transform(lambda x: x.ffill().bfill())
    )

    print("\n" + "=" * 70)
    print("After Forward/Backward Fill")
    print("=" * 70)

    print(f"WMO_WIND : {filled['WMO_WIND'].isna().sum():,}")

    print(f"WMO_PRES : {filled['WMO_PRES'].isna().sum():,}")

    # ======================================================
    # Rows Still Missing
    # ======================================================

    remaining = filled[
        filled["WMO_WIND"].isna() |
        filled["WMO_PRES"].isna()
    ]

    print("\n" + "=" * 70)
    print("Rows Still Missing After Fill")
    print("=" * 70)

    print(len(remaining))

    print("\n")

    print("=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()