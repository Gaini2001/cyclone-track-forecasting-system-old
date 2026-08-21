"""
validate_features.py

Quality gate between feature engineering and training.

The checks that matter here are temporal, not dimensional. Range checks on
latitude and wind speed cannot fail in practice; misaligned targets, lags that
span a gap, and leakage from the future all can, and all of them produce
optimistic metrics rather than crashes. Those are the invariants this module
asserts:

    * a target at horizon h is the observed position exactly h hours later,
      for the same storm
    * LAG_k equals the value k rows back within the same contiguous segment
    * consecutive rows inside a segment are exactly OBSERVATION_INTERVAL apart
    * structural NaNs sit only at segment heads (lags) and tails (targets)
    * no feature is close to perfectly correlated with its target

Failures are collected and reported together, then raised once, so a broken
dataset takes one run to diagnose instead of eight.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils import get_logger, Timer
from src.utils.config import (
    FEATURE_DATA_PATH,
    FEATURE_PARQUET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DELTA_TARGET_COLUMNS,
    TARGET_MODE,
    FORECAST_HORIZON_HOURS,
    OBSERVATION_INTERVAL,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    TROPICAL_LATITUDE_RANGE,
    MIN_WIND_SPEED,
    MAX_WIND_SPEED,
    MIN_PRESSURE,
    MAX_PRESSURE,
    CSV_NA_VALUES,
)

logger = get_logger(__name__)

# Tolerance for float comparison of reconstructed coordinates (degrees).
POSITION_TOLERANCE = 1e-6

# A feature this correlated with the target is almost certainly the target.
LEAKAGE_CORRELATION_THRESHOLD = 0.99

# Deepest lag present in the feature set.
MAX_LAG = 3


# ==========================================================
# Report
# ==========================================================

class ValidationReport:
    """
    Accumulates findings so every problem surfaces in a single run.
    """

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)
        logger.error(f"  FAIL  {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(f"  WARN  {message}")

    def ok(self, message: str) -> None:
        self.info.append(message)
        logger.info(f"  PASS  {message}")

    @property
    def passed(self) -> bool:
        return not self.errors

    def summarize(self, strict: bool = True) -> None:
        logger.info("=" * 68)
        logger.info(
            f"VALIDATION SUMMARY — {len(self.info)} passed, "
            f"{len(self.warnings)} warnings, {len(self.errors)} failures"
        )
        logger.info("=" * 68)

        for message in self.errors:
            logger.error(f"  {message}")

        if self.errors and strict:
            raise ValueError(
                f"Feature validation failed with {len(self.errors)} error(s). "
                "See the log above."
            )


# ==========================================================
# Load
# ==========================================================

def load_feature_dataset() -> pd.DataFrame:
    """
    Load the feature dataset, preferring Parquet so ISO_TIME keeps its dtype.

    The temporal checks below are impossible if timestamps arrive as strings.
    """

    logger.info("Loading feature dataset...")

    if FEATURE_PARQUET_PATH.exists():
        df = pd.read_parquet(FEATURE_PARQUET_PATH)
    else:
        df = pd.read_csv(
            FEATURE_DATA_PATH,
            parse_dates=["ISO_TIME"],
            keep_default_na=False,
            na_values=CSV_NA_VALUES,
        )

    logger.info(f"Dataset loaded. Shape: {df.shape}")
    return df


def _group_key(df: pd.DataFrame, report: ValidationReport) -> str:
    """
    Column to group by for time-relative operations.

    SEGMENT_ID marks runs of contiguous observations. Falling back to SID means
    lags may silently span gaps left by row filtering.
    """

    if "SEGMENT_ID" in df.columns:
        return "SEGMENT_ID"

    report.warn(
        "SEGMENT_ID missing — falling back to SID. Lag and target checks "
        "cannot detect gaps introduced by row filtering."
    )
    return "SID"


# ==========================================================
# Schema
# ==========================================================

def check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Every declared feature and target must exist, and they must not overlap.
    """

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing_features:
        # An error, not a warning: train.py silently trains on whatever
        # subset happens to be present, so a typo becomes an invisible
        # change to the model rather than a failure.
        report.error(f"Missing feature columns: {missing_features}")
    else:
        report.ok(f"All {len(FEATURE_COLUMNS)} feature columns present")

    target_columns = TARGET_COLUMNS if TARGET_MODE == "absolute" else DELTA_TARGET_COLUMNS
    required = [col for cols in target_columns.values() for col in cols]
    missing_targets = [c for c in required if c not in df.columns]

    if missing_targets:
        report.error(
            f"Missing target columns for TARGET_MODE={TARGET_MODE!r}: {missing_targets}"
        )
    else:
        report.ok(f"All target columns present for TARGET_MODE={TARGET_MODE!r}")

    overlap = set(FEATURE_COLUMNS) & set(required)

    if overlap:
        report.error(f"Columns used as both feature and target: {sorted(overlap)}")
    else:
        report.ok("No feature/target overlap")

    for column in ("SID", "ISO_TIME"):
        if column not in df.columns:
            report.error(f"Required identifier column missing: {column}")

    if "ISO_TIME" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["ISO_TIME"]
    ):
        report.error("ISO_TIME is not a datetime dtype — temporal checks unreliable")


# ==========================================================
# Structural Integrity
# ==========================================================

def check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    One observation per storm per timestamp.
    """

    if "SID" not in df.columns or "ISO_TIME" not in df.columns:
        return

    n_duplicates = df.duplicated(subset=["SID", "ISO_TIME"]).sum()

    if n_duplicates:
        report.error(f"{n_duplicates:,} duplicate (SID, ISO_TIME) rows")
    else:
        report.ok("No duplicate observations")


def check_segment_spacing(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Consecutive rows within a segment must be exactly one interval apart.

    This is the invariant the entire target construction rests on. If it does
    not hold, "24-hour forecast" means different things in different rows.
    """

    key = _group_key(df, report)

    if "ISO_TIME" not in df.columns:
        return

    deltas = df.groupby(key)["ISO_TIME"].diff().dropna()
    expected = pd.Timedelta(hours=OBSERVATION_INTERVAL)
    irregular = deltas[deltas != expected]

    if len(irregular):
        observed = irregular.value_counts().head(5).to_dict()
        report.error(
            f"{len(irregular):,} intra-{key} gaps are not {OBSERVATION_INTERVAL}h "
            f"(most common: {observed})"
        )
    else:
        report.ok(
            f"All intra-{key} intervals are exactly {OBSERVATION_INTERVAL}h "
            f"({len(deltas):,} checked)"
        )


# ==========================================================
# Lag Correctness
# ==========================================================

def check_lag_features(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Verify each lag column equals its source shifted within the segment.

    Catches lags computed on an unsorted frame, grouped by the wrong key, or
    carried across a discontinuity.
    """

    key = _group_key(df, report)

    lag_sources = {
        "LAT": "LAT",
        "LON": "LON",
        "WMO_WIND": "WIND",
        "WMO_PRES": "PRESSURE",
    }

    failures = 0

    for source, prefix in lag_sources.items():
        if source not in df.columns:
            continue

        for lag in range(1, MAX_LAG + 1):
            column = f"{prefix}_LAG_{lag}"

            if column not in df.columns:
                continue

            expected = df.groupby(key)[source].shift(lag)
            both_present = df[column].notna() & expected.notna()

            mismatched = (
                (df.loc[both_present, column] - expected[both_present]).abs()
                > POSITION_TOLERANCE
            ).sum()

            # A lag that is populated where the shift is undefined means the
            # value came from outside the segment.
            crossed = (df[column].notna() & expected.isna()).sum()

            if mismatched or crossed:
                failures += 1
                report.error(
                    f"{column}: {mismatched:,} value mismatches, "
                    f"{crossed:,} values crossing a {key} boundary"
                )

    if not failures:
        report.ok(f"Lag features consistent with {key} shifts")


# ==========================================================
# Target Alignment
# ==========================================================

def check_target_alignment(
    df: pd.DataFrame,
    report: ValidationReport,
    sample_size: int = None,
) -> None:
    """
    Verify each target is the observed position exactly h hours later.

    This is the single most important check in the file. It is a direct test of
    the assumption that shifting n rows equals advancing h hours -- an
    assumption that silently fails whenever the observation cadence is not what
    the configuration claims, and which turns a "24-hour forecast" into
    something easier and unreported.

    Implemented as a self-join on (SID, ISO_TIME + h) rather than by
    re-deriving the shift, so it tests the outcome instead of repeating the
    original logic.
    """

    if "ISO_TIME" not in df.columns or "SID" not in df.columns:
        return

    frame = df if sample_size is None or len(df) <= sample_size else df.sample(
        sample_size, random_state=0
    )

    positions = df[["SID", "ISO_TIME", "LAT", "LON"]].rename(
        columns={"ISO_TIME": "VALID_TIME", "LAT": "TRUE_LAT", "LON": "TRUE_LON"}
    )

    for horizon in FORECAST_HORIZON_HOURS:
        if TARGET_MODE == "absolute":
            columns = TARGET_COLUMNS.get(horizon, [])
        else:
            columns = DELTA_TARGET_COLUMNS.get(horizon, [])

        if not columns or any(c not in frame.columns for c in columns):
            continue

        target_lat_col, target_lon_col = columns

        probe = frame[["SID", "ISO_TIME", "LAT", "LON", target_lat_col, target_lon_col]]
        probe = probe.dropna(subset=[target_lat_col, target_lon_col]).copy()

        if probe.empty:
            report.warn(f"{horizon}h: no non-null targets to verify")
            continue

        probe["VALID_TIME"] = probe["ISO_TIME"] + pd.Timedelta(hours=horizon)

        merged = probe.merge(positions, on=["SID", "VALID_TIME"], how="left")

        unmatched = merged["TRUE_LAT"].isna().sum()

        if unmatched:
            report.error(
                f"{horizon}h: {unmatched:,} of {len(merged):,} targets have no "
                f"observation at ISO_TIME + {horizon}h — the target does not "
                "correspond to a real future position"
            )
            continue

        if TARGET_MODE == "absolute":
            predicted_lat = merged[target_lat_col]
            predicted_lon = merged[target_lon_col]
        else:
            predicted_lat = merged["LAT"] + merged[target_lat_col]
            predicted_lon = merged["LON"] + merged[target_lon_col]

        lat_error = (predicted_lat - merged["TRUE_LAT"]).abs()
        # Wrap the longitude comparison: a reconstructed 181.4 and an observed
        # -178.6 are the same meridian, and a naive subtraction reads 360.
        lon_error = (
            ((predicted_lon - merged["TRUE_LON"] + 180.0) % 360.0) - 180.0
        ).abs()

        mismatched = ((lat_error > POSITION_TOLERANCE) | (lon_error > POSITION_TOLERANCE)).sum()

        if mismatched:
            report.error(
                f"{horizon}h: {mismatched:,} targets do not match the observed "
                f"position {horizon}h ahead (max error "
                f"{max(lat_error.max(), lon_error.max()):.4f} deg)"
            )
        else:
            report.ok(
                f"{horizon}h targets verified against observed positions "
                f"({len(merged):,} rows)"
            )


# ==========================================================
# Missing Values
# ==========================================================

def check_missing_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Core inputs must be complete; structural NaNs must sit where expected.

    Lag NaNs belong only in the first `lag` rows of a segment; target NaNs only
    in the last `horizon / interval` rows. A NaN anywhere else is a real defect,
    which the previous "structural NaNs are expected" message could not
    distinguish from a correct one.
    """

    key = _group_key(df, report)

    core = [c for c in ("LAT", "LON", "WMO_WIND", "WMO_PRES") if c in df.columns]
    core_missing = df[core].isna().sum()

    if core_missing.sum():
        report.error(
            f"Missing values in core inputs: "
            f"{core_missing[core_missing > 0].to_dict()}"
        )
    else:
        report.ok("Core inputs (LAT, LON, WIND, PRES) complete")

    position = df.groupby(key).cumcount()
    segment_length = df.groupby(key)["ISO_TIME"].transform("size")

    # ---- Lag columns ----
    for lag in range(1, MAX_LAG + 1):
        for column in [c for c in df.columns if c.endswith(f"_LAG_{lag}")]:
            unexpected = (df[column].isna() & (position >= lag)).sum()

            if unexpected:
                report.error(
                    f"{column}: {unexpected:,} NaNs outside the first {lag} "
                    f"row(s) of a {key}"
                )

    # ---- Target columns ----
    for horizon in FORECAST_HORIZON_HOURS:
        steps = horizon // OBSERVATION_INTERVAL
        columns = (
            TARGET_COLUMNS if TARGET_MODE == "absolute" else DELTA_TARGET_COLUMNS
        ).get(horizon, [])

        for column in columns:
            if column not in df.columns:
                continue

            from_end = segment_length - position - 1
            unexpected = (df[column].isna() & (from_end >= steps)).sum()

            if unexpected:
                report.error(
                    f"{column}: {unexpected:,} NaNs outside the final {steps} "
                    f"row(s) of a {key}"
                )

    report.ok("Structural NaN placement checked")


def check_infinite_values(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Infinities usually mean a division by a zero time delta or distance.
    """

    numeric = df.select_dtypes(include=np.number)
    counts = np.isinf(numeric).sum()
    counts = counts[counts > 0]

    if not counts.empty:
        report.error(f"Infinite values: {counts.to_dict()}")
    else:
        report.ok("No infinite values")


# ==========================================================
# Physical Plausibility
# ==========================================================

def check_physical_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Bound every physical quantity on both sides.

    The previous bounds (wind >= 0, pressure >= 0) admit a 3 mb central
    pressure and a 900 kt wind. Bounds that cannot fail are not checks.
    """

    checks = [
        ("LAT", LATITUDE_RANGE, True),
        ("LON", LONGITUDE_RANGE, True),
        ("WMO_WIND", (MIN_WIND_SPEED, MAX_WIND_SPEED), True),
        ("WMO_PRES", (MIN_PRESSURE, MAX_PRESSURE), True),
    ]

    for column, (low, high), fatal in checks:
        if column not in df.columns:
            continue

        values = df[column].dropna()
        outside = ((values < low) | (values > high)).sum()

        if outside:
            message = (
                f"{column}: {outside:,} values outside [{low}, {high}] "
                f"(observed range {values.min():.2f} to {values.max():.2f})"
            )
            report.error(message) if fatal else report.warn(message)
        else:
            report.ok(f"{column} within [{low}, {high}]")

    # Advisory: geometrically valid but climatologically implausible.
    if "LAT" in df.columns:
        low, high = TROPICAL_LATITUDE_RANGE
        extreme = ((df["LAT"] < low) | (df["LAT"] > high)).sum()

        if extreme:
            report.warn(
                f"{extreme:,} rows outside {TROPICAL_LATITUDE_RANGE} latitude — "
                "unusual for tropical cyclones; check for corrupt records"
            )


# ==========================================================
# Leakage Sentinel
# ==========================================================

def check_target_leakage(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Flag features that are near-perfectly correlated with the target.

    Under TARGET_MODE="displacement" a hit here means something genuinely
    leaked. Under "absolute" it will fire immediately on LAT and LON -- which
    is not a bug in the data so much as the reason absolute targets make R^2
    meaningless: the answer is already an input.
    """

    horizon = max(FORECAST_HORIZON_HOURS)
    columns = (
        TARGET_COLUMNS if TARGET_MODE == "absolute" else DELTA_TARGET_COLUMNS
    ).get(horizon, [])

    columns = [c for c in columns if c in df.columns]
    features = [c for c in FEATURE_COLUMNS if c in df.columns]

    if not columns or not features:
        return

    subset = df[features + columns].dropna()

    if len(subset) < 100:
        report.warn("Too few complete rows for a leakage correlation check")
        return

    suspects = []

    for target in columns:
        correlations = subset[features].corrwith(subset[target]).abs()
        hits = correlations[correlations > LEAKAGE_CORRELATION_THRESHOLD]
        suspects.extend(f"{feature} ~ {target} (r={value:.4f})"
                        for feature, value in hits.items())

    if suspects:
        message = f"Features near-perfectly correlated with target: {suspects}"

        if TARGET_MODE == "absolute":
            report.warn(
                message + " — expected under TARGET_MODE='absolute'; this is "
                "why displacement targets give a more honest R^2"
            )
        else:
            report.error(message)
    else:
        report.ok("No feature correlates with the target above "
                  f"{LEAKAGE_CORRELATION_THRESHOLD}")


# ==========================================================
# Usable Sample Counts
# ==========================================================

def report_usable_samples(df: pd.DataFrame, report: ValidationReport) -> None:
    """
    Rows surviving dropna at each horizon.

    Longer horizons lose the tail of every segment, so the 72h model may train
    on materially less data than the 6h one. That difference belongs in the
    results table, not buried.
    """

    features = [c for c in FEATURE_COLUMNS if c in df.columns]

    logger.info("-" * 68)
    logger.info(f"  {'Horizon':<10} {'Usable rows':>14} {'Storms':>10} {'% of total':>12}")
    logger.info("  " + "-" * 50)

    for horizon in FORECAST_HORIZON_HOURS:
        columns = (
            TARGET_COLUMNS if TARGET_MODE == "absolute" else DELTA_TARGET_COLUMNS
        ).get(horizon, [])
        columns = [c for c in columns if c in df.columns]

        if not columns:
            continue

        usable = df.dropna(subset=features + columns)
        storms = usable["SID"].nunique() if "SID" in usable.columns else 0

        logger.info(
            f"  {str(horizon) + 'h':<10} {len(usable):>14,} {storms:>10,} "
            f"{len(usable) / len(df) * 100:>11.1f}%"
        )

        if len(usable) == 0:
            report.error(f"{horizon}h horizon has zero usable training rows")

    logger.info("-" * 68)


# ==========================================================
# Main
# ==========================================================

def validate(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    """
    Run every check and return the report.
    """

    report = ValidationReport()

    logger.info("-" * 68)
    logger.info("SCHEMA")
    logger.info("-" * 68)
    check_schema(df, report)

    logger.info("-" * 68)
    logger.info("STRUCTURE")
    logger.info("-" * 68)
    check_duplicates(df, report)
    check_segment_spacing(df, report)

    logger.info("-" * 68)
    logger.info("TEMPORAL CORRECTNESS")
    logger.info("-" * 68)
    check_lag_features(df, report)
    check_target_alignment(df, report)

    logger.info("-" * 68)
    logger.info("VALUES")
    logger.info("-" * 68)
    check_missing_values(df, report)
    check_infinite_values(df, report)
    check_physical_ranges(df, report)

    logger.info("-" * 68)
    logger.info("LEAKAGE")
    logger.info("-" * 68)
    check_target_leakage(df, report)

    logger.info("-" * 68)
    logger.info("SAMPLE AVAILABILITY")
    logger.info("-" * 68)
    report_usable_samples(df, report)

    report.summarize(strict=strict)
    return report


def main(strict: bool = True) -> ValidationReport:

    logger.info("=" * 68)
    logger.info("FEATURE VALIDATION")
    logger.info("=" * 68)

    with Timer("Feature Validation"):
        df = load_feature_dataset()
        report = validate(df, strict=strict)

    if report.passed:
        logger.info("ALL VALIDATIONS PASSED")

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the feature dataset.")
    parser.add_argument(
        "--no-strict", action="store_true",
        help="Report failures without raising (useful while iterating).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(strict=not args.no_strict)