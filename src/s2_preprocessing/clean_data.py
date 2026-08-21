"""
clean_data.py

Cleans the raw IBTrACS dataset and produces a temporally regular, leakage-free
observation table for feature engineering.

Key guarantees of the output
----------------------------
1. Every row belongs to a `main` track (spur/provisional fragments removed).
2. Every row sits on a synoptic hour (00/06/12/18 UTC), so a shift of one row
   is exactly `OBSERVATION_INTERVAL` hours. This is what makes the multi-horizon
   targets in feature_engineering meaningful.
3. `SEGMENT_ID` identifies runs of strictly contiguous 6-hourly observations.
   Row filtering can leave gaps inside a storm; grouping by SID after that would
   silently produce lag features spanning 12h or 24h. Downstream code must
   compute lags and targets per SEGMENT_ID, while train/test splits stay keyed
   on SID so no storm straddles the boundary.
4. Imputation is causal. Only forward-fill is used, so no row can ever contain
   information from a later observation. Position (LAT/LON) is never imputed --
   a filled position fabricates a stationary storm and corrupts both the motion
   features and the persistence baseline.

Pipeline
--------
load -> track type -> season -> dtypes -> longitude -> coordinates -> nature
     -> sort -> deduplicate -> synoptic hours -> causal imputation
     -> segmentation -> minimum length -> save
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.s1_ingestion.dataset_loader import load_dataset
from src.utils import get_logger, Timer
from src.utils.config import (
    RAW_DATA_PATH,
    CLEAN_DATA_PATH,
    START_YEAR,
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    DATETIME_COLUMNS,
    OBSERVATION_INTERVAL,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    CSV_NA_VALUES,
)

logger = get_logger(__name__)


# ==========================================================
# Cleaning Configuration
# ==========================================================
# Move these into config.py once you are happy with the values.

# IBTrACS track_type: "main" is the primary track; "spur" rows are secondary
# fragments of the same system and must not be treated as observations.
VALID_TRACK_TYPES = ("main",)

# IBTrACS nature codes:
#   DS = disturbance, TS = tropical, ET = extratropical,
#   SS = subtropical, NR = not reported, MX = mixture
# Restricting to TS/SS keeps the dynamics homogeneous. Including ET roughly
# doubles late-life recurving samples but mixes in a different physical regime.
# Whichever you pick, justify it in the README.
VALID_NATURES = ("TS", "SS")

# Synoptic observation hours. Derived from OBSERVATION_INTERVAL so the two
# cannot drift apart.
SYNOPTIC_HOURS = tuple(range(0, 24, OBSERVATION_INTERVAL))

# Position is never imputed; these are the fields that may be forward-filled.
IMPUTABLE_COLUMNS = ["WMO_WIND", "WMO_PRES", "STORM_SPEED", "STORM_DIR", "DIST2LAND"]

# Maximum consecutive forward-fills. 2 steps = 12 hours of carried-forward
# intensity, which is about the limit of defensibility.
MAX_FFILL_STEPS = 2

# A segment must be long enough to supply the deepest lag plus the longest
# forecast horizon. 3 lags + 12 steps (72h) + 1 = 16, with headroom.
MIN_SEGMENT_OBSERVATIONS = 20

# Columns that must be present and non-null for a row to be usable at all.
ESSENTIAL_COLUMNS = ["SID", "ISO_TIME", "LAT", "LON"]

CLEAN_PARQUET_PATH = CLEAN_DATA_PATH.with_suffix(".parquet")


def _log_drop(stage: str, before: int, after: int) -> None:
    """Log how many rows a stage removed, and what fraction that is."""

    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    logger.info(f"  {stage:<32} -{removed:>9,} rows ({pct:5.2f}%)  ->  {after:,}")


# ==========================================================
# Filter Track Type
# ==========================================================

def filter_track_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only primary ("main") tracks.

    Spur rows are secondary track fragments for the same system. Leaving them in
    creates duplicate, slightly-inconsistent observations for the same storm.
    """

    if "TRACK_TYPE" not in df.columns:
        logger.warning("TRACK_TYPE column absent — skipping track type filter.")
        return df

    before = len(df)
    track_type = df["TRACK_TYPE"].astype(str).str.strip().str.lower()
    valid = [t.lower() for t in VALID_TRACK_TYPES]

    df = df[track_type.isin(valid)].copy()

    _log_drop(f"track_type in {VALID_TRACK_TYPES}", before, len(df))
    return df


# ==========================================================
# Filter Season
# ==========================================================

def filter_by_year(df: pd.DataFrame, start_year: int = START_YEAR) -> pd.DataFrame:
    """
    Keep cyclone records from `start_year` onwards.

    Satellite coverage before ~1980 is sparse enough that best-track positions
    carry substantially larger observational error.
    """

    before = len(df)

    df = df.copy()
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df = df[df["SEASON"] >= start_year]

    _log_drop(f"season >= {start_year}", before, len(df))
    return df


# ==========================================================
# Select Required Columns
# ==========================================================

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to the required schema.

    The updated dataset_loader already reads only these columns, so this is now
    a cheap guard rather than a real projection. It is kept so the module still
    works if fed a full-width frame.
    """

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[REQUIRED_COLUMNS].copy()


# ==========================================================
# Convert Data Types
# ==========================================================

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce datetime and numeric columns, then drop rows with unusable timestamps.

    `errors="coerce"` produces NaT for malformed timestamps. Those rows must go:
    they sort to the end of every storm and corrupt the time-difference logic
    that segmentation depends on.
    """

    logger.info("Converting data types...")
    df = df.copy()

    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in NUMERIC_COLUMNS:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        coerced = df[col].isna().sum() - before_na

        if coerced > 0:
            logger.info(f"  {col}: {coerced:,} non-numeric values -> NaN")

    before = len(df)
    df = df[df["ISO_TIME"].notna()]
    _log_drop("valid ISO_TIME", before, len(df))

    return df


# ==========================================================
# Normalize Longitude
# ==========================================================

def normalize_longitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrap longitude into [-180, 180).

    Vectorized modular arithmetic. The original `.apply(lambda ...)` ran a
    Python-level call per row, which dominates runtime on a ~700k-row frame.
    """

    logger.info("Normalizing longitude to [-180, 180)...")

    df = df.copy()
    df["LON"] = ((df["LON"] + 180.0) % 360.0) - 180.0

    return df


# ==========================================================
# Validate Coordinates
# ==========================================================

def drop_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing or physically impossible positions.

    Position is the target variable's basis, so a bad fix cannot be repaired --
    only discarded. Catching it here means validate_features never has to raise.
    """

    before = len(df)

    valid = (
        df["LAT"].between(*LATITUDE_RANGE)
        & df["LON"].between(*LONGITUDE_RANGE)
        & df["LAT"].notna()
        & df["LON"].notna()
    )

    df = df[valid].copy()

    _log_drop("valid coordinates", before, len(df))
    return df


# ==========================================================
# Filter Nature
# ==========================================================

def filter_nature(df: pd.DataFrame, natures: tuple = VALID_NATURES) -> pd.DataFrame:
    """
    Keep only the storm nature classes being modelled.
    """

    if "NATURE" not in df.columns:
        logger.warning("NATURE column absent — skipping nature filter.")
        return df

    before = len(df)
    nature = df["NATURE"].astype(str).str.strip().str.upper()

    observed = nature.value_counts()
    logger.info(f"  Nature distribution: {observed.to_dict()}")

    df = df[nature.isin([n.upper() for n in natures])].copy()

    _log_drop(f"nature in {natures}", before, len(df))
    return df


# ==========================================================
# Sort
# ==========================================================

def sort_storms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort chronologically within each storm.
    """

    logger.info("Sorting storms chronologically...")
    return df.sort_values(["SID", "ISO_TIME"], kind="mergesort").reset_index(drop=True)


# ==========================================================
# Deduplicate
# ==========================================================

def remove_duplicate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate (SID, ISO_TIME) pairs, keeping the most complete row.

    The original kept whichever row happened to come first. Ranking by null
    count first means a duplicate pair resolves to the row carrying the most
    usable measurements.
    """

    before = len(df)

    df = df.copy()
    df["_null_count"] = df[NUMERIC_COLUMNS].isna().sum(axis=1)

    df = (
        df.sort_values(["SID", "ISO_TIME", "_null_count"], kind="mergesort")
        .drop_duplicates(subset=["SID", "ISO_TIME"], keep="first")
        .drop(columns="_null_count")
        .reset_index(drop=True)
    )

    _log_drop("unique (SID, ISO_TIME)", before, len(df))
    return df


# ==========================================================
# Regularize To Synoptic Hours
# ==========================================================

def filter_synoptic_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only observations on synoptic hours.

    IBTrACS interpolates many agencies onto a 3-hourly grid. Without this
    filter, `shift(1)` means 3 hours for some storms and 6 for others, so every
    "6-hour forecast" metric downstream is measuring an inconsistent interval.
    """

    before = len(df)

    hour_counts = df["ISO_TIME"].dt.hour.value_counts().sort_index()
    logger.info(f"  Observation hours present: {hour_counts.to_dict()}")

    df = df[df["ISO_TIME"].dt.hour.isin(SYNOPTIC_HOURS)].copy()

    # Non-zero minutes indicate off-grid observations that would also break the
    # fixed-interval assumption.
    off_grid = (df["ISO_TIME"].dt.minute != 0) | (df["ISO_TIME"].dt.second != 0)

    if off_grid.any():
        logger.info(f"  Dropping {off_grid.sum():,} off-grid (non-zero minute) rows.")
        df = df[~off_grid].copy()

    _log_drop(f"hour in {SYNOPTIC_HOURS}", before, len(df))
    return df


# ==========================================================
# Causal Missing Value Handling
# ==========================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing intensity fields using only past information.

    Forward-fill only, capped at MAX_FFILL_STEPS. Backward-fill is deliberately
    absent: it would copy a later observation into an earlier row that is used
    as a model input, leaking the future into the features.

    LAT and LON are excluded from imputation entirely — see drop_invalid_coordinates.
    """

    logger.info("Imputing missing values (causal forward-fill only)...")

    df = df.copy()
    fillable = [c for c in IMPUTABLE_COLUMNS if c in df.columns]

    for col in fillable:
        before_na = df[col].isna().sum()

        df[col] = df.groupby("SID")[col].ffill(limit=MAX_FFILL_STEPS)

        after_na = df[col].isna().sum()
        logger.info(
            f"  {col:<12} missing {before_na:>8,} -> {after_na:>8,} "
            f"({before_na - after_na:,} filled)"
        )

    before = len(df)
    df = df.dropna(subset=["WMO_WIND", "WMO_PRES"]).reset_index(drop=True)
    _log_drop("wind & pressure present", before, len(df))

    return df


# ==========================================================
# Track Segmentation
# ==========================================================

def assign_track_segments(
    df: pd.DataFrame,
    interval_hours: int = OBSERVATION_INTERVAL,
) -> pd.DataFrame:
    """
    Split each storm into runs of strictly contiguous observations.

    Every filter above can remove rows from the middle of a storm. Grouping by
    SID after that yields lag features and targets that silently span 12h or 24h
    instead of 6h. A SEGMENT_ID makes the discontinuity explicit: downstream
    code groups by SEGMENT_ID for anything time-relative, and by SID for
    train/test splitting.

    Must run after all row-removal stages.
    """

    logger.info("Assigning contiguous track segments...")

    df = df.copy()
    expected = pd.Timedelta(hours=interval_hours)

    time_delta = df.groupby("SID")["ISO_TIME"].diff()

    # A new segment starts at the first row of a storm (NaT delta) and wherever
    # the gap is not exactly one interval.
    is_break = time_delta.ne(expected)

    segment_number = is_break.groupby(df["SID"]).cumsum().astype(int)

    df["SEGMENT_ID"] = (
        df["SID"].astype(str) + "_S" + segment_number.astype(str).str.zfill(2)
    )

    n_storms = df["SID"].nunique()
    n_segments = df["SEGMENT_ID"].nunique()
    fragmented = n_segments - n_storms

    logger.info(f"  Storms   : {n_storms:,}")
    logger.info(f"  Segments : {n_segments:,}  (+{fragmented:,} from internal gaps)")

    return df


def validate_segment_spacing(
    df: pd.DataFrame,
    interval_hours: int = OBSERVATION_INTERVAL,
) -> None:
    """
    Assert that spacing inside every segment is exactly `interval_hours`.

    This is the invariant the entire multi-horizon target construction rests on.
    Cheap to check, and catches any future pipeline edit that reintroduces gaps.
    """

    deltas = df.groupby("SEGMENT_ID")["ISO_TIME"].diff().dropna()
    expected = pd.Timedelta(hours=interval_hours)
    bad = deltas[deltas != expected]

    if not bad.empty:
        raise AssertionError(
            f"{len(bad):,} intra-segment gaps are not {interval_hours}h "
            f"(observed: {bad.unique()[:5]})"
        )

    logger.info(f"  Spacing invariant holds: all intervals = {interval_hours}h")


def filter_short_segments(
    df: pd.DataFrame,
    min_observations: int = MIN_SEGMENT_OBSERVATIONS,
) -> pd.DataFrame:
    """
    Drop segments too short to yield lag features and long-horizon targets.
    """

    before = len(df)

    lengths = df.groupby("SEGMENT_ID")["ISO_TIME"].transform("size")
    df = df[lengths >= min_observations].reset_index(drop=True)

    _log_drop(f"segment length >= {min_observations}", before, len(df))
    logger.info(f"  Segments remaining: {df['SEGMENT_ID'].nunique():,}")

    return df


# ==========================================================
# Save / Load
# ==========================================================

def save_dataset(df: pd.DataFrame, write_csv: bool = True) -> None:
    """
    Persist the cleaned dataset.

    Parquet preserves dtypes (so ISO_TIME does not need re-parsing) and is far
    faster to read. CSV is written alongside it for compatibility with any code
    still pointing at CLEAN_DATA_PATH.
    """

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(CLEAN_PARQUET_PATH, index=False)
        logger.info(f"Saved: {CLEAN_PARQUET_PATH}")
    except Exception as exc:
        logger.warning(f"Parquet write failed ({exc}); relying on CSV.")
        write_csv = True

    if write_csv:
        df.to_csv(CLEAN_DATA_PATH, index=False)
        logger.info(f"Saved: {CLEAN_DATA_PATH}")


def load_clean_dataset() -> pd.DataFrame:
    """
    Load the cleaned dataset, preferring Parquet.

    Import this from feature_engineering instead of calling read_csv directly.
    """

    if CLEAN_PARQUET_PATH.exists():
        return pd.read_parquet(CLEAN_PARQUET_PATH)

    return pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=["ISO_TIME"],
        keep_default_na=False,
        na_values=CSV_NA_VALUES,
    )


# ==========================================================
# Summary
# ==========================================================

def cleaning_summary(df: pd.DataFrame, raw_rows: int) -> dict:
    """
    Log and return end-of-pipeline statistics.
    """

    lengths = df.groupby("SEGMENT_ID").size()

    summary = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "retention_pct": round(len(df) / raw_rows * 100, 2) if raw_rows else 0.0,
        "storms": int(df["SID"].nunique()),
        "segments": int(df["SEGMENT_ID"].nunique()),
        "median_segment_length": int(lengths.median()),
        "season_min": int(df["SEASON"].min()),
        "season_max": int(df["SEASON"].max()),
    }

    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Raw rows              : {summary['raw_rows']:,}")
    logger.info(f"  Clean rows            : {summary['clean_rows']:,}")
    logger.info(f"  Retention             : {summary['retention_pct']:.2f}%")
    logger.info(f"  Storms                : {summary['storms']:,}")
    logger.info(f"  Segments              : {summary['segments']:,}")
    logger.info(f"  Median segment length : {summary['median_segment_length']} obs")
    logger.info(
        f"  Seasons               : {summary['season_min']}-{summary['season_max']}"
    )
    logger.info("=" * 60)

    return summary


# ==========================================================
# Main Pipeline
# ==========================================================

def main(
    start_year: int = START_YEAR,
    natures: tuple = VALID_NATURES,
    min_observations: int = MIN_SEGMENT_OBSERVATIONS,
    write_csv: bool = True,
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline and return the cleaned frame.
    """

    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    with Timer("Data Cleaning Pipeline"):
        df = load_dataset(RAW_DATA_PATH)
        raw_rows = len(df)

        logger.info("-" * 60)
        logger.info("FILTERING")
        logger.info("-" * 60)

        df = filter_track_type(df)
        df = filter_by_year(df, start_year=start_year)
        df = select_columns(df)
        df = convert_data_types(df)
        df = normalize_longitude(df)
        df = drop_invalid_coordinates(df)
        df = filter_nature(df, natures=natures)

        df = sort_storms(df)
        df = remove_duplicate_observations(df)
        df = filter_synoptic_hours(df)
        df = handle_missing_values(df)

        logger.info("-" * 60)
        logger.info("SEGMENTATION")
        logger.info("-" * 60)

        df = assign_track_segments(df)
        df = filter_short_segments(df, min_observations=min_observations)
        validate_segment_spacing(df)

        cleaning_summary(df, raw_rows)
        save_dataset(df, write_csv=write_csv)

    logger.info("DATA CLEANING COMPLETE")

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean the raw IBTrACS dataset.")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument(
        "--natures", nargs="+", default=list(VALID_NATURES),
        help="IBTrACS nature codes to keep (e.g. TS SS ET).",
    )
    parser.add_argument(
        "--min-observations", type=int, default=MIN_SEGMENT_OBSERVATIONS,
        help="Minimum observations per contiguous segment.",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Write Parquet only (faster; requires updating downstream readers).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        start_year=args.start_year,
        natures=tuple(args.natures),
        min_observations=args.min_observations,
        write_csv=not args.no_csv,
    )