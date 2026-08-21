"""
train.py

Dataset preparation and splitting for cyclone track forecasting.

Two things this module is responsible for getting right:

1. The split must reflect deployment. A forecast system is trained on the past
   and run on the future, so the default holds out later seasons. A random
   partition of storm IDs trains on 2021 and tests on 1987, and storms from the
   same season share the synoptic state that drives their motion -- both make
   the test set easier than reality.

2. Split definitions must not silently drift. Split membership is persisted
   under a filename fingerprinted by the split configuration, and every cached
   split is checked for full coverage of the current dataset before it is
   reused. A cache written by a different configuration is simply not found,
   rather than loaded as though it were correct.

The prepared data keeps identifier and position columns alongside X and y.
Evaluation needs the issue-time position to reconstruct absolute forecasts from
displacement predictions, and per-storm plots need SID; recovering them later by
re-reading the source and matching row counts is fragile and unnecessary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils import get_logger, Timer
from src.utils.config import (
    FEATURE_DATA_PATH,
    FEATURE_PARQUET_PATH,
    FEATURE_COLUMNS,
    CSV_NA_VALUES,
    RANDOM_STATE,
    DEFAULT_FORECAST_HORIZON,
    TARGET_MODE,
    target_columns,
    SPLIT_STRATEGY,
    TRAIN_END_SEASON,
    VAL_END_SEASON,
    TRAIN_RATIO,
    VAL_RATIO,
    split_config,
    split_paths,
    split_fingerprint,
)

logger = get_logger(__name__)

# Columns carried alongside X and y. Not features -- context for evaluation,
# plotting, and reconstruction of absolute positions.
META_COLUMNS = [
    "SID",
    "SEGMENT_ID",
    "ISO_TIME",
    "SEASON",
    "BASIN",
    "LAT",
    "LON",
]


# ==========================================================
# Containers
# ==========================================================

@dataclass
class Split:
    """
    One partition of the dataset.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Target matrix (displacement or absolute, per TARGET_MODE).
    meta : pd.DataFrame
        Identifiers and issue-time position, row-aligned with X and y.
    """

    name: str
    X: pd.DataFrame
    y: pd.DataFrame
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.X)

    @property
    def storms(self) -> int:
        return self.meta["SID"].nunique()

    @property
    def seasons(self) -> tuple[int, int]:
        seasons = self.meta["SEASON"]
        return int(seasons.min()), int(seasons.max())


@dataclass
class PreparedData:
    """
    Train / validation / test partitions plus the split provenance.
    """

    train: Split
    test: Split
    val: Split = None
    horizon: int = DEFAULT_FORECAST_HORIZON
    target_mode: str = TARGET_MODE
    config: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.train.X.columns)

    @property
    def target_names(self) -> list[str]:
        return list(self.train.y.columns)


# ==========================================================
# Load
# ==========================================================

def load_feature_dataset() -> pd.DataFrame:
    """
    Load the feature dataset, preferring Parquet.
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

    logger.info(f"Dataset shape: {df.shape}")
    return df


# ==========================================================
# Prepare
# ==========================================================

def prepare_training_data(
    df: pd.DataFrame,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Drop rows that cannot supply a complete feature vector or target.

    Rows are lost at both ends of every segment: lag and rolling features are
    undefined at the head, targets at the tail. The retention figure is worth
    watching -- if it is very low, a single expensive feature (a long rolling
    window, say) may be costing more data than it earns.
    """

    logger.info(
        f"Preparing {forecast_horizon}h dataset (target mode: {TARGET_MODE})..."
    )

    targets = target_columns(forecast_horizon)

    missing_targets = [c for c in targets if c not in df.columns]

    if missing_targets:
        raise KeyError(
            f"Missing target columns {missing_targets}. Re-run feature "
            f"engineering, or check TARGET_MODE={TARGET_MODE!r}."
        )

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing_features:
        # Raise rather than proceed with a subset: silently training on fewer
        # features than declared makes every later comparison meaningless.
        raise KeyError(
            f"Missing feature columns: {missing_features}. Re-run feature "
            "engineering or update FEATURE_GROUPS in config."
        )

    rows_before = len(df)
    prepared = df.dropna(subset=FEATURE_COLUMNS + targets).reset_index(drop=True)

    logger.info(
        f"  Usable rows: {len(prepared):,} of {rows_before:,} "
        f"({len(prepared) / rows_before * 100:.1f}%)"
    )
    logger.info(f"  Storms: {prepared['SID'].nunique():,}")

    if prepared.empty:
        raise ValueError(
            f"No usable rows at the {forecast_horizon}h horizon. "
            "Check feature engineering output and segment lengths."
        )

    return prepared


# ==========================================================
# Split Generation
# ==========================================================

def season_split(
    df: pd.DataFrame,
    train_end: int = TRAIN_END_SEASON,
    val_end: int = VAL_END_SEASON,
) -> dict[str, np.ndarray]:
    """
    Hold out later seasons.

    The split that matches deployment: fit on history, forecast forward. Report
    this one. Storms are assigned by their season so a storm never straddles a
    boundary.
    """

    logger.info(
        f"Season split — train <= {train_end}, val {train_end + 1}-{val_end}, "
        f"test > {val_end}"
    )

    storm_seasons = df.groupby("SID")["SEASON"].min()

    return {
        "train": storm_seasons[storm_seasons <= train_end].index.to_numpy(),
        "val": storm_seasons[
            (storm_seasons > train_end) & (storm_seasons <= val_end)
        ].index.to_numpy(),
        "test": storm_seasons[storm_seasons > val_end].index.to_numpy(),
    }


def storm_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """
    Randomly partition storm IDs.

    Kept for comparison against the season split. Expect it to look better than
    the season split does -- that gap is informative, and worth reporting rather
    than hiding: it is the cost of evaluating on genuinely unseen conditions.
    """

    logger.info(
        f"Random storm split — {train_ratio:.0%} / {val_ratio:.0%} / "
        f"{1 - train_ratio - val_ratio:.0%} (seed {seed})"
    )

    # A local generator; np.random.seed() would mutate global state that other
    # components also rely on.
    rng = np.random.default_rng(seed)
    storm_ids = rng.permutation(df["SID"].unique())

    n = len(storm_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": storm_ids[:n_train],
        "val": storm_ids[n_train:n_train + n_val],
        "test": storm_ids[n_train + n_val:],
    }


def generate_split(df: pd.DataFrame, strategy: str = SPLIT_STRATEGY) -> dict:
    """
    Build a split according to the configured strategy.
    """

    if strategy == "season":
        return season_split(df)

    if strategy == "storm":
        return storm_split(df)

    raise ValueError(f"Unknown split strategy: {strategy!r}")


# ==========================================================
# Split Persistence
# ==========================================================

def save_split(assignments: dict[str, np.ndarray], config: dict) -> None:
    """
    Persist split membership plus the configuration that produced it.
    """

    paths = split_paths(config)
    paths["train"].parent.mkdir(parents=True, exist_ok=True)

    for name in ("train", "val", "test"):
        pd.DataFrame({"SID": assignments.get(name, [])}).to_csv(
            paths[name], index=False
        )

    metadata = {
        "config": config,
        "fingerprint": split_fingerprint(config),
        "counts": {name: int(len(assignments.get(name, []))) for name in
                   ("train", "val", "test")},
    }

    paths["meta"].write_text(json.dumps(metadata, indent=2))
    logger.info(f"Split saved with fingerprint {metadata['fingerprint']}")


def load_split(config: dict) -> dict[str, np.ndarray] | None:
    """
    Load a previously saved split, or None if it is absent.
    """

    paths = split_paths(config)

    if not all(paths[name].exists() for name in ("train", "test")):
        return None

    assignments = {}

    for name in ("train", "val", "test"):
        if paths[name].exists():
            assignments[name] = pd.read_csv(paths[name])["SID"].to_numpy()
        else:
            assignments[name] = np.array([])

    logger.info(f"Reusing cached split {split_fingerprint(config)}")
    return assignments


def validate_split(
    df: pd.DataFrame,
    assignments: dict[str, np.ndarray],
) -> list[str]:
    """
    Check a split against the dataset it will be applied to.

    Returns a list of problems; empty means the split is usable.

    Two failure modes matter. Overlap between partitions is outright leakage.
    Incomplete coverage is quieter: storms present in the data but absent from
    every partition are dropped from training and testing alike, which happens
    whenever the upstream filters change and a stale split is reused.
    """

    problems = []

    sets = {name: set(ids.tolist()) for name, ids in assignments.items()}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets.get(a, set()) & sets.get(b, set())

        if overlap:
            problems.append(
                f"{len(overlap)} storms appear in both {a} and {b} — leakage"
            )

    dataset_storms = set(df["SID"].unique())
    assigned = set().union(*sets.values()) if sets else set()

    unassigned = dataset_storms - assigned

    if unassigned:
        problems.append(
            f"{len(unassigned)} storms in the dataset belong to no partition — "
            "the cached split predates the current feature data"
        )

    if not sets.get("train"):
        problems.append("train partition is empty")

    if not sets.get("test"):
        problems.append("test partition is empty")

    return problems


def resolve_split(
    df: pd.DataFrame,
    strategy: str = SPLIT_STRATEGY,
    regenerate: bool = False,
) -> dict[str, np.ndarray]:
    """
    Return a validated split, reusing the cache only when it is sound.
    """

    config = split_config()

    if not regenerate:
        cached = load_split(config)

        if cached is not None:
            problems = validate_split(df, cached)

            if problems:
                for problem in problems:
                    logger.warning(f"  Cached split rejected: {problem}")
                logger.info("Regenerating split.")
            else:
                return cached

    assignments = generate_split(df, strategy=strategy)
    problems = validate_split(df, assignments)

    if problems:
        raise ValueError("Generated split is invalid: " + "; ".join(problems))

    save_split(assignments, config)
    return assignments


# ==========================================================
# Matrix Construction
# ==========================================================

def create_features_targets(
    df: pd.DataFrame,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    name: str = "split",
) -> Split:
    """
    Split a frame into feature matrix, target matrix, and metadata.
    """

    targets = target_columns(forecast_horizon)
    meta_available = [c for c in META_COLUMNS if c in df.columns]

    return Split(
        name=name,
        X=df[FEATURE_COLUMNS].reset_index(drop=True),
        y=df[targets].reset_index(drop=True),
        meta=df[meta_available].reset_index(drop=True),
    )


def summarize(data: PreparedData) -> None:
    """
    Log the composition of each partition.
    """

    logger.info("=" * 72)
    logger.info(
        f"SPLIT SUMMARY — {data.config.get('strategy')} strategy, "
        f"{data.horizon}h horizon, {data.target_mode} targets"
    )
    logger.info("=" * 72)
    logger.info(
        f"  {'Partition':<12} {'Rows':>12} {'Storms':>10} {'Seasons':>16} {'Share':>8}"
    )
    logger.info("  " + "-" * 62)

    splits = [s for s in (data.train, data.val, data.test) if s is not None and len(s)]
    total = sum(len(s) for s in splits)

    for split in splits:
        low, high = split.seasons
        logger.info(
            f"  {split.name:<12} {len(split):>12,} {split.storms:>10,} "
            f"{f'{low}-{high}':>16} {len(split) / total * 100:>7.1f}%"
        )

    logger.info("  " + "-" * 62)
    logger.info(f"  {'TOTAL':<12} {total:>12,}")
    logger.info(f"  Features         : {len(data.feature_names)}")
    logger.info(f"  Targets          : {data.target_names}")
    logger.info("=" * 72)

    if data.test is not None and data.test.storms < 50:
        logger.warning(
            f"  Only {data.test.storms} test storms — error estimates will be "
            "noisy. Consider moving the season boundary earlier."
        )


# ==========================================================
# Entry Point
# ==========================================================

def prepare_data(
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    strategy: str = SPLIT_STRATEGY,
    regenerate_split: bool = False,
    df: pd.DataFrame = None,
) -> PreparedData:
    """
    Load, filter, split, and package the dataset.

    Parameters
    ----------
    forecast_horizon : int
        Horizon in hours.
    strategy : str
        "season" (default) or "storm".
    regenerate_split : bool
        Ignore any cached split.
    df : pd.DataFrame, optional
        Pre-loaded feature frame, to avoid re-reading when preparing several
        horizons in one run.

    Returns
    -------
    PreparedData
    """

    with Timer(f"Data Preparation ({forecast_horizon}h)"):
        if df is None:
            df = load_feature_dataset()

        prepared = prepare_training_data(df, forecast_horizon=forecast_horizon)
        assignments = resolve_split(
            prepared, strategy=strategy, regenerate=regenerate_split
        )

        partitions = {}

        for name in ("train", "val", "test"):
            subset = prepared[prepared["SID"].isin(assignments.get(name, []))]

            partitions[name] = (
                create_features_targets(subset, forecast_horizon, name=name)
                if not subset.empty else None
            )

        data = PreparedData(
            train=partitions["train"],
            val=partitions["val"],
            test=partitions["test"],
            horizon=forecast_horizon,
            target_mode=TARGET_MODE,
            config=split_config(),
        )

    summarize(data)
    return data


def main(
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    strategy: str = SPLIT_STRATEGY,
    regenerate_split: bool = False,
) -> PreparedData:
    """
    Prepare data for one horizon.

    Note the changed return type: this now returns a PreparedData object rather
    than a bare (X_train, X_test, y_train, y_test) tuple, because the tuple
    discarded the validation set and the identifier columns that evaluation and
    plotting need.
    """

    logger.info("=" * 72)
    logger.info("TRAINING DATA PREPARATION")
    logger.info("=" * 72)

    return prepare_data(
        forecast_horizon=forecast_horizon,
        strategy=strategy,
        regenerate_split=regenerate_split,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training data.")
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_FORECAST_HORIZON,
        help="Forecast horizon in hours.",
    )
    parser.add_argument(
        "--strategy", choices=("season", "storm"), default=SPLIT_STRATEGY,
        help="Split strategy.",
    )
    parser.add_argument(
        "--regenerate-split", action="store_true",
        help="Ignore any cached split definition.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        forecast_horizon=args.horizon,
        strategy=args.strategy,
        regenerate_split=args.regenerate_split,
    )