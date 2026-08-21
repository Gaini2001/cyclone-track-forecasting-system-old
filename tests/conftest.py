"""
conftest.py

Shared fixtures for the cyclone track forecasting test suite.

Every fixture here is seeded. An unseeded fixture produces tests that pass on
Tuesday and fail on Wednesday, and a flaky test gets muted rather than fixed.

The fixtures deliberately include the awkward cases, because those are where
the bugs were: a storm crossing the antimeridian, a storm with a mid-track gap
that must be split into segments, and several storms at once so that grouping
errors have somewhere to show up. A single well-behaved storm exercises almost
nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


SEED = 20240615
OBSERVATIONS_PER_STORM = 26  # enough for 3 lags + a 72h target (12 steps)


def _build_storm(
    sid: str,
    basin: str,
    start_lat: float,
    start_lon: float,
    delta_lat: float,
    delta_lon: float,
    n: int = OBSERVATIONS_PER_STORM,
    season: int = 2019,
    start: str = "2019-09-01 00:00",
    wobble: float = 0.06,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    One synthetic storm on a strict 6-hourly grid.

    `wobble` adds step-to-step variation to the motion. It matters: with a
    perfectly constant velocity, the 24h displacement is exactly four times the
    last 6h step, so DELTA_LAT correlates 1.0 with the target and any
    leakage check fires on an artifact of the fixture rather than a real
    defect. Set it to 0 for tests that need an exactly straight track.
    """

    rng = rng or np.random.default_rng(SEED)

    times = pd.date_range(start, periods=n, freq="6h")
    steps = np.arange(n)

    lat_steps = delta_lat + rng.normal(0, wobble, n)
    lon_steps = delta_lon + rng.normal(0, wobble, n)
    lat_steps[0] = 0.0
    lon_steps[0] = 0.0

    latitudes = start_lat + np.cumsum(lat_steps)
    longitudes = ((start_lon + np.cumsum(lon_steps) + 180.0) % 360.0) - 180.0

    return pd.DataFrame({
        "SID": sid,
        "SEGMENT_ID": f"{sid}_S01",
        "ISO_TIME": times,
        "SEASON": season,
        "BASIN": basin,
        "SUBBASIN": "MM",
        "LAT": latitudes,
        "LON": longitudes,
        "WMO_WIND": np.clip(35 + steps * 2.5 + rng.normal(0, 2, n), 20, 160),
        "WMO_PRES": np.clip(1005 - steps * 2.0 + rng.normal(0, 2, n), 880, 1015),
        "STORM_SPEED": rng.uniform(8, 22, n),
        "STORM_DIR": rng.uniform(0, 360, n),
        "DIST2LAND": np.clip(600 - steps * 20 + rng.normal(0, 10, n), 0, 2000),
        "NATURE": "TS",
        "TRACK_TYPE": "main",
    })


@pytest.fixture
def rng():
    """Seeded generator, so any test that needs randomness stays repeatable."""
    return np.random.default_rng(SEED)


@pytest.fixture
def single_storm(rng):
    """
    One well-behaved storm. The simplest case, for basic shape assertions.
    """

    return _build_storm(
        "2019001N10080", "NA", 12.0, -60.0, 0.45, -0.70, wobble=0.0, rng=rng
    )


@pytest.fixture
def multi_storm(rng):
    """
    Three storms in different basins.

    Grouping bugs only appear with more than one group: a lag computed on an
    ungrouped frame looks perfectly correct on a single-storm fixture and
    silently pulls the previous storm's final position on a real dataset.
    """

    storms = [
        _build_storm("2019001N10080", "NA", 12.0, -60.0, 0.45, -0.70,
                     start="2019-09-01 00:00", rng=rng),
        _build_storm("2019002N15140", "WP", 15.0, 140.0, 0.55, -0.40,
                     start="2019-09-05 00:00", rng=rng),
        _build_storm("2019003N08088", "NI", 9.0, 88.0, 0.30, 0.35,
                     start="2019-10-02 00:00", rng=rng),
    ]

    return pd.concat(storms, ignore_index=True)


@pytest.fixture
def dateline_storm(rng):
    """
    A storm crossing the antimeridian.

    The original pipeline discarded these outright, taking most of the West
    Pacific with them. Any longitude arithmetic that is not wrap-aware will
    produce a ~360 degree jump here.
    """

    return _build_storm("2019004N18175", "WP", 18.0, 175.0, 0.40, 1.20, rng=rng)


@pytest.fixture
def gapped_storm(rng):
    """
    One storm whose observations have a hole in the middle.

    Cleaning removes rows -- off-synoptic times, missing intensity -- and the
    result is a storm whose consecutive rows are not consecutive in time.
    Grouping by SID after that yields "6-hour" lags that are silently 24 hours
    old, which is why SEGMENT_ID exists.
    """

    storm = _build_storm("2019005N20090", "NA", 20.0, -90.0, 0.35, -0.55, rng=rng)

    # Drop four rows from the middle, leaving a 30-hour hole.
    kept = storm.drop(index=range(10, 14)).reset_index(drop=True)

    # Re-segment as clean_data would: a new segment wherever the gap is not 6h.
    delta = kept.groupby("SID")["ISO_TIME"].diff()
    breaks = delta.ne(pd.Timedelta(hours=6))
    numbers = breaks.groupby(kept["SID"]).cumsum().astype(int)
    kept["SEGMENT_ID"] = kept["SID"] + "_S" + numbers.astype(str).str.zfill(2)

    return kept


@pytest.fixture
def clean_frame(multi_storm, dateline_storm):
    """
    A realistic cleaned dataset: several basins, including a dateline crosser.

    This is the input feature engineering should be tested against.
    """

    return pd.concat([multi_storm, dateline_storm], ignore_index=True)


@pytest.fixture
def feature_frame(clean_frame):
    """
    The clean frame with the full feature pipeline applied.

    Built once per test rather than shared at session scope, so a test that
    mutates the frame cannot affect any other.
    """

    from src.s3_features.feature_engineering import (
        add_temporal_features,
        add_lag_features,
        add_motion_features,
        add_velocity_features,
        add_turning_features,
        add_rolling_features,
        add_acceleration_features,
        add_storm_context_features,
        create_target_variables,
    )

    df = clean_frame.sort_values(["SEGMENT_ID", "ISO_TIME"]).reset_index(drop=True)

    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_motion_features(df)
    df = add_velocity_features(df)
    df = add_turning_features(df)
    df = add_rolling_features(df)
    df = add_acceleration_features(df)
    df = add_storm_context_features(df)
    df = create_target_variables(df)

    return df