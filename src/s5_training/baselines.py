"""
baselines.py

Reference forecasts that any learned model must beat to be worth anything.

Design note
-----------
Every baseline returns predictions in the *same target space as the models*
(displacement or absolute, per TARGET_MODE), as a plain (n, 2) array. They then
flow through exactly the same reconstruction and scoring path in evaluate.py.

The original computed each baseline's error internally with its own call to
track_error_statistics, which meant baselines and models were scored by
separate code. When the comparison you are making is "is the model better than
persistence", the two numbers being compared must come from the same function
or the comparison is not sound.

Baselines
---------
Persistence           the storm does not move. The conventional short-horizon
                      reference; surprisingly hard to beat at 6h.
Linear extrapolation  the storm continues at its current velocity. The
                      reference that actually matters -- a model that cannot
                      beat it has learned nothing about steering.
Climatology           the mean displacement observed in training, optionally
                      conditioned on basin.
Linear regression     ordinary least squares on the full feature set. Tells you
                      whether the gradient boosting is earning its complexity;
                      if a linear model matches it, it is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger, Timer
from src.utils.config import (
    TARGET_MODE,
    FORECAST_HORIZONS,
    OBSERVATION_INTERVAL,
)
from src.s5_training.train import PreparedData, Split

logger = get_logger(__name__)


# ==========================================================
# Helpers
# ==========================================================

def _to_target_space(
    delta_lat: np.ndarray,
    delta_lon: np.ndarray,
    split: Split,
) -> np.ndarray:
    """
    Express a predicted displacement in the active target space.

    Under "displacement" the offsets are already correct. Under "absolute" they
    are added to the issue-time position so the baseline emits the same kind of
    quantity a model would.
    """

    delta_lat = np.asarray(delta_lat, dtype=float)
    delta_lon = np.asarray(delta_lon, dtype=float)

    if TARGET_MODE == "displacement":
        return np.column_stack([delta_lat, delta_lon])

    base_lat = split.meta["LAT"].to_numpy(dtype=float)
    base_lon = split.meta["LON"].to_numpy(dtype=float)

    absolute_lat = base_lat + delta_lat
    absolute_lon = ((base_lon + delta_lon + 180.0) % 360.0) - 180.0

    return np.column_stack([absolute_lat, absolute_lon])


# ==========================================================
# Baselines
# ==========================================================

def persistence(split: Split, horizon: int, **_) -> np.ndarray:
    """
    The storm stays where it is: zero displacement.

    The standard reference at short lead times. At 6h it is genuinely
    competitive, which is why leading with a 6h headline number flatters a
    model more than it deserves.
    """

    zeros = np.zeros(len(split), dtype=float)
    return _to_target_space(zeros, zeros, split)


def linear_extrapolation(split: Split, horizon: int, **_) -> np.ndarray:
    """
    The storm continues on its current heading at its current speed.

    The reference that matters. Persistence is easy to beat; constant-velocity
    extrapolation captures most of what a track looks like over 24 hours, and
    a model that does not clearly beat it has not learned steering -- only
    inertia, which was already in the input.
    """

    steps = FORECAST_HORIZONS[horizon]

    delta_lat = split.X["DELTA_LAT"].to_numpy(dtype=float) * steps
    delta_lon = split.X["DELTA_LON"].to_numpy(dtype=float) * steps

    return _to_target_space(delta_lat, delta_lon, split)


def climatology(
    split: Split,
    horizon: int,
    train: Split = None,
    by_basin: bool = True,
    **_,
) -> np.ndarray:
    """
    The average displacement seen in training.

    Conditioning on basin makes this a real baseline rather than a formality:
    a global mean displacement averages westward Atlantic motion against
    eastward South Pacific motion and lands near zero, which just reproduces
    persistence.
    """

    if train is None:
        raise ValueError("Climatology needs the training split to average over.")

    train_deltas = _training_displacements(train, horizon)

    if by_basin and "BASIN" in train.meta.columns and "BASIN" in split.meta.columns:
        means = (
            pd.DataFrame({
                "BASIN": train.meta["BASIN"].to_numpy(),
                "DLAT": train_deltas[:, 0],
                "DLON": train_deltas[:, 1],
            })
            .groupby("BASIN")[["DLAT", "DLON"]]
            .mean()
        )

        global_mean = train_deltas.mean(axis=0)

        matched = split.meta["BASIN"].map(means["DLAT"]).to_numpy(dtype=float)
        delta_lat = np.where(np.isfinite(matched), matched, global_mean[0])

        matched = split.meta["BASIN"].map(means["DLON"]).to_numpy(dtype=float)
        delta_lon = np.where(np.isfinite(matched), matched, global_mean[1])
    else:
        mean = train_deltas.mean(axis=0)
        delta_lat = np.full(len(split), mean[0])
        delta_lon = np.full(len(split), mean[1])

    return _to_target_space(delta_lat, delta_lon, split)


def linear_regression(split: Split, horizon: int, train: Split = None, **_) -> np.ndarray:
    """
    Ordinary least squares on the full feature set.

    A complexity check rather than a physical baseline: if a linear model on
    the same features matches the gradient boosting, the nonlinearity is not
    buying anything and the honest conclusion is that the problem is close to
    linear at this horizon. That is a finding, not a failure.
    """

    from sklearn.linear_model import LinearRegression

    if train is None:
        raise ValueError("Linear regression baseline needs the training split.")

    model = LinearRegression()
    model.fit(train.X, train.y)

    return np.asarray(model.predict(split.X), dtype=float)


def _training_displacements(train: Split, horizon: int) -> np.ndarray:
    """
    Training-set displacements, whatever the target mode.
    """

    y = train.y.to_numpy(dtype=float)

    if TARGET_MODE == "displacement":
        return y

    base_lat = train.meta["LAT"].to_numpy(dtype=float)
    base_lon = train.meta["LON"].to_numpy(dtype=float)

    delta_lat = y[:, 0] - base_lat
    delta_lon = ((y[:, 1] - base_lon + 180.0) % 360.0) - 180.0

    return np.column_stack([delta_lat, delta_lon])


# ==========================================================
# Registry
# ==========================================================

BASELINE_REGISTRY = {
    "Persistence": {"fn": persistence, "needs_train": False},
    "Linear Extrapolation": {"fn": linear_extrapolation, "needs_train": False},
    "Climatology": {"fn": climatology, "needs_train": True},
    "Linear Regression": {"fn": linear_regression, "needs_train": True},
}

# The reference every skill score is measured against.
REFERENCE_BASELINE = "Persistence"


def run_all_baselines(
    data: PreparedData,
    split_name: str = "test",
    names: list[str] = None,
) -> dict[str, np.ndarray]:
    """
    Compute every baseline's predictions on the requested split.

    Returns
    -------
    dict
        Baseline name -> (n, 2) prediction array in the active target space.
        Scoring happens in evaluate.py, using the same code path as the models.
    """

    split = getattr(data, split_name)

    if split is None or len(split) == 0:
        raise ValueError(f"Split {split_name!r} is empty.")

    names = names or list(BASELINE_REGISTRY)
    predictions = {}

    logger.info(f"Computing baselines on {split_name} ({len(split):,} rows)...")

    with Timer(f"Baselines ({data.horizon}h)"):
        for name in names:
            spec = BASELINE_REGISTRY[name]

            try:
                predictions[name] = spec["fn"](
                    split,
                    data.horizon,
                    train=data.train if spec["needs_train"] else None,
                )
                logger.info(f"  {name}")
            except Exception as exc:
                logger.warning(f"  {name} failed: {exc}")

    return predictions