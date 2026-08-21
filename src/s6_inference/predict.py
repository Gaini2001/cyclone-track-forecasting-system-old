"""
predict.py

Inference interface for cyclone track forecasting.

The original was a debug script: it printed ten rows of predicted latitude and
longitude in degrees, with a leftover longitude dump for sample zero. This
turns it into something usable, and adds the artifact the project was missing --
a full forecast track from a single issue time, verified against what the storm
actually did.

Forecast strategies
-------------------
direct     One model per lead time, each predicting from the same issue time.
           Errors at 72h do not inherit errors at 6h, but it needs a trained
           model per horizon. This is the default, and it is what the trained
           artifacts support.

recursive  Step forward 6h at a time, feeding each prediction back as the next
           input. Needs only one model, but errors compound, and it requires
           forecasting intensity as well -- which this project does not model.
           Implemented with wind and pressure held constant, which is a real
           limitation and is reported as such rather than hidden.

Usage
-----
    python -m src.s6_inference.predict --list-storms
    python -m src.s6_inference.predict --sid 2019242N16145
    python -m src.s6_inference.predict --horizon 24 --samples 20
    python -m src.s6_inference.predict --export
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.utils import get_logger, Timer
from src.utils.metrics import (
    haversine_distance,
    offset_position,
    along_cross_track_error,
)
from src.utils.config import (
    TARGET_MODE,
    FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_HORIZON,
    OBSERVATION_INTERVAL,
    PREDICTION_PATH,
    FEATURE_COLUMNS,
    model_path,
)
from src.s5_training.train import prepare_data, load_feature_dataset, PreparedData, Split
from src.s5_training.save_model import load_model, read_metadata

logger = get_logger(__name__)

DEFAULT_MODEL = "xgboost"


# ==========================================================
# Model Access
# ==========================================================

def load_horizon_models(
    model_name: str = DEFAULT_MODEL,
    horizons: list[int] = None,
    feature_names: list[str] = None,
) -> dict[int, object]:
    """
    Load one model per horizon, skipping any that are missing.

    Feature names are checked against the sidecar. A model applied to a
    reordered or renamed feature set does not fail -- it returns confident
    nonsense -- so the check happens before any prediction is made.
    """

    horizons = horizons or list(FORECAST_HORIZON_HOURS)
    models = {}

    for horizon in horizons:
        path = model_path(model_name, horizon, TARGET_MODE)

        if not path.exists():
            logger.warning(f"  {model_name} @ {horizon}h: not trained")
            continue

        metadata = read_metadata(path)
        trained_on = metadata.get("feature_names")

        if feature_names and trained_on and trained_on != list(feature_names):
            logger.error(
                f"  {model_name} @ {horizon}h was trained on a different "
                "feature set — skipping. Retrain to use it."
            )
            continue

        models[horizon] = load_model(path)

    if not models:
        raise FileNotFoundError(
            f"No {model_name} models available. Train first:\n"
            f"  python -m src.s5_training.train_model --model {model_name} --horizon all"
        )

    return models


# ==========================================================
# Prediction
# ==========================================================

def predict_positions(model, X: pd.DataFrame, base_lat, base_lon) -> tuple:
    """
    Predict absolute positions.

    Model output is in whatever space TARGET_MODE dictates; under displacement
    it is an offset that must be added back to the issue-time position before
    it means anything geographic.
    """

    predictions = np.asarray(model.predict(X), dtype=float)

    if TARGET_MODE == "displacement":
        return offset_position(
            np.asarray(base_lat, dtype=float),
            np.asarray(base_lon, dtype=float),
            predictions[:, 0],
            predictions[:, 1],
        )

    return predictions[:, 0], predictions[:, 1]


def predict_split(
    model,
    split: Split,
    n_samples: int = None,
) -> pd.DataFrame:
    """
    Predict on rows of a prepared split, with verification columns.

    Returns
    -------
    pd.DataFrame
        Identifiers, forecast and observed positions, and error in kilometres.
    """

    if n_samples:
        X = split.X.iloc[:n_samples]
        y = split.y.iloc[:n_samples]
        meta = split.meta.iloc[:n_samples]
    else:
        X, y, meta = split.X, split.y, split.meta

    base_lat = meta["LAT"].to_numpy(dtype=float)
    base_lon = meta["LON"].to_numpy(dtype=float)

    pred_lat, pred_lon = predict_positions(model, X, base_lat, base_lon)

    actual = y.to_numpy(dtype=float)

    if TARGET_MODE == "displacement":
        true_lat, true_lon = offset_position(
            base_lat, base_lon, actual[:, 0], actual[:, 1]
        )
    else:
        true_lat, true_lon = actual[:, 0], actual[:, 1]

    errors = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)

    components = along_cross_track_error(
        base_lat, base_lon, true_lat, true_lon, pred_lat, pred_lon,
    )

    return pd.DataFrame({
        "SID": meta["SID"].to_numpy(),
        "ISO_TIME": meta["ISO_TIME"].to_numpy(),
        "BASIN": meta["BASIN"].to_numpy() if "BASIN" in meta else None,
        "CURRENT_LAT": base_lat,
        "CURRENT_LON": base_lon,
        "PRED_LAT": pred_lat,
        "PRED_LON": pred_lon,
        "TRUE_LAT": true_lat,
        "TRUE_LON": true_lon,
        "ERROR_KM": errors,
        "ALONG_TRACK_KM": components["along_track_km"],
        "CROSS_TRACK_KM": components["cross_track_km"],
    })


# ==========================================================
# Track Forecast
# ==========================================================

def forecast_track(
    sid: str,
    issue_time=None,
    model_name: str = DEFAULT_MODEL,
    horizons: list[int] = None,
    frame: pd.DataFrame = None,
    split_name: str = "test",
) -> pd.DataFrame:
    """
    Forecast a full track for one storm from a single issue time.

    The direct strategy: each horizon has its own model, all predicting from
    the same starting state. This is the artifact worth putting in a README --
    a forecast track laid over the observed one says more about a model in one
    image than any table of aggregate errors.

    Parameters
    ----------
    sid : str
        Storm identifier.
    issue_time : str or Timestamp, optional
        Forecast issue time. Defaults to the earliest available for the storm,
        which gives the longest verifiable track.

    Returns
    -------
    pd.DataFrame
        One row per lead time: forecast position, observed position, error.
    """

    horizons = sorted(horizons or FORECAST_HORIZON_HOURS)
    frame = frame if frame is not None else load_feature_dataset()

    rows = []
    models = None

    for horizon in horizons:
        data = prepare_data(forecast_horizon=horizon, df=frame)
        split = getattr(data, split_name)

        if split is None or len(split) == 0:
            continue

        if models is None:
            models = load_horizon_models(
                model_name, horizons, feature_names=list(split.X.columns)
            )

        if horizon not in models:
            continue

        mask = split.meta["SID"].to_numpy() == sid

        if not mask.any():
            continue

        candidates = split.meta[mask]

        if issue_time is None:
            chosen_time = candidates["ISO_TIME"].min()
        else:
            chosen_time = pd.Timestamp(issue_time)

        row_mask = mask & (split.meta["ISO_TIME"].to_numpy() == np.datetime64(chosen_time))

        if not row_mask.any():
            continue

        index = np.flatnonzero(row_mask)[:1]

        single = Split(
            "single",
            split.X.iloc[index],
            split.y.iloc[index],
            split.meta.iloc[index],
        )

        result = predict_split(models[horizon], single)
        result.insert(2, "LEAD_HOURS", horizon)
        rows.append(result)

    if not rows:
        raise ValueError(
            f"No forecastable rows for storm {sid!r} in the {split_name} split. "
            "Check the SID, or try --list-storms."
        )

    track = pd.concat(rows, ignore_index=True).sort_values("LEAD_HOURS")
    return track.reset_index(drop=True)


def log_track(track: pd.DataFrame) -> None:
    """
    Print a forecast track alongside the verifying positions.
    """

    sid = track["SID"].iloc[0]
    issue = pd.Timestamp(track["ISO_TIME"].iloc[0])

    logger.info("=" * 76)
    logger.info(f"FORECAST TRACK — storm {sid}")
    logger.info("=" * 76)
    logger.info(f"  Issued at    : {issue:%Y-%m-%d %H:%M} UTC")
    logger.info(
        f"  Start position: {track['CURRENT_LAT'].iloc[0]:.2f}N, "
        f"{track['CURRENT_LON'].iloc[0]:.2f}E"
    )
    logger.info("-" * 76)
    logger.info(
        f"  {'Lead':>6} {'Forecast':>18} {'Observed':>18} {'Error':>9} "
        f"{'Cross':>8} {'Along':>8}"
    )
    logger.info("  " + "-" * 70)

    for _, row in track.iterrows():
        logger.info(
            f"  {int(row['LEAD_HOURS']):>4}h "
            f"{row['PRED_LAT']:>9.2f},{row['PRED_LON']:>8.2f} "
            f"{row['TRUE_LAT']:>9.2f},{row['TRUE_LON']:>8.2f} "
            f"{row['ERROR_KM']:>8.1f}km "
            f"{row['CROSS_TRACK_KM']:>+7.1f} {row['ALONG_TRACK_KM']:>+7.1f}"
        )

    logger.info("-" * 76)
    logger.info(
        f"  Mean error across lead times: {track['ERROR_KM'].mean():.1f} km"
    )
    logger.info("=" * 76)


# ==========================================================
# Recursive Rollout
# ==========================================================

def forecast_recursive(
    sid: str,
    steps: int = 12,
    model_name: str = DEFAULT_MODEL,
    frame: pd.DataFrame = None,
    split_name: str = "test",
) -> pd.DataFrame:
    """
    Step a single short-horizon model forward repeatedly.

    Included for comparison against the direct strategy, with an honest
    limitation: only the position-derived features are updated between steps.
    Wind, pressure, and everything derived from them are held at their issue-time
    values, because this project does not forecast intensity. Errors therefore
    reflect both the compounding of position error and the staleness of the
    intensity state, and the two are not separable here.

    The direct strategy is the one to report. This exists so the trade-off can
    be described from measurement rather than assertion.
    """

    frame = frame if frame is not None else load_feature_dataset()

    data = prepare_data(forecast_horizon=OBSERVATION_INTERVAL, df=frame)
    split = getattr(data, split_name)

    models = load_horizon_models(
        model_name, [OBSERVATION_INTERVAL], feature_names=list(split.X.columns)
    )
    model = models[OBSERVATION_INTERVAL]

    mask = split.meta["SID"].to_numpy() == sid

    if not mask.any():
        raise ValueError(f"Storm {sid!r} not found in the {split_name} split.")

    start = np.flatnonzero(mask)[0]

    features = split.X.iloc[[start]].copy()
    lat = float(split.meta["LAT"].iloc[start])
    lon = float(split.meta["LON"].iloc[start])
    issue_time = pd.Timestamp(split.meta["ISO_TIME"].iloc[start])

    rows = []

    for step in range(1, steps + 1):
        pred_lat, pred_lon = predict_positions(model, features, [lat], [lon])
        new_lat, new_lon = float(pred_lat[0]), float(pred_lon[0])

        rows.append({
            "SID": sid,
            "LEAD_HOURS": step * OBSERVATION_INTERVAL,
            "VALID_TIME": issue_time + pd.Timedelta(hours=step * OBSERVATION_INTERVAL),
            "PRED_LAT": new_lat,
            "PRED_LON": new_lon,
        })

        features = _advance_features(features, lat, lon, new_lat, new_lon)
        lat, lon = new_lat, new_lon

    forecast = pd.DataFrame(rows)

    # Verify against observed positions where they exist.
    observed = frame.loc[frame["SID"] == sid, ["ISO_TIME", "LAT", "LON"]].rename(
        columns={"ISO_TIME": "VALID_TIME", "LAT": "TRUE_LAT", "LON": "TRUE_LON"}
    )

    forecast = forecast.merge(observed, on="VALID_TIME", how="left")
    forecast["ERROR_KM"] = haversine_distance(
        forecast["TRUE_LAT"], forecast["TRUE_LON"],
        forecast["PRED_LAT"], forecast["PRED_LON"],
    )

    return forecast


def _advance_features(
    features: pd.DataFrame,
    old_lat: float,
    old_lon: float,
    new_lat: float,
    new_lon: float,
) -> pd.DataFrame:
    """
    Roll position-derived features forward one step.

    Only what can be derived from position is updated. Intensity features stay
    frozen -- see forecast_recursive.
    """

    updated = features.copy()

    def shift_lag(prefix: str, current):
        for lag in (3, 2):
            source = f"{prefix}_LAG_{lag - 1}"
            target = f"{prefix}_LAG_{lag}"
            if target in updated.columns and source in updated.columns:
                updated[target] = updated[source].to_numpy()
        if f"{prefix}_LAG_1" in updated.columns:
            updated[f"{prefix}_LAG_1"] = current

    shift_lag("LAT", old_lat)
    shift_lag("LON", old_lon)

    delta_lat = new_lat - old_lat
    delta_lon = ((new_lon - old_lon + 180.0) % 360.0) - 180.0

    for column, value in (
        ("LAT", new_lat), ("LON", new_lon),
        ("DELTA_LAT", delta_lat), ("DELTA_LON", delta_lon),
        ("ABS_LAT", abs(new_lat)),
    ):
        if column in updated.columns:
            updated[column] = value

    if "STORM_AGE" in updated.columns:
        updated["STORM_AGE"] = updated["STORM_AGE"] + OBSERVATION_INTERVAL

    return updated


# ==========================================================
# Batch Export
# ==========================================================

def export_predictions(
    model_name: str = DEFAULT_MODEL,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    split_name: str = "test",
    path=PREDICTION_PATH,
) -> pd.DataFrame:
    """
    Write per-row predictions for the whole split.

    The output feeds the plotting scripts and makes error analysis possible
    without re-running inference.
    """

    data = prepare_data(forecast_horizon=horizon)
    split = getattr(data, split_name)

    models = load_horizon_models(
        model_name, [horizon], feature_names=list(split.X.columns)
    )

    predictions = predict_split(models[horizon], split)
    predictions.insert(2, "LEAD_HOURS", horizon)

    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)

    logger.info(f"Wrote {len(predictions):,} predictions to {path}")
    logger.info(f"  Mean error: {predictions['ERROR_KM'].mean():.1f} km")

    return predictions


def list_available_storms(
    horizon: int = DEFAULT_FORECAST_HORIZON,
    split_name: str = "test",
    limit: int = 20,
) -> pd.DataFrame:
    """
    Storms available to forecast, longest first.
    """

    data = prepare_data(forecast_horizon=horizon)
    split = getattr(data, split_name)

    summary = (
        split.meta.groupby("SID")
        .agg(
            observations=("ISO_TIME", "size"),
            first_time=("ISO_TIME", "min"),
            basin=("BASIN", "first"),
        )
        .sort_values("observations", ascending=False)
        .head(limit)
        .reset_index()
    )

    logger.info(f"Storms available in the {split_name} split ({horizon}h):")
    logger.info("\n" + summary.to_string(index=False))

    return summary


# ==========================================================
# Main
# ==========================================================

def main(
    sid: str = None,
    issue_time: str = None,
    model_name: str = DEFAULT_MODEL,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    samples: int = 10,
    recursive: bool = False,
    export: bool = False,
    list_storms: bool = False,
) -> pd.DataFrame:

    with Timer("Prediction"):
        if list_storms:
            return list_available_storms(horizon=horizon)

        if export:
            return export_predictions(model_name, horizon)

        if sid and recursive:
            forecast = forecast_recursive(sid, model_name=model_name)
            logger.info("\n" + forecast.to_string(index=False))
            logger.info(
                "  Recursive rollout holds intensity constant — see the "
                "docstring before quoting these numbers."
            )
            return forecast

        if sid:
            track = forecast_track(sid, issue_time, model_name=model_name)
            log_track(track)
            return track

        # Default: a sample of individual forecasts at one horizon.
        data = prepare_data(forecast_horizon=horizon)
        models = load_horizon_models(
            model_name, [horizon], feature_names=list(data.test.X.columns)
        )

        predictions = predict_split(models[horizon], data.test, n_samples=samples)

        logger.info(f"Sample forecasts at {horizon}h ({model_name}):")
        logger.info("\n" + predictions[
            ["SID", "CURRENT_LAT", "CURRENT_LON",
             "PRED_LAT", "PRED_LON", "TRUE_LAT", "TRUE_LON", "ERROR_KM"]
        ].round(2).to_string(index=False))
        logger.info(f"  Mean error over these {len(predictions)}: "
                    f"{predictions['ERROR_KM'].mean():.1f} km")

        return predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cyclone track inference.")
    parser.add_argument("--sid", default=None, help="Forecast a full track for this storm.")
    parser.add_argument("--issue-time", default=None, help="Issue time (default: earliest).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--horizon", type=int, default=DEFAULT_FORECAST_HORIZON)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--recursive", action="store_true",
                        help="Recursive rollout instead of direct (intensity frozen).")
    parser.add_argument("--export", action="store_true",
                        help="Write predictions for the whole split to CSV.")
    parser.add_argument("--list-storms", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        sid=args.sid,
        issue_time=args.issue_time,
        model_name=args.model,
        horizon=args.horizon,
        samples=args.samples,
        recursive=args.recursive,
        export=args.export,
        list_storms=args.list_storms,
    )