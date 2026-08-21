"""
tune_hyperparameters.py

Hyperparameter search for cyclone track forecasting, using Optuna.

What the original got wrong, and why it matters
-----------------------------------------------
* Six to eight trials over a seven-dimensional space is not Bayesian
  optimisation. Optuna's TPE sampler samples randomly until it has
  `n_startup_trials` observations (10 by default) to fit a surrogate on, so a
  study that short never engages TPE at all -- it is random search wearing a
  different name. The defaults here are set so the sampler actually runs.
* The sampler was unseeded, so the study returned a different answer each run.
* The study was not persisted, so there was no history, no resume, and no
  importance analysis.
* Results were hand-copied into config.py, losing all provenance: which
  horizon, how many trials, when, against which split.
* The objective computed great-circle distance directly on the target columns.
  Under displacement targets those are (dlat, dlon) vectors, not coordinates,
  so the "kilometres" being minimised were meaningless. Predictions are
  reconstructed against the issue-time position here.

Best parameters are written to TUNED_PARAMS_PATH, which `config.get_model_params`
reads automatically. Nothing needs to be transcribed by hand.

Usage
-----
    python -m src.s5_training.tune_hyperparameters --model xgboost --horizon 24 --trials 60
    python -m src.s5_training.tune_hyperparameters --model all --horizon all --trials 40
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.utils import get_logger, Timer
from src.utils.metrics import haversine_distance, offset_position
from src.utils.config import (
    RANDOM_STATE,
    TARGET_MODE,
    DEFAULT_FORECAST_HORIZON,
    FORECAST_HORIZON_HOURS,
    TUNED_PARAMS_PATH,
    OPTUNA_STUDY_PATH,
    split_fingerprint,
)
from src.s5_training.train import prepare_data, PreparedData, load_feature_dataset, Split

logger = get_logger(__name__)

# TPE needs this many observations before its surrogate model does anything.
# A study shorter than roughly twice this is random search.
N_STARTUP_TRIALS = 10

# Fixed high ceiling; early stopping selects the actual count per trial, so
# n_estimators is not part of the search space.
XGB_MAX_ESTIMATORS = 2000
XGB_EARLY_STOPPING_ROUNDS = 30


# ==========================================================
# Objective Metric
# ==========================================================

def track_error_km(predictions: np.ndarray, split: Split) -> float:
    """
    Mean great-circle track error in kilometres.

    Predictions arrive on whatever scale TARGET_MODE dictates. Under
    displacement they are offsets from the issue-time position and must be
    added back to it before any distance is meaningful; measuring the distance
    between two displacement vectors as though they were coordinates produces a
    number that is minimisable but means nothing.
    """

    predictions = np.asarray(predictions, dtype=float)
    actual = split.y.to_numpy(dtype=float)

    if TARGET_MODE == "displacement":
        base_lat = split.meta["LAT"].to_numpy(dtype=float)
        base_lon = split.meta["LON"].to_numpy(dtype=float)

        pred_lat, pred_lon = offset_position(
            base_lat, base_lon, predictions[:, 0], predictions[:, 1]
        )
        true_lat, true_lon = offset_position(
            base_lat, base_lon, actual[:, 0], actual[:, 1]
        )
    else:
        pred_lat, pred_lon = predictions[:, 0], predictions[:, 1]
        true_lat, true_lon = actual[:, 0], actual[:, 1]

    errors = haversine_distance(true_lat, true_lon, pred_lat, pred_lon)

    return float(np.mean(errors[np.isfinite(errors)]))


# ==========================================================
# Search Spaces
# ==========================================================

def suggest_random_forest(trial) -> dict:
    """
    Random Forest search space.

    max_depth reaches down to 6. The original space started at 15, so it could
    only explore deep trees -- and at long horizons the useful direction is
    shallower, since the test/train error ratio grows with lead time.
    """

    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 40, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20, log=True),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 1.0]
        ),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }


def suggest_xgboost(trial) -> dict:
    """
    XGBoost search space.

    n_estimators is fixed at a high ceiling rather than tuned: early stopping
    selects the count for each trial, so tuning it as well wastes the budget on
    a parameter that gets overridden anyway.
    """

    return {
        "n_estimators": XGB_MAX_ESTIMATORS,
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }


SEARCH_SPACES = {
    "random_forest": suggest_random_forest,
    "xgboost": suggest_xgboost,
}


# ==========================================================
# Model Construction
# ==========================================================

def _fit_candidate(model_name: str, params: dict, data: PreparedData):
    """
    Fit one candidate configuration on the training split.
    """

    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(**params)
        model.fit(data.train.X, data.train.y)
        return model

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        params = dict(params)
        params["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS

        model = XGBRegressor(**params)
        model.fit(
            data.train.X, data.train.y,
            eval_set=[(data.val.X, data.val.y)],
            verbose=False,
        )
        return model

    raise ValueError(f"Unknown model: {model_name!r}")


# ==========================================================
# Study
# ==========================================================

def build_objective(model_name: str, data: PreparedData, subsample: float = 1.0):
    """
    Objective returning validation track error in kilometres.
    """

    suggest = SEARCH_SPACES[model_name]

    tuning_data = data

    if subsample < 1.0:
        # Tuning on a fraction of the training rows trades a little fidelity
        # for many more trials, which is usually the better bargain when the
        # budget is fixed.
        rng = np.random.default_rng(RANDOM_STATE)
        keep = rng.random(len(data.train)) < subsample

        tuning_data = PreparedData(
            train=Split(
                "train",
                data.train.X[keep].reset_index(drop=True),
                data.train.y[keep].reset_index(drop=True),
                data.train.meta[keep].reset_index(drop=True),
            ),
            val=data.val,
            test=data.test,
            horizon=data.horizon,
            target_mode=data.target_mode,
            config=data.config,
        )

        logger.info(
            f"  Tuning on {len(tuning_data.train):,} of {len(data.train):,} rows "
            f"({subsample:.0%})"
        )

    def objective(trial) -> float:
        params = suggest(trial)
        model = _fit_candidate(model_name, params, tuning_data)

        predictions = model.predict(data.val.X)
        error = track_error_km(predictions, data.val)

        best_iteration = getattr(model, "best_iteration", None)

        if best_iteration is not None:
            trial.set_user_attr("best_iteration", int(best_iteration))

        return error

    return objective


def tune(
    model_name: str,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    n_trials: int = 50,
    timeout: int = None,
    subsample: float = 1.0,
    data: PreparedData = None,
    persist: bool = True,
) -> dict:
    """
    Run a study for one model at one horizon.

    Returns
    -------
    dict
        Best parameters, best validation error, and study provenance.
    """

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for tuning: pip install optuna"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if data is None:
        data = prepare_data(forecast_horizon=forecast_horizon)

    if data.val is None or len(data.val) == 0:
        raise ValueError(
            "Tuning requires a validation split. Check the season boundaries "
            "in config (TRAIN_END_SEASON / VAL_END_SEASON)."
        )

    if n_trials < 2 * N_STARTUP_TRIALS:
        logger.warning(
            f"  {n_trials} trials with n_startup_trials={N_STARTUP_TRIALS}: the "
            "sampler spends most of the budget on random exploration. "
            f"Use at least {2 * N_STARTUP_TRIALS} for the surrogate to matter."
        )

    fingerprint = split_fingerprint(data.config)
    study_name = f"{model_name}_{forecast_horizon}h_{TARGET_MODE}_{fingerprint}"

    logger.info("=" * 72)
    logger.info(f"TUNING {model_name.upper()} — {forecast_horizon}h horizon")
    logger.info("=" * 72)
    logger.info(f"  Study        : {study_name}")
    logger.info(f"  Train / val  : {len(data.train):,} / {len(data.val):,} rows")
    logger.info(f"  Trials       : {n_trials}")
    logger.info(f"  Objective    : mean validation track error (km)")

    storage = None

    if persist:
        # SQLite storage makes a study resumable and keeps the full trial
        # history for later analysis.
        OPTUNA_STUDY_PATH.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{OPTUNA_STUDY_PATH}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        # Seeded: an unseeded sampler makes the whole search unreproducible.
        sampler=TPESampler(seed=RANDOM_STATE, n_startup_trials=N_STARTUP_TRIALS),
        storage=storage,
        load_if_exists=True,
    )

    completed = len(study.trials)

    if completed:
        logger.info(f"  Resuming from {completed} existing trial(s)")

    objective = build_objective(model_name, data, subsample=subsample)

    with Timer(f"Optuna study ({model_name}, {forecast_horizon}h)") as timer:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )

    result = {
        "model": model_name,
        "horizon": forecast_horizon,
        "target_mode": TARGET_MODE,
        "split_fingerprint": fingerprint,
        "params": dict(study.best_params),
        "best_val_error_km": round(float(study.best_value), 2),
        "n_trials": len(study.trials),
        "study_name": study_name,
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(timer.elapsed_sec, 1),
    }

    best_iteration = study.best_trial.user_attrs.get("best_iteration")

    if best_iteration is not None:
        # Early stopping chose this; record it so the trained model does not
        # have to rediscover it.
        result["params"]["n_estimators"] = int(best_iteration) + 1
        result["best_iteration"] = int(best_iteration)

    _log_result(result, study)

    if persist:
        save_tuned_params(result)

    return result


def _log_result(result: dict, study) -> None:
    """
    Report the outcome, plus parameter importances where available.
    """

    logger.info("=" * 72)
    logger.info(f"BEST — {result['model']} @ {result['horizon']}h")
    logger.info("=" * 72)
    logger.info(f"  Validation track error : {result['best_val_error_km']:.2f} km")
    logger.info(f"  Trials completed       : {result['n_trials']}")
    logger.info(f"  Duration               : {result['duration_seconds']}s")
    logger.info("  Parameters:")

    for key, value in sorted(result["params"].items()):
        logger.info(f"    {key:<20}: {value}")

    try:
        import optuna

        if len(study.trials) >= N_STARTUP_TRIALS:
            importances = optuna.importance.get_param_importances(study)

            logger.info("  Parameter importance:")

            for key, value in list(importances.items())[:5]:
                logger.info(f"    {key:<20}: {value:.3f}")
    except Exception:
        pass  # importance is a nicety, never a reason to fail a study

    logger.info("=" * 72)


# ==========================================================
# Persistence
# ==========================================================

def save_tuned_params(result: dict) -> None:
    """
    Merge one result into the tuned-parameter file.

    `config.get_model_params` reads this, so a completed study takes effect on
    the next training run with nothing copied by hand.
    """

    store = {}

    if TUNED_PARAMS_PATH.exists():
        try:
            store = json.loads(TUNED_PARAMS_PATH.read_text())
        except json.JSONDecodeError:
            logger.warning("Existing tuned-params file is corrupt; rewriting.")

    store.setdefault(result["model"], {})[str(result["horizon"])] = result

    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PARAMS_PATH.write_text(json.dumps(store, indent=2))

    logger.info(f"  Saved to {TUNED_PARAMS_PATH}")


def load_tuned_params() -> dict:
    """
    Everything tuned so far.
    """

    if not TUNED_PARAMS_PATH.exists():
        return {}

    return json.loads(TUNED_PARAMS_PATH.read_text())


def summarize_tuning() -> pd.DataFrame:
    """
    Table of every stored tuning result. Useful for the README.
    """

    store = load_tuned_params()

    rows = [
        {
            "model": model,
            "horizon": int(horizon),
            "val_error_km": entry.get("best_val_error_km"),
            "trials": entry.get("n_trials"),
            "tuned_at": entry.get("tuned_at"),
        }
        for model, horizons in store.items()
        for horizon, entry in horizons.items()
    ]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["model", "horizon"]).reset_index(drop=True)


# ==========================================================
# Orchestration
# ==========================================================

def main(
    model: str = "all",
    horizon: str | int = DEFAULT_FORECAST_HORIZON,
    n_trials: int = 50,
    timeout: int = None,
    subsample: float = 1.0,
) -> list[dict]:
    """
    Tune every requested model at every requested horizon.
    """

    models = list(SEARCH_SPACES) if model == "all" else [model]
    horizons = (
        list(FORECAST_HORIZON_HOURS) if horizon == "all" else [int(horizon)]
    )

    frame = load_feature_dataset()
    results = []

    for h in horizons:
        data = prepare_data(forecast_horizon=h, df=frame)

        for model_name in models:
            results.append(
                tune(
                    model_name, h,
                    n_trials=n_trials,
                    timeout=timeout,
                    subsample=subsample,
                    data=data,
                )
            )

    table = summarize_tuning()

    if not table.empty:
        logger.info("\n" + table.to_string(index=False))

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune model hyperparameters.")
    parser.add_argument(
        "--model", default="all", choices=list(SEARCH_SPACES) + ["all"],
    )
    parser.add_argument(
        "--horizon", default=str(DEFAULT_FORECAST_HORIZON),
        help="Horizon in hours, or 'all'.",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help=f"Trials per study. Below {2 * N_STARTUP_TRIALS} the sampler is "
             "effectively doing random search.",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Wall-clock limit per study, in seconds.",
    )
    parser.add_argument(
        "--subsample", type=float, default=1.0,
        help="Fraction of training rows per trial (speeds up the search).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model=args.model,
        horizon=args.horizon,
        n_trials=args.trials,
        timeout=args.timeout,
        subsample=args.subsample,
    )