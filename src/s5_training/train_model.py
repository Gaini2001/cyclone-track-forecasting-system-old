"""
train_model.py

Unified training entry point for cyclone track forecasting.

Replaces train_random_forest.py and train_xgboost.py, which were the same
eighty lines with the estimator swapped. A registry keeps model-specific
details in one place each, so training every model at every horizon is a loop
rather than a copy.

What this adds over the originals
---------------------------------
* Per-horizon, per-mode artifact paths. The originals wrote every horizon to
  the 6h filename, so training five horizons produced five training runs and
  one mislabelled model.
* Early stopping against the validation split. n_estimators was previously
  fixed at 300 for every horizon; a 72h model needs different capacity from a
  6h one, and the validation set was built and then never used.
* A metadata sidecar recorded with each model: feature names and order,
  horizon, target mode, split fingerprint, hyperparameters, and fit time.
  A model file with no record of the feature order it expects is a silent
  wrong-answer generator the first time the feature set changes.

Usage
-----
    python -m src.s5_training.train_model --model xgboost --horizon 24
    python -m src.s5_training.train_model --model all --horizon all
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils import get_logger, Timer
from src.utils.config import (
    DEFAULT_FORECAST_HORIZON,
    FORECAST_HORIZON_HOURS,
    TARGET_MODE,
    RANDOM_STATE,
    get_model_params,
    model_path,
    split_fingerprint,
)
from src.s5_training.train import prepare_data, PreparedData, load_feature_dataset
from src.s5_training.save_model import save_model, capture_environment

logger = get_logger(__name__)

# Rounds without validation improvement before stopping.
EARLY_STOPPING_ROUNDS = 30


# ==========================================================
# Builders
# ==========================================================

def build_random_forest(params: dict, **_) -> RandomForestRegressor:
    """
    Random Forest regressor.

    Natively multi-output: one forest predicts both target components, and the
    trees can exploit the correlation between them.
    """

    return RandomForestRegressor(**params)


def build_xgboost(params: dict, early_stopping: bool = False, **_):
    """
    XGBoost regressor.

    Uses XGBRegressor directly on a 2-D target rather than wrapping it in
    MultiOutputRegressor. The wrapper forwards identical fit_params to every
    sub-estimator, so an eval_set built from a two-column y is rejected --
    which is why the original could not do early stopping. XGBoost has handled
    multi-output targets natively since 1.6.

    Note also that MultiOutputRegressor(n_jobs=-1) wrapping XGBRegressor(
    n_jobs=-1) oversubscribes the CPU: the two layers compete for the same
    cores and the fit runs slower than either setting alone.
    """

    from xgboost import XGBRegressor

    params = dict(params)

    if early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS

    return XGBRegressor(**params)


MODEL_REGISTRY = {
    "random_forest": {
        "builder": build_random_forest,
        "supports_early_stopping": False,
        "label": "Random Forest",
    },
    "xgboost": {
        "builder": build_xgboost,
        "supports_early_stopping": True,
        "label": "XGBoost",
    },
}


# ==========================================================
# Fit
# ==========================================================

def fit_model(
    model_name: str,
    data: PreparedData,
    params: dict = None,
    use_early_stopping: bool = True,
):
    """
    Fit one model on the prepared training split.

    Returns
    -------
    (model, info) : tuple
        The fitted estimator and a dict of fit diagnostics.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {model_name!r}. Options: {list(MODEL_REGISTRY)}"
        )

    spec = MODEL_REGISTRY[model_name]
    params = params if params is not None else get_model_params(model_name, data.horizon)

    early_stopping = (
        use_early_stopping
        and spec["supports_early_stopping"]
        and data.val is not None
        and len(data.val) > 0
    )

    logger.info("=" * 72)
    logger.info(f"{spec['label'].upper()} — {data.horizon}h horizon")
    logger.info("=" * 72)
    logger.info(f"  Training rows  : {len(data.train):,}")
    logger.info(f"  Features       : {data.train.X.shape[1]}")
    logger.info(f"  Targets        : {data.target_names}")
    logger.info(f"  Early stopping : {early_stopping}")

    model = spec["builder"](params, early_stopping=early_stopping)

    fit_kwargs = {}

    if early_stopping:
        fit_kwargs["eval_set"] = [(data.val.X, data.val.y)]
        fit_kwargs["verbose"] = False

    with Timer(f"{spec['label']} fit ({data.horizon}h)") as timer:
        model.fit(data.train.X, data.train.y, **fit_kwargs)

    info = {
        "model": model_name,
        "horizon": data.horizon,
        "target_mode": data.target_mode,
        "n_train_rows": int(len(data.train)),
        "n_val_rows": int(len(data.val)) if data.val is not None else 0,
        "n_features": int(data.train.X.shape[1]),
        "feature_names": list(data.train.X.columns),
        "target_names": list(data.train.y.columns),
        "params": {k: v for k, v in params.items()},
        "split": data.config,
        "split_fingerprint": split_fingerprint(data.config),
        "fit_seconds": round(timer.elapsed_sec, 2),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Record where early stopping landed: if it stops far short of
    # n_estimators the model was over-provisioned, and if it never stops it
    # was under-provisioned.
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is not None:
        info["best_iteration"] = int(best_iteration)
        logger.info(
            f"  Early stopping at iteration {best_iteration} "
            f"of {params.get('n_estimators')}"
        )

    return model, info


# ==========================================================
# Quick Diagnostics
# ==========================================================

def training_diagnostics(model, data: PreparedData) -> dict:
    """
    Cheap in-sample vs out-of-sample check on the raw target scale.

    Not the real evaluation -- that lives in evaluate.py and works in
    kilometres on reconstructed positions. This exists to catch gross
    overfitting immediately, before spending time on a full evaluation run.
    """

    diagnostics = {}

    for split in (data.train, data.val, data.test):
        if split is None or len(split) == 0:
            continue

        predictions = np.asarray(model.predict(split.X), dtype=float)
        actual = split.y.to_numpy(dtype=float)

        rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
        diagnostics[f"{split.name}_rmse_deg"] = round(rmse, 5)

    train_rmse = diagnostics.get("train_rmse_deg")
    test_rmse = diagnostics.get("test_rmse_deg")

    logger.info("  Target-scale RMSE (degrees):")

    for key, value in diagnostics.items():
        logger.info(f"    {key:<20}: {value}")

    if train_rmse and test_rmse and train_rmse > 0:
        ratio = test_rmse / train_rmse
        logger.info(f"    test/train ratio    : {ratio:.2f}")

        if ratio > 2.0:
            logger.warning(
                "    Test error is more than double train error — the model is "
                "memorising the training storms."
            )

    return diagnostics


# ==========================================================
# Persistence
# ==========================================================

def save_with_metadata(model, info: dict) -> None:
    """
    Save the estimator and a JSON sidecar describing how it was produced.
    """

    path = model_path(info["model"], info["horizon"], info["target_mode"])

    # save_model writes the sidecar, merging in the library versions that
    # produced the artifact.
    info = {**info, "environment": capture_environment()}
    save_model(model, path, metadata=info)

    logger.info(f"  Metadata saved : {path.with_suffix('.json')}")


# ==========================================================
# Orchestration
# ==========================================================

def train_one(
    model_name: str,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    save: bool = True,
    data: PreparedData = None,
    params: dict = None,
    use_early_stopping: bool = True,
) -> tuple:
    """
    Train a single model at a single horizon.

    Returns
    -------
    (model, data, info)
        The prepared data is returned so callers can evaluate without
        re-loading and re-splitting the dataset.
    """

    if data is None:
        data = prepare_data(forecast_horizon=forecast_horizon)

    model, info = fit_model(
        model_name, data, params=params, use_early_stopping=use_early_stopping,
    )

    info["diagnostics"] = training_diagnostics(model, data)

    if save:
        save_with_metadata(model, info)

    return model, data, info


def train_all(
    models: list[str] = None,
    horizons: list[int] = None,
    save: bool = True,
) -> list[dict]:
    """
    Train every requested model at every requested horizon.

    The feature dataset is loaded once and reused across horizons; the original
    trainers each re-read and re-split the full dataset per call.
    """

    models = models or list(MODEL_REGISTRY)
    horizons = horizons or [DEFAULT_FORECAST_HORIZON]

    frame = load_feature_dataset()
    results = []

    for horizon in horizons:
        data = prepare_data(forecast_horizon=horizon, df=frame)

        for model_name in models:
            _, _, info = train_one(
                model_name, horizon, save=save, data=data,
            )
            results.append(info)

    _log_overview(results)
    return results


def _log_overview(results: list[dict]) -> None:
    """
    Compact table of everything trained in this run.
    """

    if not results:
        return

    logger.info("=" * 72)
    logger.info("TRAINING OVERVIEW")
    logger.info("=" * 72)
    logger.info(
        f"  {'Model':<16} {'Horizon':>8} {'Rows':>10} {'Fit':>10} "
        f"{'Train RMSE':>12} {'Test RMSE':>12}"
    )
    logger.info("  " + "-" * 68)

    for info in results:
        diagnostics = info.get("diagnostics", {})
        logger.info(
            f"  {info['model']:<16} {str(info['horizon']) + 'h':>8} "
            f"{info['n_train_rows']:>10,} {info['fit_seconds']:>9.1f}s "
            f"{diagnostics.get('train_rmse_deg', float('nan')):>12.4f} "
            f"{diagnostics.get('test_rmse_deg', float('nan')):>12.4f}"
        )

    logger.info("=" * 72)
    logger.info(
        "  RMSE is in degrees on the raw target — a sanity check only. "
        "Track error in km comes from evaluate.py."
    )


def main(
    model: str = "all",
    horizon: str | int = DEFAULT_FORECAST_HORIZON,
    save: bool = True,
) -> list[dict]:
    """
    CLI entry point.
    """

    models = list(MODEL_REGISTRY) if model == "all" else [model]
    horizons = (
        list(FORECAST_HORIZON_HOURS) if horizon == "all" else [int(horizon)]
    )

    return train_all(models=models, horizons=horizons, save=save)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train track forecasting models.")
    parser.add_argument(
        "--model", default="all",
        choices=list(MODEL_REGISTRY) + ["all"],
    )
    parser.add_argument(
        "--horizon", default=str(DEFAULT_FORECAST_HORIZON),
        help="Horizon in hours, or 'all'.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Train without writing artifacts.",
    )
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Use the configured n_estimators rather than stopping on validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(model=args.model, horizon=args.horizon, save=not args.no_save)