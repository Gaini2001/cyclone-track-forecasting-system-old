"""
main.py

Pipeline entry point for the Cyclone Track Forecasting system.

Changes from the previous version
---------------------------------
* Training is a stage. The old script called `evaluate()` and relied on it to
  train the models as a side effect, which meant every evaluation retrained
  from scratch and overwrote the artifacts it was measuring.
* `ensure_directories()` is called here. config.py no longer creates
  directories at import time -- importing a configuration module should not
  touch the filesystem.
* Stages are selectable. A full run downloads several hundred megabytes and
  fits a model per horizon; being able to re-run just evaluation, or skip
  tuning, matters when iterating.
* Failures stop the run and say which stage failed, rather than surfacing three
  stages later as a confusing KeyError.

Usage
-----
    python main.py                     # clean -> features -> validate ->
                                       # train -> evaluate -> figures -> maps
    python main.py --stages maps
    python main.py --with-tuning --trials 50
    python main.py --with-importance --ablation
    python main.py --map-horizons 6 12 24 48 72
"""

from __future__ import annotations

import argparse
import sys
import time

from src.utils.config import (
    ensure_directories,
    FORECAST_HORIZON_HOURS,
    DEFAULT_FORECAST_HORIZON,
)
from src.utils.logger import configure_logging, get_logger, log_environment
from src.utils.timer import log_timing_summary

logger = get_logger(__name__)

ALL_STAGES = (
    "clean", "features", "validate", "tune", "train", "evaluate",
    "figures", "maps", "importance",
)

# Tuning and the ablation study are expensive and optional; everything else is
# the default path.
DEFAULT_STAGES = (
    "clean", "features", "validate", "train", "evaluate", "figures", "maps",
)


# ==========================================================
# Stages
# ==========================================================

def stage_clean(args) -> None:
    from src.s2_preprocessing.clean_data import main as clean_data

    clean_data()


def stage_features(args) -> None:
    from src.s3_features.feature_engineering import main as build_features

    build_features()


def stage_validate(args) -> None:
    from src.s4_validation.validate_features import main as validate

    # Strict by default: a dataset that fails validation should stop the run,
    # not train a model on itself.
    validate(strict=not args.lenient)


def stage_tune(args) -> None:
    from src.s5_training.tune_hyperparameters import main as tune

    tune(model="all", horizon=args.horizon, n_trials=args.trials)


def stage_train(args) -> None:
    from src.s5_training.train_model import main as train

    train(model="all", horizon=args.horizon)


def stage_evaluate(args) -> None:
    from src.s5_training.evaluate import main as evaluate

    table, _ = evaluate(horizon=args.horizon, make_plots=False)

    logger.info("\n" + table.to_string(index=False))


def stage_figures(args) -> None:
    """
    Dataset and evaluation figures.

    Each generator is called independently: one failing plot should not cost
    you the other seventeen at the end of a long run.
    """

    from src.s7_analysis.eda import main as eda
    from src.s7_analysis.evaluation_plots import main as evaluation_plots

    for name, task in (("eda", eda), ("evaluation plots", evaluation_plots)):
        try:
            task()
        except Exception as exc:
            logger.warning(f"  {name} failed: {type(exc).__name__}: {exc}")


def stage_maps(args) -> None:
    """
    Geographic figures: error maps and observed-vs-forecast tracks.

    Separate from `figures` because it runs inference over the test split for
    every mapped horizon, which is slower than plotting from stored metrics.
    """

    from src.s7_analysis.world_map import main as world_map

    world_map(
        horizons=tuple(args.map_horizons),
        model_name=args.model,
        sid=args.sid,
        n_storms=args.map_storms,
    )


def stage_importance(args) -> None:
    """
    Permutation importance, grouped importance, and the ablation study.

    Not in the default set: the ablation retrains the model once per feature
    group, which takes considerably longer than everything else combined.
    """

    from src.s7_analysis.feature_analysis import main as feature_analysis

    horizon = (
        DEFAULT_FORECAST_HORIZON if args.horizon == "all" else int(args.horizon)
    )

    feature_analysis(
        model_name=args.model,
        horizon=horizon,
        run_ablation=args.ablation,
    )


STAGES = {
    "clean": stage_clean,
    "features": stage_features,
    "validate": stage_validate,
    "tune": stage_tune,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "figures": stage_figures,
    "maps": stage_maps,
    "importance": stage_importance,
}


# ==========================================================
# Orchestration
# ==========================================================

def run(stages: list[str], args) -> int:
    """
    Execute the requested stages in order.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on the first stage failure.
    """

    ensure_directories()

    logger.info("=" * 72)
    logger.info("CYCLONE TRACK FORECASTING PIPELINE")
    logger.info("=" * 72)
    logger.info(f"  Stages  : {' -> '.join(stages)}")
    logger.info(f"  Horizon : {args.horizon}")

    log_environment(logger)

    started = time.perf_counter()

    for index, name in enumerate(stages, start=1):
        logger.info("")
        logger.info("#" * 72)
        logger.info(f"# STAGE {index}/{len(stages)}: {name.upper()}")
        logger.info("#" * 72)

        try:
            STAGES[name](args)
        except Exception as exc:
            logger.exception(f"Stage {name!r} failed")
            logger.error("")
            logger.error(f"Pipeline stopped at stage {index}/{len(stages)}: {name}")
            logger.error(f"  {type(exc).__name__}: {exc}")

            # Naming the resume point saves re-running the expensive stages
            # that already succeeded.
            remaining = stages[index - 1:]
            logger.error(f"  Resume with: python main.py --stages {' '.join(remaining)}")

            return 1

    elapsed = time.perf_counter() - started

    log_timing_summary(logger)

    logger.info("=" * 72)
    logger.info(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min")
    logger.info("=" * 72)
    logger.info("  Results : reports/results.md")
    logger.info("  Figures : reports/figures/")
    logger.info("  Models  : models/")
    logger.info("  Preds   : data/predictions/")

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cyclone track forecasting pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --stages maps\n"
            "  python main.py --stages evaluate figures maps\n"
            "  python main.py --with-tuning --trials 50\n"
            "  python main.py --with-importance --ablation\n"
            "  python main.py --map-horizons 6 12 24 48 72 --map-storms 8\n"
        ),
    )

    parser.add_argument(
        "--stages", nargs="+", choices=ALL_STAGES, default=None,
        help=f"Stages to run. Default: {' '.join(DEFAULT_STAGES)}",
    )
    parser.add_argument(
        "--with-tuning", action="store_true",
        help="Include the Optuna stage before training.",
    )
    parser.add_argument(
        "--with-importance", action="store_true",
        help="Include the feature analysis stage after evaluation.",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="In the importance stage, retrain without each feature group. "
             "Slow, but the only evidence that answers 'can I delete these?'",
    )
    parser.add_argument(
        "--model", default="xgboost", choices=("xgboost", "random_forest"),
        help="Model used for maps and feature analysis. Evaluation always "
             "covers both.",
    )
    parser.add_argument(
        "--map-horizons", type=int, nargs="+", default=[6, 12, 24],
        choices=list(FORECAST_HORIZON_HOURS),
        help="Lead times to map. One error map and one track figure each.",
    )
    parser.add_argument(
        "--map-storms", type=int, default=6,
        help="Storms sampled for the observed-vs-forecast track figures.",
    )
    parser.add_argument(
        "--sid", default=None,
        help="Storm for the single-storm all-horizons forecast map.",
    )
    parser.add_argument(
        "--horizon", default="all",
        help=f"Horizon in hours {list(FORECAST_HORIZON_HOURS)}, or 'all'. "
             f"Default: all",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Optuna trials per study. Below 20 the sampler is doing random "
             "search.",
    )
    parser.add_argument(
        "--lenient", action="store_true",
        help="Report validation failures without stopping. For iterating only "
             "— never for a run whose numbers you intend to publish.",
    )
    parser.add_argument(
        "--log-file", action="store_true",
        help="Write a timestamped log to reports/logs/.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    configure_logging(level=args.log_level, log_file=args.log_file)

    if args.stages:
        stages = list(args.stages)
    else:
        stages = list(DEFAULT_STAGES)

        if args.with_tuning:
            stages.insert(stages.index("train"), "tune")

        if args.with_importance or args.ablation:
            stages.append("importance")

    return run(stages, args)


if __name__ == "__main__":
    sys.exit(main())