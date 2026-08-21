"""
evaluate.py

Evaluation for cyclone track forecasting.

Principles
----------
* Evaluation loads saved models; it does not train them. The original called
  the trainers from inside main(), which re-loaded and re-split the dataset
  twice and then overwrote the artifacts it was meant to be measuring. Reported
  metrics could silently diverge from what was on disk.
* Models and baselines are scored by the same function on the same rows. A
  claim of the form "18% better than persistence" is only sound if both numbers
  came from identical code.
* Predictions are reconstructed to absolute positions before any distance is
  computed. Under displacement targets the raw outputs are offsets, and
  measuring great-circle distance between two offset vectors produces a number
  that looks plausible and means nothing.
* Every skill claim carries a confidence interval. With a few hundred test
  storms, an 8% improvement may well be sampling noise, and a paired bootstrap
  is what distinguishes the two.

Outputs
-------
reports/model_comparison.csv     full comparison table
reports/evaluation_results.json  detailed per-model metrics
reports/results.md               markdown tables ready to paste into a README
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils import get_logger, Timer
from src.utils.metrics import (
    track_errors,
    track_error_statistics,
    offset_position,
    paired_bootstrap_comparison,
    bootstrap_ci,
)
from src.utils.config import (
    TARGET_MODE,
    DEFAULT_FORECAST_HORIZON,
    FORECAST_HORIZON_HOURS,
    REPORT_DIR,
    EVALUATION_RESULTS_PATH,
    MODEL_COMPARISON_PATH,
    model_path,
)
from src.s5_training.train import prepare_data, load_feature_dataset, PreparedData, Split
from src.s5_training.train_model import MODEL_REGISTRY
from src.s5_training.baselines import run_all_baselines, REFERENCE_BASELINE
from src.s5_training.save_model import load_model

logger = get_logger(__name__)

RESULTS_MARKDOWN_PATH = REPORT_DIR / "results.md"

# Minimum test storms before skill estimates are worth quoting without heavy
# caveats.
MIN_TEST_STORMS = 50


# ==========================================================
# Reconstruction
# ==========================================================

def reconstruct_positions(
    predictions: np.ndarray,
    split: Split,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw model output and targets into absolute coordinates.

    Everything downstream -- track error, along/cross decomposition, plots --
    needs real positions. Under TARGET_MODE="displacement" the model emits
    offsets from the issue-time position, so both prediction and truth are
    added back onto it.

    Returns
    -------
    (pred_lat, pred_lon, true_lat, true_lon)
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

    return pred_lat, pred_lon, true_lat, true_lon


# ==========================================================
# Scoring
# ==========================================================

def evaluate_predictions(
    predictions: np.ndarray,
    split: Split,
    name: str,
    horizon: int,
    reference_errors: np.ndarray = None,
    with_ci: bool = True,
) -> dict:
    """
    Score one set of predictions.

    Used identically for models and baselines -- that is the point.

    Parameters
    ----------
    reference_errors : np.ndarray, optional
        Per-sample errors from the reference forecast. Supplying them adds a
        skill score with a paired bootstrap interval.
    """

    predictions = np.asarray(predictions, dtype=float)

    pred_lat, pred_lon, true_lat, true_lon = reconstruct_positions(predictions, split)

    errors = track_errors(true_lat, true_lon, pred_lat, pred_lon)

    stats = track_error_statistics(
        true_lat, true_lon,
        pred_lat, pred_lon,
        # Issue-time position: required for the along/cross-track split.
        current_lat=split.meta["LAT"].to_numpy(dtype=float),
        current_lon=split.meta["LON"].to_numpy(dtype=float),
    )

    actual = split.y.to_numpy(dtype=float)

    result = {
        "model": name,
        "horizon": horizon,
        "target_mode": TARGET_MODE,
        **stats,
        # On the raw target scale. Under displacement targets R^2 is
        # informative; under absolute targets it saturates near 1.0 because
        # the current position is both an input and nearly the answer.
        "mae_target_units": float(mean_absolute_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
    }

    if with_ci and stats["n_samples"] > 1:
        low, high = bootstrap_ci(errors)
        result["mean_ci_lower_km"] = round(low, 2)
        result["mean_ci_upper_km"] = round(high, 2)

    if reference_errors is not None:
        comparison = paired_bootstrap_comparison(errors, reference_errors)
        result.update({
            "skill_pct": round(comparison.get("skill_pct", float("nan")), 2),
            "skill_ci_lower_pct": round(comparison.get("skill_ci_lower_pct", float("nan")), 2),
            "skill_ci_upper_pct": round(comparison.get("skill_ci_upper_pct", float("nan")), 2),
            "skill_p_value": round(comparison.get("p_value", float("nan")), 4),
        })

    result["_errors"] = errors  # stripped before serialization
    return result


def evaluate_by_basin(
    predictions: np.ndarray,
    split: Split,
    name: str,
    horizon: int,
    min_samples: int = 200,
) -> list[dict]:
    """
    Per-basin breakdown.

    Aggregate error hides that the same model can be strong in one basin and
    weak in another. Basins differ in steering regime, in observation density,
    and in how often storms recurve, so a single global number is an average
    over genuinely different problems.
    """

    if "BASIN" not in split.meta.columns:
        return []

    pred_lat, pred_lon, true_lat, true_lon = reconstruct_positions(predictions, split)
    basins = split.meta["BASIN"].to_numpy()

    rows = []

    for basin in pd.unique(basins):
        mask = basins == basin

        if mask.sum() < min_samples:
            continue

        stats = track_error_statistics(
            true_lat[mask], true_lon[mask], pred_lat[mask], pred_lon[mask],
            current_lat=split.meta["LAT"].to_numpy(dtype=float)[mask],
            current_lon=split.meta["LON"].to_numpy(dtype=float)[mask],
        )

        rows.append({
            "model": name,
            "horizon": horizon,
            "basin": str(basin),
            "n_samples": stats["n_samples"],
            "mean_km": round(stats["mean_km"], 1),
            "median_km": round(stats["median_km"], 1),
            "cross_track_mae_km": round(stats.get("cross_track_mae_km", float("nan")), 1),
            "along_track_bias_km": round(stats.get("along_track_bias_km", float("nan")), 1),
        })

    return rows


# ==========================================================
# Model Loading
# ==========================================================

def load_trained_model(model_name: str, horizon: int, feature_names: list[str]):
    """
    Load a saved model and verify it matches the current feature set.

    The metadata sidecar records the feature order the model was fitted on. A
    silent mismatch -- one renamed or reordered column -- produces confident,
    wrong predictions rather than an error, so it is checked here.

    Returns None when the artifact is absent, so evaluation can proceed with
    whatever has been trained.
    """

    path = model_path(model_name, horizon, TARGET_MODE)

    if not path.exists():
        logger.warning(f"  {model_name} @ {horizon}h: no artifact at {path.name}")
        return None

    model = load_model(path)
    metadata_path = path.with_suffix(".json")

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        trained_on = metadata.get("feature_names", [])

        if trained_on and trained_on != list(feature_names):
            missing = set(trained_on) - set(feature_names)
            added = set(feature_names) - set(trained_on)

            raise ValueError(
                f"{model_name} @ {horizon}h was trained on a different feature "
                f"set. Missing now: {sorted(missing) or 'none'}; "
                f"new: {sorted(added) or 'none'}; order changed: "
                f"{trained_on != list(feature_names)}. Retrain before evaluating."
            )

    return model


# ==========================================================
# Horizon Evaluation
# ==========================================================

def evaluate_horizon(
    horizon: int,
    data: PreparedData = None,
    models: list[str] = None,
    split_name: str = "test",
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate every model and baseline at one horizon.

    Returns
    -------
    (results, basin_rows)
    """

    if data is None:
        data = prepare_data(forecast_horizon=horizon)

    split = getattr(data, split_name)

    if split is None or len(split) == 0:
        raise ValueError(f"Split {split_name!r} is empty at {horizon}h.")

    models = models or list(MODEL_REGISTRY)

    logger.info("=" * 76)
    logger.info(f"EVALUATION — {horizon}h horizon, {split_name} split")
    logger.info("=" * 76)
    logger.info(f"  Rows   : {len(split):,}")
    logger.info(f"  Storms : {split.storms:,}")

    if split.storms < MIN_TEST_STORMS:
        logger.warning(
            f"  Only {split.storms} test storms — treat skill differences "
            "smaller than the confidence intervals as unresolved."
        )

    predictions: dict[str, np.ndarray] = {}

    # ---- Baselines first: the reference must exist before scoring anything ----
    predictions.update(run_all_baselines(data, split_name=split_name))

    # ---- Models ----
    for model_name in models:
        model = load_trained_model(model_name, horizon, list(split.X.columns))

        if model is None:
            continue

        label = MODEL_REGISTRY[model_name]["label"]
        predictions[label] = np.asarray(model.predict(split.X), dtype=float)

    if REFERENCE_BASELINE not in predictions:
        raise RuntimeError(
            f"Reference baseline {REFERENCE_BASELINE!r} unavailable; "
            "skill scores cannot be computed."
        )

    # Reference errors, computed once through the same path as everything else.
    reference = evaluate_predictions(
        predictions[REFERENCE_BASELINE], split, REFERENCE_BASELINE, horizon,
    )
    reference_errors = reference["_errors"]

    results = []
    basin_rows = []

    for name, prediction in predictions.items():
        result = evaluate_predictions(
            prediction, split, name, horizon,
            reference_errors=None if name == REFERENCE_BASELINE else reference_errors,
        )
        results.append(result)
        basin_rows.extend(evaluate_by_basin(prediction, split, name, horizon))

    _log_horizon(results, horizon)
    return results, basin_rows


def _log_horizon(results: list[dict], horizon: int) -> None:
    """
    Ranked table for one horizon.
    """

    ordered = sorted(results, key=lambda r: r["mean_km"])

    logger.info("-" * 76)
    logger.info(
        f"  {'Forecast':<22} {'Mean km':>9} {'Median':>8} {'P90':>8} "
        f"{'Skill %':>9} {'p':>7}"
    )
    logger.info("  " + "-" * 70)

    for result in ordered:
        skill = result.get("skill_pct")
        p_value = result.get("skill_p_value")

        logger.info(
            f"  {result['model']:<22} {result['mean_km']:>9.1f} "
            f"{result['median_km']:>8.1f} {result['p90_km']:>8.1f} "
            f"{(f'{skill:+.1f}' if skill is not None else '  ref'):>9} "
            f"{(f'{p_value:.3f}' if p_value is not None else '    —'):>7}"
        )

    logger.info("-" * 76)

    best = ordered[0]

    if best.get("cross_track_mae_km") is not None:
        # The diagnostic worth reading: direction error versus speed error.
        logger.info(
            f"  {best['model']} error split — cross-track MAE "
            f"{best.get('cross_track_mae_km', float('nan')):.1f} km, "
            f"along-track bias {best.get('along_track_bias_km', float('nan')):+.1f} km"
        )
        logger.info(
            "  (positive along-track bias = forecasts run the storm too fast)"
        )


# ==========================================================
# Aggregation & Reporting
# ==========================================================

def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Flatten results into the headline comparison table.
    """

    rows = [
        {
            "Model": r["model"],
            "Horizon (h)": r["horizon"],
            "Mean Error (km)": round(r["mean_km"], 1),
            "95% CI": (
                f"{r['mean_ci_lower_km']:.0f}–{r['mean_ci_upper_km']:.0f}"
                if "mean_ci_lower_km" in r else "—"
            ),
            "Median (km)": round(r["median_km"], 1),
            "P90 (km)": round(r["p90_km"], 1),
            "Within 100km (%)": round(r.get("within_100km_pct", float("nan")), 1),
            # R^2 on the displacement target. Under absolute-coordinate
            # targets this saturated near 1.0 for every model and said
            # nothing, because the current position was both an input and
            # almost the answer. On displacements it measures what fraction
            # of the movement the model actually explains.
            "R2 (displacement)": round(r.get("r2", float("nan")), 3),
            # Both MAEs, then the bias. Showing cross-track MAE beside
            # along-track *bias* invites the wrong conclusion: bias cancels,
            # MAE does not, so a large scattered along-track error can look
            # small next to a cross-track magnitude.
            "Cross-track MAE (km)": round(r.get("cross_track_mae_km", float("nan")), 1),
            "Along-track MAE (km)": round(r.get("along_track_mae_km", float("nan")), 1),
            "Along-track bias (km)": round(r.get("along_track_bias_km", float("nan")), 1),
            "Skill vs Persistence (%)": (
                f"{r['skill_pct']:+.1f}" if r.get("skill_pct") is not None else "ref"
            ),
            "Skill 95% CI": (
                f"{r['skill_ci_lower_pct']:+.1f}–{r['skill_ci_upper_pct']:+.1f}"
                if r.get("skill_ci_lower_pct") is not None else "—"
            ),
            "p": r.get("skill_p_value", "—"),
            "n": r["n_samples"],
        }
        for r in results
    ]

    return (
        pd.DataFrame(rows)
        .sort_values(["Horizon (h)", "Mean Error (km)"])
        .reset_index(drop=True)
    )


def _to_markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """
    Render a DataFrame as a markdown table.

    pandas delegates to_markdown() to `tabulate`, an optional dependency. A
    missing formatting library should not destroy a completed evaluation, so
    this falls back to writing the table by hand.
    """

    try:
        return frame.to_markdown(index=index)
    except ImportError:
        logger.warning(
            "  tabulate not installed — writing plain tables. "
            "`pip install tabulate` for proper markdown."
        )

        columns = ([frame.index.name or ""] if index else []) + [
            str(c) for c in frame.columns
        ]

        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]

        for key, row in frame.iterrows():
            cells = ([str(key)] if index else []) + [str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)


def write_markdown_report(
    table: pd.DataFrame,
    basin_table: pd.DataFrame,
    path=RESULTS_MARKDOWN_PATH,
) -> None:
    """
    Emit markdown ready to paste into the README.

    A results table a reader can see without cloning the repo is worth more
    than any amount of code quality they will never look at.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Results",
        "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d')} — "
        f"target mode: `{TARGET_MODE}`_",
        "",
        "Track error is great-circle distance between forecast and observed "
        "position. Skill is the percentage reduction in mean error relative to "
        "persistence, with a paired bootstrap confidence interval; `p` is the "
        "bootstrap fraction in which the forecast failed to beat persistence.",
        "",
        "## Headline comparison",
        "",
        _to_markdown(table),
        "",
    ]

    # Error versus lead time: the single most informative view.
    if table["Horizon (h)"].nunique() > 1:
        pivot = table.pivot_table(
            index="Model", columns="Horizon (h)",
            values="Mean Error (km)", aggfunc="first",
        )
        lines += [
            "## Mean track error by lead time (km)",
            "",
            _to_markdown(pivot, index=True),
            "",
        ]

    if not basin_table.empty:
        lines += [
            "## By basin",
            "",
            _to_markdown(basin_table),
            "",
        ]

    lines += [
        "## Reading the error decomposition",
        "",
        "- **Cross-track error** is perpendicular to the storm's actual motion: "
        "a direction error, typically a missed or mistimed recurvature.",
        "- **Along-track error** is parallel to motion: a speed error. The MAE "
        "is its magnitude; the bias is its systematic component. A bias close "
        "to the MAE means the model is consistently wrong in one direction "
        "(negative = it runs storms too slow) rather than merely imprecise, "
        "which is a correctable error.",
        "",
    ]

    path.write_text("\n".join(lines))
    logger.info(f"Markdown report written to: {path}")


def save_results(
    table: pd.DataFrame,
    results: list[dict],
    basin_rows: list[dict],
) -> None:
    """
    Persist tables and detailed metrics.
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    table.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Comparison table: {MODEL_COMPARISON_PATH}")

    serializable = [
        {k: v for k, v in r.items() if not isinstance(v, np.ndarray) and not k.startswith("_")}
        for r in results
    ]

    EVALUATION_RESULTS_PATH.write_text(json.dumps(serializable, indent=2, default=float))
    logger.info(f"Detailed results: {EVALUATION_RESULTS_PATH}")

    basin_table = pd.DataFrame(basin_rows)
    write_markdown_report(table, basin_table)


# ==========================================================
# Main
# ==========================================================

def main(
    horizon: str | int = "all",
    models: list[str] = None,
    split_name: str = "test",
    make_plots: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Evaluate saved models across horizons and write the report.
    """

    horizons = (
        list(FORECAST_HORIZON_HOURS) if horizon == "all" else [int(horizon)]
    )

    all_results = []
    all_basin_rows = []

    with Timer("Full Evaluation"):
        frame = load_feature_dataset()

        for h in horizons:
            try:
                data = prepare_data(forecast_horizon=h, df=frame)
                results, basin_rows = evaluate_horizon(
                    h, data=data, models=models, split_name=split_name,
                )
                all_results.extend(results)
                all_basin_rows.extend(basin_rows)
            except Exception as exc:
                logger.warning(f"  {h}h horizon skipped: {exc}")

        if not all_results:
            raise RuntimeError(
                "Nothing evaluated. Train models first: "
                "python -m src.s5_training.train_model --model all --horizon all"
            )

        table = build_comparison_table(all_results)
        save_results(table, all_results, all_basin_rows)

    logger.info("\n" + table.to_string(index=False))

    if make_plots:
        try:
            from src.s7_analysis.evaluation_plots import generate_all_plots

            generate_all_plots(all_results)
        except Exception as exc:
            logger.warning(f"Plot generation skipped: {exc}")

    return table, all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument(
        "--horizon", default="all",
        help="Horizon in hours, or 'all'.",
    )
    parser.add_argument(
        "--split", default="test", choices=("test", "val"),
        help="Which split to evaluate on. Use val while iterating; report test.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Subset of {list(MODEL_REGISTRY)}.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        horizon=args.horizon,
        models=args.models,
        split_name=args.split,
        make_plots=not args.no_plots,
    )