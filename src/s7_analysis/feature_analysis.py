"""
feature_analysis.py

Feature importance analysis for cyclone track forecasting.

Replaces feature_importance.py, which could not run: it called
`model.named_steps["model"]` on a bare estimator and raised AttributeError
before reaching anything else.

Why this is more than a bar chart
---------------------------------
Impurity-based importance -- `.feature_importances_` -- is biased toward
high-cardinality continuous features, and it divides credit arbitrarily between
correlated ones. This feature set is full of correlated groups: LAT with its
three lags, DELTA_LAT with VELOCITY_V and DELTA_LAT_MEAN_24H. Impurity
importance scatters the credit among them in a way that depends on the random
seed as much as on the data, so the resulting ranking is close to unreadable.

Three analyses, in increasing order of how much they actually prove:

permutation   Shuffle one column on held-out data and measure how far the
              track error rises, in kilometres. Model-agnostic and measured
              where it matters, but a shuffled feature can be reconstructed
              from its correlated neighbours, which understates group effects.

grouped       Shuffle a whole correlated group at once. Permuting LAT_LAG_1
              alone changes almost nothing because LAT still carries the
              signal; permuting the position group together reveals what that
              information is worth.

ablation      Retrain without each group. The only one that answers "would I
              lose anything by deleting these features?" -- expensive, and the
              one worth putting in a writeup.

Usage
-----
    python -m src.s7_analysis.feature_analysis --horizon 24
    python -m src.s7_analysis.feature_analysis --horizon 24 --ablation
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_logger, Timer
from src.utils.metrics import haversine_distance, offset_position
from src.utils.config import (
    FEATURE_GROUPS,
    TARGET_MODE,
    DEFAULT_FORECAST_HORIZON,
    FIGURES_DIR,
    REPORT_DIR,
    model_path,
    features_excluding,
    get_model_params,
)
from src.s5_training.train import prepare_data, PreparedData, Split
from src.s5_training.save_model import load_model, read_metadata

logger = get_logger(__name__)

N_REPEATS = 5
RANDOM_SEED = 42

IMPORTANCE_CSV = REPORT_DIR / "feature_importance.csv"
ABLATION_CSV = REPORT_DIR / "feature_ablation.csv"


# ==========================================================
# Scoring
# ==========================================================

def track_error_km(model, X: pd.DataFrame, split: Split) -> float:
    """
    Mean track error in kilometres for a given feature matrix.

    Base positions come from the split metadata, not from X. That matters:
    permutation shuffles the LAT and LON columns, so reading the reference
    position out of X would corrupt the reconstruction for exactly the features
    being tested. Row order is preserved by permutation, so the captured
    metadata stays aligned.
    """

    predictions = np.asarray(model.predict(X), dtype=float)
    actual = split.y.to_numpy(dtype=float)

    base_lat = split.meta["LAT"].to_numpy(dtype=float)
    base_lon = split.meta["LON"].to_numpy(dtype=float)

    if TARGET_MODE == "displacement":
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
# Impurity Importance
# ==========================================================

def impurity_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """
    The model's built-in importance, for comparison only.

    Included so the divergence from permutation importance is visible. Where
    the two rankings disagree sharply, the impurity one is usually the one to
    distrust.
    """

    values = getattr(model, "feature_importances_", None)

    if values is None:
        return pd.DataFrame()

    if len(values) != len(feature_names):
        logger.warning(
            f"  Model reports {len(values)} importances for "
            f"{len(feature_names)} names — skipping impurity importance."
        )
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "impurity": values})
        .sort_values("impurity", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================
# Permutation Importance
# ==========================================================

def permutation_importance(
    model,
    split: Split,
    n_repeats: int = N_REPEATS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Per-feature permutation importance, expressed in kilometres.

    Importance is the increase in mean track error when a column is shuffled.
    Reporting it in kilometres rather than as a unitless score makes it
    directly comparable to the error figures elsewhere: "shuffling this feature
    costs 14 km at 24h" is a sentence with meaning.
    """

    rng = np.random.default_rng(seed)

    baseline = track_error_km(model, split.X, split)
    logger.info(f"  Baseline track error: {baseline:.1f} km")

    rows = []

    for column in split.X.columns:
        increases = []

        for _ in range(n_repeats):
            shuffled = split.X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            increases.append(track_error_km(model, shuffled, split) - baseline)

        rows.append({
            "feature": column,
            "importance_km": float(np.mean(increases)),
            "std_km": float(np.std(increases)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("importance_km", ascending=False)
        .reset_index(drop=True)
    )


def grouped_permutation_importance(
    model,
    split: Split,
    n_repeats: int = N_REPEATS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Permutation importance over whole feature groups.

    Single-feature permutation understates anything with a correlated
    substitute: shuffle LAT_LAG_1 and the model simply leans on LAT. Shuffling
    the group together removes the substitute as well, which is what makes this
    the more honest number for correlated families.
    """

    rng = np.random.default_rng(seed)
    baseline = track_error_km(model, split.X, split)

    rows = []

    for group_name, columns in FEATURE_GROUPS.items():
        present = [c for c in columns if c in split.X.columns]

        if not present:
            continue

        increases = []

        for _ in range(n_repeats):
            shuffled = split.X.copy()
            order = rng.permutation(len(shuffled))

            # One shared permutation across the group preserves the internal
            # correlation structure while destroying the link to the target.
            for column in present:
                shuffled[column] = shuffled[column].to_numpy()[order]

            increases.append(track_error_km(model, shuffled, split) - baseline)

        rows.append({
            "group": group_name,
            "n_features": len(present),
            "importance_km": float(np.mean(increases)),
            "std_km": float(np.std(increases)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("importance_km", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================
# Ablation
# ==========================================================

def group_ablation(
    model_name: str,
    horizon: int,
    data: PreparedData,
) -> pd.DataFrame:
    """
    Retrain without each feature group in turn.

    The strongest evidence available, because it lets the model recover
    whatever it can from the remaining features -- which permutation does not.
    A group whose removal costs nothing is a group you can delete, and being
    able to say "removing the rolling-motion features costs 0.4 km, so they
    were dropped" is a better line in a writeup than adding more features.

    Expensive: one full fit per group.
    """

    from src.s5_training.train_model import fit_model

    baseline_model, _ = fit_model(model_name, data, use_early_stopping=False)
    baseline_error = track_error_km(baseline_model, data.test.X, data.test)

    logger.info(f"  Full feature set: {baseline_error:.1f} km")

    rows = [{
        "removed_group": "(none)",
        "n_features": data.train.X.shape[1],
        "error_km": round(baseline_error, 2),
        "delta_km": 0.0,
    }]

    for group_name in FEATURE_GROUPS:
        remaining = [c for c in features_excluding(group_name) if c in data.train.X.columns]

        if not remaining:
            continue

        reduced = PreparedData(
            train=Split("train", data.train.X[remaining], data.train.y, data.train.meta),
            val=Split("val", data.val.X[remaining], data.val.y, data.val.meta)
            if data.val is not None else None,
            test=Split("test", data.test.X[remaining], data.test.y, data.test.meta),
            horizon=data.horizon,
            target_mode=data.target_mode,
            config=data.config,
        )

        model, _ = fit_model(model_name, reduced, use_early_stopping=False)
        error = track_error_km(model, reduced.test.X, reduced.test)

        rows.append({
            "removed_group": group_name,
            "n_features": len(remaining),
            "error_km": round(error, 2),
            "delta_km": round(error - baseline_error, 2),
        })

        logger.info(
            f"  without {group_name:<16}: {error:>7.1f} km "
            f"({error - baseline_error:+.1f})"
        )

    return (
        pd.DataFrame(rows)
        .sort_values("delta_km", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================
# Correlation
# ==========================================================

def correlated_clusters(X: pd.DataFrame, threshold: float = 0.95) -> list[list[str]]:
    """
    Groups of near-duplicate features.

    Context for reading any single-feature importance: within one of these
    clusters, which member gets the credit is close to arbitrary.
    """

    correlation = X.corr().abs()
    clusters = []
    assigned = set()

    for feature in correlation.columns:
        if feature in assigned:
            continue

        partners = correlation.index[
            (correlation[feature] > threshold) & (correlation.index != feature)
        ].tolist()

        if partners:
            cluster = [feature] + [p for p in partners if p not in assigned]
            clusters.append(cluster)
            assigned.update(cluster)

    return clusters


# ==========================================================
# Plots
# ==========================================================

def plot_importance(
    permutation: pd.DataFrame,
    grouped: pd.DataFrame,
    horizon: int,
    n_top: int = 20,
) -> None:
    """
    Permutation importance per feature and per group.
    """

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, n_top * 0.32)))

    top = permutation.head(n_top)
    axes[0].barh(
        top["feature"][::-1], top["importance_km"][::-1],
        xerr=top["std_km"][::-1], color="#2196F3", edgecolor="white", capsize=2,
    )
    axes[0].set_xlabel("Track error increase when shuffled (km)")
    axes[0].set_title(f"Permutation importance — {horizon}h")

    axes[1].barh(
        grouped["group"][::-1], grouped["importance_km"][::-1],
        xerr=grouped["std_km"][::-1], color="#FF9800", edgecolor="white", capsize=2,
    )
    axes[1].set_xlabel("Track error increase when group shuffled (km)")
    axes[1].set_title(f"Grouped permutation importance — {horizon}h")

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"feature_importance_analysis_{horizon}h.png"
    fig.savefig(path)
    plt.close(fig)

    logger.info(f"  {path.name}")


def plot_ablation(ablation: pd.DataFrame, horizon: int) -> None:
    """
    Cost of removing each feature group.
    """

    frame = ablation[ablation["removed_group"] != "(none)"]

    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(frame) * 0.4)))

    colors = ["#E53935" if v > 0 else "#43A047" for v in frame["delta_km"][::-1]]

    ax.barh(frame["removed_group"][::-1], frame["delta_km"][::-1],
            color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Change in track error when group removed (km)")
    ax.set_title(f"Feature group ablation — {horizon}h")

    fig.tight_layout()

    path = FIGURES_DIR / f"feature_ablation_{horizon}h.png"
    fig.savefig(path)
    plt.close(fig)

    logger.info(f"  {path.name}")


# ==========================================================
# Main
# ==========================================================

def main(
    model_name: str = "random_forest",
    horizon: int = DEFAULT_FORECAST_HORIZON,
    n_repeats: int = N_REPEATS,
    run_ablation: bool = False,
    n_top: int = 20,
) -> dict:
    """
    Run the feature analysis and write tables and figures.
    """

    logger.info("=" * 72)
    logger.info(f"FEATURE ANALYSIS — {model_name}, {horizon}h")
    logger.info("=" * 72)

    path = model_path(model_name, horizon, TARGET_MODE)

    if not path.exists():
        raise FileNotFoundError(
            f"No model at {path}. Train first:\n"
            f"  python -m src.s5_training.train_model --model {model_name} --horizon {horizon}"
        )

    data = prepare_data(forecast_horizon=horizon)
    model = load_model(path)

    metadata = read_metadata(path)
    trained_on = metadata.get("feature_names")

    if trained_on and trained_on != list(data.test.X.columns):
        raise ValueError(
            "The saved model was fitted on a different feature set. "
            "Retrain before analysing importance — otherwise every label here "
            "is attached to the wrong column."
        )

    results = {}

    with Timer("Permutation importance"):
        results["permutation"] = permutation_importance(
            model, data.test, n_repeats=n_repeats
        )

    with Timer("Grouped permutation importance"):
        results["grouped"] = grouped_permutation_importance(
            model, data.test, n_repeats=n_repeats
        )

    results["impurity"] = impurity_importance(model, list(data.test.X.columns))

    # ---- Report ----
    logger.info("-" * 72)
    logger.info(f"TOP {n_top} FEATURES (permutation, km)")
    logger.info("-" * 72)
    logger.info("\n" + results["permutation"].head(n_top).to_string(index=False))

    logger.info("-" * 72)
    logger.info("FEATURE GROUPS (grouped permutation, km)")
    logger.info("-" * 72)
    logger.info("\n" + results["grouped"].to_string(index=False))

    # Where impurity and permutation disagree, impurity is usually the
    # unreliable one -- worth seeing rather than assuming.
    if not results["impurity"].empty:
        merged = results["permutation"].merge(results["impurity"], on="feature")
        merged["rank_permutation"] = merged["importance_km"].rank(ascending=False)
        merged["rank_impurity"] = merged["impurity"].rank(ascending=False)
        merged["rank_gap"] = (merged["rank_permutation"] - merged["rank_impurity"]).abs()

        disagreements = merged.nlargest(5, "rank_gap")[
            ["feature", "rank_permutation", "rank_impurity", "rank_gap"]
        ]

        logger.info("-" * 72)
        logger.info("LARGEST RANKING DISAGREEMENTS (impurity vs permutation)")
        logger.info("-" * 72)
        logger.info("\n" + disagreements.to_string(index=False))

        results["comparison"] = merged

    clusters = correlated_clusters(data.test.X)

    if clusters:
        logger.info("-" * 72)
        logger.info(f"CORRELATED CLUSTERS (|r| > 0.95) — credit within each is arbitrary")
        logger.info("-" * 72)
        for cluster in clusters[:8]:
            logger.info(f"  {cluster}")

    # ---- Ablation ----
    if run_ablation:
        logger.info("-" * 72)
        logger.info("GROUP ABLATION (retraining without each group)")
        logger.info("-" * 72)

        with Timer("Ablation study"):
            results["ablation"] = group_ablation(model_name, horizon, data)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        results["ablation"].to_csv(ABLATION_CSV, index=False)
        logger.info(f"  Written to {ABLATION_CSV}")
        plot_ablation(results["ablation"], horizon)

    # ---- Save ----
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results["permutation"].to_csv(IMPORTANCE_CSV, index=False)
    logger.info(f"Feature importance written to {IMPORTANCE_CSV}")

    plot_importance(results["permutation"], results["grouped"], horizon, n_top=n_top)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse feature importance.")
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--horizon", type=int, default=DEFAULT_FORECAST_HORIZON)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--ablation", action="store_true",
        help="Retrain without each feature group. Slow, but the strongest evidence.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_name=args.model,
        horizon=args.horizon,
        n_repeats=args.repeats,
        run_ablation=args.ablation,
        n_top=args.top,
    )