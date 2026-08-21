"""
evaluation_plots.py

Figures for cyclone track forecasting evaluation.

Data sources
------------
all_results     aggregate metrics per (model, horizon) from evaluate.py
predictions     per-row forecasts from predict.py --export, used for anything
                distributional. Regenerated on demand if absent.

Figures
-------
1. error_vs_lead_time      mean track error against forecast horizon. The
                           single most informative figure here.
2. skill_vs_persistence    skill with bootstrap confidence intervals
3. error_distribution      per-model error histograms
4. cumulative_cdf          fraction of forecasts within a distance
5. along_cross_track       direction error vs speed error, the diagnostic
                           that says *how* the model fails
6. error_by_intensity      error by Saffir-Simpson category
7. feature_importance      what the model actually uses
8. forecast_track_example  one issue time projected to every lead time
9. residuals               residual structure on the target scale
10. error_by_basin         where the model works and where it does not
11. storm_lifecycle        error through one storm's lifetime, ordered by its
                           own timeline rather than by row position

Notes on what changed
---------------------
The feature importance figure never rendered in the original: it reached for
`model.named_steps["model"]` on a bare estimator, raised AttributeError, and
was swallowed by a bare except. Feature names now come from the model's
metadata sidecar rather than from a slice of the config list, which was a
tautology that returned the first n names regardless of what the model saw.

Intensity category labels were off by one against the bin edges -- category 1
is a tropical storm, not a Category 1 hurricane -- so every box in that figure
was labelled one class too strong.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_logger
from src.utils.config import (
    FIGURES_DIR,
    PREDICTION_PATH,
    INTENSITY_LABEL_NAMES,
    TARGET_MODE,
    DEFAULT_FORECAST_HORIZON,
    model_path,
)

logger = get_logger(__name__)

RANDOM_SEED = 42

# Consistent identity across every figure. Baselines are muted; models are not.
SERIES_COLORS = {
    "Random Forest": "#2196F3",
    "XGBoost": "#FF9800",
    "Persistence": "#9E9E9E",
    "Linear Extrapolation": "#E91E63",
    "Climatology": "#795548",
    "Linear Regression": "#4CAF50",
}

DEFAULT_COLOR = "#607D8B"
ACTUAL_COLOR = "#1565C0"
PREDICTED_COLOR = "#E53935"

BASELINE_NAMES = {
    "Persistence", "Linear Extrapolation", "Climatology", "Linear Regression",
}


def _color(name: str) -> str:
    return SERIES_COLORS.get(name, DEFAULT_COLOR)


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _save(fig, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  {path.name}")


def _results_frame(all_results: list[dict]) -> pd.DataFrame:
    """
    Aggregate results as a tidy frame.
    """

    return pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in all_results
    ])


# ==========================================================
# 1. Error vs Lead Time
# ==========================================================

def plot_error_vs_lead_time(all_results: list[dict]) -> None:
    """
    Mean track error against forecast horizon.

    The figure to lead with. It shows at a glance whether the model beats the
    baselines, and -- more usefully -- at which lead times it does. A model
    that beats persistence everywhere but linear extrapolation nowhere has
    learned inertia, not steering, and only this view makes that visible.
    """

    _setup_style()

    frame = _results_frame(all_results)

    if frame["horizon"].nunique() < 2:
        logger.info("  (skipping error_vs_lead_time: single horizon)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, group in frame.groupby("model"):
        group = group.sort_values("horizon")
        is_baseline = name in BASELINE_NAMES

        axes[0].plot(
            group["horizon"], group["mean_km"],
            marker="o" if not is_baseline else "s",
            linewidth=2.5 if not is_baseline else 1.5,
            linestyle="-" if not is_baseline else "--",
            alpha=1.0 if not is_baseline else 0.7,
            color=_color(name), label=name,
        )

        if {"mean_ci_lower_km", "mean_ci_upper_km"}.issubset(group.columns):
            axes[0].fill_between(
                group["horizon"],
                group["mean_ci_lower_km"], group["mean_ci_upper_km"],
                color=_color(name), alpha=0.12,
            )

    axes[0].set_xlabel("Forecast lead time (hours)")
    axes[0].set_ylabel("Mean track error (km)")
    axes[0].set_title("Track error by lead time")
    axes[0].set_xticks(sorted(frame["horizon"].unique()))
    axes[0].legend(fontsize=9)

    # Log scale reveals whether error growth is linear or compounding.
    for name, group in frame.groupby("model"):
        group = group.sort_values("horizon")
        axes[1].plot(
            group["horizon"], group["mean_km"],
            marker="o", linewidth=2, color=_color(name), label=name,
            alpha=1.0 if name not in BASELINE_NAMES else 0.6,
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Forecast lead time (hours)")
    axes[1].set_ylabel("Mean track error (km, log scale)")
    axes[1].set_title("Error growth rate")
    axes[1].set_xticks(sorted(frame["horizon"].unique()))

    fig.tight_layout()
    _save(fig, "error_vs_lead_time.png")


# ==========================================================
# 2. Skill With Confidence Intervals
# ==========================================================

def plot_skill(all_results: list[dict]) -> None:
    """
    Skill relative to persistence, with bootstrap intervals.

    An interval that crosses zero means the improvement is not distinguishable
    from sampling noise. Reporting skill without one is reporting a point
    estimate as though it were a result.
    """

    _setup_style()

    frame = _results_frame(all_results)

    if "skill_pct" not in frame.columns:
        logger.info("  (skipping skill plot: no skill scores)")
        return

    frame = frame.dropna(subset=["skill_pct"])

    if frame.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    horizons = sorted(frame["horizon"].unique())
    models = sorted(frame["model"].unique())
    width = 0.8 / max(len(models), 1)

    for i, name in enumerate(models):
        group = frame[frame["model"] == name].set_index("horizon").reindex(horizons)
        positions = np.arange(len(horizons)) + i * width - 0.4 + width / 2

        lower = group["skill_pct"] - group.get("skill_ci_lower_pct", group["skill_pct"])
        upper = group.get("skill_ci_upper_pct", group["skill_pct"]) - group["skill_pct"]

        ax.bar(
            positions, group["skill_pct"], width * 0.9,
            yerr=[lower.abs().fillna(0), upper.abs().fillna(0)],
            capsize=3, color=_color(name), label=name,
            alpha=1.0 if name not in BASELINE_NAMES else 0.65,
            edgecolor="white",
        )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(horizons)))
    ax.set_xticklabels([f"{h}h" for h in horizons])
    ax.set_xlabel("Forecast lead time")
    ax.set_ylabel("Skill vs persistence (%)")
    ax.set_title("Forecast skill relative to persistence (95% bootstrap CI)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "skill_vs_persistence.png")


# ==========================================================
# 3 & 4. Distributions
# ==========================================================

def plot_error_distribution(predictions: pd.DataFrame, horizon: int) -> None:
    """
    Error histogram. Track error is strongly right-skewed, so the mean sits
    well above the median and a handful of badly-missed recurvatures dominate
    it -- worth seeing rather than summarising.
    """

    _setup_style()

    errors = predictions["ERROR_KM"].dropna()

    if errors.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(errors, bins=60, color=SERIES_COLORS["Random Forest"],
            alpha=0.75, edgecolor="white")

    for value, label, color in (
        (errors.mean(), f"mean {errors.mean():.0f} km", "#E53935"),
        (errors.median(), f"median {errors.median():.0f} km", "#1B5E20"),
        (errors.quantile(0.9), f"P90 {errors.quantile(0.9):.0f} km", "#F57C00"),
    ):
        ax.axvline(value, color=color, linestyle="--", linewidth=2, label=label)

    ax.set_xlabel("Track error (km)")
    ax.set_ylabel("Forecasts")
    ax.set_title(f"Track error distribution — {horizon}h")
    ax.legend()

    fig.tight_layout()
    _save(fig, "error_distribution.png")


def plot_cumulative_cdf(all_results: list[dict], horizon: int = None) -> None:
    """
    Fraction of forecasts falling within a given distance.

    Easier to communicate than a percentile: "three quarters of 24h forecasts
    land within 100 km" is a sentence anyone can act on.
    """

    _setup_style()

    frame = _results_frame(all_results)

    horizon = horizon or (
        DEFAULT_FORECAST_HORIZON if DEFAULT_FORECAST_HORIZON in frame["horizon"].values
        else frame["horizon"].iloc[0]
    )

    subset = frame[frame["horizon"] == horizon]
    threshold_columns = [c for c in subset.columns if c.startswith("within_")]

    if not threshold_columns:
        return

    thresholds = sorted(int(c.split("_")[1].replace("km", "")) for c in threshold_columns)

    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in subset.iterrows():
        values = [row[f"within_{t}km_pct"] for t in thresholds]

        ax.plot(
            thresholds, values,
            marker="o", linewidth=2, color=_color(row["model"]),
            label=row["model"],
            alpha=1.0 if row["model"] not in BASELINE_NAMES else 0.65,
        )

    ax.set_xlabel("Distance threshold (km)")
    ax.set_ylabel("Forecasts within threshold (%)")
    ax.set_title(f"Cumulative accuracy — {horizon}h")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "cumulative_cdf.png")


# ==========================================================
# 5. Along / Cross-Track
# ==========================================================

def plot_along_cross_track(
    predictions: pd.DataFrame,
    all_results: list[dict],
    horizon: int,
) -> None:
    """
    Direction error against speed error.

    The most diagnostic figure available. Cross-track spread means the model
    gets the heading wrong -- typically a missed recurvature. Along-track
    offset means it gets the speed wrong, and a non-zero mean is a systematic
    bias that could be corrected. A single aggregate error hides both.
    """

    _setup_style()

    if not {"ALONG_TRACK_KM", "CROSS_TRACK_KM"}.issubset(predictions.columns):
        return

    valid = predictions[["ALONG_TRACK_KM", "CROSS_TRACK_KM"]].dropna()

    if valid.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rng = np.random.default_rng(RANDOM_SEED)
    sample = valid.iloc[rng.choice(len(valid), min(5000, len(valid)), replace=False)]

    axes[0].scatter(
        sample["ALONG_TRACK_KM"], sample["CROSS_TRACK_KM"],
        alpha=0.25, s=8, color=SERIES_COLORS["Random Forest"],
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].scatter(
        [valid["ALONG_TRACK_KM"].mean()], [valid["CROSS_TRACK_KM"].mean()],
        s=200, marker="X", color=PREDICTED_COLOR, zorder=5,
        edgecolor="white", linewidth=1.5, label="mean bias",
    )
    axes[0].set_xlabel("Along-track error (km)   →  forecast too fast")
    axes[0].set_ylabel("Cross-track error (km)   →  forecast right of track")
    axes[0].set_title(f"Error decomposition — {horizon}h")
    axes[0].legend()
    axes[0].set_aspect("equal", adjustable="datalim")

    # How the two components grow with lead time.
    frame = _results_frame(all_results)

    if "cross_track_mae_km" in frame.columns and frame["horizon"].nunique() > 1:
        best = (
            frame[~frame["model"].isin(BASELINE_NAMES)]
            if (~frame["model"].isin(BASELINE_NAMES)).any() else frame
        )
        chosen = best.loc[best.groupby("horizon")["mean_km"].idxmin()].sort_values("horizon")

        axes[1].plot(chosen["horizon"], chosen["cross_track_mae_km"],
                     marker="o", linewidth=2.5, color="#E91E63",
                     label="Cross-track MAE (direction)")
        axes[1].plot(chosen["horizon"], chosen["along_track_mae_km"],
                     marker="s", linewidth=2.5, color="#3F51B5",
                     label="Along-track MAE (speed)")
        axes[1].plot(chosen["horizon"], chosen["along_track_bias_km"].abs(),
                     marker="^", linewidth=1.5, linestyle="--", color="#3F51B5",
                     alpha=0.6, label="|Along-track bias|")

        axes[1].set_xlabel("Forecast lead time (hours)")
        axes[1].set_ylabel("Error component (km)")
        axes[1].set_title("Which component dominates")
        axes[1].set_xticks(sorted(chosen["horizon"].unique()))
        axes[1].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "along_cross_track.png")


# ==========================================================
# 6. Error by Intensity
# ==========================================================

def plot_error_by_intensity(
    predictions: pd.DataFrame,
    features: pd.DataFrame = None,
    horizon: int = None,
) -> None:
    """
    Error by Saffir-Simpson category.

    Labels come from INTENSITY_LABEL_NAMES in config. The previous hardcoded
    map was off by one against the bin edges: with seven edges, category 0 is a
    tropical depression and category 1 a tropical storm, so what was labelled
    "Cat 1" was in fact a tropical storm.
    """

    _setup_style()

    if features is None or "INTENSITY_CATEGORY" not in features.columns:
        logger.info("  (skipping error_by_intensity: no intensity column)")
        return

    frame = pd.DataFrame({
        "error": predictions["ERROR_KM"].to_numpy(),
        "category": features["INTENSITY_CATEGORY"].to_numpy(),
    }).dropna()

    if frame.empty:
        return

    frame["category"] = frame["category"].astype(int)

    categories = sorted(frame["category"].unique())
    data = [frame.loc[frame["category"] == c, "error"].to_numpy() for c in categories]
    labels = [INTENSITY_LABEL_NAMES.get(c, str(c)) for c in categories]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(11, 6))

    box = ax.boxplot(data, patch_artist=True, showfliers=False,
                     medianprops={"color": "black", "linewidth": 2})

    palette = ["#81C784", "#AED581", "#FDD835", "#FFB74D", "#FF8A65",
               "#E57373", "#B71C1C"]

    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels([f"{l}\nn={n:,}" for l, n in zip(labels, counts)])
    ax.set_xlabel("Saffir-Simpson category")
    ax.set_ylabel("Track error (km)")
    ax.set_title(
        f"Track error by intensity{f' — {horizon}h' if horizon else ''}"
    )

    fig.tight_layout()
    _save(fig, "error_by_intensity.png")


# ==========================================================
# 7. Feature Importance
# ==========================================================

def plot_feature_importance(
    model_name: str = "random_forest",
    horizon: int = None,
    n_top: int = 25,
) -> None:
    """
    Feature importance, read from the saved artifact.

    The original called `model.named_steps["model"]`, which only works on a
    Pipeline. The saved artifact is a bare estimator, so this raised
    AttributeError on every run and was silently caught -- the figure has never
    been produced. Feature names come from the metadata sidecar, which records
    the exact order the model was fitted on.
    """

    _setup_style()

    from src.s5_training.save_model import load_model, read_metadata

    horizon = horizon or DEFAULT_FORECAST_HORIZON
    path = model_path(model_name, horizon, TARGET_MODE)

    if not path.exists():
        logger.info(f"  (skipping feature_importance: {path.name} not found)")
        return

    model = load_model(path, warn_on_version_mismatch=False)
    importances = getattr(model, "feature_importances_", None)

    if importances is None:
        logger.info(f"  (skipping feature_importance: {model_name} exposes none)")
        return

    metadata = read_metadata(path)
    names = metadata.get("feature_names")

    if not names or len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
        logger.warning("  feature names unavailable; using positional labels")

    frame = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(n_top)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(frame) * 0.32)))

    ax.barh(
        frame["feature"][::-1], frame["importance"][::-1],
        color=_color("Random Forest" if model_name == "random_forest" else "XGBoost"),
        edgecolor="white",
    )

    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name.replace('_', ' ').title()} — top {len(frame)} features ({horizon}h)")

    fig.tight_layout()
    _save(fig, f"feature_importance_{model_name}_{horizon}h.png")


# ==========================================================
# 8. Forecast Track
# ==========================================================

def plot_forecast_track(
    sid: str = None,
    model_name: str = "random_forest",
) -> None:
    """
    One issue time projected to every lead time, against what happened.

    This is a forecast track. The original figure plotted the h-ahead position
    for each consecutive issue time, which traces the observed track shifted
    forward and is not a forecast of anything.
    """

    _setup_style()

    try:
        from src.s6_inference.predict import forecast_track, list_available_storms
        from src.s5_training.train import load_feature_dataset

        frame = load_feature_dataset()

        if sid is None:
            candidates = list_available_storms(limit=5)
            sid = candidates["SID"].iloc[0]

        track = forecast_track(sid, model_name=model_name, frame=frame)
    except Exception as exc:
        logger.info(f"  (skipping forecast_track: {exc})")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    start_lat = track["CURRENT_LAT"].iloc[0]
    start_lon = track["CURRENT_LON"].iloc[0]

    observed_lat = np.concatenate([[start_lat], track["TRUE_LAT"].to_numpy()])
    observed_lon = np.concatenate([[start_lon], track["TRUE_LON"].to_numpy()])
    forecast_lat = np.concatenate([[start_lat], track["PRED_LAT"].to_numpy()])
    forecast_lon = np.concatenate([[start_lon], track["PRED_LON"].to_numpy()])

    ax.plot(observed_lon, observed_lat, marker="o", markersize=7, linewidth=2.5,
            color=ACTUAL_COLOR, label="Observed", zorder=3)
    ax.plot(forecast_lon, forecast_lat, marker="X", markersize=8, linewidth=2.5,
            linestyle="--", color=PREDICTED_COLOR, label="Forecast", zorder=4)

    # Error at each lead time, drawn as the gap it actually is.
    for _, row in track.iterrows():
        ax.plot([row["TRUE_LON"], row["PRED_LON"]],
                [row["TRUE_LAT"], row["PRED_LAT"]],
                color="gray", linewidth=1, alpha=0.6, zorder=2)
        ax.annotate(
            f"{int(row['LEAD_HOURS'])}h\n{row['ERROR_KM']:.0f}km",
            (row["TRUE_LON"], row["TRUE_LAT"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color="#555",
        )

    ax.scatter([start_lon], [start_lat], s=180, marker="^", color="#1B5E20",
               zorder=5, edgecolor="white", linewidth=1.5, label="Issue time")

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"Forecast track — storm {sid}\n"
        f"issued {pd.Timestamp(track['ISO_TIME'].iloc[0]):%Y-%m-%d %H:%M} UTC, "
        f"mean error {track['ERROR_KM'].mean():.0f} km"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    _save(fig, "forecast_track_example.png")


# ==========================================================
# 9. Residuals
# ==========================================================

def plot_residuals(predictions: pd.DataFrame, horizon: int) -> None:
    """
    Residual structure.

    Under displacement targets these residuals are informative: a trend against
    the predicted value means the model systematically under- or over-shoots
    large displacements. Under absolute targets the same plot is dominated by
    the current position and shows almost nothing.
    """

    _setup_style()

    residual_lat = predictions["TRUE_LAT"] - predictions["PRED_LAT"]
    residual_lon = predictions["TRUE_LON"] - predictions["PRED_LON"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].scatter(predictions["PRED_LAT"], residual_lat,
                       alpha=0.15, s=6, color=SERIES_COLORS["Random Forest"])
    axes[0, 0].axhline(0, color=PREDICTED_COLOR, linewidth=1.5)
    axes[0, 0].set_xlabel("Predicted latitude (°)")
    axes[0, 0].set_ylabel("Residual (°)")
    axes[0, 0].set_title("Latitude residuals")

    axes[0, 1].scatter(predictions["PRED_LON"], residual_lon,
                       alpha=0.15, s=6, color=SERIES_COLORS["XGBoost"])
    axes[0, 1].axhline(0, color=PREDICTED_COLOR, linewidth=1.5)
    axes[0, 1].set_xlabel("Predicted longitude (°)")
    axes[0, 1].set_ylabel("Residual (°)")
    axes[0, 1].set_title("Longitude residuals")

    axes[1, 0].hist(residual_lat.dropna(), bins=60,
                    color=SERIES_COLORS["Random Forest"], alpha=0.75)
    axes[1, 0].axvline(0, color=PREDICTED_COLOR, linewidth=1.5)
    axes[1, 0].set_xlabel("Residual (°)")
    axes[1, 0].set_title(f"Latitude residuals (mean {residual_lat.mean():+.3f}°)")

    axes[1, 1].hist(residual_lon.dropna(), bins=60,
                    color=SERIES_COLORS["XGBoost"], alpha=0.75)
    axes[1, 1].axvline(0, color=PREDICTED_COLOR, linewidth=1.5)
    axes[1, 1].set_xlabel("Residual (°)")
    axes[1, 1].set_title(f"Longitude residuals (mean {residual_lon.mean():+.3f}°)")

    fig.suptitle(f"Residual analysis — {horizon}h", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "residuals.png")


# ==========================================================
# 10. Error by Basin
# ==========================================================

def plot_error_by_basin(predictions: pd.DataFrame, horizon: int) -> None:
    """
    Error by ocean basin.

    Basins differ in steering regime, observation density, and how often storms
    recurve. A single global mean averages over genuinely different problems.
    """

    _setup_style()

    if "BASIN" not in predictions.columns or predictions["BASIN"].isna().all():
        return

    frame = predictions[["BASIN", "ERROR_KM"]].dropna()
    counts = frame["BASIN"].value_counts()
    basins = [b for b in counts.index if counts[b] >= 50]

    if len(basins) < 2:
        logger.info("  (skipping error_by_basin: fewer than two basins)")
        return

    data = [frame.loc[frame["BASIN"] == b, "ERROR_KM"].to_numpy() for b in basins]

    fig, ax = plt.subplots(figsize=(10, 6))

    box = ax.boxplot(data, patch_artist=True, showfliers=False,
                     medianprops={"color": "black", "linewidth": 2})

    for patch in box["boxes"]:
        patch.set_facecolor(SERIES_COLORS["Random Forest"])
        patch.set_alpha(0.7)

    ax.set_xticklabels([f"{b}\nn={counts[b]:,}" for b in basins])
    ax.set_xlabel("Basin")
    ax.set_ylabel("Track error (km)")
    ax.set_title(f"Track error by basin — {horizon}h")

    fig.tight_layout()
    _save(fig, "error_by_basin.png")


# ==========================================================
# 11. Storm Lifecycle
# ==========================================================

def plot_storm_lifecycle(
    predictions: pd.DataFrame,
    sid: str = None,
    horizon: int = None,
) -> None:
    """
    Forecast error through a single storm's lifetime.

    The honest version of a "sequence" plot. Rows are ordered by the storm's
    own timeline, so consecutive points are genuinely consecutive -- unlike a
    plot indexed by row position in the test set, where adjacent points are
    unrelated observations and any apparent agreement between actual and
    predicted is a property of the sort order rather than the model.

    What it shows: forecast difficulty is not uniform across a storm's life.
    Genesis is poorly observed, recurvature is hard, and a mature storm in
    steady westward motion is easy. An aggregate error averages all three.
    """

    _setup_style()

    if "SID" not in predictions.columns:
        return

    if sid is None:
        # A storm with enough forecasts to show structure.
        counts = predictions["SID"].value_counts()
        eligible = counts[counts >= 8]

        if eligible.empty:
            logger.info("  (skipping storm_lifecycle: no storm with enough rows)")
            return

        sid = eligible.index[len(eligible) // 2]

    storm = predictions[predictions["SID"] == sid].sort_values("ISO_TIME")

    if len(storm) < 3:
        return

    hours = (
        pd.to_datetime(storm["ISO_TIME"]) - pd.to_datetime(storm["ISO_TIME"]).min()
    ).dt.total_seconds() / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(hours, storm["ERROR_KM"], marker="o", linewidth=2,
                 color=SERIES_COLORS["Random Forest"], label="Track error")
    axes[0].axhline(storm["ERROR_KM"].mean(), color=PREDICTED_COLOR,
                    linestyle="--", linewidth=1.5,
                    label=f"mean {storm['ERROR_KM'].mean():.0f} km")

    if "CROSS_TRACK_KM" in storm.columns:
        axes[0].plot(hours, storm["CROSS_TRACK_KM"].abs(), marker="s",
                     linewidth=1.5, alpha=0.7, color="#E91E63",
                     label="|cross-track|")

    axes[0].set_xlabel("Hours since first forecast")
    axes[0].set_ylabel("Error (km)")
    axes[0].set_title(f"Error through storm lifetime — {sid}")
    axes[0].legend(fontsize=9)

    # The storm's observed path, coloured by how badly it was forecast.
    scatter = axes[1].scatter(
        storm["CURRENT_LON"], storm["CURRENT_LAT"],
        c=storm["ERROR_KM"], cmap="YlOrRd", s=70,
        edgecolor="black", linewidth=0.5, zorder=3,
    )
    axes[1].plot(storm["CURRENT_LON"], storm["CURRENT_LAT"],
                 color="gray", linewidth=1.5, alpha=0.6, zorder=2)

    fig.colorbar(scatter, ax=axes[1], label="Track error (km)")

    axes[1].set_xlabel("Longitude (°)")
    axes[1].set_ylabel("Latitude (°)")
    axes[1].set_title("Where along the track errors occur")

    fig.suptitle(
        f"Storm {sid}{f' — {horizon}h forecasts' if horizon else ''}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "storm_lifecycle.png")


# ==========================================================
# Orchestration
# ==========================================================

def _load_predictions(horizon: int, model_name: str) -> pd.DataFrame | None:
    """
    Per-row predictions, generated if not already exported.
    """

    if PREDICTION_PATH.exists():
        frame = pd.read_csv(PREDICTION_PATH, parse_dates=["ISO_TIME"])

        if "LEAD_HOURS" in frame.columns and horizon in set(frame["LEAD_HOURS"]):
            return frame[frame["LEAD_HOURS"] == horizon].reset_index(drop=True)

    try:
        from src.s6_inference.predict import export_predictions

        return export_predictions(model_name=model_name, horizon=horizon)
    except Exception as exc:
        logger.warning(f"  Could not obtain per-row predictions: {exc}")
        return None


def generate_all_plots(
    all_results: list[dict],
    horizon: int = None,
    model_name: str = "random_forest",
    sid: str = None,
) -> None:
    """
    Produce every figure.

    Each plot is attempted independently: one failure does not abort the rest,
    but it is logged rather than silently swallowed.
    """

    logger.info("=" * 68)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 68)

    frame = _results_frame(all_results)

    if horizon is None:
        available = set(frame["horizon"]) if not frame.empty else set()
        horizon = (
            DEFAULT_FORECAST_HORIZON if DEFAULT_FORECAST_HORIZON in available
            else (sorted(available)[0] if available else DEFAULT_FORECAST_HORIZON)
        )

    predictions = _load_predictions(horizon, model_name)

    tasks = [
        ("error_vs_lead_time", lambda: plot_error_vs_lead_time(all_results)),
        ("skill", lambda: plot_skill(all_results)),
        ("cumulative_cdf", lambda: plot_cumulative_cdf(all_results, horizon)),
        ("feature_importance", lambda: plot_feature_importance(model_name, horizon)),
        ("forecast_track", lambda: plot_forecast_track(sid, model_name)),
    ]

    if predictions is not None and not predictions.empty:
        tasks += [
            ("error_distribution", lambda: plot_error_distribution(predictions, horizon)),
            ("along_cross_track", lambda: plot_along_cross_track(predictions, all_results, horizon)),
            ("residuals", lambda: plot_residuals(predictions, horizon)),
            ("error_by_basin", lambda: plot_error_by_basin(predictions, horizon)),
            ("storm_lifecycle", lambda: plot_storm_lifecycle(predictions, sid, horizon)),
        ]

    for name, task in tasks:
        try:
            task()
        except Exception as exc:
            # Logged at warning, not swallowed: the original hid a plot that
            # had never once rendered behind a bare except.
            logger.warning(f"  {name} failed: {type(exc).__name__}: {exc}")

    logger.info("=" * 68)
    logger.info(f"Figures written to {FIGURES_DIR}")
    logger.info("=" * 68)


def main(horizon: int = None, model_name: str = "random_forest", sid: str = None) -> None:
    """
    Regenerate figures from saved evaluation results.
    """

    import json
    from src.utils.config import EVALUATION_RESULTS_PATH

    if not EVALUATION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"No results at {EVALUATION_RESULTS_PATH}. Run evaluation first:\n"
            "  python -m src.s5_training.evaluate --horizon all"
        )

    all_results = json.loads(EVALUATION_RESULTS_PATH.read_text())
    generate_all_plots(all_results, horizon=horizon, model_name=model_name, sid=sid)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--sid", default=None, help="Storm for the forecast track figure.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(horizon=args.horizon, model_name=args.model, sid=args.sid)