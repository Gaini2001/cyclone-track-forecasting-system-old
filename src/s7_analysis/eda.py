"""
eda.py

Exploratory figures for the cyclone dataset.

Replaces track_visualization.py and prediction_visualization.py, both of which
duplicated evaluation figures that evaluation_plots.py now produces correctly.
What the project lacked was the opposite: nothing described the *input*. A
reader arriving at a results table has no idea how many storms, which basins,
which decades, or how hard the prediction problem is before any model touches
it.

Figures
-------
1. track_coverage        sampled storm tracks by basin -- what the data covers
2. dataset_composition   storms per season, basin balance, intensity spread,
                         segment lengths
3. displacement_targets  the distribution the model must actually predict, per
                         horizon. This is the problem statement in one figure:
                         if 24h displacements span 0-10 degrees, that is the
                         range a forecast has to resolve.
4. data_retention        rows surviving each pipeline stage

Usage
-----
    python -m src.s7_analysis.eda
    python -m src.s7_analysis.eda --max-tracks 400
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import get_logger, Timer
from src.utils.metrics import haversine_distance
from src.utils.config import (
    FIGURES_DIR,
    FORECAST_HORIZON_HOURS,
    OBSERVATION_INTERVAL,
    INTENSITY_LABEL_NAMES,
    DELTA_TARGET_COLUMNS,
)

logger = get_logger(__name__)

RANDOM_SEED = 42

BASIN_COLORS = {
    "NA": "#1976D2",   # North Atlantic
    "EP": "#43A047",   # Eastern Pacific
    "WP": "#E53935",   # Western Pacific
    "NI": "#FB8C00",   # North Indian
    "SI": "#8E24AA",   # South Indian
    "SP": "#00ACC1",   # South Pacific
    "SA": "#6D4C41",   # South Atlantic
}

BASIN_NAMES = {
    "NA": "North Atlantic",
    "EP": "East Pacific",
    "WP": "West Pacific",
    "NI": "North Indian",
    "SI": "South Indian",
    "SP": "South Pacific",
    "SA": "South Atlantic",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _save(fig, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"  {path.name}")


# ==========================================================
# 1. Track Coverage
# ==========================================================

def plot_track_coverage(
    df: pd.DataFrame,
    max_tracks: int = 300,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Sampled storm tracks, coloured by basin.

    Establishes coverage before any result is discussed: which basins the model
    has seen, and how unevenly. It also makes the antimeridian visible -- these
    tracks are retained now, where the original pipeline discarded every storm
    crossing 180 degrees and with it most of the West Pacific.
    """

    _setup_style()

    group_key = "SEGMENT_ID" if "SEGMENT_ID" in df.columns else "SID"

    rng = np.random.default_rng(seed)
    ids = df[group_key].unique()

    if len(ids) > max_tracks:
        ids = rng.choice(ids, max_tracks, replace=False)

    fig, ax = plt.subplots(figsize=(15, 8))

    drawn = set()

    for track_id in ids:
        track = df[df[group_key] == track_id].sort_values("ISO_TIME")

        if len(track) < 2:
            continue

        basin = track["BASIN"].iloc[0] if "BASIN" in track.columns else "NA"
        color = BASIN_COLORS.get(basin, "#607D8B")

        # Break the line where a track crosses the antimeridian, so it does not
        # draw a spurious horizontal streak across the whole map.
        lon = track["LON"].to_numpy(dtype=float)
        lat = track["LAT"].to_numpy(dtype=float)
        breaks = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1
        segments = np.split(np.arange(len(lon)), breaks)

        for segment in segments:
            if len(segment) < 2:
                continue

            ax.plot(
                lon[segment], lat[segment],
                linewidth=0.7, alpha=0.5, color=color,
                label=BASIN_NAMES.get(basin, basin) if basin not in drawn else None,
            )
            drawn.add(basin)

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Storm track coverage — {len(ids):,} tracks sampled")
    ax.set_xlim(-180, 180)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "track_coverage.png")


# ==========================================================
# 2. Dataset Composition
# ==========================================================

def plot_dataset_composition(df: pd.DataFrame) -> None:
    """
    Seasons, basins, intensity, and segment lengths.

    Four things a reader needs before believing a results table: how much data,
    from where, spanning what, and in what shape.
    """

    _setup_style()

    group_key = "SEGMENT_ID" if "SEGMENT_ID" in df.columns else "SID"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ---- Storms per season ----
    if "SEASON" in df.columns:
        per_season = df.groupby("SEASON")["SID"].nunique()

        axes[0, 0].bar(per_season.index, per_season.values,
                       color="#1976D2", edgecolor="white")
        axes[0, 0].set_xlabel("Season")
        axes[0, 0].set_ylabel("Storms")
        axes[0, 0].set_title(
            f"Storms per season ({int(per_season.index.min())}–"
            f"{int(per_season.index.max())})"
        )

    # ---- Basin balance ----
    if "BASIN" in df.columns:
        per_basin = df.groupby("BASIN")["SID"].nunique().sort_values(ascending=True)
        colors = [BASIN_COLORS.get(b, "#607D8B") for b in per_basin.index]
        labels = [BASIN_NAMES.get(b, b) for b in per_basin.index]

        axes[0, 1].barh(labels, per_basin.values, color=colors, edgecolor="white")
        axes[0, 1].set_xlabel("Storms")
        axes[0, 1].set_title("Basin coverage")

        for i, value in enumerate(per_basin.values):
            axes[0, 1].text(value, i, f" {value:,}", va="center", fontsize=9)

    # ---- Intensity distribution ----
    if "INTENSITY_CATEGORY" in df.columns:
        counts = df["INTENSITY_CATEGORY"].dropna().astype(int).value_counts().sort_index()
        labels = [INTENSITY_LABEL_NAMES.get(c, str(c)) for c in counts.index]
        palette = ["#81C784", "#AED581", "#FDD835", "#FFB74D",
                   "#FF8A65", "#E57373", "#B71C1C"]

        axes[1, 0].bar(labels, counts.values,
                       color=palette[:len(counts)], edgecolor="white")
        axes[1, 0].set_xlabel("Saffir-Simpson category")
        axes[1, 0].set_ylabel("Observations")
        axes[1, 0].set_title("Intensity distribution")
        axes[1, 0].set_yscale("log")
    elif "WMO_WIND" in df.columns:
        axes[1, 0].hist(df["WMO_WIND"].dropna(), bins=50,
                        color="#FF8A65", edgecolor="white")
        axes[1, 0].set_xlabel("Wind speed (kt)")
        axes[1, 0].set_title("Wind speed distribution")

    # ---- Segment lengths ----
    lengths = df.groupby(group_key).size()

    axes[1, 1].hist(lengths, bins=40, color="#7E57C2", edgecolor="white")
    axes[1, 1].axvline(lengths.median(), color="#E53935", linestyle="--",
                       linewidth=2, label=f"median {int(lengths.median())} obs")
    axes[1, 1].set_xlabel(f"Observations per {group_key.lower()}")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title(
        f"Track lengths (1 obs = {OBSERVATION_INTERVAL}h)"
    )
    axes[1, 1].legend()

    fig.suptitle(
        f"Dataset composition — {len(df):,} observations, "
        f"{df['SID'].nunique():,} storms",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "dataset_composition.png")


# ==========================================================
# 3. Displacement Targets
# ==========================================================

def plot_displacement_targets(df: pd.DataFrame) -> None:
    """
    What the model is actually asked to predict.

    The problem statement as a figure. Displacement magnitude grows with lead
    time, and its spread is the range a forecast has to resolve -- which is why
    predicting displacement gives an honest R^2 while predicting absolute
    position gives 0.999 regardless of skill.
    """

    _setup_style()

    available = [
        h for h in FORECAST_HORIZON_HOURS
        if all(c in df.columns for c in DELTA_TARGET_COLUMNS[h])
    ]

    if not available:
        logger.info("  (skipping displacement_targets: no delta target columns)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(available)))

    # ---- Displacement magnitude by horizon ----
    magnitudes = []

    for horizon, color in zip(available, colors):
        lat_col, lon_col = DELTA_TARGET_COLUMNS[horizon]
        subset = df[[lat_col, lon_col, "LAT", "LON"]].dropna()

        distance = haversine_distance(
            subset["LAT"], subset["LON"],
            subset["LAT"] + subset[lat_col],
            subset["LON"] + subset[lon_col],
        )

        magnitudes.append(distance)

        axes[0].hist(distance, bins=60, alpha=0.55, color=color,
                     label=f"{horizon}h", density=True)

    axes[0].set_xlabel("Displacement (km)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("How far storms move, by lead time")
    axes[0].legend(fontsize=9)

    # ---- Median displacement growth ----
    medians = [np.median(m) for m in magnitudes]
    p90s = [np.percentile(m, 90) for m in magnitudes]

    axes[1].plot(available, medians, marker="o", linewidth=2.5,
                 color="#1976D2", label="median")
    axes[1].plot(available, p90s, marker="s", linewidth=2,
                 linestyle="--", color="#E53935", label="P90")
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_ylabel("Displacement (km)")
    axes[1].set_title("Displacement grows with lead time")
    axes[1].set_xticks(available)
    axes[1].legend()

    # ---- Directional spread at the mid horizon ----
    mid = available[len(available) // 2]
    lat_col, lon_col = DELTA_TARGET_COLUMNS[mid]
    subset = df[[lat_col, lon_col]].dropna()

    rng = np.random.default_rng(RANDOM_SEED)
    sample = subset.iloc[rng.choice(len(subset), min(6000, len(subset)), replace=False)]

    axes[2].scatter(sample[lon_col], sample[lat_col],
                    alpha=0.15, s=6, color="#1976D2")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_xlabel("Δ longitude (°)")
    axes[2].set_ylabel("Δ latitude (°)")
    axes[2].set_title(f"Displacement spread at {mid}h")
    axes[2].set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    _save(fig, "displacement_targets.png")


# ==========================================================
# 4. Retention
# ==========================================================

def plot_data_retention(stages: dict[str, int]) -> None:
    """
    Rows surviving each pipeline stage.

    Worth showing rather than hiding. Most of the loss here is deliberate --
    off-synoptic rows, spur tracks, segment heads and tails -- and a reader who
    can see where it went is more likely to trust what remains.
    """

    _setup_style()

    if not stages:
        return

    names = list(stages)
    values = [stages[n] for n in names]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(names[::-1], values[::-1], color="#1976D2", edgecolor="white")

    initial = values[0] if values else 1

    for bar, value in zip(bars, values[::-1]):
        ax.text(
            bar.get_width(), bar.get_y() + bar.get_height() / 2,
            f"  {value:,} ({value / initial * 100:.0f}%)",
            va="center", fontsize=10,
        )

    ax.set_xlabel("Rows")
    ax.set_title("Data retention through the pipeline")
    ax.set_xlim(0, initial * 1.25)

    fig.tight_layout()
    _save(fig, "data_retention.png")


# ==========================================================
# Summary
# ==========================================================

def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Headline numbers for the README.
    """

    group_key = "SEGMENT_ID" if "SEGMENT_ID" in df.columns else "SID"
    lengths = df.groupby(group_key).size()

    summary = {
        "observations": len(df),
        "storms": int(df["SID"].nunique()),
        "segments": int(df[group_key].nunique()),
        "median_track_length_obs": int(lengths.median()),
        "median_track_length_hours": int(lengths.median() * OBSERVATION_INTERVAL),
    }

    if "SEASON" in df.columns:
        summary["season_range"] = (int(df["SEASON"].min()), int(df["SEASON"].max()))

    if "BASIN" in df.columns:
        summary["basins"] = df.groupby("BASIN")["SID"].nunique().to_dict()

    logger.info("=" * 68)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 68)

    for key, value in summary.items():
        logger.info(f"  {key:<26}: {value}")

    logger.info("=" * 68)

    return summary


# ==========================================================
# Main
# ==========================================================

def main(max_tracks: int = 300) -> dict:
    """
    Generate exploratory figures from the feature dataset.
    """

    from src.s5_training.train import load_feature_dataset

    logger.info("=" * 68)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 68)

    with Timer("EDA"):
        df = load_feature_dataset()
        summary = dataset_summary(df)

        for name, task in (
            ("track_coverage", lambda: plot_track_coverage(df, max_tracks=max_tracks)),
            ("dataset_composition", lambda: plot_dataset_composition(df)),
            ("displacement_targets", lambda: plot_displacement_targets(df)),
        ):
            try:
                task()
            except Exception as exc:
                logger.warning(f"  {name} failed: {type(exc).__name__}: {exc}")

    logger.info(f"Figures written to {FIGURES_DIR}")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory dataset figures.")
    parser.add_argument(
        "--max-tracks", type=int, default=300,
        help="Tracks to sample for the coverage map.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(max_tracks=args.max_tracks)