"""
world_map.py

Geographic visualisation of forecasts against observed tracks.

Two figures:

  world_forecast_tracks.png   sample storms, observed track against forecast
                              positions, on a world map
  world_error_map.png         every test forecast as a point coloured by error,
                              which answers "where does the model fail?"
                              rather than "how much does it fail by"

Basemap
-------
Uses cartopy when it is installed, which gives real coastlines and a proper
projection. Without it the figures still render on a plain lat/lon grid with a
graticule -- readable, just not pretty. Cartopy is a heavy dependency and this
is the only module that wants it, so it is optional:

    pip install cartopy

Antimeridian
------------
Tracks crossing 180 degrees are split before plotting. Without that, a storm
moving from 179E to 179W draws a horizontal line straight across the map -- a
2-degree step rendered as a 358-degree one.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils import get_logger, Timer
from src.utils.config import (
    FIGURES_DIR,
    PREDICTION_PATH,
    PREDICTIONS_DIR,
    DEFAULT_FORECAST_HORIZON,
    FORECAST_HORIZON_HOURS,
)

# Error maps are most informative at short lead times, where the spatial
# pattern is not yet swamped by the sheer magnitude of the error.
DEFAULT_MAP_HORIZONS = (6, 12, 24)

logger = get_logger(__name__)

OBSERVED_COLOR = "#1565C0"
FORECAST_COLOR = "#E53935"
LAND_COLOR = "#EAE7DC"
OCEAN_COLOR = "#F4F8FB"
COAST_COLOR = "#9AA5AD"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "font.size": 10,
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
# Basemap
# ==========================================================

def make_map_axes(fig, position=111, extent=None):
    """
    Create map axes, with coastlines when cartopy is available.

    Parameters
    ----------
    position : int or tuple
        Either a three-digit code (111) or a (rows, columns, index) tuple.
        A tuple must be unpacked -- passing it whole makes matplotlib read it
        as a bounding box.

    Returns
    -------
    (ax, transform) : tuple
        `transform` is the keyword to pass to plotting calls. With cartopy it
        is a PlateCarree CRS; without it, None. Callers pass it through so the
        same plotting code works either way.
    """

    position = tuple(position) if isinstance(position, (tuple, list)) else (position,)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = fig.add_subplot(*position, projection=ccrs.PlateCarree(central_longitude=180))

        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR)
        ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR)
        ax.add_feature(cfeature.COASTLINE, edgecolor=COAST_COLOR, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, edgecolor=COAST_COLOR, linewidth=0.3, alpha=0.5)

        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False

        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()

        return ax, ccrs.PlateCarree()

    except ImportError:
        logger.info(
            "  cartopy not installed — drawing on a plain lat/lon grid. "
            "`pip install cartopy` for coastlines."
        )

        ax = fig.add_subplot(*position)
        ax.set_facecolor(OCEAN_COLOR)
        ax.grid(True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")

        # Equator and tropics give some geographic anchoring without coastlines.
        ax.axhline(0, color=COAST_COLOR, linewidth=0.8, alpha=0.7)
        for latitude in (-23.5, 23.5):
            ax.axhline(latitude, color=COAST_COLOR, linewidth=0.5,
                       alpha=0.5, linestyle=":")

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        if extent:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-60, 60)

        return ax, None


def _plot_kwargs(transform):
    return {"transform": transform} if transform is not None else {}


def split_at_antimeridian(lon: np.ndarray, lat: np.ndarray) -> list[tuple]:
    """
    Break a track wherever it crosses 180 degrees.

    A storm stepping from 179E to 179W has moved 2 degrees. Drawn naively it
    becomes a line straight across the map.
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    breaks = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1
    segments = np.split(np.arange(len(lon)), breaks)

    return [(lon[s], lat[s]) for s in segments if len(s) > 1]


# ==========================================================
# 1. Forecast tracks
# ==========================================================

def choose_storms(
    predictions: pd.DataFrame,
    n_storms: int = 6,
    min_observations: int = 10,
    seed: int = 42,
) -> list[str]:
    """
    Pick a sample of storms with enough observations to draw a track.

    Selected once and reused across horizons, so the per-horizon figures show
    the *same* storms. Sampling independently per horizon would make the
    figures look different for reasons that have nothing to do with lead time.
    """

    counts = predictions["SID"].value_counts()
    eligible = counts[counts >= min_observations].index.tolist()

    if not eligible:
        return []

    rng = np.random.default_rng(seed)

    return list(rng.choice(eligible, size=min(n_storms, len(eligible)), replace=False))


def plot_forecast_tracks(
    predictions: pd.DataFrame,
    n_storms: int = 6,
    horizon: int = None,
    seed: int = 42,
    storms: list[str] = None,
) -> None:
    """
    Observed tracks against forecast positions for a sample of storms.

    Each storm's observed path is drawn as a line; the forecast position for
    every issue time is drawn as a marker, with a thin connector showing the
    error. Densely-spaced connectors mean the model is tracking well; visible
    connectors mean it is not.

    Parameters
    ----------
    storms : list of str, optional
        Storms to plot. Pass the same list at every horizon so the figures can
        be compared directly.
    """

    _setup_style()

    chosen = storms if storms else choose_storms(predictions, n_storms, seed=seed)

    if not chosen:
        logger.info("  (skipping forecast tracks: no storm long enough)")
        return

    chosen = [s for s in chosen if s in set(predictions["SID"])]

    if not chosen:
        logger.info(f"  (skipping {horizon}h tracks: sampled storms absent)")
        return

    subset = predictions[predictions["SID"].isin(chosen)]

    pad = 8
    extent = [
        subset[["CURRENT_LON", "PRED_LON", "TRUE_LON"]].to_numpy().min() - pad,
        subset[["CURRENT_LON", "PRED_LON", "TRUE_LON"]].to_numpy().max() + pad,
        subset[["CURRENT_LAT", "PRED_LAT", "TRUE_LAT"]].to_numpy().min() - pad,
        subset[["CURRENT_LAT", "PRED_LAT", "TRUE_LAT"]].to_numpy().max() + pad,
    ]

    # A sample spanning most of the globe is better shown globally.
    if extent[1] - extent[0] > 200:
        extent = None

    fig = plt.figure(figsize=(15, 8))
    ax, transform = make_map_axes(fig, extent=extent)
    kwargs = _plot_kwargs(transform)

    for sid in chosen:
        storm = predictions[predictions["SID"] == sid].sort_values("ISO_TIME")

        for lon, lat in split_at_antimeridian(
            storm["TRUE_LON"].to_numpy(), storm["TRUE_LAT"].to_numpy()
        ):
            ax.plot(lon, lat, color=OBSERVED_COLOR, linewidth=2, alpha=0.85,
                    zorder=3, **kwargs)

        ax.scatter(storm["PRED_LON"], storm["PRED_LAT"], s=14, marker="x",
                   color=FORECAST_COLOR, alpha=0.85, zorder=4, **kwargs)

        # Error connectors, skipping any that would span the seam.
        for _, row in storm.iterrows():
            if abs(row["PRED_LON"] - row["TRUE_LON"]) > 180:
                continue
            ax.plot(
                [row["TRUE_LON"], row["PRED_LON"]],
                [row["TRUE_LAT"], row["PRED_LAT"]],
                color="gray", linewidth=0.6, alpha=0.5, zorder=2, **kwargs,
            )

        start = storm.iloc[0]
        ax.scatter([start["CURRENT_LON"]], [start["CURRENT_LAT"]], s=60,
                   marker="^", color="#1B5E20", edgecolor="white",
                   linewidth=0.8, zorder=5, **kwargs)

    ax.legend(
        handles=[
            Line2D([], [], color=OBSERVED_COLOR, linewidth=2, label="Observed track"),
            Line2D([], [], color=FORECAST_COLOR, marker="x", linestyle="none",
                   label="Forecast position"),
            Line2D([], [], color="gray", linewidth=0.8, label="Error"),
            Line2D([], [], color="#1B5E20", marker="^", linestyle="none",
                   label="Track start"),
        ],
        loc="lower left", framealpha=0.9,
    )

    mean_error = subset["ERROR_KM"].mean()
    label = f"{horizon}h forecasts" if horizon else "forecasts"

    ax.set_title(
        f"Observed vs forecast tracks — {len(chosen)} storms, {label}\n"
        f"mean track error {mean_error:.0f} km"
    )

    filename = (
        f"world_forecast_tracks_{horizon}h.png" if horizon
        else "world_forecast_tracks.png"
    )
    _save(fig, filename)


# ==========================================================
# 2. Error map
# ==========================================================

def plot_error_map(
    predictions: pd.DataFrame,
    horizon: int = None,
    max_points: int = 8000,
    seed: int = 42,
) -> None:
    """
    Every forecast as a point at its verifying position, coloured by error.

    This is the geographic view the aggregate tables cannot give: whether the
    model fails uniformly, or concentrates its error in particular basins,
    latitudes, or recurvature regions. Clusters of dark points are where to
    look next.
    """

    _setup_style()

    frame = predictions.dropna(subset=["TRUE_LAT", "TRUE_LON", "ERROR_KM"])

    if frame.empty:
        return

    if len(frame) > max_points:
        rng = np.random.default_rng(seed)
        frame = frame.iloc[rng.choice(len(frame), max_points, replace=False)]

    fig = plt.figure(figsize=(15, 8))
    ax, transform = make_map_axes(fig)
    kwargs = _plot_kwargs(transform)

    # Percentile clipping so a handful of extreme misses do not flatten the
    # colour scale for everything else.
    vmax = float(np.percentile(frame["ERROR_KM"], 95))

    scatter = ax.scatter(
        frame["TRUE_LON"], frame["TRUE_LAT"],
        c=frame["ERROR_KM"].clip(upper=vmax),
        cmap="YlOrRd", s=9, alpha=0.6, vmin=0, vmax=vmax,
        edgecolor="none", zorder=3, **kwargs,
    )

    bar = fig.colorbar(scatter, ax=ax, orientation="horizontal",
                       pad=0.06, shrink=0.6, aspect=40)
    bar.set_label(f"Track error (km, clipped at the 95th percentile = {vmax:.0f})")

    label = f" — {horizon}h" if horizon else ""

    ax.set_title(
        f"Where forecast error occurs{label}\n"
        f"{len(frame):,} forecasts, mean {frame['ERROR_KM'].mean():.0f} km, "
        f"median {frame['ERROR_KM'].median():.0f} km"
    )

    filename = (
        f"world_error_map_{horizon}h.png" if horizon else "world_error_map.png"
    )
    _save(fig, filename)



def plot_error_map_grid(
    frames: dict[int, pd.DataFrame],
    max_points: int = 4000,
    seed: int = 42,
) -> None:
    """
    Error maps side by side across lead times.

    Each panel gets its own colour scale, with its own maximum printed in the
    title. A shared scale would be defensible for showing how error grows --
    but 6h errors are an order of magnitude smaller than 72h ones, so a shared
    scale renders the short-lead panels uniformly pale and hides the thing this
    figure exists to show: whether the *spatial pattern* of error is the same
    at every lead time, or whether particular regions only become difficult
    further out.
    """

    _setup_style()

    horizons = sorted(frames)

    if not horizons:
        return

    columns = min(len(horizons), 2)
    rows = int(np.ceil(len(horizons) / columns))

    fig = plt.figure(figsize=(9.5 * columns, 5.2 * rows))
    rng = np.random.default_rng(seed)

    for index, horizon in enumerate(horizons, start=1):
        frame = frames[horizon].dropna(subset=["TRUE_LAT", "TRUE_LON", "ERROR_KM"])

        if frame.empty:
            continue

        if len(frame) > max_points:
            frame = frame.iloc[rng.choice(len(frame), max_points, replace=False)]

        ax, transform = make_map_axes(fig, position=(rows, columns, index))
        kwargs = _plot_kwargs(transform)

        vmax = float(np.percentile(frame["ERROR_KM"], 95))

        scatter = ax.scatter(
            frame["TRUE_LON"], frame["TRUE_LAT"],
            c=frame["ERROR_KM"].clip(upper=vmax),
            cmap="YlOrRd", s=7, alpha=0.6, vmin=0, vmax=vmax,
            edgecolor="none", zorder=3, **kwargs,
        )

        bar = fig.colorbar(scatter, ax=ax, orientation="vertical",
                           pad=0.02, shrink=0.85)
        bar.set_label("km", fontsize=9)

        ax.set_title(
            f"{horizon}h — mean {frame['ERROR_KM'].mean():.0f} km "
            f"(scale to {vmax:.0f} km)",
            fontsize=12,
        )

    fig.suptitle(
        "Geographic distribution of forecast error by lead time",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()

    _save(fig, "world_error_map_by_lead_time.png")


# ==========================================================
# 3. Single storm, multiple lead times
# ==========================================================

def plot_single_storm_map(
    sid: str = None,
    model_name: str = "xgboost",
) -> None:
    """
    One storm's observed track with forecasts at every lead time, on a map.

    The clearest single image of what the model does: where it was issued,
    where the storm went, and where each lead time said it would go.
    """

    _setup_style()

    try:
        from src.s6_inference.predict import forecast_track, list_available_storms
        from src.s5_training.train import load_feature_dataset

        frame = load_feature_dataset()

        if sid is None:
            sid = list_available_storms(limit=5)["SID"].iloc[0]

        track = forecast_track(sid, model_name=model_name, frame=frame)
    except Exception as exc:
        logger.info(f"  (skipping world_storm_forecast: {exc})")
        return

    start_lat = float(track["CURRENT_LAT"].iloc[0])
    start_lon = float(track["CURRENT_LON"].iloc[0])

    lats = np.concatenate([[start_lat], track["TRUE_LAT"], track["PRED_LAT"]])
    lons = np.concatenate([[start_lon], track["TRUE_LON"], track["PRED_LON"]])

    pad = 6
    extent = [lons.min() - pad, lons.max() + pad, lats.min() - pad, lats.max() + pad]

    fig = plt.figure(figsize=(11, 9))
    ax, transform = make_map_axes(fig, extent=extent)
    kwargs = _plot_kwargs(transform)

    observed_lat = np.concatenate([[start_lat], track["TRUE_LAT"].to_numpy()])
    observed_lon = np.concatenate([[start_lon], track["TRUE_LON"].to_numpy()])
    forecast_lat = np.concatenate([[start_lat], track["PRED_LAT"].to_numpy()])
    forecast_lon = np.concatenate([[start_lon], track["PRED_LON"].to_numpy()])

    ax.plot(observed_lon, observed_lat, marker="o", markersize=7, linewidth=2.5,
            color=OBSERVED_COLOR, label="Observed", zorder=4, **kwargs)
    ax.plot(forecast_lon, forecast_lat, marker="X", markersize=9, linewidth=2.5,
            linestyle="--", color=FORECAST_COLOR, label="Forecast", zorder=5, **kwargs)

    for _, row in track.iterrows():
        ax.plot([row["TRUE_LON"], row["PRED_LON"]],
                [row["TRUE_LAT"], row["PRED_LAT"]],
                color="gray", linewidth=1, alpha=0.7, zorder=3, **kwargs)
        ax.annotate(
            f"{int(row['LEAD_HOURS'])}h · {row['ERROR_KM']:.0f}km",
            (row["TRUE_LON"], row["TRUE_LAT"]),
            xytext=(7, 7), textcoords="offset points",
            fontsize=8, color="#444",
            **({"xycoords": transform._as_mpl_transform(ax)} if transform is not None else {}),
        )

    ax.scatter([start_lon], [start_lat], s=170, marker="^", color="#1B5E20",
               edgecolor="white", linewidth=1.5, zorder=6,
               label="Issue time", **kwargs)

    ax.legend(loc="best", framealpha=0.9)
    ax.set_title(
        f"Forecast track — storm {sid}\n"
        f"issued {pd.Timestamp(track['ISO_TIME'].iloc[0]):%Y-%m-%d %H:%M} UTC, "
        f"mean error {track['ERROR_KM'].mean():.0f} km"
    )

    _save(fig, "world_storm_forecast.png")


# ==========================================================
# Main
# ==========================================================

def _load_predictions(horizon: int, model_name: str) -> pd.DataFrame | None:
    """
    Per-row predictions for one horizon.

    Exports go to a per-horizon file. `export_predictions` writes to a single
    path by default, so exporting 12h after 6h would overwrite the 6h rows --
    and a multi-horizon figure would silently end up plotting one horizon
    several times.
    """

    for path in (PREDICTIONS_DIR / f"predictions_{horizon}h.csv", PREDICTION_PATH):
        if not path.exists():
            continue

        frame = pd.read_csv(path, parse_dates=["ISO_TIME"])

        if "LEAD_HOURS" in frame.columns and horizon in set(frame["LEAD_HOURS"]):
            return frame[frame["LEAD_HOURS"] == horizon].reset_index(drop=True)

    try:
        from src.s6_inference.predict import export_predictions

        return export_predictions(
            model_name=model_name,
            horizon=horizon,
            path=PREDICTIONS_DIR / f"predictions_{horizon}h.csv",
        )
    except Exception as exc:
        logger.warning(f"  {horizon}h predictions unavailable: {exc}")
        return None


def main(
    horizons: tuple = DEFAULT_MAP_HORIZONS,
    model_name: str = "xgboost",
    sid: str = None,
    n_storms: int = 6,
    track_horizon: int = None,
) -> None:
    """
    Generate the map figures.

    Parameters
    ----------
    horizons : tuple
        Lead times to map. One error map per horizon, plus a combined panel.
    track_horizon : int, optional
        Horizon for the multi-storm track figure. Defaults to the first
        requested horizon.
    """

    horizons = tuple(horizons)

    logger.info("=" * 72)
    logger.info(
        f"WORLD MAP FIGURES — {model_name}, "
        f"{', '.join(f'{h}h' for h in horizons)}"
    )
    logger.info("=" * 72)

    with Timer("World map figures"):
        frames = {}

        for horizon in horizons:
            frame = _load_predictions(horizon, model_name)

            if frame is not None and not frame.empty:
                frames[horizon] = frame

        if not frames:
            logger.warning("No predictions available for any requested horizon.")
            return

        # One map per horizon.
        for horizon, frame in frames.items():
            try:
                plot_error_map(frame, horizon)
            except Exception as exc:
                logger.warning(f"  {horizon}h error map failed: {exc}")

        # Combined panel, for comparing the spatial pattern across lead times.
        if len(frames) > 1:
            try:
                plot_error_map_grid(frames)
            except Exception as exc:
                logger.warning(f"  error map grid failed: {exc}")

        # One track figure per horizon. The storms are chosen once, from the
        # shortest lead time (which has the most rows), and reused everywhere
        # so the figures differ only by lead time.
        track_horizons = (
            [track_horizon] if track_horizon in frames else sorted(frames)
        )

        storms = choose_storms(frames[sorted(frames)[0]], n_storms)

        for horizon in track_horizons:
            try:
                plot_forecast_tracks(
                    frames[horizon], n_storms, horizon, storms=storms
                )
            except Exception as exc:
                logger.warning(f"  {horizon}h forecast tracks failed: {exc}")

        try:
            plot_single_storm_map(sid, model_name)
        except Exception as exc:
            logger.warning(f"  storm forecast map failed: {exc}")

    logger.info(f"Figures written to {FIGURES_DIR}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World map forecast figures.")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=list(DEFAULT_MAP_HORIZONS),
                        choices=list(FORECAST_HORIZON_HOURS),
                        help="Lead times to map. One figure each, plus a "
                             "combined panel.")
    parser.add_argument("--track-horizon", type=int, default=None,
                        choices=list(FORECAST_HORIZON_HOURS),
                        help="Horizon for the multi-storm track figure.")
    parser.add_argument("--model", default="xgboost",
                        choices=("xgboost", "random_forest"))
    parser.add_argument("--sid", default=None, help="Storm for the single-track map.")
    parser.add_argument("--storms", type=int, default=6,
                        help="Storms to sample for the multi-track map.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(horizons=tuple(args.horizons), model_name=args.model,
         sid=args.sid, n_storms=args.storms,
         track_horizon=args.track_horizon)