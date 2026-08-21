"""
metrics.py

Domain-specific evaluation metrics for cyclone track forecasting.

Everything is vectorized and NaN-tolerant: non-finite pairs are masked out and
counted rather than silently poisoning an aggregate.

Contents
--------
Geometry          haversine_distance, compute_bearing, angular_difference,
                  encode_direction, offset_position
Error decomposition
                  along_cross_track_error  -- the NHC-standard split of track
                  error into a direction component and a speed component
Summaries         track_errors, track_error_statistics
Comparison        skill_score, bootstrap_ci, paired_bootstrap_comparison
"""

from __future__ import annotations

import numpy as np

from src.utils.config import EARTH_RADIUS_KM

__all__ = [
    "haversine_distance",
    "compute_bearing",
    "angular_difference",
    "encode_direction",
    "offset_position",
    "along_cross_track_error",
    "track_errors",
    "track_error_statistics",
    "skill_score",
    "bootstrap_ci",
    "paired_bootstrap_comparison",
]


# ==========================================================
# Geometry
# ==========================================================

def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance between two points, in kilometers.

    Correct across the antimeridian: the formula depends on the longitude
    difference only through periodic functions.

    Assumes a spherical Earth. Against a WGS84 geodesic this is accurate to
    roughly 0.3%, i.e. well under a kilometer at the scale of a track error,
    which is far below best-track positional uncertainty.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like
        Coordinates in degrees.

    Returns
    -------
    np.ndarray
        Distance in kilometers.

    Examples
    --------
    >>> float(round(haversine_distance(0.0, 0.0, 0.0, 1.0), 2))
    111.19
    """

    lat1, lon1, lat2, lon2 = map(
        lambda x: np.radians(np.asarray(x, dtype=float)), (lat1, lon1, lat2, lon2)
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    # Rounding can push `a` marginally above 1 for near-antipodal points,
    # which would make arcsin return NaN.
    a = np.clip(a, 0.0, 1.0)

    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))


def compute_bearing(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    undefined_as_nan: bool = True,
) -> np.ndarray:
    """
    Initial great-circle bearing from point 1 to point 2, in degrees [0, 360).

    Parameters
    ----------
    undefined_as_nan : bool
        When the two points coincide the bearing is undefined. `arctan2(0, 0)`
        returns 0, which would report a stationary storm as heading due north
        and inject a fake motion direction into the features. With this flag
        set (the default) such rows return NaN instead.

    Examples
    --------
    >>> float(compute_bearing(0.0, 0.0, 1.0, 0.0))
    0.0
    >>> float(compute_bearing(0.0, 0.0, 0.0, 1.0))
    90.0
    """

    lat1_r, lon1_r, lat2_r, lon2_r = map(
        lambda x: np.radians(np.asarray(x, dtype=float)), (lat1, lon1, lat2, lon2)
    )

    dlon = lon2_r - lon1_r

    x = np.sin(dlon) * np.cos(lat2_r)
    y = (
        np.cos(lat1_r) * np.sin(lat2_r)
        - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y)) % 360.0

    if undefined_as_nan:
        stationary = np.isclose(x, 0.0, atol=1e-12) & np.isclose(y, 0.0, atol=1e-12)
        bearing = np.where(stationary, np.nan, bearing)

    return bearing


def angular_difference(angle_a: np.ndarray, angle_b: np.ndarray) -> np.ndarray:
    """
    Signed smallest difference `a - b`, wrapped to (-180, 180].

    Examples
    --------
    >>> float(angular_difference(10.0, 350.0))
    20.0
    """

    diff = (np.asarray(angle_a, dtype=float) - np.asarray(angle_b, dtype=float) + 180.0)
    return (diff % 360.0) - 180.0


def encode_direction(degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a compass direction as (sin, cos).

    Direction is circular: 359 deg and 1 deg are adjacent, but as raw numbers
    they sit at opposite ends of the range. A tree has to burn many splits to
    reconstruct that adjacency, and a linear model cannot represent it at all.

    Apply this to BEARING and to STORM_DIR before using them as features -- the
    same treatment MONTH and HOUR already receive.

    Returns
    -------
    (sin, cos) : tuple of np.ndarray
    """

    radians = np.radians(np.asarray(degrees, dtype=float))
    return np.sin(radians), np.cos(radians)


def offset_position(
    lat: np.ndarray,
    lon: np.ndarray,
    delta_lat: np.ndarray,
    delta_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a degree displacement to a position, wrapping longitude.

    Used to reconstruct an absolute forecast position from a displacement
    prediction (TARGET_MODE == "displacement") before scoring it.
    """

    new_lat = np.asarray(lat, dtype=float) + np.asarray(delta_lat, dtype=float)
    new_lon = np.asarray(lon, dtype=float) + np.asarray(delta_lon, dtype=float)

    # Latitude cannot wrap; clamp to the pole.
    new_lat = np.clip(new_lat, -90.0, 90.0)
    new_lon = ((new_lon + 180.0) % 360.0) - 180.0

    return new_lat, new_lon


# ==========================================================
# Error Decomposition
# ==========================================================

def along_cross_track_error(
    current_lat: np.ndarray,
    current_lon: np.ndarray,
    actual_lat: np.ndarray,
    actual_lon: np.ndarray,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Decompose track error into along-track and cross-track components.

    This is the standard verification decomposition used in operational
    tropical cyclone forecasting, and it is what turns "142 km of error" into a
    diagnosis. The error vector from the verifying position to the forecast
    position is projected onto the storm's actual direction of motion:

    * cross-track  -- perpendicular to motion. A direction error: the model
      missed a recurvature or turned too early. Positive = forecast lies to the
      right of the observed track.
    * along-track  -- parallel to motion. A speed error. Positive = forecast is
      further along the track than reality, i.e. the model ran the storm too
      fast.

    A model with small cross-track error and a large positive along-track bias
    has a very different problem from one with the reverse, and they call for
    different fixes. The aggregate distance hides both.

    Parameters
    ----------
    current_lat, current_lon : array-like
        Position at forecast issue time (the origin of the motion vector).
    actual_lat, actual_lon : array-like
        Verifying (observed) position at the forecast valid time.
    pred_lat, pred_lon : array-like
        Forecast position at the valid time.

    Returns
    -------
    dict
        along_track_km, cross_track_km, total_km, motion_bearing_deg
    """

    motion_bearing = compute_bearing(
        current_lat, current_lon, actual_lat, actual_lon, undefined_as_nan=True,
    )

    error_distance = haversine_distance(
        actual_lat, actual_lon, pred_lat, pred_lon,
    )

    error_bearing = compute_bearing(
        actual_lat, actual_lon, pred_lat, pred_lon, undefined_as_nan=False,
    )

    # Angle between the error vector and the direction of travel.
    theta = np.radians(angular_difference(error_bearing, motion_bearing))

    return {
        "along_track_km": error_distance * np.cos(theta),
        "cross_track_km": error_distance * np.sin(theta),
        "total_km": error_distance,
        "motion_bearing_deg": motion_bearing,
    }


# ==========================================================
# Summaries
# ==========================================================

def track_errors(
    actual_lat: np.ndarray,
    actual_lon: np.ndarray,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
) -> np.ndarray:
    """
    Per-sample great-circle track error in km.

    Kept separate from the summary so callers can hold the raw array without
    threading a large ndarray through a dict that gets JSON-serialized.
    """

    return haversine_distance(actual_lat, actual_lon, pred_lat, pred_lon)


def track_error_statistics(
    actual_lat: np.ndarray,
    actual_lon: np.ndarray,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    current_lat: np.ndarray = None,
    current_lon: np.ndarray = None,
    reference_errors: np.ndarray = None,
    thresholds_km: tuple = (50, 100, 200, 300),
    include_raw: bool = False,
) -> dict:
    """
    Summary statistics for a set of track forecasts.

    Non-finite pairs are masked and counted rather than propagating NaN through
    every aggregate.

    Parameters
    ----------
    current_lat, current_lon : array-like, optional
        Position at issue time. When supplied, along/cross-track statistics are
        included.
    reference_errors : array-like, optional
        Per-sample errors from a baseline (usually persistence). When supplied,
        a skill score is included.
    thresholds_km : tuple
        Report the fraction of forecasts falling within each distance.
    include_raw : bool
        Attach the per-sample error array under "raw_errors". Leave False for
        anything destined for JSON.

    Returns
    -------
    dict
        All values are plain floats/ints, JSON-safe unless include_raw is set.
    """

    errors = track_errors(actual_lat, actual_lon, pred_lat, pred_lon)

    finite = np.isfinite(errors)
    n_total = int(errors.size)
    n_dropped = int(n_total - finite.sum())

    if finite.sum() == 0:
        return {
            "n_samples": 0,
            "n_dropped": n_dropped,
            "mean_km": float("nan"),
            "median_km": float("nan"),
        }

    valid = errors[finite]

    stats: dict = {
        "n_samples": int(valid.size),
        "n_dropped": n_dropped,
        "mean_km": float(np.mean(valid)),
        "median_km": float(np.median(valid)),
        "std_km": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "p90_km": float(np.percentile(valid, 90)),
        "p95_km": float(np.percentile(valid, 95)),
        "min_km": float(np.min(valid)),
        "max_km": float(np.max(valid)),
    }

    # Fraction of forecasts inside each distance band. Easier to communicate
    # than a percentile: "72% of 24h forecasts land within 100 km".
    for threshold in thresholds_km:
        stats[f"within_{threshold}km_pct"] = float(np.mean(valid <= threshold) * 100.0)

    # ---- Along / cross-track decomposition ----
    if current_lat is not None and current_lon is not None:
        components = along_cross_track_error(
            current_lat, current_lon,
            actual_lat, actual_lon,
            pred_lat, pred_lon,
        )

        along = components["along_track_km"]
        cross = components["cross_track_km"]
        ok = np.isfinite(along) & np.isfinite(cross)

        if ok.sum() > 0:
            stats.update({
                # Signed means are the bias; a large magnitude means a
                # systematic error the model could be corrected for.
                "along_track_bias_km": float(np.mean(along[ok])),
                "cross_track_bias_km": float(np.mean(cross[ok])),
                "along_track_mae_km": float(np.mean(np.abs(along[ok]))),
                "cross_track_mae_km": float(np.mean(np.abs(cross[ok]))),
            })

    # ---- Skill relative to a reference forecast ----
    if reference_errors is not None:
        reference_errors = np.asarray(reference_errors, dtype=float)

        if reference_errors.shape == errors.shape:
            both = finite & np.isfinite(reference_errors)

            if both.sum() > 0:
                stats["skill_vs_reference_pct"] = float(
                    skill_score(errors[both], reference_errors[both]) * 100.0
                )

    if include_raw:
        stats["raw_errors"] = errors

    return stats


# ==========================================================
# Comparison
# ==========================================================

def skill_score(model_errors: np.ndarray, reference_errors: np.ndarray) -> float:
    """
    Fractional error reduction relative to a reference forecast.

        skill = (reference_mean - model_mean) / reference_mean

    Positive means the model beats the reference. 0.15 is a 15% reduction in
    mean track error. Persistence is the conventional reference at short
    horizons; a climatology-and-persistence (CLIPER) blend at longer ones.
    """

    reference_mean = float(np.mean(reference_errors))

    if reference_mean <= 0:
        return float("nan")

    return (reference_mean - float(np.mean(model_errors))) / reference_mean


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for a statistic.

    Lets you report "142 km (95% CI: 138-146)" instead of a bare point
    estimate, which is the difference between a number and a result.

    Returns
    -------
    (lower, upper) : tuple of float
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = values.size

    estimates = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        estimates[i] = statistic(sample)

    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))

    return lower, upper


def paired_bootstrap_comparison(
    model_errors: np.ndarray,
    reference_errors: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test that a model genuinely beats a reference.

    Both forecasts are evaluated on the same rows, so the comparison must be
    paired -- resampling the two sets independently would inflate the variance
    and understate significance.

    `p_value` is the bootstrap fraction in which the model failed to improve on
    the reference. Below 0.05 means the improvement is unlikely to be sampling
    noise.

    Returns
    -------
    dict
        mean_difference_km (reference minus model; positive = model is better),
        ci_lower_km, ci_upper_km, skill_pct, skill_ci_lower_pct,
        skill_ci_upper_pct, p_value, n_samples
    """

    model_errors = np.asarray(model_errors, dtype=float)
    reference_errors = np.asarray(reference_errors, dtype=float)

    if model_errors.shape != reference_errors.shape:
        raise ValueError(
            "Paired comparison requires equal-length error arrays "
            f"(got {model_errors.shape} and {reference_errors.shape})."
        )

    both = np.isfinite(model_errors) & np.isfinite(reference_errors)
    model_errors = model_errors[both]
    reference_errors = reference_errors[both]

    if model_errors.size == 0:
        return {"n_samples": 0, "p_value": float("nan")}

    rng = np.random.default_rng(seed)
    n = model_errors.size

    differences = np.empty(n_boot, dtype=float)
    skills = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # same indices for both -> paired
        model_mean = float(np.mean(model_errors[idx]))
        reference_mean = float(np.mean(reference_errors[idx]))

        differences[i] = reference_mean - model_mean
        skills[i] = (
            (reference_mean - model_mean) / reference_mean
            if reference_mean > 0 else np.nan
        )

    observed_difference = float(np.mean(reference_errors) - np.mean(model_errors))

    return {
        "n_samples": int(n),
        "mean_difference_km": observed_difference,
        "ci_lower_km": float(np.percentile(differences, 100 * alpha / 2)),
        "ci_upper_km": float(np.percentile(differences, 100 * (1 - alpha / 2))),
        "skill_pct": float(skill_score(model_errors, reference_errors) * 100.0),
        "skill_ci_lower_pct": float(np.nanpercentile(skills, 100 * alpha / 2) * 100.0),
        "skill_ci_upper_pct": float(
            np.nanpercentile(skills, 100 * (1 - alpha / 2)) * 100.0
        ),
        "p_value": float(np.mean(differences <= 0)),
    }