"""
tests/test_metrics.py

Unit tests for the cyclone track forecasting metrics.

Geometry is the right place to start testing: the correct answers are known
independently of the codebase, so these tests catch real regressions rather
than just pinning current behaviour.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest

from src.utils.metrics import (
    haversine_distance,
    compute_bearing,
    angular_difference,
    encode_direction,
    offset_position,
    along_cross_track_error,
    track_error_statistics,
    skill_score,
    paired_bootstrap_comparison,
)

# One degree of latitude at the equator, for R = 6371 km.
KM_PER_DEGREE = 111.19


# ==========================================================
# Haversine Distance
# ==========================================================

class TestHaversineDistance:

    def test_zero_distance(self):
        assert haversine_distance(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)

    def test_one_degree_latitude(self):
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(KM_PER_DEGREE, abs=0.1)

    def test_one_degree_longitude_at_equator(self):
        d = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(KM_PER_DEGREE, abs=0.1)

    def test_longitude_degree_shrinks_with_latitude(self):
        """A degree of longitude at 60N is half its equatorial length."""
        equator = haversine_distance(0.0, 0.0, 0.0, 1.0)
        high_lat = haversine_distance(60.0, 0.0, 60.0, 1.0)
        assert high_lat == pytest.approx(equator * 0.5, rel=0.01)

    def test_known_city_pair(self):
        """London to Paris is about 343 km."""
        d = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
        assert d == pytest.approx(343.0, abs=5.0)

    def test_antipodal(self):
        d = haversine_distance(0.0, 0.0, 0.0, 180.0)
        assert d == pytest.approx(np.pi * 6371.0, rel=1e-6)
        assert np.isfinite(d), "clipping must prevent NaN at the antipode"

    def test_crosses_antimeridian(self):
        """179E to 179W is 2 degrees apart, not 358."""
        d = haversine_distance(0.0, 179.0, 0.0, -179.0)
        assert d == pytest.approx(2 * KM_PER_DEGREE, abs=0.5)

    def test_symmetric(self):
        forward = haversine_distance(10.0, 20.0, 30.0, 40.0)
        backward = haversine_distance(30.0, 40.0, 10.0, 20.0)
        assert forward == pytest.approx(backward)

    def test_vectorized(self):
        lat1 = np.array([0.0, 10.0, 20.0])
        result = haversine_distance(lat1, np.zeros(3), lat1 + 1.0, np.zeros(3))
        assert result.shape == (3,)
        assert np.allclose(result, KM_PER_DEGREE, atol=0.5)


# ==========================================================
# Bearing
# ==========================================================

class TestComputeBearing:

    @pytest.mark.parametrize("dlat, dlon, expected", [
        (1.0, 0.0, 0.0),      # north
        (0.0, 1.0, 90.0),     # east
        (-1.0, 0.0, 180.0),   # south
        (0.0, -1.0, 270.0),   # west
    ])
    def test_cardinal_directions(self, dlat, dlon, expected):
        b = compute_bearing(0.0, 0.0, dlat, dlon)
        assert b == pytest.approx(expected, abs=0.01)

    def test_northeast_is_about_45(self):
        b = compute_bearing(0.0, 0.0, 1.0, 1.0)
        assert b == pytest.approx(45.0, abs=0.5)

    def test_range_is_zero_to_360(self):
        rng = np.random.default_rng(0)
        lat1, lon1 = rng.uniform(-60, 60, 200), rng.uniform(-180, 180, 200)
        lat2, lon2 = rng.uniform(-60, 60, 200), rng.uniform(-180, 180, 200)

        b = compute_bearing(lat1, lon1, lat2, lon2)
        assert np.all((b >= 0) & (b < 360))

    def test_identical_points_return_nan(self):
        """A stationary storm has no heading -- it is not travelling north."""
        assert np.isnan(compute_bearing(15.0, 120.0, 15.0, 120.0))

    def test_identical_points_zero_when_disabled(self):
        b = compute_bearing(15.0, 120.0, 15.0, 120.0, undefined_as_nan=False)
        assert b == pytest.approx(0.0)


# ==========================================================
# Angular Helpers
# ==========================================================

class TestAngularHelpers:

    @pytest.mark.parametrize("a, b, expected", [
        (10.0, 350.0, 20.0),
        (350.0, 10.0, -20.0),
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 90.0),
    ])
    def test_angular_difference_wraps(self, a, b, expected):
        assert angular_difference(a, b) == pytest.approx(expected, abs=1e-9)

    def test_angular_difference_bounded(self):
        rng = np.random.default_rng(1)
        a, b = rng.uniform(0, 360, 500), rng.uniform(0, 360, 500)
        diff = angular_difference(a, b)
        assert np.all(diff > -180.0001) and np.all(diff <= 180.0001)

    def test_encode_direction_is_continuous_across_zero(self):
        """359 deg and 1 deg must be close in encoded space."""
        sin_a, cos_a = encode_direction(359.0)
        sin_b, cos_b = encode_direction(1.0)

        distance = np.hypot(sin_a - sin_b, cos_a - cos_b)
        assert distance < 0.05, "circular encoding should keep 359 and 1 adjacent"

    def test_encode_direction_unit_circle(self):
        sin_v, cos_v = encode_direction(np.array([0.0, 90.0, 180.0, 270.0]))
        assert np.allclose(sin_v ** 2 + cos_v ** 2, 1.0)


# ==========================================================
# Position Offset
# ==========================================================

class TestOffsetPosition:

    def test_simple_offset(self):
        lat, lon = offset_position(10.0, 20.0, 1.0, -2.0)
        assert lat == pytest.approx(11.0)
        assert lon == pytest.approx(18.0)

    def test_longitude_wraps(self):
        _, lon = offset_position(0.0, 179.0, 0.0, 3.0)
        assert lon == pytest.approx(-178.0)

    def test_latitude_clamped_at_pole(self):
        lat, _ = offset_position(89.0, 0.0, 5.0, 0.0)
        assert lat == pytest.approx(90.0)


# ==========================================================
# Along / Cross Track Decomposition
# ==========================================================

class TestAlongCrossTrack:

    def test_forecast_too_fast_is_positive_along_track(self):
        """
        Storm moves north from (0,0) to (1,0). Forecast overshoots to (1.5,0):
        pure speed error, no direction error.
        """
        result = along_cross_track_error(
            current_lat=0.0, current_lon=0.0,
            actual_lat=1.0, actual_lon=0.0,
            pred_lat=1.5, pred_lon=0.0,
        )

        assert result["along_track_km"] == pytest.approx(0.5 * KM_PER_DEGREE, abs=1.0)
        assert abs(result["cross_track_km"]) < 1.0

    def test_forecast_too_slow_is_negative_along_track(self):
        result = along_cross_track_error(
            current_lat=0.0, current_lon=0.0,
            actual_lat=1.0, actual_lon=0.0,
            pred_lat=0.5, pred_lon=0.0,
        )

        assert result["along_track_km"] == pytest.approx(-0.5 * KM_PER_DEGREE, abs=1.0)

    def test_forecast_right_of_track_is_positive_cross_track(self):
        """
        Storm moves north; forecast lands to the east, i.e. to the right of
        the direction of travel: pure direction error.
        """
        result = along_cross_track_error(
            current_lat=0.0, current_lon=0.0,
            actual_lat=1.0, actual_lon=0.0,
            pred_lat=1.0, pred_lon=0.5,
        )

        assert result["cross_track_km"] > 0
        assert abs(result["along_track_km"]) < 1.0

    def test_components_recover_total(self):
        """along^2 + cross^2 == total^2 for small errors."""
        result = along_cross_track_error(
            current_lat=15.0, current_lon=120.0,
            actual_lat=16.0, actual_lon=121.0,
            pred_lat=16.3, pred_lon=121.4,
        )

        recovered = np.hypot(result["along_track_km"], result["cross_track_km"])
        assert recovered == pytest.approx(result["total_km"], rel=1e-6)

    def test_perfect_forecast_has_zero_components(self):
        result = along_cross_track_error(
            current_lat=15.0, current_lon=120.0,
            actual_lat=16.0, actual_lon=121.0,
            pred_lat=16.0, pred_lon=121.0,
        )

        assert result["total_km"] == pytest.approx(0.0, abs=1e-9)


# ==========================================================
# Track Error Statistics
# ==========================================================

class TestTrackErrorStatistics:

    def test_perfect_forecast(self):
        lat = np.array([10.0, 11.0, 12.0])
        lon = np.array([20.0, 21.0, 22.0])

        stats = track_error_statistics(lat, lon, lat, lon)

        assert stats["mean_km"] == pytest.approx(0.0)
        assert stats["n_samples"] == 3
        assert stats["within_50km_pct"] == pytest.approx(100.0)

    def test_nan_predictions_are_masked_not_propagated(self):
        actual_lat = np.array([10.0, 11.0, 12.0])
        actual_lon = np.array([20.0, 21.0, 22.0])
        pred_lat = np.array([10.0, np.nan, 12.0])
        pred_lon = np.array([20.0, 21.0, 22.0])

        stats = track_error_statistics(actual_lat, actual_lon, pred_lat, pred_lon)

        assert stats["n_samples"] == 2
        assert stats["n_dropped"] == 1
        assert np.isfinite(stats["mean_km"])

    def test_empty_input_does_not_raise(self):
        empty = np.array([])
        stats = track_error_statistics(empty, empty, empty, empty)
        assert stats["n_samples"] == 0

    def test_threshold_fractions(self):
        actual_lat = np.zeros(4)
        actual_lon = np.zeros(4)
        # ~111 km, ~222 km, ~333 km, ~444 km away
        pred_lat = np.array([1.0, 2.0, 3.0, 4.0])
        pred_lon = np.zeros(4)

        stats = track_error_statistics(actual_lat, actual_lon, pred_lat, pred_lon)

        assert stats["within_200km_pct"] == pytest.approx(25.0)
        assert stats["within_300km_pct"] == pytest.approx(50.0)

    def test_median_and_percentiles_ordered(self):
        rng = np.random.default_rng(2)
        actual_lat, actual_lon = np.zeros(500), np.zeros(500)
        pred_lat = rng.normal(0, 1, 500)
        pred_lon = rng.normal(0, 1, 500)

        stats = track_error_statistics(actual_lat, actual_lon, pred_lat, pred_lon)

        assert stats["median_km"] <= stats["p90_km"] <= stats["p95_km"]
        assert stats["max_km"] >= stats["p95_km"]


# ==========================================================
# Skill and Significance
# ==========================================================

class TestSkill:

    def test_skill_score_halving_error(self):
        model = np.full(100, 50.0)
        reference = np.full(100, 100.0)
        assert skill_score(model, reference) == pytest.approx(0.5)

    def test_no_skill_when_identical(self):
        errors = np.full(100, 75.0)
        assert skill_score(errors, errors) == pytest.approx(0.0)

    def test_negative_skill_when_worse(self):
        assert skill_score(np.full(50, 150.0), np.full(50, 100.0)) < 0

    def test_paired_bootstrap_detects_real_improvement(self):
        rng = np.random.default_rng(3)
        reference = rng.gamma(2.0, 60.0, 2000)
        model = reference * 0.8  # a genuine 20% reduction

        result = paired_bootstrap_comparison(model, reference, n_boot=300)

        assert result["skill_pct"] == pytest.approx(20.0, abs=1.0)
        assert result["p_value"] < 0.05
        assert result["ci_lower_km"] > 0

    def test_paired_bootstrap_finds_no_difference_in_noise(self):
        rng = np.random.default_rng(4)
        reference = rng.gamma(2.0, 60.0, 2000)
        model = rng.gamma(2.0, 60.0, 2000)  # same distribution, independent

        result = paired_bootstrap_comparison(model, reference, n_boot=300)

        assert result["p_value"] > 0.05

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paired_bootstrap_comparison(np.zeros(10), np.zeros(5))