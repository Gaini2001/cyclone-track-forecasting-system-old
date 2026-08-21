"""
tests/test_features.py

Tests for the feature engineering pipeline.

What changed from the original
------------------------------
The previous suite mostly asserted that columns existed:

    for col in expected:
        assert col in df.columns

That catches a typo in a column name. It cannot catch a lag that crosses a
storm boundary, a target that is not actually h hours ahead, or a longitude
difference that reads 358 degrees at the antimeridian -- which were the bugs
actually present.

It also verified targets by re-implementing the code under test:

    assert df["TARGET_LAT_6H"].iloc[0] == df["LAT"].iloc[1]

which asserts that shift(-1) returns the next row. True by definition, and it
passes just as happily when rows are 3 hours apart and the "6h target" is
really a 3h target. The tests below check the target against ISO_TIME instead,
so the property being tested is the one that matters.
"""

import numpy as np
import pandas as pd
import pytest

from src.s3_features.feature_engineering import (
    add_temporal_features,
    add_lag_features,
    add_motion_features,
    add_velocity_features,
    add_turning_features,
    add_rolling_features,
    add_acceleration_features,
    add_storm_context_features,
    create_target_variables,
    wrapped_degree_diff,
)
from src.utils.config import FORECAST_HORIZONS, OBSERVATION_INTERVAL

TOLERANCE = 1e-9


# ==========================================================
# Wrapped Differences
# ==========================================================

class TestWrappedDegreeDiff:
    """The primitive that makes discarding dateline storms unnecessary."""

    def test_simple_difference(self):
        assert wrapped_degree_diff(10.0, 4.0) == pytest.approx(6.0)

    def test_eastward_across_antimeridian(self):
        """179E to 179W is +2 degrees of eastward travel, not -358."""
        assert wrapped_degree_diff(-179.0, 179.0) == pytest.approx(2.0)

    def test_westward_across_antimeridian(self):
        assert wrapped_degree_diff(179.0, -179.0) == pytest.approx(-2.0)

    def test_result_always_bounded(self, rng):
        a, b = rng.uniform(-180, 180, 1000), rng.uniform(-180, 180, 1000)
        diff = wrapped_degree_diff(a, b)
        assert np.all(diff > -180.0001) and np.all(diff <= 180.0001)


# ==========================================================
# Temporal
# ==========================================================

class TestTemporalFeatures:

    def test_components_match_timestamps(self, single_storm):
        df = add_temporal_features(single_storm)

        assert (df["MONTH"] == df["ISO_TIME"].dt.month).all()
        assert (df["HOUR"] == df["ISO_TIME"].dt.hour).all()
        assert (df["DAY_OF_YEAR"] == df["ISO_TIME"].dt.dayofyear).all()

    def test_cyclic_encodings_lie_on_unit_circle(self, single_storm):
        df = add_temporal_features(single_storm)

        assert np.allclose(df["MONTH_SIN"] ** 2 + df["MONTH_COS"] ** 2, 1.0)
        assert np.allclose(df["HOUR_SIN"] ** 2 + df["HOUR_COS"] ** 2, 1.0)

    def test_december_and_january_are_adjacent(self):
        """The reason for the encoding: month 12 and month 1 are neighbours."""
        frame = pd.DataFrame({
            "ISO_TIME": pd.to_datetime(["2019-12-15", "2020-01-15", "2019-06-15"])
        })
        df = add_temporal_features(frame)

        def distance(i, j):
            return np.hypot(
                df["MONTH_SIN"].iloc[i] - df["MONTH_SIN"].iloc[j],
                df["MONTH_COS"].iloc[i] - df["MONTH_COS"].iloc[j],
            )

        assert distance(0, 1) < distance(0, 2)

    def test_no_nans(self, single_storm):
        df = add_temporal_features(single_storm)

        for column in ("MONTH", "DAY", "HOUR", "DAY_OF_YEAR",
                       "MONTH_SIN", "HOUR_COS", "DOY_SIN"):
            assert df[column].isna().sum() == 0


# ==========================================================
# Lags
# ==========================================================

class TestLagFeatures:

    def test_lag_equals_earlier_value(self, multi_storm):
        df = add_lag_features(multi_storm)

        for lag in (1, 2, 3):
            expected = df.groupby("SEGMENT_ID")["LAT"].shift(lag)
            both = df[f"LAT_LAG_{lag}"].notna() & expected.notna()

            assert np.allclose(
                df.loc[both, f"LAT_LAG_{lag}"], expected[both], atol=TOLERANCE
            )

    def test_lags_never_cross_a_storm_boundary(self, multi_storm):
        """
        The failure mode a single-storm fixture cannot detect: an ungrouped
        shift pulls the previous storm's last position into the next storm's
        first row, and every column-existence test still passes.
        """

        df = add_lag_features(multi_storm)

        first_rows = df.groupby("SEGMENT_ID").head(1)

        assert first_rows["LAT_LAG_1"].isna().all()
        assert first_rows["WIND_LAG_1"].isna().all()

    def test_lags_do_not_span_a_time_gap(self, gapped_storm):
        """
        After cleaning removes rows, consecutive rows may be 30 hours apart.
        Grouping by SEGMENT_ID must break the lag chain there; grouping by SID
        would carry a value across the hole and label it a 6-hour lag.
        """

        df = add_lag_features(gapped_storm)

        populated = df["LAT_LAG_1"].notna()
        elapsed = df.groupby("SEGMENT_ID")["ISO_TIME"].diff()

        assert (
            elapsed[populated] == pd.Timedelta(hours=OBSERVATION_INTERVAL)
        ).all()

    def test_nan_only_at_segment_head(self, multi_storm):
        df = add_lag_features(multi_storm)
        position = df.groupby("SEGMENT_ID").cumcount()

        for lag in (1, 2, 3):
            unexpected = df[f"LAT_LAG_{lag}"].isna() & (position >= lag)
            assert unexpected.sum() == 0


# ==========================================================
# Motion
# ==========================================================

class TestMotionFeatures:

    def test_delta_matches_difference(self, multi_storm):
        df = add_motion_features(add_lag_features(multi_storm))
        valid = df["DELTA_LAT"].notna()

        assert np.allclose(
            df.loc[valid, "DELTA_LAT"],
            (df["LAT"] - df["LAT_LAG_1"])[valid],
            atol=TOLERANCE,
        )

    def test_dateline_crossing_gives_small_delta(self, dateline_storm):
        """
        A storm moving 1.2 degrees per step should never show a 358 degree
        jump, however it crosses the antimeridian.
        """

        df = add_motion_features(add_lag_features(dateline_storm))

        assert df["LON"].min() < -170 and df["LON"].max() > 170, \
            "fixture should actually cross the dateline"

        assert df["DELTA_LON"].abs().max() < 5.0

    def test_dateline_distance_is_physical(self, dateline_storm):
        df = add_motion_features(add_lag_features(dateline_storm))

        # A 6-hour step at these speeds is a few hundred km at most.
        assert df["MOVEMENT_DISTANCE"].max() < 500

    def test_segment_head_is_nan_not_zero(self, multi_storm):
        """
        A zero here would assert the storm was stationary and heading due
        north, and those fabricated rows survive into training.
        """

        df = add_motion_features(add_lag_features(multi_storm))
        first_rows = df.groupby("SEGMENT_ID").head(1)

        assert first_rows["MOVEMENT_DISTANCE"].isna().all()
        assert first_rows["BEARING"].isna().all()

    def test_bearing_encoding_is_circular(self, multi_storm):
        df = add_motion_features(add_lag_features(multi_storm))
        valid = df["BEARING"].notna()

        assert np.allclose(
            df.loc[valid, "BEARING_SIN"] ** 2 + df.loc[valid, "BEARING_COS"] ** 2,
            1.0,
        )

    def test_distance_non_negative(self, multi_storm):
        df = add_motion_features(add_lag_features(multi_storm))
        assert (df["MOVEMENT_DISTANCE"].dropna() >= 0).all()


# ==========================================================
# Velocity & Turning
# ==========================================================

class TestVelocityFeatures:

    def _pipeline(self, frame):
        return add_velocity_features(add_motion_features(add_lag_features(frame)))

    def test_speed_matches_distance_over_time(self, multi_storm):
        df = self._pipeline(multi_storm)
        valid = df["TRANSLATION_SPEED"].notna()

        assert np.allclose(
            df.loc[valid, "TRANSLATION_SPEED"],
            df.loc[valid, "MOVEMENT_DISTANCE"] / OBSERVATION_INTERVAL,
            atol=TOLERANCE,
        )

    def test_northward_motion_is_positive_v(self, single_storm):
        """Fixture moves north, so the meridional component must be positive."""
        df = self._pipeline(single_storm)
        assert df["VELOCITY_V"].dropna().mean() > 0

    def test_speeds_are_physically_plausible(self, multi_storm):
        df = self._pipeline(multi_storm)
        # Tropical cyclones translate at roughly 5-70 km/h.
        assert df["TRANSLATION_SPEED"].dropna().max() < 150


class TestTurningFeatures:

    def _pipeline(self, frame):
        return add_turning_features(
            add_velocity_features(add_motion_features(add_lag_features(frame)))
        )

    def test_straight_track_has_near_zero_turning(self, single_storm):
        """The fixture travels a constant heading; turning should vanish."""
        df = self._pipeline(single_storm)
        assert df["BEARING_CHANGE_1"].abs().max() < 1.0

    def test_turning_is_wrapped(self, multi_storm):
        """A turn from 350 to 10 degrees is +20, not -340."""
        df = self._pipeline(multi_storm)
        values = df["BEARING_CHANGE_1"].dropna()

        assert values.between(-180, 180).all()

    def test_turn_magnitude_is_absolute(self, multi_storm):
        df = self._pipeline(multi_storm)
        valid = df[["BEARING_CHANGE_1", "TURN_MAGNITUDE"]].dropna()

        assert np.allclose(
            valid["TURN_MAGNITUDE"], valid["BEARING_CHANGE_1"].abs(), atol=TOLERANCE
        )


class TestRollingFeatures:

    def _pipeline(self, frame):
        return add_rolling_features(
            add_velocity_features(add_motion_features(add_lag_features(frame)))
        )

    def test_rolling_mean_matches_manual_window(self, single_storm):
        df = self._pipeline(single_storm)

        column = "DELTA_LAT_MEAN_24H"
        valid = df[column].notna()

        expected = df["DELTA_LAT"].rolling(4, min_periods=4).mean()

        assert np.allclose(df.loc[valid, column], expected[valid], atol=TOLERANCE)

    def test_rolling_does_not_cross_segments(self, gapped_storm):
        df = self._pipeline(gapped_storm)

        # The first rows of the second segment cannot have a full window.
        second = df[df["SEGMENT_ID"].str.endswith("S02")]

        if not second.empty:
            assert second["DELTA_LAT_MEAN_24H"].head(4).isna().all()


# ==========================================================
# Context
# ==========================================================

class TestStormContextFeatures:

    def test_storm_age_is_hours_not_row_count(self, single_storm):
        """
        Derived from timestamps: a row index undercounts age wherever cleaning
        removed rows.
        """

        df = add_storm_context_features(single_storm)

        assert df["STORM_AGE"].iloc[0] == pytest.approx(0.0)
        assert df["STORM_AGE"].iloc[1] == pytest.approx(OBSERVATION_INTERVAL)
        assert df["STORM_AGE"].is_monotonic_increasing

    def test_storm_age_respects_gaps(self, gapped_storm):
        df = add_storm_context_features(gapped_storm)

        elapsed = (
            df["ISO_TIME"] - df.groupby("SID")["ISO_TIME"].transform("min")
        ).dt.total_seconds() / 3600.0

        assert np.allclose(df["STORM_AGE"], elapsed, atol=TOLERANCE)

    def test_intensity_categories_in_range(self, multi_storm):
        df = add_storm_context_features(multi_storm)
        assert df["INTENSITY_CATEGORY"].dropna().between(0, 6).all()

    def test_extreme_wind_is_strongest_not_weakest(self):
        """
        Regression test for the finite upper bin. With a top edge of 300, a
        wind above it fell outside every bin, became NaN, and was then filled
        as category 0 -- labelling the most intense storm as the weakest.
        """

        frame = pd.DataFrame({
            "SID": ["X"] * 3,
            "SEGMENT_ID": ["X_S01"] * 3,
            "ISO_TIME": pd.date_range("2019-09-01", periods=3, freq="6h"),
            "LAT": [15.0, 15.5, 16.0],
            "LON": [-60.0, -60.5, -61.0],
            "WMO_WIND": [30.0, 150.0, 400.0],
            "BASIN": ["NA"] * 3,
            "DIST2LAND": [100.0, 90.0, 80.0],
        })

        df = add_storm_context_features(frame)

        assert df["INTENSITY_CATEGORY"].iloc[0] == 0
        assert df["INTENSITY_CATEGORY"].iloc[2] == 6

    def test_basin_is_encoded(self, multi_storm):
        df = add_storm_context_features(multi_storm)

        assert "BASIN_CODE" in df.columns
        assert df["BASIN_CODE"].notna().all(), "every fixture basin should map"
        assert df["BASIN_CODE"].nunique() == multi_storm["BASIN"].nunique()

    def test_north_atlantic_basin_survives_parsing(self):
        """
        'NA' is pandas' default missing-value token and IBTrACS' code for the
        North Atlantic. Read carelessly, every Atlantic storm loses its basin.
        """

        frame = pd.DataFrame({
            "SID": ["X"] * 2,
            "SEGMENT_ID": ["X_S01"] * 2,
            "ISO_TIME": pd.date_range("2019-09-01", periods=2, freq="6h"),
            "LAT": [15.0, 15.5],
            "LON": [-60.0, -60.5],
            "WMO_WIND": [50.0, 55.0],
            "BASIN": ["NA", "NA"],
            "DIST2LAND": [100.0, 90.0],
        })

        df = add_storm_context_features(frame)
        assert df["BASIN_CODE"].notna().all()


# ==========================================================
# Targets
# ==========================================================

class TestTargetVariables:

    def test_all_horizons_present(self, multi_storm):
        df = create_target_variables(multi_storm)

        for horizon in FORECAST_HORIZONS:
            for prefix in ("TARGET_LAT", "TARGET_LON", "TARGET_DLAT", "TARGET_DLON"):
                assert f"{prefix}_{horizon}H" in df.columns

    @pytest.mark.parametrize("horizon", [6, 12, 24, 48, 72])
    def test_target_is_the_position_h_hours_later(self, multi_storm, horizon):
        """
        The central test of the whole pipeline.

        Verified against ISO_TIME by a self-join, not by re-applying the shift.
        A test that recomputes shift(-n) confirms only that shift works; this
        one fails if a row shift does not correspond to `horizon` hours, which
        is exactly what happens on 3-hourly IBTrACS rows.
        """

        df = create_target_variables(multi_storm)

        positions = df[["SID", "ISO_TIME", "LAT", "LON"]].rename(
            columns={"ISO_TIME": "VALID_TIME", "LAT": "TRUE_LAT", "LON": "TRUE_LON"}
        )

        probe = df[["SID", "ISO_TIME", f"TARGET_LAT_{horizon}H",
                    f"TARGET_LON_{horizon}H"]].dropna()
        probe = probe.assign(
            VALID_TIME=probe["ISO_TIME"] + pd.Timedelta(hours=horizon)
        )

        merged = probe.merge(positions, on=["SID", "VALID_TIME"], how="left")

        assert len(merged) > 0, "no targets to verify"
        assert merged["TRUE_LAT"].notna().all(), \
            "a target exists where no observation does"

        assert np.allclose(
            merged[f"TARGET_LAT_{horizon}H"], merged["TRUE_LAT"], atol=TOLERANCE
        )
        assert np.allclose(
            merged[f"TARGET_LON_{horizon}H"], merged["TRUE_LON"], atol=TOLERANCE
        )

    def test_displacement_reconstructs_absolute_position(self, multi_storm):
        df = create_target_variables(multi_storm)
        valid = df["TARGET_DLAT_24H"].notna()

        assert np.allclose(
            (df["LAT"] + df["TARGET_DLAT_24H"])[valid],
            df["TARGET_LAT_24H"][valid],
            atol=TOLERANCE,
        )

        reconstructed = ((df["LON"] + df["TARGET_DLON_24H"] + 180) % 360) - 180
        difference = wrapped_degree_diff(reconstructed[valid],
                                         df["TARGET_LON_24H"][valid])

        assert np.abs(difference).max() < 1e-6

    def test_displacement_is_wrapped_at_the_dateline(self, dateline_storm):
        df = create_target_variables(dateline_storm)
        assert df["TARGET_DLON_24H"].abs().max() < 30

    def test_targets_never_cross_a_segment_boundary(self, multi_storm):
        df = create_target_variables(multi_storm)

        length = df.groupby("SEGMENT_ID")["ISO_TIME"].transform("size")
        position = df.groupby("SEGMENT_ID").cumcount()
        from_end = length - position - 1

        for horizon, steps in FORECAST_HORIZONS.items():
            column = f"TARGET_LAT_{horizon}H"
            assert df.loc[from_end < steps, column].isna().all(), \
                f"{column} is populated where no future observation exists"

    def test_tail_rows_have_no_target(self, multi_storm):
        df = create_target_variables(multi_storm)

        # groupby().last() returns the last NON-NULL value, so it can never
        # see the trailing NaN this test is about. Take the final row instead.
        final_rows = df.groupby("SEGMENT_ID").tail(1)

        assert final_rows["TARGET_LAT_6H"].isna().all()


# ==========================================================
# Full Pipeline
# ==========================================================

class TestFullPipeline:

    def test_no_infinities(self, feature_frame):
        numeric = feature_frame.select_dtypes(include=np.number)
        assert not np.isinf(numeric).any().any()

    def test_row_count_preserved(self, feature_frame, clean_frame):
        """Feature engineering adds columns; it must not add or drop rows."""
        assert len(feature_frame) == len(clean_frame)

    def test_core_inputs_complete(self, feature_frame):
        for column in ("LAT", "LON", "WMO_WIND", "WMO_PRES"):
            assert feature_frame[column].isna().sum() == 0

    def test_declared_features_exist(self, feature_frame):
        from src.utils.config import FEATURE_COLUMNS

        missing = [c for c in FEATURE_COLUMNS if c not in feature_frame.columns]
        assert not missing, f"config declares features the pipeline never builds: {missing}"

    def test_no_feature_is_the_target(self, feature_frame):
        """
        A feature perfectly correlated with the target means something leaked.
        Under displacement targets this is a genuine check; under absolute
        targets LAT would trip it, which is itself the argument for
        displacement.
        """

        from src.utils.config import FEATURE_COLUMNS

        subset = feature_frame[
            [c for c in FEATURE_COLUMNS if c in feature_frame.columns]
            + ["TARGET_DLAT_24H"]
        ].dropna()

        if len(subset) < 30:
            pytest.skip("too few complete rows for a correlation check")

        correlations = subset.drop(columns="TARGET_DLAT_24H").corrwith(
            subset["TARGET_DLAT_24H"]
        ).abs()

        leaked = correlations[correlations > 0.999]
        assert leaked.empty, f"features indistinguishable from the target: {list(leaked.index)}"