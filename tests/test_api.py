"""
tests/test_api.py

Tests for the prediction API.

The one change that matters
---------------------------
The original fixture did this:

    try:
        from app.main import app
        return TestClient(app)
    except Exception:
        pytest.skip("Could not load API (model file may be missing)")

`except Exception` turns every failure into a skip. A syntax error in the API,
a missing dependency, a broken import, a model that will not unpickle -- all of
them produce a green run with zero tests executed and one quiet skip line. A
suite that passes when the thing under test cannot even be imported is worse
than no suite, because it converts "untested" into "tested and fine".

Here the fixture catches only the two conditions that legitimately mean "not
set up yet", and setting REQUIRE_API=1 turns those into failures so CI cannot
drift into permanently skipping everything.

The assertions also test behaviour rather than status codes. `-90 <= lat <= 90`
passes for a model that returns a constant zero; the tests below check that the
forecast is near the storm, that it moves in a plausible direction, and that
identical inputs give identical outputs.
"""

import os

import pytest

# 6 hours of storm motion is a few hundred km at most; a forecast further than
# this from the last observed position is not a forecast.
MAX_PLAUSIBLE_6H_DISPLACEMENT_DEG = 5.0

REQUIRE_API = os.environ.get("REQUIRE_API") == "1"


def _skip_or_fail(reason: str):
    """
    Skip when the API is not set up, fail when it is supposed to be.

    Set REQUIRE_API=1 in CI so a broken API cannot masquerade as an absent one.
    """

    if REQUIRE_API:
        pytest.fail(f"REQUIRE_API=1 but the API could not be loaded: {reason}")

    pytest.skip(reason)


@pytest.fixture
def client():
    """
    Test client for the API.

    Only ImportError and FileNotFoundError are treated as "not set up". Any
    other exception -- a TypeError in the app module, a validation error at
    import, a corrupt artifact -- propagates and fails the test, which is the
    point.
    """

    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return _skip_or_fail("fastapi is not installed")

    try:
        from app.main import app
    except ImportError as exc:
        return _skip_or_fail(f"API module not importable: {exc}")
    except FileNotFoundError as exc:
        return _skip_or_fail(f"model artifact missing: {exc}")

    return TestClient(app)


@pytest.fixture
def observations():
    """
    A short observation sequence.

    Five consecutive positions, because the model needs 3 lags plus rolling features.
    """

    return [
        {"latitude": 13.8, "longitude": 83.4, "wind_speed": 50.0, "pressure": 990.0},
        {"latitude": 14.0, "longitude": 83.7, "wind_speed": 52.0, "pressure": 988.0},
        {"latitude": 14.2, "longitude": 84.0, "wind_speed": 55.0, "pressure": 985.0},
        {"latitude": 14.5, "longitude": 84.6, "wind_speed": 58.0, "pressure": 982.0},
        {"latitude": 15.0, "longitude": 85.2, "wind_speed": 60.0, "pressure": 980.0},
    ]


# ==========================================================
# Service Endpoints
# ==========================================================

class TestServiceEndpoints:

    def test_health_reports_healthy(self, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_reports_model_state(self, client):
        """
        A health check that only says "the process is up" is not much of a
        check. It should say whether the model actually loaded, since that is
        the failure that matters in production.
        """

        payload = response = client.get("/health").json()

        assert any(
            key in payload for key in ("model_loaded", "model", "models", "models_available", "version")
        ), "health should report model state, not just process liveness"

    def test_root_responds(self, client):
        response = client.get("/")
        assert response.status_code == 200


# ==========================================================
# Prediction
# ==========================================================

class TestPredictEndpoint:

    def test_returns_a_forecast(self, client, observations):
        response = client.post(
            "/predict",
            json={"observations": observations, "forecast_horizon": 6},
        )

        assert response.status_code == 200, response.text

        data = response.json()
        assert "predicted_lat" in data and "predicted_lon" in data

    def test_forecast_is_near_the_storm(self, client, observations):
        """
        The assertion the original was missing.

        `-90 <= lat <= 90` is satisfied by a model returning a constant zero.
        A 6-hour forecast has to land within a few degrees of the last observed
        position, which a broken feature pipeline will not manage.
        """

        response = client.post(
            "/predict",
            json={"observations": observations, "forecast_horizon": 6},
        )

        assert response.status_code == 200, response.text
        data = response.json()

        last = observations[-1]

        assert abs(data["predicted_lat"] - last["latitude"]) < MAX_PLAUSIBLE_6H_DISPLACEMENT_DEG
        assert abs(data["predicted_lon"] - last["longitude"]) < MAX_PLAUSIBLE_6H_DISPLACEMENT_DEG

    def test_forecast_continues_the_observed_motion(self, client, observations):
        """
        The storm is moving northeast. A forecast that sends it southwest means
        the features reaching the model do not describe the track it was given
        -- typically a sign order or column order mismatch, which no status
        code will reveal.
        """

        response = client.post(
            "/predict",
            json={"observations": observations, "forecast_horizon": 6},
        )

        assert response.status_code == 200, response.text
        data = response.json()

        last = observations[-1]

        assert data["predicted_lat"] > last["latitude"] - 0.5, \
            "storm was moving north; forecast moves it clearly south"
        assert data["predicted_lon"] > last["longitude"] - 0.5, \
            "storm was moving east; forecast moves it clearly west"

    def test_deterministic(self, client, observations):
        """
        Identical requests must give identical answers. A difference means
        unseeded state somewhere in the serving path.
        """

        payload = {"observations": observations, "forecast_horizon": 6}

        first = client.post("/predict", json=payload).json()
        second = client.post("/predict", json=payload).json()

        assert first["predicted_lat"] == pytest.approx(second["predicted_lat"])
        assert first["predicted_lon"] == pytest.approx(second["predicted_lon"])

    @pytest.mark.parametrize("horizon", [6, 12, 24, 48, 72])
    def test_every_configured_horizon_is_served(self, client, observations, horizon):
        """
        Config declares five horizons. If the API only has a 24h model, this is
        where that surfaces -- rather than in a 500 the first time someone asks
        for 72h.
        """

        response = client.post(
            "/predict",
            json={"observations": observations, "forecast_horizon": horizon},
        )

        assert response.status_code == 200, (
            f"{horizon}h horizon not served: {response.text}"
        )

    def test_longer_horizon_moves_the_storm_further(self, client, observations):
        """
        A 72h forecast should displace the storm further than a 6h one. If they
        are similar, the horizon parameter is probably being ignored and one
        model is answering everything.
        """

        def displacement(horizon):
            response = client.post(
                "/predict",
                json={"observations": observations, "forecast_horizon": horizon},
            )

            if response.status_code != 200:
                pytest.skip(f"{horizon}h not served")

            data = response.json()
            last = observations[-1]

            return (
                abs(data["predicted_lat"] - last["latitude"])
                + abs(data["predicted_lon"] - last["longitude"])
            )

        assert displacement(72) > displacement(6)


# ==========================================================
# Validation
# ==========================================================

class TestInputValidation:

    @pytest.mark.parametrize("field, value", [
        ("latitude", 100.0),
        ("latitude", -95.0),
        ("longitude", 200.0),
        ("wind_speed", -10.0),
        ("pressure", -5.0),
    ])
    def test_out_of_range_values_rejected(self, client, observations, field, value):
        payload = {"observations": [dict(observations[-1], **{field: value})]}
        response = client.post("/predict", json=payload)

        assert response.status_code == 422, (
            f"{field}={value} should be rejected, got {response.status_code}"
        )

    def test_unsupported_horizon_rejected(self, client, observations):
        response = client.post(
            "/predict",
            json={"observations": observations, "forecast_horizon": 7},
        )

        assert response.status_code == 422

    def test_empty_observations_rejected(self, client):
        response = client.post("/predict", json={"observations": []})
        assert response.status_code == 422

    def test_missing_field_rejected(self, client):
        response = client.post(
            "/predict",
            json={"observations": [{"latitude": 15.0, "longitude": 85.0}]},
        )

        assert response.status_code == 422

    def test_insufficient_history_is_handled_explicitly(self, client, observations):
        """
        The model needs several past positions. Given one observation the API
        must either reject the request (422) or document how it fills the gap
        -- what it must not do is silently invent lag features and return a
        confident number.

        A 200 here is only acceptable if the response says the history was
        insufficient.
        """

        response = client.post(
            "/predict",
            json={"observations": observations[:1], "forecast_horizon": 6},
        )

        if response.status_code == 200:
            data = response.json()

            assert any(
                key in data for key in ("warning", "warnings", "n_observations",
                                        "history_sufficient", "note")
            ), (
                "a single observation cannot supply lag features; the response "
                "should flag that rather than returning a bare prediction"
            )
        else:
            assert response.status_code == 422