"""
main.py

Cyclone Track Forecast API.

Swagger / OpenAPI docs are served at /docs (ReDoc at /redoc) purely from the
type hints and docstrings below and in app.schemas -- nothing here builds them
explicitly.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import available_models, predict_location
from app.schemas import CycloneInput

app = FastAPI(
    title="Cyclone Track Forecast API",
    description="Forecasts a tropical cyclone's position from a short "
                 "observation history. POST to /predict; try it from /docs.",
    version="2.0",
)

# CORS_ORIGINS is a comma-separated allowlist; "*" (the default) is fine for a
# public demo and wrong for anything that carries credentials.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Cyclone Track Forecast API — see /docs"}


@app.get("/health")
def health():
    """
    Liveness plus model state. "healthy" means at least one model is loadable
    right now; "degraded" means the process is up but /predict will 503.
    """

    models = available_models()

    return {
        "status": "healthy" if models else "degraded",
        "models_available": models,
    }


@app.post("/predict")
def predict(cyclone: CycloneInput):
    """
    Forecast a position from an observation history.

    Returns 503, not 500, when the request is well-formed but no model is
    trained for the requested horizon -- that is a deployment gap, not a bad
    request.
    """

    try:
        return predict_location(cyclone.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
