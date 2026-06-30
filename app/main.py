from fastapi import FastAPI

from app.schemas import CycloneInput
from app.predictor import predict_location


app = FastAPI(
    title="Cyclone Track Forecast API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "message":
        "Cyclone Track Forecast API"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
    cyclone: CycloneInput
):

    result = predict_location(
        cyclone.model_dump()
    )

    return result