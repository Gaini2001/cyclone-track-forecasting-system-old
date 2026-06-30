
import pandas as pd

from src.training.save_model import load_model
from app.feature_builder import build_features

model = load_model()


def predict_location(data):

    features = build_features(data)

    df = pd.DataFrame(
        [features]
    )

    prediction = model.predict(df)

    return {
    "predicted_lat": float(prediction[0][0]),
    "predicted_lon": float(prediction[0][1]),

    "model": "Random Forest",

    "forecast_horizon": "6 Hours",

    "track_error_km": 64.56,
}