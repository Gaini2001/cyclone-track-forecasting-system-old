"""
Main entry point for the Cyclone Track Forecasting System.
"""

from src.ingestion.dataset_loader import main as dataset_loader
from src.preprocessing.clean_data import main as clean_data
from src.features.feature_engineering import main as feature_engineering
from src.validation.validate_features import main as validate_features
from src.training.train import main as training
from src.training.evaluate import main as evaluate

def main():
    print("=" * 60)
    print("Cyclone Track Forecasting System")
    print("=" * 60)

    dataset_loader()

    clean_data()

    feature_engineering()

    validate_features()

    training()

    evaluate()


if __name__ == "__main__":
    main()