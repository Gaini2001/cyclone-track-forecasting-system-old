from src.training.train import load_feature_dataset
from src.preprocessing.create_sequences import create_sequences

import pandas as pd
import numpy as np


def load_storm_ids():

    train_storms = pd.read_csv(
        "data/processed/train_storms.csv"
    )["SID"].values

    test_storms = pd.read_csv(
        "data/processed/test_storms.csv"
    )["SID"].values

    return train_storms, test_storms


def storm_based_sequence_split(
    X,
    y,
    sequence_sids,
):

    train_storms, test_storms = load_storm_ids()

    train_mask = np.isin(
        sequence_sids,
        train_storms
    )

    test_mask = np.isin(
        sequence_sids,
        test_storms
    )

    X_train = X[train_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_test = y[test_mask]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================
# Main
# ==========================

df = load_feature_dataset()

X, y, sids = create_sequences(df)

X_train, X_test, y_train, y_test = (
    storm_based_sequence_split(
        X,
        y,
        sids
    )
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

# ==========================================================
# Build Model
# ==========================================================

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)


model = Sequential([

    LSTM(
        64,
        input_shape=(8, 6)
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(2)

])

# ==========================================================
# Compile Model
# ==========================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ==========================================================
# Train Model
# ==========================================================

history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=10,

    batch_size=256,

    verbose=1

)

# ==========================================================
# Evaluate
# ==========================================================

loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Results")
print(f"Loss : {loss:.4f}")
print(f"MAE  : {mae:.4f}")

# ==========================================================
# Save Model
# ==========================================================

model.save(
    "models/lstm_6h.keras"
)

print(
    "\nModel saved successfully."
)



print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)