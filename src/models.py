# src/models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout  # type: ignore


def build_1d_cnn(input_shape: tuple) -> Sequential:
    """Builds and compiles a 1D Convolutional Neural Network."""
    model = Sequential(
        [
            Conv1D(
                filters=32, kernel_size=3, activation="relu", input_shape=input_shape
            ),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_model(config: dict, input_shape: tuple = None):
    """Factory function to instantiate models based on config."""
    model_type = config["model"]["type"]
    random_seed = config["pipeline"]["random_seed"]

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["model"]["rf_estimators"],
            class_weight="balanced",
            random_state=random_seed,
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_seed,
        )
    elif model_type == "cnn":
        if input_shape is None:
            raise ValueError("input_shape must be provided for CNN.")
        return build_1d_cnn(input_shape)
    else:
        raise ValueError(f"Model type {model_type} is not supported yet.")
