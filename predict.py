# predict.py
import joblib
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from src.utils import get_logger

logger = get_logger("Inference")


def load_artifacts(config: dict):
    """Loads the saved scaler, PCA, and model."""
    logger.info("Loading ML artifacts...")
    scaler = joblib.load("saved_models/scaler.pkl")
    pca = joblib.load("saved_models/pca.pkl")

    model_type = config["model"]["type"]
    if model_type == "cnn":
        model = tf.keras.models.load_model(f"saved_models/{model_type}_model.keras")
    else:
        model = joblib.load(f"saved_models/{model_type}_model.pkl")

    return scaler, pca, model


def predict_exoplanet(raw_data: np.ndarray, config: dict):
    """Processes raw star flux data and returns a prediction."""
    scaler, pca, model = load_artifacts(config)

    # 1. Preprocess the incoming data
    logger.info("Preprocessing input data...")
    data_scaled = scaler.transform(raw_data)
    data_pca = pca.transform(data_scaled)

    # 2. Predict
    logger.info("Running inference...")
    model_type = config["model"]["type"]

    if model_type == "cnn":
        # Reshape for CNN (samples, time_steps, channels)
        data_cnn = data_pca.reshape((data_pca.shape[0], data_pca.shape[1], 1))
        prediction_prob = model.predict(data_cnn, verbose=0)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]
    else:
        prediction = model.predict(data_pca)[0]

    # Standard labels for exoplanet datasets are usually 1 (no) and 2 (yes)
    # Update this if your specific dataset uses 0 and 1
    result = "EXOPLANET DETECTED" if prediction == 2 else "NO EXOPLANET"

    return result


if __name__ == "__main__":
    # 1. Load config
    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Simulate incoming data (e.g., from a telescope API)
    logger.info("Simulating incoming telescope data...")
    test_df = pd.read_csv(config["data"]["test_path"], nrows=5)

    # Take the first row, drop the label, and convert to numpy array
    target_col = config["preprocessing"]["target_column"]
    sample_raw_data = test_df.drop(columns=[target_col]).iloc[[0]].values

    # 3. Get Prediction
    result = predict_exoplanet(sample_raw_data, config)

    print("\n" + "=" * 40)
    print(f" FINAL PREDICTION: {result} ")
    print("=" * 40 + "\n")
