# main.py
import argparse
import yaml
import os

# Suppress annoying TensorFlow C++ and oneDNN warnings BEFORE importing tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from src.data import load_data, preprocess_features, balance_classes
from src.models import get_model
from src.train import train_and_evaluate
from src.utils import get_logger

logger = get_logger("ExoPipeline")


def load_config(config_path="conf/config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_training(config: dict):
    """Executes the full training and serialization pipeline."""
    logger.info("--- STARTING TRAINING PIPELINE ---")

    logger.info("Loading data...")
    X_train, y_train, X_test, y_test = load_data(
        config["data"]["train_path"],
        config["data"]["test_path"],
        config["data"]["target_column"],
        config["pipeline"].get("dev_mode", False),
    )

    logger.info("Preprocessing and scaling...")
    X_train_pca, X_test_pca, scaler, pca = preprocess_features(
        X_train, X_test, config["preprocessing"]["pca_components"]
    )

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, "saved_models/scaler.pkl")
    joblib.dump(pca, "saved_models/pca.pkl")
    logger.info("Saved Scaler and PCA artifacts.")

    logger.info(
        f"Balancing classes using {config['preprocessing']['oversample_method']}..."
    )
    X_train_bal, y_train_bal = balance_classes(
        X_train_pca,
        y_train,
        config["preprocessing"]["oversample_method"],
        config["pipeline"]["random_seed"],
    )

    input_shape = (
        (X_train_bal.shape[1], 1)
        if config["model"]["type"] in ["cnn", "lstm"]
        else None
    )
    model = get_model(config, input_shape)

    trained_model = train_and_evaluate(
        model, X_train_bal, y_train_bal, X_test_pca, y_test, config
    )
    logger.info("--- TRAINING PIPELINE COMPLETE ---")


def run_inference(config: dict):
    """Executes the inference pipeline on a sample data point."""
    logger.info("--- STARTING INFERENCE PIPELINE ---")

    try:
        scaler = joblib.load("saved_models/scaler.pkl")
        pca = joblib.load("saved_models/pca.pkl")

        model_type = config["model"]["type"]
        if model_type in ["cnn", "lstm"]:
            model = tf.keras.models.load_model(f"saved_models/{model_type}_model.keras")
        else:
            model = joblib.load(f"saved_models/{model_type}_model.pkl")
    except FileNotFoundError:
        logger.error(
            "Artifacts not found! Please run the training pipeline first using --train"
        )
        return

    logger.info("Simulating incoming telescope data...")
    test_df = pd.read_csv(config["data"]["test_path"], nrows=5)
    target_col = config["data"]["target_column"]
    raw_data = test_df.drop(columns=[target_col]).iloc[[0]].values

    logger.info("Preprocessing input data...")
    data_scaled = scaler.transform(raw_data)
    data_pca = pca.transform(data_scaled)

    logger.info(f"Running inference using {model_type}...")
    if model_type in ["cnn", "lstm"]:
        data_dl = data_pca.reshape((data_pca.shape[0], data_pca.shape[1], 1))
        prediction_prob = model.predict(data_dl, verbose=0)
        prediction = (prediction_prob > 0.5).astype(int)[0][0]
    else:
        prediction = model.predict(data_pca)[0]

    # Labels are now mapped to 0 (No Exoplanet) and 1 (Exoplanet)
    result = "EXOPLANET DETECTED" if prediction == 1 else "NO EXOPLANET"

    print("\n" + "=" * 40)
    print(f" FINAL PREDICTION: {result} ")
    print("=" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Exoplanet Detection MLE Pipeline")

    parser.add_argument(
        "--train", action="store_true", help="Run the full training pipeline"
    )
    parser.add_argument(
        "--predict", action="store_true", help="Run inference on simulated data"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in dev mode (fast execution, subset of data)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to custom config file",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.debug:
        logger.info(
            "🚨 DEBUG MODE ACTIVATED: Running fast execution on subset of data 🚨"
        )
        config["pipeline"]["dev_mode"] = True

    if args.train:
        run_training(config)
    elif args.predict:
        run_inference(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
