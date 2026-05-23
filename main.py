# main.py
import yaml
import os
import joblib
from src.data import load_data, preprocess_features, balance_classes
from src.models import get_model
from src.train import train_and_evaluate
from src.utils import get_logger

logger = get_logger(__name__)


def load_config(config_path="conf/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    logger.info("Starting Exoplanet Detection Pipeline...")
    config = load_config()

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

    # Calculate input shape for CNN if needed
    input_shape = (
        (X_train_bal.shape[1], 1) if config["model"]["type"] == "cnn" else None
    )
    model = get_model(config, input_shape)

    trained_model = train_and_evaluate(
        model, X_train_bal, y_train_bal, X_test_pca, y_test, config
    )
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
