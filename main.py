# main.py
import yaml
from src.data import load_data, preprocess_features, balance_classes
from src.models import get_model
from src.train import train_and_evaluate


def load_config(config_path="conf/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. Setup
    config = load_config()
    print("Configuration loaded.")

    # 2. Data Pipeline
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(
        config["data"]["train_path"],
        config["data"]["test_path"],
        config["data"]["target_column"],
    )

    print("Preprocessing and scaling...")
    X_train_pca, X_test_pca, _, _ = preprocess_features(
        X_train, X_test, config["preprocessing"]["pca_components"]
    )

    print(f"Balancing classes using {config['preprocessing']['oversample_method']}...")
    X_train_bal, y_train_bal = balance_classes(
        X_train_pca,
        y_train,
        config["preprocessing"]["oversample_method"],
        config["pipeline"]["random_seed"],
    )

    # 3. Modeling
    model = get_model(config)

    # 4. Training & Evaluation
    trained_model = train_and_evaluate(
        model, X_train_bal, y_train_bal, X_test_pca, y_test
    )


if __name__ == "__main__":
    main()
