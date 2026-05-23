# src/train.py
import os
import joblib
from sklearn.metrics import classification_report, balanced_accuracy_score
from src.utils import get_logger

logger = get_logger(__name__)


def train_and_evaluate(model, X_train, y_train, X_test, y_test, config: dict):
    """Trains the model, evaluates it, and saves the artifact."""
    model_type = config["model"]["type"]
    logger.info(f"Initiating training for: {model_type}")

    # 1. Training (Check if it is a Deep Learning model)
    if model_type in ["cnn", "lstm"]:
        X_train_dl = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_dl = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        epochs = (
            1
            if config["pipeline"].get("dev_mode")
            else config["model"][f"{model_type}_epochs"]
        )
        batch_size = config["model"][f"{model_type}_batch_size"]

        model.fit(X_train_dl, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        predictions = (model.predict(X_test_dl) > 0.5).astype("int32")
    else:
        # Standard ML Models (RF, LogReg, XGBoost)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    # 2. Evaluation
    logger.info("Evaluating on test set...")
    bal_acc = balanced_accuracy_score(y_test, predictions)
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")

    # ADDED zero_division=0 to suppress the sklearn warnings
    logger.info(f"\n{classification_report(y_test, predictions, zero_division=0)}")

    # 3. Serialization
    os.makedirs("saved_models", exist_ok=True)
    if model_type in ["cnn", "lstm"]:
        model_path = f"saved_models/{model_type}_model.keras"
        model.save(model_path)
    else:
        model_path = f"saved_models/{model_type}_model.pkl"
        joblib.dump(model, model_path)

    logger.info(f"Model saved successfully at {model_path}")
    return model
