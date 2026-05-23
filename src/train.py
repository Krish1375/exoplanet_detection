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

    # 1. Training
    if model_type == "cnn":
        # CNNs require a 3D shape: (batch, steps, channels)
        X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model.fit(
            X_train_cnn,
            y_train,
            epochs=config["model"]["cnn_epochs"],
            batch_size=config["model"]["cnn_batch_size"],
            verbose=1,
        )
        predictions = (model.predict(X_test_cnn) > 0.5).astype("int32")
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    # 2. Evaluation
    logger.info("Evaluating on test set...")
    bal_acc = balanced_accuracy_score(y_test, predictions)
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}")
    logger.info(f"\n{classification_report(y_test, predictions)}")

    # 3. Serialization (Saving the model)
    os.makedirs("saved_models", exist_ok=True)
    if model_type == "cnn":
        model_path = f"saved_models/{model_type}_model.keras"
        model.save(model_path)
    else:
        model_path = f"saved_models/{model_type}_model.pkl"
        joblib.dump(model, model_path)

    logger.info(f"Model saved successfully at {model_path}")
    return model
