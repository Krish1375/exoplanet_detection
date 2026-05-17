# src/train.py
from sklearn.metrics import classification_report, balanced_accuracy_score


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Trains the model and prints evaluation metrics."""
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)

    print("Evaluating on test set...")
    predictions = model.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, predictions)
    print(f"Balanced Accuracy: {bal_acc:.4f}\n")
    print(classification_report(y_test, predictions))

    return model
