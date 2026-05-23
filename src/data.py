# src/data.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler


def load_data(train_path: str, test_path: str, target_col: str, dev_mode: bool = False):
    """Loads CSVs, taking only a subset if dev_mode is True for fast testing."""
    row_limit = 200 if dev_mode else None

    train_df = pd.read_csv(train_path, nrows=row_limit)
    test_df = pd.read_csv(test_path, nrows=row_limit)

    X_train = train_df.drop(columns=[target_col])
    # XGBoost and deep learning models expect 0-indexed classes.
    # The Kepler dataset uses 1 (No) and 2 (Yes). Subtracting 1 maps them to 0 and 1.
    y_train = train_df[target_col] - 1

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col] - 1

    return X_train, y_train, X_test, y_test


def preprocess_features(X_train, X_test, pca_components: int):
    """Scales data and applies Principal Component Analysis."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, scaler, pca


def balance_classes(X_train, y_train, method: str, random_seed: int):
    """Applies oversampling to handle imbalanced exoplanet classes."""
    if method == "smote":
        sampler = SMOTE(random_state=random_seed)
    else:
        sampler = RandomOverSampler(random_state=random_seed)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
