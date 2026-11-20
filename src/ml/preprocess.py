"""Feature/target separation, preprocessing pipeline and train/test split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml.data import TARGET_COLUMN, load_dataset, prepare_dataset

# Canonical feature names and order. Used for training AND inference.
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Only these are scaled. V1-V28 are already PCA-transformed.
COLUMNS_TO_SCALE = ["Time", "Amount"]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix X from the target y."""
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X[FEATURE_COLUMNS], y


def build_preprocessor() -> ColumnTransformer:
    """Scale Time and Amount; pass the PCA features through unchanged.

    This is returned UNFITTED. It becomes part of the sklearn Pipeline, so it is
    fitted on training data only (no leakage) and reused as-is at inference.
    """
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), COLUMNS_TO_SCALE)],
        remainder="passthrough",
    )


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Reproducible, stratified 80/20 split (preserves the fraud ratio)."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_train_test_data(remove_duplicates: bool = False):
    """Load -> prepare -> split. Single entry point so training and evaluation
    always see exactly the same split."""
    df = prepare_dataset(load_dataset(), remove_duplicates=remove_duplicates)
    X, y = split_features_target(df)
    return split_train_test(X, y)
