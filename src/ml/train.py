"""Train the baseline fraud model and save the complete pipeline.

Run from the project root:
    python -m src.ml.train
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml.preprocess import build_preprocessor, get_train_test_data

MODEL_PATH = Path("models/fraud_model.pkl")


def build_pipeline() -> Pipeline:
    """Preprocessing + baseline Logistic Regression in ONE pipeline.

    The baseline configuration is the project's reference point. Do not change
    it; build new models alongside it so they can be compared against it.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )


def train_and_save(model_path: Path = MODEL_PATH, remove_duplicates: bool = False) -> Pipeline:
    X_train, X_test, y_train, y_test = get_train_test_data(remove_duplicates)

    print(f"Train rows: {len(X_train)} (fraud: {int(y_train.sum())})")
    print(f"Test rows : {len(X_test)} (fraud: {int(y_test.sum())})")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)  # scaler is fitted on training data only

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    return pipeline


if __name__ == "__main__":
    train_and_save()
