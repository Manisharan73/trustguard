"""Standalone prediction: transaction in, prediction + fraud probability out.

This module only produces model output. It makes no risk or business decisions.

Run a demo from the project root:
    python -m src.ml.predict
"""

from pathlib import Path

import joblib
import pandas as pd

from src.ml.preprocess import FEATURE_COLUMNS
from src.ml.train import MODEL_PATH


def load_model(model_path: Path = MODEL_PATH):
    """Load the saved preprocessing + classifier pipeline."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run: python -m src.ml.train"
        )
    return joblib.load(model_path)


def fraud_class_index(pipeline) -> int:
    """Column of predict_proba that corresponds to class 1 (fraud).

    Looked up from the model's classes_ instead of assuming column order.
    """
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError(f"Model has no fraud class (1). classes_ = {classes}")
    return classes.index(1)


def predict_transaction(transaction: dict, pipeline=None) -> dict:
    """Predict a single transaction.

    Returns {"prediction": 0 | 1, "probability": P(fraud)}.
    Raises ValueError if required features are missing.
    """
    if pipeline is None:
        pipeline = load_model()

    missing = [col for col in FEATURE_COLUMNS if col not in transaction]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Build the frame in the canonical training order
    row = pd.DataFrame([{col: transaction[col] for col in FEATURE_COLUMNS}])

    prediction = int(pipeline.predict(row)[0])
    probability = float(pipeline.predict_proba(row)[0, fraud_class_index(pipeline)])
    return {"prediction": prediction, "probability": probability}


if __name__ == "__main__":
    from src.ml.preprocess import get_train_test_data

    model = load_model()
    _, X_test, _, y_test = get_train_test_data()

    # One real legitimate and one real fraud transaction from the held-out test set
    for label, name in [(0, "legitimate"), (1, "fraud")]:
        idx = y_test[y_test == label].index[0]
        result = predict_transaction(X_test.loc[idx].to_dict(), model)
        print(f"\nTest transaction (actual: {name})")
        print(f"Transaction prediction: {result['prediction']}")
        print(f"Fraud probability: {result['probability']}")
