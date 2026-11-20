"""Evaluate the saved pipeline on the held-out test set.

Run from the project root:
    python -m src.ml.evaluate
"""

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.ml.predict import fraud_class_index
from src.ml.preprocess import get_train_test_data
from src.ml.train import MODEL_PATH

METRICS_PATH = Path("results/baseline_metrics.json")


def compute_metrics(pipeline, X_test, y_test) -> dict:
    """Compute all tracked metrics. Probability is taken for class 1 explicitly."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, fraud_class_index(pipeline)]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }


def evaluate(model_path: Path = MODEL_PATH, remove_duplicates: bool = False) -> dict:
    pipeline = joblib.load(model_path)
    _, X_test, _, y_test = get_train_test_data(remove_duplicates)
    metrics = compute_metrics(pipeline, X_test, y_test)

    cm = metrics["confusion_matrix"]
    print("\nBASELINE EVALUATION")
    print("-" * 40)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC   : {metrics['pr_auc']:.4f}")
    print("\nConfusion matrix:")
    print(f"[[{cm['true_negatives']:>6} {cm['false_positives']:>5}]")
    print(f" [{cm['false_negatives']:>6} {cm['true_positives']:>5}]]")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            pipeline.predict(X_test),
            target_names=["Legitimate", "Fraud"],
            digits=4,
        )
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate()
