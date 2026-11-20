"""Data-quality experiment: effect of duplicate removal on the baseline.

This is a SEPARATE experiment. It never overwrites the baseline model
(models/fraud_model.pkl) or the baseline metrics (results/baseline_metrics.json).

Run from the project root:
    python -m src.ml.experiment_duplicates
"""

import json
from pathlib import Path

import pandas as pd

from src.ml.evaluate import METRICS_PATH, compute_metrics
from src.ml.preprocess import FEATURE_COLUMNS, get_train_test_data
from src.ml.train import build_pipeline

EXPERIMENT_PATH = Path("results/experiment_remove_duplicates.json")


def measure_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Count test rows that have an exact copy in the training set."""
    train_rows = X_train[FEATURE_COLUMNS].drop_duplicates()
    flagged = X_test[FEATURE_COLUMNS].merge(
        train_rows, on=FEATURE_COLUMNS, how="left", indicator=True
    )["_merge"].eq("both").to_numpy()

    return {
        "test_rows": int(len(X_test)),
        "test_rows_with_exact_copy_in_train": int(flagged.sum()),
        "test_fraud_rows": int(y_test.sum()),
        "test_fraud_rows_with_exact_copy_in_train": int(flagged[y_test.to_numpy() == 1].sum()),
    }


def run_experiment() -> dict:
    # 1) Leakage in the REFERENCE setup (duplicates kept)
    Xtr_full, Xte_full, _, yte_full = get_train_test_data(remove_duplicates=False)
    leakage = measure_leakage(Xtr_full, Xte_full, yte_full)

    # 2) Retrain on de-duplicated data (in memory only; nothing is saved as the model)
    Xtr, Xte, ytr, yte = get_train_test_data(remove_duplicates=True)
    pipeline = build_pipeline().fit(Xtr, ytr)
    dedup_metrics = compute_metrics(pipeline, Xte, yte)
    dedup_metrics["test_rows"] = int(len(yte))
    dedup_metrics["test_fraud_rows"] = int(yte.sum())

    result = {
        "description": "Baseline Logistic Regression retrained on de-duplicated data",
        "baseline_reference": json.loads(METRICS_PATH.read_text()),
        "leakage_in_reference_split": leakage,
        "remove_duplicates_metrics": dedup_metrics,
    }

    EXPERIMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_PATH.write_text(json.dumps(result, indent=2))

    base = result["baseline_reference"]
    print("\nLEAKAGE IN REFERENCE SPLIT (duplicates kept)")
    print(f"Test rows with an exact copy in train : "
          f"{leakage['test_rows_with_exact_copy_in_train']} / {leakage['test_rows']}")
    print(f"Test FRAUD rows with a copy in train  : "
          f"{leakage['test_fraud_rows_with_exact_copy_in_train']} / {leakage['test_fraud_rows']}")
    print("\nBASELINE vs DUPLICATES REMOVED")
    print(f"{'metric':<10}{'baseline':>10}{'dedup':>10}")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        print(f"{m:<10}{base[m]:>10.4f}{dedup_metrics[m]:>10.4f}")
    print(f"\nSaved to {EXPERIMENT_PATH}")
    return result


if __name__ == "__main__":
    run_experiment()
