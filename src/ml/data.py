"""Dataset loading, inspection and preparation.

This module never modifies the raw CSV on disk. Any cleaning (such as duplicate
removal) happens on an in-memory copy and is always reported, never silent.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/creditcard.csv")
TARGET_COLUMN = "Class"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw credit card fraud dataset."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Place creditcard.csv in data/raw/."
        )
    return pd.read_csv(path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Print and return a summary of the dataset (shape, types, quality, balance)."""
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    n_fraud = int(class_counts.get(1, 0))
    n_legit = int(class_counts.get(0, 0))

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_fraud_rows": int(df[df.duplicated()][TARGET_COLUMN].sum()),
        "legitimate": n_legit,
        "fraud": n_fraud,
        "fraud_percentage": round(100 * n_fraud / len(df), 4),
    }

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape                 : {df.shape}")
    print(f"Data types            : {df.dtypes.value_counts().to_dict()}")
    print(f"Missing values        : {summary['missing_values']}")
    print(
        f"Duplicate rows        : {summary['duplicate_rows']} "
        f"({summary['duplicate_fraud_rows']} of them fraud)"
    )
    print(f"Legitimate (Class 0)  : {n_legit}")
    print(f"Fraud (Class 1)       : {n_fraud} ({summary['fraud_percentage']}%)")
    print("\nAmount by class:")
    print(df.groupby(TARGET_COLUMN)["Amount"].describe().round(2))
    print("=" * 60)
    return summary


def prepare_dataset(df: pd.DataFrame, remove_duplicates: bool = False) -> pd.DataFrame:
    """Return the dataset used for modelling.

    remove_duplicates defaults to False because the reference baseline metrics
    (Recall 0.9184, PR-AUC 0.7159, 98 fraud cases in the test set) were produced
    on the full 284,807 rows. Setting it to True is supported, but it changes
    the baseline numbers. The number of removed rows is always reported.
    """
    if not remove_duplicates:
        print("Duplicate removal: OFF (full dataset used, matches reference baseline)")
        return df.copy()

    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate removal: ON ({before - len(cleaned)} rows removed)")
    return cleaned


if __name__ == "__main__":
    data = load_dataset()
    inspect_dataset(data)
