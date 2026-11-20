"""Explainability engine for TrustGuard (Phase 7).

Explains predictions of the baseline Logistic Regression model using SHAP LinearExplainer.
Features are explained in the model's natural log-odds space:
    log_odds(P(fraud)) = base_value + sum(contributions)

Positive contributions push the log-odds towards fraud; negative contributions
push towards legitimate. Feature contributions reflect model sensitivity under
the training distribution, not real-world causation.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.explainability.schemas import FeatureContribution, ExplanationResponse
from src.ml.predict import fraud_class_index
from src.ml.preprocess import FEATURE_COLUMNS

DEFAULT_BACKGROUND_PATH = Path("data/processed/background_reference.npy")


def get_default_background(preprocessor, background_path: Path = DEFAULT_BACKGROUND_PATH) -> np.ndarray:
    """Load or compute a background reference dataset for SHAP LinearExplainer."""
    if background_path.exists():
        return np.load(background_path)

    try:
        from src.ml.data import DATA_PATH
        if DATA_PATH.exists():
            from src.ml.preprocess import get_train_test_data
            X_train, _, _, _ = get_train_test_data()
            bg = preprocessor.transform(X_train.iloc[:100])
            background_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(background_path, bg)
            return bg
    except Exception:
        pass

    # Standardized fallback: zeros vector (represents population mean of standardized features)
    return np.zeros((1, len(FEATURE_COLUMNS)))


class ExplainabilityEngine:
    """Calculate feature-level attributions and human-readable summaries using SHAP."""

    def __init__(self, pipeline, background_data: np.ndarray | None = None):
        self._validate_pipeline(pipeline)
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessing"]
        self.classifier = pipeline.named_steps["classifier"]

        # Feature names in the exact order emitted by the preprocessor
        feature_names_out = self.preprocessor.get_feature_names_out()
        self.feature_names = [name.split("__")[-1] for name in feature_names_out]

        if background_data is None:
            background_data = get_default_background(self.preprocessor)
        self.background_data = background_data

        # Initialize SHAP LinearExplainer once with the classifier and reference distribution
        self.explainer = shap.LinearExplainer(
            self.classifier,
            self.background_data,
            feature_names=self.feature_names,
        )
        self.base_value = float(self.explainer.expected_value)

    @staticmethod
    def _validate_pipeline(pipeline) -> None:
        if not hasattr(pipeline, "named_steps"):
            raise ValueError("pipeline must be an sklearn Pipeline with named_steps.")
        if "preprocessing" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
            raise ValueError("pipeline must contain 'preprocessing' and 'classifier' steps.")
        classifier = pipeline.named_steps["classifier"]
        if not hasattr(classifier, "coef_") or not hasattr(classifier, "intercept_"):
            raise ValueError("classifier must have coef_ and intercept_ attributes.")

    def _validate_transaction(self, transaction: Mapping) -> None:
        if not isinstance(transaction, Mapping):
            raise ValueError("Transaction must be a mapping of feature names to values.")
        missing = [col for col in FEATURE_COLUMNS if col not in transaction]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        for col in FEATURE_COLUMNS:
            val = transaction[col]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Feature '{col}' must be a numeric value, got {type(val).__name__}.")
            if not math.isfinite(val):
                raise ValueError(f"Feature '{col}' must be finite, got {val}.")

    def _compute_contributions(self, transaction: Mapping) -> tuple[int, float, float, list[FeatureContribution]]:
        self._validate_transaction(transaction)
        row = pd.DataFrame([{col: float(transaction[col]) for col in FEATURE_COLUMNS}])

        # 1. Model prediction and fraud probability
        prediction = int(self.pipeline.predict(row)[0])
        prob_idx = fraud_class_index(self.pipeline)
        probability = float(self.pipeline.predict_proba(row)[0, prob_idx])

        # 2. Transform input and compute SHAP values in log-odds space
        transformed = self.preprocessor.transform(row)
        shap_res = self.explainer(transformed)
        shap_values = shap_res.values[0]
        base_value = float(shap_res.base_values[0])

        # 3. Build contribution records for all features
        all_contributions = []
        for i, feat_name in enumerate(self.feature_names):
            contrib = float(shap_values[i])
            orig_val = float(transaction[feat_name])
            direction = "TOWARDS_FRAUD" if contrib > 0 else "TOWARDS_LEGITIMATE"
            all_contributions.append(
                FeatureContribution(
                    feature=feat_name,
                    feature_value=orig_val,
                    contribution=contrib,
                    direction=direction,
                )
            )

        return prediction, probability, base_value, all_contributions

    def get_all_contributions(self, transaction: Mapping) -> list[FeatureContribution]:
        """Return contributions for all features, sorted by absolute contribution descending."""
        _, _, _, all_contributions = self._compute_contributions(transaction)
        return sorted(all_contributions, key=lambda c: abs(c.contribution), reverse=True)

    def explain(self, transaction: Mapping, top_k: int = 5) -> ExplanationResponse:
        """Generate structured explanation and human-readable summary for a transaction."""
        if not isinstance(top_k, int) or top_k < 1 or top_k > len(FEATURE_COLUMNS):
            raise ValueError(
                f"top_k must be an integer between 1 and {len(FEATURE_COLUMNS)}, got {top_k}."
            )

        prediction, probability, base_value, all_contributions = self._compute_contributions(transaction)

        # Rank by descending absolute magnitude
        ranked = sorted(all_contributions, key=lambda c: abs(c.contribution), reverse=True)
        top_features = ranked[:top_k]

        summary = self._generate_summary(prediction, probability, top_features)

        return ExplanationResponse(
            prediction=prediction,
            probability=probability,
            fraud_probability=probability,
            explainer="SHAP LinearExplainer",
            explanation_space="log_odds",
            base_value=base_value,
            top_features=top_features,
            summary=summary,
            explanation=summary,
        )

    @staticmethod
    def _generate_summary(prediction: int, probability: float, top_features: list[FeatureContribution]) -> str:
        pred_label = "FRAUD" if prediction == 1 else "LEGITIMATE"
        pct = probability * 100

        fraud_contributors = [c for c in top_features if c.direction == "TOWARDS_FRAUD"]
        legit_contributors = [c for c in top_features if c.direction == "TOWARDS_LEGITIMATE"]

        parts = [f"The model predicted {pred_label} with a fraud probability of {pct:.2f}%."]

        if fraud_contributors and legit_contributors:
            fraud_names = " and ".join(c.feature for c in fraud_contributors[:2])
            legit_names = " and ".join(c.feature for c in legit_contributors[:2])
            parts.append(
                f"Among the top examined features, {fraud_names} contributed most strongly towards fraud, "
                f"while {legit_names} contributed towards legitimate."
            )
        elif fraud_contributors:
            fraud_names = " and ".join(c.feature for c in fraud_contributors[:3])
            parts.append(f"All top examined features contributed towards fraud, led by {fraud_names}.")
        elif legit_contributors:
            legit_names = " and ".join(c.feature for c in legit_contributors[:3])
            parts.append(f"All top examined features contributed towards legitimate, led by {legit_names}.")

        parts.append(
            "Feature contributions are calculated in log-odds space using SHAP LinearExplainer "
            "and indicate model sensitivity, not causal real-world relationships."
        )

        return " ".join(parts)
