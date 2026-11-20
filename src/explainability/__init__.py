"""Explainability component for TrustGuard (Phase 7).

Explains why the Logistic Regression model produced a prediction and fraud
probability for a given transaction using SHAP LinearExplainer in log-odds space.
"""

from src.explainability.engine import ExplainabilityEngine
from src.explainability.schemas import FeatureContribution, ExplanationResponse

__all__ = [
    "ExplainabilityEngine",
    "FeatureContribution",
    "ExplanationResponse",
]
