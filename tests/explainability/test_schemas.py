"""Tests for explainability Pydantic schemas (Phase 7)."""

import json
import pytest
from pydantic import ValidationError

from src.explainability.schemas import FeatureContribution, ExplanationResponse


def test_valid_feature_contribution():
    fc = FeatureContribution(
        feature="V14",
        feature_value=-4.25,
        contribution=1.35,
        direction="TOWARDS_FRAUD",
    )
    assert fc.feature == "V14"
    assert fc.feature_value == -4.25
    assert fc.contribution == 1.35
    assert fc.direction == "TOWARDS_FRAUD"


@pytest.mark.parametrize("invalid_dir", ["FRAUD", "LEGITIMATE", "TOWARDS_UNKNOWN", ""])
def test_invalid_direction_rejected(invalid_dir):
    with pytest.raises(ValidationError):
        FeatureContribution(
            feature="V14",
            feature_value=-4.25,
            contribution=1.35,
            direction=invalid_dir,
        )


def test_extra_fields_forbidden_on_feature_contribution():
    with pytest.raises(ValidationError):
        FeatureContribution(
            feature="V14",
            feature_value=-4.25,
            contribution=1.35,
            direction="TOWARDS_FRAUD",
            extra="forbidden",
        )


def test_valid_explanation_response():
    data = {
        "prediction": 1,
        "probability": 0.874,
        "fraud_probability": 0.874,
        "explainer": "SHAP LinearExplainer",
        "explanation_space": "log_odds",
        "base_value": -3.95,
        "top_features": [
            {
                "feature": "V14",
                "feature_value": -4.25,
                "contribution": 1.35,
                "direction": "TOWARDS_FRAUD",
            }
        ],
        "summary": "Sample summary",
        "explanation": "Sample summary",
    }
    resp = ExplanationResponse(**data)
    assert resp.prediction == 1
    assert resp.probability == 0.874
    assert resp.fraud_probability == 0.874
    assert len(resp.top_features) == 1
    assert resp.top_features[0].feature == "V14"

    # JSON round-trip
    dumped = json.loads(resp.model_dump_json())
    assert dumped["prediction"] == 1
    assert dumped["fraud_probability"] == 0.874


@pytest.mark.parametrize("bad_prob", [-0.1, 1.1, float("nan")])
def test_invalid_probability_rejected_in_explanation_response(bad_prob):
    with pytest.raises(ValidationError):
        ExplanationResponse(
            prediction=0,
            probability=bad_prob,
            fraud_probability=bad_prob,
            explainer="SHAP LinearExplainer",
            explanation_space="log_odds",
            base_value=0.0,
            top_features=[],
            summary="test",
            explanation="test",
        )


def test_extra_fields_forbidden_on_explanation_response():
    with pytest.raises(ValidationError):
        ExplanationResponse(
            prediction=0,
            probability=0.5,
            fraud_probability=0.5,
            explainer="SHAP LinearExplainer",
            explanation_space="log_odds",
            base_value=0.0,
            top_features=[],
            summary="test",
            explanation="test",
            extra_field="rejected",
        )
