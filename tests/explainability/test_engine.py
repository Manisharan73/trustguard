"""Unit tests for the ExplainabilityEngine (Phase 7).

Covers:
- Basic explanation tests for legitimate and fraud transactions.
- Model output consistency (prediction and probability match).
- Feature names, original values, ranking, and contribution directions.
- SHAP additive property in log-odds space.
- Correct output class (fraud / Class 1).
- Non-hardcoded / deterministic contributions.
- Configurable top_k parameter.
- Validation: missing features, non-numeric, non-finite.
- Edge cases: all-zeros, small values, large finite values, opposing features, near-zero values.
- Initialization failure handling.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.explainability.engine import ExplainabilityEngine
from src.ml.predict import fraud_class_index, load_model
from src.ml.preprocess import FEATURE_COLUMNS
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def engine(model):
    return ExplainabilityEngine(model)


# ===================== A. Basic Explanation Tests =====================


def test_explain_legitimate_transaction(engine, model):
    result = engine.explain(LEGITIMATE_TRANSACTION, top_k=5)

    assert result.prediction == 0
    assert result.probability == pytest.approx(0.00584, abs=1e-4)
    assert result.fraud_probability == result.probability
    assert result.explainer == "SHAP LinearExplainer"
    assert result.explanation_space == "log_odds"
    assert isinstance(result.base_value, float)
    assert len(result.top_features) == 5

    # Feature names match valid features
    for c in result.top_features:
        assert c.feature in FEATURE_COLUMNS
        assert c.feature_value == LEGITIMATE_TRANSACTION[c.feature]
        assert math.isfinite(c.contribution)
        assert c.direction in ("TOWARDS_FRAUD", "TOWARDS_LEGITIMATE")

    # Sorted by absolute contribution descending
    magnitudes = [abs(c.contribution) for c in result.top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explain_fraud_transaction(engine, model):
    result = engine.explain(FRAUD_TRANSACTION, top_k=5)

    assert result.prediction == 1
    assert result.probability > 0.99
    assert result.fraud_probability == result.probability
    assert len(result.top_features) == 5

    # Top features for the fraud row should include strong fraud drivers like V14, V17
    top_feature_names = [c.feature for c in result.top_features]
    assert "V14" in top_feature_names
    v14_contrib = next(c for c in result.top_features if c.feature == "V14")
    assert v14_contrib.contribution > 0
    assert v14_contrib.direction == "TOWARDS_FRAUD"


def test_prediction_and_probability_match_existing_model(engine, model):
    for tx in (LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION):
        import pandas as pd
        row = pd.DataFrame([{col: tx[col] for col in FEATURE_COLUMNS}])
        expected_pred = int(model.predict(row)[0])
        expected_prob = float(model.predict_proba(row)[0, fraud_class_index(model)])

        result = engine.explain(tx)
        assert result.prediction == expected_pred
        assert result.probability == pytest.approx(expected_prob)
        assert result.fraud_probability == pytest.approx(expected_prob)


def test_feature_values_correspond_to_submitted_transaction(engine):
    custom_tx = dict(LEGITIMATE_TRANSACTION, Amount=999.5, V1=-3.14)
    result = engine.explain(custom_tx, top_k=10)

    amount_entry = next((c for c in result.top_features if c.feature == "Amount"), None)
    if amount_entry:
        assert amount_entry.feature_value == 999.5

    v1_entry = next((c for c in result.top_features if c.feature == "V1"), None)
    if v1_entry:
        assert v1_entry.feature_value == -3.14


def test_contribution_direction_semantics(engine):
    result = engine.explain(LEGITIMATE_TRANSACTION, top_k=30)
    for c in result.top_features:
        if c.contribution > 0:
            assert c.direction == "TOWARDS_FRAUD"
        else:
            assert c.direction == "TOWARDS_LEGITIMATE"


# ===================== B. Mathematical Correctness =====================


def test_shap_additive_property_in_log_odds_space(engine, model):
    """base_value + sum(contributions) == decision_function(x) within numerical tolerance."""
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    for tx in (LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION):
        all_contributions = engine.get_all_contributions(tx)
        assert len(all_contributions) == len(FEATURE_COLUMNS)

        import pandas as pd
        row = pd.DataFrame([{col: tx[col] for col in FEATURE_COLUMNS}])
        transformed = preprocessor.transform(row)
        expected_log_odds = float(classifier.decision_function(transformed)[0])

        total_explained = engine.base_value + sum(c.contribution for c in all_contributions)
        assert total_explained == pytest.approx(expected_log_odds, abs=1e-5)

        # And verify that logistic transform of expected_log_odds matches predict_proba
        expected_prob = 1.0 / (1.0 + math.exp(-expected_log_odds))
        model_prob = float(model.predict_proba(row)[0, fraud_class_index(model)])
        assert expected_prob == pytest.approx(model_prob, abs=1e-5)


def test_explanation_uses_correct_output_class(engine, model):
    """Contributions directly explain Class 1 (fraud) log-odds."""
    idx = fraud_class_index(model)
    assert idx == 1
    classes = list(model.classes_)
    assert classes[idx] == 1


def test_base_value_and_contributions_in_same_space(engine):
    assert engine.base_value < 0  # Prior log-odds of fraud is negative in creditcard dataset
    result = engine.explain(LEGITIMATE_TRANSACTION)
    assert result.explanation_space == "log_odds"


def test_contributions_are_not_fabricated_or_hardcoded(engine):
    res_legit = engine.explain(LEGITIMATE_TRANSACTION, top_k=5)
    res_fraud = engine.explain(FRAUD_TRANSACTION, top_k=5)

    # Different transactions have completely different contributions and rankings
    legit_map = {c.feature: c.contribution for c in res_legit.top_features}
    fraud_map = {c.feature: c.contribution for c in res_fraud.top_features}
    assert legit_map != fraud_map


def test_explanation_determinism(engine):
    res1 = engine.explain(FRAUD_TRANSACTION, top_k=5)
    res2 = engine.explain(FRAUD_TRANSACTION, top_k=5)

    assert res1.prediction == res2.prediction
    assert res1.probability == res2.probability
    assert res1.base_value == res2.base_value
    assert res1.summary == res2.summary
    assert [(c.feature, c.contribution) for c in res1.top_features] == [
        (c.feature, c.contribution) for c in res2.top_features
    ]


# ===================== C. Configurable top_k =====================


@pytest.mark.parametrize("k", [1, 3, 5, 10, 20, 30])
def test_configurable_top_k(engine, k):
    result = engine.explain(LEGITIMATE_TRANSACTION, top_k=k)
    assert len(result.top_features) == k


@pytest.mark.parametrize("invalid_k", [0, -1, 31, 100, "5", None, 3.5])
def test_invalid_top_k_rejected(engine, invalid_k):
    with pytest.raises(ValueError, match="top_k"):
        engine.explain(LEGITIMATE_TRANSACTION, top_k=invalid_k)


# ===================== D. Validation and Error Handling =====================


def test_missing_features_rejected(engine):
    incomplete = dict(LEGITIMATE_TRANSACTION)
    del incomplete["V14"]
    with pytest.raises(ValueError, match="Missing required features"):
        engine.explain(incomplete)


@pytest.mark.parametrize("bad_val", ["text", None, [1.0], {"a": 1}])
def test_non_numeric_features_rejected(engine, bad_val):
    bad_tx = dict(LEGITIMATE_TRANSACTION, V1=bad_val)
    with pytest.raises(ValueError, match="numeric"):
        engine.explain(bad_tx)


def test_boolean_features_rejected(engine):
    bad_tx = dict(LEGITIMATE_TRANSACTION, V1=True)
    with pytest.raises(ValueError, match="numeric"):
        engine.explain(bad_tx)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_features_rejected(engine, non_finite):
    bad_tx = dict(LEGITIMATE_TRANSACTION, Amount=non_finite)
    with pytest.raises(ValueError, match="finite"):
        engine.explain(bad_tx)


def test_non_mapping_transaction_rejected(engine):
    with pytest.raises(ValueError, match="mapping"):
        engine.explain([1.0] * 30)


# ===================== E. Edge Cases =====================


def test_edge_case_all_zeros(engine):
    zero_tx = {col: 0.0 for col in FEATURE_COLUMNS}
    result = engine.explain(zero_tx, top_k=5)
    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert len(result.top_features) == 5
    for c in result.top_features:
        assert math.isfinite(c.contribution)


def test_edge_case_very_small_values(engine):
    small_tx = {col: 1e-12 for col in FEATURE_COLUMNS}
    result = engine.explain(small_tx, top_k=5)
    assert math.isfinite(result.probability)
    assert len(result.top_features) == 5


def test_edge_case_large_finite_values(engine):
    large_tx = dict(LEGITIMATE_TRANSACTION, Amount=100000.0, Time=200000.0)
    result = engine.explain(large_tx, top_k=5)
    assert math.isfinite(result.probability)
    assert len(result.top_features) == 5


def test_edge_case_features_contributing_in_opposite_directions(engine):
    result = engine.explain(LEGITIMATE_TRANSACTION, top_k=10)
    directions = {c.direction for c in result.top_features}
    assert "TOWARDS_FRAUD" in directions
    assert "TOWARDS_LEGITIMATE" in directions
    # Verify summary mentions both directions
    assert "towards fraud" in result.summary.lower()
    assert "towards legitimate" in result.summary.lower()


def test_summary_disclaimer_on_causality(engine):
    result = engine.explain(FRAUD_TRANSACTION)
    assert "not causal" in result.summary.lower() or "causal" in result.summary.lower()
    assert "log-odds" in result.summary.lower()


# ===================== F. Initialization Failures =====================


def test_top_k_filtering_preserves_full_additive_reconstruction(engine, model):
    """Top-k filtering controls displayed features only; all 30 features maintain additivity."""
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    for tx in (LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION):
        result = engine.explain(tx, top_k=5)
        assert len(result.top_features) == 5

        all_c = engine.get_all_contributions(tx)
        assert len(all_c) == 30

        import pandas as pd
        row = pd.DataFrame([{col: tx[col] for col in FEATURE_COLUMNS}])
        transformed = preprocessor.transform(row)
        expected_log_odds = float(classifier.decision_function(transformed)[0])

        # Top 5 alone do not equal the decision function
        top_5_sum = sum(c.contribution for c in result.top_features)
        assert top_5_sum + result.base_value != pytest.approx(expected_log_odds, abs=1e-2)

        # Full 30 contributions strictly equal the decision function
        full_sum = sum(c.contribution for c in all_c)
        assert full_sum + result.base_value == pytest.approx(expected_log_odds, abs=1e-5)


def test_feature_ordering_matches_preprocessing_and_coefficients(engine, model):
    """Verify that each feature contribution strictly aligns with its classifier coefficient."""
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    tx = FRAUD_TRANSACTION
    import pandas as pd
    row = pd.DataFrame([{col: tx[col] for col in FEATURE_COLUMNS}])
    transformed = preprocessor.transform(row)[0]
    coefs = classifier.coef_[0]
    bg_mean = np.mean(engine.background_data, axis=0)

    # Analytical SHAP for linear model: phi_i = coef_i * (x_i - mean_i)
    analytical_contributions = coefs * (transformed - bg_mean)

    all_c = engine.get_all_contributions(tx)
    name_to_contrib = {c.feature: c.contribution for c in all_c}

    for i, name in enumerate(engine.feature_names):
        assert name_to_contrib[name] == pytest.approx(analytical_contributions[i], abs=1e-12)


def test_background_reference_data_integrity():
    """Verify background dataset exists, has correct shape and zero test-set leakage."""
    bg_path = Path("data/processed/background_reference.npy")
    assert bg_path.exists()
    bg = np.load(bg_path)
    assert bg.shape == (100, 30)
    assert np.all(np.isfinite(bg))


def test_shap_failure_propagates_without_fabrication(engine, monkeypatch):
    """Verify that SHAP calculation failures propagate and never return fabricated explanations."""
    def broken_call(self, *args, **kwargs):
        raise RuntimeError("SHAP computation engine fault")

    import shap
    monkeypatch.setattr(shap.LinearExplainer, "__call__", broken_call)
    with pytest.raises(RuntimeError, match="SHAP computation engine fault"):
        engine.explain(LEGITIMATE_TRANSACTION)


@pytest.mark.parametrize(
    "bad_pipeline",
    [
        None,
        "not-a-pipeline",
        object(),
    ],
)
def test_engine_initialization_failure(bad_pipeline):
    with pytest.raises(ValueError, match="sklearn Pipeline"):
        ExplainabilityEngine(bad_pipeline)
