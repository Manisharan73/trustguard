"""Tests for the shared four-stage pipeline (no FastAPI needed).

Requires the trained model: run `python -m src.ml.train` first.
"""

import copy
from types import SimpleNamespace

import pytest

from src.api.pipeline import run_pipeline
from src.decision.engine import DecisionConfig, recommend_action
from src.guardrails.engine import GuardrailConfig, evaluate_guardrails
from src.ml.predict import load_model, predict_transaction
from src.risk.engine import RiskConfig, assess_risk
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION

EXPECTED_FIELDS = {
    "prediction",
    "probability",
    "risk_level",
    "recommended_action",
    "guardrail_status",
    "guardrail_reasons",
}


@pytest.fixture
def state():
    return SimpleNamespace(
        model=load_model(),
        risk_config=RiskConfig(),
        decision_config=DecisionConfig(),
        guardrail_config=GuardrailConfig(),
    )


def test_returns_exactly_the_six_response_fields(state):
    assert set(run_pipeline(dict(LEGITIMATE_TRANSACTION), state)) == EXPECTED_FIELDS


def test_legitimate_transaction_is_approved_and_passes(state):
    out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state)
    assert (out["prediction"], out["risk_level"]) == (0, "LOW")
    assert out["recommended_action"] == "AUTO_APPROVE"
    assert (out["guardrail_status"], out["guardrail_reasons"]) == ("PASSED", [])


def test_fraud_transaction_is_rejected_and_passes(state):
    out = run_pipeline(dict(FRAUD_TRANSACTION), state)
    assert (out["prediction"], out["risk_level"]) == (1, "HIGH")
    assert out["recommended_action"] == "AUTO_REJECT"
    assert (out["guardrail_status"], out["guardrail_reasons"]) == ("PASSED", [])


def test_negative_amount_is_escalated_without_changing_the_recommendation(state):
    out = run_pipeline(dict(LEGITIMATE_TRANSACTION, Amount=-5.0), state)
    assert out["recommended_action"] == "AUTO_APPROVE"
    assert out["guardrail_status"] == "REQUIRES_REVIEW"
    assert out["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]


@pytest.mark.parametrize("transaction", [LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION])
def test_pipeline_equals_the_engines_called_one_by_one(state, transaction):
    expected = predict_transaction(dict(transaction), state.model)
    risk = assess_risk(expected["probability"], state.risk_config)
    action = recommend_action(risk, state.decision_config)
    guardrails = evaluate_guardrails(
        dict(transaction), expected["prediction"], risk, action, state.guardrail_config
    )
    out = run_pipeline(dict(transaction), state)
    assert out["prediction"] == expected["prediction"]
    assert out["probability"] == pytest.approx(expected["probability"])
    assert out["risk_level"] == risk
    assert out["recommended_action"] == action
    assert out["guardrail_status"] == guardrails.status
    assert out["guardrail_reasons"] == list(guardrails.reasons)


def test_configs_come_from_the_state_object(state):
    state.risk_config = RiskConfig(low_threshold=0.001, high_threshold=0.5)
    out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state)
    assert (out["risk_level"], out["recommended_action"]) == ("MEDIUM", "HUMAN_REVIEW")
    assert out["guardrail_reasons"] == ["HUMAN_REVIEW_REQUIRED"]


def test_guardrail_config_comes_from_the_state_object(state):
    state.guardrail_config = GuardrailConfig(amount_field="V1")  # V1 is negative here
    out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state)
    assert out["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]


def test_input_features_are_not_modified(state):
    features = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    before = copy.deepcopy(features)
    run_pipeline(features, state)
    assert features == before


def test_there_is_no_final_decision_in_the_output(state):
    out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state)
    assert not any("final" in key for key in out)
