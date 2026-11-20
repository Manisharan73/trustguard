"""Phase 5 API integration tests: guardrail_status / guardrail_reasons in /predict.

Requires the trained model: run `python -m src.ml.train` first.

Test isolation: the `client` fixture is function-scoped, so every test enters the
app lifespan and starts from freshly created default configs. Overrides are plain
assignments to app.state (the same operation the lifespan performs).
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.decision.engine import DecisionConfig, recommend_action
from src.guardrails.engine import GuardrailConfig
from src.ml.predict import predict_transaction
from src.risk.engine import RiskConfig, assess_risk
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION

# Under this risk policy the legitimate sample (p ~ 0.006) is MEDIUM risk.
MEDIUM_FOR_LEGIT_SAMPLE = RiskConfig(low_threshold=0.001, high_threshold=0.5)

EXPECTED_FIELDS = {
    "prediction",
    "probability",
    "risk_level",
    "recommended_action",
    "guardrail_status",
    "guardrail_reasons",
}


@pytest.fixture
def client():
    with TestClient(app) as test_client:  # runs the lifespan for this test
        yield test_client
    # Leave the app in its default configuration for whatever runs next.
    app.state.risk_config = RiskConfig()
    app.state.decision_config = DecisionConfig()
    app.state.guardrail_config = GuardrailConfig()


def post(client, transaction):
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    return response.json()


# ---------- startup and configuration lifecycle ----------

def test_startup_initialises_guardrail_config_with_defaults(client):
    assert app.state.guardrail_config == GuardrailConfig()


def test_a_new_client_starts_with_default_guardrail_config():
    # Overrides made earlier must not survive into a freshly started application.
    app.state.guardrail_config = GuardrailConfig(amount_field="V1")
    with TestClient(app) as fresh_client:
        assert app.state.guardrail_config == GuardrailConfig()
        body = post(fresh_client, LEGITIMATE_TRANSACTION)
    assert body["guardrail_status"] == "PASSED"
    assert body["guardrail_reasons"] == []


# ---------- response structure ----------

def test_response_has_exactly_the_expected_fields(client):
    assert set(post(client, LEGITIMATE_TRANSACTION)) == EXPECTED_FIELDS


def test_guardrail_reasons_is_a_list_of_strings(client):
    body = post(client, LEGITIMATE_TRANSACTION)
    assert isinstance(body["guardrail_reasons"], list)
    assert all(isinstance(reason, str) for reason in body["guardrail_reasons"])


def test_response_never_claims_a_transaction_was_decided(client):
    body = post(client, LEGITIMATE_TRANSACTION)
    assert "final_decision" not in body
    assert not any("final" in key for key in body)


# ---------- normal behaviour ----------

def test_legitimate_transaction_passes_guardrails(client):
    body = post(client, LEGITIMATE_TRANSACTION)
    assert body["recommended_action"] == "AUTO_APPROVE"
    assert body["guardrail_status"] == "PASSED"
    assert body["guardrail_reasons"] == []


def test_fraud_transaction_passes_guardrails(client):
    body = post(client, FRAUD_TRANSACTION)
    assert body["recommended_action"] == "AUTO_REJECT"
    assert body["guardrail_status"] == "PASSED"
    assert body["guardrail_reasons"] == []


def test_human_review_recommendation_requires_review(client):
    app.state.risk_config = MEDIUM_FOR_LEGIT_SAMPLE
    body = post(client, LEGITIMATE_TRANSACTION)
    assert body["risk_level"] == "MEDIUM"
    assert body["recommended_action"] == "HUMAN_REVIEW"
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["HUMAN_REVIEW_REQUIRED"]


# ---------- guardrails evaluate the Decision Engine's recommendation ----------
# A recommendation produced by a (deliberately inconsistent) decision policy is
# escalated, which also shows the guardrails run AFTER the Decision Engine and
# look at ITS output.

def test_inconsistent_auto_approve_requires_review_without_changing_it(client):
    app.state.decision_config = DecisionConfig(
        policy={"LOW": "AUTO_APPROVE", "MEDIUM": "HUMAN_REVIEW", "HIGH": "AUTO_APPROVE"}
    )
    body = post(client, FRAUD_TRANSACTION)  # prediction 1, HIGH risk
    assert body["recommended_action"] == "AUTO_APPROVE"  # NOT changed by the guardrail
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["APPROVAL_CONTEXT_MISMATCH"]
    assert body["prediction"] == 1
    assert body["risk_level"] == "HIGH"


def test_inconsistent_auto_reject_requires_review_without_changing_it(client):
    app.state.decision_config = DecisionConfig(
        policy={"LOW": "AUTO_REJECT", "MEDIUM": "HUMAN_REVIEW", "HIGH": "AUTO_REJECT"}
    )
    body = post(client, LEGITIMATE_TRANSACTION)  # prediction 0, LOW risk
    assert body["recommended_action"] == "AUTO_REJECT"  # NOT changed by the guardrail
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["REJECTION_CONTEXT_MISMATCH"]
    assert body["prediction"] == 0
    assert body["risk_level"] == "LOW"


# ---------- data quality: values Pydantic accepts but guardrails escalate ----------

def test_negative_amount_requires_review_without_changing_the_recommendation(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)  # finite, so the schema accepts it
    body = post(client, payload)
    assert body["recommended_action"] == "AUTO_APPROVE"  # unchanged
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]


def test_negative_time_requires_review_without_changing_the_recommendation(client):
    payload = dict(LEGITIMATE_TRANSACTION, Time=-1.0)
    body = post(client, payload)
    assert body["recommended_action"] == "AUTO_APPROVE"  # unchanged
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["NEGATIVE_TIME"]


def test_negative_amount_on_a_rejected_transaction_still_requires_review(client):
    payload = dict(FRAUD_TRANSACTION, Amount=-5.0)
    body = post(client, payload)
    assert body["recommended_action"] == "AUTO_REJECT"
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]


def test_api_uses_the_configured_guardrail_config(client):
    # Point the "amount" check at V1, which is negative (-0.67) in this sample.
    app.state.guardrail_config = GuardrailConfig(amount_field="V1")
    body = post(client, LEGITIMATE_TRANSACTION)
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]


# ---------- original outputs are unchanged by the guardrails ----------

@pytest.mark.parametrize(
    "payload", [LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION, dict(LEGITIMATE_TRANSACTION, Amount=-5.0)]
)
def test_prediction_probability_risk_and_action_match_the_earlier_stages(client, payload):
    expected = predict_transaction(payload)
    expected_risk = assess_risk(expected["probability"], RiskConfig())
    expected_action = recommend_action(expected_risk, DecisionConfig())

    body = post(client, payload)

    assert body["prediction"] == expected["prediction"]
    assert body["probability"] == pytest.approx(expected["probability"])
    assert body["risk_level"] == expected_risk
    assert body["recommended_action"] == expected_action


# ---------- existing request validation is intact ----------

def test_missing_feature_is_still_rejected_without_guardrail_fields(client):
    payload = dict(LEGITIMATE_TRANSACTION)
    del payload["V12"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "guardrail_status" not in response.json()


def test_nan_is_still_rejected_with_a_valid_json_422(client):
    payload = dict(LEGITIMATE_TRANSACTION, V1=float("nan"))
    response = client.post(
        "/predict",
        content=json.dumps(payload),  # produces a bare NaN token
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_infinity_is_still_rejected_with_a_valid_json_422(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=float("inf"))
    response = client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
