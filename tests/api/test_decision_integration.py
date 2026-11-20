"""Phase 4 API integration tests: recommended_action in /predict.

Requires the trained model: run `python -m src.ml.train` first.

Test isolation
--------------
The `client` fixture is function-scoped. Entering `TestClient(app)` as a context
manager runs the app lifespan, which creates the model, RiskConfig and
DecisionConfig from scratch. So EVERY test starts from the default policies, and
a test that overrides a config cannot leak that override into the next test.
Overrides are plain assignments to app.state, the same operation the lifespan
itself performs.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.decision.engine import (
    VALID_ACTIONS,
    DecisionConfig,
    recommend_action,
)
from src.ml.predict import predict_transaction
from src.risk.engine import RiskConfig, assess_risk
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION

# Under this policy the legitimate sample (p ~ 0.006) is MEDIUM risk.
MEDIUM_FOR_LEGIT_SAMPLE = RiskConfig(low_threshold=0.001, high_threshold=0.5)


@pytest.fixture
def client():
    with TestClient(app) as test_client:  # runs the lifespan for this test
        yield test_client
    # Leave the app in its default configuration for whatever runs next.
    app.state.risk_config = RiskConfig()
    app.state.decision_config = DecisionConfig()


# ---------- application startup ----------

def test_startup_initialises_both_configs_with_default_policies(client):
    assert app.state.risk_config == RiskConfig()
    assert app.state.decision_config == DecisionConfig()


def test_a_new_client_starts_from_defaults_even_after_overrides():
    # Regression test for the state leak that broke three tests: overrides made
    # earlier must not survive into a freshly started application.
    app.state.risk_config = MEDIUM_FOR_LEGIT_SAMPLE
    app.state.decision_config = DecisionConfig(
        policy={"LOW": "HUMAN_REVIEW", "MEDIUM": "HUMAN_REVIEW", "HIGH": "AUTO_REJECT"}
    )
    with TestClient(app) as fresh_client:
        assert app.state.risk_config == RiskConfig()
        assert app.state.decision_config == DecisionConfig()
        body = fresh_client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["risk_level"] == "LOW"
    assert body["recommended_action"] == "AUTO_APPROVE"


# ---------- response structure ----------

def test_response_has_exactly_the_expected_fields(client):
    # Phase 5 added the two guardrail fields; the Phase 4 fields are all still present.
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert set(body.keys()) == {
        "prediction",
        "probability",
        "risk_level",
        "recommended_action",
        "guardrail_status",
        "guardrail_reasons",
    }


def test_recommended_action_is_a_valid_action(client):
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["recommended_action"] in VALID_ACTIONS


# ---------- default policy: risk level -> recommended action ----------

def test_low_risk_transaction_is_auto_approved(client):
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["risk_level"] == "LOW"
    assert body["recommended_action"] == "AUTO_APPROVE"


def test_high_risk_transaction_is_auto_rejected(client):
    body = client.post("/predict", json=FRAUD_TRANSACTION).json()
    assert body["risk_level"] == "HIGH"
    assert body["recommended_action"] == "AUTO_REJECT"


def test_recommended_action_matches_decision_engine(client):
    for transaction in (LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION):
        body = client.post("/predict", json=transaction).json()
        assert body["recommended_action"] == recommend_action(
            body["risk_level"], DecisionConfig()
        )


def test_medium_risk_goes_to_human_review(client):
    app.state.risk_config = MEDIUM_FOR_LEGIT_SAMPLE
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["risk_level"] == "MEDIUM"
    assert body["recommended_action"] == "HUMAN_REVIEW"


# ---------- the API really uses the configured policies ----------

def test_api_uses_the_configured_decision_policy(client):
    app.state.decision_config = DecisionConfig(
        policy={"LOW": "HUMAN_REVIEW", "MEDIUM": "HUMAN_REVIEW", "HIGH": "AUTO_REJECT"}
    )
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["risk_level"] == "LOW"
    assert body["recommended_action"] == "HUMAN_REVIEW"


# ---------- existing behaviour is preserved ----------

def test_prediction_probability_and_risk_are_unchanged(client):
    expected = predict_transaction(FRAUD_TRANSACTION)
    body = client.post("/predict", json=FRAUD_TRANSACTION).json()
    assert body["prediction"] == expected["prediction"]
    assert body["probability"] == pytest.approx(expected["probability"])
    assert body["risk_level"] == assess_risk(body["probability"], RiskConfig())


def test_recommended_action_does_not_change_the_prediction(client):
    # Even when the action is HUMAN_REVIEW, the model prediction stays 0.
    app.state.risk_config = MEDIUM_FOR_LEGIT_SAMPLE
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["recommended_action"] == "HUMAN_REVIEW"
    assert body["prediction"] == 0


def test_missing_feature_is_still_rejected_without_an_action(client):
    payload = dict(LEGITIMATE_TRANSACTION)
    del payload["V12"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "recommended_action" not in response.json()


def test_nan_is_still_rejected_with_a_valid_json_422(client):
    # Regression guard for the NaN serialization bug fixed by the custom
    # RequestValidationError handler in src/api/main.py.
    payload = dict(LEGITIMATE_TRANSACTION, V1=float("nan"))
    response = client.post(
        "/predict",
        content=json.dumps(payload),  # produces a bare NaN token
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)  # body is valid JSON


def test_infinity_is_still_rejected_with_a_valid_json_422(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=float("inf"))
    response = client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
