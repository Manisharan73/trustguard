"""Tests for the Phase 2 prediction API.

Requires the trained model: run `python -m src.ml.train` first.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.decision.engine import DecisionConfig, recommend_action
from src.ml.predict import predict_transaction
from src.ml.preprocess import FEATURE_COLUMNS
from src.risk.engine import RiskConfig, assess_risk
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


@pytest.fixture(scope="module")
def client():
    # Using the client as a context manager runs the lifespan (model loading)
    with TestClient(app) as test_client:
        yield test_client


# ---------- normal behaviour ----------

def test_legitimate_transaction_is_predicted_legitimate(client):
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert 0.0 <= body["probability"] < 0.5


def test_fraud_transaction_is_predicted_fraud(client):
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert 0.5 < body["probability"] <= 1.0


def test_api_matches_direct_model_call(client):
    expected = predict_transaction(LEGITIMATE_TRANSACTION)
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["prediction"] == expected["prediction"]
    assert body["probability"] == pytest.approx(expected["probability"])


# ---------- architectural constraint ----------

def test_response_has_model_output_and_risk_level_but_no_decision(client):
    # API contract: model output, risk level, a RECOMMENDED action and (Phase 5)
    # the guardrail verdict on that recommendation. "No decision" means there is
    # no FINAL decision: recommended_action is only the system's recommendation.
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert set(body.keys()) == {
        "prediction",
        "probability",
        "risk_level",
        "recommended_action",
        "guardrail_status",
        "guardrail_reasons",
    }
    assert "final_decision" not in body
    assert body["recommended_action"] == recommend_action(
        body["risk_level"], DecisionConfig()
    )


# ---------- Phase 3: risk integration ----------

def test_legitimate_transaction_has_low_risk(client):
    body = client.post("/predict", json=LEGITIMATE_TRANSACTION).json()
    assert body["risk_level"] == "LOW"


def test_fraud_transaction_has_high_risk(client):
    body = client.post("/predict", json=FRAUD_TRANSACTION).json()
    assert body["risk_level"] == "HIGH"


def test_risk_level_is_consistent_with_probability(client):
    body = client.post("/predict", json=FRAUD_TRANSACTION).json()
    assert body["risk_level"] == assess_risk(body["probability"], RiskConfig())


def test_risk_does_not_change_the_prediction(client):
    expected = predict_transaction(FRAUD_TRANSACTION)
    body = client.post("/predict", json=FRAUD_TRANSACTION).json()
    assert body["prediction"] == expected["prediction"]
    assert body["probability"] == pytest.approx(expected["probability"])


# ---------- schema / training consistency ----------

def test_sample_transactions_have_exactly_the_training_features():
    assert list(LEGITIMATE_TRANSACTION.keys()) == FEATURE_COLUMNS
    assert list(FRAUD_TRANSACTION.keys()) == FEATURE_COLUMNS


# ---------- invalid input is rejected ----------

def test_missing_feature_is_rejected(client):
    payload = dict(LEGITIMATE_TRANSACTION)
    del payload["V12"]
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_body_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_non_numeric_value_is_rejected(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount="not a number")
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_extra_field_is_rejected(client):
    payload = dict(LEGITIMATE_TRANSACTION, merchant="ACME")
    assert client.post("/predict", json=payload).status_code == 422


def test_nan_value_is_rejected(client):
    payload = dict(LEGITIMATE_TRANSACTION, V1=float("nan"))
    response = client.post(
        "/predict",
        content=json.dumps(payload),  # produces a bare NaN token
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_wrong_http_method_is_rejected(client):
    assert client.get("/predict").status_code == 405
