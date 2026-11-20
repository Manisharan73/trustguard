"""API integration tests for the /explain endpoint (Phase 7).

Covers:
- POST /explain with legitimate and fraud transactions.
- Verification of response fields (prediction, probability, explainer, top_features, summary).
- Query parameter top_k configuration and rejection of invalid top_k.
- Rejection of missing features, extra fields, non-numeric values, NaN, and Infinity.
- Regression: POST /predict remains unchanged in schema and behavior.
- Consistency between /predict and /explain.
- Concurrency safety of /explain.
"""

import json
import threading
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


# ===================== A. Endpoint Success Cases =====================


def test_explain_legitimate_transaction_api(client):
    response = client.post("/explain", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == 0
    assert 0.0 <= body["probability"] <= 0.05
    assert body["fraud_probability"] == body["probability"]
    assert body["explainer"] == "SHAP LinearExplainer"
    assert body["explanation_space"] == "log_odds"
    assert isinstance(body["base_value"], float)
    assert len(body["top_features"]) == 5
    assert "summary" in body and len(body["summary"]) > 0
    assert body["explanation"] == body["summary"]


def test_explain_fraud_transaction_api(client):
    response = client.post("/explain", json=FRAUD_TRANSACTION)
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == 1
    assert body["probability"] > 0.95
    assert body["fraud_probability"] == body["probability"]
    assert len(body["top_features"]) == 5

    # Check top fraud features
    feature_names = [f["feature"] for f in body["top_features"]]
    assert "V14" in feature_names
    v14 = next(f for f in body["top_features"] if f["feature"] == "V14")
    assert v14["direction"] == "TOWARDS_FRAUD"
    assert v14["contribution"] > 0


def test_response_contains_expected_fields(client):
    response = client.post("/explain", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    expected_fields = {
        "prediction",
        "probability",
        "fraud_probability",
        "explainer",
        "explanation_space",
        "base_value",
        "top_features",
        "summary",
        "explanation",
    }
    assert set(response.json().keys()) == expected_fields


# ===================== B. Query Parameter top_k =====================


@pytest.mark.parametrize("k", [1, 3, 7, 15, 30])
def test_explain_with_valid_top_k(client, k):
    response = client.post(f"/explain?top_k={k}", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    assert len(response.json()["top_features"]) == k


@pytest.mark.parametrize("bad_k", [0, -1, 31, "abc", "five"])
def test_explain_with_invalid_top_k_rejected(client, bad_k):
    response = client.post(f"/explain?top_k={bad_k}", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 422


# ===================== C. Request Validation =====================


def test_missing_features_rejected(client):
    incomplete = dict(LEGITIMATE_TRANSACTION)
    del incomplete["V12"]
    response = client.post("/explain", json=incomplete)
    assert response.status_code == 422


def test_extra_fields_rejected(client):
    extra = dict(LEGITIMATE_TRANSACTION, unexpected_feature=123.45)
    response = client.post("/explain", json=extra)
    assert response.status_code == 422


def test_non_numeric_features_rejected(client):
    invalid = dict(LEGITIMATE_TRANSACTION, Amount="not_a_number")
    response = client.post("/explain", json=invalid)
    assert response.status_code == 422


def test_nan_rejected_with_clean_422(client):
    nan_payload = dict(LEGITIMATE_TRANSACTION, V1=float("nan"))
    response = client.post(
        "/explain",
        content=json.dumps(nan_payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_infinity_rejected_with_clean_422(client):
    inf_payload = dict(LEGITIMATE_TRANSACTION, Amount=float("inf"))
    response = client.post(
        "/explain",
        content=json.dumps(inf_payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


# ===================== D. Regression & Consistency =====================


def test_existing_predict_endpoint_remains_unchanged(client):
    response = client.post("/predict", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "prediction",
        "probability",
        "risk_level",
        "recommended_action",
        "guardrail_status",
        "guardrail_reasons",
    }
    assert body["prediction"] == 0
    assert body["risk_level"] == "LOW"
    assert body["recommended_action"] == "AUTO_APPROVE"
    assert body["guardrail_status"] == "PASSED"


def test_prediction_and_probability_match_between_predict_and_explain(client):
    for tx in (LEGITIMATE_TRANSACTION, FRAUD_TRANSACTION):
        pred_resp = client.post("/predict", json=tx).json()
        expl_resp = client.post("/explain", json=tx).json()

        assert pred_resp["prediction"] == expl_resp["prediction"]
        assert pred_resp["probability"] == pytest.approx(expl_resp["probability"])
        assert pred_resp["probability"] == pytest.approx(expl_resp["fraud_probability"])


def test_concurrent_explain_requests(client):
    results = []

    def call():
        resp = client.post("/explain", json=LEGITIMATE_TRANSACTION)
        results.append(resp.status_code)

    threads = [threading.Thread(target=call) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 10
    assert all(code == 200 for code in results)


def test_uninitialized_explainability_engine_returns_503():
    with TestClient(app) as test_client:
        test_client.app.state.explainability_engine = None
        resp = test_client.post("/explain", json=LEGITIMATE_TRANSACTION)
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"].lower()


def test_phase6_reviews_remain_unaffected(client):
    # Ensure Phase 6 review queue and decisions are fully functional alongside /explain
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    rev_resp = client.post("/reviews", json=payload)
    assert rev_resp.status_code == 201
    review_id = rev_resp.json()["review_id"]

    decide_resp = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "auditor_1", "decision": "APPROVE", "reason": "audit verified"},
    )
    assert decide_resp.status_code == 200
    assert decide_resp.json()["review_status"] == "COMPLETED"
