"""Extended tests for Review Creation (Phase 6).

Covers:
- Review is created when guardrails return REQUIRES_REVIEW.
- Review creation is rejected when guardrails return PASSED.
- HUMAN_REVIEW recommendation correctly creates a review.
- AUTO_APPROVE with guardrail failure creates a review.
- AUTO_REJECT with guardrail failure creates a review.
- Two identical transactions can create independent reviews, unless duplicate
  prevention is an explicitly implemented requirement.
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.decision.engine import DecisionConfig
from src.guardrails.engine import GuardrailConfig
from src.human_review.models import REVIEW_PENDING, ReviewNotRequiredError
from src.human_review.repository import DEFAULT_DB_PATH, ReviewRepository
from src.human_review.service import ReviewService
from src.risk.engine import RiskConfig
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


@pytest.fixture
def client(tmp_path):
    with TestClient(app) as test_client:
        app.state.review_service = ReviewService(ReviewRepository(tmp_path / "reviews.db"))
        yield test_client
    app.state.risk_config = RiskConfig()
    app.state.decision_config = DecisionConfig()
    app.state.guardrail_config = GuardrailConfig()
    app.state.review_service = ReviewService(ReviewRepository(DEFAULT_DB_PATH))


@pytest.fixture
def service(tmp_path):
    return ReviewService(ReviewRepository(tmp_path / "service_reviews.db"))


# 1. Review is created when guardrails return REQUIRES_REVIEW


def test_review_created_when_guardrails_return_requires_review_api(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-10.0)
    response = client.post("/reviews", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert "NEGATIVE_AMOUNT" in body["guardrail_reasons"]
    assert body["review_status"] == REVIEW_PENDING
    assert body["review_id"] is not None


def test_review_created_when_guardrails_return_requires_review_service(service):
    case = service.submit_for_review(
        transaction={"Time": 10.0, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    assert case.guardrail_status == "REQUIRES_REVIEW"
    assert case.review_status == REVIEW_PENDING
    assert service.get_review(case.review_id) == case


# 2. Review creation is rejected when guardrails return PASSED


def test_review_creation_rejected_when_guardrails_return_passed_legitimate_api(client):
    # Legitimate transaction passes guardrails and has AUTO_APPROVE -> not eligible for review
    response = client.post("/reviews", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 422
    assert "does not require human review" in response.json()["detail"]
    assert client.get("/reviews/pending").json() == []


def test_review_creation_rejected_when_guardrails_return_passed_fraud_api(client):
    # Fraud transaction passes guardrails and has AUTO_REJECT -> not eligible for review
    response = client.post("/reviews", json=FRAUD_TRANSACTION)
    assert response.status_code == 422
    assert "does not require human review" in response.json()["detail"]
    assert client.get("/reviews/pending").json() == []


def test_review_creation_rejected_when_guardrails_return_passed_service(service):
    # When guardrails passed and action is not HUMAN_REVIEW, service rejects creation
    with pytest.raises(ReviewNotRequiredError, match="does not require human review"):
        service.submit_for_review(
            transaction={"Time": 10.0, "Amount": 100.0},
            prediction=0,
            probability=0.01,
            risk_level="LOW",
            recommended_action="AUTO_APPROVE",
            guardrail_status="PASSED",
            guardrail_reasons=[],
        )

    with pytest.raises(ReviewNotRequiredError, match="does not require human review"):
        service.submit_for_review(
            transaction={"Time": 10.0, "Amount": 100.0},
            prediction=1,
            probability=0.95,
            risk_level="HIGH",
            recommended_action="AUTO_REJECT",
            guardrail_status="PASSED",
            guardrail_reasons=[],
        )
    assert service.list_pending() == []


# 3. HUMAN_REVIEW recommendation correctly creates a review


def test_human_review_recommendation_creates_review_api(client):
    # Adjust thresholds so legitimate transaction falls into MEDIUM risk -> HUMAN_REVIEW
    app.state.risk_config = RiskConfig(low_threshold=0.0001, high_threshold=0.5)
    response = client.post("/reviews", json=LEGITIMATE_TRANSACTION)
    assert response.status_code == 201
    body = response.json()
    assert body["recommended_action"] == "HUMAN_REVIEW"
    assert body["risk_level"] == "MEDIUM"
    assert body["review_status"] == REVIEW_PENDING


def test_human_review_recommendation_creates_review_service(service):
    # Even if guardrail_status is PASSED, recommended_action=HUMAN_REVIEW creates review
    case = service.submit_for_review(
        transaction={"Time": 50.0, "Amount": 75.0},
        prediction=0,
        probability=0.35,
        risk_level="MEDIUM",
        recommended_action="HUMAN_REVIEW",
        guardrail_status="PASSED",
        guardrail_reasons=[],
    )
    assert case.recommended_action == "HUMAN_REVIEW"
    assert case.review_status == REVIEW_PENDING
    assert service.get_review(case.review_id) == case


# 4. AUTO_APPROVE with guardrail failure creates a review


def test_auto_approve_with_guardrail_failure_creates_review_api(client):
    # Low risk transaction with negative amount triggers guardrail failure
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-25.0)
    response = client.post("/reviews", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["recommended_action"] == "AUTO_APPROVE"
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert "NEGATIVE_AMOUNT" in body["guardrail_reasons"]
    assert body["review_status"] == REVIEW_PENDING


def test_auto_approve_with_guardrail_failure_creates_review_service(service):
    case = service.submit_for_review(
        transaction={"Time": 10.0, "Amount": -25.0},
        prediction=0,
        probability=0.005,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    assert case.recommended_action == "AUTO_APPROVE"
    assert case.guardrail_status == "REQUIRES_REVIEW"
    assert case.review_status == REVIEW_PENDING


# 5. AUTO_REJECT with guardrail failure creates a review


def test_auto_reject_with_guardrail_failure_creates_review_api(client):
    # Fraud transaction (normally AUTO_REJECT) with guardrail failure (e.g. negative Time)
    payload = dict(FRAUD_TRANSACTION, Time=-50.0)
    response = client.post("/reviews", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["recommended_action"] == "AUTO_REJECT"
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert "NEGATIVE_TIME" in body["guardrail_reasons"]
    assert body["review_status"] == REVIEW_PENDING


def test_auto_reject_with_guardrail_failure_creates_review_service(service):
    case = service.submit_for_review(
        transaction={"Time": -50.0, "Amount": 100.0},
        prediction=1,
        probability=0.99,
        risk_level="HIGH",
        recommended_action="AUTO_REJECT",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_TIME"],
    )
    assert case.recommended_action == "AUTO_REJECT"
    assert case.guardrail_status == "REQUIRES_REVIEW"
    assert case.review_status == REVIEW_PENDING


# 6. Two identical transactions can create independent reviews


def test_identical_transactions_create_independent_reviews_api(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)

    # Submit first transaction
    resp1 = client.post("/reviews", json=payload)
    assert resp1.status_code == 201
    review_1 = resp1.json()

    # Submit identical second transaction
    resp2 = client.post("/reviews", json=payload)
    assert resp2.status_code == 201
    review_2 = resp2.json()

    # Must generate distinct review IDs
    assert review_1["review_id"] != review_2["review_id"]

    # Both reviews appear in the pending queue
    pending_ids = [r["review_id"] for r in client.get("/reviews/pending").json()]
    assert review_1["review_id"] in pending_ids
    assert review_2["review_id"] in pending_ids

    # Deciding review_1 leaves review_2 pending
    decide_resp = client.post(
        f"/reviews/{review_1['review_id']}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "approved first"},
    )
    assert decide_resp.status_code == 200

    pending_ids_after = [r["review_id"] for r in client.get("/reviews/pending").json()]
    assert review_1["review_id"] not in pending_ids_after
    assert review_2["review_id"] in pending_ids_after

    # Review 2 can receive an independent decision
    decide_resp_2 = client.post(
        f"/reviews/{review_2['review_id']}/decision",
        json={"reviewer_id": "bob", "decision": "REJECT", "reason": "rejected second"},
    )
    assert decide_resp_2.status_code == 200

    # Verify both records in the database retain their separate decisions
    r1_final = client.get(f"/reviews/{review_1['review_id']}").json()
    r2_final = client.get(f"/reviews/{review_2['review_id']}").json()
    assert r1_final["human_decision"] == "APPROVE"
    assert r1_final["reviewer_id"] == "alice"
    assert r2_final["human_decision"] == "REJECT"
    assert r2_final["reviewer_id"] == "bob"


def test_identical_transactions_create_independent_reviews_service(service):
    tx = {"Time": 10.0, "Amount": -5.0}
    c1 = service.submit_for_review(
        transaction=tx,
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    c2 = service.submit_for_review(
        transaction=tx,
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    assert c1.review_id != c2.review_id
    assert {c1.review_id, c2.review_id} == {c.review_id for c in service.list_pending()}
