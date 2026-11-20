"""Extended tests for Review Lifecycle (Phase 6).

Covers:
- Pending review can be retrieved.
- Completed review can be retrieved.
- Malformed UUID returns HTTP 422.
- Valid but nonexistent UUID returns HTTP 404.
- Decision for nonexistent review returns HTTP 404.
- Duplicate decision returns HTTP 400 or 409.
- Completed review cannot be modified.
- Original transaction and ML/guardrail results remain unchanged after a human decision.
"""

import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    ReviewAlreadyCompletedError,
    ReviewNotFoundError,
)
from src.human_review.repository import DEFAULT_DB_PATH, ReviewRepository
from src.human_review.service import ReviewService
from tests.api.sample_transactions import LEGITIMATE_TRANSACTION

NONEXISTENT_UUID = "00000000-0000-4000-8000-000000000000"

ORIGINAL_SYSTEM_FIELDS = (
    "created_at",
    "transaction",
    "prediction",
    "probability",
    "risk_level",
    "recommended_action",
    "guardrail_status",
    "guardrail_reasons",
)


@pytest.fixture
def client(tmp_path):
    with TestClient(app) as test_client:
        app.state.review_service = ReviewService(ReviewRepository(tmp_path / "reviews.db"))
        yield test_client
    app.state.review_service = ReviewService(ReviewRepository(DEFAULT_DB_PATH))


@pytest.fixture
def service(tmp_path):
    return ReviewService(ReviewRepository(tmp_path / "service_reviews.db"))


def create_pending_review(client) -> dict:
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    resp = client.post("/reviews", json=payload)
    assert resp.status_code == 201
    return resp.json()


# 1. Pending review can be retrieved


def test_pending_review_can_be_retrieved_api(client):
    created = create_pending_review(client)
    review_id = created["review_id"]

    response = client.get(f"/reviews/{review_id}")
    assert response.status_code == 200
    retrieved = response.json()
    assert retrieved == created
    assert retrieved["review_status"] == REVIEW_PENDING
    assert retrieved["reviewer_id"] is None
    assert retrieved["human_decision"] is None
    assert retrieved["review_reason"] is None
    assert retrieved["reviewed_at"] is None


def test_pending_review_can_be_retrieved_service(service):
    case = service.submit_for_review(
        transaction={"Time": 10.0, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    retrieved = service.get_review(case.review_id)
    assert retrieved == case
    assert retrieved.review_status == REVIEW_PENDING


# 2. Completed review can be retrieved


def test_completed_review_can_be_retrieved_api(client):
    created = create_pending_review(client)
    review_id = created["review_id"]

    decide_resp = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "all good"},
    )
    assert decide_resp.status_code == 200
    decided_payload = decide_resp.json()

    get_resp = client.get(f"/reviews/{review_id}")
    assert get_resp.status_code == 200
    retrieved = get_resp.json()
    assert retrieved == decided_payload
    assert retrieved["review_status"] == REVIEW_COMPLETED
    assert retrieved["human_decision"] == "APPROVE"
    assert retrieved["reviewer_id"] == "alice"
    assert retrieved["review_reason"] == "all good"
    assert retrieved["reviewed_at"] is not None


def test_completed_review_can_be_retrieved_service(service):
    case = service.submit_for_review(
        transaction={"Time": 10.0, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    completed = service.submit_decision(
        review_id=case.review_id,
        reviewer_id="alice",
        decision="REJECT",
        reason="high fraud risk",
    )
    retrieved = service.get_review(case.review_id)
    assert retrieved == completed
    assert retrieved.review_status == REVIEW_COMPLETED
    assert retrieved.human_decision == "REJECT"


# 3. Malformed UUID returns HTTP 422


@pytest.mark.parametrize(
    "malformed_id",
    [
        "123",
        "abc",
        "not-a-uuid",
        "12345678-1234-1234-1234-12345678901",  # 35 chars (one short)
        "12345678-1234-1234-1234-1234567890123",  # 37 chars (one long)
        "12345678-1234-1234-1234-12345678901g",  # non-hex
        "12345678_1234_1234_1234_123456789012",  # underscores instead of hyphens
        "12345678-1234-1234-1234-12345678901Z",  # non-hex char
        " 12345678-1234-1234-1234-123456789012 ",  # leading/trailing spaces
    ],
)
def test_malformed_uuid_returns_422_api(client, malformed_id):
    # GET with malformed UUID
    resp_get = client.get(f"/reviews/{malformed_id}")
    assert resp_get.status_code == 422

    # POST decision with malformed UUID
    resp_post = client.post(
        f"/reviews/{malformed_id}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "ok"},
    )
    assert resp_post.status_code == 422


# 4. Valid but nonexistent UUID returns HTTP 404


def test_valid_nonexistent_uuid_returns_404_api(client):
    random_uuid = str(uuid.uuid4())
    resp = client.get(f"/reviews/{random_uuid}")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_valid_nonexistent_uuid_raises_not_found_service(service):
    random_uuid = str(uuid.uuid4())
    with pytest.raises(ReviewNotFoundError, match="not found"):
        service.get_review(random_uuid)


# 5. Decision for nonexistent review returns HTTP 404


def test_decision_for_nonexistent_review_returns_404_api(client):
    random_uuid = str(uuid.uuid4())
    resp = client.post(
        f"/reviews/{random_uuid}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "ok"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_decision_for_nonexistent_review_raises_not_found_service(service):
    random_uuid = str(uuid.uuid4())
    with pytest.raises(ReviewNotFoundError, match="not found"):
        service.submit_decision(
            review_id=random_uuid,
            reviewer_id="alice",
            decision="APPROVE",
            reason="ok",
        )


# 6. Duplicate decision returns HTTP 400 or 409


def test_duplicate_decision_returns_400_or_409_api(client):
    created = create_pending_review(client)
    review_id = created["review_id"]

    # First decision succeeds
    resp1 = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "first call"},
    )
    assert resp1.status_code == 200

    # Second decision returns 400 or 409 (TrustGuard returns 409 Conflict)
    resp2 = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "bob", "decision": "REJECT", "reason": "second call"},
    )
    assert resp2.status_code in (400, 409)
    assert "already completed" in resp2.json()["detail"].lower()

    # Even identical duplicate decision returns 400 or 409
    resp3 = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "first call"},
    )
    assert resp3.status_code in (400, 409)


def test_duplicate_decision_raises_error_service(service):
    case = service.submit_for_review(
        transaction={"Time": 10.0, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    service.submit_decision(case.review_id, "alice", "APPROVE", "first")
    with pytest.raises(ReviewAlreadyCompletedError, match="already completed"):
        service.submit_decision(case.review_id, "bob", "REJECT", "second")


# 7. Completed review cannot be modified


def test_completed_review_cannot_be_modified_api(client):
    created = create_pending_review(client)
    review_id = created["review_id"]

    # Complete review
    client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "done"},
    )

    # Cannot decide again
    assert client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "carol", "decision": "REJECT", "reason": "tamper"},
    ).status_code in (400, 409)

    # HTTP PUT, PATCH, DELETE are not allowed
    assert client.put(f"/reviews/{review_id}", json={"review_status": "PENDING"}).status_code == 405
    assert client.patch(f"/reviews/{review_id}", json={"human_decision": "REJECT"}).status_code == 405
    assert client.delete(f"/reviews/{review_id}").status_code == 405

    # Verify unchanged
    current = client.get(f"/reviews/{review_id}").json()
    assert current["review_status"] == REVIEW_COMPLETED
    assert current["reviewer_id"] == "alice"
    assert current["human_decision"] == "APPROVE"


# 8. Original transaction and ML/guardrail results remain unchanged after a human decision


def test_original_system_output_remains_unchanged_after_decision_api(client):
    created = create_pending_review(client)
    review_id = created["review_id"]

    # Submit decision (opposite to recommended_action if AUTO_APPROVE)
    decided_resp = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "reviewer_qa", "decision": "REJECT", "reason": "disagree with model"},
    )
    assert decided_resp.status_code == 200
    decided_body = decided_resp.json()

    # Verify in the decision response
    for field in ORIGINAL_SYSTEM_FIELDS:
        assert decided_body[field] == created[field], f"Field {field} changed in decision response"

    # Verify in a subsequent GET request
    get_body = client.get(f"/reviews/{review_id}").json()
    for field in ORIGINAL_SYSTEM_FIELDS:
        assert get_body[field] == created[field], f"Field {field} changed in get response"


def test_original_system_output_remains_unchanged_after_decision_service(service):
    tx = {"Time": 42.0, "Amount": -50.0}
    case = service.submit_for_review(
        transaction=tx,
        prediction=1,
        probability=0.88,
        risk_level="HIGH",
        recommended_action="AUTO_REJECT",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    completed = service.submit_decision(
        review_id=case.review_id,
        reviewer_id="reviewer_senior",
        decision="APPROVE",
        reason="known customer exception",
    )
    for field in (
        "review_id",
        "created_at",
        "transaction",
        "prediction",
        "probability",
        "risk_level",
        "recommended_action",
        "guardrail_status",
        "guardrail_reasons",
    ):
        assert getattr(completed, field) == getattr(case, field), f"Field {field} altered"
