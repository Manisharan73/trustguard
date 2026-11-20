"""Extended tests for Human Decisions (Phase 6).

Covers:
- APPROVE decision works.
- REJECT decision works.
- Invalid decision values are rejected.
- PENDING is not accepted as a human decision.
- Empty reviewer ID is rejected.
- Whitespace-only reviewer ID is rejected.
- Empty reason is rejected.
- Whitespace-only reason is rejected.
- Reviewer ID and reason length constraints are enforced.
- Unexpected request fields are handled according to the existing schema policy.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    ReviewValidationError,
)
from src.human_review.repository import DEFAULT_DB_PATH, ReviewRepository
from src.human_review.schemas import DecisionRequest
from src.human_review.service import (
    MAX_REASON_LENGTH,
    MAX_REVIEWER_ID_LENGTH,
    ReviewService,
)
from tests.api.sample_transactions import LEGITIMATE_TRANSACTION


@pytest.fixture
def client(tmp_path):
    with TestClient(app) as test_client:
        app.state.review_service = ReviewService(ReviewRepository(tmp_path / "reviews.db"))
        yield test_client
    app.state.review_service = ReviewService(ReviewRepository(DEFAULT_DB_PATH))


@pytest.fixture
def service(tmp_path):
    return ReviewService(ReviewRepository(tmp_path / "service_reviews.db"))


def make_pending_review(client) -> str:
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    resp = client.post("/reviews", json=payload)
    assert resp.status_code == 201
    return resp.json()["review_id"]


def make_pending_case(service):
    return service.submit_for_review(
        transaction={"Time": 1.0, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )


# 1. APPROVE decision works


def test_approve_decision_works_api(client):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "reviewer_01", "decision": "APPROVE", "reason": "Verified manually"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_status"] == REVIEW_COMPLETED
    assert body["human_decision"] == "APPROVE"
    assert body["reviewer_id"] == "reviewer_01"
    assert body["review_reason"] == "Verified manually"
    assert body["reviewed_at"] is not None


def test_approve_decision_works_service(service):
    case = make_pending_case(service)
    completed = service.submit_decision(
        review_id=case.review_id,
        reviewer_id="reviewer_01",
        decision="APPROVE",
        reason="Verified identity",
    )
    assert completed.review_status == REVIEW_COMPLETED
    assert completed.human_decision == "APPROVE"
    assert completed.reviewer_id == "reviewer_01"
    assert completed.review_reason == "Verified identity"


# 2. REJECT decision works


def test_reject_decision_works_api(client):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "reviewer_02", "decision": "REJECT", "reason": "Suspicious transaction"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_status"] == REVIEW_COMPLETED
    assert body["human_decision"] == "REJECT"
    assert body["reviewer_id"] == "reviewer_02"
    assert body["review_reason"] == "Suspicious transaction"
    assert body["reviewed_at"] is not None


def test_reject_decision_works_service(service):
    case = make_pending_case(service)
    completed = service.submit_decision(
        review_id=case.review_id,
        reviewer_id="reviewer_02",
        decision="REJECT",
        reason="Known fraudulent pattern",
    )
    assert completed.review_status == REVIEW_COMPLETED
    assert completed.human_decision == "REJECT"
    assert completed.reviewer_id == "reviewer_02"
    assert completed.review_reason == "Known fraudulent pattern"


# 3. Invalid decision values are rejected


@pytest.mark.parametrize(
    "invalid_decision",
    ["APPROVED", "REJECTED", "DECLINE", "PASS", "FAIL", "MAYBE", "approve", "reject", "Approve", "", 123, True],
)
def test_invalid_decision_values_rejected_api(client, invalid_decision):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "rev1", "decision": invalid_decision, "reason": "test"},
    )
    assert response.status_code == 422
    # Ensure review stays PENDING
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


@pytest.mark.parametrize(
    "invalid_decision",
    ["APPROVED", "REJECTED", "DECLINE", "PASS", "FAIL", "MAYBE", "approve", "reject", "Approve", "", 123, True],
)
def test_invalid_decision_values_rejected_service(service, invalid_decision):
    case = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="decision"):
        service.submit_decision(
            review_id=case.review_id,
            reviewer_id="rev1",
            decision=invalid_decision,
            reason="test",
        )
    assert service.get_review(case.review_id).review_status == REVIEW_PENDING


@pytest.mark.parametrize(
    "invalid_decision",
    ["APPROVED", "REJECTED", "DECLINE", "PASS", "FAIL", "MAYBE", "approve", "reject", "Approve", "", 123, True],
)
def test_invalid_decision_values_rejected_schema(invalid_decision):
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer_id="rev1", decision=invalid_decision, reason="test")


# 4. PENDING is not accepted as a human decision


def test_pending_not_accepted_as_human_decision_api(client):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "rev1", "decision": "PENDING", "reason": "leaving it pending"},
    )
    assert response.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


def test_pending_not_accepted_as_human_decision_service(service):
    case = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="decision"):
        service.submit_decision(
            review_id=case.review_id,
            reviewer_id="rev1",
            decision="PENDING",
            reason="leaving it pending",
        )
    assert service.get_review(case.review_id).review_status == REVIEW_PENDING


def test_pending_not_accepted_as_human_decision_schema():
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer_id="rev1", decision="PENDING", reason="leaving it pending")


# 5. Empty reviewer ID is rejected


def test_empty_reviewer_id_rejected_api(client):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "", "decision": "APPROVE", "reason": "Valid reason"},
    )
    assert response.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


def test_empty_reviewer_id_rejected_service(service):
    case = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="reviewer_id"):
        service.submit_decision(case.review_id, reviewer_id="", decision="APPROVE", reason="Valid reason")


def test_empty_reviewer_id_rejected_schema():
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer_id="", decision="APPROVE", reason="Valid reason")


# 6. Whitespace-only reviewer ID is rejected


@pytest.mark.parametrize("ws_reviewer_id", ["   ", "\t", "\n\r\t", "       "])
def test_whitespace_only_reviewer_id_rejected_api(client, ws_reviewer_id):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": ws_reviewer_id, "decision": "APPROVE", "reason": "Valid reason"},
    )
    assert response.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


@pytest.mark.parametrize("ws_reviewer_id", ["   ", "\t", "\n\r\t", "       "])
def test_whitespace_only_reviewer_id_rejected_service(service, ws_reviewer_id):
    case = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="reviewer_id"):
        service.submit_decision(case.review_id, reviewer_id=ws_reviewer_id, decision="APPROVE", reason="Valid reason")


@pytest.mark.parametrize("ws_reviewer_id", ["   ", "\t", "\n\r\t", "       "])
def test_whitespace_only_reviewer_id_rejected_schema(ws_reviewer_id):
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer_id=ws_reviewer_id, decision="APPROVE", reason="Valid reason")


# 7. Empty reason is rejected


def test_empty_reason_rejected_api(client):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "rev1", "decision": "APPROVE", "reason": ""},
    )
    assert response.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


def test_empty_reason_rejected_service(service):
    case = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="reason"):
        service.submit_decision(case.review_id, reviewer_id="rev1", decision="APPROVE", reason="")


def test_empty_reason_rejected_schema():
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer_id="rev1", decision="APPROVE", reason="")


# 8. Whitespace-only reason is rejected


@pytest.mark.parametrize("ws_reason", ["   ", "\t", "\n\t  ", "      \n  "])
def test_whitespace_only_reason_rejected_api(client, ws_reason):
    review_id = make_pending_review(client)
    response = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "rev1", "decision": "APPROVE", "reason": ws_reason},
    )
    assert response.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


@pytest.mark.parametrize("ws_reason", ["   ", "\t", "\n\t  ", "      \n  "])
def test_whitespace_only_reason_rejected_service(service, ws_reason):
    case = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="reason"):
        service.submit_decision(case.review_id, reviewer_id="rev1", decision="APPROVE", reason=ws_reason)


@pytest.mark.parametrize("ws_reason", ["   ", "\t", "\n\t  ", "      \n  "])
def test_whitespace_only_reason_rejected_schema(ws_reason):
    with pytest.raises(ValidationError):
        DecisionRequest(reviewer_id="rev1", decision="APPROVE", reason=ws_reason)


# 9. Reviewer ID and reason length constraints are enforced


def test_reviewer_id_max_length_enforced_api(client):
    review_id = make_pending_review(client)
    # Exactly MAX_REVIEWER_ID_LENGTH characters (100) -> accepted
    valid_id = "r" * MAX_REVIEWER_ID_LENGTH
    resp = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": valid_id, "decision": "APPROVE", "reason": "ok"},
    )
    assert resp.status_code == 200
    assert resp.json()["reviewer_id"] == valid_id

    # 101 characters -> rejected
    review_id_2 = make_pending_review(client)
    too_long_id = "r" * (MAX_REVIEWER_ID_LENGTH + 1)
    resp2 = client.post(
        f"/reviews/{review_id_2}/decision",
        json={"reviewer_id": too_long_id, "decision": "APPROVE", "reason": "ok"},
    )
    assert resp2.status_code == 422
    assert client.get(f"/reviews/{review_id_2}").json()["review_status"] == REVIEW_PENDING


def test_reviewer_id_max_length_enforced_service(service):
    case = make_pending_case(service)
    # 100 characters accepted
    c = service.submit_decision(
        case.review_id, reviewer_id="r" * MAX_REVIEWER_ID_LENGTH, decision="APPROVE", reason="ok"
    )
    assert len(c.reviewer_id) == MAX_REVIEWER_ID_LENGTH

    # 101 characters rejected
    case2 = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="reviewer_id"):
        service.submit_decision(
            case2.review_id, reviewer_id="r" * (MAX_REVIEWER_ID_LENGTH + 1), decision="APPROVE", reason="ok"
        )


def test_reason_max_length_enforced_api(client):
    review_id = make_pending_review(client)
    # Exactly MAX_REASON_LENGTH characters (2000) -> accepted
    valid_reason = "m" * MAX_REASON_LENGTH
    resp = client.post(
        f"/reviews/{review_id}/decision",
        json={"reviewer_id": "rev1", "decision": "APPROVE", "reason": valid_reason},
    )
    assert resp.status_code == 200
    assert resp.json()["review_reason"] == valid_reason

    # 2001 characters -> rejected
    review_id_2 = make_pending_review(client)
    too_long_reason = "m" * (MAX_REASON_LENGTH + 1)
    resp2 = client.post(
        f"/reviews/{review_id_2}/decision",
        json={"reviewer_id": "rev1", "decision": "APPROVE", "reason": too_long_reason},
    )
    assert resp2.status_code == 422
    assert client.get(f"/reviews/{review_id_2}").json()["review_status"] == REVIEW_PENDING


def test_reason_max_length_enforced_service(service):
    case = make_pending_case(service)
    # 2000 characters accepted
    c = service.submit_decision(
        case.review_id, reviewer_id="rev1", decision="APPROVE", reason="m" * MAX_REASON_LENGTH
    )
    assert len(c.review_reason) == MAX_REASON_LENGTH

    # 2001 characters rejected
    case2 = make_pending_case(service)
    with pytest.raises(ReviewValidationError, match="reason"):
        service.submit_decision(
            case2.review_id, reviewer_id="rev1", decision="APPROVE", reason="m" * (MAX_REASON_LENGTH + 1)
        )


# 10. Unexpected request fields are handled according to existing schema policy (extra="forbid")


@pytest.mark.parametrize(
    "extra_field",
    [
        {"extra": "unexpected"},
        {"comment": "should be forbidden"},
        {"prediction": 1},
        {"probability": 0.5},
        {"final_decision": "APPROVE"},
        {"risk_level": "HIGH"},
        {"guardrail_status": "PASSED"},
        {"status": "COMPLETED"},
    ],
)
def test_unexpected_request_fields_rejected_api(client, extra_field):
    review_id = make_pending_review(client)
    payload = {"reviewer_id": "rev1", "decision": "APPROVE", "reason": "ok"}
    payload.update(extra_field)
    response = client.post(f"/reviews/{review_id}/decision", json=payload)
    assert response.status_code == 422
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == REVIEW_PENDING


@pytest.mark.parametrize(
    "extra_field",
    [
        {"extra": "unexpected"},
        {"comment": "should be forbidden"},
        {"final_decision": "APPROVE"},
        {"prediction": 1},
    ],
)
def test_unexpected_request_fields_rejected_schema(extra_field):
    payload = {"reviewer_id": "rev1", "decision": "APPROVE", "reason": "ok"}
    payload.update(extra_field)
    with pytest.raises(ValidationError):
        DecisionRequest(**payload)
