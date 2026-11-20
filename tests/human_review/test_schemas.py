"""Tests for the human review Pydantic schemas (requires pydantic)."""

import json
import typing

import pytest
from pydantic import ValidationError

from src.human_review.models import VALID_DECISIONS, ReviewCase
from src.human_review.schemas import DecisionRequest, ReviewResponse


def valid_body(**overrides):
    body = {"reviewer_id": "alice", "decision": "APPROVE", "reason": "verified"}
    body.update(overrides)
    return body


def test_decision_literals_match_the_domain_constants():
    assert set(typing.get_args(DecisionRequest.model_fields["decision"].annotation)) == set(VALID_DECISIONS)


def test_a_valid_decision_request_is_accepted():
    request = DecisionRequest(**valid_body())
    assert (request.reviewer_id, request.decision, request.reason) == ("alice", "APPROVE", "verified")


def test_surrounding_whitespace_is_stripped():
    request = DecisionRequest(**valid_body(reviewer_id="  alice ", reason="\n ok \t"))
    assert (request.reviewer_id, request.reason) == ("alice", "ok")


@pytest.mark.parametrize(
    "overrides",
    [
        {"reviewer_id": ""}, {"reviewer_id": "   "}, {"reason": ""}, {"reason": " \n\t "},
        {"decision": "MAYBE"}, {"decision": "approve"}, {"decision": ""},
        {"reviewer_id": "x" * 101}, {"reason": "x" * 2001},
        {"reviewer_id": None}, {"reason": None}, {"decision": None},
        {"reviewer_id": 5}, {"reason": ["a"]},
    ],
)
def test_invalid_decision_requests_are_rejected_with_json_serializable_errors(overrides):
    with pytest.raises(ValidationError) as caught:
        DecisionRequest(**valid_body(**overrides))
    json.dumps(caught.value.errors(include_url=False))  # must not raise


@pytest.mark.parametrize("missing", ["reviewer_id", "decision", "reason"])
def test_missing_fields_are_rejected(missing):
    body = valid_body()
    del body[missing]
    with pytest.raises(ValidationError):
        DecisionRequest(**body)


@pytest.mark.parametrize(
    "extra",
    [{"prediction": 1}, {"probability": 0.1}, {"risk_level": "LOW"},
     {"recommended_action": "AUTO_APPROVE"}, {"guardrail_status": "PASSED"}, {"review_status": "PENDING"}],
)
def test_a_reviewer_cannot_supply_original_fields(extra):
    with pytest.raises(ValidationError):
        DecisionRequest(**valid_body(**extra))


def test_review_response_round_trips_a_case():
    case = ReviewCase(
        review_id="11111111-1111-4111-8111-111111111111",
        created_at="2026-01-01T00:00:00+00:00",
        review_status="COMPLETED",
        transaction={"Amount": -5.0},
        prediction=0, probability=0.0073, risk_level="LOW",
        recommended_action="AUTO_APPROVE", guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=("NEGATIVE_AMOUNT",),
        reviewer_id="alice", human_decision="REJECT", review_reason="x",
        reviewed_at="2026-01-02T00:00:00+00:00",
    )
    body = ReviewResponse.from_case(case).model_dump()
    assert body["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]
    assert body["human_decision"] == "REJECT" and body["recommended_action"] == "AUTO_APPROVE"
    assert not any("final" in key for key in body)
