"""Tests for the human review workflow rules, using real temporary SQLite files."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    ReviewAlreadyCompletedError,
    ReviewNotFoundError,
    ReviewNotRequiredError,
    ReviewValidationError,
)
from src.human_review.repository import ReviewRepository
from src.human_review.service import MAX_REASON_LENGTH, MAX_REVIEWER_ID_LENGTH, ReviewService

UNKNOWN_ID = "99999999-9999-4999-8999-999999999999"


class FakeClock:
    """Deterministic clock: every call returns a time one minute later."""

    def __init__(self):
        self._now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        self._now += timedelta(minutes=1)
        return self._now


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "reviews.db"


@pytest.fixture
def service(db_path):
    return ReviewService(ReviewRepository(db_path), clock=FakeClock())


def system_output(**overrides) -> dict:
    """What the pipeline would hand over for a transaction needing review."""
    values = dict(
        transaction={"Time": 10.0, "V1": 0.5, "Amount": -5.0},
        prediction=0,
        probability=0.0073,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    values.update(overrides)
    return values


# ======================= A. review submission =======================

def test_submit_creates_a_pending_case_preserving_the_system_output(service):
    case = service.submit_for_review(**system_output())
    assert case.review_status == REVIEW_PENDING
    assert case.prediction == 0
    assert case.probability == 0.0073
    assert case.risk_level == "LOW"
    assert case.recommended_action == "AUTO_APPROVE"
    assert case.guardrail_status == "REQUIRES_REVIEW"
    assert case.guardrail_reasons == ("NEGATIVE_AMOUNT",)
    assert case.transaction == {"Time": 10.0, "V1": 0.5, "Amount": -5.0}


def test_a_new_case_has_no_human_decision_yet(service):
    case = service.submit_for_review(**system_output())
    assert case.reviewer_id is None
    assert case.human_decision is None
    assert case.review_reason is None
    assert case.reviewed_at is None


def test_review_ids_are_unique_and_valid_uuids(service):
    ids = [service.submit_for_review(**system_output()).review_id for _ in range(60)]
    assert len(set(ids)) == 60
    for review_id in ids:
        assert str(uuid.UUID(review_id)) == review_id


def test_created_at_comes_from_the_clock(service):
    case = service.submit_for_review(**system_output())
    assert case.created_at == "2026-01-01T00:01:00+00:00"


def test_the_case_is_persisted(service):
    case = service.submit_for_review(**system_output())
    assert service.get_review(case.review_id) == case


def test_a_guardrail_that_requires_review_is_eligible(service):
    case = service.submit_for_review(**system_output(guardrail_status="REQUIRES_REVIEW"))
    assert case.review_status == REVIEW_PENDING


def test_a_human_review_recommendation_is_eligible_even_if_guardrails_passed(service):
    case = service.submit_for_review(
        **system_output(recommended_action="HUMAN_REVIEW", risk_level="MEDIUM",
                        guardrail_status="PASSED", guardrail_reasons=[])
    )
    assert case.review_status == REVIEW_PENDING


def test_a_transaction_that_needs_no_review_is_refused_and_not_stored(service):
    with pytest.raises(ReviewNotRequiredError):
        service.submit_for_review(**system_output(guardrail_status="PASSED", guardrail_reasons=[]))
    assert service.list_pending() == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"prediction": 2}, {"prediction": True}, {"prediction": "0"}, {"prediction": None},
        {"probability": 1.5}, {"probability": -0.1}, {"probability": float("nan")},
        {"probability": True}, {"probability": "0.5"},
        {"risk_level": "EXTREME"}, {"risk_level": None},
        {"recommended_action": "SHRUG"}, {"recommended_action": None},
        {"guardrail_status": "MAYBE"}, {"guardrail_status": None},
        {"guardrail_reasons": "NEGATIVE_AMOUNT"}, {"guardrail_reasons": [1, 2]},
        {"transaction": {}}, {"transaction": None}, {"transaction": [1.0]},
    ],
)
def test_invalid_system_output_raises_value_error_and_stores_nothing(service, overrides):
    with pytest.raises(ValueError):
        service.submit_for_review(**system_output(**overrides))
    assert service.list_pending() == []


def test_the_callers_transaction_is_copied_not_shared(service):
    transaction = {"Time": 10.0, "Amount": -5.0}
    case = service.submit_for_review(**system_output(transaction=transaction))
    transaction["Amount"] = 999.0
    assert case.transaction["Amount"] == -5.0


# ======================= B. pending queue =======================

def test_an_empty_queue_returns_an_empty_list(service):
    assert service.list_pending() == []


def test_multiple_pending_reviews_are_listed_oldest_first(service):
    ids = [service.submit_for_review(**system_output()).review_id for _ in range(3)]
    assert [c.review_id for c in service.list_pending()] == ids


def test_completed_reviews_are_excluded_from_the_pending_queue(service):
    first = service.submit_for_review(**system_output())
    second = service.submit_for_review(**system_output())
    service.submit_decision(first.review_id, "rev-1", "APPROVE", "fine")
    assert [c.review_id for c in service.list_pending()] == [second.review_id]


# ======================= C. human decisions =======================

@pytest.mark.parametrize("decision", ["APPROVE", "REJECT"])
def test_a_reviewer_can_approve_or_reject(service, decision):
    case = service.submit_for_review(**system_output())
    done = service.submit_decision(case.review_id, "rev-1", decision, "checked the history")
    assert done.review_status == REVIEW_COMPLETED
    assert done.human_decision == decision


def test_reviewer_identity_and_reason_are_stored(service):
    case = service.submit_for_review(**system_output())
    done = service.submit_decision(case.review_id, "alice.smith", "REJECT", "amount is impossible")
    assert done.reviewer_id == "alice.smith"
    assert done.review_reason == "amount is impossible"


def test_reviewer_id_and_reason_are_stored_stripped(service):
    case = service.submit_for_review(**system_output())
    done = service.submit_decision(case.review_id, "  alice  ", "APPROVE", "\n  verified by phone \t")
    assert done.reviewer_id == "alice"
    assert done.review_reason == "verified by phone"


def test_the_review_timestamp_is_recorded_after_creation(service):
    case = service.submit_for_review(**system_output())
    done = service.submit_decision(case.review_id, "rev-1", "APPROVE", "ok")
    assert done.reviewed_at == "2026-01-01T00:02:00+00:00"
    assert datetime.fromisoformat(done.reviewed_at) > datetime.fromisoformat(done.created_at)


def test_a_completed_review_can_be_retrieved_with_its_decision(service):
    case = service.submit_for_review(**system_output())
    done = service.submit_decision(case.review_id, "rev-1", "APPROVE", "ok")
    assert service.get_review(case.review_id) == done


def test_the_human_decision_is_independent_of_the_model_recommendation(service):
    # Recommended AUTO_APPROVE, reviewer REJECTS; recommended AUTO_REJECT, reviewer APPROVES.
    approve_case = service.submit_for_review(**system_output())
    reject_case = service.submit_for_review(
        **system_output(prediction=1, probability=0.99, risk_level="HIGH",
                        recommended_action="AUTO_REJECT", guardrail_status="REQUIRES_REVIEW")
    )
    a = service.submit_decision(approve_case.review_id, "rev-1", "REJECT", "disagree with the model")
    b = service.submit_decision(reject_case.review_id, "rev-1", "APPROVE", "known good customer")
    assert (a.recommended_action, a.human_decision) == ("AUTO_APPROVE", "REJECT")
    assert (b.recommended_action, b.human_decision) == ("AUTO_REJECT", "APPROVE")


# ======================= D. validation =======================

@pytest.mark.parametrize(
    "bad_reviewer",
    ["", "   ", "\t\n", None, 123, ["rev"], "x" * (MAX_REVIEWER_ID_LENGTH + 1)],
)
def test_invalid_reviewer_id_is_rejected(service, bad_reviewer):
    case = service.submit_for_review(**system_output())
    with pytest.raises(ReviewValidationError, match="reviewer_id"):
        service.submit_decision(case.review_id, bad_reviewer, "APPROVE", "reason")


@pytest.mark.parametrize("bad_decision", [None, "", "approve", "reject", "MAYBE", "APPROVE ", 1, True])
def test_invalid_decision_is_rejected(service, bad_decision):
    case = service.submit_for_review(**system_output())
    with pytest.raises(ReviewValidationError, match="decision"):
        service.submit_decision(case.review_id, "rev-1", bad_decision, "reason")


@pytest.mark.parametrize(
    "bad_reason",
    ["", "   ", "\n\t  ", None, 5, ["reason"], "x" * (MAX_REASON_LENGTH + 1)],
)
def test_invalid_reason_is_rejected(service, bad_reason):
    case = service.submit_for_review(**system_output())
    with pytest.raises(ReviewValidationError, match="reason"):
        service.submit_decision(case.review_id, "rev-1", "APPROVE", bad_reason)


def test_a_rejected_decision_leaves_the_review_pending(service):
    case = service.submit_for_review(**system_output())
    with pytest.raises(ReviewValidationError):
        service.submit_decision(case.review_id, "rev-1", "APPROVE", "   ")
    assert service.get_review(case.review_id).review_status == REVIEW_PENDING
    assert [c.review_id for c in service.list_pending()] == [case.review_id]


def test_validation_errors_are_value_errors():
    assert issubclass(ReviewValidationError, ValueError)


def test_the_longest_allowed_values_are_accepted(service):
    case = service.submit_for_review(**system_output())
    done = service.submit_decision(
        case.review_id, "r" * MAX_REVIEWER_ID_LENGTH, "APPROVE", "x" * MAX_REASON_LENGTH
    )
    assert len(done.reviewer_id) == MAX_REVIEWER_ID_LENGTH
    assert len(done.review_reason) == MAX_REASON_LENGTH


# ======================= E. workflow edge cases =======================

def test_getting_an_unknown_review_raises_not_found(service):
    with pytest.raises(ReviewNotFoundError):
        service.get_review(UNKNOWN_ID)


def test_deciding_an_unknown_review_raises_not_found(service):
    with pytest.raises(ReviewNotFoundError):
        service.submit_decision(UNKNOWN_ID, "rev-1", "APPROVE", "reason")


def test_a_review_can_only_be_decided_once(service):
    case = service.submit_for_review(**system_output())
    service.submit_decision(case.review_id, "rev-1", "APPROVE", "first")
    with pytest.raises(ReviewAlreadyCompletedError):
        service.submit_decision(case.review_id, "rev-2", "REJECT", "second")


def test_resubmitting_the_same_decision_is_also_refused(service):
    case = service.submit_for_review(**system_output())
    service.submit_decision(case.review_id, "rev-1", "APPROVE", "first")
    with pytest.raises(ReviewAlreadyCompletedError):
        service.submit_decision(case.review_id, "rev-1", "APPROVE", "first")


def test_the_first_decision_survives_a_second_attempt(service):
    case = service.submit_for_review(**system_output())
    first = service.submit_decision(case.review_id, "rev-1", "APPROVE", "first")
    with pytest.raises(ReviewAlreadyCompletedError):
        service.submit_decision(case.review_id, "rev-2", "REJECT", "second")
    assert service.get_review(case.review_id) == first


def test_original_system_output_is_unchanged_by_the_decision(service):
    created = service.submit_for_review(
        **system_output(prediction=1, probability=0.91, risk_level="HIGH",
                        recommended_action="AUTO_REJECT", guardrail_status="REQUIRES_REVIEW",
                        guardrail_reasons=["NEGATIVE_TIME", "REJECTION_CONTEXT_MISMATCH"])
    )
    decided = service.submit_decision(created.review_id, "rev-1", "APPROVE", "override")
    assert decided.prediction == created.prediction == 1
    assert decided.probability == created.probability == 0.91
    assert decided.risk_level == created.risk_level == "HIGH"
    assert decided.recommended_action == created.recommended_action == "AUTO_REJECT"
    assert decided.guardrail_status == created.guardrail_status == "REQUIRES_REVIEW"
    assert decided.guardrail_reasons == created.guardrail_reasons
    assert decided.transaction == created.transaction
    assert decided.created_at == created.created_at


def test_decisions_persist_after_the_database_is_reopened(db_path):
    first_service = ReviewService(ReviewRepository(db_path), clock=FakeClock())
    case = first_service.submit_for_review(**system_output())
    done = first_service.submit_decision(case.review_id, "rev-1", "REJECT", "fraud ring")
    pending = first_service.submit_for_review(**system_output())
    del first_service

    restarted = ReviewService(ReviewRepository(db_path))  # simulates an application restart
    assert restarted.get_review(case.review_id) == done
    assert [c.review_id for c in restarted.list_pending()] == [pending.review_id]
    with pytest.raises(ReviewAlreadyCompletedError):  # still protected after the restart
        restarted.submit_decision(case.review_id, "rev-9", "APPROVE", "too late")


def test_injected_id_factory_is_used(db_path):
    ids = iter(["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"])
    service = ReviewService(ReviewRepository(db_path), id_factory=lambda: next(ids))
    assert service.submit_for_review(**system_output()).review_id.startswith("aaaaaaaa")
    assert service.submit_for_review(**system_output()).review_id.startswith("bbbbbbbb")
