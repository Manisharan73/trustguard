"""Extended tests for Persistence (Phase 6).

Covers:
- Pending reviews survive a database connection closing and reopening.
- Completed reviews survive a database connection closing and reopening.
- Multiple pending and completed records survive database closing and reopening.
"""

import sqlite3
import pytest

from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    ReviewCase,
)
from src.human_review.repository import ReviewRepository
from src.human_review.service import ReviewService

ID_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
ID_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def make_sample_case(review_id: str, **overrides) -> ReviewCase:
    values = dict(
        review_id=review_id,
        created_at="2026-03-01T12:00:00+00:00",
        review_status=REVIEW_PENDING,
        transaction={"Time": 100.0, "V1": 0.123, "Amount": -15.5},
        prediction=0,
        probability=0.042,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=("NEGATIVE_AMOUNT",),
    )
    values.update(overrides)
    return ReviewCase(**values)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "persistence_test.db"


# 1. Pending reviews survive a database connection closing and reopening


def test_pending_reviews_survive_connection_closing_and_reopening_repo(db_path):
    # Step 1: Open repository connection and store a pending review
    repo1 = ReviewRepository(db_path)
    case_a = make_sample_case(ID_A)
    repo1.create(case_a)
    del repo1  # Explicitly drop repository reference

    # Step 2: Open a brand-new repository instance pointing to the same SQLite file
    repo2 = ReviewRepository(db_path)
    retrieved = repo2.get(ID_A)

    assert retrieved is not None
    assert retrieved.review_id == ID_A
    assert retrieved.review_status == REVIEW_PENDING
    assert retrieved.transaction == {"Time": 100.0, "V1": 0.123, "Amount": -15.5}
    assert retrieved.prediction == 0
    assert retrieved.probability == pytest.approx(0.042)
    assert retrieved.risk_level == "LOW"
    assert retrieved.recommended_action == "AUTO_APPROVE"
    assert retrieved.guardrail_status == "REQUIRES_REVIEW"
    assert retrieved.guardrail_reasons == ("NEGATIVE_AMOUNT",)
    assert retrieved.reviewer_id is None
    assert retrieved.human_decision is None
    assert retrieved.review_reason is None
    assert retrieved.reviewed_at is None

    # Check pending list
    pending = repo2.list_by_status(REVIEW_PENDING)
    assert len(pending) == 1
    assert pending[0].review_id == ID_A


def test_pending_reviews_survive_connection_closing_and_reopening_service(db_path):
    svc1 = ReviewService(ReviewRepository(db_path))
    case = svc1.submit_for_review(
        transaction={"Time": 50.0, "Amount": -1.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    case_id = case.review_id
    del svc1

    svc2 = ReviewService(ReviewRepository(db_path))
    retrieved = svc2.get_review(case_id)
    assert retrieved.review_id == case_id
    assert retrieved.review_status == REVIEW_PENDING
    assert [c.review_id for c in svc2.list_pending()] == [case_id]


# 2. Completed reviews survive a database connection closing and reopening


def test_completed_reviews_survive_connection_closing_and_reopening_repo(db_path):
    repo1 = ReviewRepository(db_path)
    case_b = make_sample_case(ID_B)
    repo1.create(case_b)
    completed = repo1.complete(
        review_id=ID_B,
        reviewer_id="reviewer_persisted",
        human_decision="REJECT",
        review_reason="fraud detected",
        reviewed_at="2026-03-01T12:05:00+00:00",
    )
    assert completed.review_status == REVIEW_COMPLETED
    del repo1

    repo2 = ReviewRepository(db_path)
    retrieved = repo2.get(ID_B)
    assert retrieved is not None
    assert retrieved.review_id == ID_B
    assert retrieved.review_status == REVIEW_COMPLETED
    assert retrieved.reviewer_id == "reviewer_persisted"
    assert retrieved.human_decision == "REJECT"
    assert retrieved.review_reason == "fraud detected"
    assert retrieved.reviewed_at == "2026-03-01T12:05:00+00:00"

    # Completed review does not appear in pending
    assert repo2.list_by_status(REVIEW_PENDING) == []
    # Completed review appears in completed list
    completed_list = repo2.list_by_status(REVIEW_COMPLETED)
    assert len(completed_list) == 1
    assert completed_list[0].review_id == ID_B


def test_completed_reviews_survive_connection_closing_and_reopening_service(db_path):
    svc1 = ReviewService(ReviewRepository(db_path))
    case = svc1.submit_for_review(
        transaction={"Time": 50.0, "Amount": -1.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    svc1.submit_decision(
        review_id=case.review_id,
        reviewer_id="reviewer_sv",
        decision="APPROVE",
        reason="looks good",
    )
    case_id = case.review_id
    del svc1

    svc2 = ReviewService(ReviewRepository(db_path))
    retrieved = svc2.get_review(case_id)
    assert retrieved.review_status == REVIEW_COMPLETED
    assert retrieved.reviewer_id == "reviewer_sv"
    assert retrieved.human_decision == "APPROVE"
    assert svc2.list_pending() == []


# 3. Both pending and completed records coexist and survive reconnects


def test_both_pending_and_completed_records_coexist_across_reconnects(db_path):
    repo1 = ReviewRepository(db_path)
    repo1.create(make_sample_case(ID_A))
    repo1.create(make_sample_case(ID_B))
    repo1.complete(ID_A, "rev1", "APPROVE", "ok", "2026-03-01T12:01:00+00:00")
    del repo1

    repo2 = ReviewRepository(db_path)
    case_a = repo2.get(ID_A)
    case_b = repo2.get(ID_B)

    assert case_a.review_status == REVIEW_COMPLETED
    assert case_b.review_status == REVIEW_PENDING

    pending_ids = [c.review_id for c in repo2.list_by_status(REVIEW_PENDING)]
    completed_ids = [c.review_id for c in repo2.list_by_status(REVIEW_COMPLETED)]

    assert pending_ids == [ID_B]
    assert completed_ids == [ID_A]
