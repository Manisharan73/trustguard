"""Extended tests for Repository and Database Integrity (Phase 6).

Covers:
- Duplicate review IDs are rejected.
- Invalid review statuses cannot be stored.
- Invalid human decisions cannot be stored.
- Completed reviews cannot be changed back to PENDING.
- Completed review decisions cannot be overwritten.
- Database constraints and triggers are tested directly where appropriate.
"""

import json
import sqlite3
import pytest

from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    ReviewAlreadyCompletedError,
    ReviewCase,
)
from src.human_review.repository import ReviewRepository

TEST_ID_1 = "11111111-aaaa-4aaa-8aaa-111111111111"
TEST_ID_2 = "22222222-bbbb-4bbb-8bbb-222222222222"


def make_case(review_id=TEST_ID_1, **overrides) -> ReviewCase:
    values = dict(
        review_id=review_id,
        created_at="2026-01-01T00:00:00+00:00",
        review_status=REVIEW_PENDING,
        transaction={"Time": 10.0, "V1": 0.5, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=("NEGATIVE_AMOUNT",),
    )
    values.update(overrides)
    return ReviewCase(**values)


def raw_row(**overrides) -> dict:
    row = dict(
        review_id=TEST_ID_2,
        created_at="2026-01-01T00:00:00+00:00",
        review_status="PENDING",
        transaction_json="{}",
        prediction=0,
        probability=0.5,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons_json="[]",
        reviewer_id=None,
        human_decision=None,
        review_reason=None,
        reviewed_at=None,
    )
    row.update(overrides)
    return row


def execute_raw_insert(db_path, **overrides):
    row = raw_row(**overrides)
    conn = sqlite3.connect(str(db_path))
    try:
        with conn:
            conn.execute(
                f"INSERT INTO reviews ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
    finally:
        conn.close()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "integrity_test.db"


@pytest.fixture
def repo(db_path):
    r = ReviewRepository(db_path)
    # Trigger lazy schema creation
    r.get("00000000-0000-4000-8000-000000000000")
    return r


# 1. Duplicate review IDs are rejected


def test_duplicate_review_ids_rejected_repo(repo):
    case = make_case(TEST_ID_1)
    repo.create(case)
    # Attempting to insert duplicate review_id via repo raises IntegrityError
    duplicate_case = make_case(TEST_ID_1, prediction=1)
    with pytest.raises(sqlite3.IntegrityError):
        repo.create(duplicate_case)
    # Verify original remains untouched
    assert repo.get(TEST_ID_1).prediction == 0


def test_duplicate_review_ids_rejected_direct_sql(repo, db_path):
    case = make_case(TEST_ID_1)
    repo.create(case)
    # Attempting to insert duplicate review_id via raw SQL raises UNIQUE constraint failure
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        execute_raw_insert(db_path, review_id=TEST_ID_1)


# 2. Invalid review statuses cannot be stored


@pytest.mark.parametrize(
    "invalid_status",
    [
        "PENDING_REVIEW",
        "IN_PROGRESS",
        "DONE",
        "ACCEPTED",
        "CANCELLED",
        "pending",
        "completed",
        "UNKNOWN",
        "",
    ],
)
def test_invalid_review_statuses_cannot_be_stored(repo, db_path, invalid_status):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        execute_raw_insert(db_path, review_status=invalid_status)


# 3. Invalid human decisions cannot be stored


@pytest.mark.parametrize(
    "invalid_decision",
    [
        "PENDING",
        "DECLINE",
        "PASS",
        "FAIL",
        "HOLD",
        "MAYBE",
        "approve",
        "reject",
        "INVALID",
        "",
    ],
)
def test_invalid_human_decisions_cannot_be_stored(repo, db_path, invalid_decision):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        execute_raw_insert(
            db_path,
            review_status="COMPLETED",
            reviewer_id="reviewer_1",
            human_decision=invalid_decision,
            review_reason="reason text",
            reviewed_at="2026-01-01T00:00:00+00:00",
        )


# 4. Completed reviews cannot be changed back to PENDING


def test_completed_reviews_cannot_be_changed_back_to_pending(repo, db_path):
    case = make_case(TEST_ID_1)
    repo.create(case)
    repo.complete(TEST_ID_1, "alice", "APPROVE", "verified", "2026-01-01T01:00:00+00:00")
    assert repo.get(TEST_ID_1).review_status == REVIEW_COMPLETED

    # Direct raw SQL attempting to revert to PENDING
    conn = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="completed reviews are immutable"):
            with conn:
                conn.execute(
                    """
                    UPDATE reviews
                       SET review_status = 'PENDING', reviewer_id = NULL,
                           human_decision = NULL, review_reason = NULL, reviewed_at = NULL
                     WHERE review_id = ?
                    """,
                    (TEST_ID_1,),
                )
    finally:
        conn.close()

    # Verify review in database remains COMPLETED with original reviewer data
    current = repo.get(TEST_ID_1)
    assert current.review_status == REVIEW_COMPLETED
    assert current.human_decision == "APPROVE"
    assert current.reviewer_id == "alice"


# 5. Completed review decisions cannot be overwritten


def test_completed_review_decisions_cannot_be_overwritten_repo(repo):
    case = make_case(TEST_ID_1)
    repo.create(case)
    repo.complete(TEST_ID_1, "alice", "APPROVE", "first", "2026-01-01T01:00:00+00:00")

    with pytest.raises(ReviewAlreadyCompletedError):
        repo.complete(TEST_ID_1, "bob", "REJECT", "second", "2026-01-01T02:00:00+00:00")

    persisted = repo.get(TEST_ID_1)
    assert persisted.reviewer_id == "alice"
    assert persisted.human_decision == "APPROVE"
    assert persisted.review_reason == "first"


@pytest.mark.parametrize(
    "tampering_update",
    [
        "SET human_decision = 'REJECT'",
        "SET reviewer_id = 'bob'",
        "SET review_reason = 'tampered reason'",
        "SET reviewed_at = '2026-12-31T23:59:59+00:00'",
    ],
)
def test_completed_review_decisions_cannot_be_overwritten_sql(repo, db_path, tampering_update):
    case = make_case(TEST_ID_1)
    repo.create(case)
    repo.complete(TEST_ID_1, "alice", "APPROVE", "first", "2026-01-01T01:00:00+00:00")

    conn = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="completed reviews are immutable"):
            with conn:
                conn.execute(f"UPDATE reviews {tampering_update} WHERE review_id = ?", (TEST_ID_1,))
    finally:
        conn.close()

    persisted = repo.get(TEST_ID_1)
    assert persisted.reviewer_id == "alice"
    assert persisted.human_decision == "APPROVE"
    assert persisted.review_reason == "first"


# 6. Database constraints and triggers tested directly where appropriate


def test_pending_row_cannot_carry_any_human_fields(repo, db_path):
    # A row with review_status=PENDING must have reviewer_id, human_decision, review_reason, reviewed_at all NULL
    for human_field in ("reviewer_id", "human_decision", "review_reason", "reviewed_at"):
        val = "APPROVE" if human_field == "human_decision" else "not_null"
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            execute_raw_insert(db_path, review_id=str(pytest.importorskip("uuid").uuid4()), **{human_field: val})


def test_completed_row_must_carry_all_human_fields(repo, db_path):
    # A row with review_status=COMPLETED must have all 4 human fields NOT NULL
    valid_human_data = dict(
        reviewer_id="reviewer_1",
        human_decision="APPROVE",
        review_reason="ok",
        reviewed_at="2026-01-01T00:00:00+00:00",
    )
    for field_to_null in ("reviewer_id", "human_decision", "review_reason", "reviewed_at"):
        invalid_data = dict(valid_human_data, review_status="COMPLETED")
        invalid_data[field_to_null] = None
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            execute_raw_insert(db_path, review_id=str(pytest.importorskip("uuid").uuid4()), **invalid_data)


@pytest.mark.parametrize(
    "assignment",
    [
        "review_id = '99999999-9999-4999-8999-999999999999'",
        "created_at = '2099-01-01T00:00:00+00:00'",
        "transaction_json = '{\"tampered\": true}'",
        "prediction = 1",
        "probability = 0.999",
        "risk_level = 'HIGH'",
        "recommended_action = 'AUTO_REJECT'",
        "guardrail_status = 'PASSED'",
        "guardrail_reasons_json = '[]'",
    ],
)
def test_trigger_original_output_immutable_on_pending_row(repo, db_path, assignment):
    case = make_case(TEST_ID_1)
    repo.create(case)

    conn = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="original model and guardrail output is immutable"):
            with conn:
                conn.execute(f"UPDATE reviews SET {assignment} WHERE review_id = ?", (TEST_ID_1,))
    finally:
        conn.close()

    assert repo.get(TEST_ID_1) == case


@pytest.mark.parametrize(
    "invalid_col,bad_val",
    [
        ("prediction", 2),
        ("prediction", -1),
        ("probability", -0.01),
        ("probability", 1.01),
        ("risk_level", "INVALID_RISK"),
        ("recommended_action", "INVALID_ACTION"),
        ("guardrail_status", "INVALID_STATUS"),
    ],
)
def test_direct_check_constraints_on_domain_columns(repo, db_path, invalid_col, bad_val):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        execute_raw_insert(db_path, review_id=str(pytest.importorskip("uuid").uuid4()), **{invalid_col: bad_val})
