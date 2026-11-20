"""Tests for the SQLite review repository, using real temporary database files."""

import sqlite3
import threading

import pytest

from src.human_review.models import (
    REVIEW_COMPLETED,
    REVIEW_PENDING,
    ReviewAlreadyCompletedError,
    ReviewCase,
    ReviewNotFoundError,
)
from src.human_review.repository import ReviewRepository

ID_1 = "11111111-1111-4111-8111-111111111111"
ID_2 = "22222222-2222-4222-8222-222222222222"
ID_3 = "33333333-3333-4333-8333-333333333333"


def make_case(review_id=ID_1, **overrides) -> ReviewCase:
    values = dict(
        review_id=review_id,
        created_at="2026-01-01T00:00:00+00:00",
        review_status=REVIEW_PENDING,
        transaction={"Time": 10.0, "V1": 0.5, "Amount": -5.0},
        prediction=0,
        probability=0.0073,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=("NEGATIVE_AMOUNT",),
    )
    values.update(overrides)
    return ReviewCase(**values)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "reviews.db"


@pytest.fixture
def repo(db_path):
    return ReviewRepository(db_path)


def raw_row(**overrides) -> dict:
    """All column values of a valid PENDING row, for bypassing the repository."""
    row = dict(
        review_id=ID_3,
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


def raw_insert(path, **overrides) -> None:
    row = raw_row(**overrides)
    connection = sqlite3.connect(str(path))
    try:
        with connection:
            connection.execute(
                f"INSERT INTO reviews ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                tuple(row.values()),
            )
    finally:
        connection.close()


# ---------- construction and lazy creation ----------

def test_constructing_a_repository_does_not_touch_the_disk(db_path):
    ReviewRepository(db_path)
    assert not db_path.exists()


def test_database_and_parent_directories_are_created_on_first_use(tmp_path):
    path = tmp_path / "nested" / "dir" / "reviews.db"
    ReviewRepository(path).get(ID_1)
    assert path.exists()


@pytest.mark.parametrize("bad_path", [":memory:", "", "file::memory:?cache=shared"])
def test_in_memory_or_empty_paths_are_rejected(bad_path):
    with pytest.raises(ValueError, match="file path"):
        ReviewRepository(bad_path)


def test_two_repositories_can_share_one_file(db_path):
    first, second = ReviewRepository(db_path), ReviewRepository(db_path)
    first.create(make_case())
    assert second.get(ID_1) is not None  # schema creation is idempotent


# ---------- create / get ----------

def test_create_then_get_returns_an_identical_case(repo):
    case = make_case(guardrail_reasons=("NEGATIVE_AMOUNT", "NEGATIVE_TIME"))
    repo.create(case)
    assert repo.get(ID_1) == case


def test_get_unknown_id_returns_none(repo):
    assert repo.get(ID_1) is None


def test_duplicate_review_id_is_rejected_and_not_stored_twice(repo):
    repo.create(make_case())
    with pytest.raises(sqlite3.IntegrityError):
        repo.create(make_case(prediction=1))
    assert repo.get(ID_1).prediction == 0
    assert len(repo.list_by_status(REVIEW_PENDING)) == 1


def test_non_finite_transaction_values_are_rejected_and_nothing_is_stored(repo):
    with pytest.raises(ValueError):
        repo.create(make_case(transaction={"Amount": float("nan")}))
    assert repo.get(ID_1) is None


# ---------- listing ----------

def test_list_by_status_is_oldest_first_with_insertion_order_tiebreak(repo):
    repo.create(make_case(ID_2, created_at="2026-01-02T00:00:00+00:00"))
    repo.create(make_case(ID_1, created_at="2026-01-01T00:00:00+00:00"))
    repo.create(make_case(ID_3, created_at="2026-01-01T00:00:00+00:00"))  # tie with ID_1
    assert [c.review_id for c in repo.list_by_status(REVIEW_PENDING)] == [ID_1, ID_3, ID_2]


def test_list_by_status_on_an_empty_database_is_empty(repo):
    assert repo.list_by_status(REVIEW_PENDING) == []
    assert repo.list_by_status(REVIEW_COMPLETED) == []


def test_list_by_status_rejects_an_unknown_status(repo):
    with pytest.raises(ValueError, match="Invalid review status"):
        repo.list_by_status("DONE")


def test_completed_reviews_are_not_listed_as_pending(repo):
    repo.create(make_case(ID_1))
    repo.create(make_case(ID_2))
    repo.complete(ID_1, "rev-1", "APPROVE", "looks fine", "2026-01-03T00:00:00+00:00")
    assert [c.review_id for c in repo.list_by_status(REVIEW_PENDING)] == [ID_2]
    assert [c.review_id for c in repo.list_by_status(REVIEW_COMPLETED)] == [ID_1]


# ---------- complete ----------

def test_complete_records_the_human_decision_and_marks_completed(repo):
    repo.create(make_case())
    done = repo.complete(ID_1, "rev-1", "REJECT", "suspicious", "2026-01-03T00:00:00+00:00")
    assert done.review_status == REVIEW_COMPLETED
    assert done.reviewer_id == "rev-1"
    assert done.human_decision == "REJECT"
    assert done.review_reason == "suspicious"
    assert done.reviewed_at == "2026-01-03T00:00:00+00:00"
    assert repo.get(ID_1) == done


def test_complete_leaves_the_original_output_unchanged(repo):
    original = make_case(prediction=1, probability=0.91, risk_level="HIGH",
                         recommended_action="AUTO_REJECT", guardrail_status="PASSED",
                         guardrail_reasons=())
    repo.create(original)
    done = repo.complete(ID_1, "rev-1", "APPROVE", "customer verified", "2026-01-03T00:00:00+00:00")
    for name in ("review_id", "created_at", "transaction", "prediction", "probability",
                 "risk_level", "recommended_action", "guardrail_status", "guardrail_reasons"):
        assert getattr(done, name) == getattr(original, name), name


def test_completing_twice_fails_and_keeps_the_first_decision(repo):
    repo.create(make_case())
    repo.complete(ID_1, "rev-1", "APPROVE", "first", "2026-01-03T00:00:00+00:00")
    with pytest.raises(ReviewAlreadyCompletedError):
        repo.complete(ID_1, "rev-2", "REJECT", "second", "2026-01-04T00:00:00+00:00")
    kept = repo.get(ID_1)
    assert (kept.reviewer_id, kept.human_decision, kept.review_reason) == ("rev-1", "APPROVE", "first")


def test_completing_an_unknown_review_raises_not_found(repo):
    with pytest.raises(ReviewNotFoundError):
        repo.complete(ID_1, "rev-1", "APPROVE", "x", "2026-01-03T00:00:00+00:00")


def test_only_one_of_many_concurrent_decisions_succeeds(repo):
    repo.create(make_case())
    outcomes = []

    def attempt(n):
        try:
            repo.complete(ID_1, f"rev-{n}", "APPROVE", f"reason {n}", "2026-01-03T00:00:00+00:00")
            outcomes.append("ok")
        except ReviewAlreadyCompletedError:
            outcomes.append("already")

    threads = [threading.Thread(target=attempt, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("ok") == 1
    assert outcomes.count("already") == 7


# ---------- persistence ----------

def test_decisions_persist_after_reconnecting_with_a_new_repository(db_path):
    first = ReviewRepository(db_path)
    first.create(make_case())
    first.complete(ID_1, "rev-1", "APPROVE", "ok", "2026-01-03T00:00:00+00:00")
    del first

    reopened = ReviewRepository(db_path)  # simulates an application restart
    case = reopened.get(ID_1)
    assert case.review_status == REVIEW_COMPLETED
    assert (case.reviewer_id, case.human_decision, case.review_reason) == ("rev-1", "APPROVE", "ok")
    assert reopened.list_by_status(REVIEW_PENDING) == []


def test_pending_reviews_also_persist(db_path):
    ReviewRepository(db_path).create(make_case())
    assert len(ReviewRepository(db_path).list_by_status(REVIEW_PENDING)) == 1


# ---------- SQL safety ----------

def test_sql_metacharacters_in_ids_are_treated_as_data(repo):
    repo.create(make_case())
    assert repo.get("x' OR '1'='1") is None
    with pytest.raises(ReviewNotFoundError):
        repo.complete("'; DROP TABLE reviews; --", "r", "APPROVE", "x", "2026-01-03T00:00:00+00:00")
    assert repo.get(ID_1) is not None  # table intact


def test_sql_metacharacters_in_reviewer_text_are_stored_literally(repo):
    repo.create(make_case())
    evil = "Robert'); DROP TABLE reviews;--"
    repo.complete(ID_1, evil, "APPROVE", evil, "2026-01-03T00:00:00+00:00")
    case = repo.get(ID_1)
    assert case.reviewer_id == evil and case.review_reason == evil


# ---------- the database itself refuses inconsistent rows ----------

@pytest.mark.parametrize(
    "overrides",
    [
        {"prediction": 2},
        {"probability": 1.5},
        {"probability": -0.1},
        {"risk_level": "EXTREME"},
        {"recommended_action": "SHRUG"},
        {"guardrail_status": "MAYBE"},
        {"review_status": "DONE"},
        {"human_decision": "MAYBE"},
    ],
)
def test_check_constraints_reject_invalid_values(repo, db_path, overrides):
    repo.get(ID_1)  # make sure the schema exists
    with pytest.raises(sqlite3.IntegrityError):
        raw_insert(db_path, **overrides)


def test_a_pending_row_cannot_carry_human_fields(repo, db_path):
    repo.get(ID_1)
    with pytest.raises(sqlite3.IntegrityError):
        raw_insert(db_path, reviewer_id="rev-1")


def test_a_completed_row_must_carry_all_human_fields(repo, db_path):
    repo.get(ID_1)
    with pytest.raises(sqlite3.IntegrityError):
        raw_insert(db_path, review_status="COMPLETED", reviewer_id="rev-1")


def test_completed_rows_are_immutable_at_the_database_level(repo, db_path):
    repo.create(make_case())
    repo.complete(ID_1, "rev-1", "APPROVE", "ok", "2026-01-03T00:00:00+00:00")
    connection = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute("UPDATE reviews SET review_reason = 'tampered' WHERE review_id = ?", (ID_1,))
    finally:
        connection.close()
    assert repo.get(ID_1).review_reason == "ok"


@pytest.mark.parametrize(
    "assignment",
    ["prediction = 1", "probability = 0.99", "risk_level = 'HIGH'",
     "recommended_action = 'AUTO_REJECT'", "guardrail_status = 'PASSED'",
     "guardrail_reasons_json = '[]'", "transaction_json = '{}'", "created_at = 'x'"],
)
def test_original_output_is_immutable_at_the_database_level(repo, db_path, assignment):
    repo.create(make_case())
    connection = sqlite3.connect(str(db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute(f"UPDATE reviews SET {assignment} WHERE review_id = ?", (ID_1,))
    finally:
        connection.close()
    assert repo.get(ID_1) == make_case()
