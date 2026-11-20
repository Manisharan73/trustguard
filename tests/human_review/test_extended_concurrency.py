"""Extended tests for Concurrency (Phase 6).

Covers:
- Two reviewers submit decisions for the same review simultaneously.
- Exactly one decision succeeds.
- The other request receives an appropriate conflict response.
- Database contains exactly one completed decision.
- Concurrent review creation generates unique IDs.
- Concurrent pending queue retrieval does not produce errors.
"""

import sqlite3
import threading
import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.human_review.models import (
    REVIEW_COMPLETED,
    ReviewAlreadyCompletedError,
)
from src.human_review.repository import DEFAULT_DB_PATH, ReviewRepository
from src.human_review.service import ReviewService
from tests.api.sample_transactions import LEGITIMATE_TRANSACTION


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "concurrent_reviews.db"


@pytest.fixture
def repo(db_path):
    return ReviewRepository(db_path)


@pytest.fixture
def service(repo):
    return ReviewService(repo)


@pytest.fixture
def client(db_path):
    with TestClient(app) as test_client:
        app.state.review_service = ReviewService(ReviewRepository(db_path))
        yield test_client
    app.state.review_service = ReviewService(ReviewRepository(DEFAULT_DB_PATH))


# 1. Two reviewers submit decisions for the same review simultaneously (API)


def test_concurrent_decisions_by_two_reviewers_api(client, db_path):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    created = client.post("/reviews", json=payload).json()
    review_id = created["review_id"]

    results = []
    barrier = threading.Barrier(2)

    def reviewer_action(reviewer_name, decision_val, reason_text):
        barrier.wait()
        resp = client.post(
            f"/reviews/{review_id}/decision",
            json={
                "reviewer_id": reviewer_name,
                "decision": decision_val,
                "reason": reason_text,
            },
        )
        results.append((resp.status_code, resp.json()))

    t1 = threading.Thread(target=reviewer_action, args=("alice", "APPROVE", "verified legit"))
    t2 = threading.Thread(target=reviewer_action, args=("bob", "REJECT", "suspected fraud"))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    status_codes = [code for code, _ in results]
    # Exactly one decision succeeds (200), the other receives conflict (409)
    assert status_codes.count(200) == 1
    assert status_codes.count(409) == 1

    # Database contains exactly one completed decision
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT review_status, reviewer_id, human_decision FROM reviews WHERE review_id = ?", (review_id,))
        rows = cursor.fetchall()
        assert len(rows) == 1
        status, rev_id, dec = rows[0]
        assert status == REVIEW_COMPLETED
        assert rev_id in ("alice", "bob")
        assert dec in ("APPROVE", "REJECT")
    finally:
        conn.close()

    # Subsequent GET returns the single completed decision
    get_resp = client.get(f"/reviews/{review_id}").json()
    assert get_resp["review_status"] == REVIEW_COMPLETED
    assert get_resp["reviewer_id"] == rev_id
    assert get_resp["human_decision"] == dec


# 2. Two reviewers submit decisions for the same review simultaneously (Service/Repo)


def test_concurrent_decisions_by_two_reviewers_service(service, repo, db_path):
    case = service.submit_for_review(
        transaction={"Time": 1.0, "Amount": -5.0},
        prediction=0,
        probability=0.01,
        risk_level="LOW",
        recommended_action="AUTO_APPROVE",
        guardrail_status="REQUIRES_REVIEW",
        guardrail_reasons=["NEGATIVE_AMOUNT"],
    )
    review_id = case.review_id

    outcomes = []
    barrier = threading.Barrier(2)

    def reviewer_action(reviewer_name, decision_val):
        barrier.wait()
        try:
            service.submit_decision(
                review_id=review_id,
                reviewer_id=reviewer_name,
                decision=decision_val,
                reason="concurrent decision test",
            )
            outcomes.append(("success", reviewer_name))
        except ReviewAlreadyCompletedError:
            outcomes.append(("conflict", reviewer_name))

    t1 = threading.Thread(target=reviewer_action, args=("reviewer_1", "APPROVE"))
    t2 = threading.Thread(target=reviewer_action, args=("reviewer_2", "REJECT"))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    types = [t for t, _ in outcomes]
    assert types.count("success") == 1
    assert types.count("conflict") == 1

    final_case = service.get_review(review_id)
    assert final_case.review_status == REVIEW_COMPLETED
    assert final_case.reviewer_id in ("reviewer_1", "reviewer_2")


# 3. Concurrent review creation generates unique IDs


def test_concurrent_review_creation_generates_unique_ids_api(client):
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    created_ids = []
    lock = threading.Lock()
    num_threads = 12
    barrier = threading.Barrier(num_threads)

    def create_worker():
        barrier.wait()
        resp = client.post("/reviews", json=payload)
        assert resp.status_code == 201
        rid = resp.json()["review_id"]
        with lock:
            created_ids.append(rid)

    threads = [threading.Thread(target=create_worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created_ids) == num_threads
    # All IDs must be unique
    assert len(set(created_ids)) == num_threads
    # All IDs must be valid UUIDs
    for rid in created_ids:
        assert str(uuid.UUID(rid)) == rid

    # All must appear in pending queue
    pending = client.get("/reviews/pending").json()
    pending_ids = {r["review_id"] for r in pending}
    for rid in created_ids:
        assert rid in pending_ids


def test_concurrent_review_creation_generates_unique_ids_service(service):
    num_threads = 12
    created_ids = []
    lock = threading.Lock()
    barrier = threading.Barrier(num_threads)

    def worker():
        barrier.wait()
        case = service.submit_for_review(
            transaction={"Time": 1.0, "Amount": -5.0},
            prediction=0,
            probability=0.01,
            risk_level="LOW",
            recommended_action="AUTO_APPROVE",
            guardrail_status="REQUIRES_REVIEW",
            guardrail_reasons=["NEGATIVE_AMOUNT"],
        )
        with lock:
            created_ids.append(case.review_id)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(created_ids) == num_threads
    assert len(set(created_ids)) == num_threads


# 4. Concurrent pending queue retrieval does not produce errors


def test_concurrent_pending_queue_retrieval_api(client):
    # Pre-create several reviews
    payload = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)
    r1 = client.post("/reviews", json=payload).json()["review_id"]
    r2 = client.post("/reviews", json=payload).json()["review_id"]
    r3 = client.post("/reviews", json=payload).json()["review_id"]

    errors = []
    num_readers = 10
    barrier = threading.Barrier(num_readers + 2)

    def reader_worker():
        barrier.wait()
        for _ in range(5):
            resp = client.get("/reviews/pending")
            if resp.status_code != 200:
                errors.append(f"Bad status: {resp.status_code}")
            if not isinstance(resp.json(), list):
                errors.append("Response not a list")

    def writer_worker():
        barrier.wait()
        # Add new reviews during reads
        for _ in range(3):
            client.post("/reviews", json=payload)

    def decider_worker():
        barrier.wait()
        # Complete a review during reads
        client.post(
            f"/reviews/{r1}/decision",
            json={"reviewer_id": "alice", "decision": "APPROVE", "reason": "ok"},
        )

    threads = [threading.Thread(target=reader_worker) for _ in range(num_readers)]
    threads.append(threading.Thread(target=writer_worker))
    threads.append(threading.Thread(target=decider_worker))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
