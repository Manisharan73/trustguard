"""Phase 6 API integration tests: the /reviews human-in-the-loop endpoints.

Requires the trained model: run `python -m src.ml.train` first.

Isolation
---------
* Every test that uses the `client` fixture gets a fresh application lifespan and a
  review service pointing at its OWN temporary SQLite file, so the real project
  database is never touched.
* The database file is created lazily, so tests that never create a review (and
  every pre-Phase-6 test) create no database at all.
"""

import json
import re
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.decision.engine import DecisionConfig
from src.guardrails.engine import GuardrailConfig
from src.human_review.repository import DEFAULT_DB_PATH, ReviewRepository
from src.human_review.service import ReviewService
from src.risk.engine import RiskConfig
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
UNKNOWN_ID = "99999999-9999-4999-8999-999999999999"

# Finite, so the request schema accepts it; the guardrails then require review.
NEEDS_REVIEW = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)

REVIEW_FIELDS = {
    "review_id", "review_status", "created_at", "transaction",
    "prediction", "probability", "risk_level", "recommended_action",
    "guardrail_status", "guardrail_reasons",
    "reviewer_id", "human_decision", "review_reason", "reviewed_at",
}
ORIGINAL_FIELDS = (
    "created_at", "transaction", "prediction", "probability", "risk_level",
    "recommended_action", "guardrail_status", "guardrail_reasons",
)


@pytest.fixture
def client(tmp_path):
    with TestClient(app) as test_client:  # runs the lifespan for this test
        app.state.review_service = ReviewService(ReviewRepository(tmp_path / "reviews.db"))
        yield test_client
    # Leave the app in its default configuration for whatever runs next.
    app.state.risk_config = RiskConfig()
    app.state.decision_config = DecisionConfig()
    app.state.guardrail_config = GuardrailConfig()
    app.state.review_service = ReviewService(ReviewRepository(DEFAULT_DB_PATH))  # lazy: no I/O


def create_review(client, transaction=None):
    response = client.post("/reviews", json=transaction or NEEDS_REVIEW)
    assert response.status_code == 201, response.text
    return response.json()


def decide(client, review_id, **overrides):
    body = {"reviewer_id": "alice", "decision": "APPROVE", "reason": "verified with the customer"}
    body.update(overrides)
    return client.post(f"/reviews/{review_id}/decision", json=body)


# ======================= A. review submission =======================

def test_creating_a_review_returns_201_and_a_pending_case(client):
    response = client.post("/reviews", json=NEEDS_REVIEW)
    assert response.status_code == 201
    body = response.json()
    assert set(body) == REVIEW_FIELDS
    assert UUID_RE.match(body["review_id"])
    assert body["review_status"] == "PENDING"
    assert body["guardrail_status"] == "REQUIRES_REVIEW"
    assert body["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]
    assert body["recommended_action"] == "AUTO_APPROVE"
    assert body["transaction"]["Amount"] == -5.0
    assert datetime.fromisoformat(body["created_at"])


def test_a_new_review_has_no_human_decision(client):
    body = create_review(client)
    for field in ("reviewer_id", "human_decision", "review_reason", "reviewed_at"):
        assert body[field] is None


def test_the_preserved_fields_match_what_predict_returns_for_the_same_transaction(client):
    predicted = client.post("/predict", json=NEEDS_REVIEW).json()
    review = create_review(client)
    for field in ("prediction", "risk_level", "recommended_action",
                  "guardrail_status", "guardrail_reasons"):
        assert review[field] == predicted[field], field
    assert review["probability"] == pytest.approx(predicted["probability"])


def test_review_ids_are_unique(client):
    ids = {create_review(client)["review_id"] for _ in range(5)}
    assert len(ids) == 5


def test_a_human_review_recommendation_is_queued(client):
    app.state.risk_config = RiskConfig(low_threshold=0.001, high_threshold=0.5)  # MEDIUM
    body = create_review(client, LEGITIMATE_TRANSACTION)  # clean data, but HUMAN_REVIEW
    assert body["recommended_action"] == "HUMAN_REVIEW"
    assert body["guardrail_reasons"] == ["HUMAN_REVIEW_REQUIRED"]
    assert body["review_status"] == "PENDING"


def test_a_transaction_that_needs_no_review_is_refused_with_422_and_not_stored(client):
    response = client.post("/reviews", json=LEGITIMATE_TRANSACTION)  # passes all guardrails
    assert response.status_code == 422
    assert "does not require human review" in response.json()["detail"]
    assert client.get("/reviews/pending").json() == []


def test_predict_never_creates_a_review(client):
    assert client.post("/predict", json=NEEDS_REVIEW).status_code == 200
    assert client.get("/reviews/pending").json() == []


def test_the_client_cannot_supply_the_system_output(client):
    payload = dict(NEEDS_REVIEW, prediction=1, risk_level="HIGH", recommended_action="AUTO_REJECT")
    assert client.post("/reviews", json=payload).status_code == 422  # extra fields forbidden
    assert client.get("/reviews/pending").json() == []


def test_an_invalid_transaction_body_is_rejected(client):
    payload = dict(NEEDS_REVIEW)
    del payload["V12"]
    assert client.post("/reviews", json=payload).status_code == 422


def test_nan_in_a_review_submission_is_a_valid_json_422(client):
    payload = dict(NEEDS_REVIEW, V1=float("nan"))
    response = client.post(
        "/reviews", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


# ======================= B. pending queue =======================

def test_an_empty_queue_returns_an_empty_list(client):
    response = client.get("/reviews/pending")
    assert response.status_code == 200
    assert response.json() == []


def test_multiple_pending_reviews_are_listed_oldest_first(client):
    ids = [create_review(client)["review_id"] for _ in range(3)]
    response = client.get("/reviews/pending")
    assert response.status_code == 200
    assert [r["review_id"] for r in response.json()] == ids
    assert all(set(r) == REVIEW_FIELDS for r in response.json())
    assert all(r["review_status"] == "PENDING" for r in response.json())


def test_completed_reviews_are_excluded_from_the_pending_list(client):
    first, second = create_review(client)["review_id"], create_review(client)["review_id"]
    assert decide(client, first).status_code == 200
    assert [r["review_id"] for r in client.get("/reviews/pending").json()] == [second]


def test_the_pending_route_is_not_captured_by_the_id_route(client):
    assert client.get("/reviews/pending").status_code == 200  # not 404 / 422


# ======================= C. human decisions =======================

@pytest.mark.parametrize("decision", ["APPROVE", "REJECT"])
def test_a_reviewer_can_approve_or_reject(client, decision):
    review_id = create_review(client)["review_id"]
    response = decide(client, review_id, decision=decision)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == REVIEW_FIELDS
    assert body["review_status"] == "COMPLETED"
    assert body["human_decision"] == decision


def test_reviewer_identity_reason_and_timestamp_are_recorded(client):
    review_id = create_review(client)["review_id"]
    body = decide(client, review_id, reviewer_id="bob.jones", reason="amount is impossible").json()
    assert body["reviewer_id"] == "bob.jones"
    assert body["review_reason"] == "amount is impossible"
    assert datetime.fromisoformat(body["reviewed_at"]) >= datetime.fromisoformat(body["created_at"])


def test_reviewer_id_and_reason_are_stored_stripped(client):
    review_id = create_review(client)["review_id"]
    body = decide(client, review_id, reviewer_id="  carol ", reason="\n checked \t").json()
    assert (body["reviewer_id"], body["review_reason"]) == ("carol", "checked")


def test_the_completed_decision_can_be_retrieved(client):
    review_id = create_review(client)["review_id"]
    decided = decide(client, review_id, decision="REJECT", reason="fraud ring").json()
    fetched = client.get(f"/reviews/{review_id}")
    assert fetched.status_code == 200
    assert fetched.json() == decided


def test_a_pending_review_can_be_retrieved(client):
    created = create_review(client)
    fetched = client.get(f"/reviews/{created['review_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_the_human_decision_is_independent_of_the_recommendation(client):
    review_id = create_review(client)["review_id"]  # recommended AUTO_APPROVE
    body = decide(client, review_id, decision="REJECT", reason="disagree with the model").json()
    assert body["recommended_action"] == "AUTO_APPROVE"
    assert body["human_decision"] == "REJECT"
    assert not any("final" in key for key in body)


def test_a_fraud_transaction_with_a_data_problem_can_also_be_reviewed(client):
    body = create_review(client, dict(FRAUD_TRANSACTION, Amount=-5.0))
    assert body["recommended_action"] == "AUTO_REJECT"
    assert body["guardrail_reasons"] == ["NEGATIVE_AMOUNT"]
    assert decide(client, body["review_id"], decision="APPROVE").json()["human_decision"] == "APPROVE"


# ======================= D. validation =======================

@pytest.mark.parametrize(
    "body",
    [
        {"decision": "APPROVE", "reason": "ok"},                                      # missing reviewer_id
        {"reviewer_id": "", "decision": "APPROVE", "reason": "ok"},                   # empty reviewer_id
        {"reviewer_id": "   ", "decision": "APPROVE", "reason": "ok"},                # blank reviewer_id
        {"reviewer_id": "alice", "reason": "ok"},                                     # missing decision
        {"reviewer_id": "alice", "decision": "MAYBE", "reason": "ok"},                # invalid decision
        {"reviewer_id": "alice", "decision": "approve", "reason": "ok"},              # wrong case
        {"reviewer_id": "alice", "decision": "APPROVE"},                              # missing reason
        {"reviewer_id": "alice", "decision": "APPROVE", "reason": ""},                # empty reason
        {"reviewer_id": "alice", "decision": "APPROVE", "reason": "   \n\t  "},       # whitespace-only reason
        {"reviewer_id": "x" * 101, "decision": "APPROVE", "reason": "ok"},            # too long
        {"reviewer_id": "alice", "decision": "APPROVE", "reason": "x" * 2001},        # too long
        {"reviewer_id": None, "decision": "APPROVE", "reason": "ok"},                 # wrong type
        {},                                                                           # empty body
    ],
)
def test_invalid_decision_requests_return_422_with_valid_json_and_leave_the_review_pending(client, body):
    review_id = create_review(client)["review_id"]
    response = client.post(f"/reviews/{review_id}/decision", json=body)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert client.get(f"/reviews/{review_id}").json()["review_status"] == "PENDING"


@pytest.mark.parametrize(
    "extra",
    [{"prediction": 1}, {"probability": 0.0}, {"risk_level": "HIGH"},
     {"recommended_action": "AUTO_REJECT"}, {"guardrail_status": "PASSED"},
     {"guardrail_reasons": []}, {"review_status": "COMPLETED"}],
)
def test_a_reviewer_cannot_modify_the_original_output_through_the_decision(client, extra):
    created = create_review(client)
    response = decide(client, created["review_id"], **extra)
    assert response.status_code == 422
    unchanged = client.get(f"/reviews/{created['review_id']}").json()
    assert unchanged == created


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "123", "ABCDEFAB-ABCD-4ABC-8ABC-ABCDEFABCDEF", "x' OR '1'='1"])
def test_a_malformed_review_id_returns_422(client, bad_id):
    assert client.get(f"/reviews/{bad_id}").status_code == 422
    assert decide(client, bad_id).status_code == 422


# ======================= E. workflow edge cases =======================

def test_getting_a_nonexistent_review_returns_404(client):
    response = client.get(f"/reviews/{UNKNOWN_ID}")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_deciding_a_nonexistent_review_returns_404(client):
    assert decide(client, UNKNOWN_ID).status_code == 404


def test_a_review_can_only_be_decided_once(client):
    review_id = create_review(client)["review_id"]
    assert decide(client, review_id, reviewer_id="alice", decision="APPROVE", reason="first").status_code == 200
    second = decide(client, review_id, reviewer_id="bob", decision="REJECT", reason="second")
    assert second.status_code == 409
    assert "already completed" in second.json()["detail"]


def test_the_first_decision_survives_a_second_attempt(client):
    review_id = create_review(client)["review_id"]
    first = decide(client, review_id, reviewer_id="alice", decision="APPROVE", reason="first").json()
    decide(client, review_id, reviewer_id="bob", decision="REJECT", reason="second")
    assert client.get(f"/reviews/{review_id}").json() == first


def test_resubmitting_the_identical_decision_is_also_a_409(client):
    review_id = create_review(client)["review_id"]
    assert decide(client, review_id).status_code == 200
    assert decide(client, review_id).status_code == 409


def test_a_decision_leaves_every_original_field_unchanged(client):
    created = create_review(client)
    decided = decide(client, created["review_id"], decision="REJECT").json()
    for field in ORIGINAL_FIELDS:
        assert decided[field] == created[field], field
    fetched = client.get(f"/reviews/{created['review_id']}").json()
    for field in ORIGINAL_FIELDS:
        assert fetched[field] == created[field], field


@pytest.mark.parametrize("method", ["put", "patch", "delete"])
def test_reviews_cannot_be_modified_or_deleted_through_other_methods(client, method):
    review_id = create_review(client)["review_id"]
    assert getattr(client, method)(f"/reviews/{review_id}").status_code == 405


def test_the_decision_route_does_not_accept_get(client):
    review_id = create_review(client)["review_id"]
    assert client.get(f"/reviews/{review_id}/decision").status_code == 405


# ======================= F. persistence and configuration =======================

def test_decisions_survive_an_application_restart(tmp_path, monkeypatch):
    db_file = tmp_path / "restart.db"
    monkeypatch.setenv("TRUSTGUARD_DB_PATH", str(db_file))

    with TestClient(app) as first_run:  # lifespan builds the service from the env path
        created = first_run.post("/reviews", json=NEEDS_REVIEW).json()
        decided = first_run.post(
            f"/reviews/{created['review_id']}/decision",
            json={"reviewer_id": "alice", "decision": "REJECT", "reason": "fraud"},
        ).json()
        still_pending = first_run.post("/reviews", json=NEEDS_REVIEW).json()
    assert db_file.exists()

    with TestClient(app) as second_run:  # a brand-new application start
        assert second_run.get(f"/reviews/{created['review_id']}").json() == decided
        assert [r["review_id"] for r in second_run.get("/reviews/pending").json()] == [still_pending["review_id"]]
        assert decide(second_run, created["review_id"]).status_code == 409  # still protected


def test_the_database_path_is_configurable_through_the_environment(tmp_path, monkeypatch):
    db_file = tmp_path / "custom" / "location.db"
    monkeypatch.setenv("TRUSTGUARD_DB_PATH", str(db_file))
    with TestClient(app) as test_client:
        assert test_client.post("/reviews", json=NEEDS_REVIEW).status_code == 201
    assert db_file.exists()


def test_starting_the_application_does_not_create_a_database_file(tmp_path, monkeypatch):
    db_file = tmp_path / "unused.db"
    monkeypatch.setenv("TRUSTGUARD_DB_PATH", str(db_file))
    with TestClient(app) as test_client:
        assert test_client.post("/predict", json=NEEDS_REVIEW).status_code == 200
    assert not db_file.exists()


def test_a_new_client_gets_a_fresh_review_service(tmp_path):
    with TestClient(app):
        app.state.review_service = ReviewService(ReviewRepository(tmp_path / "a.db"))
        first_service = app.state.review_service
    with TestClient(app):
        assert app.state.review_service is not first_service
