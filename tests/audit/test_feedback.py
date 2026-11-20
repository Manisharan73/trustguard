"""Tests for human review feedback tracking, override detection, and agreement metrics."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.audit.repository import AuditRepository
from src.audit.service import AuditService
from tests.api.sample_transactions import LEGITIMATE_TRANSACTION


def _dummy_outcome(action="HUMAN_REVIEW"):
    return {
        "prediction": 0,
        "probability": 0.45,
        "risk_level": "MEDIUM",
        "recommended_action": action,
        "guardrail_status": "REQUIRES_REVIEW",
        "guardrail_reasons": ["MEDIUM_RISK_REQUIRES_HUMAN_REVIEW"],
        "confidence": 0.55,
        "uncertainty": 0.45,
        "ood_status": "NORMAL",
        "ood_distance": 3.0,
        "requires_human_review": True,
        "phase5_guardrail_status": "REQUIRES_REVIEW",
        "phase5_guardrail_reasons": ["MEDIUM_RISK_REQUIRES_HUMAN_REVIEW"],
        "phase8_guardrail_status": "PASSED",
        "phase8_guardrail_reasons": [],
    }


def test_feedback_lifecycle_and_override_detection(tmp_path):
    repo = AuditRepository(tmp_path / "test_feedback.db")
    service = AuditService(repo)

    # Case 1: System said AUTO_APPROVE, human decided REJECT -> override=True, disagreement=True
    outcome_approve = _dummy_outcome(action="AUTO_APPROVE")
    record1 = service.record_transaction(
        transaction=LEGITIMATE_TRANSACTION, pipeline_outcome=outcome_approve, review_id="rev-1"
    )
    audit1, fb1 = service.record_decision(
        review_id="rev-1",
        reviewer_id="rev_bob",
        human_decision="REJECT",
        review_reason="Fraud pattern detected",
    )
    assert fb1 is not None
    assert fb1.recommended_action == "AUTO_APPROVE"
    assert fb1.human_decision == "REJECT"
    assert fb1.disagreement is True
    assert fb1.is_override is True

    # Case 2: System said AUTO_REJECT, human decided APPROVE -> override=True, disagreement=True
    outcome_reject = _dummy_outcome(action="AUTO_REJECT")
    record2 = service.record_transaction(
        transaction=LEGITIMATE_TRANSACTION, pipeline_outcome=outcome_reject, review_id="rev-2"
    )
    audit2, fb2 = service.record_decision(
        review_id="rev-2",
        reviewer_id="rev_bob",
        human_decision="APPROVE",
        review_reason="Customer confirmed high-value purchase",
    )
    assert fb2 is not None
    assert fb2.recommended_action == "AUTO_REJECT"
    assert fb2.human_decision == "APPROVE"
    assert fb2.disagreement is True
    assert fb2.is_override is True

    # Case 3: System said HUMAN_REVIEW, human decided APPROVE -> disagreement=False, override=False
    outcome_review = _dummy_outcome(action="HUMAN_REVIEW")
    record3 = service.record_transaction(
        transaction=LEGITIMATE_TRANSACTION, pipeline_outcome=outcome_review, review_id="rev-3"
    )
    audit3, fb3 = service.record_decision(
        review_id="rev-3",
        reviewer_id="rev_bob",
        human_decision="APPROVE",
        review_reason="Normal purchase activity",
    )
    assert fb3 is not None
    assert fb3.recommended_action == "HUMAN_REVIEW"
    assert fb3.human_decision == "APPROVE"
    assert fb3.disagreement is False
    assert fb3.is_override is False


def test_feedback_api_retrieval(tmp_path):
    repo = AuditRepository(tmp_path / "test_feedback_api.db")
    service = AuditService(repo)

    with TestClient(app) as client:
        app.state.audit_repository = repo
        app.state.audit_service = service

        outcome = _dummy_outcome(action="AUTO_APPROVE")
        rec = service.record_transaction(
            transaction=LEGITIMATE_TRANSACTION, pipeline_outcome=outcome, review_id="rev-test-1"
        )
        service.record_decision(
            review_id="rev-test-1",
            reviewer_id="reviewer_1",
            human_decision="REJECT",
            review_reason="Stolen card reported",
        )

        resp = client.get("/feedback")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 1
        item = data["records"][0]
        assert item["audit_id"] == rec.audit_id
        assert item["review_id"] == "rev-test-1"
        assert item["recommended_action"] == "AUTO_APPROVE"
        assert item["human_decision"] == "REJECT"
        assert item["disagreement"] is True
        assert item["is_override"] is True
