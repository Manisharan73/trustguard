"""Tests for transaction auditing: repository, service, immutability, and API endpoints."""

import json
import sqlite3
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.audit.models import AuditRecord
from src.audit.repository import AuditRepository
from src.audit.service import AuditService
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


def _make_dummy_outcome(prediction=0, probability=0.05, risk="LOW", action="AUTO_APPROVE"):
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk,
        "recommended_action": action,
        "guardrail_status": "PASSED",
        "guardrail_reasons": [],
        "confidence": 0.95,
        "uncertainty": 0.05,
        "ood_status": "NORMAL",
        "ood_distance": 2.5,
        "requires_human_review": False,
        "phase5_guardrail_status": "PASSED",
        "phase5_guardrail_reasons": [],
        "phase8_guardrail_status": "PASSED",
        "phase8_guardrail_reasons": [],
    }


def _make_audit_record(audit_id="aud-1", **overrides):
    base = {
        "audit_id": audit_id,
        "created_at": "2026-11-20T10:00:00Z",
        "transaction": LEGITIMATE_TRANSACTION,
        "prediction": 0,
        "probability": 0.05,
        "risk_level": "LOW",
        "recommended_action": "AUTO_APPROVE",
        "phase5_guardrail_status": "PASSED",
        "phase5_guardrail_reasons": (),
        "phase8_guardrail_status": "PASSED",
        "phase8_guardrail_reasons": (),
        "confidence": 0.95,
        "uncertainty": 0.05,
        "ood_status": "NORMAL",
        "ood_distance": 2.5,
        "guardrail_status": "PASSED",
        "guardrail_reasons": (),
        "requires_human_review": False,
        "review_id": None,
        "reviewer_id": None,
        "human_decision": None,
        "review_reason": None,
        "reviewed_at": None,
        "ground_truth": None,
        "labeled_at": None,
        "updated_at": "2026-11-20T10:00:00Z",
    }
    base.update(overrides)
    return AuditRecord(**base)


def test_audit_repository_crud(tmp_path):
    repo = AuditRepository(tmp_path / "test_audit.db")
    rec = _make_audit_record("aud-test-1")
    repo.create_audit(rec)

    fetched = repo.get_audit("aud-test-1")
    assert fetched is not None
    assert fetched.audit_id == "aud-test-1"
    assert fetched.prediction == 0
    assert fetched.probability == 0.05
    assert fetched.risk_level == "LOW"
    assert fetched.recommended_action == "AUTO_APPROVE"
    assert fetched.guardrail_status == "PASSED"
    assert fetched.requires_human_review is False
    assert fetched.human_decision is None


def test_audit_repository_immutability_triggers(tmp_path):
    db_file = tmp_path / "test_immutability.db"
    repo = AuditRepository(db_file)
    rec = _make_audit_record("aud-test-imm")
    repo.create_audit(rec)

    conn = sqlite3.connect(str(db_file))
    with pytest.raises(sqlite3.IntegrityError, match="original model output and recommendation in audit records are immutable"):
        conn.execute("UPDATE audit_records SET prediction = 1 WHERE audit_id = ?", ("aud-test-imm",))

    with pytest.raises(sqlite3.IntegrityError, match="original model output and recommendation in audit records are immutable"):
        conn.execute("UPDATE audit_records SET probability = 0.99 WHERE audit_id = ?", ("aud-test-imm",))

    with pytest.raises(sqlite3.IntegrityError, match="original model output and recommendation in audit records are immutable"):
        conn.execute("UPDATE audit_records SET recommended_action = 'AUTO_REJECT' WHERE audit_id = ?", ("aud-test-imm",))

    conn.close()


def test_audit_repository_decision_immutability(tmp_path):
    db_file = tmp_path / "test_decision_immutability.db"
    repo = AuditRepository(db_file)
    rec = _make_audit_record(
        "aud-test-dec",
        recommended_action="HUMAN_REVIEW",
        requires_human_review=True,
        review_id="rev-123",
    )
    repo.create_audit(rec)

    # Record first decision
    audit, fb = repo.update_decision(
        review_id="rev-123",
        reviewer_id="reviewer_alice",
        human_decision="APPROVE",
        review_reason="Verified with customer via phone",
        reviewed_at="2026-11-20T10:15:00Z",
        updated_at="2026-11-20T10:15:00Z",
        feedback_id="fb-123",
    )
    assert audit is not None
    assert audit.human_decision == "APPROVE"

    # Attempt to change decision directly via SQL
    conn = sqlite3.connect(str(db_file))
    with pytest.raises(sqlite3.IntegrityError, match="completed human decisions in audit records cannot be altered"):
        conn.execute("UPDATE audit_records SET human_decision = 'REJECT' WHERE audit_id = ?", ("aud-test-dec",))
    conn.close()


def test_audit_repository_ground_truth(tmp_path):
    repo = AuditRepository(tmp_path / "test_gt.db")
    rec = _make_audit_record("aud-test-gt")
    repo.create_audit(rec)

    # Record ground truth
    repo.set_ground_truth("aud-test-gt", ground_truth=0, labeled_at="2026-11-20T10:30:00Z")

    record = repo.get_audit("aud-test-gt")
    assert record.ground_truth == 0
    assert record.labeled_at == "2026-11-20T10:30:00Z"

    # Test invalid ground truth rejected by CHECK constraint
    with pytest.raises(sqlite3.IntegrityError):
        conn = sqlite3.connect(str(tmp_path / "test_gt.db"))
        conn.execute("UPDATE audit_records SET ground_truth = 5 WHERE audit_id = ?", ("aud-test-gt",))


def test_audit_api_endpoints(tmp_path):
    db_file = tmp_path / "api_audit.db"
    repo = AuditRepository(db_file)
    service = AuditService(repo)

    with TestClient(app) as client:
        app.state.audit_repository = repo
        app.state.audit_service = service

        # Record a transaction through service
        outcome = _make_dummy_outcome(prediction=1, probability=0.88, risk="HIGH", action="AUTO_REJECT")
        outcome["requires_human_review"] = True
        record = service.record_transaction(transaction=FRAUD_TRANSACTION, pipeline_outcome=outcome)
        audit_id = record.audit_id

        # GET /audit
        resp = client.get("/audit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] >= 1
        assert any(item["audit_id"] == audit_id for item in data["records"])

        # GET /audit/{audit_id}
        resp = client.get(f"/audit/{audit_id}")
        assert resp.status_code == 200
        item = resp.json()
        assert item["audit_id"] == audit_id
        assert item["prediction"] == 1
        assert item["risk_level"] == "HIGH"

        # GET /audit/unknown-id -> 404
        resp = client.get("/audit/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

        # POST /audit/{audit_id}/ground-truth
        resp = client.post(f"/audit/{audit_id}/ground-truth", json={"ground_truth": 1})
        assert resp.status_code == 200
        gt_data = resp.json()
        assert gt_data["audit_id"] == audit_id
        assert gt_data["ground_truth"] == 1

        # Verify ground truth persisted
        record_db = repo.get_audit(audit_id)
        assert record_db.ground_truth == 1
