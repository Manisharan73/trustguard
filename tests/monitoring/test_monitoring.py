"""Tests for monitoring and analytics services, metrics computation, and API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.audit.repository import AuditRepository
from src.audit.service import AuditService
from src.monitoring.service import MonitoringService
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


def _outcome(pred, prob, risk, action, reasons=None, requires_review=False):
    reasons = reasons or []
    return {
        "prediction": pred,
        "probability": prob,
        "risk_level": risk,
        "recommended_action": action,
        "guardrail_status": "REQUIRES_REVIEW" if requires_review else "PASSED",
        "guardrail_reasons": reasons,
        "confidence": max(prob, 1.0 - prob),
        "uncertainty": 1.0 - max(prob, 1.0 - prob),
        "ood_status": "NORMAL",
        "ood_distance": 2.0,
        "requires_human_review": requires_review,
        "phase5_guardrail_status": "REQUIRES_REVIEW" if requires_review else "PASSED",
        "phase5_guardrail_reasons": reasons,
        "phase8_guardrail_status": "PASSED",
        "phase8_guardrail_reasons": [],
    }


def test_monitoring_empty_database(tmp_path):
    repo = AuditRepository(tmp_path / "empty.db")
    monitoring = MonitoringService(repo)

    summary = monitoring.get_summary()
    assert summary.total_transactions == 0
    assert summary.auto_approve_count == 0

    perf = monitoring.get_performance_metrics()
    assert perf.has_ground_truth is False
    assert perf.accuracy is None


def test_monitoring_metrics_calculation(tmp_path):
    repo = AuditRepository(tmp_path / "metrics.db")
    audit_svc = AuditService(repo)
    monitoring = MonitoringService(repo)

    # 1. Legit transaction: AUTO_APPROVE
    rec1 = audit_svc.record_transaction(
        transaction=LEGITIMATE_TRANSACTION,
        pipeline_outcome=_outcome(0, 0.05, "LOW", "AUTO_APPROVE"),
    )
    # 2. Borderline transaction: HUMAN_REVIEW
    rec2 = audit_svc.record_transaction(
        transaction=LEGITIMATE_TRANSACTION,
        pipeline_outcome=_outcome(0, 0.45, "MEDIUM", "HUMAN_REVIEW", ["MEDIUM_RISK_REQUIRES_HUMAN_REVIEW"], requires_review=True),
        review_id="rev-1",
    )
    # Complete review with APPROVE
    audit_svc.record_decision(
        review_id="rev-1",
        reviewer_id="rev_bob",
        human_decision="APPROVE",
        review_reason="Legit user confirmed",
    )

    # 3. Fraud transaction: AUTO_REJECT
    rec3 = audit_svc.record_transaction(
        transaction=FRAUD_TRANSACTION,
        pipeline_outcome=_outcome(1, 0.90, "HIGH", "AUTO_REJECT", ["HIGH_RISK_REQUIRES_REJECTION"]),
    )

    # Summary
    summary = monitoring.get_summary()
    assert summary.total_transactions == 3
    assert summary.fraud_predictions == 1
    assert summary.legitimate_predictions == 2
    assert summary.auto_approve_count == 1
    assert summary.human_review_count == 1
    assert summary.auto_reject_count == 1
    assert summary.total_human_decisions == 1
    assert summary.disagreement_count == 0

    # Add ground truth to test model performance metrics
    audit_svc.record_ground_truth(rec1.audit_id, 0)  # TN
    audit_svc.record_ground_truth(rec2.audit_id, 0)  # TN
    audit_svc.record_ground_truth(rec3.audit_id, 1)  # TP

    perf = monitoring.get_performance_metrics()
    assert perf.has_ground_truth is True
    assert perf.labeled_samples == 3
    assert perf.confusion_matrix == [[2, 0], [0, 1]]
    assert perf.accuracy == 1.0
    assert perf.precision == 1.0
    assert perf.recall == 1.0
    assert perf.f1 == 1.0


def test_monitoring_api_endpoints(tmp_path):
    repo = AuditRepository(tmp_path / "api_monitoring.db")
    audit_svc = AuditService(repo)
    monitoring = MonitoringService(repo)

    with TestClient(app) as client:
        app.state.audit_repository = repo
        app.state.audit_service = audit_svc
        app.state.monitoring_service = monitoring

        # Record a transaction
        audit_svc.record_transaction(
            transaction=LEGITIMATE_TRANSACTION,
            pipeline_outcome=_outcome(0, 0.10, "LOW", "AUTO_APPROVE"),
        )

        resp = client.get("/monitoring/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_transactions"] == 1

        resp = client.get("/monitoring/decisions")
        assert resp.status_code == 200
        assert resp.json()["total_transactions"] == 1

        resp = client.get("/monitoring/guardrails")
        assert resp.status_code == 200
        assert resp.json()["passed_count"] == 1

        resp = client.get("/monitoring/feedback")
        assert resp.status_code == 200
        assert resp.json()["total_feedback_records"] == 0

        resp = client.get("/monitoring/performance")
        assert resp.status_code == 200
        assert resp.json()["has_ground_truth"] is False
