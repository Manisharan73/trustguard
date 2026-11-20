"""End-to-end integration test: full decision, audit, feedback, monitoring, and drift lifecycle."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.audit.repository import AuditRepository
from src.audit.service import AuditService
from src.drift.detector import DriftDetector
from src.drift.repository import DriftRepository
from src.drift.service import DriftService
from src.human_review.repository import ReviewRepository
from src.human_review.service import ReviewService
from src.ml.preprocess import get_train_test_data
from src.monitoring.service import MonitoringService
from tests.api.sample_transactions import LEGITIMATE_TRANSACTION

# Borderline transaction that requires review due to negative amount guardrail
BORDERLINE_TX = dict(LEGITIMATE_TRANSACTION, Amount=-5.0)


def test_complete_trustguard_lifecycle(tmp_path):
    db_file = tmp_path / "lifecycle.db"

    review_repo = ReviewRepository(db_file)
    review_svc = ReviewService(review_repo)
    audit_repo = AuditRepository(db_file)
    audit_svc = AuditService(audit_repo)
    monitoring_svc = MonitoringService(audit_repo)
    drift_repo = DriftRepository(db_file)
    drift_detector = DriftDetector.from_reference_file()
    drift_svc = DriftService(drift_detector, drift_repo, audit_repo)

    with TestClient(app) as client:
        app.state.review_service = review_svc
        app.state.audit_repository = audit_repo
        app.state.audit_service = audit_svc
        app.state.monitoring_service = monitoring_svc
        app.state.drift_repository = drift_repo
        app.state.drift_detector = drift_detector
        app.state.drift_service = drift_svc

        # 1. Create a review case (via /reviews)
        create_resp = client.post("/reviews", json=BORDERLINE_TX)
        assert create_resp.status_code == 201
        review_data = create_resp.json()
        review_id = review_data["review_id"]
        assert review_data["review_status"] == "PENDING"
        assert review_data["guardrail_status"] == "REQUIRES_REVIEW"

        # 2. Verify audit record created and linked
        records, total = audit_svc.list_audits(limit=10)
        assert total == 1
        audit_record = records[0]
        audit_id = audit_record.audit_id
        assert audit_record.review_id == review_id
        assert audit_record.guardrail_status == "REQUIRES_REVIEW"
        assert audit_record.human_decision is None

        # 3. Reviewer submits human decision (e.g. APPROVE)
        decision_resp = client.post(
            f"/reviews/{review_id}/decision",
            json={
                "reviewer_id": "rev_lead_1",
                "decision": "APPROVE",
                "reason": "Customer verified negative offset adjustment",
            },
        )
        assert decision_resp.status_code == 200

        # 4. Audit record updated, preserving original system prediction & recommendation
        updated_audit = audit_svc.get_audit(audit_id)
        assert updated_audit is not None
        assert updated_audit.recommended_action == audit_record.recommended_action  # Preserved!
        assert updated_audit.human_decision == "APPROVE"
        assert updated_audit.reviewer_id == "rev_lead_1"

        # 5. Feedback captured independently
        feedbacks, fb_total, _ = audit_svc.list_feedback(limit=10)
        assert fb_total == 1
        fb = feedbacks[0]
        assert fb.audit_id == audit_id
        assert fb.review_id == review_id
        assert fb.human_decision == "APPROVE"

        # 6. Monitoring reflects transaction, review, and feedback
        summary_resp = client.get("/monitoring/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert summary["total_transactions"] == 1
        assert summary["human_review_count"] == 1
        assert summary["total_human_decisions"] == 1

        # 7. Ground truth labeling
        gt_resp = client.post(f"/audit/{audit_id}/ground-truth", json={"ground_truth": 0})
        assert gt_resp.status_code == 200

        # 8. Monitoring performance metrics now compute genuine ML metrics
        perf_resp = client.get("/monitoring/performance")
        assert perf_resp.status_code == 200
        perf = perf_resp.json()
        assert perf["has_ground_truth"] is True
        assert perf["labeled_samples"] == 1
        assert perf["confusion_matrix"] == [[1, 0], [0, 0]]

        # 9. Drift detection execution
        _, X_test, _, _ = get_train_test_data()
        batch = [X_test.iloc[i].to_dict() for i in range(60)]
        drift_resp = client.post("/drift/analyze", json={"transactions": batch, "min_sample_size": 50})
        assert drift_resp.status_code == 200
        drift_data = drift_resp.json()
        assert "analysis_id" in drift_data
        assert drift_data["sample_count"] == 60

        # 10. Drift report retrieval
        latest_drift = client.get("/drift/latest")
        assert latest_drift.status_code == 200
        assert latest_drift.json()["analysis_id"] == drift_data["analysis_id"]
