"""Tests for drift detection: Population Stability Index (PSI), drift detector, and drift API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.audit.repository import AuditRepository
from src.audit.service import AuditService
from src.drift.detector import DriftDetector, calculate_psi
from src.drift.repository import DriftRepository
from src.drift.service import DriftService
from src.ml.preprocess import get_train_test_data
from tests.api.sample_transactions import LEGITIMATE_TRANSACTION


def test_calculate_psi():
    np.random.seed(42)
    # Reference decile edges from N(0, 1)
    ref_edges = list(np.percentile(np.random.normal(0, 1, 5000), np.linspace(0, 100, 11)))
    actual_similar = np.random.normal(0, 1, 1000)
    actual_shifted = np.random.normal(3, 1, 1000)

    # Identical/similar distribution should yield very low PSI (< 0.1)
    psi_similar = calculate_psi(actual_similar, ref_edges)
    assert psi_similar < 0.10

    # Heavily shifted distribution should yield high PSI (> 0.25)
    psi_shifted = calculate_psi(actual_shifted, ref_edges)
    assert psi_shifted > 0.25


def test_drift_detector_with_normal_and_drifted_data():
    detector = DriftDetector.from_reference_file()
    assert len(detector.feature_quantiles) == 30
    assert detector.ref_sample_count > 0

    # 1. Normal batch (sampled from held-out test data)
    _, X_test, _, _ = get_train_test_data()
    normal_transactions = [X_test.iloc[i].to_dict() for i in range(150)]

    normal_report = detector.evaluate(transactions=normal_transactions, min_samples=50)
    assert normal_report.overall_status in ("NO_DRIFT", "MODERATE_DRIFT")
    assert normal_report.drift_detected is False

    # 2. Artificially drifted batch (V1 shifted significantly by +15.0)
    drifted_transactions = []
    for tx in normal_transactions:
        drifted_tx = dict(tx)
        drifted_tx["V1"] += 15.0
        drifted_transactions.append(drifted_tx)

    drifted_report = detector.evaluate(transactions=drifted_transactions, min_samples=50)
    assert drifted_report.feature_metrics["V1"]["status"] == "SIGNIFICANT_DRIFT"
    assert drifted_report.feature_metrics["V1"]["psi"] > 0.20
    assert drifted_report.drift_detected is True


def test_drift_sample_size_threshold():
    detector = DriftDetector.from_reference_file()
    small_batch = [dict(LEGITIMATE_TRANSACTION) for _ in range(10)]
    report = detector.evaluate(transactions=small_batch, min_samples=50)
    assert report.overall_status == "INSUFFICIENT_DATA"
    assert report.drift_detected is False


def test_drift_api_endpoints(tmp_path):
    audit_repo = AuditRepository(tmp_path / "drift_audit.db")
    drift_repo = DriftRepository(tmp_path / "drift_reports.db")
    detector = DriftDetector.from_reference_file()
    service = DriftService(detector, drift_repo, audit_repo)

    with TestClient(app) as client:
        app.state.drift_repository = drift_repo
        app.state.drift_service = service
        app.state.drift_detector = detector

        # Generate batch of 55 transactions
        batch = [dict(LEGITIMATE_TRANSACTION) for _ in range(55)]

        # POST /drift/analyze
        resp = client.post("/drift/analyze", json={"transactions": batch, "min_sample_size": 50})
        assert resp.status_code == 200
        data = resp.json()
        assert "analysis_id" in data
        analysis_id = data["analysis_id"]
        assert data["sample_count"] == 55

        # GET /drift/latest
        resp_latest = client.get("/drift/latest")
        assert resp_latest.status_code == 200
        assert resp_latest.json()["analysis_id"] == analysis_id

        # GET /drift/reports
        resp_list = client.get("/drift/reports")
        assert resp_list.status_code == 200
        assert resp_list.json()["total_count"] >= 1

        # GET /drift/reports/{id}
        resp_detail = client.get(f"/drift/reports/{analysis_id}")
        assert resp_detail.status_code == 200
        assert resp_detail.json()["analysis_id"] == analysis_id
