"""Drift Detection service layer.

Coordinates running drift analysis on persisted audit windows or explicit transaction batches,
persisting the resulting reports, and retrieving historical analyses.
"""

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from src.audit.repository import AuditRepository
from src.drift.detector import DriftDetector
from src.drift.models import DriftReport
from src.drift.repository import DriftRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class DriftService:
    """Service for running and managing drift detection analyses."""

    def __init__(
        self,
        detector: DriftDetector,
        drift_repository: DriftRepository,
        audit_repository: AuditRepository,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = _new_id,
    ):
        self._detector = detector
        self._drift_repo = drift_repository
        self._audit_repo = audit_repository
        self._clock = clock
        self._id_factory = id_factory

    @property
    def detector(self) -> DriftDetector:
        return self._detector

    def run_analysis(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        explicit_transactions: list[dict[str, Any]] | None = None,
        min_samples: int = 20,
    ) -> DriftReport:
        """Execute drift analysis across feature distributions and model behaviors."""
        analysis_id = self._id_factory()
        now_str = self._clock().isoformat()

        if explicit_transactions is not None:
            txs = explicit_transactions
            preds = None
            probs = None
            risks = None
            actions = None
            guardrails = None
            gts = None
        else:
            audits, _ = self._audit_repo.list_audits(
                start_time=start_time,
                end_time=end_time,
                limit=100000,
            )
            txs = [a.transaction for a in audits]
            preds = [a.prediction for a in audits]
            probs = [a.probability for a in audits]
            risks = [a.risk_level for a in audits]
            actions = [a.recommended_action for a in audits]
            guardrails = [a.guardrail_status for a in audits]
            gts = [a.ground_truth for a in audits]

        report = self._detector.evaluate(
            transactions=txs,
            predictions=preds,
            probabilities=probs,
            risk_levels=risks,
            recommended_actions=actions,
            guardrail_statuses=guardrails,
            ground_truths=gts,
            window_start=start_time,
            window_end=end_time,
            analysis_id=analysis_id,
            created_at=now_str,
            min_samples=min_samples,
        )

        self._drift_repo.save_report(report)
        return report

    def get_latest_report(self) -> DriftReport | None:
        """Fetch the most recent drift report."""
        return self._drift_repo.get_latest_report()

    def get_report(self, analysis_id: str) -> DriftReport | None:
        """Fetch report by analysis_id."""
        return self._drift_repo.get_report(analysis_id)

    def list_reports(self, limit: int = 50, offset: int = 0) -> tuple[list[DriftReport], int]:
        """List historical reports."""
        return self._drift_repo.list_reports(limit=limit, offset=offset)
