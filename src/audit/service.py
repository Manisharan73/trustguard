"""Audit and Feedback service layer.

Coordinates creating audit records from pipeline outputs, linking human decisions,
recording ground-truth outcomes, and retrieving records for monitoring.
"""

import math
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from src.audit.models import AuditRecord, FeedbackRecord
from src.audit.repository import AuditRepository


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class AuditService:
    """Record lifecycle events and feedback for transactions."""

    def __init__(
        self,
        repository: AuditRepository,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = _new_id,
    ):
        self._repository = repository
        self._clock = clock
        self._id_factory = id_factory

    @property
    def repository(self) -> AuditRepository:
        return self._repository

    def record_transaction(
        self,
        *,
        transaction: Mapping[str, Any],
        pipeline_outcome: Mapping[str, Any],
        review_id: str | None = None,
    ) -> AuditRecord:
        """Create and store an audit record from transaction features and pipeline output."""
        now_str = self._clock().isoformat()
        audit_id = self._id_factory()

        # If a review_id was passed, check if we already have an audit record for this review_id
        if review_id:
            existing = self._repository.get_audit_by_review_id(review_id)
            if existing is not None:
                return existing

        p5_status = pipeline_outcome.get("phase5_guardrail_status", pipeline_outcome.get("guardrail_status", "PASSED"))
        p5_reasons = tuple(pipeline_outcome.get("phase5_guardrail_reasons", ()))
        p8_status = pipeline_outcome.get("advanced_guardrail_status")
        p8_reasons = tuple(pipeline_outcome.get("advanced_guardrail_reasons", ()))
        combined_status = pipeline_outcome.get("guardrail_status", "PASSED")
        combined_reasons = tuple(pipeline_outcome.get("guardrail_reasons", ()))

        rec_action = pipeline_outcome["recommended_action"]
        requires_review = (combined_status == "REQUIRES_REVIEW" or rec_action == "HUMAN_REVIEW")

        record = AuditRecord(
            audit_id=audit_id,
            created_at=now_str,
            transaction=dict(transaction),
            prediction=int(pipeline_outcome["prediction"]),
            probability=float(pipeline_outcome["probability"]),
            risk_level=pipeline_outcome["risk_level"],
            recommended_action=rec_action,
            phase5_guardrail_status=p5_status,
            phase5_guardrail_reasons=p5_reasons,
            phase8_guardrail_status=p8_status,
            phase8_guardrail_reasons=p8_reasons,
            confidence=pipeline_outcome.get("confidence"),
            uncertainty=pipeline_outcome.get("uncertainty"),
            ood_status=pipeline_outcome.get("ood_status"),
            ood_distance=pipeline_outcome.get("ood_distance"),
            guardrail_status=combined_status,
            guardrail_reasons=combined_reasons,
            requires_human_review=requires_review,
            review_id=review_id,
            updated_at=now_str,
        )

        self._repository.create_audit(record)
        return record

    def record_decision(
        self,
        *,
        review_id: str,
        reviewer_id: str,
        human_decision: str,
        review_reason: str,
        reviewed_at: str | None = None,
    ) -> tuple[AuditRecord | None, FeedbackRecord | None]:
        """Update audit record with the reviewer's decision and record feedback."""
        now_str = reviewed_at or self._clock().isoformat()
        feedback_id = self._id_factory()

        return self._repository.update_decision(
            review_id=review_id,
            reviewer_id=reviewer_id,
            human_decision=human_decision,
            review_reason=review_reason,
            reviewed_at=now_str,
            updated_at=now_str,
            feedback_id=feedback_id,
        )

    def record_ground_truth(self, audit_id: str, ground_truth: int) -> AuditRecord:
        """Attach verified ground-truth outcome to an audit record."""
        if ground_truth not in (0, 1):
            raise ValueError(f"Invalid ground_truth: {ground_truth}. Expected 0 or 1.")
        now_str = self._clock().isoformat()
        audit = self._repository.set_ground_truth(audit_id, ground_truth, now_str)
        if audit is None:
            raise KeyError(f"Audit record {audit_id} not found.")
        return audit

    def get_audit(self, audit_id: str) -> AuditRecord | None:
        return self._repository.get_audit(audit_id)

    def list_audits(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        action: str | None = None,
        decision: str | None = None,
        requires_review: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AuditRecord], int]:
        return self._repository.list_audits(
            start_time=start_time,
            end_time=end_time,
            action=action,
            decision=decision,
            requires_review=requires_review,
            limit=limit,
            offset=offset,
        )

    def list_feedback(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        disagreement_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[FeedbackRecord], int, int]:
        return self._repository.list_feedback(
            start_time=start_time,
            end_time=end_time,
            disagreement_only=disagreement_only,
            limit=limit,
            offset=offset,
        )
