"""Pydantic schemas for Audit and Feedback API endpoints."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from src.audit.models import AuditRecord, FeedbackRecord


class GroundTruthRequest(BaseModel):
    """Ground-truth outcome submission for model performance monitoring."""

    model_config = ConfigDict(extra="forbid")

    ground_truth: Literal[0, 1] = Field(description="0 = legitimate, 1 = fraud")


class AuditResponse(BaseModel):
    """Full lifecycle view of a transaction audit record."""

    audit_id: str
    created_at: str
    transaction: dict[str, Any]

    prediction: int
    probability: float
    risk_level: str
    recommended_action: str

    phase5_guardrail_status: str
    phase5_guardrail_reasons: list[str]
    phase8_guardrail_status: str | None = None
    phase8_guardrail_reasons: list[str] = Field(default_factory=list)
    confidence: float | None = None
    uncertainty: float | None = None
    ood_status: str | None = None
    ood_distance: float | None = None
    guardrail_status: str
    guardrail_reasons: list[str]

    requires_human_review: bool
    review_id: str | None = None
    reviewer_id: str | None = None
    human_decision: Literal["APPROVE", "REJECT"] | None = None
    review_reason: str | None = None
    reviewed_at: str | None = None

    ground_truth: Literal[0, 1] | None = None
    labeled_at: str | None = None
    updated_at: str

    @classmethod
    def from_record(cls, record: AuditRecord) -> "AuditResponse":
        return cls(
            audit_id=record.audit_id,
            created_at=record.created_at,
            transaction=record.transaction,
            prediction=record.prediction,
            probability=record.probability,
            risk_level=record.risk_level,
            recommended_action=record.recommended_action,
            phase5_guardrail_status=record.phase5_guardrail_status,
            phase5_guardrail_reasons=list(record.phase5_guardrail_reasons),
            phase8_guardrail_status=record.phase8_guardrail_status,
            phase8_guardrail_reasons=list(record.phase8_guardrail_reasons),
            confidence=record.confidence,
            uncertainty=record.uncertainty,
            ood_status=record.ood_status,
            ood_distance=record.ood_distance,
            guardrail_status=record.guardrail_status,
            guardrail_reasons=list(record.guardrail_reasons),
            requires_human_review=record.requires_human_review,
            review_id=record.review_id,
            reviewer_id=record.reviewer_id,
            human_decision=record.human_decision,  # type: ignore
            review_reason=record.review_reason,
            reviewed_at=record.reviewed_at,
            ground_truth=record.ground_truth,  # type: ignore
            labeled_at=record.labeled_at,
            updated_at=record.updated_at,
        )


class AuditListResponse(BaseModel):
    """Paginated list of audit records."""

    total_count: int
    limit: int
    offset: int
    records: list[AuditResponse]


class FeedbackResponse(BaseModel):
    """Human decision feedback record."""

    feedback_id: str
    audit_id: str
    review_id: str | None = None
    created_at: str
    recommended_action: str
    human_decision: Literal["APPROVE", "REJECT"]
    reviewer_id: str
    review_reason: str
    disagreement: bool
    is_override: bool

    @classmethod
    def from_record(cls, record: FeedbackRecord) -> "FeedbackResponse":
        return cls(
            feedback_id=record.feedback_id,
            audit_id=record.audit_id,
            review_id=record.review_id,
            created_at=record.created_at,
            recommended_action=record.recommended_action,
            human_decision=record.human_decision,  # type: ignore
            reviewer_id=record.reviewer_id,
            review_reason=record.review_reason,
            disagreement=record.disagreement,
            is_override=record.is_override,
        )


class FeedbackListResponse(BaseModel):
    """Paginated list of feedback records."""

    total_count: int
    limit: int
    offset: int
    disagreement_count: int
    records: list[FeedbackResponse]
