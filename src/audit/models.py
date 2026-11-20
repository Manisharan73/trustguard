"""Audit and Feedback domain models.

Captures the complete lifecycle of a transaction:
- Model output: prediction, probability
- Risk assessment: risk_level
- Decision Engine: recommended_action
- Guardrails: Phase 5, Phase 8, combined verdict and reason codes
- Human oversight: review necessity, reviewer identity, human decision, reason, timestamps
- Feedback: system recommendation vs. human decision, disagreement flag
- Ground truth: optional true outcome for valid model performance evaluation
"""

from dataclasses import dataclass, field
from typing import Any, Mapping
import types


@dataclass(frozen=True)
class AuditRecord:
    """Immutable audit record for a single transaction lifecycle."""

    audit_id: str
    created_at: str  # ISO-8601 UTC
    transaction: Mapping[str, Any]

    def __post_init__(self):
        # Ensure transaction dict is immutable
        object.__setattr__(self, 'transaction', types.MappingProxyType(dict(self.transaction)))

    # Model & Engine outputs
    prediction: int
    probability: float
    risk_level: str
    recommended_action: str

    # Guardrail assessments
    phase5_guardrail_status: str
    phase5_guardrail_reasons: tuple[str, ...]
    phase8_guardrail_status: str | None
    phase8_guardrail_reasons: tuple[str, ...]
    confidence: float | None
    uncertainty: float | None
    ood_status: str | None
    ood_distance: float | None
    guardrail_status: str
    guardrail_reasons: tuple[str, ...]

    # Human review & lifecycle tracking
    requires_human_review: bool
    review_id: str | None = None
    reviewer_id: str | None = None
    human_decision: str | None = None  # APPROVE or REJECT
    review_reason: str | None = None
    reviewed_at: str | None = None  # ISO-8601 UTC

    # Ground truth (if genuinely available)
    ground_truth: int | None = None
    labeled_at: str | None = None

    updated_at: str = ""


@dataclass(frozen=True)
class FeedbackRecord:
    """Explicit feedback comparing system recommendation to human decision."""

    feedback_id: str
    audit_id: str
    review_id: str | None
    created_at: str  # ISO-8601 UTC

    recommended_action: str
    human_decision: str
    reviewer_id: str
    review_reason: str

    disagreement: bool
    is_override: bool
