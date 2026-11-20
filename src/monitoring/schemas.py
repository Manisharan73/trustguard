"""Pydantic response schemas for Monitoring & Analytics."""

from pydantic import BaseModel, Field


class MonitoringSummaryResponse(BaseModel):
    """High-level summary of TrustGuard system activity and performance."""

    total_transactions: int
    fraud_predictions: int
    legitimate_predictions: int
    fraud_rate: float

    auto_approve_count: int
    human_review_count: int
    auto_reject_count: int
    escalation_rate: float

    total_human_decisions: int
    human_approvals: int
    human_rejections: int
    disagreement_count: int
    disagreement_rate: float

    labeled_transaction_count: int
    model_accuracy: float | None = None

    window_start: str | None = None
    window_end: str | None = None


class DecisionMetricsResponse(BaseModel):
    """Breakdown of system recommendations vs. human review decisions."""

    total_transactions: int
    decision_counts: dict[str, int]
    decision_percentages: dict[str, float]
    human_review_rate: float
    human_decision_counts: dict[str, int]
    human_decision_percentages: dict[str, float]

    window_start: str | None = None
    window_end: str | None = None


class GuardrailMetricsResponse(BaseModel):
    """Guardrail pass/escalation rates and reason code frequencies."""

    total_evaluated: int
    passed_count: int
    requires_review_count: int
    escalation_rate: float
    phase5_triggers_count: int
    phase8_triggers_count: int
    reason_frequencies: dict[str, int]

    window_start: str | None = None
    window_end: str | None = None


class FeedbackMetricsResponse(BaseModel):
    """Human oversight feedback metrics and system-vs-human agreement matrix."""

    total_feedback_records: int
    disagreement_count: int
    disagreement_rate: float
    override_count: int
    override_rate: float
    matrix: dict[str, int]

    window_start: str | None = None
    window_end: str | None = None


class PerformanceMetricsResponse(BaseModel):
    """Model performance metrics evaluated against genuine ground-truth labels."""

    has_ground_truth: bool
    message: str
    labeled_samples: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    confusion_matrix: list[list[int]] | None = None

    window_start: str | None = None
    window_end: str | None = None
