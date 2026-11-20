"""Drift detection domain models."""

from dataclasses import dataclass, field
from typing import Any

# Drift statuses
STATUS_NO_DRIFT = "NO_DRIFT"
STATUS_MODERATE_DRIFT = "MODERATE_DRIFT"
STATUS_SIGNIFICANT_DRIFT = "SIGNIFICANT_DRIFT"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

VALID_DRIFT_STATUSES = (
    STATUS_NO_DRIFT,
    STATUS_MODERATE_DRIFT,
    STATUS_SIGNIFICANT_DRIFT,
    STATUS_INSUFFICIENT_DATA,
)


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature: str
    psi: float
    status: str  # NO_DRIFT, MODERATE_DRIFT, SIGNIFICANT_DRIFT
    warning: str | None = None


@dataclass(frozen=True)
class DriftReport:
    """Complete multi-dimensional drift detection report."""

    analysis_id: str
    created_at: str  # ISO-8601 UTC
    window_start: str | None
    window_end: str | None
    sample_count: int
    reference_sample_count: int

    overall_status: str  # NO_DRIFT, MODERATE_DRIFT, SIGNIFICANT_DRIFT, INSUFFICIENT_DATA
    drift_detected: bool

    mean_feature_psi: float
    max_feature_psi: float
    drifted_features_count: int

    feature_metrics: dict[str, dict[str, Any]]
    behavioral_metrics: dict[str, Any]
    concept_drift: dict[str, Any]

    method: str
    thresholds: dict[str, float]
    limitations: list[str] = field(default_factory=list)
