"""Pydantic schemas for Drift Detection API endpoints."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from src.drift.models import DriftReport


class DriftAnalyzeRequest(BaseModel):
    """Request payload to trigger drift analysis on a window or batch."""

    model_config = ConfigDict(extra="forbid")

    start_time: str | None = Field(default=None, description="ISO-8601 UTC start timestamp for window analysis")
    end_time: str | None = Field(default=None, description="ISO-8601 UTC end timestamp for window analysis")
    transactions: list[dict[str, Any]] | None = Field(
        default=None, description="Optional explicit batch of transactions to analyze"
    )
    min_sample_size: int = Field(
        default=20, ge=1, le=10000, description="Minimum number of samples required to evaluate drift"
    )


class DriftReportResponse(BaseModel):
    """Statistical drift analysis report."""

    analysis_id: str
    created_at: str
    window_start: str | None = None
    window_end: str | None = None
    sample_count: int
    reference_sample_count: int

    overall_status: Literal["NO_DRIFT", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT", "INSUFFICIENT_DATA"]
    drift_detected: bool

    mean_feature_psi: float
    max_feature_psi: float
    drifted_features_count: int

    feature_metrics: dict[str, Any]
    behavioral_metrics: dict[str, Any]
    concept_drift: dict[str, Any]

    method: str
    thresholds: dict[str, float]
    limitations: list[str]

    @classmethod
    def from_report(cls, report: DriftReport) -> "DriftReportResponse":
        return cls(
            analysis_id=report.analysis_id,
            created_at=report.created_at,
            window_start=report.window_start,
            window_end=report.window_end,
            sample_count=report.sample_count,
            reference_sample_count=report.reference_sample_count,
            overall_status=report.overall_status,  # type: ignore
            drift_detected=report.drift_detected,
            mean_feature_psi=report.mean_feature_psi,
            max_feature_psi=report.max_feature_psi,
            drifted_features_count=report.drifted_features_count,
            feature_metrics=report.feature_metrics,
            behavioral_metrics=report.behavioral_metrics,
            concept_drift=report.concept_drift,
            method=report.method,
            thresholds=report.thresholds,
            limitations=list(report.limitations),
        )


class DriftReportListResponse(BaseModel):
    """Paginated list of historical drift reports."""

    total_count: int
    limit: int
    offset: int
    reports: list[DriftReportResponse]
