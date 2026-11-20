"""HTTP routes for Monitoring & Analytics."""

from typing import Annotated
from fastapi import APIRouter, HTTPException, Query, Request

from src.monitoring.schemas import (
    DecisionMetricsResponse,
    FeedbackMetricsResponse,
    GuardrailMetricsResponse,
    MonitoringSummaryResponse,
    PerformanceMetricsResponse,
)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/summary", response_model=MonitoringSummaryResponse)
def get_monitoring_summary(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
) -> MonitoringSummaryResponse:
    """Executive monitoring summary: transaction volumes, fraud rates, decisions, escalation, and feedback."""
    monitoring_service = getattr(request.app.state, "monitoring_service", None)
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is not initialized.")
    return monitoring_service.get_summary(start_time=start_time, end_time=end_time)


@router.get("/decisions", response_model=DecisionMetricsResponse)
def get_decision_metrics(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
) -> DecisionMetricsResponse:
    """Distribution of system recommendations and human review decisions."""
    monitoring_service = getattr(request.app.state, "monitoring_service", None)
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is not initialized.")
    return monitoring_service.get_decision_metrics(start_time=start_time, end_time=end_time)


@router.get("/guardrails", response_model=GuardrailMetricsResponse)
def get_guardrail_metrics(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
) -> GuardrailMetricsResponse:
    """Guardrail pass/review escalation rates and breakdown of individual reason code frequencies."""
    monitoring_service = getattr(request.app.state, "monitoring_service", None)
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is not initialized.")
    return monitoring_service.get_guardrail_metrics(start_time=start_time, end_time=end_time)


@router.get("/feedback", response_model=FeedbackMetricsResponse)
def get_feedback_metrics(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
) -> FeedbackMetricsResponse:
    """Human review feedback metrics, recommendation disagreement rates, and override counts."""
    monitoring_service = getattr(request.app.state, "monitoring_service", None)
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is not initialized.")
    return monitoring_service.get_feedback_metrics(start_time=start_time, end_time=end_time)


@router.get("/performance", response_model=PerformanceMetricsResponse)
def get_performance_metrics(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
) -> PerformanceMetricsResponse:
    """Model performance metrics evaluated strictly against genuine ground-truth labeled transactions."""
    monitoring_service = getattr(request.app.state, "monitoring_service", None)
    if monitoring_service is None:
        raise HTTPException(status_code=503, detail="Monitoring service is not initialized.")
    return monitoring_service.get_performance_metrics(start_time=start_time, end_time=end_time)
