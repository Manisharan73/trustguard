"""HTTP routes for Drift Detection."""

from typing import Annotated
from fastapi import APIRouter, HTTPException, Path, Query, Request

from src.drift.schemas import (
    DriftAnalyzeRequest,
    DriftReportListResponse,
    DriftReportResponse,
)

router = APIRouter(prefix="/drift", tags=["drift"])

_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
AnalysisId = Annotated[str, Path(pattern=_UUID_PATTERN, description="Analysis ID (UUID)")]


@router.post("/analyze", response_model=DriftReportResponse)
def analyze_drift(
    request: Request,
    payload: DriftAnalyzeRequest | None = None,
) -> DriftReportResponse:
    """Trigger statistical drift detection over a historical time window or explicit batch."""
    drift_service = getattr(request.app.state, "drift_service", None)
    if drift_service is None:
        raise HTTPException(status_code=503, detail="Drift service is not initialized.")

    start_time = payload.start_time if payload else None
    end_time = payload.end_time if payload else None
    explicit_txs = payload.transactions if payload else None
    min_samples = payload.min_sample_size if payload else 20

    report = drift_service.run_analysis(
        start_time=start_time,
        end_time=end_time,
        explicit_transactions=explicit_txs,
        min_samples=min_samples,
    )
    return DriftReportResponse.from_report(report)


@router.get("/latest", response_model=DriftReportResponse)
def get_latest_drift_report(request: Request) -> DriftReportResponse:
    """Retrieve the most recent drift detection report."""
    drift_service = getattr(request.app.state, "drift_service", None)
    if drift_service is None:
        raise HTTPException(status_code=503, detail="Drift service is not initialized.")

    report = drift_service.get_latest_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No drift reports found. Trigger an analysis first via POST /drift/analyze.")
    return DriftReportResponse.from_report(report)


@router.get("/reports", response_model=DriftReportListResponse)
def list_drift_reports(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DriftReportListResponse:
    """List historical drift reports ordered newest first."""
    drift_service = getattr(request.app.state, "drift_service", None)
    if drift_service is None:
        raise HTTPException(status_code=503, detail="Drift service is not initialized.")

    reports, total = drift_service.list_reports(limit=limit, offset=offset)
    return DriftReportListResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        reports=[DriftReportResponse.from_report(r) for r in reports],
    )


@router.get("/reports/{analysis_id}", response_model=DriftReportResponse)
def get_drift_report(analysis_id: AnalysisId, request: Request) -> DriftReportResponse:
    """Retrieve a specific historical drift report by ID."""
    drift_service = getattr(request.app.state, "drift_service", None)
    if drift_service is None:
        raise HTTPException(status_code=503, detail="Drift service is not initialized.")

    report = drift_service.get_report(analysis_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Drift report {analysis_id} not found.")
    return DriftReportResponse.from_report(report)
