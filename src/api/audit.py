"""HTTP routes for Audit and Feedback."""

from typing import Annotated
from fastapi import APIRouter, HTTPException, Path, Query, Request

from src.audit.schemas import (
    AuditListResponse,
    AuditResponse,
    FeedbackListResponse,
    GroundTruthRequest,
)

router = APIRouter(tags=["audit"])

_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
AuditId = Annotated[str, Path(pattern=_UUID_PATTERN, description="Audit Record ID (UUID)")]


@router.get("/audit", response_model=AuditListResponse)
def list_audits(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
    action: Annotated[str | None, Query(description="Filter by recommended action")] = None,
    decision: Annotated[str | None, Query(description="Filter by human decision")] = None,
    requires_review: Annotated[bool | None, Query(description="Filter by requires_human_review flag")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditListResponse:
    """Retrieve paginated audit records matching filters."""
    audit_service = getattr(request.app.state, "audit_service", None)
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Audit service is not initialized.")

    records, total = audit_service.list_audits(
        start_time=start_time,
        end_time=end_time,
        action=action,
        decision=decision,
        requires_review=requires_review,
        limit=limit,
        offset=offset,
    )
    return AuditListResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        records=[AuditResponse.from_record(r) for r in records],
    )


@router.get("/audit/{audit_id}", response_model=AuditResponse)
def get_audit(audit_id: AuditId, request: Request) -> AuditResponse:
    """Retrieve a single audit record by its unique ID."""
    audit_service = getattr(request.app.state, "audit_service", None)
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Audit service is not initialized.")

    record = audit_service.get_audit(audit_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Audit record {audit_id} not found.")
    return AuditResponse.from_record(record)


@router.post("/audit/{audit_id}/ground-truth", response_model=AuditResponse)
def record_ground_truth(
    audit_id: AuditId, payload: GroundTruthRequest, request: Request
) -> AuditResponse:
    """Attach verified ground-truth outcome (0 = legitimate, 1 = fraud) to an audit record."""
    audit_service = getattr(request.app.state, "audit_service", None)
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Audit service is not initialized.")

    try:
        record = audit_service.record_ground_truth(audit_id, payload.ground_truth)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return AuditResponse.from_record(record)


@router.get("/feedback", response_model=FeedbackListResponse)
def list_feedback(
    request: Request,
    start_time: Annotated[str | None, Query(description="Start ISO-8601 UTC timestamp")] = None,
    end_time: Annotated[str | None, Query(description="End ISO-8601 UTC timestamp")] = None,
    disagreement_only: Annotated[bool, Query(description="Filter only cases where human disagreed with recommendation")] = False,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> FeedbackListResponse:
    """Retrieve human review decision feedback records."""
    audit_service = getattr(request.app.state, "audit_service", None)
    if audit_service is None:
        raise HTTPException(status_code=503, detail="Audit service is not initialized.")

    records, total, disagree_cnt = audit_service.list_feedback(
        start_time=start_time,
        end_time=end_time,
        disagreement_only=disagreement_only,
        limit=limit,
        offset=offset,
    )
    from src.audit.schemas import FeedbackResponse

    return FeedbackListResponse(
        total_count=total,
        limit=limit,
        offset=offset,
        disagreement_count=disagree_cnt,
        records=[FeedbackResponse.from_record(r) for r in records],
    )
