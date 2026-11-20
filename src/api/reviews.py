"""HTTP routes for the human review workflow (Phase 6).

This layer only translates HTTP <-> the service: it parses requests, calls the
ReviewService and maps domain errors to status codes. No workflow rules live here.

  POST /reviews                       register a transaction for human review   201
  GET  /reviews/pending               the pending queue (oldest first)          200
  GET  /reviews/{review_id}           one review, pending or completed          200 / 404
  POST /reviews/{review_id}/decision  record the human decision                 200 / 404 / 409

Reviewer identity is a plain identifier from the request body: there is no
authentication or role-based access control in this phase.
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Request

from src.api.pipeline import run_pipeline
from src.api.schemas import TransactionRequest
from src.human_review.models import (
    ReviewAlreadyCompletedError,
    ReviewNotFoundError,
    ReviewNotRequiredError,
    ReviewValidationError,
)
from src.human_review.schemas import DecisionRequest, ReviewResponse

router = APIRouter(prefix="/reviews", tags=["human review"])

# Canonical lowercase UUID, which is what the service generates.
_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
ReviewId = Annotated[str, Path(pattern=_UUID_PATTERN, description="Review ID (UUID)")]


@router.post("", response_model=ReviewResponse, status_code=201)
def create_review(transaction: TransactionRequest, request: Request) -> ReviewResponse:
    """Assess a transaction and, if it needs human review, add it to the queue.

    The preserved prediction, risk and guardrail fields are computed here by the
    system, never supplied by the client. Returns 422 if no review is needed.
    """
    features = transaction.model_dump()
    outcome = run_pipeline(features, request.app.state)
    try:
        case = request.app.state.review_service.submit_for_review(
            transaction=features, **outcome
        )
    except ReviewNotRequiredError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    audit_service = getattr(request.app.state, "audit_service", None)
    if audit_service is not None:
        audit_meta = run_pipeline(
            features, request.app.state, include_advanced=True, include_audit_meta=True
        )
        audit_service.record_transaction(
            transaction=features,
            pipeline_outcome=audit_meta,
            review_id=case.review_id,
        )

    return ReviewResponse.from_case(case)


# NOTE: /pending must be registered before /{review_id} or it would be captured by it.
@router.get("/pending", response_model=list[ReviewResponse])
def list_pending_reviews(request: Request) -> list[ReviewResponse]:
    """The pending review queue, oldest first."""
    cases = request.app.state.review_service.list_pending()
    return [ReviewResponse.from_case(case) for case in cases]


@router.get("/{review_id}", response_model=ReviewResponse)
def get_review(review_id: ReviewId, request: Request) -> ReviewResponse:
    """One review: pending, or completed with the recorded human decision."""
    try:
        case = request.app.state.review_service.get_review(review_id)
    except ReviewNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return ReviewResponse.from_case(case)


@router.post("/{review_id}/decision", response_model=ReviewResponse)
def submit_decision(
    review_id: ReviewId, decision: DecisionRequest, request: Request
) -> ReviewResponse:
    """Record the reviewer's decision. A review can be decided exactly once."""
    try:
        case = request.app.state.review_service.submit_decision(
            review_id, decision.reviewer_id, decision.decision, decision.reason
        )
    except ReviewNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ReviewAlreadyCompletedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ReviewValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    audit_service = getattr(request.app.state, "audit_service", None)
    if audit_service is not None:
        audit_service.record_decision(
            review_id=review_id,
            reviewer_id=decision.reviewer_id,
            human_decision=decision.decision,
            review_reason=decision.reason,
            reviewed_at=case.reviewed_at,
        )

    return ReviewResponse.from_case(case)
