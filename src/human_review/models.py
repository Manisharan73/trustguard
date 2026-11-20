"""Human review domain model: constants, the ReviewCase record and domain errors.

A ReviewCase keeps four things strictly separate:
  * the model's output            -> prediction, probability
  * the system's assessment       -> risk_level, recommended_action
  * the guardrail evaluation      -> guardrail_status, guardrail_reasons
  * the HUMAN's independent call  -> human_decision (+ reviewer, reason, time)

The first three are written once when the case is created and are never changed.
The human fields stay None until a reviewer completes the case.
There is deliberately no "final_decision" field; that belongs to a later phase.
"""

from dataclasses import dataclass, field

# ---------- review status ----------
REVIEW_PENDING = "PENDING"
REVIEW_COMPLETED = "COMPLETED"
VALID_REVIEW_STATUSES = (REVIEW_PENDING, REVIEW_COMPLETED)

# ---------- human decisions ----------
DECISION_APPROVE = "APPROVE"
DECISION_REJECT = "REJECT"
VALID_DECISIONS = (DECISION_APPROVE, DECISION_REJECT)


@dataclass(frozen=True)
class ReviewCase:
    """One transaction awaiting (or having received) a human decision."""

    review_id: str
    created_at: str  # ISO-8601, UTC
    review_status: str  # PENDING or COMPLETED
    transaction: dict  # the submitted features, as the reviewer needs to see them

    # --- original system output: immutable once stored ---
    prediction: int
    probability: float
    risk_level: str
    recommended_action: str
    guardrail_status: str
    guardrail_reasons: tuple = field(default_factory=tuple)

    # --- the human's independent decision: None until COMPLETED ---
    reviewer_id: str | None = None
    human_decision: str | None = None
    review_reason: str | None = None
    reviewed_at: str | None = None  # ISO-8601, UTC


# ---------- domain errors (the API layer maps these to HTTP status codes) ----------

class ReviewError(Exception):
    """Base class for human-review workflow errors."""


class ReviewNotFoundError(ReviewError):
    """No review exists with the given ID."""


class ReviewAlreadyCompletedError(ReviewError):
    """The review already has a human decision and cannot be changed."""


class ReviewNotRequiredError(ReviewError):
    """The transaction does not need human review, so no case is created."""


class ReviewValidationError(ReviewError, ValueError):
    """A reviewer-supplied value (reviewer_id, decision, reason) is invalid."""
