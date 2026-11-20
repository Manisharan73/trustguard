"""Human review workflow rules.

Responsibility: decide WHAT is allowed. Persistence is delegated to the
repository and HTTP concerns to the API layer.

Rules enforced here
-------------------
* A case is created only for a transaction that needs review: guardrail_status is
  REQUIRES_REVIEW or the recommended action is HUMAN_REVIEW. Otherwise
  ReviewNotRequiredError (nothing is stored).
* A new case is PENDING and carries the system output exactly as supplied.
* A PENDING case can receive one human decision (APPROVE or REJECT) with a
  non-blank reviewer_id and reason. It then becomes COMPLETED.
* A COMPLETED case can never be decided again or modified.
* The human decision is stored separately; the prediction, probability, risk
  level, recommended action and guardrail results are never touched.

Reviewer identity is just an identifier supplied by the caller. This phase has
no authentication or role-based access control.
"""

import math
import numbers
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone

from src.decision.engine import HUMAN_REVIEW, RISK_LEVELS, VALID_ACTIONS
from src.human_review.models import (
    REVIEW_PENDING,
    VALID_DECISIONS,
    ReviewCase,
    ReviewNotFoundError,
    ReviewNotRequiredError,
    ReviewValidationError,
)
from src.human_review.repository import ReviewRepository

REQUIRES_REVIEW = "REQUIRES_REVIEW"  # same value the Guardrails Engine reports
VALID_GUARDRAIL_STATUSES = ("PASSED", REQUIRES_REVIEW)

MAX_REVIEWER_ID_LENGTH = 100
MAX_REASON_LENGTH = 2000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


def _clean_text(value, name: str, max_length: int) -> str:
    """Return the stripped text; reject non-strings, blank and over-long values."""
    if not isinstance(value, str):
        raise ReviewValidationError(f"{name} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise ReviewValidationError(f"{name} must not be empty or whitespace-only.")
    if len(cleaned) > max_length:
        raise ReviewValidationError(f"{name} must be at most {max_length} characters.")
    return cleaned


class ReviewService:
    """Create review cases and record human decisions."""

    def __init__(
        self,
        repository: ReviewRepository,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[], str] = _new_id,
    ):
        self._repository = repository
        self._clock = clock
        self._id_factory = id_factory

    # ---------- creating a case ----------

    def submit_for_review(
        self,
        *,
        transaction: Mapping,
        prediction: int,
        probability: float,
        risk_level: str,
        recommended_action: str,
        guardrail_status: str,
        guardrail_reasons,
        **extra,
    ) -> ReviewCase:
        """Create a PENDING review case from the system's output for a transaction.

        Raises ReviewNotRequiredError if the transaction does not need review and
        ValueError if the supplied system output is itself invalid.
        """
        self._validate_system_output(
            transaction,
            prediction,
            probability,
            risk_level,
            recommended_action,
            guardrail_status,
            guardrail_reasons,
        )

        if guardrail_status != REQUIRES_REVIEW and recommended_action != HUMAN_REVIEW:
            raise ReviewNotRequiredError(
                "Transaction does not require human review: guardrails passed and "
                "the recommended action is not HUMAN_REVIEW."
            )

        case = ReviewCase(
            review_id=self._id_factory(),
            created_at=self._clock().isoformat(),
            review_status=REVIEW_PENDING,
            transaction=dict(transaction),
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            recommended_action=recommended_action,
            guardrail_status=guardrail_status,
            guardrail_reasons=tuple(guardrail_reasons),
        )
        self._repository.create(case)
        return case

    @staticmethod
    def _validate_system_output(
        transaction,
        prediction,
        probability,
        risk_level,
        recommended_action,
        guardrail_status,
        guardrail_reasons,
    ) -> None:
        if not isinstance(transaction, Mapping) or len(transaction) == 0:
            raise ValueError("transaction must be a non-empty mapping.")
        if (
            isinstance(prediction, bool)
            or not isinstance(prediction, numbers.Integral)
            or prediction not in (0, 1)
        ):
            raise ValueError(f"Invalid prediction: {prediction!r}. Expected 0 or 1.")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, numbers.Real)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(f"Invalid probability: {probability!r}. Expected 0 to 1.")
        if not isinstance(risk_level, str) or risk_level not in RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {risk_level!r}.")
        if not isinstance(recommended_action, str) or recommended_action not in VALID_ACTIONS:
            raise ValueError(f"Invalid recommended action: {recommended_action!r}.")
        if (
            not isinstance(guardrail_status, str)
            or guardrail_status not in VALID_GUARDRAIL_STATUSES
        ):
            raise ValueError(f"Invalid guardrail status: {guardrail_status!r}.")
        if isinstance(guardrail_reasons, (str, bytes)) or not all(
            isinstance(reason, str) for reason in guardrail_reasons
        ):
            raise ValueError("guardrail_reasons must be a collection of strings.")

    # ---------- reading ----------

    def get_review(self, review_id: str) -> ReviewCase:
        """Return a review (pending or completed). Raises ReviewNotFoundError."""
        case = self._repository.get(review_id)
        if case is None:
            raise ReviewNotFoundError(f"Review {review_id} was not found.")
        return case

    def list_pending(self) -> list[ReviewCase]:
        """All PENDING reviews, oldest first."""
        return self._repository.list_by_status(REVIEW_PENDING)

    # ---------- recording the human decision ----------

    def submit_decision(
        self, review_id: str, reviewer_id: str, decision: str, reason: str
    ) -> ReviewCase:
        """Record the reviewer's independent decision on a PENDING review.

        reviewer_id and reason are stored stripped of surrounding whitespace.
        Raises ReviewValidationError for invalid input, ReviewNotFoundError for an
        unknown ID and ReviewAlreadyCompletedError if a decision already exists.
        Reviewer identity is an unauthenticated identifier in this phase.
        """
        reviewer_id = _clean_text(reviewer_id, "reviewer_id", MAX_REVIEWER_ID_LENGTH)
        if not isinstance(decision, str) or decision not in VALID_DECISIONS:
            raise ReviewValidationError(
                f"decision must be one of: {list(VALID_DECISIONS)}."
            )
        reason = _clean_text(reason, "reason", MAX_REASON_LENGTH)

        return self._repository.complete(
            review_id=review_id,
            reviewer_id=reviewer_id,
            human_decision=decision,
            review_reason=reason,
            reviewed_at=self._clock().isoformat(),
        )
