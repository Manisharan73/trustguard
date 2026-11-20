"""Pydantic schemas for the human review API.

Built-in string constraints are used instead of custom validators on purpose:
a custom validator's ValueError ends up inside the validation error's context as
an exception object, which cannot be serialized to JSON by the API's 422 handler.
Built-in constraints produce plain, serializable error details.

DecisionRequest forbids extra fields, so a reviewer cannot smuggle in a new
prediction, probability, risk level or recommended action.
"""

from dataclasses import asdict
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from src.human_review.models import ReviewCase
from src.human_review.service import MAX_REASON_LENGTH, MAX_REVIEWER_ID_LENGTH

# Whitespace is stripped first, so a whitespace-only value fails min_length=1.
ReviewerId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_REVIEWER_ID_LENGTH)
]
ReviewReason = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_REASON_LENGTH)
]


class DecisionRequest(BaseModel):
    """A reviewer's independent decision. Nothing else may be supplied."""

    model_config = ConfigDict(extra="forbid")

    reviewer_id: ReviewerId
    decision: Literal["APPROVE", "REJECT"]
    reason: ReviewReason


class ReviewResponse(BaseModel):
    """A review case: the original system output plus the (optional) human decision.

    The first group of fields is the system's output and never changes. The
    human_* / reviewer_* / review_reason / reviewed_at fields are null until the
    review is COMPLETED. There is no final_decision field.
    """

    review_id: str
    review_status: Literal["PENDING", "COMPLETED"]
    created_at: str
    transaction: dict[str, Any]

    # original system output (immutable)
    prediction: Literal[0, 1]
    probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommended_action: Literal["AUTO_APPROVE", "HUMAN_REVIEW", "AUTO_REJECT"]
    guardrail_status: Literal["PASSED", "REQUIRES_REVIEW"]
    guardrail_reasons: list[str]

    # the human's decision (null while PENDING)
    reviewer_id: str | None = None
    human_decision: Literal["APPROVE", "REJECT"] | None = None
    review_reason: str | None = None
    reviewed_at: str | None = None

    @classmethod
    def from_case(cls, case: ReviewCase) -> "ReviewResponse":
        return cls(**asdict(case))
