"""API request and response schemas.

The request schema is generated from FEATURE_COLUMNS, the same list the model is
trained with, so API field names can never drift from the training features.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.ml.preprocess import FEATURE_COLUMNS

# Illustrative payload shown in Swagger (/docs). Not a real transaction.
_EXAMPLE = {col: 0.0 for col in FEATURE_COLUMNS}
_EXAMPLE["Amount"] = 10.0

# Every feature is a required finite float:
#  - missing field          -> rejected (422)
#  - wrong type (e.g. text) -> rejected (422)
#  - NaN / Infinity         -> rejected (422)
#  - unknown extra field    -> rejected (422)
# Business-level data quality checks belong to the Phase 5 guardrail, not here.
TransactionRequest = create_model(
    "TransactionRequest",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE}),
    **{col: (float, Field(allow_inf_nan=False)) for col in FEATURE_COLUMNS},
)


class PredictionResponse(BaseModel):
    """Model output, risk assessment, recommendation and guardrail evaluation.

    prediction and probability come from the ML model. risk_level comes from the
    Risk Engine and never alters the prediction. recommended_action comes from
    the Decision Engine and is a RECOMMENDATION, not the final transaction
    decision. guardrail_status and guardrail_reasons come from the Guardrails
    Engine; they evaluate the recommendation and never change it. Nothing here
    says a transaction was actually approved or rejected. Human review and any
    final decision are later phases.
    """

    prediction: Literal[0, 1] = Field(description="0 = legitimate, 1 = fraud")
    probability: float = Field(ge=0.0, le=1.0, description="Model probability of fraud (class 1)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Risk Engine assessment of the fraud probability"
    )
    recommended_action: Literal["AUTO_APPROVE", "HUMAN_REVIEW", "AUTO_REJECT"] = Field(
        description="Decision Engine recommendation for this risk level (not the final decision)"
    )
    guardrail_status: Literal["PASSED", "REQUIRES_REVIEW"] = Field(
        description="Guardrails Engine verdict on whether the recommendation may proceed"
    )
    guardrail_reasons: list[str] = Field(
        description="Machine-readable reason codes; empty when the guardrails passed"
    )
    # Phase 8: Advanced Guardrail fields (optional; populated when include_advanced=True or configured)
    advanced_guardrail_status: Literal["PASSED", "REQUIRES_REVIEW"] | None = Field(
        default=None, description="Advanced guardrail verdict on whether the recommendation may proceed"
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Model prediction confidence max(p, 1-p)"
    )
    uncertainty: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Model prediction uncertainty (1 - confidence)"
    )
    ood_status: Literal["IN_DISTRIBUTION", "OUT_OF_DISTRIBUTION"] | None = Field(
        default=None, description="Out-of-distribution status"
    )
    ood_distance: float | None = Field(
        default=None, ge=0.0, description="Standardized Euclidean distance from reference distribution"
    )
    advanced_guardrail_reasons: list[str] | None = Field(
        default=None, description="Machine-readable reasons flagged by advanced guardrails"
    )
