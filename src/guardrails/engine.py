"""Guardrails Engine (v1): is the Decision Engine's recommendation safe to proceed with?

Responsibility: evaluate a recommendation and report PASSED or REQUIRES_REVIEW,
with machine-readable reason codes. Nothing else.

It never modifies the prediction, probability, risk level or recommended action
(the original recommendation is echoed back unchanged), never approves or
rejects a real transaction, and has no dependency on FastAPI. Human-review
workflows and any "final decision" belong to later phases.

Failure policy (applied consistently)
-------------------------------------
* Problems with the TRANSACTION DATA (non-numeric or non-finite values, negative
  Amount or Time) are expected, data-driven events. They NEVER raise: they
  produce REQUIRES_REVIEW with reason codes.
* Problems with the INTERNAL ARGUMENTS (prediction not 0/1, unknown risk level or
  action, config of the wrong type or with invalid values, transaction that is
  not a non-empty mapping) mean an upstream component broke its contract. They
  RAISE (TypeError / ValueError), like the Risk and Decision engines, and can
  never produce PASSED. Converting them into routine review results would hide
  bugs behind normal human-review traffic.
"""

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

from src.decision.engine import (
    AUTO_APPROVE,
    AUTO_REJECT,
    HUMAN_REVIEW,
    RISK_LEVELS,
    VALID_ACTIONS,
)

# ---------- guardrail status ----------
PASSED = "PASSED"
REQUIRES_REVIEW = "REQUIRES_REVIEW"

# ---------- reason codes ----------
NON_NUMERIC_FEATURE = "NON_NUMERIC_FEATURE"
NON_FINITE_FEATURE = "NON_FINITE_FEATURE"
NEGATIVE_AMOUNT = "NEGATIVE_AMOUNT"
NEGATIVE_TIME = "NEGATIVE_TIME"
APPROVAL_CONTEXT_MISMATCH = "APPROVAL_CONTEXT_MISMATCH"
REJECTION_CONTEXT_MISMATCH = "REJECTION_CONTEXT_MISMATCH"
HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"

_LOW = "LOW"
_HIGH = "HIGH"


@dataclass(frozen=True)
class GuardrailConfig:
    """Names of the transaction fields that must be non-negative."""

    amount_field: str = "Amount"
    time_field: str = "Time"


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of the guardrail evaluation.

    status:              "PASSED" or "REQUIRES_REVIEW".
    recommended_action:  the Decision Engine's recommendation, UNCHANGED.
    reasons:             machine-readable reason codes (empty when PASSED).
    """

    status: str
    recommended_action: str
    reasons: tuple = ()


# ---------- argument validation (raises; see failure policy) ----------

def _validate_config(config: GuardrailConfig) -> None:
    if not isinstance(config, GuardrailConfig):
        raise TypeError(
            f"config must be a GuardrailConfig instance, got {type(config).__name__}."
        )
    for name in ("amount_field", "time_field"):
        value = getattr(config, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"GuardrailConfig.{name} must be a non-empty string.")
    if config.amount_field == config.time_field:
        raise ValueError("GuardrailConfig.amount_field and time_field must differ.")


def _validate_context(prediction, risk_level, recommended_action) -> None:
    # bool is a subclass of int, so True/False are excluded explicitly.
    if (
        isinstance(prediction, bool)
        or not isinstance(prediction, numbers.Integral)
        or prediction not in (0, 1)
    ):
        raise ValueError(f"Invalid prediction: {prediction!r}. Expected 0 or 1.")
    if not isinstance(risk_level, str) or risk_level not in RISK_LEVELS:
        raise ValueError(
            f"Invalid risk level: {risk_level!r}. Expected one of: {list(RISK_LEVELS)}."
        )
    if not isinstance(recommended_action, str) or recommended_action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid recommended action: {recommended_action!r}. "
            f"Expected one of: {list(VALID_ACTIONS)}."
        )


def _validate_transaction(transaction) -> None:
    if not isinstance(transaction, Mapping):
        raise TypeError(
            f"transaction must be a mapping of feature names to values, "
            f"got {type(transaction).__name__}."
        )
    if len(transaction) == 0:
        raise ValueError("transaction must contain at least one feature.")


# ---------- Rule A: transaction data quality (never raises) ----------

def _is_number(value) -> bool:
    # bool is a subclass of int but is not a numeric feature value.
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    if isinstance(value, numbers.Integral):
        return True  # integers cannot be NaN/inf (and may be too large for float)
    return math.isfinite(value)


def _check_data_quality(transaction: Mapping, config: GuardrailConfig) -> list:
    non_numeric = non_finite = negative_amount = negative_time = False

    for name, value in transaction.items():
        if not _is_number(value):
            non_numeric = True
            continue
        if not _is_finite(value):
            non_finite = True
            continue
        # Sign checks only run on values already known to be finite numbers.
        if name == config.amount_field and value < 0:
            negative_amount = True
        if name == config.time_field and value < 0:
            negative_time = True

    flags = (
        (non_numeric, NON_NUMERIC_FEATURE),
        (non_finite, NON_FINITE_FEATURE),
        (negative_amount, NEGATIVE_AMOUNT),
        (negative_time, NEGATIVE_TIME),
    )
    return [code for flagged, code in flags if flagged]


# ---------- Rules B, C, D: recommendation consistency ----------

def _check_recommendation(prediction, risk_level, recommended_action) -> list:
    if recommended_action == AUTO_APPROVE:
        if prediction != 0 or risk_level != _LOW:
            return [APPROVAL_CONTEXT_MISMATCH]
    elif recommended_action == AUTO_REJECT:
        if prediction != 1 or risk_level != _HIGH:
            return [REJECTION_CONTEXT_MISMATCH]
    elif recommended_action == HUMAN_REVIEW:
        return [HUMAN_REVIEW_REQUIRED]
    return []


def evaluate_guardrails(
    transaction: Mapping,
    prediction: int,
    risk_level: str,
    recommended_action: str,
    config: GuardrailConfig,
) -> GuardrailResult:
    """Evaluate whether the recommended action may proceed without human review.

    Returns a GuardrailResult. status is PASSED only when no reason applies:
      A. data quality: every supplied value is a finite number; Amount and Time
         are not negative                       -> NON_NUMERIC_FEATURE,
                                                   NON_FINITE_FEATURE,
                                                   NEGATIVE_AMOUNT, NEGATIVE_TIME
      B. AUTO_APPROVE needs prediction 0 and LOW risk
                                                -> APPROVAL_CONTEXT_MISMATCH
      C. AUTO_REJECT needs prediction 1 and HIGH risk
                                                -> REJECTION_CONTEXT_MISMATCH
      D. HUMAN_REVIEW always requires review    -> HUMAN_REVIEW_REQUIRED

    Data-quality reasons are listed first, each code at most once. The
    context-mismatch code is added only when prediction/risk really disagree
    with the action, so a data-quality failure alone is not mislabelled as a
    mismatch. Raises TypeError/ValueError for invalid arguments (see the module
    docstring); never mutates its inputs.
    """
    _validate_config(config)
    _validate_context(prediction, risk_level, recommended_action)
    _validate_transaction(transaction)

    reasons = _check_data_quality(transaction, config)
    reasons += _check_recommendation(prediction, risk_level, recommended_action)

    return GuardrailResult(
        status=REQUIRES_REVIEW if reasons else PASSED,
        recommended_action=recommended_action,  # echoed unchanged
        reasons=tuple(reasons),
    )
