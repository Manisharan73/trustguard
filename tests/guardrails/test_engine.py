"""Unit tests for the Guardrails Engine. Imports the real implementation."""

import copy
import dataclasses

import numpy as np
import pytest

from src.decision.engine import AUTO_APPROVE, AUTO_REJECT, HUMAN_REVIEW, VALID_ACTIONS
from src.guardrails.engine import (
    APPROVAL_CONTEXT_MISMATCH,
    HUMAN_REVIEW_REQUIRED,
    NEGATIVE_AMOUNT,
    NEGATIVE_TIME,
    NON_FINITE_FEATURE,
    NON_NUMERIC_FEATURE,
    PASSED,
    REJECTION_CONTEXT_MISMATCH,
    REQUIRES_REVIEW,
    GuardrailConfig,
    GuardrailResult,
    evaluate_guardrails,
)


@pytest.fixture
def config():
    return GuardrailConfig()


@pytest.fixture
def transaction():
    """A clean transaction: every value finite and numeric, Amount/Time >= 0."""
    return {"Time": 1000.0, "V1": 0.5, "V2": -1.2, "Amount": 25.0}


def approve(transaction, config, prediction=0, risk="LOW"):
    return evaluate_guardrails(transaction, prediction, risk, AUTO_APPROVE, config)


def reject(transaction, config, prediction=1, risk="HIGH"):
    return evaluate_guardrails(transaction, prediction, risk, AUTO_REJECT, config)


# ---------- normal behaviour ----------

def test_valid_low_legitimate_auto_approve_passes(transaction, config):
    result = approve(transaction, config)
    assert result.status == PASSED
    assert result.recommended_action == "AUTO_APPROVE"
    assert result.reasons == ()


def test_valid_high_fraud_auto_reject_passes(transaction, config):
    result = reject(transaction, config)
    assert result.status == PASSED
    assert result.recommended_action == "AUTO_REJECT"
    assert result.reasons == ()


@pytest.mark.parametrize("prediction", [0, 1])
def test_human_review_always_requires_review(transaction, config, prediction):
    result = evaluate_guardrails(transaction, prediction, "MEDIUM", HUMAN_REVIEW, config)
    assert result.status == REQUIRES_REVIEW
    assert result.recommended_action == "HUMAN_REVIEW"
    assert result.reasons == (HUMAN_REVIEW_REQUIRED,)


def test_status_and_reason_values_are_the_documented_strings():
    assert PASSED == "PASSED"
    assert REQUIRES_REVIEW == "REQUIRES_REVIEW"
    assert HUMAN_REVIEW_REQUIRED == "HUMAN_REVIEW_REQUIRED"
    assert APPROVAL_CONTEXT_MISMATCH == "APPROVAL_CONTEXT_MISMATCH"
    assert REJECTION_CONTEXT_MISMATCH == "REJECTION_CONTEXT_MISMATCH"


# ---------- Rule A: data quality ----------

def test_negative_amount_requires_review(transaction, config):
    transaction["Amount"] = -0.01
    result = approve(transaction, config)
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (NEGATIVE_AMOUNT,)


def test_negative_time_requires_review(transaction, config):
    transaction["Time"] = -1.0
    result = approve(transaction, config)
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (NEGATIVE_TIME,)


def test_zero_amount_and_zero_time_are_valid_boundaries(transaction, config):
    transaction["Amount"] = 0.0
    transaction["Time"] = 0
    assert approve(transaction, config).status == PASSED


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_feature_requires_review(transaction, config, bad_value):
    transaction["V1"] = bad_value
    result = approve(transaction, config)
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (NON_FINITE_FEATURE,)


def test_nan_infinity_and_negative_infinity_are_each_reported_once(transaction, config):
    transaction.update({"V1": float("nan"), "V2": float("inf"), "Time": float("-inf")})
    result = approve(transaction, config)
    assert result.reasons == (NON_FINITE_FEATURE,)  # one code, not one per feature


@pytest.mark.parametrize("bad_value", ["abc", "10", None, [1.0], {"x": 1}, True, False])
def test_non_numeric_feature_requires_review(transaction, config, bad_value):
    transaction["V1"] = bad_value
    result = approve(transaction, config)
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (NON_NUMERIC_FEATURE,)


def test_non_numeric_amount_is_flagged_not_crashed(transaction, config):
    transaction["Amount"] = "lots"
    result = approve(transaction, config)
    assert result.reasons == (NON_NUMERIC_FEATURE,)


def test_non_finite_amount_does_not_also_report_negative_amount(transaction, config):
    transaction["Amount"] = float("-inf")
    assert approve(transaction, config).reasons == (NON_FINITE_FEATURE,)


def test_multiple_data_quality_violations_are_all_reported_in_fixed_order(transaction, config):
    transaction.update(
        {"Amount": -5.0, "Time": -1.0, "V1": float("nan"), "V2": "not a number"}
    )
    result = approve(transaction, config)
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (
        NON_NUMERIC_FEATURE,
        NON_FINITE_FEATURE,
        NEGATIVE_AMOUNT,
        NEGATIVE_TIME,
    )


def test_numpy_numbers_are_handled(transaction, config):
    transaction["V1"] = np.float64(0.25)
    transaction["V2"] = np.int64(3)
    assert approve(transaction, config).status == PASSED
    transaction["V1"] = np.float64("nan")
    assert approve(transaction, config).reasons == (NON_FINITE_FEATURE,)


def test_integers_too_large_for_float_do_not_crash(transaction, config):
    transaction["V1"] = 10**400
    assert approve(transaction, config).status == PASSED


def test_data_quality_is_reported_for_every_action(transaction, config):
    transaction["Amount"] = -1.0
    result = evaluate_guardrails(transaction, 0, "MEDIUM", HUMAN_REVIEW, config)
    assert result.reasons == (NEGATIVE_AMOUNT, HUMAN_REVIEW_REQUIRED)


# ---------- Rules B and C: recommendation consistency ----------

def test_auto_approve_with_fraud_prediction_requires_review(transaction, config):
    result = approve(transaction, config, prediction=1, risk="LOW")
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (APPROVAL_CONTEXT_MISMATCH,)


def test_auto_approve_with_high_risk_requires_review(transaction, config):
    result = approve(transaction, config, prediction=0, risk="HIGH")
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (APPROVAL_CONTEXT_MISMATCH,)


def test_auto_approve_with_medium_risk_requires_review(transaction, config):
    result = approve(transaction, config, prediction=0, risk="MEDIUM")
    assert result.reasons == (APPROVAL_CONTEXT_MISMATCH,)


def test_auto_reject_with_legitimate_prediction_requires_review(transaction, config):
    result = reject(transaction, config, prediction=0, risk="HIGH")
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (REJECTION_CONTEXT_MISMATCH,)


def test_auto_reject_with_low_risk_requires_review(transaction, config):
    result = reject(transaction, config, prediction=1, risk="LOW")
    assert result.status == REQUIRES_REVIEW
    assert result.reasons == (REJECTION_CONTEXT_MISMATCH,)


def test_auto_reject_with_medium_risk_requires_review(transaction, config):
    result = reject(transaction, config, prediction=1, risk="MEDIUM")
    assert result.reasons == (REJECTION_CONTEXT_MISMATCH,)


def test_data_quality_failure_alone_is_not_labelled_a_context_mismatch(transaction, config):
    transaction["Amount"] = -1.0  # context itself is consistent
    assert approve(transaction, config).reasons == (NEGATIVE_AMOUNT,)
    assert reject(transaction, config).reasons == (NEGATIVE_AMOUNT,)


def test_data_quality_and_context_mismatch_are_both_reported(transaction, config):
    transaction["Amount"] = -1.0
    result = approve(transaction, config, prediction=1, risk="HIGH")
    assert result.reasons == (NEGATIVE_AMOUNT, APPROVAL_CONTEXT_MISMATCH)


# ---------- the original recommendation is never changed ----------

@pytest.mark.parametrize("action", VALID_ACTIONS)
@pytest.mark.parametrize("amount", [25.0, -25.0])
def test_original_recommended_action_is_always_echoed_unchanged(transaction, config, action, amount):
    transaction["Amount"] = amount
    result = evaluate_guardrails(transaction, 0, "LOW", action, config)
    assert result.recommended_action == action


def test_evaluation_does_not_modify_the_transaction(transaction, config):
    before = copy.deepcopy(transaction)
    approve(transaction, config)
    assert transaction == before


def test_there_is_no_final_decision_field():
    field_names = {f.name for f in dataclasses.fields(GuardrailResult)}
    assert field_names == {"status", "recommended_action", "reasons"}


def test_result_is_immutable(transaction, config):
    result = approve(transaction, config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "PASSED"


# ---------- validation: invalid internal arguments raise, never pass ----------

@pytest.mark.parametrize("bad_prediction", [2, -1, None, "0", 0.0, 1.0, True, False, [0]])
def test_invalid_prediction_raises(transaction, config, bad_prediction):
    with pytest.raises(ValueError, match="Invalid prediction"):
        evaluate_guardrails(transaction, bad_prediction, "LOW", AUTO_APPROVE, config)


@pytest.mark.parametrize("bad_risk", ["low", "CRITICAL", "", None, 1, ["LOW"]])
def test_invalid_risk_level_raises(transaction, config, bad_risk):
    with pytest.raises(ValueError, match="Invalid risk level"):
        evaluate_guardrails(transaction, 0, bad_risk, AUTO_APPROVE, config)


@pytest.mark.parametrize("bad_action", ["approve", "SHRUG", "", None, 1, "AUTO_APPROVE "])
def test_invalid_action_raises(transaction, config, bad_action):
    with pytest.raises(ValueError, match="Invalid recommended action"):
        evaluate_guardrails(transaction, 0, "LOW", bad_action, config)


@pytest.mark.parametrize("bad_config", [None, {}, "config", {"amount_field": "Amount"}])
def test_config_that_is_not_a_guardrail_config_raises_type_error(transaction, bad_config):
    with pytest.raises(TypeError, match="GuardrailConfig"):
        evaluate_guardrails(transaction, 0, "LOW", AUTO_APPROVE, bad_config)


@pytest.mark.parametrize(
    "bad_config",
    [
        GuardrailConfig(amount_field=""),
        GuardrailConfig(time_field=""),
        GuardrailConfig(amount_field=None),
        GuardrailConfig(time_field=123),
        GuardrailConfig(amount_field="Same", time_field="Same"),
    ],
)
def test_guardrail_config_with_invalid_values_raises(transaction, bad_config):
    with pytest.raises(ValueError, match="GuardrailConfig"):
        evaluate_guardrails(transaction, 0, "LOW", AUTO_APPROVE, bad_config)


@pytest.mark.parametrize("bad_transaction", [None, [1.0, 2.0], "transaction", 42])
def test_transaction_that_is_not_a_mapping_raises_type_error(config, bad_transaction):
    with pytest.raises(TypeError, match="mapping"):
        evaluate_guardrails(bad_transaction, 0, "LOW", AUTO_APPROVE, config)


def test_empty_transaction_raises(config):
    with pytest.raises(ValueError, match="at least one feature"):
        evaluate_guardrails({}, 0, "LOW", AUTO_APPROVE, config)


def test_omitted_arguments_raise_type_error(transaction, config):
    with pytest.raises(TypeError):
        evaluate_guardrails(transaction, 0, "LOW", AUTO_APPROVE)  # config omitted


def test_invalid_arguments_never_return_a_passed_result(transaction, config):
    # Safety net: every invalid-argument case raises instead of returning anything.
    for args in [
        (transaction, 2, "LOW", AUTO_APPROVE, config),
        (transaction, 0, "NOPE", AUTO_APPROVE, config),
        (transaction, 0, "LOW", "NOPE", config),
        (transaction, 0, "LOW", AUTO_APPROVE, None),
    ]:
        with pytest.raises((TypeError, ValueError)):
            evaluate_guardrails(*args)


# ---------- configuration ----------

def test_default_config_values():
    config = GuardrailConfig()
    assert config.amount_field == "Amount"
    assert config.time_field == "Time"


def test_config_instances_are_equal_but_independent():
    assert GuardrailConfig() == GuardrailConfig()
    assert GuardrailConfig() is not GuardrailConfig()


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        GuardrailConfig().amount_field = "Value"


def test_custom_amount_field_is_used():
    custom = GuardrailConfig(amount_field="Value")
    tx = {"Time": 1.0, "Value": -3.0, "Amount": -3.0}
    result = evaluate_guardrails(tx, 0, "LOW", AUTO_APPROVE, custom)
    assert result.reasons == (NEGATIVE_AMOUNT,)  # "Value" checked...
    tx_ok = {"Time": 1.0, "Value": 3.0, "Amount": -3.0}
    assert evaluate_guardrails(tx_ok, 0, "LOW", AUTO_APPROVE, custom).status == PASSED  # ..."Amount" no longer is


def test_custom_time_field_is_used():
    custom = GuardrailConfig(time_field="Seconds")
    tx = {"Seconds": -1.0, "Time": -1.0, "Amount": 5.0}
    result = evaluate_guardrails(tx, 0, "LOW", AUTO_APPROVE, custom)
    assert result.reasons == (NEGATIVE_TIME,)
