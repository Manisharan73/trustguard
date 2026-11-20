"""Tests for Phase 8 Advanced Guardrails.
These tests are additive; they must not break any existing test suite.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.guardrails.advanced import (
    AdvancedGuardrailConfig,
    AdvancedGuardrailResult,
    OODDetector,
    calculate_confidence_and_uncertainty,
    evaluate_advanced_guardrails,
    combine_guardrails,
    check_prediction_context,
    validate_prediction_context,
    IN_DISTRIBUTION,
    OUT_OF_DISTRIBUTION,
    LOW_MODEL_CONFIDENCE,
    HIGH_MODEL_UNCERTAINTY,
    OUT_OF_DISTRIBUTION_REASON,
    PREDICTION_CONTEXT_MISMATCH,
)
from src.guardrails.engine import (
    GuardrailConfig,
    GuardrailResult,
    PASSED,
    REQUIRES_REVIEW,
    APPROVAL_CONTEXT_MISMATCH,
    REJECTION_CONTEXT_MISMATCH,
    HUMAN_REVIEW_REQUIRED,
    NEGATIVE_AMOUNT,
)
from src.decision.engine import AUTO_APPROVE, AUTO_REJECT, HUMAN_REVIEW
from src.risk.engine import RiskConfig, assess_risk
from src.ml.predict import predict_transaction
from src.ml.preprocess import FEATURE_COLUMNS
from tests.api.sample_transactions import FRAUD_TRANSACTION, LEGITIMATE_TRANSACTION


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def ood_detector():
    from src.ml.predict import load_model

    model = load_model()
    preprocessor = model.named_steps["preprocessing"]
    return OODDetector.from_reference_file(preprocessor=preprocessor)


@pytest.fixture(scope="module")
def loaded_model():
    from src.ml.predict import load_model
    return load_model()


# ============================================================
# A. Confidence and Uncertainty Tests
# ============================================================

class TestConfidenceCalculation:
    """Test confidence = max(p, 1-p) and uncertainty = 1 - confidence."""

    def test_probability_zero(self):
        conf, unc = calculate_confidence_and_uncertainty(0.0)
        assert conf == 1.0
        assert unc == 0.0

    def test_probability_one(self):
        conf, unc = calculate_confidence_and_uncertainty(1.0)
        assert conf == 1.0
        assert unc == 0.0

    def test_probability_half(self):
        conf, unc = calculate_confidence_and_uncertainty(0.5)
        assert conf == pytest.approx(0.5)
        assert unc == pytest.approx(0.5)

    def test_just_below_default_threshold(self):
        conf, unc = calculate_confidence_and_uncertainty(0.79)
        assert conf == pytest.approx(0.79)
        assert unc == pytest.approx(0.21)

    def test_just_above_default_threshold(self):
        conf, unc = calculate_confidence_and_uncertainty(0.81)
        assert conf == pytest.approx(0.81)
        assert unc == pytest.approx(0.19)

    def test_symmetry(self):
        """confidence(p) == confidence(1-p) for all valid p."""
        for p in [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]:
            c1, u1 = calculate_confidence_and_uncertainty(p)
            c2, u2 = calculate_confidence_and_uncertainty(1.0 - p)
            assert c1 == pytest.approx(c2)
            assert u1 == pytest.approx(u2)

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            calculate_confidence_and_uncertainty(-0.1)

    def test_invalid_above_one(self):
        with pytest.raises(ValueError):
            calculate_confidence_and_uncertainty(1.1)

    def test_invalid_nan(self):
        with pytest.raises(ValueError):
            calculate_confidence_and_uncertainty(float("nan"))

    def test_invalid_inf(self):
        with pytest.raises(ValueError):
            calculate_confidence_and_uncertainty(float("inf"))

    def test_invalid_string(self):
        with pytest.raises(TypeError):
            calculate_confidence_and_uncertainty("0.5")

    def test_invalid_bool(self):
        with pytest.raises(TypeError):
            calculate_confidence_and_uncertainty(True)

    def test_integer_accepted(self):
        """Integer 0 and 1 are valid probabilities."""
        c0, u0 = calculate_confidence_and_uncertainty(0)
        assert c0 == 1.0
        c1, u1 = calculate_confidence_and_uncertainty(1)
        assert c1 == 1.0


class TestConfidenceGuardrail:
    """Test that low-confidence predictions trigger REQUIRES_REVIEW."""

    def test_low_confidence_triggers_review(self):
        cfg = AdvancedGuardrailConfig(min_confidence=0.80, max_uncertainty=0.20)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.75,
            risk_level="MEDIUM",
            recommended_action=HUMAN_REVIEW,
            config=cfg,
            ood_detector=None,
        )
        assert result.status == REQUIRES_REVIEW
        assert LOW_MODEL_CONFIDENCE in result.reasons

    def test_high_confidence_passes(self):
        cfg = AdvancedGuardrailConfig(min_confidence=0.80, max_uncertainty=0.20)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.05,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=None,
        )
        assert result.status == PASSED
        assert LOW_MODEL_CONFIDENCE not in result.reasons
        assert HIGH_MODEL_UNCERTAINTY not in result.reasons

    def test_confidence_exactly_at_threshold(self):
        """Confidence exactly at min_confidence should PASS (not strictly below)."""
        cfg = AdvancedGuardrailConfig(min_confidence=0.80, max_uncertainty=0.20)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.80,
            risk_level="MEDIUM",
            recommended_action=HUMAN_REVIEW,
            config=cfg,
            ood_detector=None,
        )
        assert LOW_MODEL_CONFIDENCE not in result.reasons

    def test_no_duplicate_confidence_uncertainty_reasons(self):
        """LOW_MODEL_CONFIDENCE and HIGH_MODEL_UNCERTAINTY should not both appear."""
        cfg = AdvancedGuardrailConfig(min_confidence=0.95, max_uncertainty=0.05)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.6,
            risk_level="MEDIUM",
            recommended_action=HUMAN_REVIEW,
            config=cfg,
            ood_detector=None,
        )
        assert result.status == REQUIRES_REVIEW
        # Only one of the two should be present
        conf_reasons = [r for r in result.reasons if r in (LOW_MODEL_CONFIDENCE, HIGH_MODEL_UNCERTAINTY)]
        assert len(conf_reasons) == 1

    def test_disabled_confidence_check(self):
        """When require_review_on_low_confidence=False, no confidence reason added."""
        cfg = AdvancedGuardrailConfig(
            min_confidence=0.95,
            require_review_on_low_confidence=False,
        )
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.6,
            risk_level="MEDIUM",
            recommended_action=HUMAN_REVIEW,
            config=cfg,
            ood_detector=None,
        )
        assert LOW_MODEL_CONFIDENCE not in result.reasons
        assert HIGH_MODEL_UNCERTAINTY not in result.reasons


# ============================================================
# B. OOD Detection Tests
# ============================================================

class TestOODDetector:
    """Tests for Out-of-Distribution detection."""

    def test_reference_at_mean_is_in_distribution(self):
        """A transaction at exactly the reference mean should be IN_DISTRIBUTION."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds, threshold=20.0)
        tx = {col: 0.0 for col in FEATURE_COLUMNS}
        status, dist = det.detect(tx)
        assert status == IN_DISTRIBUTION
        assert dist == pytest.approx(0.0)

    def test_close_transaction_is_in_distribution(self):
        """A transaction close to the mean should be in-distribution."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds, threshold=20.0)
        tx = {col: 0.1 for col in FEATURE_COLUMNS}
        status, dist = det.detect(tx)
        assert status == IN_DISTRIBUTION
        assert dist < 20.0

    def test_extreme_transaction_is_ood(self):
        """A transaction far from the mean should be OOD."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds, threshold=10.0)
        tx = {col: 100.0 for col in FEATURE_COLUMNS}
        status, dist = det.detect(tx)
        assert status == OUT_OF_DISTRIBUTION
        assert dist > 10.0

    def test_zero_variance_features_safe(self):
        """Features with zero variance should not cause division by zero."""
        means = np.zeros(30)
        stds = np.zeros(30)  # all zero variance
        det = OODDetector(means=means, stds=stds, threshold=20.0)
        tx = {col: 5.0 for col in FEATURE_COLUMNS}
        status, dist = det.detect(tx)
        # With all zero-variance, all normalized contributions are 0
        assert dist == pytest.approx(0.0)
        assert status == IN_DISTRIBUTION

    def test_non_finite_input_returns_inf(self):
        """Non-finite values should return inf distance (OOD)."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds, threshold=20.0)
        tx = {col: 0.0 for col in FEATURE_COLUMNS}
        tx["V1"] = float("nan")
        status, dist = det.detect(tx)
        assert status == OUT_OF_DISTRIBUTION
        assert math.isinf(dist)

    def test_incorrect_feature_count_raises(self):
        """Numpy input with wrong length raises ValueError."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds)
        with pytest.raises(ValueError, match="feature length"):
            det.compute_distance(np.zeros(25))

    def test_missing_feature_raises(self):
        """Dict input missing features raises ValueError."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds)
        with pytest.raises(ValueError, match="missing required features"):
            det.compute_distance({"V1": 1.0})  # missing most features

    def test_mismatched_means_stds_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            OODDetector(means=np.zeros(30), stds=np.ones(25))

    def test_2d_means_raises(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            OODDetector(means=np.zeros((30, 2)), stds=np.ones(30))

    def test_from_reference_file_loads(self, ood_detector):
        """OOD detector loaded from reference file has correct shapes."""
        assert len(ood_detector.means) == 30
        assert len(ood_detector.stds) == 30

    def test_ood_triggers_review(self, ood_detector):
        """An extreme transaction triggers OOD review."""
        cfg = AdvancedGuardrailConfig(ood_distance_threshold=10.0, require_review_on_ood=True)
        ood_tx = dict(LEGITIMATE_TRANSACTION, Amount=1_000_000.0)
        result = evaluate_advanced_guardrails(
            transaction=ood_tx,
            prediction=0,
            probability=0.001,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=ood_detector,
        )
        assert result.status == REQUIRES_REVIEW
        assert OUT_OF_DISTRIBUTION_REASON in result.reasons

    def test_ood_disabled(self, ood_detector):
        """When ood_enabled=False, OOD check is skipped."""
        cfg = AdvancedGuardrailConfig(ood_enabled=False)
        ood_tx = dict(LEGITIMATE_TRANSACTION, Amount=1_000_000.0)
        result = evaluate_advanced_guardrails(
            transaction=ood_tx,
            prediction=0,
            probability=0.001,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=ood_detector,
        )
        assert OUT_OF_DISTRIBUTION_REASON not in result.reasons
        assert result.ood_status == IN_DISTRIBUTION

    def test_ood_no_detector(self):
        """When ood_detector is None, OOD defaults to IN_DISTRIBUTION."""
        cfg = AdvancedGuardrailConfig(ood_enabled=True)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.05,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=None,
        )
        assert result.ood_status == IN_DISTRIBUTION
        assert result.ood_distance == 0.0

    def test_numpy_input_accepted(self):
        """Numpy array input works for compute_distance."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds, threshold=20.0)
        x = np.ones(30) * 0.5
        dist = det.compute_distance(x)
        expected = math.sqrt(30 * 0.25)
        assert dist == pytest.approx(expected)

    def test_unsupported_type_raises(self):
        """Unsupported input type raises TypeError."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds)
        with pytest.raises(TypeError, match="Unsupported"):
            det.compute_distance("not a transaction")

    def test_bool_feature_returns_inf(self):
        """Bool value in a feature returns inf distance."""
        means = np.zeros(30)
        stds = np.ones(30)
        det = OODDetector(means=means, stds=stds, threshold=20.0)
        tx = {col: 0.0 for col in FEATURE_COLUMNS}
        tx["V1"] = True
        status, dist = det.detect(tx)
        assert status == OUT_OF_DISTRIBUTION


# ============================================================
# C. Prediction-Context Consistency Tests
# ============================================================

class TestPredictionContext:
    """Test prediction-context consistency checks."""

    def test_case1_valid_low_approve(self):
        """Case 1: p=0.15, prediction=0, risk=LOW, action=AUTO_APPROVE -> valid."""
        reasons = check_prediction_context(0, 0.15, "LOW", AUTO_APPROVE)
        assert reasons == []

    def test_case2_valid_medium_review_prediction_one(self):
        """Case 2: p=0.45, prediction=1, risk=MEDIUM, action=HUMAN_REVIEW -> valid."""
        reasons = check_prediction_context(1, 0.45, "MEDIUM", HUMAN_REVIEW)
        assert reasons == []

    def test_case2_valid_medium_review_prediction_zero(self):
        """p=0.45, prediction=0, risk=MEDIUM, action=HUMAN_REVIEW -> valid."""
        reasons = check_prediction_context(0, 0.45, "MEDIUM", HUMAN_REVIEW)
        assert reasons == []

    def test_case3_valid_high_reject(self):
        """Case 3: p=0.85, prediction=1, risk=HIGH, action=AUTO_REJECT -> valid."""
        reasons = check_prediction_context(1, 0.85, "HIGH", AUTO_REJECT)
        assert reasons == []

    def test_case4_prob_high_risk_low_mismatch(self):
        """Case 4: p=0.85, risk=LOW -> mismatch."""
        reasons = check_prediction_context(0, 0.85, "LOW", AUTO_APPROVE)
        assert PREDICTION_CONTEXT_MISMATCH in reasons
        reasons_hr = check_prediction_context(1, 0.85, "LOW", HUMAN_REVIEW)
        assert PREDICTION_CONTEXT_MISMATCH in reasons_hr

    def test_case5_prob_low_risk_high_mismatch(self):
        """Case 5: p=0.15, risk=HIGH -> mismatch."""
        reasons = check_prediction_context(1, 0.15, "HIGH", AUTO_REJECT)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_case6_prob_medium_risk_high_mismatch(self):
        """Case 6: p=0.45, risk=HIGH -> mismatch."""
        reasons = check_prediction_context(1, 0.45, "HIGH", AUTO_REJECT)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_case7_correct_risk_incorrect_recommendation_mismatch(self):
        """Case 7: Correct risk but incorrect recommendation -> mismatch."""
        # LOW risk must map to AUTO_APPROVE
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(0, 0.15, "LOW", HUMAN_REVIEW)
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(0, 0.15, "LOW", AUTO_REJECT)
        # MEDIUM risk must map to HUMAN_REVIEW
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(0, 0.45, "MEDIUM", AUTO_APPROVE)
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(1, 0.45, "MEDIUM", AUTO_REJECT)
        # HIGH risk must map to AUTO_REJECT
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(1, 0.85, "HIGH", HUMAN_REVIEW)
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(0, 0.85, "HIGH", AUTO_APPROVE)

    def test_case8_prediction_inconsistent_with_threshold_mismatch(self):
        """Case 8: Prediction inconsistent with binary classification threshold -> mismatch."""
        # p=0.15 with prediction=1 is inconsistent with threshold
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(1, 0.15, "LOW", AUTO_APPROVE)
        # p=0.85 with prediction=0 is inconsistent with threshold
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(0, 0.85, "HIGH", AUTO_REJECT)
        # When explicit classification_threshold is supplied
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(
            0, 0.45, "MEDIUM", HUMAN_REVIEW, classification_threshold=0.40
        )
        assert PREDICTION_CONTEXT_MISMATCH in check_prediction_context(
            1, 0.45, "MEDIUM", HUMAN_REVIEW, classification_threshold=0.50
        )
        assert [] == check_prediction_context(
            1, 0.45, "MEDIUM", HUMAN_REVIEW, classification_threshold=0.40
        )
        assert [] == check_prediction_context(
            0, 0.45, "MEDIUM", HUMAN_REVIEW, classification_threshold=0.50
        )

    def test_mismatch_approve_non_low(self):
        reasons = check_prediction_context(0, 0.5, "MEDIUM", AUTO_APPROVE)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_mismatch_reject_non_high(self):
        reasons = check_prediction_context(1, 0.5, "LOW", AUTO_REJECT)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_mismatch_approve_fraud_prediction(self):
        """AUTO_APPROVE with prediction=1 is a mismatch."""
        reasons = check_prediction_context(1, 0.1, "LOW", AUTO_APPROVE)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_mismatch_reject_legit_prediction(self):
        """AUTO_REJECT with prediction=0 is a mismatch."""
        reasons = check_prediction_context(0, 0.9, "HIGH", AUTO_REJECT)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_invalid_prediction(self):
        with pytest.raises(ValueError, match="Invalid prediction"):
            check_prediction_context(2, 0.5, "LOW", AUTO_APPROVE)

    def test_invalid_risk(self):
        with pytest.raises(ValueError, match="Invalid risk level"):
            check_prediction_context(0, 0.5, "EXTREME", AUTO_APPROVE)

    def test_invalid_action(self):
        with pytest.raises(ValueError, match="Invalid recommended action"):
            check_prediction_context(0, 0.5, "LOW", "DENY")

    def test_invalid_probability_above_one(self):
        with pytest.raises(ValueError, match="Invalid probability"):
            check_prediction_context(0, 1.5, "LOW", AUTO_APPROVE)

    def test_invalid_probability_negative(self):
        with pytest.raises(ValueError, match="Invalid probability"):
            check_prediction_context(0, -0.1, "LOW", AUTO_APPROVE)

    def test_no_raise_on_invalid_returns_mismatch(self):
        """When raise_on_invalid=False, invalid args return mismatch reason."""
        reasons = check_prediction_context(99, 0.5, "LOW", AUTO_APPROVE, raise_on_invalid=False)
        assert PREDICTION_CONTEXT_MISMATCH in reasons

    def test_context_mismatch_triggers_review(self):
        """Inconsistent prediction=0 + HIGH risk + AUTO_REJECT triggers review."""
        cfg = AdvancedGuardrailConfig(require_review_on_context_mismatch=True)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.9,
            risk_level="HIGH",
            recommended_action=AUTO_REJECT,
            config=cfg,
            ood_detector=None,
        )
        assert result.status == REQUIRES_REVIEW
        assert PREDICTION_CONTEXT_MISMATCH in result.reasons

    def test_context_mismatch_disabled(self):
        """When require_review_on_context_mismatch=False, no mismatch reason added."""
        cfg = AdvancedGuardrailConfig(require_review_on_context_mismatch=False)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.9,
            risk_level="HIGH",
            recommended_action=AUTO_REJECT,
            config=cfg,
            ood_detector=None,
        )
        assert PREDICTION_CONTEXT_MISMATCH not in result.reasons


# ============================================================
# D. Configuration Tests
# ============================================================

class TestAdvancedGuardrailConfig:
    """Test configuration validation."""

    def test_defaults(self):
        cfg = AdvancedGuardrailConfig()
        assert cfg.min_confidence == 0.80
        assert cfg.max_uncertainty == 0.20
        assert cfg.ood_distance_threshold == 20.0
        assert cfg.ood_enabled is True

    def test_min_confidence_below_range(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(min_confidence=0.3)

    def test_min_confidence_above_range(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(min_confidence=1.1)

    def test_min_confidence_bool_rejected(self):
        with pytest.raises(TypeError):
            AdvancedGuardrailConfig(min_confidence=True)

    def test_min_confidence_string_rejected(self):
        with pytest.raises(TypeError):
            AdvancedGuardrailConfig(min_confidence="0.8")

    def test_max_uncertainty_below_range(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(max_uncertainty=-0.1)

    def test_max_uncertainty_above_range(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(max_uncertainty=0.6)

    def test_ood_threshold_zero(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(ood_distance_threshold=0.0)

    def test_ood_threshold_negative(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(ood_distance_threshold=-1.0)

    def test_ood_threshold_inf(self):
        with pytest.raises(ValueError):
            AdvancedGuardrailConfig(ood_distance_threshold=float("inf"))

    def test_bool_flag_string_rejected(self):
        with pytest.raises(TypeError):
            AdvancedGuardrailConfig(ood_enabled="true")

    def test_frozen(self):
        cfg = AdvancedGuardrailConfig()
        with pytest.raises(AttributeError):
            cfg.min_confidence = 0.9  # type: ignore


# ============================================================
# E. Combine Guardrails Tests
# ============================================================

class TestCombineGuardrails:
    """Test combining Phase 5 and Phase 8 results."""

    def test_both_passed(self):
        p5 = GuardrailResult(status=PASSED, recommended_action=AUTO_APPROVE, reasons=())
        p8 = AdvancedGuardrailResult(
            status=PASSED, confidence=0.95, uncertainty=0.05,
            ood_status=IN_DISTRIBUTION, ood_distance=5.0, reasons=(),
        )
        combined = combine_guardrails(p5, p8)
        assert combined.status == PASSED
        assert combined.reasons == ()
        assert combined.recommended_action == AUTO_APPROVE

    def test_phase5_requires_review_phase8_passed(self):
        p5 = GuardrailResult(status=REQUIRES_REVIEW, recommended_action=AUTO_APPROVE, reasons=(NEGATIVE_AMOUNT,))
        p8 = AdvancedGuardrailResult(
            status=PASSED, confidence=0.95, uncertainty=0.05,
            ood_status=IN_DISTRIBUTION, ood_distance=5.0, reasons=(),
        )
        combined = combine_guardrails(p5, p8)
        assert combined.status == REQUIRES_REVIEW
        assert NEGATIVE_AMOUNT in combined.reasons

    def test_phase5_passed_phase8_requires_review(self):
        p5 = GuardrailResult(status=PASSED, recommended_action=AUTO_APPROVE, reasons=())
        p8 = AdvancedGuardrailResult(
            status=REQUIRES_REVIEW, confidence=0.6, uncertainty=0.4,
            ood_status=IN_DISTRIBUTION, ood_distance=5.0,
            reasons=(LOW_MODEL_CONFIDENCE,),
        )
        combined = combine_guardrails(p5, p8)
        assert combined.status == REQUIRES_REVIEW
        assert LOW_MODEL_CONFIDENCE in combined.reasons

    def test_both_require_review(self):
        p5 = GuardrailResult(status=REQUIRES_REVIEW, recommended_action=AUTO_APPROVE, reasons=(NEGATIVE_AMOUNT,))
        p8 = AdvancedGuardrailResult(
            status=REQUIRES_REVIEW, confidence=0.6, uncertainty=0.4,
            ood_status=OUT_OF_DISTRIBUTION, ood_distance=25.0,
            reasons=(LOW_MODEL_CONFIDENCE, OUT_OF_DISTRIBUTION_REASON),
        )
        combined = combine_guardrails(p5, p8)
        assert combined.status == REQUIRES_REVIEW
        assert NEGATIVE_AMOUNT in combined.reasons
        assert LOW_MODEL_CONFIDENCE in combined.reasons
        assert OUT_OF_DISTRIBUTION_REASON in combined.reasons

    def test_preserves_recommended_action(self):
        for action in [AUTO_APPROVE, HUMAN_REVIEW, AUTO_REJECT]:
            p5 = GuardrailResult(status=PASSED, recommended_action=action, reasons=())
            p8 = AdvancedGuardrailResult(
                status=PASSED, confidence=0.95, uncertainty=0.05,
                ood_status=IN_DISTRIBUTION, ood_distance=5.0, reasons=(),
            )
            combined = combine_guardrails(p5, p8)
            assert combined.recommended_action == action

    def test_no_duplicate_reasons(self):
        p5 = GuardrailResult(status=REQUIRES_REVIEW, recommended_action=AUTO_APPROVE, reasons=(LOW_MODEL_CONFIDENCE,))
        p8 = AdvancedGuardrailResult(
            status=REQUIRES_REVIEW, confidence=0.6, uncertainty=0.4,
            ood_status=IN_DISTRIBUTION, ood_distance=5.0,
            reasons=(LOW_MODEL_CONFIDENCE,),
        )
        combined = combine_guardrails(p5, p8)
        assert combined.reasons.count(LOW_MODEL_CONFIDENCE) == 1

    def test_phase5_mismatch_suppresses_phase8_mismatch(self):
        """If Phase 5 already has APPROVAL_CONTEXT_MISMATCH, Phase 8 PREDICTION_CONTEXT_MISMATCH is skipped."""
        p5 = GuardrailResult(
            status=REQUIRES_REVIEW, recommended_action=AUTO_APPROVE,
            reasons=(APPROVAL_CONTEXT_MISMATCH,),
        )
        p8 = AdvancedGuardrailResult(
            status=REQUIRES_REVIEW, confidence=0.9, uncertainty=0.1,
            ood_status=IN_DISTRIBUTION, ood_distance=5.0,
            reasons=(PREDICTION_CONTEXT_MISMATCH,),
        )
        combined = combine_guardrails(p5, p8)
        assert APPROVAL_CONTEXT_MISMATCH in combined.reasons
        assert PREDICTION_CONTEXT_MISMATCH not in combined.reasons

    def test_combining_preserves_all_phase5_and_phase8_reasons(self):
        """Phase 5 data-quality + Phase 8 confidence reasons are all preserved."""
        p5 = GuardrailResult(status=REQUIRES_REVIEW, recommended_action=AUTO_APPROVE, reasons=(NEGATIVE_AMOUNT,))
        cfg = AdvancedGuardrailConfig(min_confidence=0.95)
        p8 = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.6,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=None,
        )
        combined = combine_guardrails(p5, p8)
        assert combined.status == REQUIRES_REVIEW
        assert NEGATIVE_AMOUNT in combined.reasons
        # Phase 8 should have added exactly one of confidence/uncertainty
        p8_reasons_in_combined = [r for r in combined.reasons if r in (LOW_MODEL_CONFIDENCE, HIGH_MODEL_UNCERTAINTY)]
        assert len(p8_reasons_in_combined) == 1


# ============================================================
# F. Pipeline Integration Tests
# ============================================================

class TestPipelineIntegration:
    """Test that advanced guardrails integrate correctly with the pipeline."""

    def test_pipeline_exposes_advanced_fields(self, ood_detector, loaded_model):
        """Pipeline includes advanced fields when include_advanced=True."""
        from src.api.pipeline import run_pipeline

        state = SimpleNamespace(
            model=loaded_model,
            risk_config=RiskConfig(),
            decision_config=__import__("src.decision.engine", fromlist=["DecisionConfig"]).DecisionConfig(),
            guardrail_config=GuardrailConfig(),
            advanced_guardrail_config=AdvancedGuardrailConfig(min_confidence=0.80),
            ood_detector=ood_detector,
        )
        out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state, include_advanced=True)

        # Core fields present
        for key in ["prediction", "probability", "risk_level", "recommended_action",
                     "guardrail_status", "guardrail_reasons"]:
            assert key in out

        # Advanced fields present
        for key in ["advanced_guardrail_status", "confidence", "uncertainty",
                     "ood_status", "ood_distance", "advanced_guardrail_reasons"]:
            assert key in out

        # Legitimate transaction with default config should pass
        assert out["advanced_guardrail_status"] == PASSED
        assert out["guardrail_status"] == PASSED

    def test_pipeline_hides_advanced_by_default(self, loaded_model):
        """Pipeline excludes advanced fields by default."""
        from src.api.pipeline import run_pipeline

        state = SimpleNamespace(
            model=loaded_model,
            risk_config=RiskConfig(),
            decision_config=__import__("src.decision.engine", fromlist=["DecisionConfig"]).DecisionConfig(),
            guardrail_config=GuardrailConfig(),
            advanced_guardrail_config=AdvancedGuardrailConfig(),
            ood_detector=None,
        )
        out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state, include_advanced=False)

        assert "advanced_guardrail_status" not in out
        assert "confidence" not in out

    def test_pipeline_without_advanced_config(self, loaded_model):
        """Pipeline works when advanced_guardrail_config is not set."""
        from src.api.pipeline import run_pipeline

        state = SimpleNamespace(
            model=loaded_model,
            risk_config=RiskConfig(),
            decision_config=__import__("src.decision.engine", fromlist=["DecisionConfig"]).DecisionConfig(),
            guardrail_config=GuardrailConfig(),
        )
        out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state)
        assert "prediction" in out
        assert "guardrail_status" in out

    def test_pipeline_combined_guardrail_status(self, ood_detector, loaded_model):
        """Pipeline's guardrail_status reflects combined Phase 5 + Phase 8."""
        from src.api.pipeline import run_pipeline

        # Use extremely low OOD threshold to trigger Phase 8 review
        cfg = AdvancedGuardrailConfig(ood_distance_threshold=0.001, require_review_on_ood=True)
        state = SimpleNamespace(
            model=loaded_model,
            risk_config=RiskConfig(),
            decision_config=__import__("src.decision.engine", fromlist=["DecisionConfig"]).DecisionConfig(),
            guardrail_config=GuardrailConfig(),
            advanced_guardrail_config=cfg,
            ood_detector=ood_detector,
        )
        out = run_pipeline(dict(LEGITIMATE_TRANSACTION), state, include_advanced=True)
        # With threshold 0.001, virtually any real transaction triggers OOD
        assert out["guardrail_status"] == REQUIRES_REVIEW
        assert OUT_OF_DISTRIBUTION_REASON in out["guardrail_reasons"]


# ============================================================
# G. Evaluate Advanced Guardrails Edge Cases
# ============================================================

class TestEvaluateEdgeCases:
    """Edge cases for evaluate_advanced_guardrails."""

    def test_empty_transaction_raises(self):
        cfg = AdvancedGuardrailConfig()
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_advanced_guardrails(
                transaction={},
                prediction=0,
                probability=0.1,
                risk_level="LOW",
                recommended_action=AUTO_APPROVE,
                config=cfg,
            )

    def test_non_mapping_transaction_raises(self):
        cfg = AdvancedGuardrailConfig()
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_advanced_guardrails(
                transaction="not a dict",  # type: ignore
                prediction=0,
                probability=0.1,
                risk_level="LOW",
                recommended_action=AUTO_APPROVE,
                config=cfg,
            )

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError, match="AdvancedGuardrailConfig"):
            evaluate_advanced_guardrails(
                transaction=LEGITIMATE_TRANSACTION,
                prediction=0,
                probability=0.1,
                risk_level="LOW",
                recommended_action=AUTO_APPROVE,
                config="not a config",  # type: ignore
            )

    def test_invalid_prediction_raises(self):
        cfg = AdvancedGuardrailConfig()
        with pytest.raises(ValueError, match="Invalid prediction"):
            evaluate_advanced_guardrails(
                transaction=LEGITIMATE_TRANSACTION,
                prediction=5,
                probability=0.1,
                risk_level="LOW",
                recommended_action=AUTO_APPROVE,
                config=cfg,
            )

    def test_result_has_all_fields(self):
        cfg = AdvancedGuardrailConfig()
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.05,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=None,
        )
        assert hasattr(result, "status")
        assert hasattr(result, "confidence")
        assert hasattr(result, "uncertainty")
        assert hasattr(result, "ood_status")
        assert hasattr(result, "ood_distance")
        assert hasattr(result, "reasons")
        assert hasattr(result, "rules_evaluated")

    def test_multiple_reasons_preserved(self, ood_detector):
        """Multiple failure reasons should be preserved, not just the first."""
        cfg = AdvancedGuardrailConfig(
            min_confidence=0.99,
            ood_distance_threshold=0.001,
            require_review_on_low_confidence=True,
            require_review_on_ood=True,
        )
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=0,
            probability=0.6,
            risk_level="MEDIUM",
            recommended_action=HUMAN_REVIEW,
            config=cfg,
            ood_detector=ood_detector,
        )
        assert result.status == REQUIRES_REVIEW
        assert len(result.reasons) >= 2
        assert any(r in result.reasons for r in (LOW_MODEL_CONFIDENCE, HIGH_MODEL_UNCERTAINTY))
        assert OUT_OF_DISTRIBUTION_REASON in result.reasons

    def test_phase5_dup_suppression(self):
        """Phase 8 PREDICTION_CONTEXT_MISMATCH suppressed if Phase 5 already has specific mismatch."""
        cfg = AdvancedGuardrailConfig(require_review_on_context_mismatch=True)
        result = evaluate_advanced_guardrails(
            transaction=LEGITIMATE_TRANSACTION,
            prediction=1,
            probability=0.1,
            risk_level="LOW",
            recommended_action=AUTO_APPROVE,
            config=cfg,
            ood_detector=None,
            phase5_reasons=(APPROVAL_CONTEXT_MISMATCH,),
        )
        # Phase 5 already flagged, so Phase 8 should NOT add its own PREDICTION_CONTEXT_MISMATCH
        assert PREDICTION_CONTEXT_MISMATCH not in result.reasons


# ============================================================
# H. API Integration Tests
# ============================================================

class TestAPIIntegration:
    """Test API endpoint behavior with advanced guardrails."""

    def test_predict_default_excludes_advanced(self):
        """Default /predict does not include advanced fields."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.post("/predict", json=LEGITIMATE_TRANSACTION)
            assert resp.status_code == 200
            data = resp.json()
            # Core fields present
            assert "prediction" in data
            assert "guardrail_status" in data
            # Advanced fields absent (None values excluded by response_model_exclude_none)
            assert "advanced_guardrail_status" not in data

    def test_predict_include_advanced(self):
        """GET /predict?include_advanced=true includes advanced fields."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.post("/predict?include_advanced=true", json=LEGITIMATE_TRANSACTION)
            assert resp.status_code == 200
            data = resp.json()
            assert "advanced_guardrail_status" in data
            assert "confidence" in data
            assert "uncertainty" in data
            assert "ood_status" in data
            assert "ood_distance" in data
            assert "advanced_guardrail_reasons" in data

    def test_explain_unaffected(self):
        """/explain still works with Phase 8 enabled."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.post("/explain", json=LEGITIMATE_TRANSACTION)
            assert resp.status_code == 200
            data = resp.json()
            assert "prediction" in data
            assert "top_features" in data

    def test_reviews_unaffected(self):
        """/reviews endpoints still work."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        with TestClient(app) as client:
            resp = client.get("/reviews/pending")
            assert resp.status_code == 200
