"""Advanced Guardrails (Phase 8): reliability, uncertainty, OOD, and context consistency.

Responsibility: evaluate whether the ML model's prediction and the Decision Engine's
recommendation are safe to proceed with, or whether human review is required.

Guiding principles:
- Never modifies the prediction, fraud probability, risk level, or recommended action.
- Never converts a Phase 5 REQUIRES_REVIEW into PASSED.
- Multiple failure reasons are preserved.
- Out-of-Distribution (OOD) uses standardized Euclidean distance from training statistics only.
"""

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.decision.engine import (
    AUTO_APPROVE,
    AUTO_REJECT,
    HUMAN_REVIEW,
    RISK_LEVELS,
    VALID_ACTIONS,
)
from src.guardrails.engine import (
    APPROVAL_CONTEXT_MISMATCH,
    PASSED,
    REJECTION_CONTEXT_MISMATCH,
    REQUIRES_REVIEW,
    GuardrailResult,
)
from src.ml.preprocess import FEATURE_COLUMNS

# ---------- Status constants ----------
IN_DISTRIBUTION = "IN_DISTRIBUTION"
OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
VALID_OOD_STATUSES = (IN_DISTRIBUTION, OUT_OF_DISTRIBUTION)

# ---------- Reason codes ----------
LOW_MODEL_CONFIDENCE = "LOW_MODEL_CONFIDENCE"
HIGH_MODEL_UNCERTAINTY = "HIGH_MODEL_UNCERTAINTY"
OUT_OF_DISTRIBUTION_REASON = "OUT_OF_DISTRIBUTION"
PREDICTION_CONTEXT_MISMATCH = "PREDICTION_CONTEXT_MISMATCH"

VALID_REASONS = (
    LOW_MODEL_CONFIDENCE,
    HIGH_MODEL_UNCERTAINTY,
    OUT_OF_DISTRIBUTION_REASON,
    PREDICTION_CONTEXT_MISMATCH,
)

DEFAULT_OOD_PATH = Path("data/processed/ood_reference.npz")


@dataclass(frozen=True)
class AdvancedGuardrailConfig:
    """Configuration for advanced guardrails.

    min_confidence:                     Minimum acceptable confidence max(p, 1-p).
    max_uncertainty:                    Maximum acceptable uncertainty (1 - confidence).
    ood_distance_threshold:             Standardized Euclidean distance threshold for OOD.
    ood_enabled:                        Whether OOD detection is evaluated.
    require_review_on_low_confidence:   Whether low confidence requires human review.
    require_review_on_ood:              Whether OOD requires human review.
    require_review_on_context_mismatch: Whether prediction-context mismatch requires review.
    expose_in_predict:                  Whether /predict includes Phase 8 fields by default.
    """

    min_confidence: float = 0.80
    max_uncertainty: float = 0.20
    ood_distance_threshold: float = 20.0
    ood_enabled: bool = True
    require_review_on_low_confidence: bool = True
    require_review_on_ood: bool = True
    require_review_on_context_mismatch: bool = True
    expose_in_predict: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.min_confidence, bool) or not isinstance(self.min_confidence, numbers.Real):
            raise TypeError(f"min_confidence must be a float, got {type(self.min_confidence).__name__}.")
        if not math.isfinite(self.min_confidence) or not (0.5 <= self.min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be between 0.5 and 1.0, got {self.min_confidence}.")

        if isinstance(self.max_uncertainty, bool) or not isinstance(self.max_uncertainty, numbers.Real):
            raise TypeError(f"max_uncertainty must be a float, got {type(self.max_uncertainty).__name__}.")
        if not math.isfinite(self.max_uncertainty) or not (0.0 <= self.max_uncertainty <= 0.5):
            raise ValueError(f"max_uncertainty must be between 0.0 and 0.5, got {self.max_uncertainty}.")

        if isinstance(self.ood_distance_threshold, bool) or not isinstance(self.ood_distance_threshold, numbers.Real):
            raise TypeError(f"ood_distance_threshold must be a float, got {type(self.ood_distance_threshold).__name__}.")
        if not math.isfinite(self.ood_distance_threshold) or self.ood_distance_threshold <= 0:
            raise ValueError(f"ood_distance_threshold must be positive, got {self.ood_distance_threshold}.")

        for flag in (
            "ood_enabled",
            "require_review_on_low_confidence",
            "require_review_on_ood",
            "require_review_on_context_mismatch",
            "expose_in_predict",
        ):
            val = getattr(self, flag)
            if not isinstance(val, bool):
                raise TypeError(f"{flag} must be a boolean, got {type(val).__name__}.")


@dataclass(frozen=True)
class AdvancedGuardrailResult:
    """Structured outcome of advanced guardrail evaluation.

    status:          "PASSED" or "REQUIRES_REVIEW".
    confidence:      max(p, 1-p).
    uncertainty:     1 - confidence.
    ood_status:      "IN_DISTRIBUTION" or "OUT_OF_DISTRIBUTION".
    ood_distance:    standardized Euclidean distance from reference distribution.
    reasons:         tuple of machine-readable reason codes (empty when PASSED).
    rules_evaluated: dictionary of evaluated rules and details.
    """

    status: str
    confidence: float
    uncertainty: float
    ood_status: str
    ood_distance: float
    reasons: tuple[str, ...] = ()
    rules_evaluated: dict[str, Any] = field(default_factory=dict)


def calculate_confidence_and_uncertainty(probability: float) -> tuple[float, float]:
    """Calculate confidence and uncertainty for binary classification.

    confidence = max(probability, 1 - probability)
    uncertainty = 1 - confidence
    """
    if isinstance(probability, bool) or not isinstance(probability, numbers.Real):
        raise TypeError(f"probability must be a real number, got {type(probability).__name__}.")
    if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be between 0.0 and 1.0, got {probability}.")

    prob = float(probability)
    confidence = max(prob, 1.0 - prob)
    uncertainty = 1.0 - confidence
    return confidence, uncertainty


class OODDetector:
    """Out-of-distribution detector using standardized Euclidean distance.

    Calculates the distance of incoming preprocessed transactions from the
    training data reference distribution:
        d(x) = sqrt(sum(((x_i - mean_i) / std_i)^2))

    Features with zero variance (std < 1e-12) are handled safely by setting
    their normalized contribution to zero.
    """

    def __init__(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        threshold: float = 20.0,
        feature_names: list[str] | None = None,
        preprocessor: Any = None,
    ):
        self.means = np.asarray(means, dtype=float)
        self.stds = np.asarray(stds, dtype=float)
        if self.means.ndim != 1 or self.stds.ndim != 1:
            raise ValueError("means and stds must be 1-dimensional arrays.")
        if len(self.means) != len(self.stds):
            raise ValueError(f"means and stds length mismatch: {len(self.means)} vs {len(self.stds)}.")

        self.threshold = float(threshold)
        self.feature_names = feature_names or list(FEATURE_COLUMNS)
        self.preprocessor = preprocessor

    @classmethod
    def from_reference_file(
        cls,
        path: Path | str = DEFAULT_OOD_PATH,
        threshold: float = 20.0,
        preprocessor: Any = None,
    ) -> "OODDetector":
        """Load reference statistics from a saved .npz file, or compute if missing."""
        path = Path(path)
        if path.exists():
            data = np.load(path)
            means = data["means"]
            stds = data["stds"]
            feature_names = list(data["feature_names"]) if "feature_names" in data else list(FEATURE_COLUMNS)
            return cls(means=means, stds=stds, threshold=threshold, feature_names=feature_names, preprocessor=preprocessor)

        if preprocessor is not None:
            try:
                from src.ml.preprocess import get_train_test_data

                X_train, _, _, _ = get_train_test_data()
                X_train_proc = preprocessor.transform(X_train)
                means = np.mean(X_train_proc, axis=0)
                stds = np.std(X_train_proc, axis=0)
                feature_names = [name.split("__")[-1] for name in preprocessor.get_feature_names_out()]
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, means=means, stds=stds, feature_names=feature_names)
                return cls(means=means, stds=stds, threshold=threshold, feature_names=feature_names, preprocessor=preprocessor)
            except Exception:
                pass

        # Fallback to standard unit normal reference
        n_features = len(FEATURE_COLUMNS)
        return cls(means=np.zeros(n_features), stds=np.ones(n_features), threshold=threshold, preprocessor=preprocessor)

    def compute_distance(self, transaction: Mapping | np.ndarray | pd.DataFrame) -> float:
        """Compute standardized Euclidean distance for a transaction."""
        if isinstance(transaction, Mapping):
            # Check all required features are present
            missing = [c for c in FEATURE_COLUMNS if c not in transaction]
            if missing:
                raise ValueError(f"Transaction missing required features: {missing}.")

            # Check for non-finite values
            for c in FEATURE_COLUMNS:
                val = transaction[c]
                if isinstance(val, bool) or not isinstance(val, numbers.Real) or not math.isfinite(val):
                    return float("inf")

            if self.preprocessor is not None:
                df = pd.DataFrame([transaction])[FEATURE_COLUMNS]
                x = self.preprocessor.transform(df)[0]
            else:
                x = np.array([transaction[c] for c in FEATURE_COLUMNS], dtype=float)
        elif isinstance(transaction, np.ndarray):
            x = np.asarray(transaction, dtype=float).ravel()
            if len(x) != len(self.means):
                raise ValueError(f"Input feature length {len(x)} does not match reference {len(self.means)}.")
            if not np.all(np.isfinite(x)):
                return float("inf")
        elif isinstance(transaction, pd.DataFrame):
            if self.preprocessor is not None:
                x = self.preprocessor.transform(transaction[FEATURE_COLUMNS])[0]
            else:
                x = transaction[FEATURE_COLUMNS].to_numpy(dtype=float)[0]
        else:
            raise TypeError(f"Unsupported transaction type: {type(transaction).__name__}.")

        zero_var = self.stds < 1e-12
        z = np.zeros_like(x, dtype=float)
        valid = ~zero_var
        z[valid] = (x[valid] - self.means[valid]) / self.stds[valid]
        z[zero_var] = 0.0

        return float(np.sqrt(np.sum(z**2)))

    def detect(self, transaction: Mapping | np.ndarray | pd.DataFrame, threshold: float | None = None) -> tuple[str, float]:
        """Classify transaction as IN_DISTRIBUTION or OUT_OF_DISTRIBUTION."""
        thresh = self.threshold if threshold is None else float(threshold)
        dist = self.compute_distance(transaction)
        if not math.isfinite(dist) or dist > thresh:
            return OUT_OF_DISTRIBUTION, dist
        return IN_DISTRIBUTION, dist


def validate_prediction_context(prediction: Any, probability: Any, risk_level: Any, recommended_action: Any) -> None:
    """Validate internal arguments for prediction context consistency.

    Raises TypeError or ValueError on invalid argument contracts.
    """
    if isinstance(prediction, bool) or not isinstance(prediction, numbers.Integral) or prediction not in (0, 1):
        raise ValueError(f"Invalid prediction: {prediction!r}. Expected 0 or 1.")

    if isinstance(probability, bool) or not isinstance(probability, numbers.Real):
        raise TypeError(f"probability must be a real number, got {type(probability).__name__}.")
    if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
        raise ValueError(f"Invalid probability: {probability!r}. Expected 0 to 1.")

    if not isinstance(risk_level, str) or risk_level not in RISK_LEVELS:
        raise ValueError(f"Invalid risk level: {risk_level!r}. Expected one of: {list(RISK_LEVELS)}.")

    if not isinstance(recommended_action, str) or recommended_action not in VALID_ACTIONS:
        raise ValueError(f"Invalid recommended action: {recommended_action!r}. Expected one of: {list(VALID_ACTIONS)}.")


def check_prediction_context(
    prediction: Any,
    probability: Any,
    risk_level: Any,
    recommended_action: Any,
    raise_on_invalid: bool = True,
    low_threshold: float = 0.30,
    high_threshold: float = 0.70,
    classification_threshold: float | None = None,
) -> list[str]:
    """Check consistency across prediction, probability, risk level, and recommended action.

    Expected mappings (from the existing Risk Engine and Decision Engine):
      0.00 <= p < low_threshold (0.30)           -> LOW    -> AUTO_APPROVE
      low_threshold <= p < high_threshold (0.70) -> MEDIUM -> HUMAN_REVIEW
      high_threshold <= p <= 1.00                -> HIGH   -> AUTO_REJECT

    Additionally validates:
      - AUTO_APPROVE requires prediction == 0 and risk == "LOW"
      - AUTO_REJECT requires prediction == 1 and risk == "HIGH"
      - HUMAN_REVIEW requires risk == "MEDIUM" (and permits prediction 0 or 1)
      - Probability must match the risk level:
          p < low_threshold requires risk == "LOW"
          low_threshold <= p < high_threshold requires risk == "MEDIUM"
          p >= high_threshold requires risk == "HIGH"
      - Prediction must not contradict classification threshold:
          If classification_threshold is supplied, prediction == 1 iff p >= classification_threshold
          Otherwise, prediction cannot be 1 when p < low_threshold, and cannot be 0 when p >= high_threshold
    """
    if raise_on_invalid:
        validate_prediction_context(prediction, probability, risk_level, recommended_action)
    else:
        try:
            validate_prediction_context(prediction, probability, risk_level, recommended_action)
        except (ValueError, TypeError):
            return [PREDICTION_CONTEXT_MISMATCH]

    # 1. Risk Engine mapping consistency: probability -> risk_level
    if probability < low_threshold:
        expected_risk = "LOW"
    elif probability < high_threshold:
        expected_risk = "MEDIUM"
    else:
        expected_risk = "HIGH"

    if risk_level != expected_risk:
        return [PREDICTION_CONTEXT_MISMATCH]

    # 2. Decision Engine mapping consistency: risk_level -> recommended_action
    expected_actions = {
        "LOW": AUTO_APPROVE,
        "MEDIUM": HUMAN_REVIEW,
        "HIGH": AUTO_REJECT,
    }
    if recommended_action != expected_actions.get(risk_level):
        return [PREDICTION_CONTEXT_MISMATCH]

    # 3. Prediction consistency with recommendation and binary classification threshold
    if recommended_action == AUTO_APPROVE and prediction != 0:
        return [PREDICTION_CONTEXT_MISMATCH]
    if recommended_action == AUTO_REJECT and prediction != 1:
        return [PREDICTION_CONTEXT_MISMATCH]

    if classification_threshold is not None:
        expected_pred = 1 if probability >= classification_threshold else 0
        if prediction != expected_pred:
            return [PREDICTION_CONTEXT_MISMATCH]
    else:
        if probability < low_threshold and prediction != 0:
            return [PREDICTION_CONTEXT_MISMATCH]
        if probability >= high_threshold and prediction != 1:
            return [PREDICTION_CONTEXT_MISMATCH]

    return []


def evaluate_advanced_guardrails(
    transaction: Mapping,
    prediction: int,
    probability: float,
    risk_level: str,
    recommended_action: str,
    config: AdvancedGuardrailConfig,
    ood_detector: OODDetector | None = None,
    phase5_reasons: tuple[str, ...] | list[str] | None = None,
    risk_config: Any = None,
) -> AdvancedGuardrailResult:
    """Evaluate advanced guardrails: confidence, uncertainty, OOD, and context consistency.

    Returns an AdvancedGuardrailResult with status PASSED or REQUIRES_REVIEW and
    machine-readable reason codes.
    """
    if not isinstance(config, AdvancedGuardrailConfig):
        raise TypeError(f"config must be an AdvancedGuardrailConfig instance, got {type(config).__name__}.")
    validate_prediction_context(prediction, probability, risk_level, recommended_action)

    if not isinstance(transaction, Mapping) or len(transaction) == 0:
        raise ValueError("transaction must be a non-empty mapping.")

    reasons: list[str] = []
    rules_evaluated: dict[str, Any] = {}

    # Rule A: Confidence and Uncertainty
    confidence, uncertainty = calculate_confidence_and_uncertainty(probability)
    rules_evaluated["confidence"] = confidence
    rules_evaluated["uncertainty"] = uncertainty

    if config.require_review_on_low_confidence:
        if confidence < config.min_confidence:
            reasons.append(LOW_MODEL_CONFIDENCE)
        # Only add HIGH_MODEL_UNCERTAINTY if LOW_MODEL_CONFIDENCE wasn't already added
        # (they are complementary: confidence < threshold <==> uncertainty > 1 - threshold)
        elif uncertainty > config.max_uncertainty:
            reasons.append(HIGH_MODEL_UNCERTAINTY)

    # Rule B: Out-of-Distribution Detection
    if config.ood_enabled and ood_detector is not None:
        ood_status, ood_distance = ood_detector.detect(transaction, threshold=config.ood_distance_threshold)
        rules_evaluated["ood_status"] = ood_status
        rules_evaluated["ood_distance"] = ood_distance
        if ood_status == OUT_OF_DISTRIBUTION and config.require_review_on_ood:
            reasons.append(OUT_OF_DISTRIBUTION_REASON)
    else:
        ood_status = IN_DISTRIBUTION
        ood_distance = 0.0
        rules_evaluated["ood_status"] = ood_status
        rules_evaluated["ood_distance"] = ood_distance

    # Rule C: Prediction-Context Consistency
    low_thresh = getattr(risk_config, "low_threshold", 0.30)
    high_thresh = getattr(risk_config, "high_threshold", 0.70)
    ctx_reasons = check_prediction_context(
        prediction,
        probability,
        risk_level,
        recommended_action,
        low_threshold=low_thresh,
        high_threshold=high_thresh,
    )
    p5 = phase5_reasons or ()
    # If Phase 5 already flagged context mismatch, avoid redundant duplicate reason
    if PREDICTION_CONTEXT_MISMATCH in ctx_reasons and (
        APPROVAL_CONTEXT_MISMATCH in p5 or REJECTION_CONTEXT_MISMATCH in p5
    ):
        ctx_reasons = [r for r in ctx_reasons if r != PREDICTION_CONTEXT_MISMATCH]

    if ctx_reasons and config.require_review_on_context_mismatch:
        reasons.extend(ctx_reasons)
    rules_evaluated["context_consistency"] = "PASSED" if not ctx_reasons else "FAILED"

    status = REQUIRES_REVIEW if reasons else PASSED

    return AdvancedGuardrailResult(
        status=status,
        confidence=confidence,
        uncertainty=uncertainty,
        ood_status=ood_status,
        ood_distance=ood_distance,
        reasons=tuple(reasons),
        rules_evaluated=rules_evaluated,
    )


def combine_guardrails(
    phase5_result: GuardrailResult,
    phase8_result: AdvancedGuardrailResult,
) -> GuardrailResult:
    """Combine Phase 5 and Phase 8 guardrail evaluations.

    Invariants:
    - Status is REQUIRES_REVIEW if either Phase 5 or Phase 8 requires review.
    - Never overrides a Phase 5 failure with a Phase 8 pass.
    - Never converts REQUIRES_REVIEW into PASSED.
    - Preserves all Phase 5 reason codes, then appends Phase 8 reasons without duplicates.
    - Avoids redundant context-mismatch duplication if Phase 5 already flagged approval/rejection mismatch.
    - Preserves recommended_action unchanged.
    """
    status = (
        REQUIRES_REVIEW
        if (phase5_result.status == REQUIRES_REVIEW or phase8_result.status == REQUIRES_REVIEW)
        else PASSED
    )

    combined_reasons = list(phase5_result.reasons)
    for r in phase8_result.reasons:
        if r not in combined_reasons:
            if r == PREDICTION_CONTEXT_MISMATCH and (
                APPROVAL_CONTEXT_MISMATCH in combined_reasons or REJECTION_CONTEXT_MISMATCH in combined_reasons
            ):
                continue
            combined_reasons.append(r)

    return GuardrailResult(
        status=status,
        recommended_action=phase5_result.recommended_action,
        reasons=tuple(combined_reasons),
    )
