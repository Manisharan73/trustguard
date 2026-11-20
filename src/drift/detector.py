"""Statistical drift detector using Population Stability Index (PSI).

Complements Phase 8 OOD detection:
- Phase 8 OOD checks if an INDIVIDUAL transaction is an outlier.
- Drift detection checks whether a BATCH or TIME WINDOW of transactions
  has shifted systematically compared to baseline training data.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.drift.models import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_MODERATE_DRIFT,
    STATUS_NO_DRIFT,
    STATUS_SIGNIFICANT_DRIFT,
    DriftReport,
)
from src.ml.preprocess import FEATURE_COLUMNS

DEFAULT_REFERENCE_PATH = Path("data/processed/drift_reference.json")

# Industry standard PSI thresholds
PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25


def calculate_psi(
    actual: np.ndarray,
    reference_edges: list[float],
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) against reference decile bin edges.

    PSI = sum((A_i - E_i) * ln(A_i / E_i))

    Bins:
    - Bins are defined by 11 quantile edges from the reference distribution (10 bins).
    - Expected frequency per bin: 0.10 (10%).
    - Epsilon is added to avoid division by zero or ln(0).
    """
    n = len(actual)
    if n == 0:
        return 0.0

    valid = actual[np.isfinite(actual)]
    if len(valid) == 0:
        return 1.0  # Entirely invalid/inf implies extreme shift

    # Bin the actual samples
    edges = np.array(reference_edges, dtype=float)
    # Ensure inner edges are monotonically non-decreasing
    edges = np.sort(edges)

    num_bins = len(edges) - 1
    expected_pct = 1.0 / num_bins

    # Count samples in each bin
    counts = np.zeros(num_bins, dtype=float)
    for val in valid:
        # find bin index
        idx = np.searchsorted(edges[1:-1], val)
        counts[idx] += 1

    actual_pcts = counts / len(valid)

    # Smooth zero bins
    actual_pcts = np.maximum(actual_pcts, epsilon)
    actual_pcts = actual_pcts / np.sum(actual_pcts)

    exp_pcts = np.full(num_bins, expected_pct)

    psi_val = np.sum((actual_pcts - exp_pcts) * np.log(actual_pcts / exp_pcts))
    return float(max(0.0, psi_val))


class DriftDetector:
    """Evaluates data drift, behavioral drift, and concept drift."""

    def __init__(self, reference_data: dict[str, Any]):
        self.reference_data = reference_data
        self.feature_quantiles = reference_data.get("feature_quantiles", {})
        self.probability_quantiles = reference_data.get("probability_quantiles", [])
        self.ref_sample_count = reference_data.get("sample_count", 227845)
        self.ref_fraud_rate = reference_data.get("predicted_fraud_rate", 0.0017)
        self.ref_risks = reference_data.get("risk_proportions", {"LOW": 0.98, "MEDIUM": 0.015, "HIGH": 0.005})
        self.ref_actions = reference_data.get("action_proportions", {"AUTO_APPROVE": 0.98, "HUMAN_REVIEW": 0.015, "AUTO_REJECT": 0.005})

    @classmethod
    def from_reference_file(cls, path: Path | str = DEFAULT_REFERENCE_PATH) -> "DriftDetector":
        path = Path(path)
        if not path.exists():
            from src.ml.preprocess import get_train_test_data
            from src.ml.predict import load_model
            from src.risk.engine import RiskConfig, assess_risk
            from src.decision.engine import DecisionConfig, recommend_action

            X_train, _, y_train, _ = get_train_test_data()
            model = load_model()
            probs = model.predict_proba(X_train)[:, 1]
            preds = model.predict(X_train)
            risk_cfg = RiskConfig()
            decision_cfg = DecisionConfig()
            risks = [assess_risk(p, risk_cfg) for p in probs]
            actions = [recommend_action(r, decision_cfg) for r in risks]

            ref = {
                "sample_count": len(X_train),
                "fraud_rate": float(np.mean(y_train)),
                "predicted_fraud_rate": float(np.mean(preds)),
                "feature_quantiles": {},
                "probability_quantiles": [float(x) for x in np.percentile(probs, np.linspace(0, 100, 11))],
                "risk_proportions": {
                    "LOW": float(np.mean([r == "LOW" for r in risks])),
                    "MEDIUM": float(np.mean([r == "MEDIUM" for r in risks])),
                    "HIGH": float(np.mean([r == "HIGH" for r in risks])),
                },
                "action_proportions": {
                    "AUTO_APPROVE": float(np.mean([a == "AUTO_APPROVE" for a in actions])),
                    "HUMAN_REVIEW": float(np.mean([a == "HUMAN_REVIEW" for a in actions])),
                    "AUTO_REJECT": float(np.mean([a == "AUTO_REJECT" for a in actions])),
                },
            }
            for col in FEATURE_COLUMNS:
                vals = X_train[col].to_numpy(dtype=float)
                ref["feature_quantiles"][col] = [float(x) for x in np.percentile(vals, np.linspace(0, 100, 11))]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(ref, f, indent=2)

        with open(path) as f:
            data = json.load(f)
        return cls(data)

    def evaluate(
        self,
        *,
        transactions: list[dict[str, Any]],
        predictions: list[int] | None = None,
        probabilities: list[float] | None = None,
        risk_levels: list[str] | None = None,
        recommended_actions: list[str] | None = None,
        guardrail_statuses: list[str] | None = None,
        ground_truths: list[int | None] | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
        analysis_id: str = "",
        created_at: str = "",
        min_samples: int = 20,
    ) -> DriftReport:
        """Run statistical drift detection across data, model behavior, and concept."""
        n = len(transactions)

        thresholds = {
            "psi_moderate": PSI_MODERATE_THRESHOLD,
            "psi_significant": PSI_SIGNIFICANT_THRESHOLD,
            "min_sample_size": float(min_samples),
        }

        limitations = [
            "PSI assumes independent and identically distributed observations within the analyzed window.",
            "Decile binning approximates continuous distributions; small sample sizes (< 30) may yield noisy PSI.",
            "Concept drift requires verified ground-truth labels and cannot be confirmed from model predictions alone.",
        ]

        if n < min_samples:
            return DriftReport(
                analysis_id=analysis_id,
                created_at=created_at,
                window_start=window_start,
                window_end=window_end,
                sample_count=n,
                reference_sample_count=self.ref_sample_count,
                overall_status=STATUS_INSUFFICIENT_DATA,
                drift_detected=False,
                mean_feature_psi=0.0,
                max_feature_psi=0.0,
                drifted_features_count=0,
                feature_metrics={},
                behavioral_metrics={"warning": f"Insufficient sample size ({n} < {min_samples})."},
                concept_drift={"performance_drift_confirmed": False, "message": "Insufficient sample size."},
                method="Population Stability Index (PSI)",
                thresholds=thresholds,
                limitations=limitations,
            )

        # 1. Feature Data Drift
        feature_metrics: dict[str, dict[str, Any]] = {}
        psis: list[float] = []
        drifted_count = 0

        for col in FEATURE_COLUMNS:
            ref_edges = self.feature_quantiles.get(col)
            if not ref_edges:
                continue

            vals = []
            for tx in transactions:
                v = tx.get(col)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    vals.append(float(v))
                else:
                    vals.append(float("nan"))

            arr = np.array(vals, dtype=float)
            psi = calculate_psi(arr, ref_edges)
            psis.append(psi)

            if psi >= PSI_SIGNIFICANT_THRESHOLD:
                status = STATUS_SIGNIFICANT_DRIFT
                drifted_count += 1
            elif psi >= PSI_MODERATE_THRESHOLD:
                status = STATUS_MODERATE_DRIFT
            else:
                status = STATUS_NO_DRIFT

            feature_metrics[col] = {
                "psi": round(psi, 4),
                "status": status,
            }

        mean_psi = float(np.mean(psis)) if psis else 0.0
        max_psi = float(np.max(psis)) if psis else 0.0

        if max_psi >= PSI_SIGNIFICANT_THRESHOLD:
            overall_status = STATUS_SIGNIFICANT_DRIFT
            drift_detected = True
        elif max_psi >= PSI_MODERATE_THRESHOLD or mean_psi >= PSI_MODERATE_THRESHOLD:
            overall_status = STATUS_MODERATE_DRIFT
            drift_detected = False
        else:
            overall_status = STATUS_NO_DRIFT
            drift_detected = False

        # 2. Behavioral Drift
        behavioral: dict[str, Any] = {}
        if predictions:
            pred_fraud_rate = float(np.mean([p == 1 for p in predictions]))
            behavioral["predicted_fraud_rate"] = round(pred_fraud_rate, 4)
            behavioral["baseline_fraud_rate"] = round(self.ref_fraud_rate, 4)
            behavioral["fraud_rate_shift"] = round(pred_fraud_rate - self.ref_fraud_rate, 4)

        if probabilities and self.probability_quantiles:
            prob_arr = np.array([p for p in probabilities if math.isfinite(p)], dtype=float)
            prob_psi = calculate_psi(prob_arr, self.probability_quantiles)
            behavioral["probability_psi"] = round(prob_psi, 4)
            behavioral["probability_drift_status"] = (
                STATUS_SIGNIFICANT_DRIFT if prob_psi >= PSI_SIGNIFICANT_THRESHOLD
                else STATUS_MODERATE_DRIFT if prob_psi >= PSI_MODERATE_THRESHOLD
                else STATUS_NO_DRIFT
            )

        if risk_levels:
            r_counts = {k: 0 for k in ("LOW", "MEDIUM", "HIGH")}
            for r in risk_levels:
                if r in r_counts:
                    r_counts[r] += 1
            behavioral["risk_distribution"] = {k: round(v / n, 4) for k, v in r_counts.items()}
            behavioral["baseline_risk_distribution"] = self.ref_risks

        if recommended_actions:
            a_counts = {k: 0 for k in ("AUTO_APPROVE", "HUMAN_REVIEW", "AUTO_REJECT")}
            for a in recommended_actions:
                if a in a_counts:
                    a_counts[a] += 1
            behavioral["action_distribution"] = {k: round(v / n, 4) for k, v in a_counts.items()}
            behavioral["baseline_action_distribution"] = self.ref_actions

        if guardrail_statuses:
            esc_rate = float(np.mean([s == "REQUIRES_REVIEW" for s in guardrail_statuses]))
            behavioral["guardrail_escalation_rate"] = round(esc_rate, 4)

        # 3. Concept Drift / Performance Degradation
        concept_drift: dict[str, Any] = {}
        labeled_pairs = [
            (p, gt) for p, gt in zip(predictions or [], ground_truths or [])
            if gt is not None and gt in (0, 1)
        ]

        if len(labeled_pairs) >= 10:
            tp = sum(1 for p, gt in labeled_pairs if p == 1 and gt == 1)
            fp = sum(1 for p, gt in labeled_pairs if p == 1 and gt == 0)
            fn = sum(1 for p, gt in labeled_pairs if p == 0 and gt == 1)
            tn = sum(1 for p, gt in labeled_pairs if p == 0 and gt == 0)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            concept_drift["performance_drift_confirmed"] = True
            concept_drift["labeled_sample_count"] = len(labeled_pairs)
            concept_drift["current_f1"] = round(f1, 4)
            concept_drift["current_precision"] = round(prec, 4)
            concept_drift["current_recall"] = round(rec, 4)
            concept_drift["baseline_f1"] = 0.1144  # Baseline reference metric
            concept_drift["f1_change"] = round(f1 - 0.1144, 4)
        else:
            concept_drift["performance_drift_confirmed"] = False
            concept_drift["message"] = (
                f"Insufficient ground-truth labels ({len(labeled_pairs)} samples). "
                "Concept drift cannot yet be confirmed; monitor data and behavioral drift indicators."
            )

        return DriftReport(
            analysis_id=analysis_id,
            created_at=created_at,
            window_start=window_start,
            window_end=window_end,
            sample_count=n,
            reference_sample_count=self.ref_sample_count,
            overall_status=overall_status,
            drift_detected=drift_detected,
            mean_feature_psi=round(mean_psi, 4),
            max_feature_psi=round(max_psi, 4),
            drifted_features_count=drifted_count,
            feature_metrics=feature_metrics,
            behavioral_metrics=behavioral,
            concept_drift=concept_drift,
            method="Population Stability Index (PSI)",
            thresholds=thresholds,
            limitations=limitations,
        )
