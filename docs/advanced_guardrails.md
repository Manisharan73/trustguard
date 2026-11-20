# Phase 8: Advanced Guardrails

## Architecture

Phase 8 adds an **Advanced Guardrails** layer after the existing Phase 5 (Guardrails v1) in the TrustGuard pipeline:

```
Transaction Input
→ ML Prediction (Phase 2)
→ Risk Assessment (Phase 3)
→ Decision Engine (Phase 4)
→ Guardrails v1 (Phase 5)
→ Advanced Guardrails (Phase 8)     ← NEW
→ Human-in-the-Loop (Phase 6)
→ Explainability (Phase 7)
```

Advanced Guardrails evaluate whether a prediction is **sufficiently reliable** for automated action. They never modify the prediction, probability, risk level, or recommended action — they only determine whether the recommendation is safe to proceed or requires human review.

**Module:** [`src/guardrails/advanced.py`](file:///home/manisharan/Downloads/TrustGuard/src/guardrails/advanced.py)

---

## Guardrail Components

### A. Confidence and Uncertainty Guard

Evaluates model confidence using the existing fraud probability.

**Formulas:**

$$\text{confidence} = \max(p, 1 - p)$$

$$\text{uncertainty} = 1 - \text{confidence}$$

Where \(p\) is the fraud probability from the ML model.

| Configuration | Default | Range |
|---|---|---|
| `min_confidence` | 0.80 | [0.5, 1.0] |
| `max_uncertainty` | 0.20 | [0.0, 0.5] |

**Trigger:** If `confidence < min_confidence`, adds `LOW_MODEL_CONFIDENCE`. If only `uncertainty > max_uncertainty` (and confidence threshold is not violated), adds `HIGH_MODEL_UNCERTAINTY`. At most one of these two reasons is added per evaluation to avoid redundancy.

> [!IMPORTANT]
> High confidence does not mean the prediction is correct. It means the model is decisive about its output. A confident model can still be wrong.

### B. Out-of-Distribution (OOD) Guard

Detects transactions that fall outside the model's familiar input distribution using **standardized Euclidean distance**:

$$d(\mathbf{x}) = \sqrt{\sum_{i=1}^{30} \left(\frac{x_i - \mu_i}{\sigma_i}\right)^2}$$

Where:
- \(x_i\) is the preprocessed feature value
- \(\mu_i\) is the training-data mean for feature \(i\)
- \(\sigma_i\) is the training-data standard deviation for feature \(i\)

**Reference statistics** are pre-computed from the training set (transformed through the fitted preprocessing pipeline) and saved to `data/processed/ood_reference.npz`. The test set is never used.

**Zero-variance handling:** Features with \(\sigma_i < 10^{-12}\) contribute zero to the distance.

| Configuration | Default |
|---|---|
| `ood_distance_threshold` | 20.0 |
| `ood_enabled` | `True` |
| `require_review_on_ood` | `True` |

**Trigger:** If `distance > threshold`, adds `OUT_OF_DISTRIBUTION`.

> [!NOTE]
> OOD detection is a risk signal, not proof of fraud. It indicates the model may be operating outside its training distribution and predictions may be less reliable.

### C. Prediction-Context Consistency Guard

Validates consistency across prediction, risk level, and recommended action based on the existing Risk Engine and Decision Engine mappings:

| Risk Level | Expected Action | Required Prediction |
|---|---|---|
| LOW | AUTO_APPROVE | 0 (legitimate) |
| MEDIUM | HUMAN_REVIEW | 0 or 1 |
| HIGH | AUTO_REJECT | 1 (fraud) |

HUMAN_REVIEW is valid for any prediction/risk combination.

| Configuration | Default |
|---|---|
| `require_review_on_context_mismatch` | `True` |

**Trigger:** If the combination is inconsistent, adds `PREDICTION_CONTEXT_MISMATCH`.

---

## Configuration

```python
from src.guardrails.advanced import AdvancedGuardrailConfig

config = AdvancedGuardrailConfig(
    min_confidence=0.80,              # Minimum model confidence
    max_uncertainty=0.20,             # Maximum model uncertainty
    ood_distance_threshold=20.0,      # Standardized Euclidean distance threshold
    ood_enabled=True,                 # Enable OOD detection
    require_review_on_low_confidence=True,
    require_review_on_ood=True,
    require_review_on_context_mismatch=True,
    expose_in_predict=False,          # Include Phase 8 fields in /predict by default
)
```

All configuration values are validated:
- `min_confidence` must be in [0.5, 1.0]
- `max_uncertainty` must be in [0.0, 0.5]
- `ood_distance_threshold` must be positive and finite
- Boolean flags must be actual `bool` values (not strings or integers)
- Configuration is frozen (immutable after creation)

---

## Reason Codes

| Code | Meaning |
|---|---|
| `LOW_MODEL_CONFIDENCE` | Model confidence below minimum threshold |
| `HIGH_MODEL_UNCERTAINTY` | Model uncertainty above maximum threshold |
| `OUT_OF_DISTRIBUTION` | Transaction is outside the training distribution |
| `PREDICTION_CONTEXT_MISMATCH` | Inconsistency between prediction, risk, and action |

Multiple reasons are preserved — the system does not stop at the first failure.

---

## Integration with Phase 5

The `combine_guardrails()` function merges Phase 5 and Phase 8 results:

- Status is `REQUIRES_REVIEW` if **either** phase requires review
- A Phase 5 failure is **never** overridden by a Phase 8 pass
- All Phase 5 reason codes are preserved
- Phase 8 reasons are appended without duplicates
- If Phase 5 already flagged `APPROVAL_CONTEXT_MISMATCH` or `REJECTION_CONTEXT_MISMATCH`, the Phase 8 `PREDICTION_CONTEXT_MISMATCH` is suppressed to avoid redundancy
- The original `recommended_action` is preserved unchanged

---

## API Response Changes

### Default `/predict` (backward compatible)

```json
{
  "prediction": 0,
  "probability": 0.0058,
  "risk_level": "LOW",
  "recommended_action": "AUTO_APPROVE",
  "guardrail_status": "PASSED",
  "guardrail_reasons": []
}
```

### `/predict?include_advanced=true`

```json
{
  "prediction": 0,
  "probability": 0.0058,
  "risk_level": "LOW",
  "recommended_action": "AUTO_APPROVE",
  "guardrail_status": "PASSED",
  "guardrail_reasons": [],
  "advanced_guardrail_status": "PASSED",
  "confidence": 0.9942,
  "uncertainty": 0.0058,
  "ood_status": "IN_DISTRIBUTION",
  "ood_distance": 4.72,
  "advanced_guardrail_reasons": []
}
```

The `guardrail_status` and `guardrail_reasons` always reflect the **combined** Phase 5 + Phase 8 verdict, regardless of whether advanced fields are exposed.

---

## Human Review Integration

Transactions flagged by Phase 8 enter the existing Phase 6 review workflow automatically:
- The combined `guardrail_status` determines whether review is required
- Original prediction, probability, risk level, and recommendation are preserved
- Phase 5 and Phase 8 reason codes are stored together in `guardrail_reasons`
- Review lifecycle (creation, decision, immutability) is unchanged

---

## Known Limitations

1. **OOD detection is distance-based only.** It uses standardized Euclidean distance, which assumes feature independence. More sophisticated methods (e.g., Mahalanobis distance, isolation forests) could be added in future phases.

2. **Confidence ≠ correctness.** A model can be highly confident and wrong. The confidence guard only catches uncertain predictions.

3. **Static reference statistics.** The OOD reference means and standard deviations are computed once from training data. They do not adapt to distribution shifts over time.

4. **No ensemble or multi-model checks.** The system evaluates a single model's output. Disagreement between multiple models could provide stronger reliability signals.
