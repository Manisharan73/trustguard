# TrustGuard — Phase 7: Explainability

## 1. What Explainability Means in TrustGuard

In TrustGuard, explainability answers a fundamental operational question:
**Why did the machine learning model produce a particular prediction and fraud probability for a given transaction?**

TrustGuard is not an opaque classification system. Operating in high-stakes financial fraud detection requires defensible, auditable reasoning:
- Investigators and human reviewers need to understand why an alert triggered.
- Dispute handlers and compliance officers require transparent attribution for adverse actions.
- Model validators need to confirm that model decisions align with known risk factors rather than spurious artifacts.

Explainability provides local, feature-level attribution: it quantifies exactly how much each individual input feature shifted the model's output towards or away from a fraud prediction.

---

## 2. Why SHAP Was Selected

**SHAP (SHapley Additive exPlanations)** was selected as TrustGuard's explainability framework because of its solid theoretical foundation in cooperative game theory (Shapley values).

Key reasons:
1. **Local Accuracy (Additivity)**: The sum of feature contributions plus the reference base value strictly equals the model's raw decision output for the instance.
2. **Missingness & Consistency**: Features with zero influence receive zero attribution, and if a feature's effect increases, its attribution never decreases.
3. **Exact Linear Explanations**: For generalized linear models such as Logistic Regression, `shap.LinearExplainer` computes exact Shapley attributions analytically without sampling approximations or perturbation variance.
4. **Computational Efficiency**: `LinearExplainer` executes in less than 3 milliseconds per transaction, making it safe for low-latency production APIs without bottlenecking downstream guardrails.

---

## 3. How SHAP Works at a Conceptual Level

Shapley values originate in cooperative game theory, where players collaborate in a game to produce a payout. A player's Shapley value represents their average marginal contribution across all possible coalitions of players.

In machine learning:
- The **"players"** are the transaction features ($Time, Amount, V_1, \dots, V_{28}$).
- The **"game"** is the model's prediction score for a specific transaction.
- The **"payout"** is the difference between the model's actual prediction and the expected prediction over a reference background distribution ($E[f(X)]$).

SHAP computes how much the presence of each feature value moves the prediction away from the baseline expected value.

---

## 4. Why Logistic Regression Is Suitable for Linear Explainability

The TrustGuard baseline model is a balanced Logistic Regression pipeline. A logistic regression model defines its decision surface via a linear combination of transformed features:

$$f(x) = \beta_0 + \sum_{i=1}^{p} \beta_i x_i$$

where $f(x)$ represents the **log-odds** (logit) of fraud:

$$\text{log-odds} = \ln\left(\frac{P(\text{fraud})}{1 - P(\text{fraud})}\right)$$

Because the model is strictly additive in log-odds space, each feature's contribution can be calculated directly and cleanly:

$$\phi_i(x) = \beta_i (x_i - E[X_i])$$

where:
- $\beta_i$ is the model coefficient for feature $i$.
- $x_i$ is the preprocessed (scaled or passed-through) value of feature $i$.
- $E[X_i]$ is the expected value of feature $i$ across the background reference distribution.

This linear formulation avoids the approximations required by deep or tree-based ensemble models and satisfies the additivity property with machine precision ($10^{-14}$).

---

## 5. Background Reference Dataset & Generation Method

To evaluate marginal contributions $x_i - E[X_i]$, SHAP requires a background reference distribution:

1. **Source Data**: The reference background is derived strictly from the **training set** (`X_train.iloc[:100]`), originating from the reproducible stratified 80/20 train/test split in `src/ml/preprocess.py`.
2. **Leakage Prevention**: Zero rows from the held-out test set (`X_test`) are included. The test split (56,962 rows) remains strictly unseen.
3. **Preprocessing Pipeline Alignment**: The 100 training rows are transformed using the model's pre-fitted `ColumnTransformer` (standardizing `Time` and `Amount` using training statistics and passing `V1`–`V28` through).
4. **Feature Ordering**: The column ordering of the background matrix matches the classifier's input space exactly:
   $$\text{Columns} = [\text{scale\_\_Time}, \text{scale\_\_Amount}, \text{remainder\_\_V1}, \dots, \text{remainder\_\_V28}]$$
5. **Persistence & Loading**: The transformed matrix is persisted as a compact array at `data/processed/background_reference.npy` (24 KB). At application startup, the file is loaded with `np.load()` in $<1$ ms without modifying, retraining, or refitting the model artifact.

---

## 6. What a Base Value Represents

The **base value** ($\phi_0$) is the reference score: the expected value of the model's output over the background reference dataset:

$$\text{base\_value} = E[f(X_{\text{background}})] = \beta_0 + \sum_{i=1}^{p} \beta_i E[X_{\text{background}, i}]$$

In TrustGuard:
- The base value is `-3.948099` in log-odds space.
- Applying the logistic sigmoid function:
  $$\sigma(\text{base\_value}) = \frac{1}{1 + e^{-(-3.948099)}} \approx 0.0189 \quad (1.89\%)$$
- This represents the model's baseline expectation of fraud before observing the specific attributes of an incoming transaction.
- Every individual transaction's feature contributions shift the prediction upward or downward from this baseline.

---

## 7. What Positive and Negative Contributions Mean

All contributions in TrustGuard explain the **fraud class (Class = 1)**:

- **Positive Contribution (`TOWARDS_FRAUD`)**: The feature value increases the log-odds of fraud relative to the background baseline, pushing the model toward predicting fraud.
- **Negative Contribution (`TOWARDS_LEGITIMATE`)**: The feature value decreases the log-odds of fraud relative to the background baseline, pushing the model toward predicting legitimate.
- **Magnitude (`|contribution|`)**: The absolute value indicates the strength of the influence. Features with larger absolute contributions exerted more leverage over the model's score.

---

## 8. Difference Between Feature Contribution and Causation

A feature contribution reflects **statistical model sensitivity**, not real-world causation:
- A high positive contribution for `V14` means that within the trained logistic regression model, the value of `V14` strongly increased the computed fraud score.
- It does **not** prove that `V14` caused the transaction to be fraudulent in reality.
- Observational credit card features may correlate with fraud patterns without being the underlying mechanism of criminal activity.
- Explanations should be used to inspect and audit model behavior, not as unilateral legal or causal determinations.

---

## 9. Difference Between Log-Odds and Probability

It is critical to distinguish between the model's linear scoring space (**log-odds**) and its calibrated output space (**probability**):

1. **Log-Odds Space ($-\infty$ to $+\infty$)**:
   - Logistic regression is linear in log-odds: $\text{score} = \beta_0 + \sum \beta_i x_i$.
   - SHAP contributions are strictly additive in this space: $\text{base\_value} + \sum_{i=1}^{30} \phi_i = \text{score}$.
   - TrustGuard measures and reports feature contributions in log-odds because additivity holds exactly.

2. **Probability Space ($0.0$ to $1.0$)**:
   - The fraud probability is obtained by passing the log-odds through the non-linear sigmoid transform:
     $$P(\text{fraud}) = \sigma(\text{score}) = \frac{1}{1 + e^{-\text{score}}}$$
   - Because the sigmoid function is non-linear and S-shaped, a $+1.0$ shift in log-odds does not correspond to a constant change in probability: near 50%, a small log-odds shift causes a large probability change, whereas near 0% or 100%, the same shift causes a very small probability change.
   - Reporting contributions in log-odds space preserves mathematical accuracy and avoids distorting feature rank order.

---

## 10. How to Interpret the API Response

The `POST /explain` endpoint returns structured JSON:

```json
{
  "prediction": 1,
  "probability": 0.9999999999999964,
  "fraud_probability": 0.9999999999999964,
  "explainer": "SHAP LinearExplainer",
  "explanation_space": "log_odds",
  "base_value": -3.948099486428257,
  "top_features": [
    {
      "feature": "V14",
      "feature_value": -6.17428834800643,
      "contribution": 9.59744350650615,
      "direction": "TOWARDS_FRAUD"
    },
    {
      "feature": "V17",
      "feature_value": -6.53652073527011,
      "contribution": 7.307597476723742,
      "direction": "TOWARDS_FRAUD"
    },
    {
      "feature": "V10",
      "feature_value": -4.88114292689057,
      "contribution": 5.950070683887223,
      "direction": "TOWARDS_FRAUD"
    },
    {
      "feature": "V16",
      "feature_value": -4.84969238709652,
      "contribution": 5.3631294464356225,
      "direction": "TOWARDS_FRAUD"
    },
    {
      "feature": "V12",
      "feature_value": -4.68638689759229,
      "contribution": 5.154027300853589,
      "direction": "TOWARDS_FRAUD"
    }
  ],
  "summary": "The model predicted FRAUD with a fraud probability of 100.00%. All top examined features contributed towards fraud, led by V14 and V17 and V10. Feature contributions are calculated in log-odds space using SHAP LinearExplainer and indicate model sensitivity, not causal real-world relationships.",
  "explanation": "The model predicted FRAUD with a fraud probability of 100.00%. All top examined features contributed towards fraud, led by V14 and V17 and V10. Feature contributions are calculated in log-odds space using SHAP LinearExplainer and indicate model sensitivity, not causal real-world relationships."
}
```

### Interpretation Guide:
- `prediction`: The binary decision (`1` = fraud, `0` = legitimate).
- `fraud_probability`: The calibrated model probability $P(\text{fraud})$.
- `base_value`: Baseline log-odds ($-3.948099$).
- `top_features`: The $k$ most influential features ordered by descending absolute impact ($|\text{contribution}|$).
- `direction`: Clearly signals whether the feature increased risk (`TOWARDS_FRAUD`) or decreased risk (`TOWARDS_LEGITIMATE`).
- `summary`: A concise natural-language narrative for human reviewers.

---

## 11. Real Examples & Exact Mathematical Reconstruction

The API displays the top $k$ features (default $k=5$) for human readability. However, **all 30 features** participate in the model's additive score. The mathematical reconstruction below explicitly separates the 5 top displayed features from the remaining 25 features.

### Example 1: High-Risk Fraud Transaction (Held-Out Test Set)

- **Prediction**: `1` (FRAUD)
- **Fraud Probability**: `99.99999999999964%`
- **Model Output (`decision_function`)**: `+33.265216` log-odds
- **Base Value ($\phi_0$)**: `-3.948099`

#### Top 5 Displayed Features:

| Feature | Feature Value | Contribution (Log-Odds) | Direction |
| :--- | :--- | :--- | :--- |
| **V14** | `-6.174288` | `+9.597444` | `TOWARDS_FRAUD` |
| **V17** | `-6.536521` | `+7.307597` | `TOWARDS_FRAUD` |
| **V10** | `-4.881143` | `+5.950071` | `TOWARDS_FRAUD` |
| **V16** | `-4.849692` | `+5.363129` | `TOWARDS_FRAUD` |
| **V12** | `-4.686387` | `+5.154027` | `TOWARDS_FRAUD` |

#### Exact Additive Decomposition:
$$\sum_{i=1}^{5} \phi_i = 9.597444 + 7.307597 + 5.950071 + 5.363129 + 5.154027 = +33.372268$$
$$\sum_{i=6}^{30} \phi_i = +3.841047 \quad (\text{net contribution of remaining 25 features})$$
$$\sum_{i=1}^{30} \phi_i = 33.372268 + 3.841047 = +37.213316 \quad (\text{sum across all 30 features})$$
$$\text{Total Log-Odds} = \phi_0 + \sum_{i=1}^{30} \phi_i = -3.948099 + 37.213316 = +33.265216$$

$$\text{Probability} = \sigma(33.265216) = \frac{1}{1 + e^{-33.265216}} = 0.9999999999999964 \quad (99.99\%)$$

---

### Example 2: Low-Risk Legitimate Transaction (Held-Out Test Set)

- **Prediction**: `0` (LEGITIMATE)
- **Fraud Probability**: `0.5842%`
- **Model Output (`decision_function`)**: `-5.136793` log-odds
- **Base Value ($\phi_0$)**: `-3.948099`

#### Top 5 Displayed Features:

| Feature | Feature Value | Contribution (Log-Odds) | Direction |
| :--- | :--- | :--- | :--- |
| **V7** | `+1.885879` | `-1.346862` | `TOWARDS_LEGITIMATE` |
| **V4** | `-1.328366` | `-1.089433` | `TOWARDS_LEGITIMATE` |
| **V5** | `+1.388996` | `+1.066780` | `TOWARDS_FRAUD` |
| **V2** | `+1.408105` | `+0.922762` | `TOWARDS_FRAUD` |
| **V16** | `-0.741398` | `+0.813729` | `TOWARDS_FRAUD` |

#### Exact Additive Decomposition:
$$\sum_{i=1}^{5} \phi_i = -1.346862 - 1.089433 + 1.066780 + 0.922762 + 0.813729 = +0.366976$$
$$\sum_{i=6}^{30} \phi_i = -1.555669 \quad (\text{net contribution of remaining 25 features})$$
$$\sum_{i=1}^{30} \phi_i = 0.366976 - 1.555669 = -1.188693 \quad (\text{sum across all 30 features})$$
$$\text{Total Log-Odds} = \phi_0 + \sum_{i=1}^{30} \phi_i = -3.948099 - 1.188693 = -5.136793$$

$$\text{Probability} = \sigma(-5.136793) = \frac{1}{1 + e^{5.136793}} = 0.00584217 \quad (0.58\%)$$

---

## 12. Limitations of the Explanation

1. **Model-Specific Fidelity**: SHAP explains the *trained Logistic Regression model*, not the underlying physical transaction mechanism. If the model has learned a biased or confounded relationship, SHAP faithfully reports that relationship.
2. **Feature Interdependence**: PCA transformations applied to features `V1`–`V28` make direct business interpretation of individual $V_i$ features abstract without the original PCA projection matrix.
3. **Linearity Assumption**: Logistic regression assumes linear additive effects in log-odds. The model does not capture non-linear feature interactions (e.g., interaction between `Time` and `Amount`), and SHAP LinearExplainer reflects this linear structure.
4. **Reference Distribution Dependency**: Contributions are measured relative to the reference background distribution. Using a different background population would shift the base value and marginal attributions accordingly.
