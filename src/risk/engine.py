"""Risk Engine: converts a fraud probability into a risk level.

Responsibility: Probability -> LOW / MEDIUM / HIGH. Nothing else.
It does not change the ML prediction, make decisions, or apply guardrails,
and it has no dependency on FastAPI.
"""

from dataclasses import dataclass


@dataclass
class RiskConfig:
    """Risk policy. These are initial policy values, not statistically
    optimized thresholds; they are expected to be evaluated and tuned later."""

    low_threshold: float = 0.30
    high_threshold: float = 0.70


def assess_risk(probability: float, config: RiskConfig) -> str:
    """
    Convert fraud probability into a risk level.

    0.00 <= p < low_threshold            -> LOW
    low_threshold <= p < high_threshold  -> MEDIUM
    high_threshold <= p <= 1.00          -> HIGH
    """

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0 < config.low_threshold < config.high_threshold < 1:
        raise ValueError(
            "Risk thresholds must satisfy: "
            "0 < low_threshold < high_threshold < 1."
        )

    if probability < config.low_threshold:
        return "LOW"

    if probability < config.high_threshold:
        return "MEDIUM"

    return "HIGH"
