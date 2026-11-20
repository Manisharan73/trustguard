"""Decision Engine: converts a risk level into a recommended action.

Responsibility: risk level -> AUTO_APPROVE / HUMAN_REVIEW / AUTO_REJECT.

It consumes only the risk level. It does not see the probability or the ML
prediction, does not recalculate risk, and has no dependency on FastAPI.
Guardrails and human-review workflows are later phases.

The output is a RECOMMENDATION, not the final transaction decision.
"""

from dataclasses import dataclass, field

# Possible recommended actions
AUTO_APPROVE = "AUTO_APPROVE"
HUMAN_REVIEW = "HUMAN_REVIEW"
AUTO_REJECT = "AUTO_REJECT"

VALID_ACTIONS = (AUTO_APPROVE, HUMAN_REVIEW, AUTO_REJECT)

# Risk levels produced by the Risk Engine
RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")


def _default_policy() -> dict:
    """The initial decision policy. Edit this table to change the policy."""
    return {
        "LOW": AUTO_APPROVE,
        "MEDIUM": HUMAN_REVIEW,
        "HIGH": AUTO_REJECT,
    }


@dataclass
class DecisionConfig:
    """Decision policy: which action is recommended for each risk level."""

    policy: dict = field(default_factory=_default_policy)


def _validate_policy(policy: dict) -> None:
    """The policy must cover every risk level and use only valid actions."""
    if not isinstance(policy, dict) or set(policy) != set(RISK_LEVELS):
        raise ValueError(
            f"Decision policy must define exactly these risk levels: {list(RISK_LEVELS)}."
        )
    for risk_level, action in policy.items():
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action {action!r} for risk level {risk_level}. "
                f"Expected one of: {list(VALID_ACTIONS)}."
            )


def recommend_action(risk_level: str, config: DecisionConfig) -> str:
    """
    Convert a risk level into a recommended action.

    The risk level must be exactly "LOW", "MEDIUM" or "HIGH" (case-sensitive).
    Anything else raises ValueError. The engine fails closed: an unknown risk
    level is never turned into AUTO_APPROVE.
    """

    if not isinstance(risk_level, str) or risk_level not in RISK_LEVELS:
        raise ValueError(
            f"Unsupported risk level: {risk_level!r}. "
            f"Expected one of: {list(RISK_LEVELS)}."
        )

    if not isinstance(config, DecisionConfig):
        raise TypeError(
            f"config must be a DecisionConfig instance, got {type(config).__name__}."
        )

    _validate_policy(config.policy)

    return config.policy[risk_level]
