"""Unit tests for the Decision Engine. Imports the real implementation."""

import pytest

from src.decision.engine import (
    AUTO_APPROVE,
    AUTO_REJECT,
    HUMAN_REVIEW,
    RISK_LEVELS,
    VALID_ACTIONS,
    DecisionConfig,
    recommend_action,
)
from src.risk.engine import RiskConfig, assess_risk


@pytest.fixture
def config():
    return DecisionConfig()


# ---------- the decision policy ----------

def test_low_risk_is_auto_approved(config):
    assert recommend_action("LOW", config) == "AUTO_APPROVE"


def test_medium_risk_goes_to_human_review(config):
    assert recommend_action("MEDIUM", config) == "HUMAN_REVIEW"


def test_high_risk_is_auto_rejected(config):
    assert recommend_action("HIGH", config) == "AUTO_REJECT"


def test_default_policy_table(config):
    assert config.policy == {
        "LOW": AUTO_APPROVE,
        "MEDIUM": HUMAN_REVIEW,
        "HIGH": AUTO_REJECT,
    }


def test_every_risk_level_maps_to_a_valid_action(config):
    for risk_level in RISK_LEVELS:
        assert recommend_action(risk_level, config) in VALID_ACTIONS


# ---------- invalid / missing risk levels (engine fails closed) ----------

@pytest.mark.parametrize(
    "bad_value",
    [None, "", "low", "Medium", "CRITICAL", " HIGH", "HIGH ", 0, 1.5, ["LOW"]],
)
def test_invalid_risk_level_raises(config, bad_value):
    with pytest.raises(ValueError, match="Unsupported risk level"):
        recommend_action(bad_value, config)


def test_omitted_risk_level_raises_type_error(config):
    with pytest.raises(TypeError):
        recommend_action(config=config)


def test_omitted_config_raises_type_error():
    with pytest.raises(TypeError):
        recommend_action("LOW")


# ---------- configuration ----------

def test_custom_policy_changes_the_recommendation():
    strict = DecisionConfig(
        policy={"LOW": HUMAN_REVIEW, "MEDIUM": HUMAN_REVIEW, "HIGH": AUTO_REJECT}
    )
    assert recommend_action("LOW", strict) == "HUMAN_REVIEW"
    assert recommend_action("HIGH", strict) == "AUTO_REJECT"


def test_policy_missing_a_risk_level_raises():
    incomplete = DecisionConfig(policy={"LOW": AUTO_APPROVE, "MEDIUM": HUMAN_REVIEW})
    with pytest.raises(ValueError, match="Decision policy"):
        recommend_action("LOW", incomplete)


def test_policy_with_an_extra_risk_level_raises():
    extra = DecisionConfig(
        policy={
            "LOW": AUTO_APPROVE,
            "MEDIUM": HUMAN_REVIEW,
            "HIGH": AUTO_REJECT,
            "CRITICAL": AUTO_REJECT,
        }
    )
    with pytest.raises(ValueError, match="Decision policy"):
        recommend_action("LOW", extra)


def test_policy_with_an_invalid_action_raises():
    bad_action = DecisionConfig(
        policy={"LOW": AUTO_APPROVE, "MEDIUM": HUMAN_REVIEW, "HIGH": "SHRUG"}
    )
    with pytest.raises(ValueError, match="Invalid action"):
        recommend_action("LOW", bad_action)


def test_policy_that_is_not_a_dict_raises():
    with pytest.raises(ValueError, match="Decision policy"):
        recommend_action("LOW", DecisionConfig(policy=None))


def test_default_configs_do_not_share_policy_state():
    first = DecisionConfig()
    first.policy["LOW"] = HUMAN_REVIEW  # mutate one instance
    assert DecisionConfig().policy["LOW"] == AUTO_APPROVE


def test_custom_medium_policy_changes_the_recommendation():
    custom = DecisionConfig(
        policy={"LOW": AUTO_APPROVE, "MEDIUM": AUTO_REJECT, "HIGH": AUTO_REJECT}
    )
    assert recommend_action("MEDIUM", custom) == "AUTO_REJECT"
    assert recommend_action("LOW", custom) == "AUTO_APPROVE"


def test_empty_policy_raises():
    with pytest.raises(ValueError, match="Decision policy"):
        recommend_action("LOW", DecisionConfig(policy={}))


@pytest.mark.parametrize(
    "bad_config",
    [
        None,
        {},
        "policy",
        {"LOW": AUTO_APPROVE, "MEDIUM": HUMAN_REVIEW, "HIGH": AUTO_REJECT},  # a bare policy dict
    ],
)
def test_config_that_is_not_a_decision_config_raises_type_error(bad_config):
    with pytest.raises(TypeError, match="DecisionConfig"):
        recommend_action("LOW", bad_config)


# ---------- Risk Engine -> Decision Engine chain ----------

@pytest.mark.parametrize(
    "probability, expected_risk, expected_action",
    [
        (0.10, "LOW", "AUTO_APPROVE"),
        (0.30, "MEDIUM", "HUMAN_REVIEW"),
        (0.50, "MEDIUM", "HUMAN_REVIEW"),
        (0.70, "HIGH", "AUTO_REJECT"),
        (0.95, "HIGH", "AUTO_REJECT"),
    ],
)
def test_probability_to_action_chain(probability, expected_risk, expected_action):
    risk_level = assess_risk(probability, RiskConfig())
    assert risk_level == expected_risk
    assert recommend_action(risk_level, DecisionConfig()) == expected_action
