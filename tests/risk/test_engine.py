"""Unit tests for the Risk Engine. Imports the real implementation."""

import pytest

from src.risk.engine import RiskConfig, assess_risk


@pytest.fixture
def config():
    return RiskConfig()


# ---------- normal cases ----------

def test_low_risk(config):
    assert assess_risk(0.10, config) == "LOW"


def test_medium_risk(config):
    assert assess_risk(0.50, config) == "MEDIUM"


def test_high_risk(config):
    assert assess_risk(0.90, config) == "HIGH"


# ---------- boundaries (lower bound inclusive, upper bound exclusive) ----------

def test_low_threshold_boundary_is_medium(config):
    assert assess_risk(0.30, config) == "MEDIUM"


def test_just_below_low_threshold_is_low(config):
    assert assess_risk(0.2999, config) == "LOW"


def test_high_threshold_boundary_is_high(config):
    assert assess_risk(0.70, config) == "HIGH"


def test_just_below_high_threshold_is_medium(config):
    assert assess_risk(0.6999, config) == "MEDIUM"


def test_zero_probability_is_low(config):
    assert assess_risk(0.0, config) == "LOW"


def test_one_probability_is_high(config):
    assert assess_risk(1.0, config) == "HIGH"


# ---------- invalid probabilities ----------

def test_negative_probability_raises(config):
    with pytest.raises(ValueError, match="Probability"):
        assess_risk(-0.2, config)


def test_probability_above_one_raises(config):
    with pytest.raises(ValueError, match="Probability"):
        assess_risk(1.4, config)


def test_nan_probability_raises(config):
    with pytest.raises(ValueError, match="Probability"):
        assess_risk(float("nan"), config)


# ---------- configuration ----------

def test_custom_thresholds_change_the_outcome():
    custom = RiskConfig(low_threshold=0.10, high_threshold=0.50)
    assert assess_risk(0.05, custom) == "LOW"
    assert assess_risk(0.20, custom) == "MEDIUM"  # LOW under the default policy
    assert assess_risk(0.60, custom) == "HIGH"    # MEDIUM under the default policy


def test_invalid_threshold_ordering_raises():
    with pytest.raises(ValueError, match="Risk thresholds"):
        assess_risk(0.5, RiskConfig(low_threshold=0.70, high_threshold=0.30))


def test_equal_thresholds_raise():
    with pytest.raises(ValueError, match="Risk thresholds"):
        assess_risk(0.5, RiskConfig(low_threshold=0.50, high_threshold=0.50))


def test_invalid_low_threshold_raises():
    with pytest.raises(ValueError, match="Risk thresholds"):
        assess_risk(0.5, RiskConfig(low_threshold=0.0, high_threshold=0.70))


def test_invalid_high_threshold_raises():
    with pytest.raises(ValueError, match="Risk thresholds"):
        assess_risk(0.5, RiskConfig(low_threshold=0.30, high_threshold=1.0))
