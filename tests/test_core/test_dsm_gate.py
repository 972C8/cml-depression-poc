"""Tests for src/core/dsm_gate.py - DSM-Gate Logic & Episode Decision Module.

Tests aligned with DSM-5 two-stage gate logic:
  Stage 1: N-of-M per-indicator presence
  Stage 2: Episode aggregation (count + core check)
"""

import pytest

from src.core.config import (
    AnalysisConfig,
    BiomarkerWeight,
    DSMGateConfig,
    EMAConfig,
    EpisodeConfig,
    ExperimentContextEvalConfig,
    IndicatorConfig,
    MarkerMembership,
    MarkerMembershipSet,
    ContextAssumptionDef,
    ContextConditionDef,
)
from src.core.dsm_gate import (
    DSMGate,
    EpisodeDecision,
    IndicatorGateResult,
    apply_indicator_gate,
    compute_episode_decision,
)


def _minimal_context_eval() -> ExperimentContextEvalConfig:
    """Create a minimal context evaluation config for tests.

    Bypasses loading from YAML files which may not be available in test env.
    """
    return ExperimentContextEvalConfig(
        marker_memberships={
            "people_in_room": MarkerMembership(
                sets={
                    "low": MarkerMembershipSet(
                        type="triangular", params=[0.0, 0.0, 3.0]
                    ),
                    "high": MarkerMembershipSet(
                        type="triangular", params=[2.0, 10.0, 10.0]
                    ),
                }
            ),
        },
        context_assumptions={
            "solitary_digital": ContextAssumptionDef(
                conditions=[
                    ContextConditionDef(
                        marker="people_in_room", fuzzy_set="high", weight=1.0
                    ),
                ],
                operator="WEIGHTED_MEAN",
            ),
        },
        neutral_threshold=0.3,
        ema=EMAConfig(),
    )


def _make_config(**kwargs) -> AnalysisConfig:
    """Create AnalysisConfig with minimal context_evaluation to bypass broken YAML loading."""
    if "context_evaluation" not in kwargs:
        kwargs["context_evaluation"] = _minimal_context_eval()
    return AnalysisConfig(**kwargs)


def _default_test_config() -> AnalysisConfig:
    """Create a default test config mimicking get_default_config() structure.

    Uses minimal context_evaluation to avoid YAML loading issues in tests.
    """
    return _make_config(
        indicators={
            "social_withdrawal": IndicatorConfig(
                biomarkers={
                    "speech_activity": BiomarkerWeight(
                        weight=0.5, direction="higher_is_worse"
                    ),
                    "connections": BiomarkerWeight(
                        weight=0.5, direction="lower_is_worse"
                    ),
                }
            ),
            "diminished_interest": IndicatorConfig(
                biomarkers={
                    "activity_level": BiomarkerWeight(
                        weight=1.0, direction="lower_is_worse"
                    ),
                }
            ),
            "sleep_disturbance": IndicatorConfig(
                biomarkers={
                    "sleep_duration": BiomarkerWeight(
                        weight=0.5, direction="lower_is_worse"
                    ),
                    "awakenings": BiomarkerWeight(
                        weight=0.5, direction="higher_is_worse"
                    ),
                }
            ),
        },
        context_weights={},
        dsm_gate_defaults=DSMGateConfig(theta=0.5, m_window=14, gate_need=10),
        episode=EpisodeConfig(
            min_indicators=5,
            core_indicators=("social_withdrawal", "diminished_interest"),
        ),
    )


class TestIndicatorGateResult:
    """Tests for IndicatorGateResult dataclass."""

    def test_creation_basic(self):
        """Test IndicatorGateResult creation with valid values."""
        result = IndicatorGateResult(
            indicator_name="test_indicator",
            presence_flag=True,
            days_above_threshold=10,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True, True, False, True) * 3 + (True, True),
            insufficient_data=False,
        )
        assert result.indicator_name == "test_indicator"
        assert result.presence_flag is True
        assert result.days_above_threshold == 10
        assert result.days_evaluated == 14
        assert result.window_size == 14
        assert result.required_days == 10
        assert result.threshold == 0.5
        assert len(result.daily_flags) == 14
        assert result.insufficient_data is False

    def test_immutable(self):
        """Test that IndicatorGateResult is frozen/immutable."""
        result = IndicatorGateResult(
            indicator_name="test",
            presence_flag=True,
            days_above_threshold=10,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(),
            insufficient_data=False,
        )
        with pytest.raises(AttributeError):
            result.presence_flag = False

    def test_has_required_days_field(self):
        """Test that IndicatorGateResult has the required_days field for N-of-M."""
        result = IndicatorGateResult(
            indicator_name="test",
            presence_flag=False,
            days_above_threshold=0,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(),
            insufficient_data=False,
        )
        assert result.required_days == 10

    def test_insufficient_data_flag(self):
        """Test IndicatorGateResult with insufficient data."""
        result = IndicatorGateResult(
            indicator_name="test",
            presence_flag=False,
            days_above_threshold=0,
            days_evaluated=3,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(),
            insufficient_data=True,
        )
        assert result.insufficient_data is True
        assert result.days_evaluated < result.window_size


class TestEpisodeDecision:
    """Tests for EpisodeDecision dataclass."""

    def test_creation_basic(self):
        """Test EpisodeDecision creation with valid values."""
        indicator_result = IndicatorGateResult(
            indicator_name="test",
            presence_flag=True,
            days_above_threshold=10,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(),
            insufficient_data=False,
        )
        from datetime import UTC, datetime
        from types import MappingProxyType

        decision = EpisodeDecision(
            episode_likely=True,
            indicators_present=6,
            core_indicator_present=True,
            core_indicators_present=("social_withdrawal",),
            all_indicator_results=MappingProxyType({"test": indicator_result}),
            decision_rationale="6 indicators present",
            min_indicators_required=5,
            timestamp=datetime.now(UTC),
        )
        assert decision.episode_likely is True
        assert decision.indicators_present == 6
        assert decision.core_indicator_present is True
        assert "social_withdrawal" in decision.core_indicators_present
        assert decision.min_indicators_required == 5

    def test_immutable(self):
        """Test that EpisodeDecision is frozen/immutable."""
        from datetime import UTC, datetime
        from types import MappingProxyType

        decision = EpisodeDecision(
            episode_likely=True,
            indicators_present=5,
            core_indicator_present=True,
            core_indicators_present=(),
            all_indicator_results=MappingProxyType({}),
            decision_rationale="test",
            min_indicators_required=5,
            timestamp=datetime.now(UTC),
        )
        with pytest.raises(AttributeError):
            decision.episode_likely = False

    def test_all_indicator_results_immutable(self):
        """Test that all_indicator_results mapping is immutable."""
        from datetime import UTC, datetime
        from types import MappingProxyType

        decision = EpisodeDecision(
            episode_likely=False,
            indicators_present=0,
            core_indicator_present=False,
            core_indicators_present=(),
            all_indicator_results=MappingProxyType({}),
            decision_rationale="test",
            min_indicators_required=5,
            timestamp=datetime.now(UTC),
        )
        with pytest.raises(TypeError):
            decision.all_indicator_results["new_key"] = None


class TestApplyIndicatorGate:
    """Tests for apply_indicator_gate function (Stage 1: N-of-M presence)."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic config for testing (theta=0.5, M=14, N=10)."""
        return _make_config(
            indicators={
                "test_indicator": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            dsm_gate_defaults=DSMGateConfig(theta=0.5, m_window=14, gate_need=10),
        )

    def test_all_days_above_threshold(self, basic_config):
        """Test with all days above threshold — present (14 >= 10)."""
        daily_likelihoods = [0.8] * 14
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.presence_flag is True
        assert result.days_above_threshold == 14
        assert result.days_evaluated == 14
        assert result.required_days == 10
        assert result.insufficient_data is False
        assert all(result.daily_flags)

    def test_no_days_above_threshold(self, basic_config):
        """Test with no days above threshold — not present (0 < 10)."""
        daily_likelihoods = [0.3] * 14
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.presence_flag is False
        assert result.days_above_threshold == 0
        assert result.days_evaluated == 14
        assert result.insufficient_data is False
        assert not any(result.daily_flags)

    def test_exactly_at_gate_need(self, basic_config):
        """Test with exactly N=10 days above threshold — present."""
        daily_likelihoods = [0.6] * 10 + [0.3] * 4
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.days_above_threshold == 10
        assert result.presence_flag is True

    def test_one_below_gate_need(self, basic_config):
        """Test with N-1=9 days above threshold — not present."""
        daily_likelihoods = [0.6] * 9 + [0.3] * 5
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.days_above_threshold == 9
        assert result.presence_flag is False

    def test_some_days_above_threshold(self, basic_config):
        """Test with 5 days above threshold — not present (5 < 10)."""
        daily_likelihoods = [0.6] * 5 + [0.3] * 9
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.days_above_threshold == 5
        assert result.days_evaluated == 14
        assert result.presence_flag is False

    def test_insufficient_data_handling(self, basic_config):
        """Test insufficient data handling (less than M days)."""
        daily_likelihoods = [0.8] * 10
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.insufficient_data is True
        assert result.days_evaluated == 10
        assert len(result.daily_flags) == 10
        assert result.days_above_threshold == 10
        # presence_flag is False because insufficient data (< m_window days)
        assert result.presence_flag is False

    def test_empty_input(self, basic_config):
        """Test with empty daily_likelihoods."""
        result = apply_indicator_gate("test_indicator", [], basic_config)

        assert result.presence_flag is False
        assert result.insufficient_data is True
        assert result.days_evaluated == 0
        assert result.daily_flags == ()
        assert result.required_days == 10

    def test_per_indicator_theta_override(self):
        """Test per-indicator theta override (m_window and gate_need stay global)."""
        config = _make_config(
            indicators={
                "custom_indicator": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    },
                    dsm_gate=DSMGateConfig(theta=0.7, m_window=7, gate_need=5),
                )
            },
            context_weights={},
            dsm_gate_defaults=DSMGateConfig(theta=0.5, m_window=14, gate_need=10),
        )

        daily_likelihoods = [0.75] * 14
        result = apply_indicator_gate("custom_indicator", daily_likelihoods, config)

        assert result.threshold == 0.7  # Per-indicator theta used
        assert result.window_size == 14  # Global m_window always used
        assert result.required_days == 10  # Global gate_need always used
        assert result.presence_flag is True  # 14 >= 10

    def test_per_indicator_m_window_not_overridable(self):
        """Test that per-indicator dsm_gate does NOT override m_window or gate_need.

        m_window and gate_need are global parameters. Per-indicator overrides
        only affect theta.
        """
        config = _make_config(
            indicators={
                "custom_indicator": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    },
                    dsm_gate=DSMGateConfig(theta=0.6, m_window=7, gate_need=5),
                )
            },
            context_weights={},
            dsm_gate_defaults=DSMGateConfig(theta=0.5, m_window=14, gate_need=10),
        )

        # Provide 14 days of data
        daily_likelihoods = [0.65] * 14
        result = apply_indicator_gate("custom_indicator", daily_likelihoods, config)

        # m_window should be 14 (global), not 7 (per-indicator)
        assert result.window_size == 14
        assert result.days_evaluated == 14
        assert result.required_days == 10

    def test_uses_last_m_days(self, basic_config):
        """Test that only last M days are used."""
        daily_likelihoods = [0.9] * 6 + [0.3] * 14
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.days_evaluated == 14
        assert result.days_above_threshold == 0
        assert result.presence_flag is False

    def test_theta_boundary_exactly_equal(self, basic_config):
        """Test that theta boundary is >= (exactly theta counts as above)."""
        daily_likelihoods = [0.5] * 14
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.days_above_threshold == 14
        assert result.presence_flag is True

    def test_theta_boundary_just_below(self, basic_config):
        """Test that just below theta doesn't count."""
        daily_likelihoods = [0.49999] * 14
        result = apply_indicator_gate("test_indicator", daily_likelihoods, basic_config)

        assert result.days_above_threshold == 0
        assert result.presence_flag is False


# ============================================================================
# Tests for Episode Decision (Stage 2: Aggregation)
# ============================================================================


class TestComputeEpisodeDecision:
    """Tests for compute_episode_decision function (Stage 2: count + core check)."""

    @pytest.fixture
    def episode_config(self):
        """Create a config for episode testing."""
        return _make_config(
            indicators={
                "social_withdrawal": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                ),
                "diminished_interest": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                ),
            },
            context_weights={},
            dsm_gate_defaults=DSMGateConfig(theta=0.5, m_window=14, gate_need=10),
            episode=EpisodeConfig(
                min_indicators=5,
                core_indicators=("social_withdrawal", "diminished_interest"),
            ),
        )

    def _make_gate_result(
        self, name: str, presence: bool, days_above: int = 10
    ) -> IndicatorGateResult:
        """Helper to create IndicatorGateResult with given presence flag."""
        return IndicatorGateResult(
            indicator_name=name,
            presence_flag=presence,
            days_above_threshold=days_above,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True,) * days_above + (False,) * (14 - days_above),
            insufficient_data=False,
        )

    def test_episode_likely_5_present_with_core(self, episode_config):
        """5 indicators present including one core — episode likely."""
        gate_results = {
            "social_withdrawal": self._make_gate_result("social_withdrawal", True),
            "diminished_interest": self._make_gate_result("diminished_interest", True),
            "indicator_3": self._make_gate_result("indicator_3", True),
            "indicator_4": self._make_gate_result("indicator_4", True),
            "indicator_5": self._make_gate_result("indicator_5", True),
        }
        decision = compute_episode_decision(gate_results, episode_config)

        assert decision.episode_likely is True
        assert decision.indicators_present == 5
        assert decision.core_indicator_present is True
        assert "social_withdrawal" in decision.core_indicators_present

    def test_episode_likely_6_present_with_core(self, episode_config):
        """6 indicators present including core — episode likely."""
        gate_results = {
            "social_withdrawal": self._make_gate_result("social_withdrawal", True),
            "indicator_2": self._make_gate_result("indicator_2", True),
            "indicator_3": self._make_gate_result("indicator_3", True),
            "indicator_4": self._make_gate_result("indicator_4", True),
            "indicator_5": self._make_gate_result("indicator_5", True),
            "indicator_6": self._make_gate_result("indicator_6", True),
        }
        decision = compute_episode_decision(gate_results, episode_config)

        assert decision.episode_likely is True
        assert decision.indicators_present == 6

    def test_episode_not_likely_only_4_present(self, episode_config):
        """Only 4 indicators present — episode NOT likely (need >= 5)."""
        gate_results = {
            "social_withdrawal": self._make_gate_result("social_withdrawal", True),
            "diminished_interest": self._make_gate_result("diminished_interest", True),
            "indicator_3": self._make_gate_result("indicator_3", True),
            "indicator_4": self._make_gate_result("indicator_4", True),
            "indicator_5": self._make_gate_result("indicator_5", False, days_above=5),
        }
        decision = compute_episode_decision(gate_results, episode_config)

        assert decision.episode_likely is False
        assert decision.indicators_present == 4

    def test_episode_not_likely_no_core_present(self, episode_config):
        """5 indicators present but no core — episode NOT likely."""
        gate_results = {
            "social_withdrawal": self._make_gate_result(
                "social_withdrawal", False, days_above=3
            ),
            "diminished_interest": self._make_gate_result(
                "diminished_interest", False, days_above=2
            ),
            "indicator_3": self._make_gate_result("indicator_3", True),
            "indicator_4": self._make_gate_result("indicator_4", True),
            "indicator_5": self._make_gate_result("indicator_5", True),
            "indicator_6": self._make_gate_result("indicator_6", True),
            "indicator_7": self._make_gate_result("indicator_7", True),
        }
        decision = compute_episode_decision(gate_results, episode_config)

        assert decision.episode_likely is False
        assert decision.indicators_present == 5
        assert decision.core_indicator_present is False

    def test_episode_likely_one_core_present_other_absent(self, episode_config):
        """One core present, other not — still sufficient for episode."""
        gate_results = {
            "social_withdrawal": self._make_gate_result("social_withdrawal", True),
            "diminished_interest": self._make_gate_result(
                "diminished_interest", False, days_above=4
            ),
            "indicator_3": self._make_gate_result("indicator_3", True),
            "indicator_4": self._make_gate_result("indicator_4", True),
            "indicator_5": self._make_gate_result("indicator_5", True),
            "indicator_6": self._make_gate_result("indicator_6", True),
        }
        decision = compute_episode_decision(gate_results, episode_config)

        assert decision.episode_likely is True
        assert decision.core_indicator_present is True
        assert "social_withdrawal" in decision.core_indicators_present
        assert "diminished_interest" not in decision.core_indicators_present

    def test_decision_rationale_generated(self, episode_config):
        """Test decision rationale generation."""
        gate_results = {
            "social_withdrawal": self._make_gate_result("social_withdrawal", True),
        }
        decision = compute_episode_decision(gate_results, episode_config)

        assert "NOT MET" in decision.decision_rationale

    def test_timestamp_present(self, episode_config):
        """Test that timestamp is set."""
        decision = compute_episode_decision({}, episode_config)

        assert decision.timestamp is not None
        from datetime import datetime

        assert isinstance(decision.timestamp, datetime)

    def test_empty_gate_results(self, episode_config):
        """Test with empty gate results."""
        decision = compute_episode_decision({}, episode_config)

        assert decision.episode_likely is False
        assert decision.indicators_present == 0
        assert decision.core_indicator_present is False


class TestDeterminism:
    """Tests for determinism (same inputs = same outputs)."""

    def test_apply_indicator_gate_deterministic(self):
        """Test that apply_indicator_gate is deterministic."""
        config = _default_test_config()
        daily_likelihoods = [0.6, 0.4, 0.8, 0.3] * 4

        result1 = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)
        result2 = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result1.presence_flag == result2.presence_flag
        assert result1.days_above_threshold == result2.days_above_threshold
        assert result1.daily_flags == result2.daily_flags

    def test_compute_episode_decision_deterministic(self):
        """Test that compute_episode_decision is deterministic (except timestamp)."""
        config = _default_test_config()

        gate_result = IndicatorGateResult(
            indicator_name="social_withdrawal",
            presence_flag=True,
            days_above_threshold=10,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True,) * 10 + (False,) * 4,
            insufficient_data=False,
        )
        gate_results = {"social_withdrawal": gate_result}

        decision1 = compute_episode_decision(gate_results, config)
        decision2 = compute_episode_decision(gate_results, config)

        assert decision1.episode_likely == decision2.episode_likely
        assert decision1.indicators_present == decision2.indicators_present
        assert decision1.core_indicator_present == decision2.core_indicator_present
        assert decision1.decision_rationale == decision2.decision_rationale


class TestDSMGateClass:
    """Tests for DSMGate class."""

    def test_init(self):
        """Test DSMGate initialization."""
        config = _default_test_config()
        gate = DSMGate(config)
        assert gate._config == config

    def test_apply_gate(self):
        """Test DSMGate.apply_gate method."""
        config = _default_test_config()
        gate = DSMGate(config)

        daily_likelihoods = [0.8] * 14
        result = gate.apply_gate("social_withdrawal", daily_likelihoods)

        assert isinstance(result, IndicatorGateResult)
        assert result.presence_flag is True

    def test_apply_all_gates(self):
        """Test DSMGate.apply_all_gates method."""
        config = _default_test_config()
        gate = DSMGate(config)

        indicator_scores = {
            "social_withdrawal": [0.8] * 14,
            "diminished_interest": [0.3] * 14,
            "sleep_disturbance": [0.6] * 14,
        }
        results = gate.apply_all_gates(indicator_scores)

        assert len(results) == 3
        assert results["social_withdrawal"].presence_flag is True
        assert results["diminished_interest"].presence_flag is False

    def test_compute_episode(self):
        """Test DSMGate.compute_episode method."""
        config = _default_test_config()
        gate = DSMGate(config)

        gate_result = IndicatorGateResult(
            indicator_name="social_withdrawal",
            presence_flag=True,
            days_above_threshold=10,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True,) * 10 + (False,) * 4,
            insufficient_data=False,
        )
        gate_results = {"social_withdrawal": gate_result}

        decision = gate.compute_episode(gate_results)

        assert isinstance(decision, EpisodeDecision)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_day_input(self):
        """Test with single day input."""
        config = _default_test_config()
        result = apply_indicator_gate("social_withdrawal", [0.8], config)

        assert result.insufficient_data is True
        assert result.days_evaluated == 1
        assert len(result.daily_flags) == 1
        assert result.daily_flags[0] is True
        assert result.days_above_threshold == 1

    def test_indicator_not_in_config(self):
        """Test with indicator not defined in config."""
        config = _make_config(
            indicators={
                "other_indicator": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
        )
        result = apply_indicator_gate("unknown_indicator", [0.8] * 14, config)

        assert result.indicator_name == "unknown_indicator"
        assert result.threshold == 0.6  # Default theta

    def test_very_long_history(self):
        """Test with very long history (uses only last M days)."""
        config = _default_test_config()
        daily_likelihoods = [0.3] * 986 + [0.8] * 14
        result = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result.days_evaluated == 14
        assert result.days_above_threshold == 14
        assert result.presence_flag is True

    def test_negative_likelihood_values(self):
        """Test with negative likelihood values (edge case)."""
        config = _default_test_config()
        daily_likelihoods = [-0.5] * 14
        result = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result.days_above_threshold == 0
        assert result.presence_flag is False

    def test_likelihood_values_above_one(self):
        """Test with likelihood values above 1.0 (edge case)."""
        config = _default_test_config()
        daily_likelihoods = [1.5] * 14
        result = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result.days_above_threshold == 14
        assert result.presence_flag is True

    def test_nan_values_treated_as_zero(self):
        """Test that NaN values are treated as 0.0 (below threshold)."""
        config = _default_test_config()
        daily_likelihoods = [float("nan")] * 14
        result = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result.days_above_threshold == 0
        assert result.presence_flag is False
        assert all(flag is False for flag in result.daily_flags)

    def test_infinity_values_treated_as_zero(self):
        """Test that infinity values are treated as 0.0 (below threshold)."""
        config = _default_test_config()
        daily_likelihoods = [float("inf")] * 7 + [float("-inf")] * 7
        result = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result.days_above_threshold == 0
        assert result.presence_flag is False

    def test_mixed_valid_and_nan_values(self):
        """Test with mixed valid and NaN values."""
        config = _default_test_config()
        daily_likelihoods = [0.8] * 10 + [float("nan")] * 4
        result = apply_indicator_gate("social_withdrawal", daily_likelihoods, config)

        assert result.days_above_threshold == 10
        # presence_flag is True because 10 >= gate_need (10)
        assert result.presence_flag is True


class TestLogging:
    """Tests for logging behavior."""

    def test_empty_input_logs_warning(self, caplog):
        """Test that empty input logs a warning."""
        import logging

        config = _default_test_config()
        with caplog.at_level(logging.WARNING):
            apply_indicator_gate("test_indicator", [], config)

        assert "No daily likelihoods provided" in caplog.text
        assert "test_indicator" in caplog.text

    def test_nan_values_log_warning(self, caplog):
        """Test that NaN values log a warning."""
        import logging

        config = _default_test_config()
        with caplog.at_level(logging.WARNING):
            apply_indicator_gate("test_indicator", [float("nan")] * 14, config)

        assert "NaN/Infinity values" in caplog.text
        assert "14" in caplog.text

    def test_episode_decision_logs_info(self, caplog):
        """Test that episode decision logs at INFO level."""
        import logging

        config = _default_test_config()
        gate_result = IndicatorGateResult(
            indicator_name="social_withdrawal",
            presence_flag=True,
            days_above_threshold=10,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True,) * 14,
            insufficient_data=False,
        )
        with caplog.at_level(logging.INFO):
            compute_episode_decision({"social_withdrawal": gate_result}, config)

        assert "Episode decision" in caplog.text
        assert "likely=" in caplog.text


# ============================================================================
# Integration Tests: Daily Series to DSM-Gate
# ============================================================================


class TestDSMGateWithDailySeries:
    """Integration tests for DSM-Gate with daily series."""

    @pytest.fixture
    def multi_day_config(self):
        """Create a config with 7-day window and gate_need=5 for easier testing."""
        return _make_config(
            indicators={
                "social_withdrawal": IndicatorConfig(
                    biomarkers={
                        "speech_activity": BiomarkerWeight(
                            weight=0.5, direction="higher_is_worse"
                        ),
                        "connections": BiomarkerWeight(
                            weight=0.5, direction="lower_is_worse"
                        ),
                    },
                    dsm_gate=DSMGateConfig(theta=0.5, m_window=7, gate_need=5),
                ),
                "diminished_interest": IndicatorConfig(
                    biomarkers={
                        "activity_level": BiomarkerWeight(
                            weight=1.0, direction="lower_is_worse"
                        ),
                    },
                    dsm_gate=DSMGateConfig(theta=0.5, m_window=7, gate_need=5),
                ),
            },
            context_weights={},
            dsm_gate_defaults=DSMGateConfig(theta=0.5, m_window=7, gate_need=5),
            episode=EpisodeConfig(
                min_indicators=2,
                core_indicators=("social_withdrawal", "diminished_interest"),
            ),
        )

    def test_daily_flags_track_each_day(self, multi_day_config):
        """Test that daily_flags correctly tracks each day's threshold evaluation."""
        gate = DSMGate(multi_day_config)

        daily_scores = [0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9]
        result = gate.apply_gate("social_withdrawal", daily_scores)

        expected_flags = (True, False, True, False, True, False, True)
        assert result.daily_flags == expected_flags
        assert result.days_above_threshold == 4
        # 4 < 5 (gate_need), so not present
        assert result.presence_flag is False

    def test_episode_with_both_indicators_present(self, multi_day_config):
        """Test episode decision when both indicators pass N-of-M."""
        gate = DSMGate(multi_day_config)

        # Both core indicators above threshold on all 7 days (7 >= 5)
        indicator_daily_scores = {
            "social_withdrawal": [0.6, 0.7, 0.6, 0.7, 0.6, 0.7, 0.6],
            "diminished_interest": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        }

        gate_results = gate.apply_all_gates(indicator_daily_scores)
        episode = gate.compute_episode(gate_results)

        assert episode.episode_likely is True
        assert episode.core_indicator_present is True

    def test_episode_fails_when_indicator_not_present(self, multi_day_config):
        """Test episode fails when an indicator does not pass N-of-M."""
        gate = DSMGate(multi_day_config)

        # social_withdrawal only above threshold on 2 of 7 days (2 < 5)
        indicator_daily_scores = {
            "social_withdrawal": [0.6, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3],
            "diminished_interest": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        }

        gate_results = gate.apply_all_gates(indicator_daily_scores)
        episode = gate.compute_episode(gate_results)

        # Only 1 indicator present, need 2
        assert episode.episode_likely is False
        assert episode.indicators_present == 1

    def test_partial_data_still_computes(self, multi_day_config):
        """Test that partial data (less than M days) still computes."""
        gate = DSMGate(multi_day_config)

        daily_scores = [0.6, 0.7, 0.8, 0.6, 0.7]
        result = gate.apply_gate("social_withdrawal", daily_scores)

        assert result.insufficient_data is True
        assert result.days_evaluated == 5
        assert len(result.daily_flags) == 5
        assert result.days_above_threshold == 5

    def test_single_day_analysis_window(self):
        """Test backwards compatibility: single-day analysis still works."""
        config = _default_test_config()
        gate = DSMGate(config)

        result = gate.apply_gate("social_withdrawal", [0.8])

        assert result.insufficient_data is True
        assert result.days_evaluated == 1
        assert result.days_above_threshold == 1
        assert result.daily_flags == (True,)
