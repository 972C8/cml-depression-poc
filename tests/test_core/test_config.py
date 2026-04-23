"""Tests for src/core/config.py - Analysis Configuration Module."""

from pathlib import Path

import pytest

from src.core.config import (
    AnalysisConfig,
    BaselineDefaults,
    BiomarkerMembershipFunction,
    BiomarkerWeight,
    ConfigurationError,
    ContextAssumptionDef,
    ContextConditionDef,
    DSMGateConfig,
    EMAConfig,
    EpisodeConfig,
    ExperimentContextEvalConfig,
    IndicatorConfig,
    MarkerMembership,
    MarkerMembershipSet,
    MembershipFunction,
    WindowConfig,
    get_default_config,
    load_config,
)


class TestBiomarkerWeight:
    """Tests for BiomarkerWeight model."""

    def test_valid_higher_is_worse(self):
        """Test BiomarkerWeight with higher_is_worse direction."""
        bw = BiomarkerWeight(weight=0.5, direction="higher_is_worse")
        assert bw.weight == 0.5
        assert bw.direction == "higher_is_worse"

    def test_valid_lower_is_worse(self):
        """Test BiomarkerWeight with lower_is_worse direction."""
        bw = BiomarkerWeight(weight=0.3, direction="lower_is_worse")
        assert bw.weight == 0.3
        assert bw.direction == "lower_is_worse"

    def test_invalid_direction_raises(self):
        """Test that invalid direction raises validation error."""
        with pytest.raises(ValueError):
            BiomarkerWeight(weight=0.5, direction="invalid")

    def test_weight_bounds_upper(self):
        """Test that weight > 1.0 raises validation error."""
        with pytest.raises(ValueError):
            BiomarkerWeight(weight=1.5, direction="higher_is_worse")

    def test_weight_bounds_lower(self):
        """Test that weight < 0 raises validation error."""
        with pytest.raises(ValueError):
            BiomarkerWeight(weight=-0.1, direction="higher_is_worse")

    def test_weight_exactly_zero(self):
        """Test that weight=0 is valid."""
        bw = BiomarkerWeight(weight=0.0, direction="higher_is_worse")
        assert bw.weight == 0.0

    def test_weight_exactly_one(self):
        """Test that weight=1.0 is valid."""
        bw = BiomarkerWeight(weight=1.0, direction="lower_is_worse")
        assert bw.weight == 1.0

    def test_immutable(self):
        """Test that BiomarkerWeight is frozen/immutable."""
        bw = BiomarkerWeight(weight=0.5, direction="higher_is_worse")
        with pytest.raises((ValueError, AttributeError)):
            bw.weight = 0.7


class TestBiomarkerMembershipFunction:
    """Tests for BiomarkerMembershipFunction model."""

    def test_triangular_valid(self):
        """Test valid triangular membership function."""
        fn = BiomarkerMembershipFunction(
            type="triangular", params={"l": -2.0, "m": 0.0, "h": 2.0}
        )
        assert fn.type == "triangular"
        assert fn.params == {"l": -2.0, "m": 0.0, "h": 2.0}

    def test_sigmoid_valid(self):
        """Test valid sigmoid membership function."""
        fn = BiomarkerMembershipFunction(type="sigmoid", params={"x0": 1.0, "k": 2.0})
        assert fn.type == "sigmoid"
        assert fn.params == {"x0": 1.0, "k": 2.0}

    def test_exponential_ramp_valid(self):
        """Test valid exponential_ramp membership function."""
        fn = BiomarkerMembershipFunction(
            type="exponential_ramp", params={"tau": 0.5, "lam": 1.5}
        )
        assert fn.type == "exponential_ramp"
        assert fn.params == {"tau": 0.5, "lam": 1.5}

    def test_gaussian_valid(self):
        """Test valid gaussian membership function."""
        fn = BiomarkerMembershipFunction(
            type="gaussian", params={"c": 0.0, "sigma": 1.0}
        )
        assert fn.type == "gaussian"
        assert fn.params == {"c": 0.0, "sigma": 1.0}

    def test_missing_params_raises(self):
        """Test that missing required parameters raises ValueError."""
        with pytest.raises(ValueError, match="Missing required parameters"):
            BiomarkerMembershipFunction(type="triangular", params={"l": -2.0})

    def test_extra_params_allowed(self):
        """Test that extra parameters are allowed."""
        fn = BiomarkerMembershipFunction(
            type="sigmoid", params={"x0": 1.0, "k": 2.0, "extra": 3.0}
        )
        assert fn.params["extra"] == 3.0


class TestBaselineDefaults:
    """Tests for BaselineDefaults model (Story 4.1b)."""

    def test_valid_creation(self):
        """Test BaselineDefaults with valid values."""
        bd = BaselineDefaults(mean=0.5, std=0.15, min_data_points=7)
        assert bd.mean == 0.5
        assert bd.std == 0.15
        assert bd.min_data_points == 7

    def test_default_min_data_points(self):
        """Test that min_data_points defaults to 7."""
        bd = BaselineDefaults(mean=0.5, std=0.15)
        assert bd.min_data_points == 7

    def test_std_must_be_positive(self):
        """Test that std <= 0 raises validation error."""
        with pytest.raises(ValueError):
            BaselineDefaults(mean=0.5, std=0.0)
        with pytest.raises(ValueError):
            BaselineDefaults(mean=0.5, std=-0.1)

    def test_min_data_points_must_be_at_least_one(self):
        """Test that min_data_points < 1 raises validation error."""
        with pytest.raises(ValueError):
            BaselineDefaults(mean=0.5, std=0.15, min_data_points=0)
        with pytest.raises(ValueError):
            BaselineDefaults(mean=0.5, std=0.15, min_data_points=-1)

    def test_min_data_points_exactly_one_valid(self):
        """Test that min_data_points=1 is valid."""
        bd = BaselineDefaults(mean=0.5, std=0.15, min_data_points=1)
        assert bd.min_data_points == 1

    def test_immutable(self):
        """Test that BaselineDefaults is frozen/immutable."""
        bd = BaselineDefaults(mean=0.5, std=0.15)
        with pytest.raises((ValueError, AttributeError)):
            bd.mean = 0.7

    def test_negative_mean_valid(self):
        """Test that negative mean is valid (no constraint)."""
        bd = BaselineDefaults(mean=-0.5, std=0.15)
        assert bd.mean == -0.5


class TestDSMGateConfig:
    """Tests for DSMGateConfig model.

    Story 6.15: n_days removed, validate_window removed.
    DSMGateConfig now only has theta and m_window.
    """

    def test_default_values(self):
        """Test DSMGateConfig with default parameter values."""
        config = DSMGateConfig()
        assert config.theta == 0.5
        assert config.m_window == 14

    def test_custom_values(self):
        """Test DSMGateConfig with custom parameter values."""
        config = DSMGateConfig(theta=0.7, m_window=10)
        assert config.theta == 0.7
        assert config.m_window == 10

    def test_no_n_days_field(self):
        """Test that n_days field no longer exists (Story 6.15)."""
        config = DSMGateConfig()
        assert not hasattr(config, "n_days")

    def test_immutable(self):
        """Test that DSMGateConfig is frozen/immutable."""
        config = DSMGateConfig()
        with pytest.raises((ValueError, AttributeError)):
            config.theta = 0.8


class TestEMAConfig:
    """Tests for EMAConfig model."""

    def test_default_values(self):
        """Test EMAConfig with default parameter values."""
        config = EMAConfig()
        assert config.alpha == 0.3
        assert config.hysteresis == 0.1
        assert config.dwell_time == 2

    def test_custom_values(self):
        """Test EMAConfig with custom parameter values."""
        config = EMAConfig(alpha=0.5, hysteresis=0.05, dwell_time=3)
        assert config.alpha == 0.5
        assert config.hysteresis == 0.05
        assert config.dwell_time == 3

    def test_alpha_upper_bound(self):
        """Test that alpha <= 1.0 is valid."""
        config = EMAConfig(alpha=1.0)
        assert config.alpha == 1.0

    def test_alpha_lower_bound_exclusive(self):
        """Test that alpha > 0 (exclusive) is enforced."""
        with pytest.raises(ValueError):
            EMAConfig(alpha=0.0)

    def test_alpha_exceeds_one_raises(self):
        """Test that alpha > 1.0 raises validation error."""
        with pytest.raises(ValueError):
            EMAConfig(alpha=1.5)

    def test_immutable(self):
        """Test that EMAConfig is frozen/immutable."""
        config = EMAConfig()
        with pytest.raises((ValueError, AttributeError)):
            config.alpha = 0.5


# =============================================================================
# Story 6.13: ExperimentContextEvalConfig Tests
# =============================================================================


class TestMarkerMembershipSet:
    """Tests for MarkerMembershipSet model (Story 6.13)."""

    def test_triangular_valid(self):
        """Test MarkerMembershipSet with valid triangular params."""
        mset = MarkerMembershipSet(type="triangular", params=[0.0, 0.5, 1.0])
        assert mset.type == "triangular"
        assert mset.params == [0.0, 0.5, 1.0]

    def test_trapezoidal_valid(self):
        """Test MarkerMembershipSet with valid trapezoidal params."""
        mset = MarkerMembershipSet(type="trapezoidal", params=[0.0, 0.3, 0.7, 1.0])
        assert mset.type == "trapezoidal"
        assert mset.params == [0.0, 0.3, 0.7, 1.0]

    def test_triangular_wrong_params_count_raises(self):
        """Test that triangular with wrong param count raises error."""
        with pytest.raises(ValueError, match="3 parameters"):
            MarkerMembershipSet(type="triangular", params=[0.0, 0.5, 0.7, 1.0])

    def test_trapezoidal_wrong_params_count_raises(self):
        """Test that trapezoidal with wrong param count raises error."""
        with pytest.raises(ValueError, match="4 parameters"):
            MarkerMembershipSet(type="trapezoidal", params=[0.0, 0.5, 1.0])

    def test_immutable(self):
        """Test that MarkerMembershipSet is frozen/immutable."""
        mset = MarkerMembershipSet(type="triangular", params=[0.0, 0.5, 1.0])
        with pytest.raises((ValueError, AttributeError)):
            mset.type = "trapezoidal"


class TestMarkerMembership:
    """Tests for MarkerMembership model (Story 6.13)."""

    def test_valid_with_multiple_sets(self):
        """Test MarkerMembership with multiple fuzzy sets."""
        mm = MarkerMembership(
            sets={
                "low": MarkerMembershipSet(type="triangular", params=[0.0, 0.0, 0.5]),
                "high": MarkerMembershipSet(type="triangular", params=[0.5, 1.0, 1.0]),
            }
        )
        assert len(mm.sets) == 2
        assert "low" in mm.sets
        assert "high" in mm.sets

    def test_immutable(self):
        """Test that MarkerMembership is frozen/immutable."""
        mm = MarkerMembership(
            sets={"low": MarkerMembershipSet(type="triangular", params=[0.0, 0.0, 0.5])}
        )
        with pytest.raises((ValueError, AttributeError)):
            mm.sets = {}


class TestContextConditionDef:
    """Tests for ContextConditionDef model (Story 6.13)."""

    def test_valid_condition(self):
        """Test valid ContextConditionDef."""
        cond = ContextConditionDef(
            marker="people_in_room", fuzzy_set="high", weight=0.6
        )
        assert cond.marker == "people_in_room"
        assert cond.fuzzy_set == "high"
        assert cond.weight == 0.6

    def test_alias_set_to_fuzzy_set(self):
        """Test that 'set' alias works for fuzzy_set field."""
        # Using alias for compatibility with YAML format
        cond = ContextConditionDef(marker="test", set="high", weight=0.5)
        assert cond.fuzzy_set == "high"

    def test_weight_bounds_valid(self):
        """Test weight boundary values."""
        cond1 = ContextConditionDef(marker="test", fuzzy_set="low", weight=0.0)
        assert cond1.weight == 0.0
        cond2 = ContextConditionDef(marker="test", fuzzy_set="high", weight=1.0)
        assert cond2.weight == 1.0

    def test_weight_out_of_bounds_raises(self):
        """Test that weight > 1.0 raises error."""
        with pytest.raises(ValueError):
            ContextConditionDef(marker="test", fuzzy_set="high", weight=1.5)

    def test_immutable(self):
        """Test that ContextConditionDef is frozen/immutable."""
        cond = ContextConditionDef(marker="test", fuzzy_set="high", weight=0.5)
        with pytest.raises((ValueError, AttributeError)):
            cond.weight = 0.8


class TestContextAssumptionDef:
    """Tests for ContextAssumptionDef model (Story 6.13)."""

    def test_valid_assumption_weights_sum_to_one(self):
        """Test valid assumption with weights summing to 1.0."""
        assumption = ContextAssumptionDef(
            conditions=[
                ContextConditionDef(
                    marker="people_in_room", fuzzy_set="high", weight=0.6
                ),
                ContextConditionDef(
                    marker="ambient_noise", fuzzy_set="moderate", weight=0.4
                ),
            ],
            operator="WEIGHTED_MEAN",
        )
        assert len(assumption.conditions) == 2
        assert assumption.operator == "WEIGHTED_MEAN"

    def test_weights_not_sum_to_one_raises(self):
        """Test that weights not summing to 1.0 raises error."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ContextAssumptionDef(
                conditions=[
                    ContextConditionDef(marker="test1", fuzzy_set="high", weight=0.3),
                    ContextConditionDef(marker="test2", fuzzy_set="low", weight=0.3),
                ],
            )

    def test_empty_conditions_allowed(self):
        """Test that empty conditions list is allowed (for threshold-only)."""
        assumption = ContextAssumptionDef(conditions=[], operator="WEIGHTED_MEAN")
        assert len(assumption.conditions) == 0

    def test_default_operator_is_weighted_mean(self):
        """Test that the default operator is WEIGHTED_MEAN."""
        assumption = ContextAssumptionDef(
            conditions=[
                ContextConditionDef(marker="test", fuzzy_set="high", weight=1.0),
            ],
        )
        assert assumption.operator == "WEIGHTED_MEAN"

    def test_immutable(self):
        """Test that ContextAssumptionDef is frozen/immutable."""
        assumption = ContextAssumptionDef(
            conditions=[
                ContextConditionDef(marker="test", fuzzy_set="high", weight=1.0),
            ],
        )
        with pytest.raises((ValueError, AttributeError)):
            assumption.operator = "WEIGHTED_MEAN"


class TestExperimentContextEvalConfig:
    """Tests for ExperimentContextEvalConfig model (Story 6.13)."""

    def test_valid_config(self):
        """Test valid ExperimentContextEvalConfig."""
        config = ExperimentContextEvalConfig(
            marker_memberships={
                "people_in_room": MarkerMembership(
                    sets={
                        "low": MarkerMembershipSet(type="triangular", params=[0, 0, 2]),
                        "high": MarkerMembershipSet(
                            type="triangular", params=[3, 5, 5]
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
                ),
            },
            neutral_threshold=0.3,
            ema=EMAConfig(alpha=0.3, hysteresis=0.1, dwell_time=2),
        )
        assert "people_in_room" in config.marker_memberships
        assert "solitary_digital" in config.context_assumptions
        assert config.neutral_threshold == 0.3
        assert config.ema.alpha == 0.3

    def test_neutral_threshold_default(self):
        """Test that neutral_threshold defaults to 0.3."""
        config = ExperimentContextEvalConfig(
            marker_memberships={},
            context_assumptions={},
            ema=EMAConfig(),
        )
        assert config.neutral_threshold == 0.3

    def test_neutral_threshold_bounds(self):
        """Test neutral_threshold boundary values."""
        config1 = ExperimentContextEvalConfig(
            marker_memberships={},
            context_assumptions={},
            neutral_threshold=0.0,
            ema=EMAConfig(),
        )
        assert config1.neutral_threshold == 0.0

        config2 = ExperimentContextEvalConfig(
            marker_memberships={},
            context_assumptions={},
            neutral_threshold=1.0,
            ema=EMAConfig(),
        )
        assert config2.neutral_threshold == 1.0

    def test_neutral_threshold_out_of_bounds_raises(self):
        """Test that neutral_threshold > 1.0 raises error."""
        with pytest.raises(ValueError):
            ExperimentContextEvalConfig(
                marker_memberships={},
                context_assumptions={},
                neutral_threshold=1.5,
                ema=EMAConfig(),
            )

    def test_immutable(self):
        """Test that ExperimentContextEvalConfig is frozen/immutable."""
        config = ExperimentContextEvalConfig(
            marker_memberships={},
            context_assumptions={},
            ema=EMAConfig(),
        )
        with pytest.raises((ValueError, AttributeError)):
            config.neutral_threshold = 0.5


class TestGetDefaultConfigContextEvaluation:
    """Tests for context_evaluation field in get_default_config (Story 6.13)."""

    def test_has_context_evaluation(self):
        """Test that default config has context_evaluation field."""
        config = get_default_config()
        assert config.context_evaluation is not None
        assert isinstance(config.context_evaluation, ExperimentContextEvalConfig)

    def test_context_evaluation_has_marker_memberships(self):
        """Test that context_evaluation has marker_memberships."""
        config = get_default_config()
        assert len(config.context_evaluation.marker_memberships) > 0
        assert "people_in_room" in config.context_evaluation.marker_memberships

    def test_context_evaluation_has_context_assumptions(self):
        """Test that context_evaluation has context_assumptions."""
        config = get_default_config()
        assert len(config.context_evaluation.context_assumptions) > 0
        assert "solitary_digital" in config.context_evaluation.context_assumptions

    def test_context_evaluation_has_ema(self):
        """Test that context_evaluation contains EMA config."""
        config = get_default_config()
        assert config.context_evaluation.ema is not None
        assert config.context_evaluation.ema.alpha == 0.3
        assert config.context_evaluation.ema.hysteresis == 0.05
        assert config.context_evaluation.ema.dwell_time == 1

    def test_context_evaluation_marker_membership_structure(self):
        """Test that marker_memberships have correct structure."""
        config = get_default_config()
        people_in_room = config.context_evaluation.marker_memberships["people_in_room"]
        assert "low" in people_in_room.sets
        assert "high" in people_in_room.sets
        assert people_in_room.sets["low"].type in ["triangular", "trapezoidal"]


class TestEpisodeConfig:
    """Tests for EpisodeConfig model (Story 4.9)."""

    def test_default_values(self):
        """Test EpisodeConfig with default parameter values."""
        config = EpisodeConfig()
        assert config.min_indicators == 5
        assert config.core_indicators == ("1_depressed_mood", "2_loss_of_interest")

    def test_custom_min_indicators(self):
        """Test EpisodeConfig with custom min_indicators."""
        config = EpisodeConfig(min_indicators=3)
        assert config.min_indicators == 3

    def test_custom_core_indicators(self):
        """Test EpisodeConfig with custom core_indicators."""
        config = EpisodeConfig(core_indicators=("indicator_a", "indicator_b"))
        assert config.core_indicators == ("indicator_a", "indicator_b")

    def test_core_indicators_list_converted_to_tuple(self):
        """Test that core_indicators list is converted to tuple."""
        config = EpisodeConfig(core_indicators=["a", "b", "c"])
        assert config.core_indicators == ("a", "b", "c")
        assert isinstance(config.core_indicators, tuple)

    def test_min_indicators_lower_bound(self):
        """Test that min_indicators >= 1 is enforced."""
        with pytest.raises(ValueError):
            EpisodeConfig(min_indicators=0)

    def test_min_indicators_upper_bound(self):
        """Test that min_indicators <= 9 is enforced."""
        with pytest.raises(ValueError):
            EpisodeConfig(min_indicators=10)

    def test_min_indicators_boundary_one(self):
        """Test that min_indicators=1 is valid."""
        config = EpisodeConfig(min_indicators=1)
        assert config.min_indicators == 1

    def test_min_indicators_boundary_nine(self):
        """Test that min_indicators=9 is valid."""
        config = EpisodeConfig(min_indicators=9)
        assert config.min_indicators == 9

    def test_immutable(self):
        """Test that EpisodeConfig is frozen/immutable."""
        config = EpisodeConfig()
        with pytest.raises((ValueError, AttributeError)):
            config.min_indicators = 3

    def test_empty_core_indicators_allowed(self):
        """Test that empty core_indicators tuple is allowed."""
        config = EpisodeConfig(core_indicators=())
        assert config.core_indicators == ()


class TestWindowConfig:
    """Tests for WindowConfig model (Story 6.2)."""

    def test_default_values(self):
        """Test WindowConfig with default parameter values."""
        config = WindowConfig()
        assert config.size_minutes == 15
        assert config.aggregation_method == "mean"
        assert config.min_readings == 1

    def test_custom_window_size(self):
        """Test WindowConfig with custom window size."""
        config = WindowConfig(size_minutes=30)
        assert config.size_minutes == 30

    def test_custom_aggregation_method(self):
        """Test WindowConfig with custom aggregation method."""
        config = WindowConfig(aggregation_method="median")
        assert config.aggregation_method == "median"

    def test_custom_min_readings(self):
        """Test WindowConfig with custom min_readings."""
        config = WindowConfig(min_readings=3)
        assert config.min_readings == 3

    @pytest.mark.parametrize("size", [5, 10, 15, 30, 60])
    def test_valid_window_sizes(self, size):
        """Test all valid window sizes."""
        config = WindowConfig(size_minutes=size)
        assert config.size_minutes == size

    def test_invalid_window_size_raises(self):
        """Test that invalid window sizes raise validation error."""
        with pytest.raises(ValueError):
            WindowConfig(size_minutes=7)

    def test_invalid_window_size_zero_raises(self):
        """Test that window size of 0 raises validation error."""
        with pytest.raises(ValueError):
            WindowConfig(size_minutes=0)

    def test_invalid_window_size_negative_raises(self):
        """Test that negative window sizes raise validation error."""
        with pytest.raises(ValueError):
            WindowConfig(size_minutes=-15)

    @pytest.mark.parametrize("method", ["mean", "median", "max", "min"])
    def test_valid_aggregation_methods(self, method):
        """Test all valid aggregation methods."""
        config = WindowConfig(aggregation_method=method)
        assert config.aggregation_method == method

    def test_invalid_aggregation_method_raises(self):
        """Test that invalid aggregation methods raise validation error."""
        with pytest.raises(ValueError):
            WindowConfig(aggregation_method="invalid")

    def test_min_readings_lower_bound(self):
        """Test that min_readings >= 1 is enforced."""
        with pytest.raises(ValueError):
            WindowConfig(min_readings=0)

    def test_min_readings_boundary_one(self):
        """Test that min_readings=1 is valid."""
        config = WindowConfig(min_readings=1)
        assert config.min_readings == 1

    def test_immutable(self):
        """Test that WindowConfig is frozen/immutable."""
        config = WindowConfig()
        with pytest.raises((ValueError, AttributeError)):
            config.size_minutes = 30


class TestMembershipFunction:
    """Tests for MembershipFunction model."""

    def test_triangular_valid(self):
        """Test MembershipFunction with triangular type and 3 params."""
        mf = MembershipFunction(type="triangular", params=[0.2, 0.5, 0.8])
        assert mf.type == "triangular"
        assert mf.params == [0.2, 0.5, 0.8]

    def test_trapezoidal_valid(self):
        """Test MembershipFunction with trapezoidal type and 4 params."""
        mf = MembershipFunction(type="trapezoidal", params=[0.1, 0.3, 0.6, 0.9])
        assert mf.type == "trapezoidal"
        assert mf.params == [0.1, 0.3, 0.6, 0.9]

    def test_triangular_wrong_param_count_raises(self):
        """Test that triangular with != 3 params raises validation error."""
        with pytest.raises(ValueError, match="triangular.*3 parameters"):
            MembershipFunction(type="triangular", params=[0.2, 0.5])

    def test_trapezoidal_wrong_param_count_raises(self):
        """Test that trapezoidal with != 4 params raises validation error."""
        with pytest.raises(ValueError, match="trapezoidal.*4 parameters"):
            MembershipFunction(type="trapezoidal", params=[0.1, 0.3, 0.6])

    def test_immutable(self):
        """Test that MembershipFunction is frozen/immutable."""
        mf = MembershipFunction(type="triangular", params=[0.2, 0.5, 0.8])
        with pytest.raises((ValueError, AttributeError)):
            mf.type = "trapezoidal"


class TestIndicatorConfig:
    """Tests for IndicatorConfig with weight validation."""

    def test_weights_sum_to_one(self):
        """Test IndicatorConfig with weights that sum to 1.0."""
        config = IndicatorConfig(
            biomarkers={
                "metric_a": BiomarkerWeight(weight=0.6, direction="higher_is_worse"),
                "metric_b": BiomarkerWeight(weight=0.4, direction="lower_is_worse"),
            }
        )
        assert len(config.biomarkers) == 2
        assert config.biomarkers["metric_a"].weight == 0.6
        assert config.biomarkers["metric_b"].weight == 0.4

    def test_weights_sum_within_tolerance(self):
        """Test that weights within tolerance (0.999-1.001) are accepted."""
        config = IndicatorConfig(
            biomarkers={
                "a": BiomarkerWeight(weight=0.3333, direction="higher_is_worse"),
                "b": BiomarkerWeight(weight=0.3333, direction="higher_is_worse"),
                "c": BiomarkerWeight(weight=0.3334, direction="lower_is_worse"),
            }
        )
        assert len(config.biomarkers) == 3

    def test_weights_not_sum_to_one_raises(self):
        """Test that weights not summing to 1.0 raises validation error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            IndicatorConfig(
                biomarkers={
                    "metric_a": BiomarkerWeight(
                        weight=0.5, direction="higher_is_worse"
                    ),
                    "metric_b": BiomarkerWeight(weight=0.4, direction="lower_is_worse"),
                }
            )

    def test_with_dsm_gate_override(self):
        """Test IndicatorConfig with optional DSM-gate override."""
        config = IndicatorConfig(
            biomarkers={
                "metric_a": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
            },
            dsm_gate=DSMGateConfig(theta=0.7, m_window=10),
        )
        assert config.dsm_gate is not None
        assert config.dsm_gate.theta == 0.7

    def test_without_dsm_gate_override(self):
        """Test IndicatorConfig without DSM-gate override (None)."""
        config = IndicatorConfig(
            biomarkers={
                "metric_a": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
            }
        )
        assert config.dsm_gate is None

    def test_min_biomarkers_default(self):
        """Test that min_biomarkers defaults to 1."""
        config = IndicatorConfig(
            biomarkers={
                "metric_a": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
            }
        )
        assert config.min_biomarkers == 1

    def test_min_biomarkers_custom(self):
        """Test IndicatorConfig with custom min_biomarkers."""
        config = IndicatorConfig(
            biomarkers={
                "a": BiomarkerWeight(weight=0.5, direction="higher_is_worse"),
                "b": BiomarkerWeight(weight=0.5, direction="lower_is_worse"),
            },
            min_biomarkers=2,
        )
        assert config.min_biomarkers == 2

    def test_immutable(self):
        """Test that IndicatorConfig is frozen/immutable."""
        config = IndicatorConfig(
            biomarkers={
                "metric_a": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
            }
        )
        with pytest.raises((ValueError, AttributeError)):
            config.min_biomarkers = 2


class TestAnalysisConfigBiomarkerDefaults:
    """Tests for AnalysisConfig biomarker_defaults field (Story 4.1b)."""

    def test_biomarker_defaults_empty_dict_valid(self):
        """Test that biomarker_defaults can be empty dict."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            biomarker_defaults={},
        )
        assert config.biomarker_defaults == {}

    def test_biomarker_defaults_default_is_empty_dict(self):
        """Test that biomarker_defaults defaults to empty dict."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
        )
        assert config.biomarker_defaults == {}

    def test_biomarker_defaults_with_values(self):
        """Test biomarker_defaults with BaselineDefaults values."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            biomarker_defaults={
                "speech_activity": BaselineDefaults(
                    mean=0.5, std=0.15, min_data_points=7
                ),
                "voice_energy": BaselineDefaults(mean=0.55, std=0.18),
            },
        )
        assert "speech_activity" in config.biomarker_defaults
        assert "voice_energy" in config.biomarker_defaults
        assert config.biomarker_defaults["speech_activity"].mean == 0.5
        assert config.biomarker_defaults["voice_energy"].std == 0.18


class TestAnalysisConfigTimezone:
    """Tests for AnalysisConfig timezone field (Story 4.1b)."""

    def test_timezone_default_europe_zurich(self):
        """Test that timezone defaults to Europe/Zurich."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
        )
        assert config.timezone == "Europe/Zurich"

    def test_valid_timezone_europe_zurich(self):
        """Test AnalysisConfig with valid Europe/Zurich timezone."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            timezone="Europe/Zurich",
        )
        assert config.timezone == "Europe/Zurich"

    def test_valid_timezone_america_new_york(self):
        """Test AnalysisConfig with valid America/New_York timezone."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            timezone="America/New_York",
        )
        assert config.timezone == "America/New_York"

    def test_invalid_timezone_raises(self):
        """Test that invalid timezone string raises validation error."""
        with pytest.raises(ValueError, match="Invalid timezone"):
            AnalysisConfig(
                indicators={
                    "test": IndicatorConfig(
                        biomarkers={
                            "m": BiomarkerWeight(
                                weight=1.0, direction="higher_is_worse"
                            )
                        }
                    )
                },
                context_weights={},
                timezone="Invalid/Timezone",
            )

    def test_invalid_timezone_garbage_raises(self):
        """Test that garbage timezone string raises validation error."""
        with pytest.raises(ValueError, match="Invalid timezone"):
            AnalysisConfig(
                indicators={
                    "test": IndicatorConfig(
                        biomarkers={
                            "m": BiomarkerWeight(
                                weight=1.0, direction="higher_is_worse"
                            )
                        }
                    )
                },
                context_weights={},
                timezone="not-a-timezone",
            )


class TestAnalysisConfig:
    """Tests for AnalysisConfig main configuration class."""

    def test_minimal_config(self):
        """Test AnalysisConfig with minimal required fields."""
        config = AnalysisConfig(
            indicators={
                "test_indicator": IndicatorConfig(
                    biomarkers={
                        "metric_a": BiomarkerWeight(
                            weight=1.0, direction="higher_is_worse"
                        )
                    }
                )
            },
            context_weights={},
        )
        assert len(config.indicators) == 1

    def test_full_config(self):
        """Test AnalysisConfig with all fields populated."""
        config = AnalysisConfig(
            indicators={
                "indicator_1": IndicatorConfig(
                    biomarkers={
                        "a": BiomarkerWeight(weight=0.5, direction="higher_is_worse"),
                        "b": BiomarkerWeight(weight=0.5, direction="lower_is_worse"),
                    }
                )
            },
            context_weights={"social": {"a": 1.5, "b": 0.7}},
            membership_functions={
                "low": MembershipFunction(type="triangular", params=[0.0, 0.0, 0.5])
            },
            dsm_gate_defaults=DSMGateConfig(theta=0.6, m_window=21),
        )
        assert config.dsm_gate_defaults.theta == 0.6
        # Story 6.13: EMA is now part of context_evaluation
        assert config.context_evaluation.ema is not None

    def test_to_dict_method(self):
        """Test to_dict() serialization method."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert "indicators" in result
        assert "context_weights" in result

    def test_context_weights_positive_multipliers(self):
        """Test that context weight multipliers are positive."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={"social": {"metric_a": 1.5, "metric_b": 0.8}},
        )
        assert config.context_weights["social"]["metric_a"] == 1.5

    def test_context_weights_negative_multiplier_raises(self):
        """Test that negative context weight multipliers raise validation error."""
        with pytest.raises(ValueError, match="positive"):
            AnalysisConfig(
                indicators={
                    "test": IndicatorConfig(
                        biomarkers={
                            "m": BiomarkerWeight(
                                weight=1.0, direction="higher_is_worse"
                            )
                        }
                    )
                },
                context_weights={"social": {"metric_a": -0.5}},
            )

    def test_immutable(self):
        """Test that AnalysisConfig is frozen/immutable."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
        )
        with pytest.raises((ValueError, AttributeError)):
            config.timezone = "Europe/London"

    def test_episode_default(self):
        """Test that episode defaults to EpisodeConfig() (Story 4.9)."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
        )
        assert config.episode is not None
        assert config.episode.min_indicators == 5
        assert config.episode.core_indicators == (
            "1_depressed_mood",
            "2_loss_of_interest",
        )

    def test_episode_custom(self):
        """Test AnalysisConfig with custom episode configuration (Story 4.9)."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            episode=EpisodeConfig(min_indicators=3, core_indicators=("a", "b")),
        )
        assert config.episode.min_indicators == 3
        assert config.episode.core_indicators == ("a", "b")


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML configuration file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            """
indicators:
  test_indicator:
    biomarkers:
      metric_a:
        weight: 0.6
        direction: higher_is_worse
      metric_b:
        weight: 0.4
        direction: lower_is_worse
context_weights:
  social:
    metric_a: 1.2
"""
        )
        config = load_config(config_file)
        assert isinstance(config, AnalysisConfig)
        assert "test_indicator" in config.indicators

    def test_load_valid_json(self, tmp_path):
        """Test loading a valid JSON configuration file."""
        config_file = tmp_path / "test_config.json"
        config_file.write_text(
            """
{
  "indicators": {
    "test": {
      "biomarkers": {
        "m": {"weight": 1.0, "direction": "higher_is_worse"}
      }
    }
  },
  "context_weights": {}
}
"""
        )
        config = load_config(config_file)
        assert isinstance(config, AnalysisConfig)
        assert "test" in config.indicators

    def test_load_nonexistent_file_raises(self):
        """Test that loading non-existent file raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_config("/nonexistent/path.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        """Test that loading invalid YAML raises ConfigurationError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: :")
        with pytest.raises(ConfigurationError):
            load_config(config_file)

    def test_load_validation_failure_raises(self, tmp_path):
        """Test that validation failures raise ConfigurationError."""
        config_file = tmp_path / "bad_validation.yaml"
        config_file.write_text(
            """
indicators:
  test:
    biomarkers:
      m:
        weight: 0.5
        direction: higher_is_worse
context_weights: {}
"""
        )
        with pytest.raises(ConfigurationError, match="must sum to 1.0"):
            load_config(config_file)

    def test_load_with_path_object(self, tmp_path):
        """Test load_config accepts Path objects."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
indicators:
  t:
    biomarkers:
      m: {weight: 1.0, direction: higher_is_worse}
context_weights: {}
"""
        )
        config = load_config(Path(config_file))
        assert isinstance(config, AnalysisConfig)

    def test_load_config_with_timezone_and_biomarker_defaults(self, tmp_path):
        """Test load_config with new fields: timezone and biomarker_defaults (Story 4.1b)."""
        config_file = tmp_path / "test_new_fields.yaml"
        config_file.write_text(
            """
timezone: "America/Los_Angeles"
biomarker_defaults:
  speech_activity:
    mean: 0.5
    std: 0.15
    min_data_points: 10
  voice_energy:
    mean: 0.6
    std: 0.2
indicators:
  test_indicator:
    biomarkers:
      metric_a:
        weight: 1.0
        direction: higher_is_worse
context_weights: {}
"""
        )
        config = load_config(config_file)
        assert config.timezone == "America/Los_Angeles"
        assert len(config.biomarker_defaults) == 2
        assert config.biomarker_defaults["speech_activity"].mean == 0.5
        assert config.biomarker_defaults["speech_activity"].min_data_points == 10
        assert config.biomarker_defaults["voice_energy"].std == 0.2

    def test_load_config_backward_compatible_without_new_fields(self, tmp_path):
        """Test load_config still works without timezone/biomarker_defaults (backward compat)."""
        config_file = tmp_path / "old_config.yaml"
        config_file.write_text(
            """
indicators:
  test:
    biomarkers:
      m: {weight: 1.0, direction: higher_is_worse}
context_weights: {}
"""
        )
        config = load_config(config_file)
        # Should use defaults
        assert config.timezone == "Europe/Zurich"
        assert config.biomarker_defaults == {}

    def test_load_config_with_episode_section(self, tmp_path):
        """Test load_config with episode section (Story 4.9)."""
        config_file = tmp_path / "test_episode.yaml"
        config_file.write_text(
            """
indicators:
  test:
    biomarkers:
      m: {weight: 1.0, direction: higher_is_worse}
context_weights: {}
episode:
  min_indicators: 3
  core_indicators:
    - indicator_a
    - indicator_b
"""
        )
        config = load_config(config_file)
        assert config.episode.min_indicators == 3
        assert config.episode.core_indicators == ("indicator_a", "indicator_b")

    def test_load_config_backward_compatible_without_episode(self, tmp_path):
        """Test load_config works without episode section (backward compat) (Story 4.9)."""
        config_file = tmp_path / "no_episode.yaml"
        config_file.write_text(
            """
indicators:
  test:
    biomarkers:
      m: {weight: 1.0, direction: higher_is_worse}
context_weights: {}
"""
        )
        config = load_config(config_file)
        # Should use default EpisodeConfig
        assert config.episode.min_indicators == 5
        assert config.episode.core_indicators == (
            "1_depressed_mood",
            "2_loss_of_interest",
        )


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_valid_config(self):
        """Test that get_default_config returns a valid AnalysisConfig."""
        config = get_default_config()
        assert isinstance(config, AnalysisConfig)

    def test_has_timezone(self):
        """Test that default config has timezone set (Story 4.1b)."""
        config = get_default_config()
        assert config.timezone == "Europe/Zurich"

    def test_has_mdd_indicators(self):
        """Test that default config includes MDD-focused indicators."""
        config = get_default_config()
        assert "1_depressed_mood" in config.indicators
        assert "2_loss_of_interest" in config.indicators
        assert "4_insomnia_hypersomnia" in config.indicators

    def test_all_indicators_valid(self):
        """Test that all default indicators have valid configurations."""
        config = get_default_config()
        assert len(config.indicators) > 0
        for indicator_config in config.indicators.values():
            # Check weights sum to 1.0 (validation should pass)
            total = sum(bw.weight for bw in indicator_config.biomarkers.values())
            assert 0.999 <= total <= 1.001

    def test_has_context_weights(self):
        """Test that default config includes context weights."""
        config = get_default_config()
        assert len(config.context_weights) > 0
        assert "solitary_digital" in config.context_weights

    def test_has_dsm_gate_defaults(self):
        """Test that default config has DSM-gate defaults."""
        config = get_default_config()
        assert config.dsm_gate_defaults is not None
        assert config.dsm_gate_defaults.theta == 0.6
        assert config.dsm_gate_defaults.m_window == 14

    def test_has_ema_config(self):
        """Test that default config has EMA configuration (Story 6.13: in context_evaluation)."""
        config = get_default_config()
        # Story 6.13: EMA is now nested under context_evaluation
        assert config.context_evaluation is not None
        assert config.context_evaluation.ema is not None
        assert config.context_evaluation.ema.alpha == 0.3

    def test_has_episode_config(self):
        """Test that default config has episode configuration (Story 4.9)."""
        config = get_default_config()
        assert config.episode is not None
        assert config.episode.min_indicators == 5
        assert "1_depressed_mood" in config.episode.core_indicators
        assert "2_loss_of_interest" in config.episode.core_indicators


class TestLoadConfigEdgeCases:
    """Tests for load_config edge cases discovered in code review."""

    def test_load_empty_file_raises(self, tmp_path):
        """Test that loading empty config file raises clear ConfigurationError."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        with pytest.raises(ConfigurationError, match="empty"):
            load_config(config_file)

    def test_load_comment_only_file_raises(self, tmp_path):
        """Test that file with only comments raises ConfigurationError."""
        config_file = tmp_path / "comments.yaml"
        config_file.write_text("# This is just a comment\n# Nothing else")
        with pytest.raises(ConfigurationError, match="empty"):
            load_config(config_file)

    def test_load_unsupported_extension_raises(self, tmp_path):
        """Test that unsupported file extension raises ConfigurationError."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("indicators: {}")
        with pytest.raises(ConfigurationError, match="Unsupported file extension"):
            load_config(config_file)

    def test_load_valid_yml_extension(self, tmp_path):
        """Test that .yml extension is supported (not just .yaml)."""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(
            """
indicators:
  test:
    biomarkers:
      m:
        weight: 1.0
        direction: higher_is_worse
context_weights: {}
"""
        )
        config = load_config(config_file)
        assert isinstance(config, AnalysisConfig)
        assert "test" in config.indicators


class TestContextWeightsEdgeCases:
    """Tests for context weight edge cases."""

    def test_context_weights_zero_multiplier_raises(self):
        """Test that zero context weight multiplier raises validation error."""
        with pytest.raises(ValueError, match="positive"):
            AnalysisConfig(
                indicators={
                    "test": IndicatorConfig(
                        biomarkers={
                            "m": BiomarkerWeight(
                                weight=1.0, direction="higher_is_worse"
                            )
                        }
                    )
                },
                context_weights={"social": {"metric_a": 0.0}},
            )


class TestSerializationRoundTrip:
    """Tests for configuration serialization and deserialization."""

    def test_to_dict_round_trip(self):
        """Test that config can be serialized and deserialized correctly."""
        import json

        original = get_default_config()
        data = original.to_dict()

        # Verify it's JSON-serializable
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)

        # Verify it can be deserialized back to AnalysisConfig
        restored = AnalysisConfig(**restored_data)

        # Verify key properties match
        assert list(original.indicators.keys()) == list(restored.indicators.keys())
        assert original.dsm_gate_defaults.theta == restored.dsm_gate_defaults.theta

    def test_to_dict_includes_timezone(self):
        """Test that to_dict() serializes timezone correctly (Story 4.1b)."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            timezone="Europe/Zurich",
        )
        data = config.to_dict()
        assert "timezone" in data
        assert data["timezone"] == "Europe/Zurich"

    def test_to_dict_includes_biomarker_defaults(self):
        """Test that to_dict() serializes biomarker_defaults correctly (Story 4.1b)."""
        config = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            biomarker_defaults={
                "speech_activity": BaselineDefaults(
                    mean=0.5, std=0.15, min_data_points=7
                )
            },
        )
        data = config.to_dict()
        assert "biomarker_defaults" in data
        assert "speech_activity" in data["biomarker_defaults"]
        assert data["biomarker_defaults"]["speech_activity"]["mean"] == 0.5
        assert data["biomarker_defaults"]["speech_activity"]["std"] == 0.15
        assert data["biomarker_defaults"]["speech_activity"]["min_data_points"] == 7

    def test_to_dict_round_trip_with_new_fields(self):
        """Test round-trip with timezone and biomarker_defaults (Story 4.1b)."""
        import json

        original = AnalysisConfig(
            indicators={
                "test": IndicatorConfig(
                    biomarkers={
                        "m": BiomarkerWeight(weight=1.0, direction="higher_is_worse")
                    }
                )
            },
            context_weights={},
            timezone="America/New_York",
            biomarker_defaults={
                "speech_activity": BaselineDefaults(
                    mean=0.5, std=0.15, min_data_points=10
                )
            },
        )
        data = original.to_dict()

        # Round-trip via JSON
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = AnalysisConfig(**restored_data)

        # Verify new fields preserved
        assert restored.timezone == "America/New_York"
        assert "speech_activity" in restored.biomarker_defaults
        assert restored.biomarker_defaults["speech_activity"].mean == 0.5
        assert restored.biomarker_defaults["speech_activity"].min_data_points == 10

    def test_to_dict_yaml_round_trip(self, tmp_path):
        """Test that config can be written to YAML and re-loaded."""
        import yaml

        original = get_default_config()
        data = original.to_dict()

        # Write to YAML
        config_file = tmp_path / "round_trip.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(data, f)

        # Re-load
        restored = load_config(config_file)

        # Verify key properties match (use set for order-independent comparison)
        assert set(original.indicators.keys()) == set(restored.indicators.keys())
