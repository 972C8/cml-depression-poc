"""Unit tests for context evaluator.

Tests ContextEvaluator, ContextResult, and context assumptions.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.config import MembershipFunction
from src.core.context.evaluator import (
    ContextAssumption,
    ContextAssumptionConfig,
    ContextCondition,
    ContextEvaluationConfig,
    ContextEvaluator,
    ContextResult,
    MarkerMembershipConfig,
    get_default_context_config,
    load_context_config,
)
from src.core.data_reader import ContextRecord


class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_context_result_contains_all_required_fields(self) -> None:
        """Test ContextResult contains all required fields."""
        result = ContextResult(
            active_context="solitary_digital",
            confidence_scores={"solitary_digital": 0.8, "neutral": 0.2},
            raw_scores={"solitary_digital": 0.75, "neutral": 0.25},
            smoothed=True,
            markers_used=("people_in_room", "ambient_noise"),
            markers_missing=("network_activity_level",),
            timestamp=datetime.now(UTC),
        )

        assert result.active_context == "solitary_digital"
        assert "solitary_digital" in result.confidence_scores
        assert "neutral" in result.confidence_scores
        assert result.smoothed is True
        assert "people_in_room" in result.markers_used
        assert "network_activity_level" in result.markers_missing
        assert result.timestamp is not None

    def test_context_result_is_frozen(self) -> None:
        """Test ContextResult is immutable (frozen dataclass)."""
        result = ContextResult(
            active_context="solitary_digital",
            confidence_scores={"solitary_digital": 0.8},
            raw_scores={"solitary_digital": 0.8},
            smoothed=True,
            markers_used=("people_in_room",),
            markers_missing=(),
            timestamp=datetime.now(UTC),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.active_context = "other"


class TestContextAssumption:
    """Tests for ContextAssumption evaluation."""

    def test_solitary_digital_context_low_people_high_network(self) -> None:
        """Test solitary_digital context detected with low people, high network."""
        assumption = ContextAssumption(
            name="solitary_digital",
            conditions={
                "people_in_room": ("low", 0.4),
                "network_activity_level": ("high", 0.6),
            },
            operator="WEIGHTED_MEAN",
        )

        marker_memberships = {
            "people_in_room": {"low": 0.9, "medium": 0.0, "high": 0.0},
            "network_activity_level": {"low": 0.0, "medium": 0.0, "high": 0.8},
        }

        score, used, missing = assumption.evaluate(marker_memberships)

        # Weighted mean: (0.4 * 0.9 + 0.6 * 0.8) / 1.0 = 0.84
        assert score == pytest.approx(0.84)
        assert "people_in_room" in used
        assert "network_activity_level" in used
        assert len(missing) == 0

    def test_missing_markers_handled_gracefully(self) -> None:
        """Test missing markers handled gracefully."""
        assumption = ContextAssumption(
            name="solitary_digital",
            conditions={
                "people_in_room": ("low", 0.4),
                "network_activity_level": ("high", 0.6),
            },
            operator="WEIGHTED_MEAN",
        )

        # Only one marker available
        marker_memberships = {
            "people_in_room": {"low": 0.9, "medium": 0.0, "high": 0.0},
        }

        score, used, missing = assumption.evaluate(marker_memberships)

        # Should use available marker only, renormalised: (0.4 * 0.9) / 0.4 = 0.9
        assert score == pytest.approx(0.9)
        assert "people_in_room" in used
        assert "network_activity_level" in missing

    def test_threshold_fallback_when_no_conditions(self) -> None:
        """Test threshold is used as fallback when no conditions."""
        assumption = ContextAssumption(
            name="neutral",
            conditions={},
            threshold=0.3,
        )

        score, used, missing = assumption.evaluate({})

        assert score == 0.3
        assert len(used) == 0
        assert len(missing) == 0


class TestContextAssumptionWeightSensitivity:
    """Tests verifying that condition weights affect the aggregated score (AC9)."""

    def test_different_weights_produce_different_scores(self) -> None:
        """Two conditions with equal membership but different weights must produce different contributions."""
        memberships = {
            "marker_a": {"high": 0.8},
            "marker_b": {"high": 0.4},
        }

        assumption_a = ContextAssumption(
            name="test_a",
            conditions={
                "marker_a": ("high", 0.7),
                "marker_b": ("high", 0.3),
            },
            operator="WEIGHTED_MEAN",
        )

        assumption_b = ContextAssumption(
            name="test_b",
            conditions={
                "marker_a": ("high", 0.3),
                "marker_b": ("high", 0.7),
            },
            operator="WEIGHTED_MEAN",
        )

        score_a, _, _ = assumption_a.evaluate(memberships)
        score_b, _, _ = assumption_b.evaluate(memberships)

        # score_a = (0.7*0.8 + 0.3*0.4) / 1.0 = 0.68
        # score_b = (0.3*0.8 + 0.7*0.4) / 1.0 = 0.52
        assert score_a == pytest.approx(0.68)
        assert score_b == pytest.approx(0.52)
        assert score_a != pytest.approx(score_b)

    def test_equal_membership_any_weights_produces_same_score(self) -> None:
        """All conditions with equal membership — score equals that membership regardless of weights."""
        memberships = {
            "marker_a": {"high": 0.6},
            "marker_b": {"high": 0.6},
        }

        assumption = ContextAssumption(
            name="test",
            conditions={
                "marker_a": ("high", 0.9),
                "marker_b": ("high", 0.1),
            },
            operator="WEIGHTED_MEAN",
        )

        score, _, _ = assumption.evaluate(memberships)

        # (0.9*0.6 + 0.1*0.6) / 1.0 = 0.6
        assert score == pytest.approx(0.6)

    def test_zero_membership_with_small_weight_score_not_zero(self) -> None:
        """One condition with membership 0.0 and weight 0.1 — score is not zero (unlike fuzzy minimum)."""
        memberships = {
            "marker_a": {"high": 0.0},
            "marker_b": {"high": 0.8},
        }

        assumption = ContextAssumption(
            name="test",
            conditions={
                "marker_a": ("high", 0.1),
                "marker_b": ("high", 0.9),
            },
            operator="WEIGHTED_MEAN",
        )

        score, _, _ = assumption.evaluate(memberships)

        # (0.1*0.0 + 0.9*0.8) / 1.0 = 0.72
        # With fuzzy minimum, this would be 0.0
        assert score == pytest.approx(0.72)
        assert score > 0.0


class TestContextEvaluator:
    """Tests for ContextEvaluator class."""

    def _create_context_record(
        self,
        name: str,
        value: float,
        timestamp: datetime | None = None,
    ) -> ContextRecord:
        """Helper to create a ContextRecord."""
        return ContextRecord(
            id="test-id",
            user_id="test-user",
            timestamp=timestamp or datetime.now(UTC),
            context_type="environment",
            name=name,
            value=value,
            raw_value={name: value},
            metadata=None,
        )

    def test_solitary_digital_detected(self) -> None:
        """Test solitary_digital detected with low people, high network."""
        evaluator = ContextEvaluator()

        markers = [
            self._create_context_record("people_in_room", 0.5),  # Low (0-10 count)
            self._create_context_record("network_activity_level", 0.9),  # High
        ]

        result = evaluator.evaluate(markers, apply_smoothing=False)

        assert result.active_context == "solitary_digital"

    def test_neutral_fallback_when_no_strong_context(self) -> None:
        """Test neutral fallback when no strong context matches."""
        evaluator = ContextEvaluator()

        # Medium values - no strong signal
        markers = [
            self._create_context_record("people_in_room", 3),  # Medium (0-10 count)
            self._create_context_record("ambient_noise", 0.5),  # Moderate
            self._create_context_record("network_activity_level", 0.5),  # Medium
        ]

        result = evaluator.evaluate(markers, apply_smoothing=False)

        # Should have some score for neutral due to threshold
        assert "neutral" in result.confidence_scores

    def test_missing_markers_handled_gracefully(self) -> None:
        """Test missing markers are tracked correctly."""
        evaluator = ContextEvaluator()

        # Only provide one marker
        markers = [
            self._create_context_record("people_in_room", 7),
        ]

        result = evaluator.evaluate(markers, apply_smoothing=False)

        assert "people_in_room" in result.markers_used
        # Other markers should be in missing (network for solitary_digital)
        assert len(result.markers_missing) > 0

    def test_markers_used_and_missing_are_accurate(self) -> None:
        """Test markers_used and markers_missing are accurate."""
        evaluator = ContextEvaluator()

        markers = [
            self._create_context_record("people_in_room", 7),
            self._create_context_record("ambient_noise", 0.5),
        ]

        result = evaluator.evaluate(markers, apply_smoothing=False)

        assert "people_in_room" in result.markers_used
        assert "ambient_noise" in result.markers_used
        # network_activity_level was not provided
        assert "network_activity_level" in result.markers_missing

    def test_ema_smoothing_applied_when_enabled(self) -> None:
        """Test EMA smoothing is applied when enabled."""
        evaluator = ContextEvaluator()

        markers = [
            self._create_context_record("people_in_room", 7),
            self._create_context_record("ambient_noise", 0.5),
        ]

        # First evaluation
        result1 = evaluator.evaluate(markers, apply_smoothing=True)
        assert result1.smoothed is True

        # Second evaluation with different values
        markers2 = [
            self._create_context_record("people_in_room", 1.0),
            self._create_context_record("ambient_noise", 0.2),
        ]
        result2 = evaluator.evaluate(markers2, apply_smoothing=True)

        # Smoothed scores should differ from raw due to EMA
        # (raw scores would drop sharply, smoothed less so)
        assert result2.smoothed is True

    def test_batch_evaluation_maintains_state(self) -> None:
        """Test batch evaluation maintains EMA state across evaluations."""
        evaluator = ContextEvaluator()

        now = datetime.now(UTC)
        markers_list = [
            [
                self._create_context_record("people_in_room", 7.0, now),
                self._create_context_record("ambient_noise", 0.5, now),
            ],
            [
                self._create_context_record(
                    "people_in_room", 1.0, now + timedelta(minutes=1)
                ),
                self._create_context_record(
                    "ambient_noise", 0.2, now + timedelta(minutes=1)
                ),
            ],
            [
                self._create_context_record(
                    "people_in_room", 0.5, now + timedelta(minutes=2)
                ),
                self._create_context_record(
                    "ambient_noise", 0.1, now + timedelta(minutes=2)
                ),
            ],
        ]

        results = evaluator.evaluate_batch(markers_list, apply_smoothing=True)

        assert len(results) == 3
        # All should be smoothed
        assert all(r.smoothed for r in results)

    def test_no_markers_returns_neutral(self) -> None:
        """Test empty markers returns neutral context."""
        evaluator = ContextEvaluator()

        result = evaluator.evaluate([], apply_smoothing=False)

        assert result.active_context == "neutral"
        assert result.confidence_scores == {"neutral": 1.0}
        assert len(result.markers_used) == 0

    def test_reset_clears_smoother_state(self) -> None:
        """Test reset clears EMA smoother state."""
        evaluator = ContextEvaluator()

        markers = [
            self._create_context_record("people_in_room", 7),
        ]

        # Build up state
        evaluator.evaluate(markers, apply_smoothing=True)

        # Reset
        evaluator.reset()

        # State should be cleared (smoother reset)
        assert evaluator._smoother.get_active_context() is None


class TestContextEvaluationConfig:
    """Tests for configuration loading."""

    def test_get_default_context_config(self) -> None:
        """Test default config has expected structure."""
        config = get_default_context_config()

        assert "people_in_room" in config.marker_memberships
        assert "ambient_noise" in config.marker_memberships
        assert "network_activity_level" in config.marker_memberships

        assert "solitary_digital" in config.context_assumptions
        assert "neutral" in config.context_assumptions

    def test_load_context_config_from_file(self) -> None:
        """Test loading config from YAML file."""
        config = load_context_config("config/context_evaluation.yaml")

        assert "people_in_room" in config.marker_memberships
        assert "solitary_digital" in config.context_assumptions

    def test_load_context_config_file_not_found(self) -> None:
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            load_context_config("nonexistent.yaml")


class TestMarkerMembershipConfig:
    """Tests for MarkerMembershipConfig."""

    def test_get_membership_fns_returns_defined_sets(self) -> None:
        """Test get_membership_fns returns all defined sets."""
        config = MarkerMembershipConfig(
            low=MembershipFunction(type="triangular", params=[0, 0, 2]),
            high=MembershipFunction(type="triangular", params=[3, 5, 5]),
        )

        fns = config.get_membership_fns()

        assert "low" in fns
        assert "high" in fns
        assert "medium" not in fns  # Not defined


class TestContextCondition:
    """Tests for ContextCondition model."""

    def test_context_condition_alias(self) -> None:
        """Test ContextCondition handles 'set' field alias."""
        # 'set' is the alias, 'fuzzy_set' is the actual field name
        condition = ContextCondition(set="high", weight=0.6)
        assert condition.fuzzy_set == "high"
        assert condition.weight == 0.6
