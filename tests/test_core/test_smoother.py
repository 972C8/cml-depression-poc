"""Unit tests for EMA smoother.

Tests exponential moving average smoothing with hysteresis and dwell time.
"""

import pytest

from src.core.context.smoother import EMASmoother


class TestEMASmootherBasic:
    """Basic tests for EMASmoother."""

    def test_first_value_returns_value_directly(self) -> None:
        """Test first value (no previous) returns value directly."""
        smoother = EMASmoother(alpha=0.3)
        result = smoother.smooth("test_context", 0.8)
        assert result == 0.8

    def test_smoothing_reduces_variance(self) -> None:
        """Test smoothing reduces variance in oscillating signal."""
        smoother = EMASmoother(alpha=0.3)

        # Feed oscillating values
        values = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        results = [smoother.smooth("ctx", v) for v in values]

        # Calculate variance of input vs output
        import statistics

        input_variance = statistics.variance(values)
        output_variance = statistics.variance(results)

        # Smoothed output should have lower variance
        assert output_variance < input_variance

    def test_smoothing_formula_correctness(self) -> None:
        """Test EMA formula: smoothed = alpha * current + (1 - alpha) * previous."""
        alpha = 0.4
        smoother = EMASmoother(alpha=alpha)

        # First value
        first = smoother.smooth("ctx", 1.0)
        assert first == 1.0

        # Second value: expected = 0.4 * 0.5 + 0.6 * 1.0 = 0.8
        second = smoother.smooth("ctx", 0.5)
        expected = alpha * 0.5 + (1 - alpha) * 1.0
        assert second == pytest.approx(expected)


class TestEMASmootherHysteresis:
    """Tests for hysteresis behavior."""

    def test_hysteresis_prevents_small_oscillations(self) -> None:
        """Test hysteresis prevents switching on small differences."""
        smoother = EMASmoother(alpha=0.5, hysteresis=0.2, dwell_time=1)

        # Set initial context
        smoother.should_switch_context("ctx_a", 0.6, None, 0.0)

        # Small difference - should not switch
        decision = smoother.should_switch_context("ctx_b", 0.65, "ctx_a", 0.6)
        assert not decision.should_switch
        assert decision.blocked_reason == "hysteresis"

    def test_hysteresis_allows_large_difference(self) -> None:
        """Test hysteresis allows switching on large differences."""
        smoother = EMASmoother(alpha=0.5, hysteresis=0.1, dwell_time=1)

        # Set initial context
        smoother.should_switch_context("ctx_a", 0.4, None, 0.0)

        # Large difference - should switch
        decision = smoother.should_switch_context("ctx_b", 0.8, "ctx_a", 0.4)
        assert decision.should_switch
        assert decision.blocked_reason is None


class TestEMASmootherDwellTime:
    """Tests for dwell time behavior."""

    def test_dwell_time_delays_transitions(self) -> None:
        """Test dwell time requires consecutive readings before switch."""
        smoother = EMASmoother(alpha=0.5, hysteresis=0.1, dwell_time=3)

        # Set initial context
        smoother.should_switch_context("ctx_a", 0.4, None, 0.0)

        # First reading above threshold - not enough dwell time
        decision1 = smoother.should_switch_context("ctx_b", 0.8, "ctx_a", 0.4)
        assert not decision1.should_switch
        assert decision1.blocked_reason == "dwell_time"
        assert decision1.dwell_progress == 1

        # Second reading - still not enough
        decision2 = smoother.should_switch_context("ctx_b", 0.8, "ctx_a", 0.4)
        assert not decision2.should_switch
        assert decision2.blocked_reason == "dwell_time"
        assert decision2.dwell_progress == 2

        # Third reading - dwell time satisfied
        decision3 = smoother.should_switch_context("ctx_b", 0.8, "ctx_a", 0.4)
        assert decision3.should_switch
        assert decision3.blocked_reason is None
        assert decision3.dwell_progress == 3

    def test_dwell_time_resets_on_context_change(self) -> None:
        """Test dwell counter resets when candidate context changes."""
        smoother = EMASmoother(alpha=0.5, hysteresis=0.1, dwell_time=3)

        # Set initial context
        smoother.should_switch_context("ctx_a", 0.4, None, 0.0)

        # First reading for ctx_b
        decision1 = smoother.should_switch_context("ctx_b", 0.8, "ctx_a", 0.4)
        assert decision1.dwell_progress == 1

        # Switch to ctx_c - ctx_c starts at dwell 1
        decision2 = smoother.should_switch_context("ctx_c", 0.8, "ctx_a", 0.4)
        assert decision2.dwell_progress == 1
        assert decision2.candidate_context == "ctx_c"

        # Back to ctx_b - counter continues from 2 (ctx_b wasn't reset)
        decision3 = smoother.should_switch_context("ctx_b", 0.8, "ctx_a", 0.4)
        assert not decision3.should_switch  # Only at count 2, need 3
        assert decision3.dwell_progress == 2


class TestEMASmootherReset:
    """Tests for reset behavior."""

    def test_reset_clears_state(self) -> None:
        """Test reset clears all internal state."""
        smoother = EMASmoother(alpha=0.3)

        # Build up some state
        smoother.smooth("ctx_a", 0.5)
        smoother.smooth("ctx_a", 0.6)
        smoother.should_switch_context("ctx_a", 0.6, None, 0.0)

        # Reset
        smoother.reset()

        # State should be cleared
        assert smoother.get_active_context() is None

        # Next smooth should treat as first value
        result = smoother.smooth("ctx_a", 0.8)
        assert result == 0.8


class TestEMASmootherDeterminism:
    """Tests for deterministic behavior."""

    def test_same_inputs_produce_same_outputs(self) -> None:
        """Test determinism: same inputs = same outputs."""
        inputs = [0.5, 0.6, 0.4, 0.7, 0.3]

        # First run
        smoother1 = EMASmoother(alpha=0.3)
        outputs1 = [smoother1.smooth("ctx", v) for v in inputs]

        # Second run with same inputs
        smoother2 = EMASmoother(alpha=0.3)
        outputs2 = [smoother2.smooth("ctx", v) for v in inputs]

        # Should be identical
        assert outputs1 == outputs2


class TestEMASmootherValidation:
    """Tests for parameter validation."""

    def test_invalid_alpha_raises(self) -> None:
        """Test alpha must be in (0, 1]."""
        with pytest.raises(ValueError, match="alpha must be in"):
            EMASmoother(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            EMASmoother(alpha=1.5)

    def test_negative_hysteresis_raises(self) -> None:
        """Test hysteresis must be >= 0."""
        with pytest.raises(ValueError, match="hysteresis must be >= 0"):
            EMASmoother(hysteresis=-0.1)

    def test_negative_dwell_time_raises(self) -> None:
        """Test dwell_time must be >= 0."""
        with pytest.raises(ValueError, match="dwell_time must be >= 0"):
            EMASmoother(dwell_time=-1)
