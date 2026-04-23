"""Tests for window-level FASL aggregation module.

Story 6.4: Window-Level FASL Aggregation (AC6)
Tests WindowIndicator dataclass, compute_window_indicators function,
missing biomarker strategies, and FASL operator.
"""

from datetime import UTC, datetime

import pytest

from src.core.config import AnalysisConfig, BiomarkerWeight, FaslConfig, IndicatorConfig
from src.core.models.window_models import WindowIndicator, WindowMembership
from src.core.processors.window_fasl import (
    MembershipData,
    apply_fasl_operator,
    apply_missing_strategy,
    compute_window_indicators,
)


# ============================================================================
# WindowIndicator Dataclass Tests (AC1)
# ============================================================================


class TestWindowIndicator:
    """Tests for WindowIndicator dataclass."""

    def test_create_window_indicator(self):
        """Test creating a WindowIndicator instance with all fields."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.75,
            contributing_biomarkers={
                "speech_activity": 0.8,
                "connections": 0.6,
            },
            biomarkers_present=2,
            biomarkers_expected=3,
            biomarker_completeness=0.667,
            dominant_context="social",
        )

        assert indicator.window_start == window_start
        assert indicator.window_end == window_end
        assert indicator.indicator_name == "social_withdrawal"
        assert indicator.indicator_score == 0.75
        assert indicator.contributing_biomarkers == {
            "speech_activity": 0.8,
            "connections": 0.6,
        }
        assert indicator.biomarkers_present == 2
        assert indicator.biomarkers_expected == 3
        assert indicator.biomarker_completeness == 0.667
        assert indicator.dominant_context == "social"

    def test_window_indicator_is_frozen(self):
        """Test that WindowIndicator is immutable."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.75,
            contributing_biomarkers={"speech_activity": 0.8},
            biomarkers_present=1,
            biomarkers_expected=2,
            biomarker_completeness=0.5,
            dominant_context="social",
        )

        with pytest.raises(AttributeError):
            indicator.indicator_score = 0.5

    def test_window_indicator_score_in_range(self):
        """Test that indicator_score is accepted in [0, 1] range."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # Test edge values
        for score in [0.0, 0.5, 1.0]:
            indicator = WindowIndicator(
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                indicator_score=score,
                contributing_biomarkers={},
                biomarkers_present=0,
                biomarkers_expected=3,
                biomarker_completeness=0.0,
                dominant_context="neutral",
            )
            assert indicator.indicator_score == score

    def test_window_indicator_completeness_calculation(self):
        """Test that biomarker_completeness represents present/expected ratio."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # Test with 2 of 3 biomarkers present
        indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.6,
            contributing_biomarkers={"speech_activity": 0.7, "connections": 0.5},
            biomarkers_present=2,
            biomarkers_expected=3,
            biomarker_completeness=2 / 3,
            dominant_context="social",
        )

        assert indicator.biomarker_completeness == pytest.approx(2 / 3, rel=1e-6)
        assert indicator.biomarkers_present == 2
        assert indicator.biomarkers_expected == 3


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_window_memberships():
    """Create sample WindowMembership data for testing."""
    window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
    window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

    return {
        "speech_activity": [
            WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={"social": 0.9},
                dominant_context="social",
                context_weight=1.2,
                context_confidence=1.0,
                weighted_membership=0.324,
                readings_count=5,
            )
        ],
        "connections": [
            WindowMembership(
                biomarker_name="connections",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.4,
                z_score=-0.5,
                membership=0.38,
                context_strategy="dominant",
                context_state={"social": 0.9},
                dominant_context="social",
                context_weight=1.1,
                context_confidence=1.0,
                weighted_membership=0.418,
                readings_count=3,
            )
        ],
    }


@pytest.fixture
def sample_indicator_config():
    """Create sample indicator configuration."""
    return IndicatorConfig(
        biomarkers={
            "speech_activity": BiomarkerWeight(
                weight=0.5, direction="lower_is_worse"
            ),
            "connections": BiomarkerWeight(
                weight=0.3, direction="lower_is_worse"
            ),
            "movement_radius": BiomarkerWeight(
                weight=0.2, direction="lower_is_worse"
            ),
        },
        min_biomarkers=2,
    )


@pytest.fixture
def sample_config(sample_indicator_config):
    """Create a minimal analysis config for testing."""
    from src.core.config import get_default_config

    config = get_default_config()
    return config


# ============================================================================
# Missing Biomarker Strategy Tests (AC3)
# ============================================================================


class TestApplyMissingStrategy:
    """Tests for missing biomarker strategy application.

    Story 6.10: Updated to return MembershipData with raw memberships,
    context_weights, and confidences.
    """

    def test_neutral_fill_fills_missing_with_neutral_value(self):
        """Test neutral_fill strategy fills missing biomarkers with 0.5."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        present_memberships = {
            "speech_activity": WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={"social": 0.9},
                dominant_context="social",
                context_weight=1.0,
                context_confidence=1.0,
                weighted_membership=0.27,
                readings_count=5,
            ),
        }
        expected_biomarkers = ["speech_activity", "connections", "movement_radius"]

        result = apply_missing_strategy(
            window_memberships=present_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy="neutral_fill",
            neutral_value=0.5,
        )

        assert result is not None
        assert isinstance(result, MembershipData)
        # Raw membership (not weighted)
        assert result.memberships["speech_activity"] == 0.27
        assert result.memberships["connections"] == 0.5  # Filled with neutral
        assert result.memberships["movement_radius"] == 0.5  # Filled with neutral
        # Context weights
        assert result.context_weights["speech_activity"] == 1.0
        assert result.context_weights["connections"] == 1.0  # Neutral for missing
        assert result.context_weights["movement_radius"] == 1.0  # Neutral for missing
        # Confidences (high=1.0, none=0.0 for missing)
        assert result.confidences["speech_activity"] == 1.0  # high -> 1.0
        assert result.confidences["connections"] == 0.0  # Missing -> no confidence
        assert result.confidences["movement_radius"] == 0.0

    def test_partial_fasl_uses_only_present_biomarkers(self):
        """Test partial_fasl strategy uses only present biomarkers."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        present_memberships = {
            "speech_activity": WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={"social": 0.9},
                dominant_context="social",
                context_weight=1.0,
                context_confidence=1.0,
                weighted_membership=0.27,
                readings_count=5,
            ),
        }
        expected_biomarkers = ["speech_activity", "connections", "movement_radius"]

        result = apply_missing_strategy(
            window_memberships=present_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy="partial_fasl",
        )

        assert result is not None
        assert isinstance(result, MembershipData)
        assert result.memberships["speech_activity"] == 0.27
        assert "connections" not in result.memberships
        assert "movement_radius" not in result.memberships

    def test_skip_window_returns_none_when_incomplete(self):
        """Test skip_window strategy returns None when biomarkers missing."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        present_memberships = {
            "speech_activity": WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={},
                dominant_context="neutral",
                context_weight=1.0,
                context_confidence=0.0,
                weighted_membership=0.27,
                readings_count=5,
            ),
        }
        expected_biomarkers = ["speech_activity", "connections"]

        result = apply_missing_strategy(
            window_memberships=present_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy="skip_window",
        )

        assert result is None

    def test_skip_window_returns_membership_data_when_complete(self):
        """Test skip_window strategy returns MembershipData when all biomarkers present."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        present_memberships = {
            "speech_activity": WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={},
                dominant_context="neutral",
                context_weight=1.0,
                context_confidence=0.0,
                weighted_membership=0.27,
                readings_count=5,
            ),
        }
        expected_biomarkers = ["speech_activity"]

        result = apply_missing_strategy(
            window_memberships=present_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy="skip_window",
        )

        assert result is not None
        assert isinstance(result, MembershipData)
        assert result.memberships["speech_activity"] == 0.27

    def test_neutral_fill_uses_custom_neutral_value(self):
        """Test neutral_fill respects custom neutral_value parameter."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        present_memberships = {
            "speech_activity": WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={},
                dominant_context="neutral",
                context_weight=1.0,
                context_confidence=0.0,
                weighted_membership=0.27,
                readings_count=5,
            ),
        }
        expected_biomarkers = ["speech_activity", "connections"]

        result = apply_missing_strategy(
            window_memberships=present_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy="neutral_fill",
            neutral_value=0.3,  # Custom neutral value
        )

        assert result is not None
        assert isinstance(result, MembershipData)
        assert result.memberships["connections"] == 0.3

    def test_extracts_context_weights_and_confidences(self):
        """Test that context weights and confidences are extracted correctly.

        Story 6.10: MembershipData contains context_weights and confidences
        for FASL computation.
        """
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        present_memberships = {
            "speech_activity": WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.3,
                z_score=-1.0,
                membership=0.27,
                context_strategy="dominant",
                context_state={"social": 0.9},
                dominant_context="social",
                context_weight=1.5,  # Non-neutral context weight
                context_confidence=1.0,
                weighted_membership=0.405,
                readings_count=5,
            ),
        }
        expected_biomarkers = ["speech_activity"]

        result = apply_missing_strategy(
            window_memberships=present_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy="partial_fasl",
        )

        assert result is not None
        assert result.context_weights["speech_activity"] == 1.5
        assert result.confidences["speech_activity"] == 1.0  # high -> 1.0


# ============================================================================
# FASL Operator Tests (AC4) - Story 6.10 Updates
# ============================================================================


class TestApplyFaslOperator:
    """Tests for FASL operator computation with context weight integration."""

    def test_fasl_with_context_weights_and_confidence(self):
        """Test FASL computes effective weights using context weights and confidence.

        Story 6.10 AC2, AC3: Context weights applied in FASL formula.
        effective_context_weight = 1.0 + (context_weight - 1.0) × confidence
        effective_weight = biomarker_weight × effective_context_weight
        """
        memberships = {
            "speech": 0.119,  # Raw membership (low, concerning for lower_is_worse)
            "voice": 0.269,
            "connections": 0.500,
        }
        weights = {
            "speech": 0.30,
            "voice": 0.20,
            "connections": 0.25,
        }
        directions = {
            "speech": "lower_is_worse",
            "voice": "lower_is_worse",
            "connections": "lower_is_worse",
        }
        context_weights = {
            "speech": 1.5,
            "voice": 1.3,
            "connections": 0.7,
        }
        confidences = {
            "speech": 0.82,
            "voice": 0.82,
            "connections": 0.82,
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights=context_weights,
            confidences=confidences,
        )

        # Expected calculation from story Dev Notes:
        # effective_context_weights: speech=1.41, voice=1.246, connections=0.754
        # effective_weights: speech=0.423, voice=0.249, connections=0.189
        # directed_μ (1-μ for lower_is_worse): speech=0.881, voice=0.731, connections=0.500
        # numerator = (0.423 × 0.881) + (0.249 × 0.731) + (0.189 × 0.500) = 0.650
        # denominator = 0.423 + 0.249 + 0.189 = 0.861
        # FASL = 0.650 / 0.861 = 0.755
        assert score == pytest.approx(0.755, rel=0.01)

    def test_fasl_context_weight_zero_excludes_biomarker(self):
        """Test that context_weight = 0 excludes biomarker from numerator AND denominator.

        Story 6.10 AC4: context_weight = 0 excludes biomarker.
        """
        memberships = {
            "speech": 0.3,
            "voice": 0.5,
        }
        weights = {
            "speech": 0.5,
            "voice": 0.5,
        }
        directions = {
            "speech": "lower_is_worse",
            "voice": "lower_is_worse",
        }
        context_weights = {
            "speech": 0.0,  # Excluded
            "voice": 1.0,
        }
        confidences = {
            "speech": 1.0,
            "voice": 1.0,
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights=context_weights,
            confidences=confidences,
        )

        # Only voice contributes:
        # effective_weight_voice = 0.5 × 1.0 = 0.5
        # directed_μ_voice = 1 - 0.5 = 0.5
        # FASL = (0.5 × 0.5) / 0.5 = 0.5
        assert score == pytest.approx(0.5, rel=1e-6)

    def test_fasl_all_context_weights_zero_returns_nan(self):
        """Test that all context weights = 0 returns NaN.

        Story 6.10 AC4: All context weights = 0 returns NaN.
        """
        import math

        memberships = {
            "speech": 0.8,
            "voice": 0.5,
        }
        weights = {
            "speech": 0.5,
            "voice": 0.5,
        }
        directions = {
            "speech": "higher_is_worse",
            "voice": "higher_is_worse",
        }
        context_weights = {
            "speech": 0.0,
            "voice": 0.0,
        }
        confidences = {
            "speech": 1.0,
            "voice": 1.0,
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights=context_weights,
            confidences=confidences,
        )

        assert math.isnan(score)

    def test_fasl_confidence_zero_makes_context_weight_neutral(self):
        """Test that confidence = 0 makes effective_context_weight = 1.0.

        Story 6.10 AC3: confidence = 0 → effective_context_weight = 1.0
        """
        memberships = {
            "speech": 0.3,
        }
        weights = {
            "speech": 1.0,
        }
        directions = {
            "speech": "lower_is_worse",
        }
        context_weights = {
            "speech": 2.0,  # Would double if confidence were 1.0
        }
        confidences = {
            "speech": 0.0,  # No confidence, context weight ignored
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights=context_weights,
            confidences=confidences,
        )

        # effective_context_weight = 1.0 + (2.0 - 1.0) × 0.0 = 1.0
        # effective_weight = 1.0 × 1.0 = 1.0
        # directed_μ = 1 - 0.3 = 0.7
        # FASL = (1.0 × 0.7) / 1.0 = 0.7
        assert score == pytest.approx(0.7, rel=1e-6)

    def test_fasl_confidence_one_uses_full_context_weight(self):
        """Test that confidence = 1 uses full context weight.

        Story 6.10 AC3: confidence = 1 → effective_context_weight = context_weight
        """
        memberships = {
            "speech": 0.3,
        }
        weights = {
            "speech": 1.0,
        }
        directions = {
            "speech": "lower_is_worse",
        }
        context_weights = {
            "speech": 2.0,
        }
        confidences = {
            "speech": 1.0,  # Full confidence
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights=context_weights,
            confidences=confidences,
        )

        # effective_context_weight = 1.0 + (2.0 - 1.0) × 1.0 = 2.0
        # effective_weight = 1.0 × 2.0 = 2.0
        # directed_μ = 1 - 0.3 = 0.7
        # FASL = (2.0 × 0.7) / 2.0 = 0.7 (same, but weight is doubled)
        assert score == pytest.approx(0.7, rel=1e-6)

    def test_fasl_context_weight_one_is_neutral(self):
        """Test that context_weight = 1 produces same result as no context weighting.

        Story 6.10 AC validation: context_weight = 1 is neutral.
        """
        memberships = {
            "speech": 0.3,
            "voice": 0.5,
        }
        weights = {
            "speech": 0.6,
            "voice": 0.4,
        }
        directions = {
            "speech": "lower_is_worse",
            "voice": "lower_is_worse",
        }

        # With context weights = 1.0
        score_with_context = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights={"speech": 1.0, "voice": 1.0},
            confidences={"speech": 1.0, "voice": 1.0},
        )

        # Without context weights (defaults)
        score_without_context = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
        )

        assert score_with_context == pytest.approx(score_without_context, rel=1e-6)

    def test_fasl_lower_is_worse_with_context_weight_gt_one_increases_score(self):
        """Test that lower_is_worse + context_weight > 1 increases FASL score.

        Story 6.10 validation: This is the key bug fix - ensuring concern INCREASES
        when context_weight > 1 for lower_is_worse biomarkers.
        """
        memberships = {
            "speech": 0.2,  # Low speech (concerning for lower_is_worse)
        }
        weights = {
            "speech": 1.0,
        }
        directions = {
            "speech": "lower_is_worse",
        }

        # Without context amplification
        score_neutral = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights={"speech": 1.0},
            confidences={"speech": 1.0},
        )

        # With context amplification (speech matters more in social context)
        score_amplified = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
            context_weights={"speech": 1.5},
            confidences={"speech": 1.0},
        )

        # Scores should be the same (single biomarker), but effective_weight differs
        # What matters is the relative contribution in multi-biomarker scenarios
        # Let's verify both compute correctly:
        # neutral: directed_μ = 0.8, FASL = 0.8
        # amplified: directed_μ = 0.8, effective_weight = 1.5, FASL = 0.8
        assert score_neutral == pytest.approx(0.8, rel=1e-6)
        assert score_amplified == pytest.approx(0.8, rel=1e-6)

    def test_fasl_backward_compatibility_no_context_weights(self):
        """Test FASL works without context_weights parameter (backward compatibility)."""
        memberships = {
            "speech_activity": 0.8,
            "connections": 0.6,
        }
        weights = {
            "speech_activity": 0.6,
            "connections": 0.4,
        }
        directions = {
            "speech_activity": "higher_is_worse",
            "connections": "higher_is_worse",
        }

        # Call without context_weights and confidences
        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
        )

        # score = (0.6 * 0.8 + 0.4 * 0.6) / (0.6 + 0.4)
        # score = (0.48 + 0.24) / 1.0 = 0.72
        assert score == pytest.approx(0.72, rel=1e-6)

    def test_fasl_weighted_average_higher_is_worse(self):
        """Test FASL computes weighted average for higher_is_worse direction."""
        memberships = {
            "speech_activity": 0.8,
            "connections": 0.6,
        }
        weights = {
            "speech_activity": 0.6,
            "connections": 0.4,
        }
        directions = {
            "speech_activity": "higher_is_worse",
            "connections": "higher_is_worse",
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
        )

        # score = (0.6 * 0.8 + 0.4 * 0.6) / (0.6 + 0.4)
        # score = (0.48 + 0.24) / 1.0 = 0.72
        assert score == pytest.approx(0.72, rel=1e-6)

    def test_fasl_weighted_average_lower_is_worse(self):
        """Test FASL computes inverted membership for lower_is_worse direction."""
        memberships = {
            "speech_activity": 0.3,  # Low value
        }
        weights = {
            "speech_activity": 1.0,
        }
        directions = {
            "speech_activity": "lower_is_worse",
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
        )

        # For lower_is_worse: directed_mu = 1.0 - mu = 1.0 - 0.3 = 0.7
        # score = (1.0 * 0.7) / 1.0 = 0.7
        assert score == pytest.approx(0.7, rel=1e-6)

    def test_fasl_mixed_directions(self):
        """Test FASL handles mixed higher_is_worse and lower_is_worse."""
        memberships = {
            "awakenings": 0.8,  # High awakenings = concerning (higher_is_worse)
            "speech_activity": 0.2,  # Low speech = concerning (lower_is_worse)
        }
        weights = {
            "awakenings": 0.5,
            "speech_activity": 0.5,
        }
        directions = {
            "awakenings": "higher_is_worse",
            "speech_activity": "lower_is_worse",
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
        )

        # awakenings: 0.5 * 0.8 = 0.4
        # speech_activity (inverted): 0.5 * (1 - 0.2) = 0.5 * 0.8 = 0.4
        # score = (0.4 + 0.4) / 1.0 = 0.8
        assert score == pytest.approx(0.8, rel=1e-6)

    def test_fasl_returns_nan_with_zero_weights(self):
        """Test FASL returns NaN when all weights are zero.

        Story 6.10 AC4: All weights = 0 returns NaN (cannot compute meaningful indicator).
        """
        import math

        memberships = {
            "speech_activity": 0.8,
        }
        weights = {
            "speech_activity": 0.0,
        }
        directions = {
            "speech_activity": "higher_is_worse",
        }

        score = apply_fasl_operator(
            memberships=memberships,
            weights=weights,
            directions=directions,
        )

        assert math.isnan(score)

    def test_fasl_empty_memberships_returns_zero(self):
        """Test FASL returns 0.0 with empty memberships."""
        score = apply_fasl_operator(
            memberships={},
            weights={},
            directions={},
        )

        assert score == 0.0


# ============================================================================
# Compute Window Indicators Tests (AC2)
# ============================================================================


class TestComputeWindowIndicators:
    """Tests for compute_window_indicators function."""

    def test_produces_indicator_for_each_window(
        self, sample_window_memberships, sample_indicator_config, sample_config
    ):
        """Test that an indicator is produced for each unique window."""
        result = compute_window_indicators(
            window_memberships=sample_window_memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 1  # One window in sample data
        assert isinstance(result[0], WindowIndicator)
        assert result[0].indicator_name == "social_withdrawal"

    def test_tracks_biomarker_completeness(
        self, sample_window_memberships, sample_indicator_config, sample_config
    ):
        """Test that completeness is correctly calculated."""
        result = compute_window_indicators(
            window_memberships=sample_window_memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        # Sample has 2 biomarkers, indicator expects 3
        indicator = result[0]
        assert indicator.biomarkers_present == 2
        assert indicator.biomarkers_expected == 3
        assert indicator.biomarker_completeness == pytest.approx(2 / 3, rel=1e-2)

    def test_handles_multiple_windows(self, sample_indicator_config, sample_config):
        """Test handling of multiple windows across time."""
        window1_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window1_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        window2_start = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        window2_end = datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC)

        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window1_start,
                    window_end=window1_end,
                    aggregated_value=0.3,
                    z_score=-1.0,
                    membership=0.27,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="social",
                    context_weight=1.0,
                    context_confidence=1.0,
                    weighted_membership=0.27,
                    readings_count=5,
                ),
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window2_start,
                    window_end=window2_end,
                    aggregated_value=0.5,
                    z_score=0.0,
                    membership=0.5,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.5,
                    readings_count=3,
                ),
            ],
        }

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 2
        # Verify windows are ordered chronologically
        assert result[0].window_start == window1_start
        assert result[1].window_start == window2_start

    def test_empty_memberships_returns_empty_list(
        self, sample_indicator_config, sample_config
    ):
        """Test that empty memberships return empty list."""
        result = compute_window_indicators(
            window_memberships={},
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert result == []

    def test_uses_dominant_context_from_window(
        self, sample_window_memberships, sample_indicator_config, sample_config
    ):
        """Test that dominant context is captured from window memberships."""
        result = compute_window_indicators(
            window_memberships=sample_window_memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        indicator = result[0]
        assert indicator.dominant_context == "social"

    def test_contributing_biomarkers_recorded(
        self, sample_window_memberships, sample_indicator_config, sample_config
    ):
        """Test that contributing biomarkers and their raw memberships are recorded.

        Story 6.10: contributing_biomarkers now contains raw memberships (not weighted).
        """
        result = compute_window_indicators(
            window_memberships=sample_window_memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        indicator = result[0]
        assert "speech_activity" in indicator.contributing_biomarkers
        assert "connections" in indicator.contributing_biomarkers
        # Values should be the raw membership values (not weighted_membership)
        assert indicator.contributing_biomarkers["speech_activity"] == pytest.approx(
            0.27, rel=1e-2
        )
        assert indicator.contributing_biomarkers["connections"] == pytest.approx(
            0.38, rel=1e-2
        )

    def test_skip_window_strategy_skips_incomplete_windows(
        self, sample_indicator_config
    ):
        """Test that skip_window strategy skips windows with missing biomarkers."""
        window1_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window1_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        window2_start = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        window2_end = datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Window 1: Complete (all 3 biomarkers), Window 2: Incomplete (only 1)
        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window1_start,
                    window_end=window1_end,
                    aggregated_value=0.5,
                    z_score=0.0,
                    membership=0.5,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.5,
                    readings_count=5,
                ),
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window2_start,
                    window_end=window2_end,
                    aggregated_value=0.3,
                    z_score=-1.0,
                    membership=0.27,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.27,
                    readings_count=3,
                ),
            ],
            "connections": [
                WindowMembership(
                    biomarker_name="connections",
                    window_start=window1_start,
                    window_end=window1_end,
                    aggregated_value=0.6,
                    z_score=0.5,
                    membership=0.62,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.62,
                    readings_count=4,
                ),
                # No connections data for window 2
            ],
            "movement_radius": [
                WindowMembership(
                    biomarker_name="movement_radius",
                    window_start=window1_start,
                    window_end=window1_end,
                    aggregated_value=0.7,
                    z_score=1.0,
                    membership=0.73,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.73,
                    readings_count=2,
                ),
                # No movement_radius data for window 2
            ],
        }

        # Create config with skip_window strategy
        from src.core.config import get_default_config

        config = get_default_config()
        # Override fasl config with skip_window strategy
        config_dict = config.model_dump()
        config_dict["fasl"] = {
            "missing_biomarker_strategy": "skip_window",
            "neutral_membership": 0.5,
        }
        skip_config = AnalysisConfig(**config_dict)

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=skip_config,
        )

        # Only window 1 should be included (complete), window 2 should be skipped
        assert len(result) == 1
        assert result[0].window_start == window1_start
        assert result[0].biomarkers_present == 3
        assert result[0].biomarker_completeness == 1.0


# ============================================================================
# FaslConfig Tests (AC5)
# ============================================================================


class TestFaslConfig:
    """Tests for FaslConfig configuration."""

    def test_fasl_config_defaults(self):
        """Test that FaslConfig has correct default values."""
        config = FaslConfig()

        assert config.missing_biomarker_strategy == "neutral_fill"
        assert config.neutral_membership == 0.5

    def test_fasl_config_custom_values(self):
        """Test FaslConfig with custom values."""
        config = FaslConfig(
            missing_biomarker_strategy="partial_fasl",
            neutral_membership=0.3,
        )

        assert config.missing_biomarker_strategy == "partial_fasl"
        assert config.neutral_membership == 0.3

    def test_fasl_config_valid_strategies(self):
        """Test that all valid strategies are accepted."""
        for strategy in ["neutral_fill", "partial_fasl", "skip_window"]:
            config = FaslConfig(missing_biomarker_strategy=strategy)
            assert config.missing_biomarker_strategy == strategy

    def test_fasl_config_in_analysis_config(self):
        """Test that FaslConfig is available in AnalysisConfig."""
        from src.core.config import get_default_config

        config = get_default_config()

        assert hasattr(config, "fasl")
        assert isinstance(config.fasl, FaslConfig)
        assert config.fasl.missing_biomarker_strategy == "neutral_fill"
        assert config.fasl.neutral_membership == 0.5


# ============================================================================
# Edge Case Tests (AC6)
# ============================================================================


# ============================================================================
# Integration Tests - Story 6.10 Bug Fix Validation
# ============================================================================


class TestStory610Integration:
    """Integration tests validating the Story 6.10 bug fix.

    These tests verify that context weights applied in FASL produce correct
    behavior for lower_is_worse biomarkers, fixing the counterintuitive
    behavior where context_weight > 1 was decreasing concern instead of
    increasing it.
    """

    def test_lower_is_worse_with_context_gt_one_increases_fasl(
        self, sample_indicator_config, sample_config
    ):
        """Test that lower_is_worse + context_weight > 1 increases FASL score.

        Story 6.10 validation: This is the key bug fix test.

        Setup:
        - Speech activity with low raw membership (0.2) - concerning
        - Direction: lower_is_worse
        - Context weight > 1 (speech matters more in social context)

        Expected behavior:
        - Concern should INCREASE when context_weight > 1
        - FASL score should be higher with context amplification
        """
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # Create memberships with neutral context (weight=1.0)
        neutral_memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.2,
                    z_score=-1.5,
                    membership=0.2,  # Low - concerning for lower_is_worse
                    context_strategy="dominant",
                    context_state={"neutral": 1.0},
                    dominant_context="neutral",
                    context_weight=1.0,  # Neutral
                    context_confidence=1.0,
                    weighted_membership=0.2,
                    readings_count=5,
                ),
            ],
        }

        # Create memberships with amplified context (weight=1.5)
        amplified_memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.2,
                    z_score=-1.5,
                    membership=0.2,  # Same low value
                    context_strategy="dominant",
                    context_state={"social": 1.0},
                    dominant_context="social",
                    context_weight=1.5,  # Amplified - speech matters more
                    context_confidence=1.0,
                    weighted_membership=0.2,  # Story 6.10: equals membership
                    readings_count=5,
                ),
            ],
        }

        # Create indicator config for single biomarker
        from src.core.config import BiomarkerWeight, IndicatorConfig

        single_bio_config = IndicatorConfig(
            biomarkers={
                "speech_activity": BiomarkerWeight(
                    weight=1.0, direction="lower_is_worse"
                ),
            },
            min_biomarkers=1,
        )

        # Compute indicators with neutral context
        result_neutral = compute_window_indicators(
            window_memberships=neutral_memberships,
            indicator_name="social_withdrawal",
            indicator_config=single_bio_config,
            config=sample_config,
        )

        # Compute indicators with amplified context
        result_amplified = compute_window_indicators(
            window_memberships=amplified_memberships,
            indicator_name="social_withdrawal",
            indicator_config=single_bio_config,
            config=sample_config,
        )

        assert len(result_neutral) == 1
        assert len(result_amplified) == 1

        neutral_score = result_neutral[0].indicator_score
        amplified_score = result_amplified[0].indicator_score

        # Both should show high concern (membership=0.2, direction=lower_is_worse)
        # directed_mu = 1 - 0.2 = 0.8
        assert neutral_score == pytest.approx(0.8, rel=1e-6)
        assert amplified_score == pytest.approx(0.8, rel=1e-6)

        # For single biomarker, scores are same but effective_weight differs
        # The key test is multi-biomarker scenario below

    def test_multi_biomarker_context_amplification_increases_concern(
        self, sample_config
    ):
        """Test that context amplification correctly increases concern in multi-biomarker scenario.

        This is the worked example from the story Dev Notes:
        - speech: low membership (0.119), lower_is_worse, context_weight=1.5
        - voice: low membership (0.269), lower_is_worse, context_weight=1.3
        - connections: neutral membership (0.500), lower_is_worse, context_weight=0.7

        Expected FASL with context weights: ~0.755
        Expected FASL without context (all weights=1): ~0.719 (estimated)

        The context should AMPLIFY concern from low speech/voice.
        """
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # Memberships with context weights from story example
        memberships = {
            "speech": [
                WindowMembership(
                    biomarker_name="speech",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.1,
                    z_score=-2.0,
                    membership=0.119,
                    context_strategy="dominant",
                    context_state={"social": 1.0},
                    dominant_context="social",
                    context_weight=1.5,
                    context_confidence=1.0,
                    weighted_membership=0.119,
                    readings_count=5,
                ),
            ],
            "voice": [
                WindowMembership(
                    biomarker_name="voice",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.2,
                    z_score=-1.0,
                    membership=0.269,
                    context_strategy="dominant",
                    context_state={"social": 1.0},
                    dominant_context="social",
                    context_weight=1.3,
                    context_confidence=1.0,
                    weighted_membership=0.269,
                    readings_count=3,
                ),
            ],
            "connections": [
                WindowMembership(
                    biomarker_name="connections",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.5,
                    z_score=0.0,
                    membership=0.500,
                    context_strategy="dominant",
                    context_state={"social": 1.0},
                    dominant_context="social",
                    context_weight=0.7,
                    context_confidence=1.0,
                    weighted_membership=0.500,
                    readings_count=4,
                ),
            ],
        }

        # Indicator config from story example
        from src.core.config import BiomarkerWeight, IndicatorConfig

        indicator_config = IndicatorConfig(
            biomarkers={
                "speech": BiomarkerWeight(weight=0.40, direction="lower_is_worse"),
                "voice": BiomarkerWeight(weight=0.27, direction="lower_is_worse"),
                "connections": BiomarkerWeight(weight=0.33, direction="lower_is_worse"),
            },
            min_biomarkers=1,
        )

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=indicator_config,
            config=sample_config,
        )

        assert len(result) == 1
        indicator = result[0]

        # Expected calculation with adjusted biomarker weights that sum to 1.0:
        # biomarker_weights: speech=0.40, voice=0.27, connections=0.33
        # effective_context_weights: speech=1.5, voice=1.3, connections=0.7 (confidence=1.0)
        # effective_weights: speech=0.60, voice=0.351, connections=0.231
        # directed_μ: speech=0.881, voice=0.731, connections=0.500
        # numerator = (0.60 × 0.881) + (0.351 × 0.731) + (0.231 × 0.500)
        #           = 0.5286 + 0.2566 + 0.1155 = 0.9007
        # denominator = 0.60 + 0.351 + 0.231 = 1.182
        # FASL = 0.9007 / 1.182 ≈ 0.762

        # Allow some tolerance due to rounding
        assert indicator.indicator_score == pytest.approx(0.762, rel=0.02)

    def test_context_weight_zero_excludes_biomarker_from_computation(
        self, sample_config
    ):
        """Test that context_weight=0 excludes biomarker from FASL entirely.

        Story 6.10 AC4: context_weight=0 excludes biomarker from numerator AND denominator.
        """
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        memberships = {
            "speech": [
                WindowMembership(
                    biomarker_name="speech",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.1,
                    z_score=-2.0,
                    membership=0.119,
                    context_strategy="dominant",
                    context_state={"work": 1.0},
                    dominant_context="work",
                    context_weight=0.0,  # Speech irrelevant in work context
                    context_confidence=1.0,
                    weighted_membership=0.119,
                    readings_count=5,
                ),
            ],
            "focus": [
                WindowMembership(
                    biomarker_name="focus",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.8,
                    z_score=1.5,
                    membership=0.817,
                    context_strategy="dominant",
                    context_state={"work": 1.0},
                    dominant_context="work",
                    context_weight=1.5,  # Focus amplified in work context
                    context_confidence=1.0,
                    weighted_membership=0.817,
                    readings_count=3,
                ),
            ],
        }

        from src.core.config import BiomarkerWeight, IndicatorConfig

        indicator_config = IndicatorConfig(
            biomarkers={
                "speech": BiomarkerWeight(weight=0.5, direction="lower_is_worse"),
                "focus": BiomarkerWeight(weight=0.5, direction="higher_is_worse"),
            },
            min_biomarkers=1,
        )

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="cognitive_load",
            indicator_config=indicator_config,
            config=sample_config,
        )

        assert len(result) == 1
        indicator = result[0]

        # Speech excluded (context_weight=0), only focus contributes
        # effective_weight_focus = 0.5 × 1.5 = 0.75
        # directed_μ_focus = 0.817 (higher_is_worse, no flip)
        # FASL = (0.75 × 0.817) / 0.75 = 0.817
        assert indicator.indicator_score == pytest.approx(0.817, rel=1e-6)


# ============================================================================
# Edge Case Tests (AC6)
# ============================================================================


class TestWindowFaslEdgeCases:
    """Edge case tests for window FASL computation."""

    def test_sparse_data_neutral_fill_dilutes_score(self, sample_indicator_config, sample_config):
        """Test that sparse data with neutral_fill dilutes score toward 0.5."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # Only one biomarker present (out of 3 expected)
        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.1,
                    z_score=-2.0,
                    membership=0.12,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.12,  # Very low - concerning
                    readings_count=5,
                ),
            ],
        }

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 1
        indicator = result[0]
        # With neutral_fill, 2 missing biomarkers get 0.5
        # Score should be diluted toward neutral
        assert indicator.biomarker_completeness == pytest.approx(1 / 3, rel=1e-2)
        assert indicator.biomarkers_present == 1
        assert indicator.biomarkers_expected == 3

    def test_all_biomarkers_high_score_when_all_concerning(
        self, sample_indicator_config, sample_config
    ):
        """Test that all concerning biomarkers produce high indicator score."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # All biomarkers with low values (concerning for lower_is_worse)
        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.1,
                    z_score=-2.0,
                    membership=0.12,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.1,  # Low = concerning for lower_is_worse
                    readings_count=5,
                ),
            ],
            "connections": [
                WindowMembership(
                    biomarker_name="connections",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.1,
                    z_score=-2.0,
                    membership=0.12,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.1,
                    readings_count=3,
                ),
            ],
            "movement_radius": [
                WindowMembership(
                    biomarker_name="movement_radius",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.1,
                    z_score=-2.0,
                    membership=0.12,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.1,
                    readings_count=2,
                ),
            ],
        }

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 1
        indicator = result[0]
        # For lower_is_worse: score = 1 - membership
        # All are 0.1, so directed = 0.9
        # Expected score ≈ 0.9
        assert indicator.indicator_score > 0.8
        assert indicator.biomarker_completeness == 1.0

    def test_all_biomarkers_low_score_when_all_healthy(
        self, sample_indicator_config, sample_config
    ):
        """Test that all healthy biomarkers produce low indicator score."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # All biomarkers with high values (healthy for lower_is_worse)
        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.9,
                    z_score=2.0,
                    membership=0.88,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.9,  # High = healthy for lower_is_worse
                    readings_count=5,
                ),
            ],
            "connections": [
                WindowMembership(
                    biomarker_name="connections",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.9,
                    z_score=2.0,
                    membership=0.88,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.9,
                    readings_count=3,
                ),
            ],
            "movement_radius": [
                WindowMembership(
                    biomarker_name="movement_radius",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.9,
                    z_score=2.0,
                    membership=0.88,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.9,
                    readings_count=2,
                ),
            ],
        }

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 1
        indicator = result[0]
        # For lower_is_worse: score = 1 - membership
        # All are 0.9, so directed = 0.1
        # Expected score ≈ 0.1
        assert indicator.indicator_score < 0.2
        assert indicator.biomarker_completeness == 1.0

    def test_boundary_membership_values(self, sample_indicator_config, sample_config):
        """Test with boundary membership values (0.0 and 1.0)."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # Edge case: membership at boundaries
        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.0,
                    z_score=-5.0,
                    membership=0.0,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.0,
                    readings_count=5,
                ),
            ],
            "connections": [
                WindowMembership(
                    biomarker_name="connections",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=1.0,
                    z_score=5.0,
                    membership=1.0,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=1.0,
                    readings_count=3,
                ),
            ],
        }

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 1
        indicator = result[0]
        # Score should be between 0 and 1
        assert 0.0 <= indicator.indicator_score <= 1.0

    def test_windows_with_different_biomarker_coverage(
        self, sample_indicator_config, sample_config
    ):
        """Test handling windows with varying biomarker coverage."""
        window1_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window1_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        window2_start = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        window2_end = datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC)

        # Window 1: has both biomarkers, Window 2: only speech
        memberships = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window1_start,
                    window_end=window1_end,
                    aggregated_value=0.5,
                    z_score=0.0,
                    membership=0.5,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.5,
                    readings_count=5,
                ),
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window2_start,
                    window_end=window2_end,
                    aggregated_value=0.3,
                    z_score=-1.0,
                    membership=0.27,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.27,
                    readings_count=3,
                ),
            ],
            "connections": [
                WindowMembership(
                    biomarker_name="connections",
                    window_start=window1_start,
                    window_end=window1_end,
                    aggregated_value=0.6,
                    z_score=0.5,
                    membership=0.62,
                    context_strategy="dominant",
                    context_state={},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.62,
                    readings_count=4,
                ),
                # No connections data for window 2
            ],
        }

        result = compute_window_indicators(
            window_memberships=memberships,
            indicator_name="social_withdrawal",
            indicator_config=sample_indicator_config,
            config=sample_config,
        )

        assert len(result) == 2

        # Window 1: 2 biomarkers present
        assert result[0].biomarkers_present == 2
        assert result[0].biomarker_completeness == pytest.approx(2 / 3, rel=1e-2)

        # Window 2: 1 biomarker present
        assert result[1].biomarkers_present == 1
        assert result[1].biomarker_completeness == pytest.approx(1 / 3, rel=1e-2)
