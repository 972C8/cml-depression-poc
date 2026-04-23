"""Tests for window membership module.

Story 6.3: Membership with Context Weighting (AC9)
Tests WindowMembership dataclass, z-score to membership mapping,
and compute_window_memberships function.
"""

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.core.baseline_config import BaselineDefinition, BaselineFile
from src.core.config import get_default_config
from src.core.context.history import ContextState
from src.core.context.strategies import ContextStrategyResult
from src.core.models.window_models import WindowAggregate, WindowMembership
from src.core.processors.window_membership import (
    BaselineError,
    compute_window_memberships,
    z_score_to_membership,
)


# ============================================================================
# WindowMembership Dataclass Tests (AC1)
# ============================================================================


class TestWindowMembership:
    """Tests for WindowMembership dataclass."""

    def test_create_window_membership(self):
        """Test creating a WindowMembership instance with all fields."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        membership = WindowMembership(
            biomarker_name="speech_activity",
            window_start=window_start,
            window_end=window_end,
            aggregated_value=0.6,
            z_score=1.5,
            membership=0.817,
            context_strategy="dominant",
            context_state={"social": 0.8, "solitary": 0.2},
            dominant_context="social",
            context_weight=1.2,
            context_confidence=1.0,
            weighted_membership=0.9804,
            readings_count=5,
        )

        assert membership.biomarker_name == "speech_activity"
        assert membership.window_start == window_start
        assert membership.window_end == window_end
        assert membership.aggregated_value == 0.6
        assert membership.z_score == 1.5
        assert membership.membership == 0.817
        assert membership.context_strategy == "dominant"
        assert membership.context_state == {"social": 0.8, "solitary": 0.2}
        assert membership.dominant_context == "social"
        assert membership.context_weight == 1.2
        assert membership.context_confidence == 1.0
        assert membership.weighted_membership == 0.9804
        assert membership.readings_count == 5

    def test_window_membership_is_frozen(self):
        """Test that WindowMembership is immutable."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        membership = WindowMembership(
            biomarker_name="speech_activity",
            window_start=window_start,
            window_end=window_end,
            aggregated_value=0.6,
            z_score=1.5,
            membership=0.817,
            context_strategy="dominant",
            context_state={"social": 0.8},
            dominant_context="social",
            context_weight=1.2,
            context_confidence=1.0,
            weighted_membership=0.9804,
            readings_count=5,
        )

        with pytest.raises(AttributeError):
            membership.weighted_membership = 0.5

    def test_window_membership_context_confidence_values(self):
        """Test that context_confidence accepts float values."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        base_kwargs = {
            "biomarker_name": "speech_activity",
            "window_start": window_start,
            "window_end": window_end,
            "aggregated_value": 0.6,
            "z_score": 0.0,
            "membership": 0.5,
            "context_strategy": "dominant",
            "context_state": {},
            "dominant_context": "neutral",
            "context_weight": 1.0,
            "weighted_membership": 0.5,
            "readings_count": 1,
        }

        # Test float confidence values
        for confidence in [1.0, 0.5, 0.0, 0.85]:
            membership = WindowMembership(**base_kwargs, context_confidence=confidence)
            assert membership.context_confidence == confidence

    def test_window_membership_context_strategy_values(self):
        """Test that context_strategy accepts valid values."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        base_kwargs = {
            "biomarker_name": "speech_activity",
            "window_start": window_start,
            "window_end": window_end,
            "aggregated_value": 0.6,
            "z_score": 0.0,
            "membership": 0.5,
            "context_state": {},
            "dominant_context": "neutral",
            "context_weight": 1.0,
            "context_confidence": 0.0,
            "weighted_membership": 0.5,
            "readings_count": 1,
        }

        # Test all valid strategy values
        for strategy in ["dominant", "time_weighted", "reading_weighted"]:
            membership = WindowMembership(**base_kwargs, context_strategy=strategy)
            assert membership.context_strategy == strategy


# ============================================================================
# Z-Score to Membership Tests (AC6)
# ============================================================================


class TestZScoreToMembership:
    """Tests for z-score to membership sigmoid mapping."""

    def test_z_score_zero_gives_half(self):
        """Test that z=0 maps to membership=0.5 (baseline)."""
        result = z_score_to_membership(0.0)
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_z_score_positive_gives_above_half(self):
        """Test that positive z-scores map to membership > 0.5."""
        for z in [0.5, 1.0, 2.0, 3.0]:
            result = z_score_to_membership(z)
            assert result > 0.5
            assert result < 1.0

    def test_z_score_negative_gives_below_half(self):
        """Test that negative z-scores map to membership < 0.5."""
        for z in [-0.5, -1.0, -2.0, -3.0]:
            result = z_score_to_membership(z)
            assert result < 0.5
            assert result > 0.0

    def test_z_score_three_approximately_95_percent(self):
        """Test that z=3 gives approximately 0.95 (AC6 specification)."""
        result = z_score_to_membership(3.0)
        assert result == pytest.approx(0.9526, rel=0.01)  # ~95%

    def test_z_score_minus_three_approximately_5_percent(self):
        """Test that z=-3 gives approximately 0.05 (AC6 specification)."""
        result = z_score_to_membership(-3.0)
        assert result == pytest.approx(0.0474, rel=0.01)  # ~5%

    def test_z_score_one_approximately_73_percent(self):
        """Test that z=1 gives approximately 0.73."""
        result = z_score_to_membership(1.0)
        assert result == pytest.approx(0.731, rel=0.01)

    def test_z_score_minus_one_approximately_27_percent(self):
        """Test that z=-1 gives approximately 0.27."""
        result = z_score_to_membership(-1.0)
        assert result == pytest.approx(0.269, rel=0.01)

    def test_z_score_symmetric(self):
        """Test that sigmoid is symmetric around z=0."""
        for z in [1.0, 2.0, 3.0]:
            pos = z_score_to_membership(z)
            neg = z_score_to_membership(-z)
            assert pos + neg == pytest.approx(1.0, rel=1e-6)


# ============================================================================
# Compute Window Memberships Tests (AC5)
# ============================================================================


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def config():
    """Create a test configuration."""
    return get_default_config()


@pytest.fixture
def baseline_config():
    """Create a baseline configuration with test values."""
    return BaselineFile(
        baselines={
            "speech_activity": BaselineDefinition(mean=0.5, std=0.2),
            "unknown_biomarker": BaselineDefinition(mean=0.5, std=0.2),
        }
    )


@pytest.fixture
def sample_window_aggregates():
    """Create sample WindowAggregate data."""
    window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
    window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
    timestamps = (
        window_start + timedelta(minutes=5),
        window_start + timedelta(minutes=10),
    )

    return {
        "speech_activity": [
            WindowAggregate(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.6,
                readings_count=2,
                readings_timestamps=timestamps,
            )
        ]
    }


class TestComputeWindowMemberships:
    """Tests for compute_window_memberships function."""

    def test_computes_membership_for_each_aggregate(
        self, mock_session, config, baseline_config, sample_window_aggregates
    ):
        """Test that memberships are computed for all aggregates."""
        # Mock the context strategy
        with patch(
            "src.core.processors.window_membership.get_window_context"
        ) as mock_context:
            mock_context.return_value = ContextStrategyResult(
                dominant_context="social",
                context_state={"social": 0.9},
                context_weight=1.0,
                confidence=1.0,
                strategy_used="dominant",
            )

            result = compute_window_memberships(
                window_aggregates=sample_window_aggregates,
                user_id="user1",
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                baseline_config=baseline_config,
            )

            assert "speech_activity" in result
            assert len(result["speech_activity"]) == 1

    def test_computes_z_score_correctly(
        self, mock_session, config, baseline_config, sample_window_aggregates
    ):
        """Test that z-score is computed as (value - mean) / std."""
        with patch(
            "src.core.processors.window_membership.get_window_context"
        ) as mock_context:
            mock_context.return_value = ContextStrategyResult(
                dominant_context="neutral",
                context_state={"neutral": 1.0},
                context_weight=1.0,
                confidence=0.0,
                strategy_used="dominant",
            )

            result = compute_window_memberships(
                window_aggregates=sample_window_aggregates,
                user_id="user1",
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                baseline_config=baseline_config,
            )

            membership = result["speech_activity"][0]
            # z = (0.6 - 0.5) / 0.2 = 0.5
            assert membership.z_score == pytest.approx(0.5, rel=1e-6)

    def test_context_weight_passed_through_not_applied(
        self, mock_session, config, baseline_config, sample_window_aggregates
    ):
        """Test that context_weight is passed through but NOT applied to membership.

        Story 6.10: Context weights are applied in FASL formula, not in membership
        computation. This prevents counterintuitive behavior for lower_is_worse biomarkers.
        """
        with patch(
            "src.core.processors.window_membership.get_window_context"
        ) as mock_context:
            mock_context.return_value = ContextStrategyResult(
                dominant_context="social",
                context_state={"social": 0.9},
                context_weight=1.2,  # Weight > 1
                confidence=1.0,
                strategy_used="dominant",
            )

            result = compute_window_memberships(
                window_aggregates=sample_window_aggregates,
                user_id="user1",
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                baseline_config=baseline_config,
            )

            membership = result["speech_activity"][0]
            # Story 6.10: weighted_membership = membership (no context weight applied)
            assert membership.weighted_membership == pytest.approx(
                membership.membership, rel=1e-6
            )
            # Context weight is still available for FASL step
            assert membership.context_weight == 1.2
            assert membership.context_confidence == 1.0

    def test_raises_error_when_biomarker_missing_from_baseline(
        self, mock_session, config
    ):
        """Test that BaselineError is raised when biomarker not in baseline."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        aggregates = {
            "missing_biomarker": [
                WindowAggregate(
                    biomarker_name="missing_biomarker",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.5,
                    readings_count=1,
                    readings_timestamps=(window_start,),
                )
            ]
        }

        baseline_config = BaselineFile(
            baselines={"other_biomarker": BaselineDefinition(mean=0.5, std=0.1)}
        )

        with pytest.raises(BaselineError) as exc_info:
            compute_window_memberships(
                window_aggregates=aggregates,
                user_id="user1",
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                baseline_config=baseline_config,
            )

        assert "missing_biomarker" in str(exc_info.value)


# ============================================================================
# Edge Case Tests (AC8)
# ============================================================================


class TestWindowMembershipEdgeCases:
    """Tests for edge cases in window membership computation."""

    def test_handles_low_std_with_minimum_floor(self, mock_session, config):
        """Test that very low std gets replaced with minimum floor."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        aggregates = {
            "speech_activity": [
                WindowAggregate(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.5,
                    readings_count=1,
                    readings_timestamps=(window_start,),
                )
            ]
        }

        # Very low std that should be replaced with minimum
        baseline_config = BaselineFile(
            baselines={"speech_activity": BaselineDefinition(mean=0.5, std=0.00001)}
        )

        with patch(
            "src.core.processors.window_membership.get_window_context"
        ) as mock_context:
            mock_context.return_value = ContextStrategyResult(
                dominant_context="neutral",
                context_state={"neutral": 1.0},
                context_weight=1.0,
                confidence=0.0,
                strategy_used="dominant",
            )

            result = compute_window_memberships(
                window_aggregates=aggregates,
                user_id="user1",
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                baseline_config=baseline_config,
            )

            # Should still produce valid membership (not crash)
            membership = result["speech_activity"][0]
            assert 0.0 <= membership.membership <= 1.0

    def test_clamps_weighted_membership_to_one(self, mock_session, config):
        """Test that weighted_membership is clamped to [0, 1]."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        # High value that will produce high membership
        aggregates = {
            "speech_activity": [
                WindowAggregate(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=1.0,  # High value
                    readings_count=1,
                    readings_timestamps=(window_start,),
                )
            ]
        }

        # Small std = high z-score
        baseline_config = BaselineFile(
            baselines={"speech_activity": BaselineDefinition(mean=0.5, std=0.1)}
        )

        with patch(
            "src.core.processors.window_membership.get_window_context"
        ) as mock_context:
            mock_context.return_value = ContextStrategyResult(
                dominant_context="social",
                context_state={"social": 1.0},
                context_weight=1.5,  # Weight > 1 that could push over 1.0
                confidence=1.0,
                strategy_used="dominant",
            )

            result = compute_window_memberships(
                window_aggregates=aggregates,
                user_id="user1",
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                baseline_config=baseline_config,
            )

            membership = result["speech_activity"][0]
            assert membership.weighted_membership <= 1.0
            assert membership.weighted_membership >= 0.0

    def test_empty_aggregates_returns_empty_dict(self, mock_session, config, baseline_config):
        """Test that empty aggregates returns empty result."""
        result = compute_window_memberships(
            window_aggregates={},
            user_id="user1",
            indicator_name="social_withdrawal",
            session=mock_session,
            config=config,
            baseline_config=baseline_config,
        )

        assert result == {}
