"""Tests for context strategies module.

Story 6.3: Membership with Context Weighting (AC9)
Tests for dominant, time-weighted, and reading-weighted context strategies.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import get_default_config
from src.core.context.history import ContextSegment, ContextState
from src.core.context.strategies import (
    ContextStrategyResult,
    get_window_context,
    get_window_context_dominant,
    get_window_context_reading_weighted,
    get_window_context_time_weighted,
)

# ============================================================================
# Fixtures
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
def window_start():
    """Create a standard window start time."""
    return datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)


@pytest.fixture
def window_end(window_start):
    """Create a standard window end time (15 minutes)."""
    return window_start + timedelta(minutes=15)


# ============================================================================
# ContextStrategyResult Tests
# ============================================================================


class TestContextStrategyResult:
    """Tests for ContextStrategyResult dataclass."""

    def test_create_strategy_result(self):
        """Test creating a ContextStrategyResult instance."""
        result = ContextStrategyResult(
            dominant_context="social",
            context_state={"social": 0.8, "solitary": 0.2},
            context_weight=1.2,
            confidence=1.0,
            strategy_used="dominant",
        )

        assert result.dominant_context == "social"
        assert result.context_state == {"social": 0.8, "solitary": 0.2}
        assert result.context_weight == 1.2
        assert result.confidence == 1.0
        assert result.strategy_used == "dominant"

    def test_strategy_result_is_frozen(self):
        """Test that ContextStrategyResult is immutable."""
        result = ContextStrategyResult(
            dominant_context="social",
            context_state={"social": 0.8},
            context_weight=1.2,
            confidence=1.0,
            strategy_used="dominant",
        )

        with pytest.raises(AttributeError):
            result.context_weight = 1.5


# ============================================================================
# Strategy Dispatcher Tests
# ============================================================================


class TestGetWindowContext:
    """Tests for the get_window_context dispatcher function."""

    def test_dispatcher_routes_to_dominant(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dispatcher routes to dominant strategy."""
        with patch(
            "src.core.context.strategies.get_window_context_dominant"
        ) as mock_dominant:
            mock_dominant.return_value = ContextStrategyResult(
                dominant_context="social",
                context_state={"social": 1.0},
                context_weight=1.0,
                confidence=1.0,
                strategy_used="dominant",
            )

            result = get_window_context(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                strategy="dominant",
            )

            mock_dominant.assert_called_once()
            assert result.strategy_used == "dominant"

    def test_dispatcher_routes_to_time_weighted(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dispatcher routes to time-weighted strategy."""
        with patch(
            "src.core.context.strategies.get_window_context_time_weighted"
        ) as mock_time:
            mock_time.return_value = ContextStrategyResult(
                dominant_context="social",
                context_state={"social": 1.0},
                context_weight=1.0,
                confidence=1.0,
                strategy_used="time_weighted",
            )

            result = get_window_context(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                strategy="time_weighted",
            )

            mock_time.assert_called_once()
            assert result.strategy_used == "time_weighted"

    def test_dispatcher_routes_to_reading_weighted(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dispatcher routes to reading-weighted strategy."""
        timestamps = (window_start, window_start + timedelta(minutes=5))

        with patch(
            "src.core.context.strategies.get_window_context_reading_weighted"
        ) as mock_reading:
            mock_reading.return_value = ContextStrategyResult(
                dominant_context="social",
                context_state={"social": 1.0},
                context_weight=1.0,
                confidence=1.0,
                strategy_used="reading_weighted",
            )

            result = get_window_context(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                strategy="reading_weighted",
                readings_timestamps=timestamps,
            )

            mock_reading.assert_called_once()
            assert result.strategy_used == "reading_weighted"

    def test_dispatcher_raises_for_unknown_strategy(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dispatcher raises ValueError for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown context strategy"):
            get_window_context(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                strategy="invalid",  # type: ignore[arg-type]
            )

    def test_dispatcher_raises_for_reading_weighted_without_timestamps(
        self, mock_session, config, window_start, window_end
    ):
        """Test that reading-weighted strategy requires timestamps."""
        with pytest.raises(
            ValueError, match="readings_timestamps required for reading_weighted"
        ):
            get_window_context(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
                strategy="reading_weighted",
                readings_timestamps=None,
            )


# ============================================================================
# Dominant Strategy Tests (AC2)
# ============================================================================


class TestDominantStrategy:
    """Tests for the dominant (midpoint) context strategy."""

    def test_dominant_uses_midpoint(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dominant strategy uses window midpoint."""
        expected_midpoint = window_start + timedelta(minutes=7, seconds=30)

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_at_timestamp.return_value = ContextState(
                dominant_context="social",
                confidence=0.9,
                all_scores={"social": 0.9, "solitary": 0.1},
                timestamp=expected_midpoint,
            )

            get_window_context_dominant(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            # Verify midpoint was used
            call_args = mock_service.get_context_at_timestamp.call_args
            actual_timestamp = call_args[1]["timestamp"]
            assert actual_timestamp == expected_midpoint

    def test_dominant_returns_context_state(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dominant strategy returns correct context state."""
        midpoint = window_start + timedelta(minutes=7, seconds=30)

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_at_timestamp.return_value = ContextState(
                dominant_context="social",
                confidence=0.9,
                all_scores={"social": 0.9, "solitary": 0.1},
                timestamp=midpoint,
            )

            result = get_window_context_dominant(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            assert result.dominant_context == "social"
            assert result.context_state == {"social": 0.9, "solitary": 0.1}
            assert result.strategy_used == "dominant"

    def test_dominant_returns_neutral_for_missing_context(
        self, mock_session, config, window_start, window_end
    ):
        """Test that dominant strategy returns neutral when no context exists."""
        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_at_timestamp.return_value = None

            result = get_window_context_dominant(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            assert result.dominant_context == "neutral"
            assert result.confidence == 0.0
            assert result.context_weight == config.context_history.neutral_weight


# ============================================================================
# Time-Weighted Strategy Tests (AC3)
# ============================================================================


class TestTimeWeightedStrategy:
    """Tests for the time-weighted context strategy."""

    def test_time_weighted_uses_timeline(
        self, mock_session, config, window_start, window_end
    ):
        """Test that time-weighted strategy queries timeline."""
        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_timeline.return_value = [
                ContextSegment(
                    context="social",
                    confidence=0.9,
                    start=window_start,
                    end=window_end,
                )
            ]

            get_window_context_time_weighted(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            mock_service.get_context_timeline.assert_called_once_with(
                user_id="user1",
                start=window_start,
                end=window_end,
            )

    def test_time_weighted_blends_multiple_contexts(
        self, mock_session, config, window_start, window_end
    ):
        """Test that time-weighted strategy blends multiple context segments."""
        midpoint = window_start + timedelta(minutes=7, seconds=30)

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            # First half is social, second half is solitary
            mock_service.get_context_timeline.return_value = [
                ContextSegment(
                    context="social",
                    confidence=0.9,
                    start=window_start,
                    end=midpoint,
                ),
                ContextSegment(
                    context="solitary",
                    confidence=0.8,
                    start=midpoint,
                    end=window_end,
                ),
            ]

            result = get_window_context_time_weighted(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            # Both contexts should be present in state with ~50% proportion each
            assert "social" in result.context_state
            assert "solitary" in result.context_state
            # Each context covered half the window, so their weights should be equal
            # (context_weight * 0.5 for each, using neutral_weight=1.0 as default)
            assert result.context_state["social"] == pytest.approx(0.5, rel=0.01)
            assert result.context_state["solitary"] == pytest.approx(0.5, rel=0.01)
            assert result.strategy_used == "time_weighted"
            # Full coverage = high confidence
            assert result.confidence == 1.0

    def test_time_weighted_returns_neutral_for_empty_timeline(
        self, mock_session, config, window_start, window_end
    ):
        """Test that time-weighted returns neutral for empty timeline."""
        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_timeline.return_value = []

            result = get_window_context_time_weighted(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            assert result.dominant_context == "neutral"
            assert result.confidence == 0.0

    def test_time_weighted_handles_partial_coverage(
        self, mock_session, config, window_start, window_end
    ):
        """Test that time-weighted fills uncovered portions with neutral weight."""
        # Context only covers first half of window
        half_point = window_start + timedelta(minutes=7, seconds=30)

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            # Timeline only covers first half
            mock_service.get_context_timeline.return_value = [
                ContextSegment(
                    context="social",
                    confidence=0.9,
                    start=window_start,
                    end=half_point,
                ),
            ]

            result = get_window_context_time_weighted(
                user_id="user1",
                window_start=window_start,
                window_end=window_end,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            # Both social and neutral should be present (50% each)
            assert "social" in result.context_state
            assert "neutral" in result.context_state
            # Social covers 50%, neutral fills the other 50%
            assert result.context_state["social"] == pytest.approx(0.5, rel=0.01)
            assert result.context_state["neutral"] == pytest.approx(0.5, rel=0.01)
            # 50% coverage = low confidence
            assert result.confidence == 0.5


# ============================================================================
# Reading-Weighted Strategy Tests (AC4)
# ============================================================================


class TestReadingWeightedStrategy:
    """Tests for the reading-weighted context strategy."""

    def test_reading_weighted_queries_each_timestamp(
        self, mock_session, config, window_start
    ):
        """Test that reading-weighted strategy queries each reading timestamp."""
        timestamps = (
            window_start,
            window_start + timedelta(minutes=5),
            window_start + timedelta(minutes=10),
        )

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_at_timestamp.return_value = ContextState(
                dominant_context="social",
                confidence=0.9,
                all_scores={"social": 0.9},
                timestamp=window_start,
            )

            get_window_context_reading_weighted(
                user_id="user1",
                readings_timestamps=timestamps,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            # Should have been called once per timestamp
            assert mock_service.get_context_at_timestamp.call_count == 3

    def test_reading_weighted_averages_weights(
        self, mock_session, config, window_start
    ):
        """Test that reading-weighted strategy averages context weights."""
        timestamps = (
            window_start,
            window_start + timedelta(minutes=5),
        )

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            mock_service.get_context_at_timestamp.return_value = ContextState(
                dominant_context="social",
                confidence=0.9,
                all_scores={"social": 0.9},
                timestamp=window_start,
            )

            result = get_window_context_reading_weighted(
                user_id="user1",
                readings_timestamps=timestamps,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            assert result.dominant_context == "social"
            assert result.strategy_used == "reading_weighted"
            # All readings had same context, so proportions should be 1.0
            assert result.context_state.get("social", 0.0) == 1.0

    def test_reading_weighted_returns_neutral_for_empty_timestamps(
        self, mock_session, config
    ):
        """Test that reading-weighted returns neutral for empty timestamps."""
        result = get_window_context_reading_weighted(
            user_id="user1",
            readings_timestamps=(),
            indicator_name="social_withdrawal",
            session=mock_session,
            config=config,
        )

        assert result.dominant_context == "neutral"
        assert result.confidence == 0.0

    def test_reading_weighted_handles_mixed_context(
        self, mock_session, config, window_start
    ):
        """Test reading-weighted with some readings having no context."""
        timestamps = (
            window_start,
            window_start + timedelta(minutes=5),
        )

        with patch("src.core.context.strategies.ContextHistoryService") as MockService:
            mock_service = MockService.return_value
            # First reading has context, second doesn't
            mock_service.get_context_at_timestamp.side_effect = [
                ContextState(
                    dominant_context="social",
                    confidence=0.9,
                    all_scores={"social": 0.9},
                    timestamp=window_start,
                ),
                None,  # No context for second reading
            ]

            result = get_window_context_reading_weighted(
                user_id="user1",
                readings_timestamps=timestamps,
                indicator_name="social_withdrawal",
                session=mock_session,
                config=config,
            )

            # Should have both social and neutral in state
            assert "social" in result.context_state or "neutral" in result.context_state
            # Confidence should be low (only 50% coverage)
            assert result.confidence == 0.5
