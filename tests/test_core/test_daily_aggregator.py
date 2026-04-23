"""Tests for daily summary aggregation module.

Story 6.5: Daily Summary Computation
Tests DailyIndicatorSummary dataclass and compute_daily_summary function.
Likelihood uses consecutive-window peak mean (max mean of k consecutive windows).
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from src.core.config import get_default_config
from src.core.models.daily_summary import DailyIndicatorSummary
from src.core.models.window_models import WindowIndicator
from src.core.processors.daily_aggregator import (
    _max_consecutive_mean,
    compute_daily_summary,
)


# ============================================================================
# DailyIndicatorSummary Dataclass Tests
# ============================================================================


class TestDailyIndicatorSummary:
    """Tests for DailyIndicatorSummary dataclass."""

    def test_create_daily_indicator_summary(self):
        """Test creating a DailyIndicatorSummary instance with all fields."""
        summary = DailyIndicatorSummary(
            date=date(2025, 1, 15),
            indicator_name="social_withdrawal",
            likelihood=0.55,
            window_scores=(),
            total_windows=80,
            expected_windows=96,
            data_coverage=0.833,
            average_biomarker_completeness=0.85,
            context_availability=0.70,
        )

        assert summary.date == date(2025, 1, 15)
        assert summary.indicator_name == "social_withdrawal"
        assert summary.likelihood == 0.55
        assert summary.total_windows == 80
        assert summary.expected_windows == 96
        assert summary.data_coverage == 0.833
        assert summary.average_biomarker_completeness == 0.85
        assert summary.context_availability == 0.70

    def test_daily_indicator_summary_is_frozen(self):
        """Test that DailyIndicatorSummary is immutable."""
        summary = DailyIndicatorSummary(
            date=date(2025, 1, 15),
            indicator_name="social_withdrawal",
            likelihood=0.55,
            window_scores=(),
            total_windows=80,
            expected_windows=96,
            data_coverage=0.833,
            average_biomarker_completeness=0.85,
            context_availability=0.70,
        )

        with pytest.raises(AttributeError):
            summary.likelihood = 0.5


# ============================================================================
# Helper: create WindowIndicator list
# ============================================================================


def _make_window_indicators(
    scores: list[float],
    base_time: datetime | None = None,
    indicator_name: str = "social_withdrawal",
    completeness: float = 1.0,
    context: str = "neutral",
) -> list[WindowIndicator]:
    """Create a list of WindowIndicators from scores."""
    if base_time is None:
        base_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)

    indicators = []
    for i, score in enumerate(scores):
        window_start = base_time + timedelta(minutes=15 * i)
        window_end = window_start + timedelta(minutes=15)
        indicators.append(
            WindowIndicator(
                window_start=window_start,
                window_end=window_end,
                indicator_name=indicator_name,
                indicator_score=score,
                contributing_biomarkers={},
                biomarkers_present=2,
                biomarkers_expected=2,
                biomarker_completeness=completeness,
                dominant_context=context,
            )
        )
    return indicators


# ============================================================================
# _max_consecutive_mean Unit Tests
# ============================================================================


class TestMaxConsecutiveMean:
    """Tests for the _max_consecutive_mean helper function."""

    def test_fewer_scores_than_k_returns_mean(self):
        """When n < k, fall back to mean of all scores."""
        scores = [0.2, 0.4, 0.6]
        assert _max_consecutive_mean(scores, k=5) == pytest.approx(0.4, rel=1e-6)

    def test_exactly_k_scores_returns_mean(self):
        """When n == k, return mean of all scores."""
        scores = [0.2, 0.4, 0.6, 0.8]
        assert _max_consecutive_mean(scores, k=4) == pytest.approx(0.5, rel=1e-6)

    def test_single_score(self):
        """Single score with k=1 returns that score."""
        assert _max_consecutive_mean([0.7], k=1) == pytest.approx(0.7, rel=1e-6)

    def test_peak_at_start(self):
        """Peak consecutive segment at the start of the list."""
        scores = [0.9, 0.8, 0.7, 0.1, 0.1, 0.1]
        # k=3: best segment is [0.9, 0.8, 0.7] = mean 0.8
        assert _max_consecutive_mean(scores, k=3) == pytest.approx(0.8, rel=1e-6)

    def test_peak_at_end(self):
        """Peak consecutive segment at the end of the list."""
        scores = [0.1, 0.1, 0.1, 0.7, 0.8, 0.9]
        # k=3: best segment is [0.7, 0.8, 0.9] = mean 0.8
        assert _max_consecutive_mean(scores, k=3) == pytest.approx(0.8, rel=1e-6)

    def test_peak_in_middle(self):
        """Peak consecutive segment in the middle of the list."""
        scores = [0.1, 0.1, 0.8, 0.9, 0.8, 0.1, 0.1]
        # k=3: best segment is [0.8, 0.9, 0.8] = mean 0.8333
        assert _max_consecutive_mean(scores, k=3) == pytest.approx(
            0.8333, rel=1e-3
        )

    def test_uniform_scores(self):
        """All identical scores — any segment gives the same mean."""
        scores = [0.5] * 10
        assert _max_consecutive_mean(scores, k=4) == pytest.approx(0.5, rel=1e-6)

    def test_scattered_spikes_diluted(self):
        """Non-consecutive spikes should NOT produce a high score."""
        # Spikes at positions 0, 3, 6 — never 3 in a row
        scores = [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1]
        result = _max_consecutive_mean(scores, k=3)
        # Best consecutive-3 windows: [0.9,0.1,0.1]=0.367, [0.1,0.1,0.9]=0.367,
        # [0.1,0.9,0.1]=0.367, etc.
        assert result == pytest.approx(0.3667, rel=1e-2)

    def test_k_equals_1_returns_max(self):
        """With k=1, the result is simply the maximum score."""
        scores = [0.1, 0.3, 0.9, 0.2]
        assert _max_consecutive_mean(scores, k=1) == pytest.approx(0.9, rel=1e-6)


# ============================================================================
# Compute Daily Summary Tests
# ============================================================================


class TestComputeDailySummary:
    """Tests for compute_daily_summary function."""

    def test_empty_indicators_returns_none(self):
        """Test that empty indicators return None."""
        config = get_default_config()
        assert compute_daily_summary([], config) is None

    def test_no_data_for_target_date_returns_none(self):
        """Test when no data exists for target date."""
        config = get_default_config()
        indicators = _make_window_indicators([0.5])
        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 16)
        )
        assert summary is None

    def test_likelihood_is_peak_consecutive_mean(self):
        """Test that likelihood equals max mean of k consecutive windows."""
        config = get_default_config()
        k = config.episode.peak_window_k  # default 4

        # 8 windows: low except for a k-window peak in the middle
        scores = [0.1, 0.1, 0.8, 0.9, 0.8, 0.7, 0.1, 0.1]
        indicators = _make_window_indicators(scores)

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary is not None
        # Best 4 consecutive: [0.8, 0.9, 0.8, 0.7] = mean 0.8
        assert summary.likelihood == pytest.approx(0.8, rel=1e-6)

    def test_single_window(self):
        """Test with single window — likelihood equals that score."""
        config = get_default_config()
        indicators = _make_window_indicators([0.7])

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary is not None
        assert summary.likelihood == pytest.approx(0.7, rel=1e-6)
        assert summary.total_windows == 1

    def test_fewer_windows_than_k_falls_back_to_mean(self):
        """When fewer windows than k, likelihood is mean of all windows."""
        config = get_default_config()
        k = config.episode.peak_window_k  # 4
        scores = [0.3, 0.7]  # only 2 windows < k=4
        indicators = _make_window_indicators(scores)

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary is not None
        assert summary.likelihood == pytest.approx(0.5, rel=1e-6)

    def test_uniform_scores(self):
        """Test with all identical scores."""
        config = get_default_config()
        indicators = _make_window_indicators([0.6] * 10)

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary.likelihood == pytest.approx(0.6, rel=1e-6)

    def test_scattered_spikes_diluted_by_consecutive_requirement(self):
        """Scattered spikes should not inflate the daily likelihood."""
        config = get_default_config()
        k = config.episode.peak_window_k  # 4

        # High spikes at positions 0, 4, 8 — never k in a row
        scores = [0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        indicators = _make_window_indicators(scores)

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        # No 4-consecutive-window segment has more than one spike
        # Best segment mean should be around 0.3 (one 0.9 + three 0.1)
        assert summary.likelihood == pytest.approx(0.3, rel=1e-2)

    def test_quality_metrics(self):
        """Test quality metrics computation."""
        config = get_default_config()
        indicators = _make_window_indicators(
            [0.5, 0.6, 0.7],
            completeness=0.667,
            context="work",
        )

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary.total_windows == 3
        assert summary.expected_windows == 96
        assert summary.data_coverage == pytest.approx(3 / 96, rel=1e-2)
        assert summary.average_biomarker_completeness == pytest.approx(0.667, rel=1e-2)
        assert summary.context_availability == pytest.approx(1.0, rel=1e-6)

    def test_context_availability_with_neutral(self):
        """Test context availability counts non-neutral windows."""
        config = get_default_config()
        base_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)

        indicators = []
        contexts = ["neutral", "work", "neutral", "work"]
        for i, ctx in enumerate(contexts):
            window_start = base_time + timedelta(minutes=15 * i)
            window_end = window_start + timedelta(minutes=15)
            indicators.append(
                WindowIndicator(
                    window_start=window_start,
                    window_end=window_end,
                    indicator_name="social_withdrawal",
                    indicator_score=0.5,
                    contributing_biomarkers={},
                    biomarkers_present=2,
                    biomarkers_expected=2,
                    biomarker_completeness=1.0,
                    dominant_context=ctx,
                )
            )

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary.context_availability == pytest.approx(0.5, rel=1e-6)

    def test_date_filtering(self):
        """Test that only windows for target date are included."""
        config = get_default_config()
        indicators = _make_window_indicators([0.5, 0.6, 0.7])

        # Add window from different date
        wrong_date = WindowIndicator(
            window_start=datetime(2025, 1, 16, 9, 0, 0, tzinfo=UTC),
            window_end=datetime(2025, 1, 16, 9, 15, 0, tzinfo=UTC),
            indicator_name="social_withdrawal",
            indicator_score=0.9,
            contributing_biomarkers={},
            biomarkers_present=2,
            biomarkers_expected=2,
            biomarker_completeness=1.0,
            dominant_context="work",
        )

        summary = compute_daily_summary(
            indicators + [wrong_date], config, target_date=date(2025, 1, 15)
        )

        assert summary.total_windows == 3
        # 0.9 from Jan 16 should NOT be included

    def test_inferred_date_from_first_window(self):
        """Test date inference when target_date not provided."""
        config = get_default_config()
        indicators = _make_window_indicators([0.5])

        summary = compute_daily_summary(indicators, config)

        assert summary is not None
        assert summary.date == date(2025, 1, 15)

    def test_mixed_indicator_names_raises_error(self):
        """Test that mixed indicator names raise ValueError."""
        config = get_default_config()
        base_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        indicators = [
            WindowIndicator(
                window_start=base_time,
                window_end=base_time + timedelta(minutes=15),
                indicator_name="social_withdrawal",
                indicator_score=0.5,
                contributing_biomarkers={},
                biomarkers_present=1,
                biomarkers_expected=1,
                biomarker_completeness=1.0,
                dominant_context="neutral",
            ),
            WindowIndicator(
                window_start=base_time + timedelta(minutes=15),
                window_end=base_time + timedelta(minutes=30),
                indicator_name="different_indicator",
                indicator_score=0.6,
                contributing_biomarkers={},
                biomarkers_present=1,
                biomarkers_expected=1,
                biomarker_completeness=1.0,
                dominant_context="neutral",
            ),
        ]

        with pytest.raises(ValueError, match="same indicator"):
            compute_daily_summary(indicators, config, target_date=date(2025, 1, 15))

    def test_full_day_96_windows_with_episode(self):
        """Test with full day of 96 windows — episode should be detected by peak."""
        config = get_default_config()
        k = config.episode.peak_window_k  # 4
        base_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC)

        # Mostly low scores, with a 1-hour episode (4 windows) of high scores
        scores = []
        for i in range(96):
            hour = i // 4
            if 10 <= hour < 11:  # 10:00-11:00 episode
                scores.append(0.85)
            else:
                scores.append(0.1)

        indicators = _make_window_indicators(scores, base_time=base_time)

        summary = compute_daily_summary(
            indicators, config, target_date=date(2025, 1, 15)
        )

        assert summary.total_windows == 96
        assert summary.expected_windows == 96
        assert summary.data_coverage == pytest.approx(1.0, rel=1e-6)

        # Peak k=4 consecutive windows are the episode: mean of [0.85]*4 = 0.85
        assert summary.likelihood == pytest.approx(0.85, rel=1e-4)
