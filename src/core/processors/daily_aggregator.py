"""Daily summary aggregation module.

Story 6.5: Daily Summary Computation

Aggregates window-level indicator scores into a daily likelihood
using a consecutive-window peak mean (sliding window max-mean).
The daily likelihood is the maximum mean of any k consecutive windows,
where k is configurable via episode.peak_window_k.
"""

import logging
import statistics
from datetime import date

from src.core.config import AnalysisConfig
from src.core.models.daily_summary import DailyIndicatorSummary
from src.core.models.window_models import WindowIndicator

__all__ = [
    "compute_daily_summary",
]

_logger = logging.getLogger(__name__)


def _max_consecutive_mean(scores: list[float], k: int) -> float:
    """Compute the maximum mean of any k consecutive elements.

    Uses an O(n) sliding window sum. When fewer than k scores are
    available, falls back to the mean of all scores.

    Args:
        scores: List of window-level scores (chronologically ordered)
        k: Number of consecutive windows

    Returns:
        Maximum mean of any k consecutive windows
    """
    n = len(scores)
    if n <= k:
        return statistics.mean(scores)

    window_sum = sum(scores[:k])
    max_sum = window_sum

    for i in range(1, n - k + 1):
        window_sum += scores[i + k - 1] - scores[i - 1]
        if window_sum > max_sum:
            max_sum = window_sum

    return max_sum / k


def compute_daily_summary(
    window_indicators: list[WindowIndicator],
    config: AnalysisConfig,
    target_date: date | None = None,
) -> DailyIndicatorSummary | None:
    """Compute daily summary from window-level indicators.

    Aggregates window-level FASL indicator scores into a single daily
    likelihood using a consecutive-window peak mean: the maximum mean
    of any k consecutive windows, where k = config.episode.peak_window_k.

    Story 6.5: Daily Summary Computation

    Args:
        window_indicators: List of WindowIndicator for a single indicator type,
            sorted chronologically (from Story 6.4)
        config: Analysis configuration with window settings
        target_date: Optional date to filter for; if None, infers from first window

    Returns:
        DailyIndicatorSummary with likelihood and quality metrics, or None if no data
    """
    if not window_indicators:
        _logger.debug("No window indicators provided for daily summary")
        return None

    window_size_minutes = getattr(config.window, "size_minutes", 15)

    # Filter by target date if specified
    if target_date:
        window_indicators = [
            wi for wi in window_indicators if wi.window_start.date() == target_date
        ]
        if not window_indicators:
            _logger.debug("No window indicators for date %s", target_date)
            return None
    else:
        # Infer date from first window
        target_date = window_indicators[0].window_start.date()
        window_indicators = [
            wi for wi in window_indicators if wi.window_start.date() == target_date
        ]

    # Validate all windows are for same indicator
    indicator_name = window_indicators[0].indicator_name
    if not all(wi.indicator_name == indicator_name for wi in window_indicators):
        raise ValueError("All window indicators must be for the same indicator")

    # Sort chronologically
    window_indicators = sorted(window_indicators, key=lambda wi: wi.window_start)

    # Extract scores
    scores = [wi.indicator_score for wi in window_indicators]
    total_windows = len(scores)

    if total_windows == 0:
        return None

    # Daily likelihood = max mean of any k consecutive windows
    k = config.episode.peak_window_k
    likelihood = _max_consecutive_mean(scores, k)

    # Quality Metrics
    expected_windows = (24 * 60) // window_size_minutes
    data_coverage = total_windows / expected_windows if expected_windows > 0 else 0.0

    completeness_values = [wi.biomarker_completeness for wi in window_indicators]
    average_biomarker_completeness = (
        statistics.mean(completeness_values) if completeness_values else 0.0
    )

    high_confidence_contexts = sum(
        1 for wi in window_indicators if wi.dominant_context != "neutral"
    )
    context_availability = (
        high_confidence_contexts / total_windows if total_windows > 0 else 0.0
    )

    # Build window scores for traceability
    window_scores = tuple(
        {
            "window_start": wi.window_start.strftime("%H:%M"),
            "score": wi.indicator_score,
            "context": wi.dominant_context,
        }
        for wi in window_indicators
    )

    summary = DailyIndicatorSummary(
        date=target_date,
        indicator_name=indicator_name,
        likelihood=likelihood,
        window_scores=window_scores,
        total_windows=total_windows,
        expected_windows=expected_windows,
        data_coverage=data_coverage,
        average_biomarker_completeness=average_biomarker_completeness,
        context_availability=context_availability,
    )

    _logger.info(
        "Computed daily summary for '%s' on %s: "
        "likelihood=%.3f, coverage=%.1f%%",
        indicator_name,
        target_date,
        likelihood,
        data_coverage * 100,
    )

    return summary
