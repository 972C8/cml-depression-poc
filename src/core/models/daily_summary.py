"""Daily summary data models.

Story 6.5: Daily Summary Computation

Defines immutable data structure for aggregating window-level indicator
scores into a daily likelihood via consecutive-window peak mean.
"""

from dataclasses import dataclass
from datetime import date

__all__ = [
    "DailyIndicatorSummary",
]


@dataclass(frozen=True)
class DailyIndicatorSummary:
    """Daily summary of window indicator scores.

    Aggregates window-level FASL indicator scores into a single daily
    likelihood using a consecutive-window peak mean: the maximum mean
    of any k consecutive windows (k = episode.peak_window_k).

    Story 6.5: Daily Summary Computation

    Attributes:
        date: Date of the summary
        indicator_name: Name of the indicator (e.g., "social_withdrawal")
        likelihood: Max mean of k consecutive window FASL scores [0, 1]
        total_windows: Total number of windows with data
        expected_windows: Expected windows for full day (e.g., 96 for 15-min)
        data_coverage: Ratio of actual to expected windows (total / expected)
        average_biomarker_completeness: Mean biomarker completeness across windows
        context_availability: Proportion of windows with non-neutral context
    """

    # Identity
    date: date
    indicator_name: str

    # Daily likelihood (max mean of k consecutive windows)
    likelihood: float

    # Window-level scores used in likelihood computation
    # Each entry: {"window_start": "HH:MM", "score": float, "context": str}
    window_scores: tuple[dict, ...]

    # Quality Metrics
    total_windows: int
    expected_windows: int
    data_coverage: float
    average_biomarker_completeness: float
    context_availability: float
