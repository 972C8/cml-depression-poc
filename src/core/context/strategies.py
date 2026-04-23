"""Context strategy implementations for window membership computation.

Story 6.3: Membership with Context Weighting (AC2, AC3, AC4)

This module provides three strategies for determining the context state
within a time window:
- Dominant: Use context at window midpoint
- Time-weighted: Blend weights by time proportion within window
- Reading-weighted: Average weights across individual readings

Each strategy returns context information that will be used to weight
biomarker membership values.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from src.core.config import AnalysisConfig
from src.core.context.history import ContextHistoryService

__all__ = [
    "ContextStrategyResult",
    "get_window_context",
    "get_window_context_dominant",
    "get_window_context_reading_weighted",
    "get_window_context_time_weighted",
]

logger = logging.getLogger(__name__)

# Type alias for context strategy names
ContextStrategy = Literal["dominant", "time_weighted", "reading_weighted"]


@dataclass(frozen=True)
class ContextStrategyResult:
    """Result of applying a context strategy to a window.

    Contains the resolved context state and metadata about how it was determined.

    Attributes:
        dominant_context: The highest-confidence context name
        context_state: Dict of all context names to their confidence/weight scores
        context_weight: The weight multiplier for the target indicator
        confidence: Raw float confidence in [0.0, 1.0]
        strategy_used: Which strategy was applied
    """

    dominant_context: str
    context_state: dict[str, float]
    context_weight: float
    confidence: float
    strategy_used: ContextStrategy


def get_window_context(
    user_id: str,
    window_start: datetime,
    window_end: datetime,
    indicator_name: str,
    session: Session,
    config: AnalysisConfig,
    strategy: ContextStrategy = "dominant",
    readings_timestamps: tuple[datetime, ...] | None = None,
    context_evaluation_run_id: uuid.UUID | None = None,
) -> ContextStrategyResult:
    """Get context for a window using the specified strategy.

    Dispatcher function that routes to the appropriate strategy implementation.

    Story 6.14: When context_evaluation_run_id is provided, context queries
    are filtered to only use records from that specific run.

    Args:
        user_id: User identifier
        window_start: Start of the time window (inclusive)
        window_end: End of the time window (exclusive)
        indicator_name: Target indicator for weight lookup
        session: SQLAlchemy database session
        config: Analysis configuration
        strategy: Strategy to use ('dominant', 'time_weighted', 'reading_weighted')
        readings_timestamps: Required for 'reading_weighted' strategy
        context_evaluation_run_id: Optional UUID to filter context queries

    Returns:
        ContextStrategyResult with resolved context and weight

    Raises:
        ValueError: If strategy is unknown or readings_timestamps missing for reading_weighted
    """
    if strategy == "dominant":
        return get_window_context_dominant(
            user_id=user_id,
            window_start=window_start,
            window_end=window_end,
            indicator_name=indicator_name,
            session=session,
            config=config,
            context_evaluation_run_id=context_evaluation_run_id,
        )
    elif strategy == "time_weighted":
        return get_window_context_time_weighted(
            user_id=user_id,
            window_start=window_start,
            window_end=window_end,
            indicator_name=indicator_name,
            session=session,
            config=config,
            context_evaluation_run_id=context_evaluation_run_id,
        )
    elif strategy == "reading_weighted":
        if readings_timestamps is None:
            raise ValueError(
                "readings_timestamps required for reading_weighted strategy"
            )
        return get_window_context_reading_weighted(
            user_id=user_id,
            readings_timestamps=readings_timestamps,
            indicator_name=indicator_name,
            session=session,
            config=config,
            context_evaluation_run_id=context_evaluation_run_id,
        )
    else:
        raise ValueError(f"Unknown context strategy: {strategy}")


def get_window_context_dominant(
    user_id: str,
    window_start: datetime,
    window_end: datetime,
    indicator_name: str,
    session: Session,
    config: AnalysisConfig,
    context_evaluation_run_id: uuid.UUID | None = None,
) -> ContextStrategyResult:
    """Get context using the dominant (midpoint) strategy.

    Uses the context active at the window midpoint. This is the simplest
    and most efficient strategy.

    AC2: Dominant Strategy
    - Use context active at window midpoint
    - Look up via get_context_at_timestamp() from Story 6.1
    - Return ContextState with dominant context and confidence

    Story 6.14: When context_evaluation_run_id is provided, only context
    records from that run are considered.

    Args:
        user_id: User identifier
        window_start: Start of the time window (inclusive)
        window_end: End of the time window (exclusive)
        indicator_name: Target indicator for weight lookup
        session: SQLAlchemy database session
        config: Analysis configuration
        context_evaluation_run_id: Optional UUID to filter context queries

    Returns:
        ContextStrategyResult with context at window midpoint
    """
    # Calculate window midpoint
    window_duration = window_end - window_start
    midpoint = window_start + window_duration / 2

    # Get context history service (with optional run_id filter)
    history_service = ContextHistoryService(
        session, config, context_evaluation_run_id=context_evaluation_run_id
    )

    # Look up context at midpoint
    context_state = history_service.get_context_at_timestamp(
        user_id=user_id,
        timestamp=midpoint,
    )

    # Handle missing context
    if context_state is None:
        return _create_neutral_result(config, "dominant")

    # Check staleness
    staleness_hours = config.context_history.staleness_hours
    age = midpoint - context_state.timestamp
    age_hours = age.total_seconds() / 3600

    if age_hours > staleness_hours:
        # Stale - use neutral
        return _create_neutral_result(config, "dominant")

    # Pass through raw float confidence from context history record
    confidence = context_state.confidence

    # Get context weight for indicator
    context_weight = _get_context_weight_for_indicator(
        context_state.dominant_context, indicator_name, config
    )

    return ContextStrategyResult(
        dominant_context=context_state.dominant_context,
        context_state=context_state.all_scores,
        context_weight=context_weight,
        confidence=confidence,
        strategy_used="dominant",
    )


def get_window_context_time_weighted(
    user_id: str,
    window_start: datetime,
    window_end: datetime,
    indicator_name: str,
    session: Session,
    config: AnalysisConfig,
    context_evaluation_run_id: uuid.UUID | None = None,
) -> ContextStrategyResult:
    """Get context using time-weighted blending.

    Gets the full timeline via get_context_timeline() and blends weights
    by time proportion within the window.

    AC3: Time-Weighted Strategy
    - Get full timeline via get_context_timeline() from Story 6.1
    - Blend weights by time proportion within window
    - Return weighted average of context weights

    Story 6.14: When context_evaluation_run_id is provided, only context
    records from that run are included in the timeline.

    Args:
        user_id: User identifier
        window_start: Start of the time window (inclusive)
        window_end: End of the time window (exclusive)
        indicator_name: Target indicator for weight lookup
        session: SQLAlchemy database session
        config: Analysis configuration
        context_evaluation_run_id: Optional UUID to filter context queries

    Returns:
        ContextStrategyResult with time-weighted context blend
    """
    # Get context history service (with optional run_id filter)
    history_service = ContextHistoryService(
        session, config, context_evaluation_run_id=context_evaluation_run_id
    )

    # Get context timeline for window
    timeline = history_service.get_context_timeline(
        user_id=user_id,
        start=window_start,
        end=window_end,
    )

    # Handle empty timeline
    if not timeline:
        return _create_neutral_result(config, "time_weighted")

    # Calculate window duration in seconds for proportion calculation
    window_duration_seconds = (window_end - window_start).total_seconds()
    if window_duration_seconds <= 0:
        return _create_neutral_result(config, "time_weighted")

    # Blend context weights by time proportion
    blended_weights: dict[str, float] = {}
    total_covered_seconds = 0.0

    for segment in timeline:
        # Calculate segment overlap with window
        segment_start = max(segment.start, window_start)
        segment_end = min(segment.end, window_end)

        # Ensure we have valid overlap
        if segment_end <= segment_start:
            continue

        overlap_seconds = (segment_end - segment_start).total_seconds()
        proportion = overlap_seconds / window_duration_seconds
        total_covered_seconds += overlap_seconds

        # Get weight for this context
        context_weight = _get_context_weight_for_indicator(
            segment.context, indicator_name, config
        )

        # Accumulate weighted contribution
        if segment.context not in blended_weights:
            blended_weights[segment.context] = 0.0
        blended_weights[segment.context] += context_weight * proportion

    # Handle case where timeline doesn't cover entire window
    if total_covered_seconds < window_duration_seconds:
        uncovered_proportion = 1.0 - (total_covered_seconds / window_duration_seconds)
        # Use neutral weight for uncovered portion
        neutral_weight = config.context_history.neutral_weight
        if "neutral" not in blended_weights:
            blended_weights["neutral"] = 0.0
        blended_weights["neutral"] += neutral_weight * uncovered_proportion

    # Find dominant context (highest weighted contribution)
    dominant_context = max(blended_weights, key=lambda x: blended_weights[x])
    total_weight = sum(blended_weights.values())

    # Use coverage ratio directly as confidence
    coverage = total_covered_seconds / window_duration_seconds

    return ContextStrategyResult(
        dominant_context=dominant_context,
        context_state=blended_weights,
        context_weight=total_weight,
        confidence=coverage,
        strategy_used="time_weighted",
    )


def get_window_context_reading_weighted(
    user_id: str,
    readings_timestamps: tuple[datetime, ...],
    indicator_name: str,
    session: Session,
    config: AnalysisConfig,
    context_evaluation_run_id: uuid.UUID | None = None,
) -> ContextStrategyResult:
    """Get context using reading-weighted averaging.

    Looks up context for each reading timestamp and averages the weights.
    Most granular but computationally expensive (O(readings) lookups).

    AC4: Reading-Weighted Strategy
    - Look up context for each reading timestamp
    - Average weights across readings
    - Most granular but computationally expensive

    Story 6.14: When context_evaluation_run_id is provided, only context
    records from that run are used for lookups.

    Args:
        user_id: User identifier
        readings_timestamps: Timestamps of all readings in the window
        indicator_name: Target indicator for weight lookup
        session: SQLAlchemy database session
        config: Analysis configuration
        context_evaluation_run_id: Optional UUID to filter context queries

    Returns:
        ContextStrategyResult with reading-weighted context average
    """
    if not readings_timestamps:
        return _create_neutral_result(config, "reading_weighted")

    # Get context history service (with optional run_id filter)
    history_service = ContextHistoryService(
        session, config, context_evaluation_run_id=context_evaluation_run_id
    )

    # Collect context weights for each reading
    context_counts: dict[str, int] = {}
    weight_sum = 0.0
    readings_with_context = 0

    for timestamp in readings_timestamps:
        context_state = history_service.get_context_at_timestamp(
            user_id=user_id,
            timestamp=timestamp,
        )

        if context_state is not None:
            # Get weight for this context
            context_weight = _get_context_weight_for_indicator(
                context_state.dominant_context, indicator_name, config
            )
            weight_sum += context_weight
            readings_with_context += 1

            # Track context occurrences
            if context_state.dominant_context not in context_counts:
                context_counts[context_state.dominant_context] = 0
            context_counts[context_state.dominant_context] += 1
        else:
            # Use neutral for readings without context
            neutral_weight = config.context_history.neutral_weight
            weight_sum += neutral_weight

            if "neutral" not in context_counts:
                context_counts["neutral"] = 0
            context_counts["neutral"] += 1

    # Calculate average weight
    total_readings = len(readings_timestamps)
    avg_weight = weight_sum / total_readings if total_readings > 0 else 1.0

    # Find dominant context (most frequent)
    dominant_context = (
        max(context_counts, key=lambda x: context_counts[x])
        if context_counts
        else "neutral"
    )

    # Build context state (proportions)
    context_state = {
        ctx: count / total_readings for ctx, count in context_counts.items()
    }

    # Use coverage ratio directly as confidence
    coverage = readings_with_context / total_readings if total_readings > 0 else 0.0

    return ContextStrategyResult(
        dominant_context=dominant_context,
        context_state=context_state,
        context_weight=avg_weight,
        confidence=coverage,
        strategy_used="reading_weighted",
    )


def _create_neutral_result(
    config: AnalysisConfig,
    strategy: ContextStrategy,
) -> ContextStrategyResult:
    """Create a neutral context result for fallback cases.

    Used when no context history exists or context is stale.

    Args:
        config: Analysis configuration
        strategy: Strategy being used

    Returns:
        ContextStrategyResult with neutral context and weight=1.0
    """
    neutral_weight = config.context_history.neutral_weight
    return ContextStrategyResult(
        dominant_context="neutral",
        context_state={"neutral": neutral_weight},
        context_weight=neutral_weight,
        confidence=0.0,
        strategy_used=strategy,
    )


def _get_context_weight_for_indicator(
    context_name: str,
    indicator_name: str,
    config: AnalysisConfig,
) -> float:
    """Get the context weight multiplier for a specific indicator.

    Looks up the weight from config.context_weights[context][biomarker].
    For windowed analysis, we use a single weight per indicator rather
    than per-biomarker weights.

    Args:
        context_name: Name of the active context
        indicator_name: Target indicator name
        config: Analysis configuration

    Returns:
        Weight multiplier (defaults to 1.0 if not configured)
    """
    # Get context-specific weights
    context_weights = config.context_weights.get(context_name, {})

    # For indicator-level weighting, we could average across biomarkers
    # or use a dedicated indicator weight. For now, return 1.0 as base
    # and let the actual biomarker weights be applied during FASL.

    # Check if there's an indicator-level override
    if indicator_name in context_weights:
        return context_weights[indicator_name]

    # Default: neutral weight
    return config.context_history.neutral_weight
