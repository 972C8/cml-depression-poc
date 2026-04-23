"""Window aggregation functions.

Story 6.2: Window Aggregation Module (AC2, AC3, AC4, AC5)

Provides functions for aggregating biomarker readings into fixed time windows
using tumbling window approach with configurable aggregation methods.
"""

import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from src.core.data_reader import BiomarkerRecord
from src.core.models.window_models import WindowAggregate

logger = logging.getLogger(__name__)

__all__ = [
    "floor_to_window",
    "aggregate_into_windows",
]

# Valid window sizes in minutes
VALID_WINDOW_SIZES = frozenset({5, 10, 15, 30, 60})


def floor_to_window(timestamp: datetime, window_minutes: int) -> datetime:
    """Floor timestamp to window boundary.

    Aligns a timestamp to the nearest window boundary that is at or before
    the given timestamp. Windows are aligned to clock time.

    Args:
        timestamp: Input timestamp (must be timezone-aware)
        window_minutes: Window size in minutes (must be 5, 10, 15, 30, or 60)

    Returns:
        Timestamp floored to window boundary, preserving timezone

    Raises:
        ValueError: If window_minutes is not a valid size

    Example:
        >>> floor_to_window(datetime(2025, 1, 15, 9, 23, 41, tzinfo=UTC), 15)
        datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
    """
    if window_minutes not in VALID_WINDOW_SIZES:
        raise ValueError(
            f"Invalid window_minutes={window_minutes}. "
            f"Must be one of: {sorted(VALID_WINDOW_SIZES)}"
        )

    # Calculate minutes since midnight
    minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
    # Floor to window boundary
    floored_minutes = (minutes_since_midnight // window_minutes) * window_minutes

    return timestamp.replace(
        hour=floored_minutes // 60,
        minute=floored_minutes % 60,
        second=0,
        microsecond=0,
    )


def _compute_aggregate(
    values: list[float],
    method: str,
) -> float:
    """Compute aggregate value using specified method.

    Args:
        values: List of values to aggregate (must be non-empty)
        method: Aggregation method ("mean", "median", "max", "min")

    Returns:
        Aggregated value

    Raises:
        ValueError: If method is not recognized
    """
    if method == "mean":
        return statistics.mean(values)
    elif method == "median":
        return statistics.median(values)
    elif method == "max":
        return max(values)
    elif method == "min":
        return min(values)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_into_windows(
    records: list[BiomarkerRecord],
    window_size_minutes: int = 15,
    aggregation_method: str = "mean",
    min_readings: int = 1,
) -> dict[str, list[WindowAggregate]]:
    """Aggregate biomarker records into time windows.

    Groups biomarker records by name and then by time window, computing
    an aggregate value for each window using the specified method.

    Uses O(N) algorithm with defaultdict for efficient grouping.

    Args:
        records: List of BiomarkerRecord from DataReader
        window_size_minutes: Window size in minutes (default: 15)
        aggregation_method: Aggregation method ("mean", "median", "max", "min")
        min_readings: Minimum readings required for valid window (default: 1)

    Returns:
        Dict mapping biomarker names to lists of WindowAggregate,
        sorted by window_start within each biomarker

    Raises:
        ValueError: If aggregation_method is not recognized or min_readings < 1

    Example:
        >>> result = aggregate_into_windows(records, 15, "mean")
        >>> result["speech_activity"][0].aggregated_value
        0.6
    """
    if not records:
        logger.debug("aggregate_into_windows called with empty records")
        return {}

    # Validate min_readings
    if min_readings < 1:
        raise ValueError(f"min_readings must be >= 1, got {min_readings}")

    # Validate aggregation method early
    valid_methods = {"mean", "median", "max", "min"}
    if aggregation_method not in valid_methods:
        raise ValueError(
            f"Unknown aggregation method: {aggregation_method}. "
            f"Must be one of: {valid_methods}"
        )

    logger.debug(
        "Aggregating %d records into %d-minute windows using %s (min_readings=%d)",
        len(records),
        window_size_minutes,
        aggregation_method,
        min_readings,
    )

    # Group records by (biomarker_name, window_start)
    # Structure: {biomarker_name: {window_start: [(value, timestamp), ...]}}
    grouped: dict[str, dict[datetime, list[tuple[float, datetime]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in records:
        window_start = floor_to_window(record.timestamp, window_size_minutes)
        grouped[record.name][window_start].append((record.value, record.timestamp))

    # Build WindowAggregate instances
    result: dict[str, list[WindowAggregate]] = {}
    skipped_windows = 0

    for biomarker_name, windows in grouped.items():
        aggregates = []
        for window_start, readings in windows.items():
            # Skip windows with fewer than min_readings (AC6)
            if len(readings) < min_readings:
                skipped_windows += 1
                continue

            values = [r[0] for r in readings]
            timestamps = tuple(r[1] for r in readings)

            window_end = window_start + timedelta(minutes=window_size_minutes)

            aggregate = WindowAggregate(
                biomarker_name=biomarker_name,
                window_start=window_start,
                window_end=window_end,
                aggregated_value=_compute_aggregate(values, aggregation_method),
                readings_count=len(readings),
                readings_timestamps=timestamps,
                aggregation_method=aggregation_method,
            )
            aggregates.append(aggregate)

        # Sort by window_start
        aggregates.sort(key=lambda x: x.window_start)
        result[biomarker_name] = aggregates

    if skipped_windows > 0:
        logger.debug(
            "Skipped %d windows with fewer than %d readings",
            skipped_windows,
            min_readings,
        )

    logger.debug(
        "Aggregated into %d biomarkers with %d total windows",
        len(result),
        sum(len(w) for w in result.values()),
    )

    return result
