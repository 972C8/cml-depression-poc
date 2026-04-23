"""Data queries for Analysis Pipeline Transparency feature.

Story 6.11: Analysis Pipeline Transparency

Provides data access functions for querying indicator computation details
from the database to enable step-by-step transparency visualization.
"""

import statistics
from datetime import date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB

from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import ContextHistoryRecord, Indicator

logger = get_logger(__name__)

__all__ = [
    "get_available_daily_indicators",
    "get_daily_indicator_summary",
    "get_context_history_for_date",
    "get_biomarker_aggregates_for_date",
    "get_window_aggregates_for_date",
    "get_indicator_config",
    "get_all_window_times",
    "get_baselines_for_user",
    "get_biomarker_defaults",
    "get_context_weights_config",
    "compute_membership_for_display",
    "compute_fasl_for_display",
    "get_all_window_fasl_scores",
    "get_window_indicator_details",
    "get_window_indicator_times",
    "BiomarkerAggregateStats",
    "WindowAggregateInfo",
    "MembershipComputation",
    "FASLComputation",
    "DailyIndicatorOption",
]


class DailyIndicatorOption:
    """Represents a daily indicator option for the selector dropdown.

    Attributes:
        indicator_date: Date of the indicator
        indicator_name: Name of the indicator (e.g., "social_withdrawal")
        display_label: Formatted label for dropdown (e.g., "2026-01-22 - social_withdrawal")
        indicator_id: UUID of the Indicator record
    """

    def __init__(
        self,
        indicator_date: date,
        indicator_name: str,
        indicator_id: str,
    ) -> None:
        self.indicator_date = indicator_date
        self.indicator_name = indicator_name
        self.indicator_id = indicator_id
        self.display_label = f"{indicator_date.isoformat()} - {indicator_name}"

    def __repr__(self) -> str:
        return f"DailyIndicatorOption({self.display_label})"


def get_available_daily_indicators(user_id: str) -> list[DailyIndicatorOption]:
    """Get list of available daily indicators for dropdown selection.

    Queries Indicator records that were created by windowed analysis pipeline
    (identified by computation_log.source = "windowed_analysis").

    Args:
        user_id: User ID to filter by

    Returns:
        List of DailyIndicatorOption sorted by date (most recent first),
        then by indicator name alphabetically. Empty list if no data.
    """
    try:
        with SessionLocal() as db:
            # Query indicators where source is windowed_analysis
            # Using JSONB containment operator for efficient querying
            query = (
                select(Indicator)
                .where(Indicator.user_id == user_id)
                .where(
                    Indicator.computation_log.cast(JSONB).contains(
                        {"source": "windowed_analysis"}
                    )
                )
                .order_by(Indicator.timestamp.desc(), Indicator.indicator_type, Indicator.id.desc())
            )

            results = db.execute(query).scalars().all()

        options = []
        seen: set[tuple[date, str]] = set()  # (date, indicator_type) pairs

        for ind in results:
            # Extract date from timestamp
            indicator_date = ind.timestamp.date()
            key = (indicator_date, ind.indicator_type)

            # Skip duplicates - keep only the first (most recent) for each date/indicator
            if key in seen:
                continue
            seen.add(key)

            options.append(
                DailyIndicatorOption(
                    indicator_date=indicator_date,
                    indicator_name=ind.indicator_type,
                    indicator_id=str(ind.id),
                )
            )

        return options

    except Exception:
        logger.error(
            "Failed to get available daily indicators for user %s",
            user_id,
            exc_info=True,
        )
        return []


def get_daily_indicator_summary(indicator_id: str) -> dict[str, Any] | None:
    """Get full daily indicator summary data for a specific indicator.

    Retrieves the complete computation_log which contains all the
    multi-dimensional metrics stored by save_daily_summaries().

    Args:
        indicator_id: UUID string of the Indicator record

    Returns:
        Dict containing the full computation_log with severity, duration,
        persistence, quality metrics and likelihood. None if not found.
    """
    try:
        with SessionLocal() as db:
            from uuid import UUID

            query = select(Indicator).where(Indicator.id == UUID(indicator_id))
            result = db.execute(query).scalar_one_or_none()

            if result is None:
                return None

            # Return the full computation log plus some top-level fields
            summary = result.computation_log.copy() if result.computation_log else {}
            summary["indicator_id"] = str(result.id)
            summary["user_id"] = result.user_id
            summary["timestamp"] = result.timestamp.isoformat()
            summary["analysis_run_id"] = str(result.analysis_run_id)
            summary["value"] = result.value  # The likelihood value

            return summary

    except Exception:
        logger.error(
            "Failed to get daily indicator summary for %s",
            indicator_id,
            exc_info=True,
        )
        return None


def get_context_history_for_date(
    user_id: str,
    target_date: date,
) -> list[dict[str, Any]]:
    """Get all context evaluations for a specific date.

    Queries ContextHistoryRecord for all evaluations on the given date.

    Args:
        user_id: User ID to filter by
        target_date: Date to get context history for

    Returns:
        List of dicts with context history data, sorted by evaluation time.
        Each dict contains: evaluated_at, dominant_context, confidence,
        context_state, raw_scores, and stabilization fields.
    """
    try:
        with SessionLocal() as db:
            # Build datetime range for the target date
            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())

            query = (
                select(ContextHistoryRecord)
                .where(ContextHistoryRecord.user_id == user_id)
                .where(ContextHistoryRecord.evaluated_at >= start_dt)
                .where(ContextHistoryRecord.evaluated_at <= end_dt)
                .order_by(ContextHistoryRecord.evaluated_at)
            )

            results = db.execute(query).scalars().all()

        history = []
        for record in results:
            history.append(
                {
                    "id": str(record.id),
                    "evaluated_at": record.evaluated_at,
                    "dominant_context": record.dominant_context,
                    "confidence": record.confidence,
                    "context_state": record.context_state,
                    "raw_scores": record.raw_scores,
                    "evaluation_trigger": record.evaluation_trigger,
                    "sensors_used": record.sensors_used,
                    "switch_blocked": record.switch_blocked,
                    "switch_blocked_reason": record.switch_blocked_reason,
                    "candidate_context": record.candidate_context,
                    "score_difference": record.score_difference,
                    "dwell_progress": record.dwell_progress,
                    "dwell_required": record.dwell_required,
                }
            )

        return history

    except Exception:
        logger.error(
            "Failed to get context history for user %s on %s",
            user_id,
            target_date,
            exc_info=True,
        )
        return []


class BiomarkerAggregateStats:
    """Statistics about a biomarker's window aggregates for a day.

    Attributes:
        biomarker_name: Name of the biomarker
        windows_with_data: Number of windows containing data
        expected_windows: Expected windows for full day (96 for 15-min)
        coverage: Percentage of expected windows with data
        daily_mean: Mean of aggregated values across windows
        daily_std: Standard deviation of aggregated values
        value_min: Minimum aggregated value
        value_max: Maximum aggregated value
    """

    def __init__(
        self,
        biomarker_name: str,
        windows_with_data: int,
        expected_windows: int,
        daily_mean: float,
        daily_std: float,
        value_min: float,
        value_max: float,
    ) -> None:
        self.biomarker_name = biomarker_name
        self.windows_with_data = windows_with_data
        self.expected_windows = expected_windows
        self.coverage = windows_with_data / expected_windows if expected_windows > 0 else 0.0
        self.daily_mean = daily_mean
        self.daily_std = daily_std
        self.value_min = value_min
        self.value_max = value_max


def get_biomarker_aggregates_for_date(
    user_id: str,
    target_date: date,
    window_size_minutes: int = 15,
) -> list[BiomarkerAggregateStats]:
    """Get window-aggregated biomarker statistics for a specific date.

    Computes on-demand since WindowAggregate is not persisted.

    Args:
        user_id: User ID to filter by
        target_date: Date to get biomarker data for
        window_size_minutes: Window size in minutes (default: 15)

    Returns:
        List of BiomarkerAggregateStats sorted by biomarker name.
        Empty list if no data.
    """
    try:
        with SessionLocal() as db:
            from src.core.data_reader import DataReader
            from src.core.processors.window_aggregator import aggregate_into_windows

            # Build datetime range for the target date
            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())

            # Read raw biomarker data
            reader = DataReader(db)
            records = reader.read_biomarkers(
                user_id=user_id,
                start_time=start_dt,
                end_time=end_dt,
            )

            if not records:
                return []

            # Aggregate into windows
            aggregates_by_biomarker = aggregate_into_windows(
                records=records,
                window_size_minutes=window_size_minutes,
                aggregation_method="mean",
            )

            # Calculate expected windows for a full day
            expected_windows = (24 * 60) // window_size_minutes

            # Build stats for each biomarker
            stats_list = []
            for biomarker_name, aggregates in aggregates_by_biomarker.items():
                if not aggregates:
                    continue

                values = [agg.aggregated_value for agg in aggregates]
                windows_count = len(aggregates)

                # Calculate std, handling single-value case
                if len(values) > 1:
                    std = statistics.stdev(values)
                else:
                    std = 0.0

                stats = BiomarkerAggregateStats(
                    biomarker_name=biomarker_name,
                    windows_with_data=windows_count,
                    expected_windows=expected_windows,
                    daily_mean=statistics.mean(values),
                    daily_std=std,
                    value_min=min(values),
                    value_max=max(values),
                )
                stats_list.append(stats)

            # Sort by biomarker name
            stats_list.sort(key=lambda x: x.biomarker_name)
            return stats_list

    except Exception:
        logger.error(
            "Failed to get biomarker aggregates for user %s on %s",
            user_id,
            target_date,
            exc_info=True,
        )
        return []


class WindowAggregateInfo:
    """Information about a single window aggregate.

    Attributes:
        biomarker_name: Name of the biomarker
        window_start: Start time of the window
        window_end: End time of the window
        aggregated_value: Computed aggregate value
        readings_count: Number of readings in this window
        aggregation_method: Method used (e.g., "mean")
    """

    def __init__(
        self,
        biomarker_name: str,
        window_start: datetime,
        window_end: datetime,
        aggregated_value: float,
        readings_count: int,
        aggregation_method: str,
    ) -> None:
        self.biomarker_name = biomarker_name
        self.window_start = window_start
        self.window_end = window_end
        self.aggregated_value = aggregated_value
        self.readings_count = readings_count
        self.aggregation_method = aggregation_method

    @property
    def window_label(self) -> str:
        """Format window as HH:MM - HH:MM."""
        return f"{self.window_start.strftime('%H:%M')} - {self.window_end.strftime('%H:%M')}"


def get_all_window_times(
    user_id: str,
    target_date: date,
    window_size_minutes: int = 15,
) -> list[tuple[datetime, datetime]]:
    """Get all unique window times for a date.

    Returns list of (window_start, window_end) tuples sorted by time.
    Used for window selector dropdowns.

    Args:
        user_id: User ID
        target_date: Date to get windows for
        window_size_minutes: Window size in minutes

    Returns:
        List of (start, end) datetime tuples, sorted by start time
    """
    try:
        with SessionLocal() as db:

            from src.core.data_reader import DataReader
            from src.core.processors.window_aggregator import aggregate_into_windows

            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())

            reader = DataReader(db)
            records = reader.read_biomarkers(
                user_id=user_id,
                start_time=start_dt,
                end_time=end_dt,
            )

            if not records:
                return []

            aggregates_by_biomarker = aggregate_into_windows(
                records=records,
                window_size_minutes=window_size_minutes,
            )

            # Collect all unique window times
            window_times: set[tuple[datetime, datetime]] = set()
            for aggregates in aggregates_by_biomarker.values():
                for agg in aggregates:
                    window_times.add((agg.window_start, agg.window_end))

            # Sort by start time (most recent first for dropdown)
            return sorted(window_times, key=lambda x: x[0], reverse=True)

    except Exception:
        logger.error(
            "Failed to get window times for user %s on %s",
            user_id,
            target_date,
            exc_info=True,
        )
        return []


def get_window_aggregates_for_date(
    user_id: str,
    target_date: date,
    window_size_minutes: int = 15,
) -> dict[datetime, dict[str, WindowAggregateInfo]]:
    """Get all window aggregates for a date, organized by window start time.

    Returns dict: {window_start: {biomarker_name: WindowAggregateInfo}}

    Args:
        user_id: User ID
        target_date: Date to get aggregates for
        window_size_minutes: Window size in minutes

    Returns:
        Dict mapping window_start to dict of biomarker name to WindowAggregateInfo
    """
    try:
        with SessionLocal() as db:
            from src.core.data_reader import DataReader
            from src.core.processors.window_aggregator import aggregate_into_windows

            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())

            reader = DataReader(db)
            records = reader.read_biomarkers(
                user_id=user_id,
                start_time=start_dt,
                end_time=end_dt,
            )

            if not records:
                return {}

            aggregates_by_biomarker = aggregate_into_windows(
                records=records,
                window_size_minutes=window_size_minutes,
            )

            # Reorganize by window
            result: dict[datetime, dict[str, WindowAggregateInfo]] = {}
            for biomarker_name, aggregates in aggregates_by_biomarker.items():
                for agg in aggregates:
                    if agg.window_start not in result:
                        result[agg.window_start] = {}

                    result[agg.window_start][biomarker_name] = WindowAggregateInfo(
                        biomarker_name=biomarker_name,
                        window_start=agg.window_start,
                        window_end=agg.window_end,
                        aggregated_value=agg.aggregated_value,
                        readings_count=agg.readings_count,
                        aggregation_method=agg.aggregation_method,
                    )

            return result

    except Exception:
        logger.error(
            "Failed to get window aggregates for user %s on %s",
            user_id,
            target_date,
            exc_info=True,
        )
        return {}


def get_indicator_config(indicator_name: str) -> dict | None:
    """Get indicator configuration from config/indicators.yaml.

    Args:
        indicator_name: Name of the indicator (e.g., "social_withdrawal")

    Returns:
        Dict with indicator configuration including biomarkers, weights, directions.
        None if not found or error.
    """
    try:
        from pathlib import Path

        import yaml

        config_path = Path("config/indicators.yaml")
        if not config_path.exists():
            logger.warning("Indicator config not found: %s", config_path)
            return None

        with open(config_path) as f:
            all_configs = yaml.safe_load(f)

        return all_configs.get(indicator_name)

    except Exception:
        logger.error(
            "Failed to load indicator config for %s",
            indicator_name,
            exc_info=True,
        )
        return None


def get_baselines_for_user(user_id: str) -> dict[str, dict]:
    """Get all baseline values for a user.

    Args:
        user_id: User ID

    Returns:
        Dict mapping biomarker name to {"mean": x, "std": y, "source": "user-specific"}
    """
    try:
        with SessionLocal() as db:
            from src.core.processors.baseline_repository import BaselineRepository

            repo = BaselineRepository(db)
            baselines = repo.get_all_baselines(user_id)

            return {
                name: {
                    "mean": baseline.mean,
                    "std": baseline.std,
                    "source": "user-specific",
                    "data_points": baseline.data_points,
                }
                for name, baseline in baselines.items()
            }

    except Exception:
        logger.error("Failed to get baselines for user %s", user_id, exc_info=True)
        return {}


def get_biomarker_defaults() -> dict[str, dict]:
    """Get population default baselines from config.

    Returns:
        Dict mapping biomarker name to {"mean": x, "std": y}
        Only includes entries that are dicts with mean/std keys.
    """
    try:
        from pathlib import Path

        import yaml

        config_path = Path("config/biomarker_defaults.yaml")
        if not config_path.exists():
            return {}

        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}

        # Filter to only include biomarker entries (dicts with mean/std)
        # Exclude global config params like z_score_bounds, generic_baseline, etc.
        biomarker_defaults = {}
        for key, value in all_config.items():
            if isinstance(value, dict) and "mean" in value and "std" in value:
                biomarker_defaults[key] = value

        return biomarker_defaults

    except Exception:
        logger.error("Failed to load biomarker defaults", exc_info=True)
        return {}


class MembershipComputation:
    """Holds membership computation details for display.

    Attributes:
        biomarker_name: Name of the biomarker
        aggregated_value: Value from window aggregate
        baseline_mean: Baseline mean used
        baseline_std: Baseline std used
        baseline_source: "user-specific" or "population-default"
        z_score: Computed z-score
        membership: Raw membership [0, 1]
        direction: "higher_is_worse" or "lower_is_worse"
        directed_membership: Membership after direction adjustment
    """

    def __init__(
        self,
        biomarker_name: str,
        aggregated_value: float,
        baseline_mean: float,
        baseline_std: float,
        baseline_source: str,
        z_score: float,
        membership: float,
        direction: str,
    ) -> None:
        self.biomarker_name = biomarker_name
        self.aggregated_value = aggregated_value
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.baseline_source = baseline_source
        self.z_score = z_score
        self.membership = membership
        self.direction = direction

        # Compute directed membership
        if direction == "lower_is_worse":
            self.directed_membership = 1.0 - membership
        else:
            self.directed_membership = membership


def compute_membership_for_display(
    user_id: str,
    target_date: date,
    indicator_name: str,
    window_start: datetime,
) -> list[MembershipComputation]:
    """Compute membership values for display in transparency UI.

    Args:
        user_id: User ID
        target_date: Date of analysis
        indicator_name: Indicator being analyzed
        window_start: Start time of the window to compute for

    Returns:
        List of MembershipComputation objects for each biomarker
    """
    import math

    try:
        # Get indicator config for biomarker list and directions
        indicator_config = get_indicator_config(indicator_name)
        if not indicator_config:
            return []

        biomarker_configs = indicator_config.get("biomarkers", {})

        # Get window aggregates
        all_aggregates = get_window_aggregates_for_date(user_id, target_date)
        window_aggregates = all_aggregates.get(window_start, {})

        # Get user baselines and defaults
        user_baselines = get_baselines_for_user(user_id)
        defaults = get_biomarker_defaults()

        results = []
        for bio_name, bio_config in biomarker_configs.items():
            direction = bio_config.get("direction", "higher_is_worse")

            # Get aggregated value
            if bio_name in window_aggregates:
                agg_value = window_aggregates[bio_name].aggregated_value
            else:
                # Missing biomarker - use neutral 0.5
                results.append(
                    MembershipComputation(
                        biomarker_name=bio_name,
                        aggregated_value=float("nan"),
                        baseline_mean=float("nan"),
                        baseline_std=float("nan"),
                        baseline_source="N/A (missing data)",
                        z_score=float("nan"),
                        membership=0.5,
                        direction=direction,
                    )
                )
                continue

            # Get baseline
            if bio_name in user_baselines:
                baseline = user_baselines[bio_name]
                mean = baseline["mean"]
                std = baseline["std"]
                source = "user-specific"
            elif bio_name in defaults:
                default = defaults[bio_name]
                mean = default.get("mean", 0.5)
                std = default.get("std", 0.1)
                source = "population-default"
            else:
                # Generic fallback
                mean = 0.5
                std = 0.2
                source = "generic-fallback"

            # Compute z-score
            if std > 0:
                z_score = (agg_value - mean) / std
            else:
                z_score = 0.0

            # Compute membership (sigmoid)
            membership = 1.0 / (1.0 + math.exp(-z_score))

            results.append(
                MembershipComputation(
                    biomarker_name=bio_name,
                    aggregated_value=agg_value,
                    baseline_mean=mean,
                    baseline_std=std,
                    baseline_source=source,
                    z_score=z_score,
                    membership=membership,
                    direction=direction,
                )
            )

        return results

    except Exception:
        logger.error(
            "Failed to compute memberships for user %s indicator %s",
            user_id,
            indicator_name,
            exc_info=True,
        )
        return []


def get_context_weights_config() -> dict[str, dict[str, float]]:
    """Get context weight configuration from config/context_weights.yaml.

    Returns:
        Dict mapping context name to dict of biomarker weights.
        Example: {"solitary_digital": {"speech_activity": 1.5, ...}}
    """
    try:
        from pathlib import Path

        import yaml

        config_path = Path("config/context_weights.yaml")
        if not config_path.exists():
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    except Exception:
        logger.error("Failed to load context weights config", exc_info=True)
        return {}


class FASLComputation:
    """Holds FASL computation details for display.

    Attributes:
        window_start: Window start time
        window_end: Window end time
        dominant_context: Active context for this window
        context_confidence: Confidence in context detection
        indicator_score: Final FASL indicator score [0, 1]
        contributions: List of contribution details per biomarker
        primary_driver: Biomarker with highest contribution
    """

    def __init__(
        self,
        window_start: datetime,
        window_end: datetime,
        dominant_context: str,
        context_confidence: float,
        indicator_score: float,
        contributions: list[dict],
    ) -> None:
        self.window_start = window_start
        self.window_end = window_end
        self.dominant_context = dominant_context
        self.context_confidence = context_confidence
        self.indicator_score = indicator_score
        self.contributions = contributions

        # Identify primary driver (highest contribution)
        if contributions:
            sorted_contribs = sorted(
                contributions, key=lambda x: x.get("contribution", 0), reverse=True
            )
            self.primary_driver = sorted_contribs[0].get("biomarker", "N/A")
            self.primary_contribution = sorted_contribs[0].get("contribution", 0)
        else:
            self.primary_driver = "N/A"
            self.primary_contribution = 0.0

    @property
    def window_label(self) -> str:
        """Format window as HH:MM - HH:MM."""
        return f"{self.window_start.strftime('%H:%M')} - {self.window_end.strftime('%H:%M')}"


def compute_fasl_for_display(
    user_id: str,
    target_date: date,
    indicator_name: str,
    window_start: datetime,
) -> FASLComputation | None:
    """Compute FASL indicator score for display in transparency UI.

    Args:
        user_id: User ID
        target_date: Date of analysis
        indicator_name: Indicator being analyzed
        window_start: Start time of the window to compute for

    Returns:
        FASLComputation object with all details, or None on error
    """
    import math

    try:
        # Get indicator config
        indicator_config = get_indicator_config(indicator_name)
        if not indicator_config:
            return None

        biomarker_configs = indicator_config.get("biomarkers", {})

        # Get memberships
        memberships = compute_membership_for_display(
            user_id=user_id,
            target_date=target_date,
            indicator_name=indicator_name,
            window_start=window_start,
        )

        if not memberships:
            return None

        # Get context for this window
        context_history = get_context_history_for_date(user_id, target_date)

        # Find context for this window (closest evaluation before or at window_start)
        dominant_context = "neutral"
        context_confidence = 0.0

        for record in reversed(context_history):
            eval_at = record.get("evaluated_at")
            if eval_at and eval_at <= window_start:
                dominant_context = record.get("dominant_context", "neutral")
                context_confidence = record.get("confidence", 0.0)
                break

        # Get context weights for this context
        context_weights_config = get_context_weights_config()
        context_biomarker_weights = context_weights_config.get(dominant_context, {})

        # Compute FASL contributions
        contributions = []
        numerator = 0.0
        denominator = 0.0

        for m in memberships:
            bio_config = biomarker_configs.get(m.biomarker_name, {})
            biomarker_weight = bio_config.get("weight", 0)

            # Get context weight
            context_weight = context_biomarker_weights.get(m.biomarker_name, 1.0)

            # Compute effective context weight
            effective_context_weight = 1.0 + (context_weight - 1.0) * context_confidence
            effective_weight = biomarker_weight * effective_context_weight

            if effective_weight == 0 or math.isnan(m.aggregated_value):
                contributions.append({
                    "biomarker": m.biomarker_name,
                    "directed_membership": m.directed_membership if not math.isnan(m.aggregated_value) else 0.5,
                    "biomarker_weight": biomarker_weight,
                    "context_weight": context_weight,
                    "effective_weight": effective_weight,
                    "contribution": 0.0,
                    "is_missing": math.isnan(m.aggregated_value),
                })
                if not math.isnan(m.aggregated_value):
                    denominator += effective_weight
                continue

            contribution = effective_weight * m.directed_membership
            numerator += contribution
            denominator += effective_weight

            contributions.append({
                "biomarker": m.biomarker_name,
                "directed_membership": m.directed_membership,
                "biomarker_weight": biomarker_weight,
                "context_weight": context_weight,
                "effective_weight": effective_weight,
                "contribution": contribution,
                "is_missing": False,
            })

        # Compute final score
        if denominator > 0:
            indicator_score = numerator / denominator
        else:
            indicator_score = 0.0

        # Calculate window_end
        from datetime import timedelta
        window_end = window_start + timedelta(minutes=15)

        return FASLComputation(
            window_start=window_start,
            window_end=window_end,
            dominant_context=dominant_context,
            context_confidence=context_confidence,
            indicator_score=indicator_score,
            contributions=contributions,
        )

    except Exception:
        logger.error(
            "Failed to compute FASL for user %s indicator %s",
            user_id,
            indicator_name,
            exc_info=True,
        )
        return None


def get_window_indicator_times(
    analysis_run_id: str,
    indicator_name: str,
) -> list[tuple[datetime, datetime]]:
    """Get available window timestamps for persisted window indicators.

    Story 6.17: Queries distinct window timestamps from persisted window
    indicator rows for the selector dropdown.

    Args:
        analysis_run_id: Analysis run UUID string
        indicator_name: Indicator name (e.g., "1_depressed_mood")

    Returns:
        List of (window_start, window_end) tuples, sorted by start time
        (most recent first). Empty list if no data.
    """
    try:
        from uuid import UUID

        indicator_type = f"{indicator_name}_window"

        with SessionLocal() as db:
            query = (
                select(Indicator)
                .where(Indicator.analysis_run_id == UUID(analysis_run_id))
                .where(Indicator.indicator_type == indicator_type)
                .order_by(Indicator.timestamp.desc())
            )

            results = db.execute(query).scalars().all()

        window_times = []
        for ind in results:
            comp_log = ind.computation_log or {}
            window_end_str = comp_log.get("window_end")
            if window_end_str:
                window_end = datetime.fromisoformat(window_end_str)
            else:
                # Fallback: assume 15-minute windows
                from datetime import timedelta
                window_end = ind.timestamp + timedelta(minutes=15)
            window_times.append((ind.timestamp, window_end))

        return window_times

    except Exception:
        logger.error(
            "Failed to get window indicator times for run %s indicator %s",
            analysis_run_id,
            indicator_name,
            exc_info=True,
        )
        return []


def get_window_indicator_details(
    analysis_run_id: str,
    indicator_name: str,
    window_start: datetime,
) -> dict[str, Any] | None:
    """Get persisted window indicator details for a specific window.

    Story 6.17: Queries the computation_log from a persisted window indicator
    row, which contains full FASL contribution details.

    Args:
        analysis_run_id: Analysis run UUID string
        indicator_name: Indicator name (e.g., "1_depressed_mood")
        window_start: Start time of the window

    Returns:
        Dict with full computation_log data including fasl_contributions,
        or None if not found.
    """
    try:
        from uuid import UUID

        indicator_type = f"{indicator_name}_window"

        with SessionLocal() as db:
            query = (
                select(Indicator)
                .where(Indicator.analysis_run_id == UUID(analysis_run_id))
                .where(Indicator.indicator_type == indicator_type)
                .where(Indicator.timestamp == window_start)
            )

            result = db.execute(query).scalar_one_or_none()

        if result is None:
            return None

        details = result.computation_log.copy() if result.computation_log else {}
        details["indicator_id"] = str(result.id)
        details["value"] = result.value

        return details

    except Exception:
        logger.error(
            "Failed to get window indicator details for run %s indicator %s at %s",
            analysis_run_id,
            indicator_name,
            window_start,
            exc_info=True,
        )
        return None


# Deprecated: compute_fasl_for_display and get_all_window_fasl_scores recompute
# independently from the pipeline. Step 6 now reads from persisted data via
# get_window_indicator_details and get_window_indicator_times (Story 6.17).


def get_all_window_fasl_scores(
    user_id: str,
    target_date: date,
    indicator_name: str,
    analysis_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get FASL indicator scores for all windows of a day.

    Uses the core pipeline functions to compute window-level FASL scores
    for every window in a single pass (efficient).

    Args:
        user_id: User ID
        target_date: Date to compute for
        indicator_name: Indicator name (e.g., "1_depressed_mood")
        analysis_run_id: Optional analysis run ID to look up the baseline used

    Returns:
        List of dicts with window_start, window_end, score, context,
        sorted by window_start. Empty list on error.
    """
    try:
        from pathlib import Path
        from uuid import UUID

        from src.core.baseline_config import load_baseline_file
        from src.core.config import get_default_config
        from src.core.data_reader import DataReader
        from src.core.processors.window_aggregator import aggregate_into_windows
        from src.core.processors.window_fasl import compute_window_indicators
        from src.core.processors.window_membership import compute_window_memberships

        config = get_default_config()
        indicator_config = config.indicators.get(indicator_name)
        if not indicator_config:
            return []

        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())

        with SessionLocal() as db:
            # Determine baseline file path from analysis run or fallback
            baseline_name = None
            if analysis_run_id:
                from src.shared.models import AnalysisRun

                run = db.execute(
                    select(AnalysisRun).where(AnalysisRun.id == UUID(analysis_run_id))
                ).scalar_one_or_none()
                if run and run.config_snapshot:
                    baseline_info = run.config_snapshot.get("baseline", {})
                    baseline_name = baseline_info.get("source")

            # Build baseline path
            baselines_dir = Path("config/baselines")
            if baseline_name:
                baseline_path = baselines_dir / f"{baseline_name}.json"
            else:
                # Fallback: use first available baseline file
                available = sorted(baselines_dir.glob("*.json"))
                if not available:
                    logger.warning("No baseline files found in %s", baselines_dir)
                    return []
                baseline_path = available[0]

            baseline = load_baseline_file(baseline_path)

            reader = DataReader(db)
            records = reader.read_biomarkers(
                user_id=user_id,
                start_time=start_dt,
                end_time=end_dt,
            )

            if not records:
                return []

            # Aggregate into windows
            aggregates_by_biomarker = aggregate_into_windows(
                records=records,
                window_size_minutes=config.window.size_minutes,
            )

            # Compute memberships
            memberships = compute_window_memberships(
                window_aggregates=aggregates_by_biomarker,
                user_id=user_id,
                indicator_name=indicator_name,
                session=db,
                config=config,
                baseline_config=baseline,
            )

            if not memberships:
                return []

            # Compute FASL indicators
            window_indicators = compute_window_indicators(
                window_memberships=memberships,
                indicator_name=indicator_name,
                indicator_config=indicator_config,
                config=config,
            )

            # Filter to target date and sort
            results = []
            for wi in sorted(window_indicators, key=lambda w: w.window_start):
                if wi.window_start.date() == target_date:
                    results.append({
                        "window_start": wi.window_start,
                        "window_end": wi.window_end,
                        "score": wi.indicator_score,
                        "context": wi.dominant_context,
                    })

            return results

    except Exception:
        logger.error(
            "Failed to get all window FASL scores for user %s indicator %s on %s",
            user_id,
            indicator_name,
            target_date,
            exc_info=True,
        )
        return []
