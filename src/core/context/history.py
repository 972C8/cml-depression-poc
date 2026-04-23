"""Context History Service for temporal context binding.

This module provides infrastructure for storing and retrieving context
evaluations over time, enabling accurate context-aware weighting by
binding the correct historical context to each biomarker window.

Story 6.1: Context History Infrastructure
"""

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Any

# Type alias for progress callback: (current_iteration, total_iterations)
ProgressCallback = Callable[[int, int], None] | None

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import AnalysisConfig
from src.core.context.evaluator import ContextEvaluator, ContextResult
from src.core.data_reader import ContextRecord, DataReader
from src.shared.models import ContextHistoryRecord

__all__ = [
    "ContextCoverageResult",
    "ContextHistoryService",
    "ContextHistoryStatus",
    "ContextSegment",
    "ContextState",
    "SensorSnapshot",
]

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes (Task 2)
# ============================================================================


class ContextHistoryStatus(str, Enum):
    """Status returned by ensure_context_history_exists.

    Attributes:
        ALREADY_POPULATED: Context history already exists for the entire range
        GAPS_FOUND: Some gaps were detected (and potentially filled)
        EVALUATIONS_ADDED: New context evaluations were added to fill gaps
        NO_SENSOR_DATA: No sensor data available to evaluate context
    """

    ALREADY_POPULATED = "already_populated"
    GAPS_FOUND = "gaps_found"
    EVALUATIONS_ADDED = "evaluations_added"
    NO_SENSOR_DATA = "no_sensor_data"


@dataclass(frozen=True)
class ContextState:
    """Represents context state at a point in time.

    Attributes:
        dominant_context: The highest-confidence context name
        confidence: Confidence score for dominant context (0-1)
        all_scores: Dict of all context names to their confidence scores
        timestamp: When this context was evaluated
    """

    dominant_context: str
    confidence: float
    all_scores: dict[str, float]
    timestamp: datetime


@dataclass(frozen=True)
class ContextSegment:
    """A time segment with consistent context state.

    Used by get_context_timeline to return periods where context remained stable.

    Attributes:
        context: The dominant context during this segment
        confidence: Average confidence during this segment
        start: Start timestamp of segment
        end: End timestamp of segment
        duration_minutes: Duration of segment in minutes
    """

    context: str
    confidence: float
    start: datetime
    end: datetime

    @property
    def duration_minutes(self) -> float:
        """Calculate duration in minutes."""
        return (self.end - self.start).total_seconds() / 60.0


@dataclass(frozen=True)
class SensorSnapshot:
    """Raw sensor values at a point in time.

    Attributes:
        timestamp: When sensors were read
        values: Dict mapping sensor/marker name to value
        marker_types: List of marker types present
    """

    timestamp: datetime
    values: dict[str, Any]
    marker_types: list[str]


@dataclass
class EnsureHistoryResult:
    """Result of ensure_context_history_exists operation.

    Attributes:
        status: The status of the operation
        gaps_found: Number of gaps detected
        evaluations_added: Number of new evaluations added
        message: Human-readable summary
    """

    status: ContextHistoryStatus
    gaps_found: int
    evaluations_added: int
    message: str


# ============================================================================
# Context History Service (Task 3-7)
# ============================================================================


@dataclass
class ContextCoverageResult:
    """Result of checking context coverage for a date range.

    Story 6.14: Used to validate selected context run coverage.

    Attributes:
        dates_covered: Number of dates with context data
        missing_dates: List of dates without context data
        coverage_ratio: Ratio of covered to total dates (0.0 to 1.0)
    """

    dates_covered: int
    missing_dates: list[date]
    coverage_ratio: float


class ContextHistoryService:
    """Service for managing context history records.

    Provides methods to:
    - Ensure context history exists for a time range (with gap filling)
    - Find gaps in context history
    - Populate context history from historical sensor data
    - Retrieve context at a specific timestamp (forward-fill)
    - Get context timeline for time-weighted analysis

    Story 6.14: When context_evaluation_run_id is provided, all queries
    are filtered to only return records from that specific run.
    """

    def __init__(
        self,
        session: Session,
        config: AnalysisConfig | None = None,
        evaluator: ContextEvaluator | None = None,
        data_reader: DataReader | None = None,
        context_evaluation_run_id: uuid.UUID | None = None,
    ) -> None:
        """Initialize context history service.

        Args:
            session: SQLAlchemy database session
            config: Analysis configuration (uses defaults if None)
            evaluator: Context evaluator instance (creates new if None)
            data_reader: Data reader for sensor data (creates new if None)
            context_evaluation_run_id: Optional UUID to filter queries by.
                When provided, only records from this run are returned.
        """
        self._session = session
        self._config = config
        self._evaluator = evaluator or ContextEvaluator(analysis_config=config)
        self._data_reader = data_reader or DataReader(session)
        self._logger = logging.getLogger(__name__)
        self._context_evaluation_run_id = context_evaluation_run_id

        # Get configuration from AnalysisConfig or use defaults
        if config and hasattr(config, "context_history"):
            ctx_config = config.context_history
            self._evaluation_interval_minutes = ctx_config.evaluation_interval_minutes
            self._staleness_hours = ctx_config.staleness_hours
            self._neutral_weight = ctx_config.neutral_weight
            self._clock_skew_tolerance_minutes = ctx_config.clock_skew_tolerance_minutes
        else:
            # Defaults matching ContextHistoryConfig
            self._evaluation_interval_minutes = 15
            self._staleness_hours = 2.0
            self._neutral_weight = 1.0
            self._clock_skew_tolerance_minutes = 5

    # -------------------------------------------------------------------------
    # Core Methods (AC3)
    # -------------------------------------------------------------------------

    def ensure_context_history_exists(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        context_evaluation_run_id: uuid.UUID | None = None,
        progress_callback: ProgressCallback = None,
    ) -> EnsureHistoryResult:
        """Ensure context history exists for the given time range.

        Main entry point called before analysis runs. Checks for gaps in
        context history and fills them by evaluating historical sensor data.

        Note: This method adds records to the session but does NOT commit.
        The caller is responsible for calling session.commit() after this
        method returns if new evaluations were added.

        Args:
            user_id: User identifier
            start: Start of time range
            end: End of time range
            context_evaluation_run_id: Optional run ID to link created records to
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            EnsureHistoryResult with status and counts
        """
        self._logger.info(
            f"Ensuring context history for user={user_id}, "
            f"range={start.isoformat()} to {end.isoformat()}"
        )

        # When a run_id is provided (experiment mode), evaluate the entire range
        # without checking for gaps - each run creates its own evaluations
        if context_evaluation_run_id is not None:
            total_added = self.populate_context_for_range(
                user_id, start, end, context_evaluation_run_id, progress_callback
            )
            if total_added == 0:
                return EnsureHistoryResult(
                    status=ContextHistoryStatus.NO_SENSOR_DATA,
                    gaps_found=0,
                    evaluations_added=0,
                    message="No sensor data available for evaluation",
                )
            return EnsureHistoryResult(
                status=ContextHistoryStatus.EVALUATIONS_ADDED,
                gaps_found=0,
                evaluations_added=total_added,
                message=f"Created {total_added} context evaluations for run",
            )

        # Standard mode: Find gaps in context history
        gaps = self.find_context_history_gaps(user_id, start, end)

        if not gaps:
            self._logger.debug(f"No gaps found for user={user_id}")
            return EnsureHistoryResult(
                status=ContextHistoryStatus.ALREADY_POPULATED,
                gaps_found=0,
                evaluations_added=0,
                message="Context history already exists for the entire range",
            )

        # Populate context for each gap
        total_added = 0
        for gap_start, gap_end in gaps:
            added = self.populate_context_for_range(
                user_id, gap_start, gap_end, context_evaluation_run_id, progress_callback
            )
            total_added += added

        if total_added == 0:
            return EnsureHistoryResult(
                status=ContextHistoryStatus.NO_SENSOR_DATA,
                gaps_found=len(gaps),
                evaluations_added=0,
                message=f"Found {len(gaps)} gaps but no sensor data available to fill",
            )

        return EnsureHistoryResult(
            status=ContextHistoryStatus.EVALUATIONS_ADDED,
            gaps_found=len(gaps),
            evaluations_added=total_added,
            message=f"Added {total_added} context evaluations for {len(gaps)} gaps",
        )

    def check_context_coverage(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> ContextCoverageResult:
        """Check context coverage for a date range using the selected run.

        Story 6.14: AC4 - Validates that the selected context run covers
        the analysis date range. Returns coverage details for UI warnings.

        Args:
            user_id: User identifier
            start: Start of analysis time range
            end: End of analysis time range

        Returns:
            ContextCoverageResult with coverage details
        """
        # Build query to find dates with context data
        stmt = (
            select(ContextHistoryRecord.evaluated_at)
            .where(ContextHistoryRecord.user_id == user_id)
            .where(ContextHistoryRecord.evaluated_at >= start)
            .where(ContextHistoryRecord.evaluated_at <= end)
        )

        # Filter by run_id if set
        if self._context_evaluation_run_id is not None:
            stmt = stmt.where(
                ContextHistoryRecord.context_evaluation_run_id
                == self._context_evaluation_run_id
            )

        results = self._session.execute(stmt).scalars().all()

        # Extract unique dates with context data
        dates_with_context: set[date] = set()
        for ts in results:
            dates_with_context.add(ts.date())

        # Generate all expected dates in range
        all_dates: set[date] = set()
        current_date = start.date()
        end_date = end.date()
        while current_date <= end_date:
            all_dates.add(current_date)
            current_date += timedelta(days=1)

        # Find missing dates
        missing_dates = sorted(all_dates - dates_with_context)
        dates_covered = len(dates_with_context)
        total_dates = len(all_dates)

        coverage_ratio = dates_covered / total_dates if total_dates > 0 else 0.0

        self._logger.debug(
            f"Context coverage for user={user_id}: {dates_covered}/{total_dates} "
            f"dates ({coverage_ratio:.1%}), missing: {len(missing_dates)}"
        )

        return ContextCoverageResult(
            dates_covered=dates_covered,
            missing_dates=missing_dates,
            coverage_ratio=coverage_ratio,
        )

    def find_context_history_gaps(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Find time ranges lacking context history.

        Uses evaluation_interval to detect gaps where context history is missing.

        Args:
            user_id: User identifier
            start: Start of time range
            end: End of time range

        Returns:
            List of (gap_start, gap_end) tuples representing missing ranges
        """
        interval = timedelta(minutes=self._evaluation_interval_minutes)

        # Query existing context history records
        stmt = (
            select(ContextHistoryRecord.evaluated_at)
            .where(ContextHistoryRecord.user_id == user_id)
            .where(ContextHistoryRecord.evaluated_at >= start)
            .where(ContextHistoryRecord.evaluated_at <= end)
            .order_by(ContextHistoryRecord.evaluated_at)
        )
        results = self._session.execute(stmt).scalars().all()
        existing_timestamps = set(results)

        # Generate expected timestamps based on interval
        gaps: list[tuple[datetime, datetime]] = []
        current = start
        gap_start: datetime | None = None

        while current <= end:
            # Check if we have a record within tolerance of current
            has_record = any(
                abs((ts - current).total_seconds()) < interval.total_seconds() / 2
                for ts in existing_timestamps
            )

            if not has_record:
                if gap_start is None:
                    gap_start = current
            else:
                if gap_start is not None:
                    # End the gap at the previous interval
                    gap_end = current - interval
                    if gap_end > gap_start:
                        gaps.append((gap_start, gap_end))
                    gap_start = None

            current += interval

        # Handle trailing gap
        if gap_start is not None:
            gaps.append((gap_start, end))

        self._logger.debug(f"Found {len(gaps)} gaps for user={user_id}")
        return gaps

    def populate_context_for_range(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        context_evaluation_run_id: uuid.UUID | None = None,
        progress_callback: ProgressCallback = None,
    ) -> int:
        """Populate context history by evaluating historical sensor data.

        Reads historical sensor/context marker data and uses the ContextEvaluator
        to determine what context was active at each evaluation point. Applies
        EMA smoothing for temporal continuity.

        Note: This method does NOT commit the transaction. The caller is
        responsible for calling session.commit() after this method returns.

        Args:
            user_id: User identifier
            start: Start of time range
            end: End of time range
            context_evaluation_run_id: Optional run ID to link records to
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            Count of evaluations added
        """
        interval = timedelta(minutes=self._evaluation_interval_minutes)
        staleness_threshold = timedelta(hours=self._staleness_hours)
        added_count = 0
        current = start
        last_eval_time: datetime | None = None

        # Calculate total iterations for progress tracking
        total_iterations = int((end - start).total_seconds() / interval.total_seconds()) + 1
        current_iteration = 0

        # Initialize smoother state from prior history for EMA continuity
        self._initialize_smoother_state(user_id, start)

        while current <= end:
            current_iteration += 1
            # Report progress every 10 iterations to avoid UI overhead
            if progress_callback and current_iteration % 10 == 0:
                progress_callback(current_iteration, total_iterations)
            # Check if context already exists (idempotency) - but skip this check
            # when creating evaluations for a specific run (experiment mode)
            if context_evaluation_run_id is None and self.context_exists_at(
                user_id, current
            ):
                # Update last_eval_time to maintain gap tracking
                last_eval_time = current
                current += interval
                continue

            # Check for large time gaps - reset smoother if continuity is broken
            if last_eval_time is not None:
                gap = current - last_eval_time
                if gap > staleness_threshold:
                    self._logger.debug(
                        f"Large gap detected ({gap}), resetting smoother state"
                    )
                    self._evaluator.reset()

            # Get sensor snapshot at this time
            snapshot = self.get_sensor_snapshot_at(user_id, current)

            if snapshot is None:
                # No sensor data available, skip this timestamp
                current += interval
                continue

            # Convert snapshot to context records for evaluator
            context_records = self._snapshot_to_context_records(snapshot, user_id)

            # Evaluate context with EMA smoothing enabled
            result = self._evaluator.evaluate(context_records, apply_smoothing=True)

            # Store result (includes both raw and smoothed scores)
            # Use "experiment" trigger when linked to a run, "backfill" otherwise
            trigger = "experiment" if context_evaluation_run_id else "backfill"
            self._store_context_result(
                user_id=user_id,
                timestamp=current,
                result=result,
                snapshot=snapshot,
                trigger=trigger,
                context_evaluation_run_id=context_evaluation_run_id,
            )
            added_count += 1
            last_eval_time = current
            current += interval

        if added_count > 0:
            self._logger.info(
                f"Added {added_count} context evaluations for user={user_id}"
            )

        return added_count

    def _initialize_smoother_state(
        self,
        user_id: str,
        start: datetime,
    ) -> None:
        """Initialize EMA smoother state from most recent prior evaluation.

        Ensures EMA smoothing continuity when backfilling new evaluations
        by loading the previous smoothed values and active context.

        Args:
            user_id: User identifier
            start: Start timestamp of the new evaluation range
        """
        staleness_threshold = timedelta(hours=self._staleness_hours)

        # Find most recent context history before start
        stmt = (
            select(ContextHistoryRecord)
            .where(ContextHistoryRecord.user_id == user_id)
            .where(ContextHistoryRecord.evaluated_at < start)
            .order_by(ContextHistoryRecord.evaluated_at.desc())
            .limit(1)
        )
        prior_record = self._session.execute(stmt).scalar_one_or_none()

        if prior_record is None:
            # No prior history - start fresh
            self._evaluator.reset()
            self._logger.debug(
                f"No prior context history for user={user_id}, starting fresh"
            )
            return

        # Check if prior record is too stale for continuity
        age = start - prior_record.evaluated_at
        if age > staleness_threshold:
            self._evaluator.reset()
            self._logger.debug(
                f"Prior context history too stale (age={age}), starting fresh"
            )
            return

        # Initialize smoother with prior state
        self._evaluator.initialize_state(
            previous_values=prior_record.context_state,
            active_context=prior_record.dominant_context,
        )
        self._logger.debug(
            f"Initialized smoother from prior record at {prior_record.evaluated_at}"
        )

    # -------------------------------------------------------------------------
    # Support Methods (AC4)
    # -------------------------------------------------------------------------

    def get_sensor_snapshot_at(
        self,
        user_id: str,
        timestamp: datetime,
    ) -> SensorSnapshot | None:
        """Get sensor values at a specific timestamp.

        Uses forward-fill logic: returns most recent sensor readings before
        the given timestamp.

        Args:
            user_id: User identifier
            timestamp: Target timestamp

        Returns:
            SensorSnapshot with sensor values, or None if no data available
        """
        # Look back up to staleness_hours for sensor data
        lookback = timestamp - timedelta(hours=self._staleness_hours)

        # Read context markers using DataReader
        context_markers = self._data_reader.read_context_markers(
            user_id=user_id,
            start_time=lookback,
            end_time=timestamp,
        )

        if not context_markers:
            return None

        # Extract latest value for each marker type (forward-fill)
        latest_values: dict[str, tuple[datetime, Any]] = {}
        marker_types: set[str] = set()

        for marker in context_markers:
            marker_types.add(marker.context_type)
            key = marker.name
            if key not in latest_values or marker.timestamp > latest_values[key][0]:
                latest_values[key] = (marker.timestamp, marker.value)

        if not latest_values:
            return None

        return SensorSnapshot(
            timestamp=timestamp,
            values={name: value for name, (_, value) in latest_values.items()},
            marker_types=list(marker_types),
        )

    def context_exists_at(
        self,
        user_id: str,
        timestamp: datetime,
    ) -> bool:
        """Check if context history exists at a specific timestamp.

        Used for idempotency when populating context history.

        Args:
            user_id: User identifier
            timestamp: Target timestamp

        Returns:
            True if context record exists, False otherwise
        """
        # Allow tolerance for exact timestamp matching
        tolerance = timedelta(seconds=30)

        stmt = (
            select(ContextHistoryRecord.id)
            .where(ContextHistoryRecord.user_id == user_id)
            .where(ContextHistoryRecord.evaluated_at >= timestamp - tolerance)
            .where(ContextHistoryRecord.evaluated_at <= timestamp + tolerance)
            .limit(1)
        )
        result = self._session.execute(stmt).scalar_one_or_none()
        return result is not None

    # -------------------------------------------------------------------------
    # Retrieval Functions (AC5)
    # -------------------------------------------------------------------------

    def get_context_at_timestamp(
        self,
        user_id: str,
        timestamp: datetime,
        max_staleness: timedelta | None = None,
    ) -> ContextState | None:
        """Get context state at a specific timestamp using forward-fill.

        Returns the most recent context evaluation before the timestamp.
        If the context is older than max_staleness, returns None.

        Story 6.14: When context_evaluation_run_id is set on the service,
        only records from that run are considered. Returns None (triggering
        neutral weight fallback) for timestamps not covered by the run.

        Args:
            user_id: User identifier
            timestamp: Target timestamp
            max_staleness: Maximum age before returning None (default: staleness_hours)

        Returns:
            ContextState if found within staleness, None otherwise
        """
        if max_staleness is None:
            max_staleness = timedelta(hours=self._staleness_hours)

        # Allow clock skew tolerance
        adjusted_timestamp = timestamp + timedelta(
            minutes=self._clock_skew_tolerance_minutes
        )

        # Find most recent context evaluation before timestamp
        stmt = (
            select(ContextHistoryRecord)
            .where(ContextHistoryRecord.user_id == user_id)
            .where(ContextHistoryRecord.evaluated_at <= adjusted_timestamp)
        )

        # Story 6.14: Filter by run_id when set
        if self._context_evaluation_run_id is not None:
            stmt = stmt.where(
                ContextHistoryRecord.context_evaluation_run_id
                == self._context_evaluation_run_id
            )

        stmt = stmt.order_by(ContextHistoryRecord.evaluated_at.desc()).limit(1)
        record = self._session.execute(stmt).scalar_one_or_none()

        if record is None:
            self._logger.debug(
                f"No context history found for user={user_id} before {timestamp}"
            )
            return None

        # Check staleness
        age = timestamp - record.evaluated_at
        if age > max_staleness:
            self._logger.debug(
                f"Context at {record.evaluated_at} is stale "
                f"(age={age}, max={max_staleness})"
            )
            return None

        return ContextState(
            dominant_context=record.dominant_context,
            confidence=record.confidence,
            all_scores=record.context_state,
            timestamp=record.evaluated_at,
        )

    def get_context_timeline(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
    ) -> list[ContextSegment]:
        """Get sequence of context states with durations.

        Returns context segments showing when each context was active.
        Used by time-weighted context strategy in Story 6.3.

        Story 6.14: When context_evaluation_run_id is set on the service,
        only records from that run are included in the timeline.

        Args:
            user_id: User identifier
            start: Start of time range
            end: End of time range

        Returns:
            List of ContextSegment with context states and durations
        """
        # Query all context history records in range
        stmt = (
            select(ContextHistoryRecord)
            .where(ContextHistoryRecord.user_id == user_id)
            .where(ContextHistoryRecord.evaluated_at >= start)
            .where(ContextHistoryRecord.evaluated_at <= end)
        )

        # Story 6.14: Filter by run_id when set
        if self._context_evaluation_run_id is not None:
            stmt = stmt.where(
                ContextHistoryRecord.context_evaluation_run_id
                == self._context_evaluation_run_id
            )

        stmt = stmt.order_by(ContextHistoryRecord.evaluated_at)
        records = self._session.execute(stmt).scalars().all()

        if not records:
            return []

        segments: list[ContextSegment] = []
        current_segment_start = records[0].evaluated_at
        current_context = records[0].dominant_context
        confidence_sum = records[0].confidence
        confidence_count = 1

        for i in range(1, len(records)):
            record = records[i]

            if record.dominant_context != current_context:
                # Context changed - close current segment
                segments.append(
                    ContextSegment(
                        context=current_context,
                        confidence=confidence_sum / confidence_count,
                        start=current_segment_start,
                        end=records[i - 1].evaluated_at,
                    )
                )
                # Start new segment
                current_segment_start = record.evaluated_at
                current_context = record.dominant_context
                confidence_sum = record.confidence
                confidence_count = 1
            else:
                # Same context - accumulate confidence
                confidence_sum += record.confidence
                confidence_count += 1

        # Close final segment
        segments.append(
            ContextSegment(
                context=current_context,
                confidence=confidence_sum / confidence_count,
                start=current_segment_start,
                end=records[-1].evaluated_at,
            )
        )

        return segments

    # -------------------------------------------------------------------------
    # Edge Case Handling (AC7)
    # -------------------------------------------------------------------------

    def get_neutral_context(self) -> ContextState:
        """Get neutral context state for fallback when no history available.

        Used when:
        - No context history exists for a timestamp
        - Context history is stale (older than staleness_hours)

        Returns:
            ContextState with neutral context and neutral_weight confidence
        """
        return ContextState(
            dominant_context="neutral",
            confidence=self._neutral_weight,
            all_scores={"neutral": self._neutral_weight},
            timestamp=datetime.now(UTC),
        )

    def get_context_or_neutral(
        self,
        user_id: str,
        timestamp: datetime,
        max_staleness: timedelta | None = None,
    ) -> ContextState:
        """Get context at timestamp or fall back to neutral.

        Convenience method that handles edge cases by returning neutral
        context when no history exists or staleness is exceeded.

        Args:
            user_id: User identifier
            timestamp: Target timestamp
            max_staleness: Maximum age before returning neutral (default: staleness_hours)

        Returns:
            ContextState - either from history or neutral fallback
        """
        state = self.get_context_at_timestamp(user_id, timestamp, max_staleness)
        if state is None:
            self._logger.debug(
                f"Using neutral context fallback for user={user_id} at {timestamp}"
            )
            return self.get_neutral_context()
        return state

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _snapshot_to_context_records(
        self,
        snapshot: SensorSnapshot,
        user_id: str,
    ) -> list[ContextRecord]:
        """Convert sensor snapshot to context records for evaluator.

        Args:
            snapshot: Sensor snapshot with values and marker types
            user_id: User identifier for the records

        Returns:
            List of ContextRecord objects suitable for ContextEvaluator
        """
        import uuid

        records = []
        for name, value in snapshot.values.items():
            # Determine context type from marker name using robust matching:
            # 1. Exact match
            # 2. Marker name starts with type (e.g., "location_home" -> "location")
            # 3. Type is a word boundary prefix (e.g., "app_usage" matches "app")
            # 4. Default to "context_marker"
            context_type = "context_marker"
            name_lower = name.lower()

            for mt in snapshot.marker_types:
                mt_lower = mt.lower()
                # Exact match
                if name_lower == mt_lower:
                    context_type = mt
                    break
                # Name starts with type followed by underscore or end
                if name_lower.startswith(mt_lower + "_") or name_lower.startswith(
                    mt_lower + "."
                ):
                    context_type = mt
                    break

            # Convert value to float for ContextRecord
            numeric_value = value if isinstance(value, int | float) else 0.0

            records.append(
                ContextRecord(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    timestamp=snapshot.timestamp,
                    context_type=context_type,
                    name=name,
                    value=numeric_value,
                    raw_value={name: value},
                )
            )
        return records

    def _store_context_result(
        self,
        user_id: str,
        timestamp: datetime,
        result: ContextResult,
        snapshot: SensorSnapshot | None,
        trigger: str,
        context_evaluation_run_id: uuid.UUID | None = None,
    ) -> None:
        """Store context evaluation result to database.

        Stores both EMA-smoothed scores (context_state) and raw pre-smoothing
        scores (raw_scores) for pipeline transparency.

        Args:
            user_id: User identifier
            timestamp: Evaluation timestamp
            result: ContextResult from evaluator
            snapshot: SensorSnapshot with marker values
            trigger: Trigger type (backfill, on_demand, manual)
            context_evaluation_run_id: Optional run ID to link this record to
        """
        record = ContextHistoryRecord(
            user_id=user_id,
            evaluated_at=timestamp,
            context_state=result.confidence_scores,
            raw_scores=result.raw_scores,
            dominant_context=result.active_context,
            confidence=result.confidence_scores.get(result.active_context, 0.0),
            evaluation_trigger=trigger,
            sensors_used=list(result.markers_used) if result.markers_used else None,
            sensor_snapshot=snapshot.values if snapshot else None,
            # Step 5: Stabilization transparency fields
            switch_blocked=result.switch_blocked,
            switch_blocked_reason=result.switch_blocked_reason,
            candidate_context=result.candidate_context,
            score_difference=result.score_difference,
            dwell_progress=result.dwell_progress,
            dwell_required=result.dwell_required,
            # Story 6.13: Link to evaluation run
            context_evaluation_run_id=context_evaluation_run_id,
        )
        self._session.add(record)
