"""Context evaluation data loading from context history.

Story 6.9: Provides functions to load context history records and analyze results.
Replaces the old on-the-fly evaluation approach with read-only access to stored history.
"""

from datetime import datetime

import pandas as pd
from sqlalchemy import select

from src.core.context.history import ContextHistoryService
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import ContextHistoryRecord

logger = get_logger(__name__)


def load_context_history_records(
    user_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    context_evaluation_run_id: str | None = None,
) -> pd.DataFrame:
    """Load context history records from database as DataFrame.

    Queries ContextHistoryRecord table for the specified user and time range.
    Returns a DataFrame with columns matching the stored history schema.

    Args:
        user_id: User ID to load history for
        start: Start datetime (inclusive), or None for no lower bound
        end: End datetime (inclusive), or None for no upper bound
        limit: Maximum number of records to return (most recent first when no
            date range specified), or None for no limit
        context_evaluation_run_id: Optional UUID string to filter by specific
            context evaluation run (Story 6.13)

    Returns:
        DataFrame with context history records:
            - evaluated_at: datetime
            - dominant_context: str
            - confidence: float
            - context_state: dict (EMA-smoothed scores)
            - raw_scores: dict | None (pre-smoothing scores)
            - evaluation_trigger: str
            - sensors_used: list[str] | None
            - sensor_snapshot: dict | None
            - context_evaluation_run_id: str | None (Story 6.13)
    """
    import uuid as uuid_module

    try:
        with SessionLocal() as db:
            stmt = select(ContextHistoryRecord).where(
                ContextHistoryRecord.user_id == user_id
            )

            if start is not None:
                stmt = stmt.where(ContextHistoryRecord.evaluated_at >= start)
            if end is not None:
                stmt = stmt.where(ContextHistoryRecord.evaluated_at <= end)

            # Story 6.13: Filter by context evaluation run
            if context_evaluation_run_id is not None:
                run_uuid = uuid_module.UUID(context_evaluation_run_id)
                stmt = stmt.where(
                    ContextHistoryRecord.context_evaluation_run_id == run_uuid
                )

            # Order by evaluated_at (ascending for date range, descending for limit)
            if limit is not None and start is None and end is None:
                # When getting latest N without date range, order descending
                stmt = stmt.order_by(ContextHistoryRecord.evaluated_at.desc())
            else:
                stmt = stmt.order_by(ContextHistoryRecord.evaluated_at)

            if limit is not None:
                stmt = stmt.limit(limit)

            records = db.execute(stmt).scalars().all()

        if not records:
            return _empty_history_df()

        # Convert records to DataFrame
        data = [
            {
                "evaluated_at": r.evaluated_at,
                "dominant_context": r.dominant_context,
                "confidence": r.confidence,
                "context_state": r.context_state,
                "raw_scores": r.raw_scores,
                "evaluation_trigger": r.evaluation_trigger,
                "sensors_used": r.sensors_used,
                "sensor_snapshot": r.sensor_snapshot,
                # Step 5: Stabilization transparency fields
                "switch_blocked": r.switch_blocked,
                "switch_blocked_reason": r.switch_blocked_reason,
                "candidate_context": r.candidate_context,
                "score_difference": r.score_difference,
                "dwell_progress": r.dwell_progress,
                "dwell_required": r.dwell_required,
                # Story 6.13: Context evaluation run reference
                "context_evaluation_run_id": (
                    str(r.context_evaluation_run_id)
                    if r.context_evaluation_run_id
                    else None
                ),
                # Step 6: Computation metadata
                "created_at": r.created_at,
            }
            for r in records
        ]

        return pd.DataFrame(data)

    except Exception:
        logger.error("Failed to load context history records", exc_info=True)
        return _empty_history_df()


def get_context_history_status(
    user_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict:
    """Get context history status and ensure history exists.

    Calls ContextHistoryService.ensure_context_history_exists() for the
    specified range and returns a user-friendly status dict.

    Args:
        user_id: User ID to check
        start: Start datetime, or None for entire history (earliest data)
        end: End datetime, or None for entire history (now)

    Returns:
        Dict with keys:
            - status: str (already_populated, gaps_found, evaluations_added, no_sensor_data)
            - gaps_found: int
            - evaluations_added: int
            - message: str
    """
    from zoneinfo import ZoneInfo

    from sqlalchemy import func

    from src.shared.models import Context

    try:
        with SessionLocal() as db:
            # If start is None, find earliest context marker for user
            if start is None:
                earliest = db.execute(
                    select(func.min(Context.timestamp)).where(
                        Context.user_id == user_id
                    )
                ).scalar()
                if earliest is None:
                    # No data at all
                    return {
                        "status": "no_sensor_data",
                        "gaps_found": 0,
                        "evaluations_added": 0,
                        "message": "No context marker data found for user",
                    }
                start = earliest

            # If end is None, use now
            if end is None:
                end = datetime.now(ZoneInfo("UTC"))

            service = ContextHistoryService(db)
            result = service.ensure_context_history_exists(user_id, start, end)

            # Commit if evaluations were added
            if result.evaluations_added > 0:
                db.commit()

            return {
                "status": result.status.value,
                "gaps_found": result.gaps_found,
                "evaluations_added": result.evaluations_added,
                "message": result.message,
            }

    except Exception:
        logger.error("Failed to get context history status", exc_info=True)
        return {
            "status": "error",
            "gaps_found": 0,
            "evaluations_added": 0,
            "message": "Failed to check context history status",
        }


def _empty_history_df() -> pd.DataFrame:
    """Return empty DataFrame with correct schema for context history."""
    return pd.DataFrame(
        columns=[
            "evaluated_at",
            "dominant_context",
            "confidence",
            "context_state",
            "raw_scores",
            "evaluation_trigger",
            "sensors_used",
            "sensor_snapshot",
            # Step 5: Stabilization transparency fields
            "switch_blocked",
            "switch_blocked_reason",
            "candidate_context",
            "score_difference",
            "dwell_progress",
            "dwell_required",
            # Step 6: Computation metadata
            "created_at",
        ]
    )


def detect_context_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Detect context transitions from context history records.

    Args:
        df: DataFrame from load_context_history_records with columns:
            - evaluated_at (or timestamp): datetime
            - dominant_context (or active_context): str

    Returns:
        DataFrame with transitions: timestamp, from_context, to_context, duration_minutes
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(
            columns=["timestamp", "from_context", "to_context", "duration_minutes"]
        )

    # Support both old column names and new column names for compatibility
    time_col = "evaluated_at" if "evaluated_at" in df.columns else "timestamp"
    context_col = (
        "dominant_context" if "dominant_context" in df.columns else "active_context"
    )

    df = df.sort_values(time_col)
    transitions = []

    prev_context = df.iloc[0][context_col]
    prev_time = df.iloc[0][time_col]

    for _, row in df.iloc[1:].iterrows():
        if row[context_col] != prev_context:
            duration = (row[time_col] - prev_time).total_seconds() / 60
            transitions.append(
                {
                    "timestamp": row[time_col],
                    "from_context": prev_context,
                    "to_context": row[context_col],
                    "duration_minutes": round(duration, 1),
                }
            )
            prev_context = row[context_col]
            prev_time = row[time_col]

    return pd.DataFrame(transitions)


def calculate_context_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate time distribution across contexts.

    Args:
        df: DataFrame from load_context_history_records with column:
            - dominant_context (or active_context): str

    Returns:
        DataFrame with: context, count, percentage
    """
    if df.empty:
        return pd.DataFrame(columns=["context", "count", "percentage"])

    # Support both old column names and new column names for compatibility
    context_col = (
        "dominant_context" if "dominant_context" in df.columns else "active_context"
    )

    counts = df[context_col].value_counts()
    total = len(df)

    return pd.DataFrame(
        {
            "context": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(1),
        }
    )


def generate_evaluation_csv_filename(
    user_id: str, start: datetime, end: datetime
) -> str:
    """Generate CSV export filename for context evaluation data.

    Args:
        user_id: User ID
        start: Start datetime
        end: End datetime

    Returns:
        Filename string like context_eval_user123_20240101_20240131.csv
    """
    return f"context_eval_{user_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"


def calculate_page_indices(
    current_page: int, page_size: int, total_rows: int
) -> tuple[int, int]:
    """Calculate start and end indices for pagination.

    Args:
        current_page: Current page number (1-indexed)
        page_size: Number of rows per page
        total_rows: Total number of rows

    Returns:
        Tuple of (start_idx, end_idx) for DataFrame slicing
    """
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    return start_idx, end_idx


def calculate_total_pages(total_rows: int, page_size: int) -> int:
    """Calculate total number of pages.

    Args:
        total_rows: Total number of rows
        page_size: Number of rows per page

    Returns:
        Total number of pages (minimum 1)
    """
    if total_rows == 0:
        return 1
    return (total_rows + page_size - 1) // page_size
