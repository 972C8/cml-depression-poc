"""Context evaluation run data management functions (Story 6.13).

Functions for creating, listing, and managing ContextEvaluationRun records
that enable experimentation with different context evaluation configurations.
"""

import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from src.core.config import ExperimentContextEvalConfig, get_default_config
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import ContextEvaluationRun, ContextHistoryRecord

logger = get_logger(__name__)


def create_context_evaluation_run(
    user_id: str,
    context_eval_config: ExperimentContextEvalConfig,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> ContextEvaluationRun:
    """Create a new context evaluation run record.

    Args:
        user_id: User ID for which context will be evaluated
        context_eval_config: ExperimentContextEvalConfig containing all evaluation params
        start_time: Start of evaluated time period (None = all history)
        end_time: End of evaluated time period (None = all history)

    Returns:
        Created ContextEvaluationRun instance
    """
    with SessionLocal() as session:
        # Serialize config to dict for storage
        config_dict = {
            "marker_memberships": {
                marker: {
                    "sets": {
                        set_name: {
                            "type": set_def.type,
                            "params": list(set_def.params),
                        }
                        for set_name, set_def in mm.sets.items()
                    }
                }
                for marker, mm in context_eval_config.marker_memberships.items()
            },
            "context_assumptions": {
                ctx: {
                    "conditions": [
                        {
                            "marker": c.marker,
                            "set": c.fuzzy_set,
                            "weight": c.weight,
                        }
                        for c in assumption.conditions
                    ],
                    "operator": assumption.operator,
                }
                for ctx, assumption in context_eval_config.context_assumptions.items()
            },
            "neutral_threshold": context_eval_config.neutral_threshold,
            "ema": {
                "alpha": context_eval_config.ema.alpha,
                "hysteresis": context_eval_config.ema.hysteresis,
                "dwell_time": context_eval_config.ema.dwell_time,
            },
        }

        run = ContextEvaluationRun(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            config_snapshot=config_dict,
            evaluation_count=0,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        logger.info(
            "Created context evaluation run: %s for user %s (%s to %s)",
            run.id,
            user_id,
            start_time,
            end_time,
        )
        session.expunge(run)
        return run


def update_run_evaluation_count(run_id: uuid.UUID, count: int) -> None:
    """Update the evaluation_count for a run after processing.

    Args:
        run_id: UUID of the run to update
        count: Number of evaluations created
    """
    with SessionLocal() as session:
        stmt = select(ContextEvaluationRun).where(ContextEvaluationRun.id == run_id)
        run = session.execute(stmt).scalar_one_or_none()
        if run:
            run.evaluation_count = count
            session.commit()
            logger.info("Updated run %s evaluation count to %d", run_id, count)


def list_context_evaluation_runs(
    user_id: str,
    limit: int = 20,
) -> pd.DataFrame:
    """List context evaluation runs for a user.

    Args:
        user_id: User ID to filter by
        limit: Maximum number of runs to return

    Returns:
        DataFrame with run metadata
    """
    try:
        with SessionLocal() as session:
            stmt = (
                select(ContextEvaluationRun)
                .where(ContextEvaluationRun.user_id == user_id)
                .order_by(ContextEvaluationRun.created_at.desc())
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()

        if not results:
            return pd.DataFrame(
                columns=[
                    "id",
                    "user_id",
                    "start_time",
                    "end_time",
                    "evaluation_count",
                    "created_at",
                ]
            )

        data = [
            {
                "id": str(run.id),
                "user_id": run.user_id,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "evaluation_count": run.evaluation_count,
                "created_at": run.created_at,
                "config_snapshot": run.config_snapshot,
            }
            for run in results
        ]
        return pd.DataFrame(data)

    except Exception:
        logger.error("Failed to list context evaluation runs", exc_info=True)
        return pd.DataFrame(
            columns=[
                "id",
                "user_id",
                "start_time",
                "end_time",
                "evaluation_count",
                "created_at",
            ]
        )


def get_context_evaluation_run(run_id: str) -> ContextEvaluationRun | None:
    """Get a context evaluation run by ID.

    Args:
        run_id: UUID string of the run

    Returns:
        ContextEvaluationRun or None if not found
    """
    try:
        with SessionLocal() as session:
            stmt = select(ContextEvaluationRun).where(
                ContextEvaluationRun.id == uuid.UUID(run_id)
            )
            run = session.execute(stmt).scalar_one_or_none()
            if run:
                session.expunge(run)
            return run
    except Exception:
        logger.error("Failed to get context evaluation run %s", run_id, exc_info=True)
        return None


def load_context_history_by_run(
    run_id: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Load context history records for a specific evaluation run.

    Args:
        run_id: UUID string of the context evaluation run
        limit: Maximum number of records to return

    Returns:
        DataFrame with context history records for the run
    """
    try:
        run_uuid = uuid.UUID(run_id)
        with SessionLocal() as session:
            stmt = (
                select(ContextHistoryRecord)
                .where(ContextHistoryRecord.context_evaluation_run_id == run_uuid)
                .order_by(ContextHistoryRecord.evaluated_at.desc())
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()

        if not results:
            return pd.DataFrame()

        data = [
            {
                "id": str(r.id),
                "user_id": r.user_id,
                "evaluated_at": r.evaluated_at,
                "dominant_context": r.dominant_context,
                "confidence": r.confidence,
                "context_state": r.context_state,
                "raw_scores": r.raw_scores,
                "sensors_used": r.sensors_used,
                "sensor_snapshot": r.sensor_snapshot,
                "evaluation_trigger": r.evaluation_trigger,
                "switch_blocked": r.switch_blocked,
                "switch_blocked_reason": r.switch_blocked_reason,
                "candidate_context": r.candidate_context,
                "score_difference": r.score_difference,
                "dwell_progress": r.dwell_progress,
                "dwell_required": r.dwell_required,
                "context_evaluation_run_id": str(r.context_evaluation_run_id)
                if r.context_evaluation_run_id
                else None,
                "created_at": r.created_at,
            }
            for r in results
        ]
        return pd.DataFrame(data)

    except Exception:
        logger.error("Failed to load context history for run %s", run_id, exc_info=True)
        return pd.DataFrame()


def check_context_run_coverage(
    run_id: str,
    user_id: str,
    start_time: datetime,
    end_time: datetime,
) -> dict:
    """Check if a context evaluation run covers dates with actual biomarker data.

    Story 6.14: AC4 - Validates that the selected context run covers
    the analysis date range. Only warns about dates that have biomarker
    data but lack context data (not dates outside the data range).

    Args:
        run_id: UUID string of the context evaluation run
        user_id: User ID
        start_time: Start of analysis window
        end_time: End of analysis window

    Returns:
        Dict with coverage details:
        - dates_with_data: Number of dates with biomarker data
        - dates_with_context: Number of dates with context data
        - dates_missing_context: List of ISO date strings with data but no context
        - coverage_ratio: 0.0 to 1.0 (context coverage of dates WITH data)
        - selected_range_days: Total days in selected range (for info)
    """
    from datetime import date, timedelta

    from src.shared.models import Biomarker

    try:
        run_uuid = uuid.UUID(run_id)
        with SessionLocal() as session:
            # Query dates with biomarker data in the selected range
            biomarker_stmt = (
                select(Biomarker.timestamp)
                .where(Biomarker.user_id == user_id)
                .where(Biomarker.timestamp >= start_time)
                .where(Biomarker.timestamp <= end_time)
            )
            biomarker_results = session.execute(biomarker_stmt).scalars().all()

            # Query dates with context data for this run
            context_stmt = (
                select(ContextHistoryRecord.evaluated_at)
                .where(ContextHistoryRecord.user_id == user_id)
                .where(ContextHistoryRecord.context_evaluation_run_id == run_uuid)
                .where(ContextHistoryRecord.evaluated_at >= start_time)
                .where(ContextHistoryRecord.evaluated_at <= end_time)
            )
            context_results = session.execute(context_stmt).scalars().all()

        # Extract unique dates with biomarker data
        dates_with_biomarker_data: set[date] = set()
        for ts in biomarker_results:
            dates_with_biomarker_data.add(ts.date())

        # Extract unique dates with context data
        dates_with_context: set[date] = set()
        for ts in context_results:
            dates_with_context.add(ts.date())

        # Calculate selected range size for info
        current_date = start_time.date()
        end_date = end_time.date()
        selected_range_days = (end_date - current_date).days + 1

        # Find dates that HAVE biomarker data but DON'T have context data
        # These are the only dates that matter for the warning
        dates_missing_context = sorted(dates_with_biomarker_data - dates_with_context)

        # Coverage is relative to dates with actual data, not selected range
        dates_with_data_count = len(dates_with_biomarker_data)
        dates_with_context_in_data = len(dates_with_biomarker_data & dates_with_context)
        coverage_ratio = (
            dates_with_context_in_data / dates_with_data_count
            if dates_with_data_count > 0
            else 1.0  # No data = no coverage issue
        )

        return {
            "dates_with_data": dates_with_data_count,
            "dates_with_context": dates_with_context_in_data,
            "dates_missing_context": [d.isoformat() for d in dates_missing_context],
            "coverage_ratio": coverage_ratio,
            "selected_range_days": selected_range_days,
            # Legacy fields for backward compatibility
            "dates_covered": dates_with_context_in_data,
            "dates_total": dates_with_data_count,
            "missing_dates": [d.isoformat() for d in dates_missing_context],
        }

    except Exception:
        logger.error(
            "Failed to check context run coverage for %s", run_id, exc_info=True
        )
        return {
            "dates_with_data": 0,
            "dates_with_context": 0,
            "dates_missing_context": [],
            "coverage_ratio": 1.0,
            "selected_range_days": 0,
            "dates_covered": 0,
            "dates_total": 0,
            "missing_dates": [],
        }


def get_default_context_eval_config() -> ExperimentContextEvalConfig:
    """Get the default ExperimentContextEvalConfig from YAML files.

    Returns:
        ExperimentContextEvalConfig loaded from config files
    """
    config = get_default_config()
    return config.context_evaluation


def delete_context_evaluation_run(run_id: str) -> bool:
    """Delete a context evaluation run and its associated history records.

    Args:
        run_id: UUID string of the run to delete

    Returns:
        True if deleted, False if not found or error
    """
    try:
        run_uuid = uuid.UUID(run_id)
        with SessionLocal() as session:
            # First, remove the run_id from associated history records
            # (foreign key is SET NULL on delete, so this is optional but explicit)
            session.query(ContextHistoryRecord).filter(
                ContextHistoryRecord.context_evaluation_run_id == run_uuid
            ).update({ContextHistoryRecord.context_evaluation_run_id: None})

            # Delete the run
            stmt = select(ContextEvaluationRun).where(
                ContextEvaluationRun.id == run_uuid
            )
            run = session.execute(stmt).scalar_one_or_none()
            if run is None:
                return False

            session.delete(run)
            session.commit()
            logger.info("Deleted context evaluation run: %s", run_id)
            return True

    except Exception:
        logger.error(
            "Failed to delete context evaluation run %s", run_id, exc_info=True
        )
        return False
