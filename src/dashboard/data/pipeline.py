"""Pipeline trace data loading functions for dashboard."""

from uuid import UUID

import pandas as pd
from sqlalchemy import select

from src.core.pipeline import PipelineTrace, get_pipeline_trace
from src.dashboard.components.filters import get_display_timezone
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import AnalysisRun

logger = get_logger(__name__)


def load_user_analysis_runs(user_id: str, limit: int = 20) -> pd.DataFrame:
    """Load analysis runs for dropdown selection.

    Args:
        user_id: User ID to filter by
        limit: Maximum number of runs to return

    Returns:
        DataFrame with: run_id, start_time, end_time, created_at, has_trace, display_label
    """
    try:
        with SessionLocal() as session:
            stmt = (
                select(AnalysisRun)
                .where(AnalysisRun.user_id == user_id)
                .order_by(AnalysisRun.created_at.desc())
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()

            if not results:
                return _empty_runs_df()

            data = [
                {
                    "run_id": str(run.id),
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "created_at": run.created_at,
                    "has_trace": run.pipeline_trace is not None,
                    "display_label": _format_run_label(run),
                }
                for run in results
            ]
            return pd.DataFrame(data)

    except Exception:
        logger.error("Failed to load analysis runs for user %s", user_id, exc_info=True)
        return _empty_runs_df()


def get_trace_for_run(run_id: str) -> tuple[PipelineTrace | None, dict | None]:
    """Get pipeline trace and config for a run.

    Args:
        run_id: UUID string of analysis run

    Returns:
        Tuple of (PipelineTrace or None, config_snapshot dict or None)
    """
    try:
        with SessionLocal() as session:
            trace = get_pipeline_trace(UUID(run_id), session)

            # Also get config snapshot
            stmt = select(AnalysisRun).where(AnalysisRun.id == UUID(run_id))
            run = session.execute(stmt).scalar_one_or_none()
            config = run.config_snapshot if run else None

            return trace, config

    except Exception:
        logger.error("Failed to get trace for run %s", run_id, exc_info=True)
        return None, None


def _format_run_label(run: AnalysisRun) -> str:
    """Format run for dropdown display."""
    run_id_short = str(run.id)[:8]
    tz = get_display_timezone()
    created_at = run.created_at
    local_time = created_at.astimezone(tz) if created_at.tzinfo else created_at.replace(tzinfo=tz)
    created = local_time.strftime("%Y-%m-%d %H:%M")
    return f"{run_id_short}... | {created}"


def _empty_runs_df() -> pd.DataFrame:
    """Return empty DataFrame for runs."""
    return pd.DataFrame(
        columns=[
            "run_id",
            "start_time",
            "end_time",
            "created_at",
            "has_trace",
            "display_label",
        ]
    )
