"""Analysis run data loading and processing functions."""

from uuid import UUID

import pandas as pd
from sqlalchemy import select

from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import AnalysisRun

logger = get_logger(__name__)


def load_analysis_runs(
    user_id: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Load recent analysis runs for a user.

    Args:
        user_id: User ID to filter by
        limit: Maximum number of runs to return

    Returns:
        DataFrame with: run_id, user_id, start_time, end_time, created_at
    """
    try:
        with SessionLocal() as db:
            stmt = (
                select(AnalysisRun)
                .where(AnalysisRun.user_id == user_id)
                .order_by(AnalysisRun.created_at.desc())
                .limit(limit)
            )
            results = db.execute(stmt).scalars().all()

        if not results:
            return _empty_runs_df()

        rows = [
            {
                "run_id": str(run.id),
                "user_id": run.user_id,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "created_at": run.created_at,
            }
            for run in results
        ]
        return pd.DataFrame(rows)

    except Exception:
        logger.error("Failed to load analysis runs", exc_info=True)
        return _empty_runs_df()


def get_analysis_run_summary(run_id: str) -> dict | None:
    """Get summary for a specific analysis run.

    Args:
        run_id: UUID of analysis run

    Returns:
        Dict with run details or None if not found
    """
    try:
        with SessionLocal() as db:
            stmt = select(AnalysisRun).where(AnalysisRun.id == UUID(run_id))
            run = db.execute(stmt).scalar_one_or_none()

        if not run:
            return None

        return {
            "run_id": str(run.id),
            "user_id": run.user_id,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "created_at": run.created_at,
            "config_snapshot": run.config_snapshot,
            "pipeline_trace": run.pipeline_trace,
        }

    except Exception:
        logger.error("Failed to get analysis run %s", run_id, exc_info=True)
        return None


def _empty_runs_df() -> pd.DataFrame:
    """Return empty DataFrame for analysis runs."""
    return pd.DataFrame(
        columns=["run_id", "user_id", "start_time", "end_time", "created_at"]
    )
