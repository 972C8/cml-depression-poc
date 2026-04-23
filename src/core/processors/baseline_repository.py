"""Baseline repository for database access."""

import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.shared.models import UserBaseline

logger = logging.getLogger(__name__)


class BaselineRepository:
    """Repository for managing user baseline statistics."""

    def __init__(self, session):
        """Initialize repository.

        Args:
            session: SQLAlchemy session

        """
        self._session = session
        self._logger = logging.getLogger(__name__)

    def get_baseline(self, user_id: str, biomarker_name: str):
        """Get baseline for user and biomarker.

        Args:
            user_id: User ID
            biomarker_name: Name of biomarker

        Returns:
            UserBaseline instance or None

        """
        stmt = select(UserBaseline).where(
            UserBaseline.user_id == user_id,
            UserBaseline.biomarker_name == biomarker_name,
        )
        result = self._session.execute(stmt)
        return result.scalar_one_or_none()

    def save_baseline(self, baseline: UserBaseline) -> None:
        """Save or update baseline (upsert).

        Args:
            baseline: UserBaseline instance

        """
        # Use PostgreSQL INSERT ... ON CONFLICT UPDATE (upsert)
        stmt = insert(UserBaseline).values(
            user_id=baseline.user_id,
            biomarker_name=baseline.biomarker_name,
            mean=baseline.mean,
            std=baseline.std,
            percentile_25=baseline.percentile_25,
            percentile_75=baseline.percentile_75,
            data_points=baseline.data_points,
            window_start=baseline.window_start,
            window_end=baseline.window_end,
        )

        # On conflict (user_id, biomarker_name), update all fields
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id", "biomarker_name"],
            set_={
                "mean": stmt.excluded.mean,
                "std": stmt.excluded.std,
                "percentile_25": stmt.excluded.percentile_25,
                "percentile_75": stmt.excluded.percentile_75,
                "data_points": stmt.excluded.data_points,
                "window_start": stmt.excluded.window_start,
                "window_end": stmt.excluded.window_end,
            },
        )

        self._session.execute(stmt)
        self._session.commit()
        self._logger.info(
            f"Saved baseline for {baseline.user_id}/{baseline.biomarker_name}"
        )

    def get_all_baselines(self, user_id: str) -> dict[str, UserBaseline]:
        """Get all baselines for user.

        Args:
            user_id: User ID

        Returns:
            Dictionary mapping biomarker name to UserBaseline

        """
        stmt = select(UserBaseline).where(UserBaseline.user_id == user_id)
        result = self._session.execute(stmt)
        baselines = result.scalars().all()

        return {baseline.biomarker_name: baseline for baseline in baselines}
