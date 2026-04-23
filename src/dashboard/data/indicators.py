"""Indicator data loading and processing functions."""

from datetime import datetime
from typing import Literal

import pandas as pd
from sqlalchemy import select

from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import Indicator

logger = get_logger(__name__)


def load_indicators(
    user_id: str,
    start: datetime,
    end: datetime,
    indicator_types: list[str] | None = None,
    presence_filter: Literal["all", "present", "absent"] = "all",
) -> pd.DataFrame:
    """Load indicator data for a user and time range.

    Args:
        user_id: User ID to filter by
        start: Start datetime (inclusive)
        end: End datetime (inclusive)
        indicator_types: Optional list of types to include
        presence_filter: Filter by presence flag ("all", "present", "absent")

    Returns:
        DataFrame with indicator data
    """
    try:
        with SessionLocal() as db:
            query = (
                select(Indicator)
                .where(Indicator.user_id == user_id)
                .where(Indicator.timestamp >= start)
                .where(Indicator.timestamp <= end)
                .order_by(Indicator.timestamp.desc())
            )

            if indicator_types:
                query = query.where(Indicator.indicator_type.in_(indicator_types))

            if presence_filter == "present":
                query = query.where(Indicator.presence_flag == True)  # noqa: E712
            elif presence_filter == "absent":
                query = query.where(
                    (Indicator.presence_flag == False)  # noqa: E712
                    | (Indicator.presence_flag.is_(None))
                )

            results = db.execute(query).scalars().all()

        rows = []
        for ind in results:
            rows.append(
                {
                    "timestamp": ind.timestamp,
                    "indicator_type": ind.indicator_type,
                    "likelihood": ind.value,
                    "presence_flag": ind.presence_flag,
                    "data_reliability_score": ind.data_reliability_score,
                    "context_used": ind.context_used or "N/A",
                    "analysis_run_id": str(ind.analysis_run_id),
                }
            )

        if not rows:
            return _empty_indicator_df()

        return pd.DataFrame(rows)

    except Exception:
        logger.error("Failed to load indicators", exc_info=True)
        return _empty_indicator_df()


def _empty_indicator_df() -> pd.DataFrame:
    """Return empty DataFrame with correct schema."""
    return pd.DataFrame(
        columns=[
            "timestamp",
            "indicator_type",
            "likelihood",
            "presence_flag",
            "data_reliability_score",
            "context_used",
            "analysis_run_id",
        ]
    )


def filter_by_types(df: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Filter DataFrame by indicator types."""
    if not types:
        return df
    return df[df["indicator_type"].isin(types)]


def generate_csv_filename(user_id: str, start: datetime, end: datetime) -> str:
    """Generate CSV export filename."""
    return (
        f"indicators_{user_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )


def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics per indicator type.

    Returns:
        DataFrame with: Indicator, Count, Mean Likelihood, Std Dev, % Present
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Indicator", "Count", "Mean Likelihood", "Std Dev", "% Present"]
        )

    # Calculate present count separately to avoid FutureWarning
    presence_counts = (
        df.groupby("indicator_type")["presence_flag"]
        .apply(lambda x: (x == True).sum())  # noqa: E712
        .reset_index(name="Present_Count")
    )

    stats = (
        df.groupby("indicator_type")
        .agg(
            Count=("likelihood", "count"),
            Mean_Likelihood=("likelihood", "mean"),
            Std_Dev=("likelihood", "std"),
        )
        .reset_index()
    )

    stats = stats.merge(presence_counts, on="indicator_type")

    stats["% Present"] = (stats["Present_Count"] / stats["Count"] * 100).round(1)
    stats = stats.drop(columns=["Present_Count"])
    stats.columns = ["Indicator", "Count", "Mean Likelihood", "Std Dev", "% Present"]
    stats["Mean Likelihood"] = stats["Mean Likelihood"].round(4)
    stats["Std Dev"] = stats["Std Dev"].round(4)

    return stats


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
