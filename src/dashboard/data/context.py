"""Context marker data loading and processing functions."""

from datetime import datetime

import pandas as pd
from sqlalchemy import select

from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import Context

logger = get_logger(__name__)


def load_context_markers(
    user_id: str,
    start: datetime,
    end: datetime,
    context_types: list[str] | None = None,
) -> pd.DataFrame:
    """Load context marker data and expand JSON values to rows.

    Args:
        user_id: User ID to filter by
        start: Start datetime (inclusive)
        end: End datetime (inclusive)
        context_types: Optional list of types to include (e.g., "environment")

    Returns:
        DataFrame with columns: timestamp, type, name, value, source
    """
    try:
        with SessionLocal() as db:
            query = (
                select(Context)
                .where(Context.user_id == user_id)
                .where(Context.timestamp >= start)
                .where(Context.timestamp <= end)
                .order_by(Context.timestamp.desc())
            )
            if context_types:
                query = query.where(Context.context_type.in_(context_types))

            results = db.execute(query).scalars().all()

        # Expand JSON values to rows
        rows = []
        for ctx in results:
            source = ctx.metadata_.get("source", "") if ctx.metadata_ else ""
            for name, value in ctx.value.items():
                rows.append(
                    {
                        "timestamp": ctx.timestamp,
                        "type": ctx.context_type,
                        "name": name,
                        "value": float(value),
                        "source": source,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "type", "name", "value", "source"]
            )

        return pd.DataFrame(rows)

    except Exception:
        logger.error("Failed to load context markers", exc_info=True)
        return pd.DataFrame(columns=["timestamp", "type", "name", "value", "source"])


def filter_by_names(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Filter DataFrame by context marker names.

    Args:
        df: DataFrame with context data
        names: List of marker names to include (empty = all)

    Returns:
        Filtered DataFrame
    """
    if not names:
        return df
    return df[df["name"].isin(names)]


def generate_csv_filename(user_id: str, start: datetime, end: datetime) -> str:
    """Generate CSV export filename.

    Args:
        user_id: User ID
        start: Start datetime
        end: End datetime

    Returns:
        Filename string like context_user123_20240101_20240131.csv
    """
    return f"context_{user_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"


def calculate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics per context marker name.

    Args:
        df: DataFrame with context data

    Returns:
        DataFrame with columns: Marker, Count, Mean, Std Dev, Min, Max
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Marker", "Count", "Mean", "Std Dev", "Min", "Max"]
        )

    stats = (
        df.groupby("name")["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
        .reset_index()
    )
    stats.columns = ["Marker", "Count", "Mean", "Std Dev", "Min", "Max"]
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
