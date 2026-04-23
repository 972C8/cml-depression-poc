"""Timeline data loading and aggregation functions."""

from datetime import datetime

import pandas as pd

from src.dashboard.data.biomarkers import load_biomarkers
from src.dashboard.data.context_evaluation import load_context_history_records
from src.dashboard.data.indicators import load_indicators
from src.shared.logging import get_logger

logger = get_logger(__name__)


def load_timeline_biomarkers(
    user_id: str,
    start: datetime,
    end: datetime,
    names: list[str] | None = None,
    resolution_minutes: int = 15,
) -> pd.DataFrame:
    """Load biomarker data optimized for timeline visualization.

    Args:
        user_id: User ID to filter by
        start: Start datetime (inclusive)
        end: End datetime (inclusive)
        names: Optional list of biomarker names to include
        resolution_minutes: Aggregate to this time resolution

    Returns:
        DataFrame with: timestamp, name, value (aggregated)
    """
    df = load_biomarkers(user_id=user_id, start=start, end=end)

    if df.empty:
        return _empty_timeline_df()

    if names:
        df = df[df["name"].isin(names)]

    if df.empty:
        return _empty_timeline_df()

    # Aggregate to time resolution
    df = _aggregate_to_resolution(df, resolution_minutes, "name", "value")

    return df.sort_values("timestamp")


def load_timeline_indicators(
    user_id: str,
    start: datetime,
    end: datetime,
    types: list[str] | None = None,
    resolution_minutes: int = 15,
) -> pd.DataFrame:
    """Load indicator data optimized for timeline visualization.

    Args:
        user_id: User ID to filter by
        start: Start datetime (inclusive)
        end: End datetime (inclusive)
        types: Optional list of indicator types to include
        resolution_minutes: Aggregate to this time resolution

    Returns:
        DataFrame with: timestamp, indicator_type, likelihood (aggregated)
    """
    df = load_indicators(user_id=user_id, start=start, end=end, indicator_types=types)

    if df.empty:
        return _empty_indicator_timeline_df()

    # Aggregate to time resolution
    df = _aggregate_to_resolution(
        df, resolution_minutes, "indicator_type", "likelihood"
    )

    return df.sort_values("timestamp")


def load_context_periods(
    user_id: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load context periods for background shading.

    Uses stored context history records instead of on-the-fly evaluation.

    Args:
        user_id: User ID
        start: Start datetime
        end: End datetime

    Returns:
        DataFrame with: start_time, end_time, context (for vrect shading)
    """
    df = load_context_history_records(
        user_id=user_id,
        start=start,
        end=end,
    )

    if df.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "context"])

    # Convert point data to periods
    periods = []
    df = df.sort_values("evaluated_at")

    prev_context = None
    period_start = None

    for _, row in df.iterrows():
        if row["dominant_context"] != prev_context:
            if prev_context is not None and period_start is not None:
                periods.append(
                    {
                        "start_time": period_start,
                        "end_time": row["evaluated_at"],
                        "context": prev_context,
                    }
                )
            period_start = row["evaluated_at"]
            prev_context = row["dominant_context"]

    # Close final period
    if prev_context is not None and period_start is not None:
        periods.append(
            {
                "start_time": period_start,
                "end_time": df.iloc[-1]["evaluated_at"],
                "context": prev_context,
            }
        )

    return pd.DataFrame(periods)


def _aggregate_to_resolution(
    df: pd.DataFrame,
    resolution_minutes: int,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Aggregate data to time resolution.

    Args:
        df: DataFrame with timestamp column
        resolution_minutes: Target resolution in minutes
        group_col: Column to group by (e.g., 'name', 'indicator_type')
        value_col: Column to aggregate (e.g., 'value', 'likelihood')

    Returns:
        Aggregated DataFrame
    """
    if resolution_minutes <= 0:
        return df

    # Floor timestamps to resolution
    df = df.copy()
    df["timestamp"] = df["timestamp"].dt.floor(f"{resolution_minutes}min")

    # Aggregate by timestamp and group
    agg_df = df.groupby(["timestamp", group_col])[value_col].mean().reset_index()

    return agg_df


def _empty_timeline_df() -> pd.DataFrame:
    """Return empty DataFrame for biomarker timeline."""
    return pd.DataFrame(columns=["timestamp", "name", "value"])


def _empty_indicator_timeline_df() -> pd.DataFrame:
    """Return empty DataFrame for indicator timeline."""
    return pd.DataFrame(columns=["timestamp", "indicator_type", "likelihood"])


def generate_timeline_csv_filename(
    user_id: str, start: datetime, end: datetime, data_type: str
) -> str:
    """Generate CSV export filename for timeline data."""
    return f"timeline_{data_type}_{user_id}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
