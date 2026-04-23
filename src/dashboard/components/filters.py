"""Filter components for dashboard user and time selection."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from sqlalchemy import func, select

from src.dashboard.data.config import get_current_config
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import Biomarker, Context, Indicator

logger = get_logger(__name__)


def get_display_timezone() -> ZoneInfo:
    """Get the timezone from config for display purposes.

    Returns:
        ZoneInfo instance for the configured timezone
    """
    config = get_current_config()
    return ZoneInfo(config.timezone)


def get_available_users() -> list[str]:
    """Get list of distinct user IDs from database.

    Returns:
        List of user IDs, empty list on error.
    """
    try:
        with SessionLocal() as db:
            result = (
                db.execute(
                    select(func.distinct(Biomarker.user_id)).order_by(Biomarker.user_id)
                )
                .scalars()
                .all()
            )
            return list(result)
    except Exception:
        logger.error("Failed to get users", exc_info=True)
        return []


def get_preset_range(preset: str) -> tuple[datetime, datetime]:
    """Get date range for a preset selection.

    Args:
        preset: One of '14d', '30d', '3m', 'custom'

    Returns:
        Tuple of (start_datetime, end_datetime) in config timezone
    """
    tz = get_display_timezone()
    end = datetime.now(tz)
    if preset == "14d":
        start = end - timedelta(days=14)
    elif preset == "30d":
        start = end - timedelta(days=30)
    elif preset == "3m":
        start = end - timedelta(days=90)
    else:  # custom
        start = end - timedelta(days=14)  # default
    return start, end


def init_filter_session_state() -> None:
    """Initialize filter session state with defaults."""
    tz = get_display_timezone()
    if "selected_user_id" not in st.session_state:
        st.session_state["selected_user_id"] = None
    if "selected_start_date" not in st.session_state:
        st.session_state["selected_start_date"] = datetime.now(tz) - timedelta(days=14)
    if "selected_end_date" not in st.session_state:
        st.session_state["selected_end_date"] = datetime.now(tz)


def user_selector(key: str = "user_select") -> str | None:
    """Render user selection dropdown.

    Args:
        key: Unique key for the Streamlit widget

    Returns:
        Selected user_id or None if no selection
    """
    init_filter_session_state()

    users = get_available_users()

    if not users:
        st.warning("No users found in database")
        return None

    # Get current selection - check widget state first (updated by Streamlit on interaction),
    # then fall back to our session state for cross-page persistence
    current_user = st.session_state.get(key) or st.session_state.get("selected_user_id")

    # Find index of current selection
    index = 0
    if current_user and current_user in users:
        index = users.index(current_user)

    selected = st.selectbox(
        "Select User",
        options=users,
        index=index,
        key=key,
    )

    # Update session state
    st.session_state["selected_user_id"] = selected
    return selected


def time_range_selector(key: str = "time_range") -> tuple[datetime, datetime]:
    """Render time range selection with presets and date pickers.

    Args:
        key: Unique key prefix for the Streamlit widgets

    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    init_filter_session_state()

    # Preset buttons
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    def apply_preset(preset: str) -> None:
        """Apply a date preset and update all relevant session state keys."""
        start, end = get_preset_range(preset)
        st.session_state["selected_start_date"] = start
        st.session_state["selected_end_date"] = end
        # Also update the widget keys directly so date pickers reflect the change
        st.session_state[f"{key}_start"] = start.date() if isinstance(start, datetime) else start
        st.session_state[f"{key}_end"] = end.date() if isinstance(end, datetime) else end

    with preset_col1:
        if st.button("Last 14 Days", key=f"{key}_14d"):
            apply_preset("14d")
            st.rerun()

    with preset_col2:
        if st.button("Last 30 Days", key=f"{key}_30d"):
            apply_preset("30d")
            st.rerun()

    with preset_col3:
        if st.button("Last 3 Months", key=f"{key}_3m"):
            apply_preset("3m")
            st.rerun()

    with preset_col4:
        if st.button("Custom", key=f"{key}_custom"):
            pass  # Just allow manual date selection

    # Date pickers
    col1, col2 = st.columns(2)

    tz = get_display_timezone()

    # Note: Don't pass 'value' if the widget key already exists in session state
    # to avoid Streamlit's "default value but also session state" warning
    start_key = f"{key}_start"
    end_key = f"{key}_end"

    with col1:
        start_value = st.session_state.get(
            "selected_start_date", datetime.now(tz) - timedelta(days=14)
        )
        # Convert to date for date_input if needed
        if isinstance(start_value, datetime):
            start_value = start_value.date()

        if start_key in st.session_state:
            start_date = st.date_input(
                "Start Date",
                key=start_key,
            )
        else:
            start_date = st.date_input(
                "Start Date",
                value=start_value,
                key=start_key,
            )

    with col2:
        end_value = st.session_state.get("selected_end_date", datetime.now(tz))
        if isinstance(end_value, datetime):
            end_value = end_value.date()

        if end_key in st.session_state:
            end_date = st.date_input(
                "End Date",
                key=end_key,
            )
        else:
            end_date = st.date_input(
                "End Date",
                value=end_value,
                key=end_key,
            )

    # Validate end_date >= start_date
    if end_date < start_date:
        st.error("End date must be after start date")
        end_date = start_date

    # Convert to datetime with timezone
    start_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
    end_datetime = datetime.combine(end_date, datetime.max.time(), tzinfo=tz)

    # Update session state
    st.session_state["selected_start_date"] = start_datetime
    st.session_state["selected_end_date"] = end_datetime

    return start_datetime, end_datetime


def get_selection_summary(
    user_id: str, start: datetime, end: datetime
) -> dict[str, int | bool]:
    """Get data counts for current selection.

    Args:
        user_id: Selected user ID
        start: Start datetime
        end: End datetime

    Returns:
        Dict with counts for biomarkers, contexts, indicators, and has_data flag
    """
    try:
        with SessionLocal() as db:
            biomarker_count = (
                db.execute(
                    select(func.count(Biomarker.id))
                    .where(Biomarker.user_id == user_id)
                    .where(Biomarker.timestamp >= start)
                    .where(Biomarker.timestamp <= end)
                ).scalar()
                or 0
            )

            context_count = (
                db.execute(
                    select(func.count(Context.id))
                    .where(Context.user_id == user_id)
                    .where(Context.timestamp >= start)
                    .where(Context.timestamp <= end)
                ).scalar()
                or 0
            )

            indicator_count = (
                db.execute(
                    select(func.count(Indicator.id))
                    .where(Indicator.user_id == user_id)
                    .where(Indicator.timestamp >= start)
                    .where(Indicator.timestamp <= end)
                ).scalar()
                or 0
            )

            return {
                "biomarkers": biomarker_count,
                "contexts": context_count,
                "indicators": indicator_count,
                "has_data": biomarker_count > 0 or context_count > 0,
            }
    except Exception:
        logger.error("Failed to get selection summary", exc_info=True)
        return {"biomarkers": 0, "contexts": 0, "indicators": 0, "has_data": False}


def render_selection_summary(
    user_id: str | None, start: datetime, end: datetime
) -> None:
    """Render data summary metrics for current selection.

    Args:
        user_id: Selected user ID (or None)
        start: Start datetime
        end: End datetime
    """
    if user_id is None:
        st.info("Select a user to see data summary")
        return

    summary = get_selection_summary(user_id, start, end)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Biomarkers", summary["biomarkers"])

    with col2:
        st.metric("Context Records", summary["contexts"])

    with col3:
        st.metric("Indicators", summary["indicators"])

    if not summary["has_data"]:
        st.warning("No data available for the selected user and time range")


def render_filter_sidebar() -> tuple[str | None, datetime, datetime]:
    """Render complete filter sidebar with user and time selection.

    Convenience wrapper that renders user selector, time range selector,
    and selection summary in the sidebar.

    Returns:
        Tuple of (user_id, start_datetime, end_datetime)

    .. deprecated::
        Use render_user_sidebar() for user-only sidebar and
        render_inline_date_range() for inline date selection.
    """
    with st.sidebar:
        st.subheader("Filters")

        user_id = user_selector()

        st.divider()

        start, end = time_range_selector()

        st.divider()

        render_selection_summary(user_id, start, end)

    return user_id, start, end


def render_user_sidebar() -> str | None:
    """Render sidebar with user selection only.

    Use this when date filtering is handled inline in the page content.

    Returns:
        Selected user_id or None if no selection
    """
    init_filter_session_state()

    with st.sidebar:
        st.subheader("Filters")
        user_id = user_selector()

    return user_id


def render_inline_date_range(
    key_prefix: str = "inline_date",
    show_presets: bool = True,
    presets: list[str] | None = None,
) -> tuple[datetime, datetime]:
    """Render date range selector inline (not in sidebar).

    Use this in the main content area for page-specific date filtering.

    Args:
        key_prefix: Unique key prefix for widgets (use different prefixes per page)
        show_presets: Whether to show preset buttons
        presets: List of preset keys to show ("14d", "30d", "3m"). Defaults to all three.

    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    init_filter_session_state()

    tz = get_display_timezone()

    # Get current values
    start_value = st.session_state.get(
        "selected_start_date", datetime.now(tz) - timedelta(days=14)
    )
    end_value = st.session_state.get("selected_end_date", datetime.now(tz))

    if isinstance(start_value, datetime):
        start_value = start_value.date()
    if isinstance(end_value, datetime):
        end_value = end_value.date()

    _preset_labels = {"14d": "Last 14 Days", "30d": "Last 30 Days", "3m": "Last 3 Months"}
    active_presets = presets if presets is not None else ["14d", "30d", "3m"]

    # Compact single-row layout: [presets] [start] [to] [end]
    if show_presets and active_presets:
        n = len(active_presets)
        cols = st.columns([1] * n + [0.3, 1.5, 0.3, 1.5])
        preset_cols = cols[:n]
        date_cols = [cols[n + 1], cols[n + 3]]

        def apply_inline_preset(preset: str) -> None:
            """Apply a date preset and update all relevant session state keys."""
            start, end = get_preset_range(preset)
            st.session_state["selected_start_date"] = start
            st.session_state["selected_end_date"] = end
            # Also update the widget keys directly so date pickers reflect the change
            st.session_state[f"{key_prefix}_start"] = start.date() if isinstance(start, datetime) else start
            st.session_state[f"{key_prefix}_end"] = end.date() if isinstance(end, datetime) else end

        # Preset buttons
        for i, preset_key in enumerate(active_presets):
            with preset_cols[i]:
                if st.button(_preset_labels[preset_key], key=f"{key_prefix}_{preset_key}", use_container_width=True):
                    apply_inline_preset(preset_key)
                    st.rerun()

        # "to" label
        with cols[n]:
            st.write("")  # Spacer

        with cols[n + 2]:
            st.markdown(
                "<div style='text-align:center; padding-top:0.5rem; color:#888;'>to</div>",
                unsafe_allow_html=True,
            )
    else:
        cols = st.columns([1, 0.2, 1])
        date_cols = [cols[0], cols[2]]
        with cols[1]:
            st.markdown(
                "<div style='text-align:center; padding-top:0.5rem; color:#888;'>to</div>",
                unsafe_allow_html=True,
            )

    # Date pickers
    # Note: Don't pass 'value' if the widget key already exists in session state
    # to avoid Streamlit's "default value but also session state" warning
    start_key = f"{key_prefix}_start"
    end_key = f"{key_prefix}_end"

    with date_cols[0]:
        if start_key in st.session_state:
            start_date = st.date_input(
                "Start",
                key=start_key,
                label_visibility="collapsed",
            )
        else:
            start_date = st.date_input(
                "Start",
                value=start_value,
                key=start_key,
                label_visibility="collapsed",
            )

    with date_cols[1]:
        if end_key in st.session_state:
            end_date = st.date_input(
                "End",
                key=end_key,
                label_visibility="collapsed",
            )
        else:
            end_date = st.date_input(
                "End",
                value=end_value,
                key=end_key,
                label_visibility="collapsed",
            )

    # Validate end_date >= start_date
    if end_date < start_date:
        st.error("End date must be after start date")
        end_date = start_date

    # Convert to datetime with timezone
    start_datetime = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
    end_datetime = datetime.combine(end_date, datetime.max.time(), tzinfo=tz)

    # Update session state
    st.session_state["selected_start_date"] = start_datetime
    st.session_state["selected_end_date"] = end_datetime

    return start_datetime, end_datetime
