"""Shared layout components for dashboard pages."""

from datetime import datetime

import streamlit as st
from sqlalchemy import text

from src.dashboard.components.filters import get_display_timezone
from src.shared.database import SessionLocal
from src.shared.logging import get_logger

logger = get_logger(__name__)


def check_database_connection() -> tuple[bool, str]:
    """Check if database connection is healthy."""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True, "Connected"
    except Exception as e:
        logger.error("Database connection failed", exc_info=True)
        return False, str(e)


def get_last_analysis_timestamp() -> datetime | None:
    """Get timestamp of most recent analysis run."""
    try:
        from sqlalchemy import select

        from src.shared.models import AnalysisRun

        with SessionLocal() as db:
            result = db.execute(
                select(AnalysisRun.created_at)
                .order_by(AnalysisRun.created_at.desc())
                .limit(1)
            ).scalar()
            return result
    except Exception:
        logger.error("Failed to get last analysis timestamp", exc_info=True)
        return None


def render_page_header(title: str, icon: str, subtitle: str | None = None) -> None:
    """Render consistent page header.

    Args:
        title: Page title text
        icon: Emoji icon for the page
        subtitle: Optional subtitle text
    """
    st.title(f"{icon} {title}")
    if subtitle:
        st.caption(subtitle)


def render_sidebar_status() -> None:
    """Render system status in sidebar."""
    with st.sidebar:
        st.subheader("System Status")

        # Database connection status
        is_connected, status_msg = check_database_connection()
        if is_connected:
            st.success(f"Database: {status_msg}")
        else:
            st.error(f"Database: {status_msg}")

        # Last analysis run
        last_run = get_last_analysis_timestamp()
        if last_run:
            tz = get_display_timezone()
            local_time = last_run.astimezone(tz) if last_run.tzinfo else last_run.replace(tzinfo=tz)
            st.caption(f"Last analysis: {local_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("No analysis runs yet")

        st.divider()


def render_footer(show_version: bool = True) -> None:
    """Render page footer with timestamp and version.

    Args:
        show_version: Whether to show version information
    """
    st.divider()
    footer_cols = st.columns([3, 1])

    with footer_cols[0]:
        tz = get_display_timezone()
        st.caption(f"Generated: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}")

    if show_version:
        with footer_cols[1]:
            st.caption("MT_POC v0.1.0")


def get_page_config(
    page_title: str,
    page_icon: str,
    layout: str = "wide",
) -> dict:
    """Get consistent page configuration dict.

    Args:
        page_title: Title for browser tab
        page_icon: Emoji icon
        layout: Page layout mode ("wide" or "centered")

    Returns:
        Dict with page configuration
    """
    return {
        "page_title": f"{page_title} - MT_POC",
        "page_icon": page_icon,
        "layout": layout,
        "initial_sidebar_state": "expanded",
    }
