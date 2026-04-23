"""Home page - Dashboard entry point with navigation and system overview."""


import streamlit as st
from sqlalchemy import func, select

from src.dashboard.components.filters import get_display_timezone
from src.dashboard.components.layout import (
    check_database_connection,
    render_sidebar_status,
)
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import AnalysisRun, Biomarker, Context, Indicator

logger = get_logger(__name__)


def get_data_counts() -> dict[str, int]:
    """Query database for summary counts."""
    try:
        with SessionLocal() as db:
            user_count = (
                db.execute(
                    select(func.count(func.distinct(Biomarker.user_id)))
                ).scalar()
                or 0
            )
            biomarker_count = db.execute(select(func.count(Biomarker.id))).scalar() or 0
            context_count = db.execute(select(func.count(Context.id))).scalar() or 0
            indicator_count = db.execute(select(func.count(Indicator.id))).scalar() or 0
            return {
                "users": user_count,
                "biomarkers": biomarker_count,
                "contexts": context_count,
                "indicators": indicator_count,
            }
    except Exception:
        logger.error("Failed to get data counts", exc_info=True)
        return {"users": 0, "biomarkers": 0, "contexts": 0, "indicators": 0}


def get_analysis_stats() -> dict:
    """Get analysis run statistics."""
    try:
        with SessionLocal() as db:
            total_runs = db.execute(select(func.count(AnalysisRun.id))).scalar() or 0
            last_run = db.execute(
                select(AnalysisRun.created_at)
                .order_by(AnalysisRun.created_at.desc())
                .limit(1)
            ).scalar()
            return {"total_runs": total_runs, "last_run": last_run}
    except Exception:
        logger.error("Failed to get analysis stats", exc_info=True)
        return {"total_runs": 0, "last_run": None}


# Track current page for cross-page state management
st.session_state["_current_page"] = "home"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
render_sidebar_status()

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.title("Multimodal Biomarker Analysis")
st.markdown(
    "Transform speech and network biomarkers into actionable mental health indicators "
    "through transparent, white-box analysis."
)

# System status bar
db_ok, db_msg = check_database_connection()
counts = get_data_counts()
analysis_stats = get_analysis_stats()

status_cols = st.columns([1, 1, 3])

with status_cols[0]:
    if analysis_stats["last_run"]:
        tz = get_display_timezone()
        last = analysis_stats["last_run"]
        local_time = last.astimezone(tz) if last.tzinfo else last.replace(tzinfo=tz)
        st.info(f"Last run: {local_time.strftime('%b %d, %H:%M')}")
    else:
        st.warning("No analysis runs yet")

with status_cols[1]:
    st.info(f"{analysis_stats['total_runs']} total runs")

st.divider()

# ---------------------------------------------------------------------------
# Navigation Cards
# ---------------------------------------------------------------------------
st.subheader("Dashboard")

CARD_HEIGHT = 180

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown("#### ⚙️ Analysis")
        st.caption(
            "Run the analysis pipeline on biomarker data. "
            "Inspect how raw signals become indicators."
        )
        st.page_link("pages/1_⚙️_Analysis.py", label="Run Analysis")

with nav_col2:
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown("#### 📋 Context")
        st.caption(
            "Evaluate how environmental context affects "
            "biomarker interpretation and confidence."
        )
        st.page_link("pages/2_📋_Context.py", label="Evaluate Context")

with nav_col3:
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown("#### ⚗️ Experiment")
        st.caption(
            "Adjust analysis parameters and compare results. "
            "Test different configurations."
        )
        st.page_link("pages/3_⚗️_Experiment.py", label="Open Experiment")

nav_col4, nav_col5, nav_col6 = st.columns(3)

with nav_col4:
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown("#### 🧪 Mock Data")
        st.caption(
            "Generate synthetic biomarker and context data "
            "with configurable scenarios."
        )
        st.page_link("pages/4_🧪_Generate_Mock_Data.py", label="Generate Data")

with nav_col5:
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown("#### 📊 Data")
        st.caption(
            "Browse raw biomarkers, context records, and "
            "computed indicators. Export data."
        )
        st.page_link("pages/5_📊_Data.py", label="View Data")

with nav_col6:
    with st.container(border=True, height=CARD_HEIGHT):
        st.markdown("#### 📚 API")
        st.caption(
            "See the API documentation for sending data records (biomarkers,context) and reading results."
        )
        st.page_link("http://localhost:8000/docs", label="View API")

st.divider()

# ---------------------------------------------------------------------------
# Data Overview & Quick Actions
# ---------------------------------------------------------------------------
st.subheader("Data Overview")

metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Users", counts["users"])
with metric_cols[1]:
    st.metric("Biomarkers", f"{counts['biomarkers']:,}")
with metric_cols[2]:
    st.metric("Context Records", f"{counts['contexts']:,}")
with metric_cols[3]:
    st.metric("Indicators", f"{counts['indicators']:,}")

if counts["biomarkers"] == 0:
    st.info(
        "No biomarker data yet. Generate mock data or ingest data via the API "
        "to get started.",
        icon="💡",
    )

st.divider()

# ---------------------------------------------------------------------------
# Getting Started
# ---------------------------------------------------------------------------
st.subheader("Getting Started")
st.markdown(
    """
        **New to MT_POC?** Follow these steps to run your first analysis:

        1. **Generate Data** — Use the Mock Data page to create test biomarkers and context
        2. **Evaluate Context** - Understand how context changes through time
        3. **Run Analysis** — Process the data through the analysis pipeline
        4. **View Results** — Explore all results through the transparent pipeline
        5. **Experiment** — Use the experiment page to adjust parameters and compare different configurations
        6. **Explore Data** — Use the data page to browse raw and computed data, export as needed
        """
)
