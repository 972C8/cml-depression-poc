"""Data page - unified view of biomarkers, indicators, and context markers.

This page consolidates raw data views from the former Biomarkers, Indicators,
and Context pages into a single unified data exploration interface.
"""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

from src.dashboard.components.charts import (  # noqa: E402
    add_context_shading,
    render_biomarker_timeline_chart,
    render_indicator_timeline_chart,
)
from src.dashboard.components.filters import (  # noqa: E402
    init_filter_session_state,
    render_inline_date_range,
    render_user_sidebar,
)
from src.dashboard.components.layout import (  # noqa: E402
    render_footer,
    render_page_header,
    render_sidebar_status,
)
from src.dashboard.data.biomarkers import (  # noqa: E402
    calculate_page_indices as bio_page_indices,
)
from src.dashboard.data.biomarkers import (
    calculate_summary_stats as bio_summary_stats,
)
from src.dashboard.data.biomarkers import (
    calculate_total_pages as bio_total_pages,
)
from src.dashboard.data.biomarkers import (
    filter_by_names as bio_filter_by_names,
)
from src.dashboard.data.biomarkers import (
    generate_csv_filename as bio_csv_filename,
)
from src.dashboard.data.biomarkers import (
    load_biomarkers,
)
from src.dashboard.data.context import (  # noqa: E402
    calculate_page_indices as context_page_indices,
)
from src.dashboard.data.context import (
    calculate_summary_stats as context_summary_stats,
)
from src.dashboard.data.context import (
    calculate_total_pages as context_total_pages,
)
from src.dashboard.data.context import (
    filter_by_names as context_filter_by_names,
)
from src.dashboard.data.context import (
    generate_csv_filename as context_csv_filename,
)
from src.dashboard.data.context import (
    load_context_markers,
)
from src.dashboard.data.context_evaluation import (  # noqa: E402
    calculate_page_indices as eval_page_indices,
)
from src.dashboard.data.context_evaluation import (
    calculate_total_pages as eval_total_pages,
)
from src.dashboard.data.context_evaluation import (
    generate_evaluation_csv_filename,
    load_context_history_records,
)
from src.dashboard.data.indicators import (  # noqa: E402
    calculate_page_indices as ind_page_indices,
)
from src.dashboard.data.indicators import (
    calculate_summary_stats as ind_summary_stats,
)
from src.dashboard.data.indicators import (
    calculate_total_pages as ind_total_pages,
)
from src.dashboard.data.indicators import (
    generate_csv_filename as ind_csv_filename,
)
from src.dashboard.data.indicators import (
    load_indicators,
)
from src.dashboard.data.timeline import (  # noqa: E402
    generate_timeline_csv_filename,
    load_context_periods,
    load_timeline_biomarkers,
    load_timeline_indicators,
)

# ==============================================================================
# Session State Initialization
# ==============================================================================

# Biomarkers session state
if "data_bio_types_select" not in st.session_state:
    st.session_state["data_bio_types_select"] = []
if "data_bio_names_select" not in st.session_state:
    st.session_state["data_bio_names_select"] = []
if "data_bio_page_size" not in st.session_state:
    st.session_state["data_bio_page_size"] = 50
if "data_bio_current_page" not in st.session_state:
    st.session_state["data_bio_current_page"] = 1
if "data_bio_view_mode" not in st.session_state:
    st.session_state["data_bio_view_mode"] = "Table"
if "data_bio_timeline_resolution" not in st.session_state:
    st.session_state["data_bio_timeline_resolution"] = 15
if "data_bio_show_context" not in st.session_state:
    st.session_state["data_bio_show_context"] = True
if "data_bio_timeline_select" not in st.session_state:
    st.session_state["data_bio_timeline_select"] = []

# Indicators session state
if "data_ind_types_select" not in st.session_state:
    st.session_state["data_ind_types_select"] = []
if "data_ind_presence_radio" not in st.session_state:
    st.session_state["data_ind_presence_radio"] = "All"
if "data_ind_page_size" not in st.session_state:
    st.session_state["data_ind_page_size"] = 50
if "data_ind_current_page" not in st.session_state:
    st.session_state["data_ind_current_page"] = 1
if "data_ind_view_mode" not in st.session_state:
    st.session_state["data_ind_view_mode"] = "Table"
if "data_ind_timeline_resolution" not in st.session_state:
    st.session_state["data_ind_timeline_resolution"] = 15
if "data_ind_show_context" not in st.session_state:
    st.session_state["data_ind_show_context"] = True
if "data_ind_dsm_threshold" not in st.session_state:
    st.session_state["data_ind_dsm_threshold"] = 0.5
if "data_ind_timeline_select" not in st.session_state:
    st.session_state["data_ind_timeline_select"] = []

# Context markers session state
if "data_ctx_type_filter" not in st.session_state:
    st.session_state["data_ctx_type_filter"] = []
if "data_ctx_name_filter" not in st.session_state:
    st.session_state["data_ctx_name_filter"] = []
if "data_ctx_raw_page" not in st.session_state:
    st.session_state["data_ctx_raw_page"] = 1
if "data_ctx_raw_page_size" not in st.session_state:
    st.session_state["data_ctx_raw_page_size"] = 50

# Context evaluation session state
if "data_ctx_eval_page" not in st.session_state:
    st.session_state["data_ctx_eval_page"] = 1
if "data_ctx_eval_page_size" not in st.session_state:
    st.session_state["data_ctx_eval_page_size"] = 50

# Initialize filter session state
init_filter_session_state()

# Track current page for cross-page state management
st.session_state["_current_page"] = "data"

# Render user selector in sidebar
user_id = render_user_sidebar()

# System status at end of sidebar
render_sidebar_status()

# Page content
render_page_header(
    "Data",
    "📊",
    "Explore biomarkers, indicators, and context marker data",
)

if user_id is None:
    st.warning("Please select a user from the sidebar to view data.")
    render_footer()
    st.stop()

# ==============================================================================
# Inline Date Range Selection
# ==============================================================================

start_datetime, end_datetime = render_inline_date_range(key_prefix="data_page")
st.divider()

# ==============================================================================
# Main Tabs
# ==============================================================================

tab_bio, tab_ctx, tab_ind = st.tabs(["Biomarkers", "Context", "Indicators"])

# ==============================================================================
# BIOMARKERS TAB
# ==============================================================================

with tab_bio:
    # View mode toggle
    bio_view_mode = st.radio(
        "View Mode",
        options=["Table", "Timeline"],
        horizontal=True,
        key="data_bio_view_mode",
    )

    st.divider()

    if bio_view_mode == "Timeline":
        # Timeline view
        st.subheader("Chart Settings")
        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            bio_resolution = st.selectbox(
                "Time Resolution (min)",
                options=[5, 15, 30, 60],
                key="data_bio_timeline_resolution",
            )

        with settings_col2:
            bio_show_context = st.checkbox(
                "Show Context Shading",
                key="data_bio_show_context",
            )

        # Load timeline data
        bio_timeline_df = load_timeline_biomarkers(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
            resolution_minutes=bio_resolution,
        )

        context_periods = None
        if bio_show_context:
            context_periods = load_context_periods(
                user_id=user_id,
                start=start_datetime,
                end=end_datetime,
            )

        # Metric selection
        st.subheader("Metric Selection")
        available_biomarkers = (
            sorted(bio_timeline_df["name"].unique().tolist())
            if not bio_timeline_df.empty
            else []
        )
        current_selection = st.session_state.get("data_bio_timeline_select", [])
        st.session_state["data_bio_timeline_select"] = [
            b for b in current_selection if b in available_biomarkers
        ]
        selected_biomarkers = st.multiselect(
            "Biomarkers",
            options=available_biomarkers,
            help="Select biomarkers to display. Leave empty for all.",
            key="data_bio_timeline_select",
        )

        st.divider()

        # Render chart
        if bio_timeline_df.empty:
            st.info("No biomarker data found for the selected filters.")
        else:
            st.subheader("Biomarker Timeline")
            fig = render_biomarker_timeline_chart(
                df=bio_timeline_df,
                selected_names=selected_biomarkers or None,
                show_threshold=False,
            )
            if bio_show_context and context_periods is not None:
                fig = add_context_shading(fig, context_periods)
            st.plotly_chart(fig, use_container_width=True)

            # Data summary
            with st.expander("Data Summary", expanded=False):
                st.write(f"- Records: {len(bio_timeline_df):,}")
                st.write(f"- Unique metrics: {bio_timeline_df['name'].nunique()}")
                st.write(
                    f"- Time range: {bio_timeline_df['timestamp'].min()} to {bio_timeline_df['timestamp'].max()}"
                )

            # CSV Export
            st.divider()
            csv_bio = bio_timeline_df.to_csv(index=False)
            st.download_button(
                label="Export Timeline CSV",
                data=csv_bio,
                file_name=generate_timeline_csv_filename(
                    user_id, start_datetime, end_datetime, "biomarkers"
                ),
                mime="text/csv",
                key="bio_timeline_export",
            )

    else:
        # Table view
        st.subheader("Filters")

        col1, col2 = st.columns(2)

        with col1:
            type_options = ["speech", "network"]
            selected_bio_types = st.multiselect(
                "Biomarker Type",
                options=type_options,
                help="Filter by biomarker type. Leave empty to show all types.",
                key="data_bio_types_select",
            )

        # Load data with type filter applied
        bio_df = load_biomarkers(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
            biomarker_types=selected_bio_types if selected_bio_types else None,
        )

        with col2:
            available_names = (
                sorted(bio_df["name"].unique().tolist()) if not bio_df.empty else []
            )
            current_selection = st.session_state.get("data_bio_names_select", [])
            st.session_state["data_bio_names_select"] = [
                n for n in current_selection if n in available_names
            ]
            selected_bio_names = st.multiselect(
                "Biomarker Name",
                options=available_names,
                help="Filter by biomarker name. Leave empty to show all names.",
                key="data_bio_names_select",
            )

        # Apply name filter
        filtered_bio_df = bio_filter_by_names(bio_df, selected_bio_names)

        # Reset page when filters change
        bio_filter_hash = hash(
            (
                user_id,
                str(start_datetime),
                str(end_datetime),
                tuple(selected_bio_types),
                tuple(selected_bio_names),
            )
        )
        if (
            "data_bio_prev_filter_hash" not in st.session_state
            or st.session_state["data_bio_prev_filter_hash"] != bio_filter_hash
        ):
            st.session_state["data_bio_current_page"] = 1
            st.session_state["data_bio_prev_filter_hash"] = bio_filter_hash

        st.divider()

        st.subheader("Biomarker Data")

        if filtered_bio_df.empty:
            st.info("No biomarker data found for the selected filters.")
        else:
            # Pagination controls
            total_bio_rows = len(filtered_bio_df)
            page_sizes = [25, 50, 100, 250]

            pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 2, 1])

            with pag_col1:
                bio_page_size = st.selectbox(
                    "Rows per page",
                    options=page_sizes,
                    index=page_sizes.index(st.session_state["data_bio_page_size"]),
                    key="data_bio_page_size_select",
                )
                st.session_state["data_bio_page_size"] = bio_page_size

            bio_total_pgs = bio_total_pages(total_bio_rows, bio_page_size)
            bio_current_page = min(
                st.session_state.get("data_bio_current_page", 1), bio_total_pgs
            )

            with pag_col2:
                st.write("")
                st.write(f"Page {bio_current_page} of {bio_total_pgs}")

            with pag_col3:
                st.write("")
                st.write(f"Total: {total_bio_rows} rows")

            with pag_col4:
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    if st.button(
                        "Prev", disabled=bio_current_page <= 1, key="bio_prev_page"
                    ):
                        st.session_state["data_bio_current_page"] = bio_current_page - 1
                        st.rerun()
                with nav_col2:
                    if st.button(
                        "Next",
                        disabled=bio_current_page >= bio_total_pgs,
                        key="bio_next_page",
                    ):
                        st.session_state["data_bio_current_page"] = bio_current_page + 1
                        st.rerun()

            # Slice data for current page
            start_idx, end_idx = bio_page_indices(
                bio_current_page, bio_page_size, total_bio_rows
            )
            bio_page_df = filtered_bio_df.iloc[start_idx:end_idx]

            # Data table
            column_config = {
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss",
                    width="medium",
                ),
                "type": st.column_config.TextColumn(
                    "Type",
                    width="small",
                ),
                "name": st.column_config.TextColumn(
                    "Biomarker Name",
                    width="medium",
                ),
                "value": st.column_config.NumberColumn(
                    "Value",
                    format="%.4f",
                    width="small",
                ),
                "source": st.column_config.TextColumn(
                    "Source",
                    width="small",
                ),
            }

            st.dataframe(
                bio_page_df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # Export and statistics
            export_col, stats_col = st.columns([1, 2])

            with export_col:
                csv_data = filtered_bio_df.to_csv(index=False)
                filename = bio_csv_filename(user_id, start_datetime, end_datetime)

                st.download_button(
                    label="Export to CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    key="bio_table_export",
                )

            with stats_col:
                with st.expander("Summary Statistics", expanded=False):
                    stats_df = bio_summary_stats(filtered_bio_df)
                    if not stats_df.empty:
                        st.dataframe(
                            stats_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No statistics available for current selection.")


# ==============================================================================
# INDICATORS TAB
# ==============================================================================

with tab_ind:
    # View mode toggle
    ind_view_mode = st.radio(
        "View Mode",
        options=["Table", "Timeline"],
        horizontal=True,
        key="data_ind_view_mode",
    )

    st.divider()

    if ind_view_mode == "Timeline":
        # Timeline view
        st.subheader("Chart Settings")
        settings_col1, settings_col2, settings_col3 = st.columns(3)

        with settings_col1:
            ind_resolution = st.selectbox(
                "Time Resolution (min)",
                options=[5, 15, 30, 60],
                key="data_ind_timeline_resolution",
            )

        with settings_col2:
            ind_threshold = st.slider(
                "DSM Threshold",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="data_ind_dsm_threshold",
            )

        with settings_col3:
            ind_show_context = st.checkbox(
                "Show Context Shading",
                key="data_ind_show_context",
            )

        # Load timeline data
        ind_timeline_df = load_timeline_indicators(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
            resolution_minutes=ind_resolution,
        )

        ind_context_periods = None
        if ind_show_context:
            ind_context_periods = load_context_periods(
                user_id=user_id,
                start=start_datetime,
                end=end_datetime,
            )

        # Metric selection
        st.subheader("Metric Selection")
        available_indicators = (
            sorted(ind_timeline_df["indicator_type"].unique().tolist())
            if not ind_timeline_df.empty
            else []
        )
        current_ind_selection = st.session_state.get("data_ind_timeline_select", [])
        st.session_state["data_ind_timeline_select"] = [
            i for i in current_ind_selection if i in available_indicators
        ]
        selected_indicators = st.multiselect(
            "Indicators",
            options=available_indicators,
            help="Select indicators to display. Leave empty for all.",
            key="data_ind_timeline_select",
        )

        st.divider()

        # Render chart
        if ind_timeline_df.empty:
            st.info("No indicator data found for the selected filters.")
        else:
            st.subheader("Indicator Timeline")
            fig = render_indicator_timeline_chart(
                df=ind_timeline_df,
                selected_types=selected_indicators or None,
                threshold_value=ind_threshold,
            )
            if ind_show_context and ind_context_periods is not None:
                fig = add_context_shading(fig, ind_context_periods)
            st.plotly_chart(fig, use_container_width=True)

            # Data summary
            with st.expander("Data Summary", expanded=False):
                st.write(f"- Records: {len(ind_timeline_df):,}")
                st.write(f"- Unique types: {ind_timeline_df['indicator_type'].nunique()}")
                st.write(
                    f"- Time range: {ind_timeline_df['timestamp'].min()} to {ind_timeline_df['timestamp'].max()}"
                )

            # CSV Export
            st.divider()
            csv_ind = ind_timeline_df.to_csv(index=False)
            st.download_button(
                label="Export Timeline CSV",
                data=csv_ind,
                file_name=generate_timeline_csv_filename(
                    user_id, start_datetime, end_datetime, "indicators"
                ),
                mime="text/csv",
                key="ind_timeline_export",
            )

    else:
        # Table view
        st.subheader("Filters")
        col1, col2 = st.columns(2)

        # Load data first to get available types
        initial_ind_df = load_indicators(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
        )

        with col1:
            available_ind_types = (
                sorted(initial_ind_df["indicator_type"].unique().tolist())
                if not initial_ind_df.empty
                else []
            )
            current_ind_type_selection = st.session_state.get(
                "data_ind_types_select", []
            )
            st.session_state["data_ind_types_select"] = [
                t for t in current_ind_type_selection if t in available_ind_types
            ]
            selected_ind_types = st.multiselect(
                "Indicator Type",
                options=available_ind_types,
                help="Filter by indicator type. Leave empty to show all.",
                key="data_ind_types_select",
            )

        with col2:
            presence_options = {
                "All": "all",
                "Present Only": "present",
                "Absent Only": "absent",
            }
            presence_selection = st.radio(
                "Presence Flag",
                options=list(presence_options.keys()),
                horizontal=True,
                key="data_ind_presence_radio",
            )
            presence_filter = presence_options[presence_selection]

        # Reload with filters applied
        filtered_ind_df = load_indicators(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
            indicator_types=selected_ind_types if selected_ind_types else None,
            presence_filter=presence_filter,
        )

        # Reset page when filters change
        ind_filter_hash = hash(
            (
                user_id,
                str(start_datetime),
                str(end_datetime),
                tuple(selected_ind_types),
                presence_filter,
            )
        )
        if (
            "data_ind_prev_filter_hash" not in st.session_state
            or st.session_state["data_ind_prev_filter_hash"] != ind_filter_hash
        ):
            st.session_state["data_ind_current_page"] = 1
            st.session_state["data_ind_prev_filter_hash"] = ind_filter_hash

        st.divider()

        st.subheader("Indicator Data")

        if filtered_ind_df.empty:
            st.info("No indicator data found for the selected filters.")
        else:
            # Pagination controls
            total_ind_rows = len(filtered_ind_df)
            page_sizes = [25, 50, 100, 250]

            pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 2, 1])

            with pag_col1:
                ind_page_size = st.selectbox(
                    "Rows per page",
                    options=page_sizes,
                    index=page_sizes.index(st.session_state["data_ind_page_size"]),
                    key="data_ind_page_size_select",
                )
                st.session_state["data_ind_page_size"] = ind_page_size

            ind_total_pgs = ind_total_pages(total_ind_rows, ind_page_size)
            ind_current_page = min(
                st.session_state.get("data_ind_current_page", 1), ind_total_pgs
            )

            with pag_col2:
                st.write("")
                st.write(f"Page {ind_current_page} of {ind_total_pgs}")

            with pag_col3:
                st.write("")
                st.write(f"Total: {total_ind_rows} rows")

            with pag_col4:
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    if st.button(
                        "Prev", disabled=ind_current_page <= 1, key="ind_prev_page"
                    ):
                        st.session_state["data_ind_current_page"] = ind_current_page - 1
                        st.rerun()
                with nav_col2:
                    if st.button(
                        "Next",
                        disabled=ind_current_page >= ind_total_pgs,
                        key="ind_next_page",
                    ):
                        st.session_state["data_ind_current_page"] = ind_current_page + 1
                        st.rerun()

            # Slice data for current page
            start_idx, end_idx = ind_page_indices(
                ind_current_page, ind_page_size, total_ind_rows
            )
            ind_page_df = filtered_ind_df.iloc[start_idx:end_idx].copy()

            # Row styling for presence flag
            def style_presence_rows(row):
                if row["presence_flag"] is True:
                    return ["background-color: #d4edda"] * len(row)
                else:
                    return ["background-color: #f8f9fa"] * len(row)

            # Column configuration
            ind_column_config = {
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss",
                    width="medium",
                ),
                "indicator_type": st.column_config.TextColumn(
                    "Indicator",
                    width="medium",
                ),
                "likelihood": st.column_config.NumberColumn(
                    "Likelihood",
                    format="%.4f",
                    width="small",
                ),
                "presence_flag": st.column_config.CheckboxColumn(
                    "Present",
                    help="DSM-gate presence flag",
                    width="small",
                ),
                "data_reliability_score": st.column_config.NumberColumn(
                    "Data Reliability",
                    format="%.3f",
                    width="small",
                ),
                "context_used": st.column_config.TextColumn(
                    "Context",
                    width="small",
                ),
                "analysis_run_id": st.column_config.TextColumn(
                    "Analysis Run ID",
                    help="Click to copy full ID",
                    width="medium",
                ),
            }

            # Apply styling and display
            styled_df = ind_page_df.style.apply(style_presence_rows, axis=1)
            st.dataframe(
                styled_df,
                column_config=ind_column_config,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # Export and statistics
            export_col, stats_col = st.columns([1, 2])

            with export_col:
                csv_data = filtered_ind_df.to_csv(index=False)
                filename = ind_csv_filename(user_id, start_datetime, end_datetime)

                st.download_button(
                    label="Export to CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    key="ind_table_export",
                )

            with stats_col:
                with st.expander("Summary Statistics", expanded=False):
                    stats_df = ind_summary_stats(filtered_ind_df)
                    if not stats_df.empty:
                        st.dataframe(
                            stats_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No statistics available for current selection.")


# ==============================================================================
# CONTEXT MARKERS TAB
# ==============================================================================

with tab_ctx:
    ctx_subtab1, ctx_subtab2 = st.tabs(["Evaluated Contexts", "Raw Context Markers"])

    # Evaluated Contexts subtab (stored context history)
    with ctx_subtab1:
        # Load context history from database
        eval_df = load_context_history_records(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
        )

        st.divider()

        if eval_df.empty:
            st.info("No evaluation data available for the selected filters.")
        else:
            # Pagination controls
            total_eval_rows = len(eval_df)
            page_sizes = [25, 50, 100, 250]

            pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 2, 1])

            with pag_col1:
                eval_page_size = st.selectbox(
                    "Rows per page",
                    options=page_sizes,
                    index=page_sizes.index(st.session_state["data_ctx_eval_page_size"]),
                    key="data_ctx_eval_page_size_select",
                )
                st.session_state["data_ctx_eval_page_size"] = eval_page_size

            eval_total_pgs = eval_total_pages(total_eval_rows, eval_page_size)
            eval_current_page = min(
                st.session_state.get("data_ctx_eval_page", 1), eval_total_pgs
            )

            with pag_col2:
                st.write("")
                st.write(f"Page {eval_current_page} of {eval_total_pgs}")

            with pag_col3:
                st.write("")
                st.write(f"Total: {total_eval_rows} rows")

            with pag_col4:
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    if st.button(
                        "Prev", disabled=eval_current_page <= 1, key="ctx_eval_prev"
                    ):
                        st.session_state["data_ctx_eval_page"] = eval_current_page - 1
                        st.rerun()
                with nav_col2:
                    if st.button(
                        "Next",
                        disabled=eval_current_page >= eval_total_pgs,
                        key="ctx_eval_next",
                    ):
                        st.session_state["data_ctx_eval_page"] = eval_current_page + 1
                        st.rerun()

            # Slice data for current page
            start_idx, end_idx = eval_page_indices(
                eval_current_page, eval_page_size, total_eval_rows
            )
            eval_page_df = eval_df.iloc[start_idx:end_idx]

            # Prepare display DataFrame - extract scores from context_state dict
            display_df = eval_page_df[["evaluated_at", "dominant_context", "confidence", "context_state", "evaluation_trigger"]].copy()

            # Extract individual context scores from context_state dict
            display_df["solitary_digital_score"] = display_df["context_state"].apply(
                lambda x: x.get("solitary_digital", 0.0) if isinstance(x, dict) else 0.0
            )
            display_df["neutral_score"] = display_df["context_state"].apply(
                lambda x: x.get("neutral", 0.0) if isinstance(x, dict) else 0.0
            )

            # Select columns for display
            display_df = display_df[[
                "evaluated_at",
                "dominant_context",
                "confidence",
                "solitary_digital_score",
                "neutral_score",
                "evaluation_trigger",
            ]]

            eval_column_config = {
                "evaluated_at": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss",
                    width="medium",
                ),
                "dominant_context": st.column_config.TextColumn(
                    "Dominant Context",
                    width="medium",
                ),
                "confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.4f",
                    width="small",
                ),
                "solitary_digital_score": st.column_config.NumberColumn(
                    "Solitary Score",
                    format="%.4f",
                    width="small",
                ),
                "neutral_score": st.column_config.NumberColumn(
                    "Neutral Score",
                    format="%.4f",
                    width="small",
                ),
                "evaluation_trigger": st.column_config.TextColumn(
                    "Trigger",
                    width="small",
                ),
            }

            st.dataframe(
                display_df,
                column_config=eval_column_config,
                use_container_width=True,
                hide_index=True,
            )

            # Export
            csv_data = eval_df.to_csv(index=False)
            filename = generate_evaluation_csv_filename(
                user_id, start_datetime, end_datetime
            )

            st.download_button(
                label="Export Evaluated Contexts to CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="ctx_eval_export",
            )

    # Raw Context Markers subtab
    with ctx_subtab2:
        # Marker-specific filters
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            type_options = ["environment"]
            selected_ctx_types = st.multiselect(
                "Context Type",
                options=type_options,
                default=st.session_state["data_ctx_type_filter"],
                help="Filter by context type. Leave empty to show all types.",
                key="data_ctx_type_filter_select",
            )
            st.session_state["data_ctx_type_filter"] = selected_ctx_types

        # Load raw markers data
        markers_df = load_context_markers(
            user_id=user_id,
            start=start_datetime,
            end=end_datetime,
            context_types=selected_ctx_types if selected_ctx_types else None,
        )

        with filter_col2:
            available_marker_names = (
                sorted(markers_df["name"].unique().tolist())
                if not markers_df.empty
                else []
            )
            current_name_selection = [
                n
                for n in st.session_state.get("data_ctx_name_filter", [])
                if n in available_marker_names
            ]
            selected_marker_names = st.multiselect(
                "Marker Name",
                options=available_marker_names,
                default=current_name_selection,
                help="Filter by marker name. Leave empty to show all.",
                key="data_ctx_name_filter_select",
            )
            st.session_state["data_ctx_name_filter"] = selected_marker_names

        # Apply name filter
        filtered_markers_df = context_filter_by_names(markers_df, selected_marker_names)

        st.divider()

        if filtered_markers_df.empty:
            st.info("No context markers found for the selected filters.")
        else:
            # Sparkline section
            with st.expander("Context Timeline", expanded=False):
                marker_names = sorted(filtered_markers_df["name"].unique())

                if len(marker_names) > 0:
                    for marker in marker_names:
                        marker_data = filtered_markers_df[
                            filtered_markers_df["name"] == marker
                        ].sort_values("timestamp")
                        if len(marker_data) > 1:
                            st.caption(f"**{marker}**")
                            st.line_chart(
                                marker_data.set_index("timestamp")["value"],
                                height=100,
                                use_container_width=True,
                            )
                        elif len(marker_data) == 1:
                            st.caption(
                                f"**{marker}**: {marker_data.iloc[0]['value']:.4f} (single point)"
                            )
                else:
                    st.info("No timeline data available for selected markers.")

            # Pagination controls
            total_marker_rows = len(filtered_markers_df)
            page_sizes = [25, 50, 100, 250]

            pag_col1, pag_col2, pag_col3, pag_col4 = st.columns([1, 1, 2, 1])

            with pag_col1:
                marker_page_size = st.selectbox(
                    "Rows per page",
                    options=page_sizes,
                    index=page_sizes.index(st.session_state["data_ctx_raw_page_size"]),
                    key="data_ctx_raw_page_size_select",
                )
                st.session_state["data_ctx_raw_page_size"] = marker_page_size

            marker_total_pgs = context_total_pages(total_marker_rows, marker_page_size)
            marker_current_page = min(
                st.session_state.get("data_ctx_raw_page", 1), marker_total_pgs
            )

            with pag_col2:
                st.write("")
                st.write(f"Page {marker_current_page} of {marker_total_pgs}")

            with pag_col3:
                st.write("")
                st.write(f"Total: {total_marker_rows} rows")

            with pag_col4:
                nav_col1, nav_col2 = st.columns(2)
                with nav_col1:
                    if st.button(
                        "Prev", disabled=marker_current_page <= 1, key="ctx_raw_prev"
                    ):
                        st.session_state["data_ctx_raw_page"] = marker_current_page - 1
                        st.rerun()
                with nav_col2:
                    if st.button(
                        "Next",
                        disabled=marker_current_page >= marker_total_pgs,
                        key="ctx_raw_next",
                    ):
                        st.session_state["data_ctx_raw_page"] = marker_current_page + 1
                        st.rerun()

            # Slice data for current page
            start_idx, end_idx = context_page_indices(
                marker_current_page, marker_page_size, total_marker_rows
            )
            marker_page_df = filtered_markers_df.iloc[start_idx:end_idx]

            # Data table
            marker_column_config = {
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss",
                    width="medium",
                ),
                "type": st.column_config.TextColumn(
                    "Context Type",
                    width="small",
                ),
                "name": st.column_config.TextColumn(
                    "Marker Name",
                    width="medium",
                ),
                "value": st.column_config.NumberColumn(
                    "Value",
                    format="%.4f",
                    width="small",
                ),
                "source": st.column_config.TextColumn(
                    "Source",
                    width="small",
                ),
            }

            st.dataframe(
                marker_page_df,
                column_config=marker_column_config,
                use_container_width=True,
                hide_index=True,
            )

            # Export and statistics
            export_col, stats_col = st.columns([1, 2])

            with export_col:
                marker_csv = filtered_markers_df.to_csv(index=False)
                marker_filename = context_csv_filename(
                    user_id, start_datetime, end_datetime
                )

                st.download_button(
                    label="Export Markers to CSV",
                    data=marker_csv,
                    file_name=marker_filename,
                    mime="text/csv",
                    key="ctx_markers_export",
                )

            with stats_col:
                with st.expander("Summary Statistics", expanded=False):
                    stats_df = context_summary_stats(filtered_markers_df)
                    if not stats_df.empty:
                        st.dataframe(
                            stats_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No statistics available for current selection.")


render_footer()
