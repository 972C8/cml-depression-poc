"""Analysis page - trigger analysis and view results."""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

from src.dashboard.actions.analysis import trigger_analysis  # noqa: E402
from src.dashboard.components.analysis_pipeline_transparency import (  # noqa: E402
    render_analysis_pipeline_transparency,
)
from src.dashboard.components.baseline_selector import (  # noqa: E402
    render_baseline_selector,
)

# comparison.py functions deprecated per Story 6.12 - now using side-by-side results
from src.dashboard.components.config_viewer import render_config_viewer  # noqa: E402
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
from src.dashboard.components.pipeline_viewer import (  # noqa: E402
    render_config_snapshot,
)
from src.dashboard.components.results_summary import (  # noqa: E402
    render_results_overview,
)
from src.dashboard.data.config import (  # noqa: E402
    config_to_yaml,
    get_current_config,
    reload_config,
)
from src.dashboard.data.context_evaluation import (  # noqa: E402
    load_context_history_records,
)
from src.dashboard.data.context_runs import (  # noqa: E402
    check_context_run_coverage,
    list_context_evaluation_runs,
)
import plotly.graph_objects as go  # noqa: E402

# Context colors for charts
CONTEXT_COLORS = {
    "solitary_digital": "#2ca02c",  # Green
    "adversarial_social_digital_gaming": "#d62728",  # Red
    "neutral": "#7f7f7f",  # Gray
}
from src.dashboard.data.experiments import (  # noqa: E402
    get_experiment_config,
    list_experiments,
)
from src.dashboard.data.pipeline import (  # noqa: E402
    get_trace_for_run,
    load_user_analysis_runs,
)

# Initialize session state for page-specific data
if "analysis_last_result" not in st.session_state:
    st.session_state["analysis_last_result"] = None
if "analysis_running" not in st.session_state:
    st.session_state["analysis_running"] = False
if "results_selected_run_id" not in st.session_state:
    st.session_state["results_selected_run_id"] = None
if "comparison_selected_run_id" not in st.session_state:
    st.session_state["comparison_selected_run_id"] = None
# Track previous primary run to detect changes
if "_previous_primary_run_id" not in st.session_state:
    st.session_state["_previous_primary_run_id"] = None

# Reset selections when entering this page (detect page change)
if st.session_state.get("_current_page") != "analysis":
    st.session_state["results_selected_run_id"] = None
    st.session_state["comparison_selected_run_id"] = None
    st.session_state["analysis_last_result"] = None
    # Reset baseline to default on page entry
    st.session_state.pop("analysis_baseline_selected_value", None)
    st.session_state["_current_page"] = "analysis"

# Initialize filter session state
init_filter_session_state()

# Render user selector in sidebar
user_id = render_user_sidebar()

# System status at end of sidebar
render_sidebar_status()

# Page content
render_page_header("Analysis", "⚙️", "Trigger analysis and view results")

if user_id is None:
    st.warning("Please select a user from the sidebar to run analysis.")
    render_footer()
    st.stop()

# ==============================================================================
# ANALYSIS TRIGGER SECTION
# ==============================================================================

# Load available experiments for selector (needed for both display and logic)
experiments_df = list_experiments(limit=20)
experiment_options = {"default": "Default Config"}
if not experiments_df.empty:
    for _, row in experiments_df.iterrows():
        experiment_options[row["id"]] = f"{row['name']}"

# Step 1: Analysis Period
st.markdown(
    """
    <div style="
        padding: 0.75rem 1rem;
        background: #e8f4f8;
        border-radius: 0.5rem;
        border-left: 3px solid #17a2b8;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    ">
        <div style="
            width: 28px;
            height: 28px;
            background: #17a2b8;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        ">1</div>
        <div>
            <div style="font-size: 0.95rem; font-weight: 500; color: #1a1a2e;">
                Select 14-Day Analysis Period
            </div>
            <div style="font-size: 0.8rem; color: #666;">
                Select the date range to analyze
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
start_datetime, end_datetime = render_inline_date_range(key_prefix="analysis_date", presets=["14d"])

st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

# Step 2: Configuration
st.markdown(
    """
    <div style="
        padding: 0.75rem 1rem;
        background: #fff3e6;
        border-radius: 0.5rem;
        border-left: 3px solid #fd7e14;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    ">
        <div style="
            width: 28px;
            height: 28px;
            background: #fd7e14;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        ">2</div>
        <div>
            <div style="font-size: 0.95rem; font-weight: 500; color: #1a1a2e;">
                Load Analysis Parameters
            </div>
            <div style="font-size: 0.8rem; color: #666;">
                Create Experiments to try different parameters
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
selected_exp_key = st.selectbox(
    "Select experiment",
    options=list(experiment_options.keys()),
    format_func=lambda x: experiment_options[x],
    key="config_selector",
    label_visibility="collapsed",
)


# Configuration details toggle (below the columns)
if st.toggle("View configuration details", value=False, key="show_config_details"):
    if selected_exp_key == "default":
        config = get_current_config()
        config_source = "Default configuration from config/*.yaml"
    else:
        config = get_experiment_config(selected_exp_key)
        config_source = f"Experiment: {experiment_options[selected_exp_key]}"

    col_src, col_btn = st.columns([3, 1])
    with col_src:
        st.caption(config_source)
    with col_btn:
        if selected_exp_key == "default":
            if st.button("Reload", use_container_width=True):
                config = reload_config()
                st.success("Configuration reloaded!")
                st.rerun()

    if config:
        render_config_viewer(config)
        if st.toggle("Show raw YAML", value=False, key="show_raw_yaml"):
            st.code(config_to_yaml(config), language="yaml")
    else:
        st.warning("Could not load experiment configuration")

st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

# Step 3: Context Evaluation
st.markdown(
    """
    <div style="
        padding: 0.75rem 1rem;
        background: #e6f3ff;
        border-radius: 0.5rem;
        border-left: 3px solid #0d6efd;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    ">
        <div style="
            width: 28px;
            height: 28px;
            background: #0d6efd;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        ">3</div>
        <div>
            <div style="font-size: 0.95rem; font-weight: 500; color: #1a1a2e;">
                Select Context Evaluation
            </div>
            <div style="font-size: 0.8rem; color: #666;">
                Choose which context evaluation run to use for context-aware analysis
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load context evaluation runs for the user
context_runs_df = list_context_evaluation_runs(user_id=user_id, limit=20)

if context_runs_df.empty:
    st.warning(
        "**Context evaluation required.** No context runs found. "
        "Go to the **Context** page to create a context evaluation run before running analysis."
    )
    selected_context_run_id = None
else:
    # Build options for dropdown (Story 6.14 AC6: removed "None" option)
    from src.dashboard.components.filters import get_display_timezone

    display_tz = get_display_timezone()
    context_run_options = []

    for _, run in context_runs_df.iterrows():
        # Convert timestamp to display timezone
        created_ts = run["created_at"]
        if hasattr(created_ts, "astimezone"):
            if created_ts.tzinfo is None:
                from datetime import UTC

                created_ts = created_ts.replace(tzinfo=UTC)
            created_ts = created_ts.astimezone(display_tz)
        run_label = (
            f"{run['id'][:8]}... | {created_ts.strftime('%Y-%m-%d %H:%M:%S')} "
            f"({run['evaluation_count']} evaluations)"
        )
        context_run_options.append((run["id"], run_label))

    selected_context_run_key = st.selectbox(
        "Select context evaluation run",
        options=[opt[0] for opt in context_run_options],
        format_func=lambda x: next(
            opt[1] for opt in context_run_options if opt[0] == x
        ),
        index=0,  # Default to most recent run
        key="context_run_selector",
        label_visibility="collapsed",
        help="Select which context evaluation run to use for context-aware weighting",
    )
    selected_context_run_id = selected_context_run_key

    # Show evaluation details toggle
    if selected_context_run_id and st.toggle(
        "Show evaluation details", value=False, key="show_context_eval_details"
    ):
        # Find the selected run's details
        selected_run = context_runs_df[
            context_runs_df["id"] == selected_context_run_id
        ].iloc[0]

        st.caption(f"{selected_run['evaluation_count']} evaluations in this run")

        # Load history for the selected run and render confidence chart
        history_df = load_context_history_records(
            user_id=user_id,
            context_evaluation_run_id=selected_context_run_id,
        )

        if not history_df.empty:
            # Render confidence scores over time chart
            fig = go.Figure()

            for context in CONTEXT_COLORS.keys():
                scores = []
                timestamps = []
                for _, row in history_df.iterrows():
                    ctx_state = row.get("context_state", {})
                    if ctx_state and context in ctx_state:
                        scores.append(ctx_state[context])
                        timestamps.append(row["evaluated_at"])

                if scores:
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=scores,
                            mode="lines",
                            name=context.replace("_", " ").title(),
                            line={"color": CONTEXT_COLORS[context], "width": 2},
                            hovertemplate=f"{context}<br>Score: %{{y:.3f}}<br>Time: %{{x}}<extra></extra>",
                        )
                    )

            fig.update_layout(
                title="Context Confidence Scores Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence Score",
                yaxis={"range": [0, 1.05]},
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1,
                },
                hovermode="x unified",
                height=300,
                margin={"t": 40, "b": 40, "l": 40, "r": 20},
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No evaluation data available for this run.")

        # Show config snapshot if available
        config_snapshot = selected_run.get("config_snapshot")
        if config_snapshot:
            with st.expander("Configuration used", expanded=False):
                st.json(config_snapshot)

    # Story 6.14 AC4: Check coverage and show warning if partial
    if selected_context_run_id:
        coverage = check_context_run_coverage(
            run_id=selected_context_run_id,
            user_id=user_id,
            start_time=start_datetime,
            end_time=end_datetime,
        )
        # Only warn if there are dates WITH biomarker data that lack context
        if coverage["dates_missing_context"]:
            missing_count = len(coverage["dates_missing_context"])
            coverage_pct = coverage["coverage_ratio"] * 100
            st.warning(
                f"⚠️ **Partial Coverage:** Context run covers "
                f"{coverage['dates_with_context']}/{coverage['dates_with_data']} dates "
                f"with biomarker data ({coverage_pct:.0f}%). "
                f"{missing_count} date(s) with data but no context: "
                f"{', '.join(coverage['dates_missing_context'][:5])}"
                f"{'...' if missing_count > 5 else ''}. "
                f"\n\n**Missing dates will use neutral context weights "
                f"(no effect on indicator weighting).**"
            )

st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

# Step 4: Baseline
st.markdown(
    """
    <div style="
        padding: 0.75rem 1rem;
        background: #e8f5e9;
        border-radius: 0.5rem;
        border-left: 3px solid #28a745;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    ">
        <div style="
            width: 28px;
            height: 28px;
            background: #28a745;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
        ">4</div>
        <div>
            <div style="font-size: 0.95rem; font-weight: 500; color: #1a1a2e;">
                Load User Baseline
            </div>
            <div style="font-size: 0.8rem; color: #666;">
                The analysis fundamentally looks at the deviation from the user's baseline. To obtain meaningful results (ie. abnormal biomarkers or indicators compared to what is typical for the user), a good baseline of what is 'normal' for the user is essential.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
selected_baseline = render_baseline_selector(key_prefix="analysis_baseline")
st.session_state["selected_baseline"] = selected_baseline

# Run button with prominent styling
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# Story 6.14: Require context evaluation run (AC6 - removed "None" option)
run_disabled = st.session_state.get("analysis_running", False) or selected_context_run_id is None

run_button = st.button(
    "Run Analysis",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
    help="Execute the analysis pipeline with the selected configuration"
    if selected_context_run_id
    else "Create a context evaluation run on the Context page first",
)

# Handle analysis trigger
if run_button:
    # Get selected baseline from session state (Story 4.14)
    baseline_config = st.session_state.get("selected_baseline")

    # Validate baseline is selected (required)
    if baseline_config is None:
        st.error(
            "Please select a baseline before running analysis. "
            "Choose a predefined baseline or upload a custom one."
        )
    else:
        st.session_state["analysis_running"] = True
        st.session_state["analysis_last_result"] = None

        # Determine experiment_id to pass (None for default)
        experiment_id = None if selected_exp_key == "default" else selected_exp_key

        # Story 6.14: Convert selected_context_run_id to UUID if provided
        context_run_uuid = None
        if selected_context_run_id:
            import uuid as uuid_module
            try:
                context_run_uuid = uuid_module.UUID(selected_context_run_id)
            except (ValueError, TypeError):
                st.error(f"Invalid context run ID format: {selected_context_run_id}")
                st.stop()

        with st.status("Running analysis pipeline...", expanded=True) as status:
            def update_progress(step: int, total: int, description: str) -> None:
                status.update(label=f"Step {step}/{total}: {description}")

            result = trigger_analysis(
                user_id=user_id,
                start_time=start_datetime,
                end_time=end_datetime,
                baseline_config=baseline_config,
                experiment_id=experiment_id,
                context_evaluation_run_id=context_run_uuid,
                progress_callback=update_progress,
            )
            st.session_state["analysis_last_result"] = result
            # Set selected run to the latest one
            if result.success:
                st.session_state["results_selected_run_id"] = result.run_id
                status.update(label="Analysis complete!", state="complete")
            else:
                status.update(label="Analysis failed", state="error")

        st.session_state["analysis_running"] = False
        st.rerun()

# Display last result status
last_result = st.session_state.get("analysis_last_result")
if last_result is not None:
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    if last_result.success:
        st.markdown(
            f"""
            <div style="
                padding: 1rem 1.25rem;
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border-radius: 0.5rem;
                border-left: 4px solid #28a745;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            ">
                <span style="font-size: 1.5rem;">✓</span>
                <div>
                    <div style="font-weight: 600; color: #155724;">Analysis completed successfully</div>
                    <div style="font-size: 0.85rem; color: #155724; opacity: 0.8;">
                        Run ID: {last_result.run_id[:8]}...
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                padding: 1rem 1.25rem;
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border-radius: 0.5rem;
                border-left: 4px solid #dc3545;
            ">
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.5rem;">✗</span>
                    <div style="font-weight: 600; color: #721c24;">Analysis failed</div>
                </div>
                <div style="font-size: 0.85rem; color: #721c24; padding-left: 2.25rem;">
                    <div><strong>Error:</strong> {last_result.error_code}</div>
                    <div><strong>Message:</strong> {last_result.error_message}</div>
                    <div><strong>Step:</strong> {last_result.error_step or 'unknown'}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# RESULTS SECTION
# ==============================================================================

st.divider()

# Section header
st.markdown(
    """
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    ">
        <div style="
            font-size: 1.75rem;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">📊</div>
        <div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #1a1a2e;">
                Results
            </div>
            <div style="font-size: 0.85rem; color: #666;">
                View and explore analysis run outputs
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load available runs for this user
runs_df = load_user_analysis_runs(user_id=user_id, limit=20)

if runs_df.empty:
    st.markdown(
        """
        <div style="
            padding: 2rem;
            text-align: center;
            background: #f8f9fa;
            border-radius: 0.5rem;
            border: 1px dashed #dee2e6;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>
            <div style="color: #6c757d;">No analysis runs found. Run an analysis above to see results.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Run selector with label
    selector_col1, selector_col2 = st.columns([1, 3])
    with selector_col1:
        st.markdown(
            "<div style='padding-top: 0.5rem; font-weight: 500; color: #495057;'>Select run:</div>",
            unsafe_allow_html=True,
        )
    with selector_col2:
        # Build options with placeholder
        run_options = [None] + runs_df["run_id"].tolist()

        def format_run_option(x):
            if x is None:
                return "Select a run..."
            return runs_df[runs_df["run_id"] == x]["display_label"].iloc[0]

        # Find index of selected run in session state
        default_index = 0
        current_selected = st.session_state.get("results_selected_run_id")
        if current_selected and current_selected in runs_df["run_id"].values:
            default_index = run_options.index(current_selected)

        selected_run_id = st.selectbox(
            "Run",
            options=run_options,
            format_func=format_run_option,
            index=default_index,
            key="results_run_selector",
            label_visibility="collapsed",
        )

    if selected_run_id:
        # Store selected run in session state
        st.session_state["results_selected_run_id"] = selected_run_id

        # AC2: Reset comparison when primary changes to same run
        previous_primary = st.session_state.get("_previous_primary_run_id")
        if previous_primary != selected_run_id:
            # Primary run changed - reset comparison if it equals new primary
            current_comparison = st.session_state.get("comparison_selected_run_id")
            if current_comparison == selected_run_id:
                st.session_state["comparison_selected_run_id"] = None
            st.session_state["_previous_primary_run_id"] = selected_run_id

        # AC2: Comparison run selector
        comparison_selector_col1, comparison_selector_col2 = st.columns([1, 3])
        with comparison_selector_col1:
            st.markdown(
                "<div style='padding-top: 0.5rem; font-weight: 500; color: #495057;'>Compare to:</div>",
                unsafe_allow_html=True,
            )
        with comparison_selector_col2:
            # Build comparison options - exclude primary run
            comparison_options = [None] + [
                r for r in runs_df["run_id"].tolist() if r != selected_run_id
            ]

            def format_comparison_option(x):
                if x is None:
                    return "None (no comparison)"
                return runs_df[runs_df["run_id"] == x]["display_label"].iloc[0]

            # Find index of selected comparison in session state
            comparison_default_index = 0
            current_comparison = st.session_state.get("comparison_selected_run_id")
            if current_comparison and current_comparison in comparison_options:
                comparison_default_index = comparison_options.index(current_comparison)

            comparison_run_id = st.selectbox(
                "Comparison Run",
                options=comparison_options,
                format_func=format_comparison_option,
                index=comparison_default_index,
                key="comparison_run_selector",
                label_visibility="collapsed",
            )

            # Store comparison run in session state
            st.session_state["comparison_selected_run_id"] = comparison_run_id

        # Load pipeline trace for selected run
        trace, config_snapshot = get_trace_for_run(selected_run_id)

        # Load comparison trace if comparison run selected
        comparison_trace, comparison_config = None, None
        if comparison_run_id:
            comparison_trace, comparison_config = get_trace_for_run(comparison_run_id)
            # AC8: Handle case where comparison run has no trace
            if comparison_trace is None:
                st.warning(
                    f"No pipeline trace available for comparison run "
                    f"({comparison_run_id[:8]}...). Showing single run view."
                )
                # Reset comparison to avoid confusion
                st.session_state["comparison_selected_run_id"] = None
                comparison_run_id = None

        if trace is None:
            st.warning("No pipeline trace available for this run.")
        else:
            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

            # AC3: Side-by-side layout when comparison run is selected
            if comparison_run_id and comparison_trace is not None:
                # Side-by-side mode
                col_primary, col_comparison = st.columns(2)

                with col_primary:
                    # AC7: Visual differentiation - primary column header
                    st.markdown(
                        f"""
                        <div style="
                            padding: 0.75rem 1rem;
                            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                            border-radius: 0.5rem;
                            border-left: 4px solid #28a745;
                            margin-bottom: 1rem;
                        ">
                            <div style="font-weight: 600; color: #155724;">
                                Primary Run: {selected_run_id[:8]}...
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Results Overview
                    st.markdown("#### Results Overview")
                    render_results_overview(
                        trace.to_dict() if hasattr(trace, "to_dict") else None,
                        session_prefix="primary_",
                    )

                    # Pipeline Transparency
                    render_analysis_pipeline_transparency(
                        user_id,
                        session_prefix="primary_",
                        run_id=selected_run_id,
                    )

                    # Configuration Used
                    st.markdown("#### Configuration Used")
                    render_config_snapshot(
                        config_snapshot,
                        label=f"View Config: {selected_run_id[:8]}...",
                    )

                with col_comparison:
                    # AC7: Visual differentiation - comparison column header
                    st.markdown(
                        f"""
                        <div style="
                            padding: 0.75rem 1rem;
                            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                            border-radius: 0.5rem;
                            border-left: 4px solid #1976d2;
                            margin-bottom: 1rem;
                        ">
                            <div style="font-weight: 600; color: #0d47a1;">
                                Comparison Run: {comparison_run_id[:8]}...
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Results Overview
                    st.markdown("#### Results Overview")
                    render_results_overview(
                        comparison_trace.to_dict()
                        if hasattr(comparison_trace, "to_dict")
                        else None,
                        session_prefix="comparison_",
                    )

                    # Pipeline Transparency
                    render_analysis_pipeline_transparency(
                        user_id,
                        session_prefix="comparison_",
                        run_id=comparison_run_id,
                    )

                    # Configuration Used
                    st.markdown("#### Configuration Used")
                    render_config_snapshot(
                        comparison_config,
                        label=f"View Config: {comparison_run_id[:8]}...",
                    )

            else:
                # Single column mode (no comparison)
                # Results Overview subsection
                st.markdown(
                    """
                    <div style="
                        font-size: 1rem;
                        font-weight: 600;
                        color: #1a1a2e;
                        padding: 0.5rem 0;
                        border-bottom: 2px solid #11998e;
                        margin-bottom: 0.75rem;
                        display: inline-block;
                    ">Results Overview</div>
                    """,
                    unsafe_allow_html=True,
                )
                render_results_overview(
                    trace.to_dict() if hasattr(trace, "to_dict") else None,
                )

                # Analysis Pipeline Transparency (Story 6.11)
                render_analysis_pipeline_transparency(user_id)

                # Configuration Used subsection
                st.markdown(
                    """
                    <div style="
                        font-size: 1rem;
                        font-weight: 600;
                        color: #1a1a2e;
                        padding: 0.5rem 0;
                        border-bottom: 2px solid #11998e;
                        margin-bottom: 0.75rem;
                        margin-top: 1rem;
                        display: inline-block;
                    ">Configuration Used</div>
                    """,
                    unsafe_allow_html=True,
                )
                render_config_snapshot(config_snapshot)


render_footer()
