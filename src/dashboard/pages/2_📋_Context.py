"""Unified Context page - view stored context history and historical data.

Story 6.9: Migrated to use ContextHistoryService for read-only access to
stored context evaluations. This page displays what the analysis pipeline
used, not live re-evaluations.

Features:
- Hero section showing current context from stored history
- Pipeline Transparency showing step-by-step evaluation breakdown
- Historical trends charts from context_history table
- Context Timeline Visualization with segments
"""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402, I001
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from src.core.config import EMAConfig  # noqa: E402
from src.core.context.evaluator import (  # noqa: E402
    ContextEvaluationConfig,
    ContextEvaluator,
)
from src.core.context.history import ContextHistoryService  # noqa: E402
from src.dashboard.components.filters import (  # noqa: E402
    get_display_timezone,
    init_filter_session_state,
    render_user_sidebar,
)
from src.dashboard.components.layout import (  # noqa: E402
    render_footer,
    render_page_header,
    render_sidebar_status,
)
from src.dashboard.data.context_evaluation import (  # noqa: E402
    detect_context_transitions,
    load_context_history_records,
)
from src.dashboard.data.context_runs import (  # noqa: E402
    create_context_evaluation_run,
    get_default_context_eval_config,
    list_context_evaluation_runs,
    update_run_evaluation_count,
)
from src.dashboard.data.experiments import (  # noqa: E402
    get_experiment_config,
    list_experiments,
)
from src.shared.database import SessionLocal  # noqa: E402
from src.shared.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# Context colors for consistent visualization
CONTEXT_COLORS = {
    "solitary_digital": "#2ca02c",  # Green
    "adversarial_social_digital_gaming": "#d62728",  # Red
    "neutral": "#7f7f7f",  # Gray
}


# ==============================================================================
# Pipeline Transparency Helper Functions
# ==============================================================================


def generate_membership_ascii(
    func_type: str,
    params: list[float],
    value: float,
    width: int = 40,
) -> str:
    """Generate ASCII visualization of a membership function with value marker.

    Args:
        func_type: "triangular" or "trapezoidal"
        params: Function parameters
        value: Current value to mark on the chart
        width: Width of the ASCII chart

    Returns:
        Multi-line ASCII string
    """
    if func_type == "triangular":
        left, peak, right = params
        x_min, x_max = left, right
    else:  # trapezoidal
        left, left_peak, right_peak, right = params
        x_min, x_max = left, right

    # Extend range slightly for visibility
    x_range = x_max - x_min
    x_min = max(0, x_min - x_range * 0.1)
    x_max = x_max + x_range * 0.1

    # Build the chart
    lines = []
    height = 5

    for row in range(height, -1, -1):
        y = row / height
        line = ""
        for col in range(width):
            x = x_min + (x_max - x_min) * col / (width - 1)

            # Calculate membership at this x
            if func_type == "triangular":
                left, peak, right = params
                if x <= left or x >= right:
                    membership = 0.0
                elif x == peak:
                    membership = 1.0
                elif x < peak:
                    denom = peak - left
                    membership = (x - left) / denom if denom != 0 else 1.0
                else:
                    denom = right - peak
                    membership = (right - x) / denom if denom != 0 else 1.0
            else:  # trapezoidal
                left, left_peak, right_peak, right = params
                if x <= left or x >= right:
                    membership = 0.0
                elif left_peak <= x <= right_peak:
                    membership = 1.0
                elif x < left_peak:
                    denom = left_peak - left
                    membership = (x - left) / denom if denom != 0 else 1.0
                else:
                    denom = right - right_peak
                    membership = (right - x) / denom if denom != 0 else 1.0

            # Check if this point is on the curve
            if abs(membership - y) < 0.15:
                line += "#"
            elif row == 0:
                line += "-"
            else:
                line += " "

        # Add y-axis label
        if row == height:
            lines.append(f"1.0 |{line}")
        elif row == 0:
            lines.append(f"0.0 |{line}")
        else:
            lines.append(f"    |{line}")

    # Add x-axis with value marker
    marker_pos = (
        int((value - x_min) / (x_max - x_min) * (width - 1)) if x_max != x_min else 0
    )
    marker_pos = max(0, min(width - 1, marker_pos))
    x_axis = " " * marker_pos + "^" + " " * (width - marker_pos - 1)
    lines.append(f"    +{x_axis}")
    lines.append(f"      x={value:.2f}")

    return "\n".join(lines)


def get_membership_formula_explanation(
    func_type: str,
    params: list[float],
    value: float,
    result: float,
) -> str:
    """Generate a formula explanation for a membership calculation.

    Args:
        func_type: "triangular" or "trapezoidal"
        params: Function parameters
        value: Input value
        result: Calculated membership degree

    Returns:
        Human-readable formula explanation
    """
    if func_type == "triangular":
        left, peak, right = params
        if value <= left or value >= right:
            return f"Value {value:.2f} outside [{left}, {right}] -> **0.0**"
        elif value == peak:
            return f"Value {value:.2f} at peak ({peak}) -> **1.0**"
        elif value < peak:
            formula = f"({value:.2f} - {left}) / ({peak} - {left})"
            return f"(x - left) / (peak - left) = {formula} = **{result:.3f}**"
        else:
            formula = f"({right} - {value:.2f}) / ({right} - {peak})"
            return f"(right - x) / (right - peak) = {formula} = **{result:.3f}**"
    else:  # trapezoidal
        left, left_peak, right_peak, right = params
        if value <= left or value >= right:
            return f"Value {value:.2f} outside [{left}, {right}] -> **0.0**"
        elif left_peak <= value <= right_peak:
            return (
                f"Value {value:.2f} on plateau [{left_peak}, {right_peak}] -> **1.0**"
            )
        elif value < left_peak:
            formula = f"({value:.2f} - {left}) / ({left_peak} - {left})"
            return f"(x - left) / (lp - left) = {formula} = **{result:.3f}**"
        else:
            formula = f"({right} - {value:.2f}) / ({right} - {right_peak})"
            return f"(right - x) / (right - rp) = {formula} = **{result:.3f}**"


def get_pipeline_transparency_data(
    sensor_snapshot: dict,
    context_state: dict,
    sensors_used: list[str],
    raw_scores_stored: dict | None = None,
) -> dict | None:
    """Build pipeline transparency data from stored context history.

    Uses stored sensor_snapshot to compute memberships for display, and uses
    stored raw_scores and smoothed scores (context_state) for accurate pipeline
    representation.

    Args:
        sensor_snapshot: Stored marker values from context history record
        context_state: Stored EMA-smoothed context scores from context history record
        sensors_used: List of sensors that were used
        raw_scores_stored: Stored raw (pre-smoothing) scores, if available

    Returns:
        Dict with membership calculations and scores for transparency display
    """
    if not sensor_snapshot:
        return None

    try:
        # Load config from YAML file for transparency display
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "context_evaluation.yaml"
        )
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Build evaluator from the actual YAML config (not defaults)
        config = ContextEvaluationConfig(**raw_config)
        evaluator = ContextEvaluator(context_config=config)

        # Compute memberships from stored sensor values (for Step 2 display)
        memberships = evaluator._compute_marker_memberships(sensor_snapshot)

        # Use stored raw scores if available, otherwise recompute for backwards compat
        if raw_scores_stored:
            raw_scores = raw_scores_stored
        else:
            # Legacy records without raw_scores - recompute from memberships
            raw_scores = {}
            for ctx_name, assumption in evaluator._assumptions.items():
                score, _, _ = assumption.evaluate(memberships)
                raw_scores[ctx_name] = score

        # Determine winner from stored context_state (smoothed scores)
        sorted_scores = sorted(context_state.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_scores[0] if sorted_scores else ("neutral", 0.0)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0.0)
        delta = winner[1] - runner_up[1]

        # Load EMA config from YAML (authoritative source)
        ema_path = Path(__file__).parent.parent.parent.parent / "config" / "ema.yaml"
        with open(ema_path) as f:
            ema_data = yaml.safe_load(f)
        ema_config = EMAConfig(**ema_data)

        # Build detailed context assumption info for Step 3
        context_assumptions_detail = {}
        for ctx_name, assumption_cfg in raw_config.get(
            "context_assumptions", {}
        ).items():
            if "conditions" in assumption_cfg:
                context_assumptions_detail[ctx_name] = {
                    "conditions": assumption_cfg["conditions"],
                    "operator": assumption_cfg.get("operator", "WEIGHTED_MEAN"),
                }
            elif "threshold" in assumption_cfg:
                context_assumptions_detail[ctx_name] = {
                    "threshold": assumption_cfg["threshold"],
                }

        # Determine markers missing
        expected_markers = set(raw_config.get("marker_memberships", {}).keys())
        actual_markers = set(sensor_snapshot.keys())
        markers_missing = list(expected_markers - actual_markers)

        return {
            # Step 1: Input markers
            "marker_values": sensor_snapshot,
            # Step 2: Fuzzy membership with config
            "memberships": memberships,
            "membership_config": raw_config.get("marker_memberships", {}),
            # Step 3: Raw context scores with assumption config
            "raw_scores": raw_scores,
            "context_assumptions": context_assumptions_detail,
            # Step 4: EMA-smoothed scores (from stored context_state)
            "smoothed_scores": context_state,
            # Flag indicating if raw scores came from storage or were recomputed
            "raw_scores_from_storage": raw_scores_stored is not None,
            # General info
            "markers_used": sensors_used,
            "markers_missing": markers_missing,
            "winner": winner,
            "runner_up": runner_up,
            "delta": delta,
            # Configuration reference
            "config": {
                "ema_alpha": ema_config.alpha,
                "hysteresis_threshold": ema_config.hysteresis,
                "dwell_time": ema_config.dwell_time,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline transparency data: {e}", exc_info=True)
        return None


# ==============================================================================
# Session State Initialization
# ==============================================================================

# Initialize filter session state
init_filter_session_state()

# Track current page for cross-page state management
st.session_state["_current_page"] = "context"

# Render user selector in sidebar (no date filter)
user_id = render_user_sidebar()

# System status at end of sidebar
render_sidebar_status()

# Page content
render_page_header(
    "Context",
    "📊",
    "Configure and run context evaluation",
)

if user_id is None:
    st.warning("Please select a user from the sidebar to view context data.")
    render_footer()
    st.stop()

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
        ">1</div>
        <div>
            <div style="font-size: 0.95rem; font-weight: 500; color: #1a1a2e;">
                Load Evaluation Parameters
            </div>
            <div style="font-size: 0.8rem; color: #666;">
                Create Experiments to try different parameters 
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Context Evaluation Settings (Story 6.13)
# ==============================================================================

# Initialize session state with namespaced keys
if "ctx_selected_experiment_id" not in st.session_state:
    st.session_state["ctx_selected_experiment_id"] = None
if "ctx_selected_run_id" not in st.session_state:
    st.session_state["ctx_selected_run_id"] = None

# --- Experiment Selection ---
experiments_df = list_experiments(limit=20)
experiment_options = [("default", "Default Configuration (from YAML files)")]

if not experiments_df.empty:
    for _, exp in experiments_df.iterrows():
        exp_label = f"{exp['name']} ({exp['created_at'].strftime('%Y-%m-%d %H:%M')})"
        experiment_options.append((exp["id"], exp_label))

selected_exp_key = st.selectbox(
    "Configuration Source",
    options=[opt[0] for opt in experiment_options],
    format_func=lambda x: next(opt[1] for opt in experiment_options if opt[0] == x),
    index=0,
    key="ctx_experiment_selector",
    help="Select an experiment configuration or use defaults",
)

# Get the actual config based on selection
if selected_exp_key == "default":
    selected_context_eval_config = get_default_context_eval_config()
    config_source_label = "Default YAML"
else:
    exp_config = get_experiment_config(selected_exp_key)
    if exp_config:
        selected_context_eval_config = exp_config.context_evaluation
        config_source_label = f"Experiment: {next(opt[1] for opt in experiment_options if opt[0] == selected_exp_key)}"
    else:
        st.warning("Failed to load experiment config, using defaults")
        selected_context_eval_config = get_default_context_eval_config()
        config_source_label = "Default YAML (fallback)"

# View configuration details toggle
if st.toggle("View configuration details", value=False, key="ctx_config_details_toggle"):
    tab_ema, tab_markers, tab_assumptions = st.tabs(
        ["EMA Settings", "Marker Memberships", "Context Assumptions"]
    )

    with tab_ema:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alpha", f"{selected_context_eval_config.ema.alpha:.2f}")
        with col2:
            st.metric("Hysteresis", f"{selected_context_eval_config.ema.hysteresis:.2f}")
        with col3:
            st.metric("Dwell Time", selected_context_eval_config.ema.dwell_time)

        st.caption(
            "Alpha: smoothing factor (higher = less smoothing). "
            "Hysteresis: buffer to prevent jitter. "
            "Dwell time: periods before transition."
        )

        st.metric("Neutral Threshold", f"{selected_context_eval_config.neutral_threshold:.2f}")

    with tab_markers:
        for marker, mm in selected_context_eval_config.marker_memberships.items():
            with st.expander(f"**{marker}**", expanded=False):
                marker_data = []
                for set_name, set_def in mm.sets.items():
                    marker_data.append({
                        "Set": set_name,
                        "Type": set_def.type,
                        "Params": str(list(set_def.params)),
                    })
                st.dataframe(
                    pd.DataFrame(marker_data),
                    hide_index=True,
                    use_container_width=True,
                )

    with tab_assumptions:
        for ctx_name, assumption in selected_context_eval_config.context_assumptions.items():
            with st.expander(f"**{ctx_name}**", expanded=False):
                assumption_data = []
                for cond in assumption.conditions:
                    assumption_data.append({
                        "Marker": cond.marker,
                        "Required Set": cond.fuzzy_set,
                        "Weight": f"{cond.weight:.2f}",
                    })
                st.dataframe(
                    pd.DataFrame(assumption_data),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(f"Operator: {assumption.operator}")

# --- Run Evaluation Button ---
run_eval_clicked = st.button(
    "Run Evaluation",
    type="primary",
    key="ctx_run_eval_button",
    help="Create a new context evaluation run with the selected configuration",
    use_container_width=True,
)

# --- Handle Run Evaluation Click ---
if run_eval_clicked:
    with st.status("Running context evaluation...", expanded=True) as status:
        try:
            from datetime import timedelta
            from datetime import datetime as dt
            from datetime import timezone

            from sqlalchemy import func, select
            from src.shared.models import Context

            status.update(label="Finding data range...")

            # First, find the user's actual data range
            with SessionLocal() as db:
                range_query = select(
                    func.min(Context.timestamp),
                    func.max(Context.timestamp),
                ).where(Context.user_id == user_id)
                result = db.execute(range_query).one()
                data_start, data_end = result

            if data_start is None or data_end is None:
                status.update(label="No data found", state="error")
                st.warning("No context marker data found for this user.")
                st.stop()

            # Add small buffer to ensure we capture edge cases
            start_time = data_start - timedelta(hours=1)
            end_time = data_end + timedelta(hours=1)

            status.update(label="Creating evaluation run...")

            # Create the run record
            run = create_context_evaluation_run(
                user_id=user_id,
                context_eval_config=selected_context_eval_config,
            )
            # Auto-select the new run in the dropdown
            st.session_state["ctx_selected_run_id"] = str(run.id)
            st.session_state["ctx_run_selector"] = str(run.id)

            # Create evaluator from the selected config
            evaluator = ContextEvaluator.from_experiment_config(
                selected_context_eval_config
            )

            # Progress callback for evaluation loop
            def update_eval_progress(current: int, total: int) -> None:
                status.update(label=f"Evaluating: {current}/{total} timestamps")

            status.update(label="Evaluating context history...")

            # Run evaluation for the user's data range
            with SessionLocal() as db:
                service = ContextHistoryService(db, evaluator=evaluator)
                eval_result = service.ensure_context_history_exists(
                    user_id=user_id,
                    start=start_time,
                    end=end_time,
                    context_evaluation_run_id=run.id,
                    progress_callback=update_eval_progress,
                )
                db.commit()

            # Update run's evaluation count
            update_run_evaluation_count(run.id, eval_result.evaluations_added)

            status.update(label="Evaluation complete!", state="complete")
            st.success(
                f"Created {eval_result.evaluations_added} context evaluations "
                f"(found {eval_result.gaps_found} gaps)"
            )
            st.rerun()
        except Exception as e:
            status.update(label="Evaluation failed", state="error")
            st.error(f"Failed to run evaluation: {e}")
            logger.error("Failed to run context evaluation", exc_info=True)

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
                View and explore evaluation run outputs
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Load Context Evaluations for Selected Run
# ==============================================================================

# --- Context Run Selector ---
runs_df = list_context_evaluation_runs(user_id=user_id, limit=20)

if runs_df.empty:
    st.info("No evaluation runs yet. Click **Run Evaluation** to create one.")
    selected_run_id = None
else:
    # Add placeholder option at the start
    run_options = [(None, "Select a run...")]
    display_tz = get_display_timezone()
    for _, run in runs_df.iterrows():
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
        run_options.append((run["id"], run_label))

    selected_run_key = st.selectbox(
        "Select Run",
        options=[opt[0] for opt in run_options],
        format_func=lambda x: next(opt[1] for opt in run_options if opt[0] == x),
        index=0,
        key="ctx_run_selector",
        help="Select an evaluation run to view its results",
    )
    selected_run_id = selected_run_key if selected_run_key is not None else None
    st.session_state["ctx_selected_run_id"] = selected_run_id

# Only show results if a run is selected
if selected_run_id is None:
    render_footer()
    st.stop()

# Load evaluations for the selected run (no limit - show all evaluations in run)
latest_history_df = load_context_history_records(
    user_id=user_id,
    context_evaluation_run_id=selected_run_id,
)

# Display status
has_data = not latest_history_df.empty

if not has_data:
    st.info("No evaluations found for this run.")


# ==============================================================================
# HERO SECTION - Current Context Status (AC #1)
# ==============================================================================

# Evaluation selector dropdown (uses latest_history_df - no date filter)
selected_record = None
if not latest_history_df.empty:
    # Format timestamps for dropdown display
    display_tz = get_display_timezone()

    def format_eval_timestamp(row: pd.Series) -> str:
        """Format evaluation timestamp for dropdown display."""
        ts = row["evaluated_at"]
        if hasattr(ts, "astimezone"):
            if ts.tzinfo is None:
                from datetime import UTC

                ts = ts.replace(tzinfo=UTC)
            ts = ts.astimezone(display_tz)
        ctx = row["dominant_context"].replace("_", " ").title()
        confidence = row["confidence"]
        return f"{ts.strftime('%Y-%m-%d %H:%M:%S')} — {ctx} ({confidence:.0%})"

    # Build options list (most recent first for easier selection)
    eval_options = []
    for idx in range(len(latest_history_df) - 1, -1, -1):
        row = latest_history_df.iloc[idx]
        label = format_eval_timestamp(row)
        eval_options.append((idx, label))

    # Default to most recent (first in reversed list)
    # Key includes run_id so selection resets when run changes
    selected_idx = st.selectbox(
        "Select Evaluation",
        options=[opt[0] for opt in eval_options],
        format_func=lambda idx: next(opt[1] for opt in eval_options if opt[0] == idx),
        index=0,
        key=f"ctx_eval_selector_{selected_run_id}",
        help="Select a recent context evaluation to inspect its pipeline details",
    )

    selected_record = latest_history_df.iloc[selected_idx].to_dict()

if selected_record:
    dominant_context = selected_record.get("dominant_context", "neutral")
    confidence = selected_record.get("confidence", 0.0)
    evaluated_at = selected_record.get("evaluated_at")
    trigger = selected_record.get("evaluation_trigger", "unknown")
    context_state = selected_record.get("context_state") or {}
    # Step 5: Stabilization transparency fields
    switch_blocked = selected_record.get("switch_blocked")
    switch_blocked_reason = selected_record.get("switch_blocked_reason")
    candidate_context = selected_record.get("candidate_context")
    score_difference = selected_record.get("score_difference")
    dwell_progress = selected_record.get("dwell_progress")
    dwell_required = selected_record.get("dwell_required")
    # Step 6: Computation metadata
    created_at = selected_record.get("created_at")

    # Get all context scores sorted by confidence
    all_scores = sorted(context_state.items(), key=lambda x: x[1], reverse=True)

    # Main context display
    dominant_color = CONTEXT_COLORS.get(dominant_context, "#7f7f7f")

    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 1.5rem 1rem;
            margin-bottom: 1rem;
            border-radius: 0.75rem;
            background: linear-gradient(135deg, {dominant_color}15, {dominant_color}05);
            border: 1px solid {dominant_color}30;
        ">
            <div style="font-size: 0.9rem; color: #888; margin-bottom: 0.25rem;">
                Selected Context
            </div>
            <div style="
                font-size: 2rem;
                font-weight: 600;
                color: {dominant_color};
                margin-bottom: 0.25rem;
            ">
                {dominant_context.replace('_', ' ').title()}
            </div>
            <div style="font-size: 1rem; font-weight: 500; color: #666;">
                Confidence: {confidence:.0%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Other contexts as horizontal bars
    if all_scores:
        st.markdown(
            "<div style='font-size: 0.85rem; color: #888; margin-bottom: 0.5rem;'>"
            "All Context Scores"
            "</div>",
            unsafe_allow_html=True,
        )

        for ctx_name, ctx_score in all_scores:
            ctx_color = CONTEXT_COLORS.get(ctx_name, "#7f7f7f")
            is_dominant = ctx_name == dominant_context
            bar_width = max(ctx_score * 100, 2)  # Minimum 2% width for visibility

            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    margin-bottom: 0.4rem;
                    font-size: 0.85rem;
                ">
                    <div style="
                        width: 130px;
                        color: {'#333' if is_dominant else '#666'};
                        font-weight: {'600' if is_dominant else '400'};
                    ">
                        {ctx_name.replace('_', ' ').title()}
                    </div>
                    <div style="
                        flex: 1;
                        height: 8px;
                        background: #f0f0f0;
                        border-radius: 4px;
                        overflow: hidden;
                        margin: 0 0.75rem;
                    ">
                        <div style="
                            width: {bar_width}%;
                            height: 100%;
                            background: {ctx_color};
                            border-radius: 4px;
                            opacity: {1.0 if is_dominant else 0.6};
                        "></div>
                    </div>
                    <div style="
                        width: 45px;
                        text-align: right;
                        color: {'#333' if is_dominant else '#888'};
                        font-weight: {'600' if is_dominant else '400'};
                    ">
                        {ctx_score:.0%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Timestamp and trigger as subtle footer (convert to configured timezone)
    display_tz = get_display_timezone()
    if hasattr(evaluated_at, "astimezone"):
        # If datetime is naive (no tzinfo), assume it's UTC
        if evaluated_at.tzinfo is None:
            from datetime import UTC

            evaluated_at_aware = evaluated_at.replace(tzinfo=UTC)
        else:
            evaluated_at_aware = evaluated_at
        local_ts = evaluated_at_aware.astimezone(display_tz)
        ts_str = local_ts.strftime("%Y-%m-%d %H:%M:%S")
    elif hasattr(evaluated_at, "strftime"):
        ts_str = evaluated_at.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = str(evaluated_at)

    trigger_labels = {
        "backfill": "🔄 backfill",
        "on_demand": "▶️ on-demand",
        "manual": "✋ manual",
    }
    trigger_label = trigger_labels.get(trigger, f"{trigger}")

    st.caption(f"Context at: {ts_str} ({trigger_label})")

else:
    st.info(
        "📋 No context history available for the selected time range. "
        "Run an analysis to generate context history, or ensure context markers exist."
    )


# ==============================================================================
# PIPELINE TRANSPARENCY SECTION (AC #3)
# ==============================================================================

if selected_record:
    st.divider()

    # Debug: show which record is being used for pipeline transparency
    record_run_id = selected_record.get("context_evaluation_run_id", "None")
    record_eval_at = selected_record.get("evaluated_at", "None")

    # Build summary
    sensors_used = selected_record.get("sensors_used") or []
    sensor_snapshot = selected_record.get("sensor_snapshot") or {}
    context_state = selected_record.get("context_state") or {}
    raw_scores_stored = selected_record.get("raw_scores")

    # Get pipeline transparency data
    pipeline_data = get_pipeline_transparency_data(
        sensor_snapshot=sensor_snapshot,
        context_state=context_state,
        sensors_used=sensors_used,
        raw_scores_stored=raw_scores_stored,
    )

    if pipeline_data is None:
        st.warning(
            "Pipeline transparency data could not be loaded. "
            "Sensor snapshot may be missing for this evaluation."
        )
    else:
        # Collapsed summary
        markers_used_count = len(pipeline_data["markers_used"])
        markers_missing_count = len(pipeline_data["markers_missing"])
        winner = pipeline_data["winner"]
        runner_up = pipeline_data["runner_up"]

        summary = (
            f"{markers_used_count} markers used, {markers_missing_count} missing — "
            f"{winner[0]} ({winner[1]:.2f})"
        )
        if runner_up[0]:
            summary += f" > {runner_up[0]} ({runner_up[1]:.2f})"

        # Pipeline Transparency Section Header
        st.markdown("### Pipeline Transparency")
        st.caption(
            "Step-by-step breakdown of how context was computed. "
            "Click each step to expand."
        )

        # ================================================================
        # Step 1: Extract Latest Values
        # ================================================================
        with st.expander("Step 1: Extract Latest Values", expanded=False):
            st.caption(
                "The pipeline extracts the most recent value for each marker. "
                "Context evaluation reflects the *current* situation, not historical averages."
            )

            marker_cols = st.columns(3)
            for i, (marker, value) in enumerate(pipeline_data["marker_values"].items()):
                with marker_cols[i % 3]:
                    st.metric(
                        label=marker.replace("_", " ").title(),
                        value=f"{value:.2f}",
                    )

            if pipeline_data["markers_missing"]:
                st.warning(
                    f"**Missing markers:** {', '.join(pipeline_data['markers_missing'])} — "
                    "These are expected by some context assumptions but not present in data."
                )

        # ================================================================
        # Step 2: Compute Fuzzy Membership
        # ================================================================
        with st.expander("Step 2: Compute Fuzzy Membership", expanded=False):
            st.caption(
                "Each numeric value is converted into membership degrees across fuzzy sets "
                "(e.g., low, medium, high). A value can belong to multiple sets simultaneously."
            )

            membership_config = pipeline_data.get("membership_config", {})

            for marker, sets in pipeline_data["memberships"].items():
                value = pipeline_data["marker_values"].get(marker, 0)
                marker_config = membership_config.get(marker, {})

                # Summary line
                sets_str = ", ".join(
                    f"**{k}**={v:.2f}" for k, v in sets.items() if v > 0
                )
                if not sets_str:
                    sets_str = "all sets = 0.0"
                st.markdown(f"**`{marker}`** = {value:.2f} -> {sets_str}")

                # Toggleable details with visualization
                if st.toggle(
                    f"Show {marker} details",
                    value=False,
                    key=f"step2_{marker}_details",
                ):
                    for set_name, membership_value in sets.items():
                        set_config = marker_config.get(set_name, {})
                        func_type = set_config.get("type", "triangular")
                        params = set_config.get("params", [0, 0, 1])

                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.markdown(f"**{set_name}** ({func_type})")
                            st.markdown(f"Parameters: `{params}`")
                            st.markdown(f"Result: **{membership_value:.3f}**")

                            # Formula explanation
                            explanation = get_membership_formula_explanation(
                                func_type, params, value, membership_value
                            )
                            st.markdown(f"*{explanation}*")

                        with col2:
                            # ASCII visualization
                            ascii_chart = generate_membership_ascii(
                                func_type, params, value, width=30
                            )
                            st.code(ascii_chart, language=None)

        # ================================================================
        # Step 3: Evaluate Context Assumptions
        # ================================================================
        with st.expander("Step 3: Evaluate Context Assumptions", expanded=False):
            st.caption(
                "Each context has an assumption — a weighted combination of marker memberships. "
                "The score shows how well current data matches each context definition."
            )

            context_assumptions = pipeline_data.get("context_assumptions", {})

            for ctx, raw_score in pipeline_data["raw_scores"].items():
                assumption = context_assumptions.get(ctx, {})

                if "threshold" in assumption:
                    st.markdown(
                        f"**`{ctx}`**: fallback threshold = {assumption['threshold']}"
                    )
                    continue

                conditions = assumption.get("conditions", {})
                operator = assumption.get("operator", "WEIGHTED_MEAN")

                st.markdown(f"**`{ctx}`** = **{raw_score:.3f}**")

                if st.toggle(
                    f"Show {ctx} calculation",
                    value=False,
                    key=f"step3_{ctx}_calc",
                ):
                    # Show the assumption configuration
                    st.markdown("**Assumption Definition** (from config):")
                    for marker_name, cond in conditions.items():
                        required_set = cond.get("set", "?")
                        weight = cond.get("weight", 0)
                        st.markdown(
                            f"- `{marker_name}` requires **{required_set}** "
                            f"(weight: {weight})"
                        )

                    st.markdown(f"Operator: **{operator}**")

                    st.markdown("---")
                    st.markdown("**Calculation:**")

                    # Build the formula
                    formula_parts = []
                    calculation_parts = []
                    total_weight = 0

                    for marker_name, cond in conditions.items():
                        required_set = cond.get("set", "?")
                        weight = cond.get("weight", 0)
                        memberships = pipeline_data["memberships"]
                        membership = memberships.get(marker_name, {}).get(
                            required_set, 0
                        )

                        formula_parts.append(
                            f"{weight} x {required_set}({marker_name})"
                        )
                        calculation_parts.append(f"{weight} x {membership:.3f}")
                        total_weight += weight

                    if total_weight > 0:
                        st.markdown(
                            "Formula: `score = Sum(weight x membership) / Sum(weights)`"
                        )
                        formula_str = " + ".join(formula_parts)
                        calc_str = " + ".join(calculation_parts)
                        st.markdown(f"= ({formula_str}) / {total_weight:.1f}")
                        st.markdown(f"= ({calc_str}) / {total_weight:.1f}")

                        # Calculate the weighted sum
                        weighted_sum = sum(
                            conditions[m].get("weight", 0)
                            * pipeline_data["memberships"]
                            .get(m, {})
                            .get(conditions[m].get("set", ""), 0)
                            for m in conditions
                        )
                        st.markdown(
                            f"= {weighted_sum:.3f} / {total_weight:.1f} "
                            f"= **{raw_score:.3f}**"
                        )

        # ================================================================
        # Step 4: Apply EMA Smoothing
        # ================================================================
        config = pipeline_data.get("config", {})
        with st.expander("Step 4: Apply EMA Smoothing", expanded=False):
            alpha = config.get("ema_alpha", 0.3)

            st.caption(
                f"Exponential Moving Average dampens rapid fluctuations. "
                f"Current α = **{alpha}** means {alpha:.0%} weight on new value, "
                f"{1-alpha:.0%} weight on history."
            )

            # Warn if raw scores had to be recomputed (legacy records)
            if not pipeline_data.get("raw_scores_from_storage", False):
                st.warning(
                    "⚠️ **Legacy Record:** Raw scores were recomputed from sensor snapshot. "
                    "This evaluation was created before EMA smoothing was enabled during backfill. "
                    "Re-run 'Evaluate Context' to generate records with proper EMA smoothing."
                )

            # Build detailed EMA data for each context
            ema_details = []
            for ctx in pipeline_data["raw_scores"]:
                raw = pipeline_data["raw_scores"].get(ctx, 0)
                smoothed = pipeline_data["smoothed_scores"].get(ctx, 0)
                delta = smoothed - raw

                # Infer previous smoothed value from EMA formula:
                # smoothed_t = α × raw_t + (1-α) × smoothed_{t-1}
                # Therefore: smoothed_{t-1} = (smoothed_t - α × raw_t) / (1-α)
                if abs(1 - alpha) > 0.001:  # Avoid division by zero
                    prev_smoothed = (smoothed - alpha * raw) / (1 - alpha)
                else:
                    prev_smoothed = raw  # Edge case: alpha ≈ 1

                # Calculate smoothing impact
                if abs(raw) > 0.001:
                    pct_change = (delta / raw) * 100
                else:
                    pct_change = 0.0

                ema_details.append(
                    {
                        "context": ctx,
                        "raw": raw,
                        "smoothed": smoothed,
                        "delta": delta,
                        "pct_change": pct_change,
                        "prev_smoothed": prev_smoothed,
                        "is_winner": ctx == winner[0],
                    }
                )

            # Sort by smoothed score descending
            ema_details.sort(key=lambda x: x["smoothed"], reverse=True)

            # Display as styled table
            st.markdown("**EMA Results:**")

            for detail in ema_details:
                ctx = detail["context"]
                raw = detail["raw"]
                smoothed = detail["smoothed"]
                delta = detail["delta"]
                pct_change = detail["pct_change"]
                prev_smoothed = detail["prev_smoothed"]
                is_winner = detail["is_winner"]

                # Direction and color indicators
                if delta > 0.01:
                    direction = "↑"
                    delta_color = "#28a745"  # Green - smoothing pushed up
                elif delta < -0.01:
                    direction = "↓"
                    delta_color = "#dc3545"  # Red - smoothing pushed down
                else:
                    direction = "→"
                    delta_color = "#6c757d"  # Gray - minimal change

                # Winner styling
                ctx_display = f"**{ctx}**" if is_winner else ctx
                winner_badge = " 🏆" if is_winner else ""

                # Main result line
                st.markdown(
                    f"**`{ctx}`**{winner_badge}: "
                    f"{raw:.3f} → **{smoothed:.3f}** "
                    f"<span style='color:{delta_color}'>{direction} ({delta:+.3f})</span>",
                    unsafe_allow_html=True,
                )

                # Expandable details for each context
                if st.toggle(
                    f"Show {ctx} calculation",
                    value=False,
                    key=f"step4_ema_{ctx}",
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Values:**")
                        st.markdown(f"- Raw score (this eval): `{raw:.4f}`")
                        st.markdown(f"- Smoothed score: `{smoothed:.4f}`")
                        st.markdown(
                            f"- Previous smoothed (inferred): `{prev_smoothed:.4f}`"
                        )
                        st.markdown(f"- Delta (smoothed - raw): `{delta:+.4f}`")
                        if abs(pct_change) > 0.1:
                            st.markdown(f"- Relative change: `{pct_change:+.1f}%`")

                    with col2:
                        st.markdown("**Calculation:**")
                        st.code(
                            f"smoothed = α × raw + (1-α) × prev\n"
                            f"        = {alpha} × {raw:.4f} + {1-alpha} × {prev_smoothed:.4f}\n"
                            f"        = {alpha * raw:.4f} + {(1-alpha) * prev_smoothed:.4f}\n"
                            f"        = {smoothed:.4f}",
                            language=None,
                        )

                        # Interpretation
                        if abs(delta) < 0.01:
                            st.info("Minimal smoothing effect - raw ≈ smoothed")
                        elif delta > 0:
                            st.success(
                                f"History pulled score UP by {abs(delta):.3f} "
                                f"(prev was {prev_smoothed:.3f})"
                            )
                        else:
                            st.error(
                                f"History pulled score DOWN by {abs(delta):.3f} "
                                f"(prev was {prev_smoothed:.3f})"
                            )

            # Show formula toggle
            if st.toggle(
                "Show EMA formula reference", value=False, key="step4_ema_formula"
            ):
                st.markdown("---")
                st.markdown("**EMA Formula:**")
                st.latex(
                    r"\text{smoothed}_t = \alpha \times \text{raw}_t + "
                    r"(1 - \alpha) \times \text{smoothed}_{t-1}"
                )
                st.markdown(f"With α = {alpha}:")
                st.latex(
                    rf"\text{{smoothed}}_t = {alpha} \times \text{{raw}}_t + "
                    rf"{1-alpha:.1f} \times \text{{smoothed}}_{{t-1}}"
                )
                st.markdown(
                    "**Interpretation:** The smoothed value is a weighted average of "
                    "the current raw score and the previous smoothed value. "
                    f"With α={alpha}, each new reading contributes {alpha:.0%} "
                    f"while history contributes {1-alpha:.0%}."
                )

        # ================================================================
        # Step 5: Apply Hysteresis & Dwell Time
        # ================================================================
        hysteresis_threshold = config.get("hysteresis_threshold", 0.1)
        dwell_time_required = config.get("dwell_time", 2)
        with st.expander("Step 5: Apply Hysteresis & Dwell Time", expanded=False):
            st.caption(
                "Stabilization prevents rapid context oscillations. "
                "A new context must beat the current one by a margin AND maintain leadership."
            )

            if st.toggle("Show stabilization rules", value=False, key="step5_rules"):
                st.markdown("**Hysteresis:**")
                st.markdown(
                    f"To switch contexts, the challenger must exceed the current "
                    f"winner by at least **{hysteresis_threshold}** (threshold)."
                )
                st.markdown("**Dwell Time:**")
                st.markdown(
                    f"After passing hysteresis, the challenger must win for "
                    f"**{dwell_time_required} consecutive** readings before switching."
                )

            # Display actual stabilization data if available
            if switch_blocked is not None:
                if switch_blocked:
                    # Switch was blocked - show why
                    block_color = "#ff9800"  # Orange for blocked
                    reason_label = (
                        "Hysteresis"
                        if switch_blocked_reason == "hysteresis"
                        else "Dwell Time"
                        if switch_blocked_reason == "dwell_time"
                        else switch_blocked_reason or "Unknown"
                    )
                    st.markdown(
                        f"""
                        <div style="
                            padding: 0.75rem 1rem;
                            border-radius: 0.5rem;
                            background: {block_color}15;
                            border-left: 4px solid {block_color};
                            margin: 0.5rem 0;
                        ">
                            <strong>Switch Blocked:</strong> {reason_label}<br/>
                            <strong>Candidate Context:</strong> {candidate_context or 'N/A'}<br/>
                            <strong>Score Difference:</strong> {f'{score_difference:.3f}' if score_difference is not None else 'N/A'}<br/>
                            <strong>Dwell Progress:</strong> {dwell_progress or 0} / {dwell_required or dwell_time_required}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"The challenger '{candidate_context}' would have won, but stabilization "
                        f"kept '{dominant_context}' active."
                    )
                else:
                    # Switch was NOT blocked - context change allowed or no challenger
                    st.success(
                        "No stabilization block applied. "
                        "Context determined directly by highest smoothed score."
                    )
            else:
                # Legacy data without stabilization fields
                st.info(
                    "Note: Stabilization state not available for this record. "
                    "This record was stored before Step 5 transparency was implemented."
                )

        # ================================================================
        # Step 6: Produce Result
        # ================================================================
        with st.expander("Step 6: Produce Result", expanded=False):
            st.caption(
                "The active context is determined with confidence scores for all contexts."
            )

            # Winner announcement
            winner_color = CONTEXT_COLORS.get(winner[0], "#7f7f7f")
            st.markdown(
                f"""
                <div style="
                    padding: 1rem;
                    border-radius: 0.5rem;
                    background: linear-gradient(135deg, {winner_color}30, {winner_color}10);
                    border-left: 4px solid {winner_color};
                    margin: 0.5rem 0;
                ">
                    <strong>Active Context:</strong> {winner[0].replace('_', ' ').title()}<br/>
                    <strong>Confidence:</strong> {winner[1]:.1%}<br/>
                    <strong>Margin:</strong> +{pipeline_data['delta']:.3f} over {runner_up[0] or 'none'}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Evaluation metadata
            st.markdown("---")
            st.markdown("**Evaluation Metadata:**")
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.markdown(f"**Trigger:** {trigger}")
                ctx_ts_fmt = (
                    evaluated_at.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(evaluated_at, "strftime")
                    else str(evaluated_at)
                )
                st.markdown(f"**Context Timestamp:** {ctx_ts_fmt}")
                computed_at_fmt = (
                    created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(created_at, "strftime")
                    else str(created_at)
                )
                st.markdown(f"**Computed At:** {computed_at_fmt}")
            with meta_col2:
                st.markdown(f"**Dominant Context:** {dominant_context}")
                st.markdown(f"**Confidence:** {confidence:.1%}")

        # ================================================================
        # Configuration Used
        # ================================================================
        st.markdown("### Configuration Used")
        st.caption("All configurations used for this context evaluation.")

        with st.expander("View Configuration Snapshot", expanded=False):
            # --- Marker Memberships ---
            st.markdown("#### Marker Membership Functions")
            st.caption(
                "Source: `config/context_evaluation.yaml` → `marker_memberships`"
            )

            membership_config = pipeline_data.get("membership_config", {})
            for marker_name, sets in membership_config.items():
                st.markdown(f"**{marker_name}:**")
                sets_table = []
                for set_name, set_config in sets.items():
                    func_type = set_config.get("type", "triangular")
                    params = set_config.get("params", [])
                    sets_table.append(
                        {
                            "Set": set_name,
                            "Type": func_type,
                            "Parameters": str(params),
                        }
                    )
                st.dataframe(sets_table, use_container_width=True, hide_index=True)

            st.divider()

            # --- Context Assumptions ---
            st.markdown("#### Context Assumptions")
            st.caption(
                "Source: `config/context_evaluation.yaml` → `context_assumptions`"
            )

            context_assumptions = pipeline_data.get("context_assumptions", {})
            for ctx_name, assumption in context_assumptions.items():
                st.markdown(f"**{ctx_name}:**")

                if "threshold" in assumption:
                    st.markdown(f"- Fallback threshold: `{assumption['threshold']}`")
                else:
                    conditions = assumption.get("conditions", {})
                    operator = assumption.get("operator", "WEIGHTED_MEAN")

                    conditions_table = []
                    for marker, cond in conditions.items():
                        conditions_table.append(
                            {
                                "Marker": marker,
                                "Required Set": cond.get("set", "?"),
                                "Weight": f"{cond.get('weight', 0):.2f}",
                            }
                        )
                    st.dataframe(
                        conditions_table, use_container_width=True, hide_index=True
                    )
                    st.caption(f"Operator: `{operator}`")

            st.divider()

            # --- Stabilization Settings ---
            st.markdown("#### Stabilization Settings")
            st.caption("Source: `config/ema.yaml`")

            stabilization_table = [
                {
                    "Parameter": "EMA Alpha (α)",
                    "Value": f"{config.get('ema_alpha', 0.3)}",
                    "Description": "Weight for new values in smoothing",
                },
                {
                    "Parameter": "Hysteresis Threshold",
                    "Value": f"{config.get('hysteresis_threshold', 0.1)}",
                    "Description": "Min margin to trigger context switch",
                },
                {
                    "Parameter": "Dwell Time",
                    "Value": f"{config.get('dwell_time', 2)} readings",
                    "Description": "Consecutive wins needed to switch",
                },
            ]
            st.dataframe(stabilization_table, use_container_width=True, hide_index=True)


# ==============================================================================
# HISTORICAL TRENDS CHARTS (AC #4)
# ==============================================================================

st.divider()
st.subheader("Historical Trends")

# Load history for charts (filtered by selected run)
history_df = load_context_history_records(
    user_id=user_id,
    context_evaluation_run_id=selected_run_id,
)

if not history_df.empty:
    # Confidence Score Chart from context_state
    def render_confidence_chart(
        data: pd.DataFrame,
        context_filter: list[str] | None = None,
    ) -> go.Figure:
        """Render line chart of context confidence scores over time."""
        fig = go.Figure()

        contexts_to_show = (
            context_filter if context_filter else list(CONTEXT_COLORS.keys())
        )

        # Extract scores from context_state column
        for context in contexts_to_show:
            if context not in CONTEXT_COLORS:
                continue

            # Extract score for this context from each row's context_state
            scores = []
            timestamps = []
            for _, row in data.iterrows():
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
            title="Confidence Scores Over Time",
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
            height=400,
        )

        return fig

    def add_transition_markers(
        fig: go.Figure, transitions: pd.DataFrame
    ) -> tuple[go.Figure, list[dict]]:
        """Add vertical lines at context transition points and return rapid transitions."""
        rapid_transitions = []

        for _, row in transitions.iterrows():
            is_rapid = row["duration_minutes"] < 5
            if is_rapid:
                rapid_transitions.append(row.to_dict())

            fig.add_shape(
                type="line",
                x0=row["timestamp"],
                x1=row["timestamp"],
                y0=0,
                y1=1,
                yref="paper",
                line={
                    "color": "red" if is_rapid else "orange",
                    "width": 1,
                    "dash": "dot",
                },
                opacity=0.7 if is_rapid else 0.4,
            )

        return fig, rapid_transitions

    # Render confidence chart
    confidence_fig = render_confidence_chart(history_df)

    # Add transition markers
    transitions_df = detect_context_transitions(history_df)
    if not transitions_df.empty:
        confidence_fig, rapid_list = add_transition_markers(
            confidence_fig, transitions_df
        )

        if rapid_list:
            st.warning(
                f"**Rapid Context Switching Detected:** {len(rapid_list)} transitions occurred "
                "within 5 minutes. This may indicate data quality issues or highly variable context."
            )
    else:
        rapid_list = []

    st.plotly_chart(confidence_fig, use_container_width=True)

    # Transition log
    if not transitions_df.empty:
        with st.expander("Transition Log", expanded=False):
            st.dataframe(
                transitions_df,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        "Timestamp", format="YYYY-MM-DD HH:mm:ss"
                    ),
                    "from_context": st.column_config.TextColumn("From Context"),
                    "to_context": st.column_config.TextColumn("To Context"),
                    "duration_minutes": st.column_config.NumberColumn(
                        "Duration in Previous (min)", format="%.1f"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )


else:
    st.info("No context history found for the selected run.")


# ==============================================================================
# CONTEXT TIMELINE VISUALIZATION (AC #5)
# ==============================================================================

if not history_df.empty:
    # Get context timeline from service (uses data range from loaded history)
    try:
        # Derive date range from loaded data
        timeline_start = history_df["evaluated_at"].min()
        timeline_end = history_df["evaluated_at"].max()

        with SessionLocal() as db:
            service = ContextHistoryService(db)
            segments = service.get_context_timeline(
                user_id=user_id,
                start=timeline_start,
                end=timeline_end,
            )

        if segments:
            # Build segment data
            segment_data = []
            for seg in segments:
                segment_data.append(
                    {
                        "context": seg.context.replace("_", " ").title(),
                        "start": seg.start,
                        "end": seg.end,
                        "duration_min": seg.duration_minutes,
                        "avg_confidence": seg.confidence,
                    }
                )

            seg_df = pd.DataFrame(segment_data)

            # Timeline visualization using Plotly
            fig = go.Figure()

            for _, row in seg_df.iterrows():
                ctx_name = row["context"].lower().replace(" ", "_")
                color = CONTEXT_COLORS.get(ctx_name, "#7f7f7f")

                fig.add_trace(
                    go.Scatter(
                        x=[row["start"], row["end"]],
                        y=[row["context"], row["context"]],
                        mode="lines",
                        line={"color": color, "width": 20},
                        name=row["context"],
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{row['context']}</b><br>"
                            f"Duration: {row['duration_min']:.1f} min<br>"
                            f"Confidence: {row['avg_confidence']:.1%}<br>"
                            f"Start: %{{x}}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title="Context Segments",
                xaxis_title="Time",
                yaxis_title="",
                height=200,
                margin={"t": 40, "b": 40},
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No context segments found for the selected time range.")

    except Exception as e:
        logger.error(f"Failed to load context timeline: {e}", exc_info=True)
        st.error("Failed to load context timeline segments.")


# ==============================================================================
# About Section
# ==============================================================================

st.divider()

with st.expander("About Context Evaluation", expanded=False):
    st.markdown(
        """
**This page lets you run and explore context evaluations.**

Context evaluation uses fuzzy logic to determine what situation a user is in
(e.g., social gathering, solitary digital, work/study) based on sensor data
like location, screen time, and app usage.

**Workflow:**

1. **Select Config Source** — Choose the fuzzy logic configuration to use:
   - *Default Config* — The standard configuration from YAML files
   - *Experiments* — Custom configurations for A/B testing different parameters

2. **Run Evaluation** — Processes the user's entire sensor history and creates
   a new evaluation run with context classifications at regular intervals

3. **Select Run** — Choose from previous evaluation runs to compare results
   from different configurations or time periods

4. **Explore Results** — View the selected evaluation's details:
   - *Hero Section* — Current context and confidence scores
   - *Pipeline Transparency* — Step-by-step breakdown of how context was computed
   - *Historical Trends* — Charts showing context changes over time

**Key Concepts:**

- **Evaluation Runs** — Each run is versioned with its configuration snapshot,
  allowing reproducible experiments and comparison between different settings
- **EMA Smoothing** — Exponential moving average smooths rapid fluctuations
- **Hysteresis** — Prevents oscillation by requiring a margin to switch contexts
- **Dwell Time** — Requires sustained readings before confirming a context switch
"""
    )


render_footer()
