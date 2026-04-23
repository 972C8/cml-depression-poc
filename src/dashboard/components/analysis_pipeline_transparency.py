"""Analysis Pipeline Transparency Component.

Story 6.11: Analysis Pipeline Transparency

Provides an indicator-centric transparency view that allows users to select
any daily indicator and see a step-by-step breakdown of how it was computed.
This is DIFFERENT from pipeline_transparency.py which shows run-level steps.

This component shows:
- Step 2: Context Overview (per window)
- Step 3: All Biomarkers Overview
- Step 4: Relevant Biomarkers for Indicator
- Step 5: Membership Computation
- Step 6: Window-Level FASL Calculation
- Step 7: Daily Summary Computation
- Configuration Summary
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.data.indicator_transparency import (
    DailyIndicatorOption,
    compute_fasl_for_display,
    compute_membership_for_display,
    get_all_window_times,
    get_available_daily_indicators,
    get_biomarker_aggregates_for_date,
    get_context_history_for_date,
    get_context_weights_config,
    get_daily_indicator_summary,
    get_indicator_config,
    get_window_aggregates_for_date,
    get_window_indicator_details,
)
from src.shared.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "render_analysis_pipeline_transparency",
    "render_indicator_selector",
]


def render_indicator_selector(
    user_id: str | None,
    session_prefix: str = "",
    run_id: str | None = None,
) -> DailyIndicatorOption | None:
    """Render the daily indicator selector dropdown.

    AC1: Daily Indicator Selector
    - Dropdown format: YYYY-MM-DD - indicator_name
    - Sorted by date (most recent first)
    - Shows "No analysis results available" when no data

    Args:
        user_id: Selected user ID, or None if no user selected
        session_prefix: Prefix for session state keys to support side-by-side
            comparison mode. Use "primary_" or "comparison_" when in
            side-by-side mode.
        run_id: Optional run ID to filter indicators for a specific run

    Returns:
        Selected DailyIndicatorOption or None if no selection/no data
    """
    if user_id is None:
        st.info("Select a user to view analysis pipeline transparency.")
        return None

    # Get available indicators (optionally filtered by run_id in future)
    options = get_available_daily_indicators(user_id)

    if not options:
        st.warning(
            "No analysis results available. Run the analysis pipeline first "
            "to generate indicator data."
        )
        return None

    # Render dropdown - let Streamlit manage selection state via key
    selected_label = st.selectbox(
        "Select Indicator to Inspect",
        options=[opt.display_label for opt in options],
        help="Select a daily indicator to see the step-by-step computation breakdown",
        key=f"{session_prefix}indicator_transparency_selector",
    )

    # Find and return selected option
    for opt in options:
        if opt.display_label == selected_label:
            return opt

    return None


def render_analysis_pipeline_transparency(
    user_id: str | None,
    session_prefix: str = "",
    run_id: str | None = None,
) -> None:
    """Render the complete analysis pipeline transparency view.

    This is the main entry point for the transparency component.
    Shows indicator selector and all computation steps.

    Args:
        user_id: Selected user ID, or None if no user selected
        session_prefix: Prefix for session state keys to support side-by-side
            comparison mode. Use "primary_" or "comparison_" when in
            side-by-side mode. Default "" for single-column mode.
        run_id: Optional run ID to filter indicators for a specific run
    """
    # Early return if no user selected - don't show anything
    if user_id is None:
        return

    # Check if there are any indicators to show before rendering the section
    options = get_available_daily_indicators(user_id)
    if not options:
        return

    # Now render the section header since we have data
    st.markdown("### Analysis Pipeline Transparency")
    st.caption(
        "Select an indicator to see a step-by-step breakdown of how it was computed. "
        "Each step shows the actual values, formulas, and configurations used."
    )

    # Render indicator selector (will have data since we checked above)
    selected_option = render_indicator_selector(
        user_id, session_prefix=session_prefix, run_id=run_id
    )

    if selected_option is None:
        return

    # Load the daily indicator summary data
    summary = get_daily_indicator_summary(selected_option.indicator_id)

    if summary is None:
        st.error(f"Failed to load data for indicator: {selected_option.display_label}")
        return

    # Store in session state for use by step renderers (with prefix)
    st.session_state[f"{session_prefix}current_indicator_summary"] = summary
    st.session_state[
        f"{session_prefix}current_indicator_date"
    ] = selected_option.indicator_date
    st.session_state[
        f"{session_prefix}current_indicator_name"
    ] = selected_option.indicator_name
    st.session_state[f"{session_prefix}current_user_id"] = user_id

    # Show selected indicator summary card
    likelihood = summary.get("likelihood", 0)
    quality = summary.get("quality", {})
    duration = summary.get("duration", {})
    coverage = quality.get("data_coverage", 0)
    total_windows = duration.get("total_windows", 0)
    date_str = selected_option.indicator_date.strftime("%Y-%m-%d")
    weekday = selected_option.indicator_date.strftime("%A")

    st.markdown(
        f"""
<div style="
    display: flex;
    align-items: center;
    gap: 2rem;
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    background: linear-gradient(135deg, #3b82f610, #3b82f605);
    border: 1px solid #3b82f625;
    margin-bottom: 1rem;
">
    <div style="flex: 1;">
        <div style="font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Indicator</div>
        <div style="font-size: 1.25rem; font-weight: 600; color: #3b82f6;">{selected_option.indicator_name}</div>
        <div style="font-size: 0.85rem; color: #6b7280; margin-top: 0.25rem;">{date_str} <span style="opacity: 0.7;">({weekday})</span></div>
    </div>
    <div style="text-align: center; padding: 0 1.5rem; border-left: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb;">
        <div style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Daily Likelihood</div>
        <div style="font-size: 1.75rem; font-weight: 700; color: #374151;">{likelihood:.3f}</div>
    </div>
    <div style="display: flex; gap: 1.5rem;">
        <div style="text-align: center;">
            <div style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Coverage</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #374151;">{coverage:.0%}</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px;">Windows</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #374151;">{total_windows}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Render each step (placeholder for now - will be implemented in subsequent tasks)
    st.markdown("#### Computation Steps")
    st.caption(
        "Click each step to expand and see the detailed computation breakdown. "
        "Steps 2-7 follow the analysis pipeline sequence. "
        "Non-computation steps were omitted."
    )

    # Step 2: Context Overview (Task 2)
    render_step_2_context_overview(user_id, selected_option, session_prefix)

    # Step 3: All Biomarkers (Task 3)
    render_step_3_all_biomarkers(user_id, selected_option, session_prefix)

    # Step 4: Relevant Biomarkers (Task 4)
    render_step_4_relevant_biomarkers(user_id, selected_option, session_prefix)

    # Step 5: Membership Computation (Task 5)
    render_step_5_membership_computation(user_id, selected_option, session_prefix)

    # Step 6: Window FASL (Task 6)
    render_step_6_window_fasl(user_id, selected_option, session_prefix)

    # Step 7: Daily Summary (Task 7)
    render_step_7_daily_summary(summary, session_prefix)

    # Configuration Summary (Task 8)
    render_config_summary(user_id, selected_option.indicator_name, session_prefix)


def render_step_2_context_overview(
    user_id: str,
    option: DailyIndicatorOption,
    session_prefix: str = "",
) -> None:
    """Render Step 2: Context Overview.

    AC2: Step 2 - Context Overview
    - Show summary text explaining context varies per window with link to /context page
    - Display context table: Window | Dominant Context | Confidence
    - Show context distribution summary (e.g., "neutral 72 windows 75%...")
    - Data source: context_history table filtered by user and date

    Args:
        user_id: User ID for context history lookup
        option: Selected indicator option with date information
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    with st.expander("Step 2: Context Overview", expanded=False):
        st.markdown(
            """
**Context affects how biomarkers are weighted.** The same low speech activity might be:
- **Concerning** at a social gathering (context weight > 1.0)
- **Normal** when alone at home (context weight < 1.0)

Context is evaluated per-window, so different times of day may have different contexts.
For detailed context evaluation configuration, see the [Context page](/Context).
"""
        )

        # Load context history for the selected day
        context_history = get_context_history_for_date(user_id, option.indicator_date)

        if not context_history:
            st.warning(
                "No context history found for this day. Context weights default to neutral (1.0)."
            )
            return

        # Calculate context distribution
        context_counts: dict[str, int] = {}
        for record in context_history:
            ctx = record.get("dominant_context", "unknown")
            context_counts[ctx] = context_counts.get(ctx, 0) + 1

        total_windows = len(context_history)

        # Show context distribution summary
        st.markdown("#### Context Distribution")
        distribution_parts = []
        for ctx, count in sorted(context_counts.items(), key=lambda x: -x[1]):
            pct = count / total_windows * 100
            distribution_parts.append(f"**{ctx}** {count} windows ({pct:.1f}%)")

        st.markdown(" | ".join(distribution_parts))

        # Show context table with toggle for full details
        st.markdown("#### Context by Window")
        show_details = st.toggle(
            "Show detailed context table",
            value=False,
            key=f"{session_prefix}step2_context_details",
        )

        if show_details:
            # Build table data
            table_data = []
            for record in context_history:
                eval_time = record.get("evaluated_at")
                if eval_time:
                    window_str = eval_time.strftime("%H:%M")
                else:
                    window_str = "N/A"

                table_data.append(
                    {
                        "Window": window_str,
                        "Dominant Context": record.get("dominant_context", "unknown"),
                        "Confidence": f"{record.get('confidence', 0):.2f}",
                        "Trigger": record.get("evaluation_trigger", "N/A"),
                    }
                )

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
            )

            # Show stabilization info if any switches were blocked
            blocked_count = sum(1 for r in context_history if r.get("switch_blocked"))
            if blocked_count > 0:
                st.caption(
                    f"Note: {blocked_count} context switches were blocked by stabilization "
                    "(hysteresis or dwell time)."
                )
        else:
            st.caption(f"Enable toggle to see all {total_windows} context evaluations.")


def render_step_3_all_biomarkers(
    user_id: str,
    option: DailyIndicatorOption,
    session_prefix: str = "",
) -> None:
    """Render Step 3: All Biomarkers Overview.

    AC3: Step 3 - All Biomarkers Overview
    - Show all biomarkers with data for selected day
    - Display table: Biomarker | Windows with Data | Coverage | Daily Mean | Daily Std
    - Calculate coverage as percentage of expected windows (96 for 15-min)
    - Data source: WindowAggregate records grouped by biomarker (computed on-demand)

    Args:
        user_id: User ID for biomarker data lookup
        option: Selected indicator option with date information
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    with st.expander("Step 3: All Biomarkers Overview", expanded=False):
        st.markdown(
            """
**Overview of all available biomarker data for this day.**

This step shows which biomarkers have data and their coverage across the day's time windows.
Coverage helps identify data gaps that may affect analysis quality.
"""
        )

        # Get biomarker aggregate stats
        stats_list = get_biomarker_aggregates_for_date(user_id, option.indicator_date)

        if not stats_list:
            st.warning(
                "No biomarker data found for this day. "
                "Analysis cannot proceed without biomarker readings."
            )
            return

        # Summary metrics
        total_biomarkers = len(stats_list)
        avg_coverage = sum(s.coverage for s in stats_list) / total_biomarkers
        total_windows = sum(s.windows_with_data for s in stats_list)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Biomarkers", total_biomarkers)
        with col2:
            st.metric("Avg Coverage", f"{avg_coverage:.1%}")
        with col3:
            st.metric("Total Windows", total_windows)

        # Build table data
        st.markdown("#### Biomarker Data Summary")
        table_data = []
        for stats in stats_list:
            table_data.append(
                {
                    "Biomarker": stats.biomarker_name,
                    "Windows": stats.windows_with_data,
                    "Expected": stats.expected_windows,
                    "Coverage": f"{stats.coverage:.1%}",
                    "Daily Mean": f"{stats.daily_mean:.4f}",
                    "Daily Std": f"{stats.daily_std:.4f}",
                    "Range": f"[{stats.value_min:.3f}, {stats.value_max:.3f}]",
                }
            )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
        )

        # Coverage notes
        low_coverage = [s for s in stats_list if s.coverage < 0.3]
        if low_coverage:
            st.warning(
                f"Low coverage biomarkers ({len(low_coverage)}): "
                f"{', '.join(s.biomarker_name for s in low_coverage)}. "
                "These may reduce analysis reliability."
            )


def render_step_4_relevant_biomarkers(
    user_id: str,
    option: DailyIndicatorOption,
    session_prefix: str = "",
) -> None:
    """Render Step 4: Relevant Biomarkers for Indicator.

    AC4: Step 4 - Relevant Biomarkers for Indicator
    - Add window selector dropdown (format: HH:MM - HH:MM, default: latest by time)
    - Show indicator configuration panel from config/indicators.yaml
    - Display biomarker weights and directions in table format
    - Show window aggregates table: Biomarker | Aggregated Value | Readings Count | Aggregation Method
    - Handle missing biomarkers with warning
    - Show expandable window configuration reference

    Args:
        user_id: User ID
        option: Selected indicator option
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    with st.expander("Step 4: Relevant Biomarkers for Indicator", expanded=False):
        st.markdown(
            """
**Which biomarkers contribute to this indicator?**

This step shows the indicator's configuration (which biomarkers, weights, and directions)
and the actual aggregated values for a specific time window.
"""
        )

        # Load indicator config
        indicator_config = get_indicator_config(option.indicator_name)
        if indicator_config is None:
            st.error(f"Indicator configuration not found for: {option.indicator_name}")
            return

        biomarker_configs = indicator_config.get("biomarkers", {})

        # Show indicator configuration
        st.markdown(f"#### Indicator: {option.indicator_name}")
        st.markdown("**Biomarker Configuration:**")

        config_table = []
        for bio_name, bio_config in biomarker_configs.items():
            config_table.append(
                {
                    "Biomarker": bio_name,
                    "Weight": f"{bio_config.get('weight', 0):.2f}",
                    "Direction": bio_config.get("direction", "N/A"),
                }
            )

        st.dataframe(config_table, use_container_width=True, hide_index=True)

        # Get window times for selector
        window_times = get_all_window_times(user_id, option.indicator_date)

        if not window_times:
            st.warning("No window data available for this day.")
            return

        # Window selector (default: latest by time, which is first after reverse sort)
        window_labels = [
            f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
            for start, end in window_times
        ]

        selected_window_label = st.selectbox(
            "Select Window",
            options=window_labels,
            index=0,
            help="Select a time window to view biomarker aggregates",
            key=f"{session_prefix}step4_window_selector",
        )

        # Find the selected window
        selected_idx = window_labels.index(selected_window_label)
        selected_window_start, selected_window_end = window_times[selected_idx]

        st.caption(f"Selected: {selected_window_label}")

        # Get all window aggregates
        all_aggregates = get_window_aggregates_for_date(user_id, option.indicator_date)
        window_aggregates = all_aggregates.get(selected_window_start, {})

        # Show window aggregates table for relevant biomarkers
        st.markdown("**Window Aggregates:**")
        aggregates_table = []
        missing_biomarkers = []

        for bio_name in biomarker_configs.keys():
            if bio_name in window_aggregates:
                agg = window_aggregates[bio_name]
                aggregates_table.append(
                    {
                        "Biomarker": bio_name,
                        "Aggregated Value": f"{agg.aggregated_value:.4f}",
                        "Readings Count": agg.readings_count,
                        "Method": agg.aggregation_method,
                    }
                )
            else:
                missing_biomarkers.append(bio_name)
                aggregates_table.append(
                    {
                        "Biomarker": bio_name,
                        "Aggregated Value": "N/A (missing)",
                        "Readings Count": 0,
                        "Method": "-",
                    }
                )

        st.dataframe(aggregates_table, use_container_width=True, hide_index=True)

        st.markdown("**Step 4: Result**")
        st.caption(
            "Biomarker readings are aggregated into 15 minute windows for later analysis."
        )
        aggregates_table = []
        missing_biomarkers = []
        for bio_name in biomarker_configs.keys():
            if bio_name in window_aggregates:
                agg = window_aggregates[bio_name]
                aggregates_table.append(
                    {
                        "Biomarker": bio_name,
                        "Aggregated Value": f"{agg.aggregated_value:.4f}",
                    }
                )
            else:
                missing_biomarkers.append(bio_name)
                aggregates_table.append(
                    {
                        "Biomarker": bio_name,
                        "Aggregated Value": "N/A (missing)",
                    }
                )

        st.dataframe(aggregates_table, use_container_width=True, hide_index=True)

        # Show warning for missing biomarkers
        if missing_biomarkers:
            st.warning(
                f"Missing data for {len(missing_biomarkers)} biomarker(s): "
                f"{', '.join(missing_biomarkers)}. "
                "Neutral fill (0.5) will be used in FASL calculation."
            )

        # Show detailed computation per biomarker (toggle)
        show_details = st.toggle(
            "Show detailed computations",
            value=False,
            key=f"{session_prefix}step4_show_details",
        )

        if show_details:
            st.markdown("#### Per-Biomarker Aggregation Details")
            for bio_name in biomarker_configs.keys():
                with st.container():
                    st.markdown(f"**{bio_name}**")

                    if bio_name not in window_aggregates:
                        st.warning(
                            "No data in this window - will use neutral fill (0.5) in membership computation"
                        )
                        st.divider()
                        continue

                    agg = window_aggregates[bio_name]
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Raw Readings:**")
                        st.code(
                            f"readings_count = {agg.readings_count}\nwindow_start = {selected_window_start.strftime('%H:%M')}\nwindow_end = {selected_window_end.strftime('%H:%M')}"
                        )

                    with col2:
                        st.markdown("**Aggregation:**")
                        st.code(
                            f"method = {agg.aggregation_method}\naggregated_value = {agg.aggregated_value:.6f}"
                        )

                    st.divider()

        # Expandable formula reference
        show_formulas = st.toggle(
            "Show formula reference",
            value=False,
            key=f"{session_prefix}step4_show_formulas",
        )

        if show_formulas:
            st.markdown("#### Formula Reference")
            st.markdown("**Window Aggregation (mean):**")
            st.latex(r"\bar{x}_{\text{window}} = \frac{1}{n} \sum_{i=1}^{n} x_i")

            st.markdown("**Window Boundaries:**")
            st.latex(r"t_{\text{start}} = \lfloor t / \Delta \rfloor \times \Delta")
            st.caption("Where Δ = 15 minutes (window size)")

            st.markdown("**Missing Data Handling:**")
            st.code(
                "if readings_count == 0:\n    aggregated_value = None  # Handled as neutral fill in Step 5"
            )

        # Expandable window configuration reference
        show_window_config = st.toggle(
            "Show window configuration",
            value=False,
            key=f"{session_prefix}step4_window_config",
        )

        if show_window_config:
            st.markdown("**Window Configuration:**")
            st.code(
                """
Window Size: 15 minutes
Aggregation Method: mean
Window Boundaries: Aligned to clock time
Expected Windows per Day: 96
"""
            )
            st.caption(
                "Windows are non-overlapping (tumbling). "
                "Each reading contributes to exactly one window."
            )


def render_step_5_membership_computation(
    user_id: str,
    option: DailyIndicatorOption,
    session_prefix: str = "",
) -> None:
    """Render Step 5: Membership Computation.

    AC5: Step 5 - Membership Computation
    - Independent window selector (default: latest by time)
    - For each biomarker, show computation panel:
      - Baseline values (mean, std, source)
      - Z-score calculation with formula and result
      - Sigmoid membership calculation with formula
      - Direction inversion if applicable
    - Display membership summary table with all values
    - Show expandable formula reference section

    Args:
        user_id: User ID
        option: Selected indicator option
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    import math

    with st.expander("Step 5: Membership Computation", expanded=False):
        st.markdown(
            """
**How are raw values converted to membership scores?**

This step normalizes biomarker values using baselines and maps them to [0, 1] membership
using a sigmoid function. Direction handling ensures that concerning values always
produce high membership scores.
"""
        )

        # Get window times for selector
        window_times = get_all_window_times(user_id, option.indicator_date)

        if not window_times:
            st.warning("No window data available for this day.")
            return

        # Window selector (independent from Step 4)
        window_labels = [
            f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
            for start, end in window_times
        ]

        selected_window_label = st.selectbox(
            "Select Window",
            options=window_labels,
            index=0,
            help="Select a time window to view membership computations",
            key=f"{session_prefix}step5_window_selector",
        )

        # Find the selected window
        selected_idx = window_labels.index(selected_window_label)
        selected_window_start, _ = window_times[selected_idx]

        # Compute memberships for display
        memberships = compute_membership_for_display(
            user_id=user_id,
            target_date=option.indicator_date,
            indicator_name=option.indicator_name,
            window_start=selected_window_start,
        )

        if not memberships:
            st.warning("Could not compute memberships for this window.")
            return

        # Show membership summary table
        st.markdown("#### Membership Summary")
        summary_table = []
        for m in memberships:
            if math.isnan(m.aggregated_value):
                value_str = "N/A (missing)"
                z_str = "N/A"
                raw_mem_str = "0.500"
            else:
                value_str = f"{m.aggregated_value:.4f}"
                z_str = f"{m.z_score:.2f}"
                raw_mem_str = f"{m.membership:.3f}"

            summary_table.append(
                {
                    "Biomarker": m.biomarker_name,
                    "Value": value_str,
                    "Baseline (mean)": f"{m.baseline_mean:.3f}"
                    if not math.isnan(m.baseline_mean)
                    else "N/A",
                    "Baseline (std)": f"{m.baseline_std:.3f}"
                    if not math.isnan(m.baseline_std)
                    else "N/A",
                    "Source": m.baseline_source,
                    "Z-Score": z_str,
                    "Raw Membership": raw_mem_str,
                    "Direction": m.direction,
                    "Directed": f"{m.directed_membership:.3f}",
                }
            )

        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        # Show results table
        st.markdown("#### Step 5: Result")
        st.caption(
            "For every window aggregate from step 4, compute how unusual the value is compared to the user's baseline, mapped to a [0, 1] membership score where 0.5 is neutral and outliers mean highly unusual/concerning"
        )
        summary_table = []
        for m in memberships:
            summary_table.append(
                {
                    "Biomarker": m.biomarker_name,
                    "Directed": f"{m.directed_membership:.3f}",
                }
            )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        # Show detailed computation per biomarker (toggle)
        show_details = st.toggle(
            "Show detailed computations",
            value=False,
            key=f"{session_prefix}step5_show_details",
        )

        if show_details:
            st.markdown("#### Per-Biomarker Computation Details")
            for m in memberships:
                with st.container():
                    st.markdown(f"**{m.biomarker_name}**")

                    if math.isnan(m.aggregated_value):
                        st.warning("Missing data - neutral fill (0.5) used")
                        st.divider()
                        continue

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Baseline:**")
                        st.code(
                            f"mean = {m.baseline_mean:.4f}\nstd = {m.baseline_std:.4f}\nsource: {m.baseline_source}"
                        )

                    with col2:
                        st.markdown("**Z-Score:**")
                        st.code(
                            f"z = (value - mean) / std\nz = ({m.aggregated_value:.4f} - {m.baseline_mean:.4f}) / {m.baseline_std:.4f}\nz = {m.z_score:.4f}"
                        )

                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("**Sigmoid Membership:**")
                        st.code(
                            f"membership = 1 / (1 + exp(-z))\nmembership = 1 / (1 + exp(-{m.z_score:.4f}))\nmembership = {m.membership:.4f}"
                        )

                    with col4:
                        st.markdown("**Direction Adjustment:**")
                        if m.direction == "lower_is_worse":
                            st.code(
                                f"direction = lower_is_worse\ndirected = 1 - membership\ndirected = 1 - {m.membership:.4f}\ndirected = {m.directed_membership:.4f}"
                            )
                        else:
                            st.code(
                                f"direction = higher_is_worse\ndirected = membership\ndirected = {m.membership:.4f}"
                            )

                    st.divider()

        # Expandable formula reference
        show_formulas = st.toggle(
            "Show formula reference",
            value=False,
            key=f"{session_prefix}step5_show_formulas",
        )

        if show_formulas:
            st.markdown("#### Formula Reference")
            st.markdown("**Z-Score Normalization:**")
            st.latex(r"z = \frac{x - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}")

            st.markdown("**Sigmoid Membership:**")
            st.latex(r"\mu = \frac{1}{1 + e^{-z}}")

            st.markdown("**Direction Handling:**")
            st.latex(
                r"\mu_{\text{directed}} = \begin{cases} 1 - \mu & \text{if lower\_is\_worse} \\ \mu & \text{if higher\_is\_worse} \end{cases}"
            )


def render_step_6_window_fasl(
    user_id: str,
    option: DailyIndicatorOption,
    session_prefix: str = "",
) -> None:
    """Render Step 6: Window-Level FASL Calculation.

    Story 6.17: Reads from persisted window indicator records instead of
    recomputing independently. Falls back to recomputation if no persisted
    data is available.

    Args:
        user_id: User ID
        option: Selected indicator option
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    with st.expander("Step 6: Window-Level FASL Calculation", expanded=False):
        st.markdown(
            """
**How are biomarker memberships combined into an indicator score?**

FASL (Fuzzy-Aggregated Symptom Likelihood) combines directed memberships using
weighted averaging. Context weights amplify or dampen biomarker contributions
based on the detected context.
"""
        )

        # Get analysis_run_id from the daily indicator summary
        summary = st.session_state.get(f"{session_prefix}current_indicator_summary", {})
        analysis_run_id = summary.get("analysis_run_id")

        # --- Overview chart: all window indicator scores for this day ---
        window_scores = summary.get("window_scores", [])
        if window_scores:
            st.markdown("#### Window Indicator Scores Overview")
            st.caption(
                f"All {len(window_scores)} window-level FASL scores for "
                f"**{option.indicator_name}** on {option.indicator_date}."
            )

            chart_data = pd.DataFrame(
                {
                    "Window": [ws["window_start"] for ws in window_scores],
                    "Score": [ws["score"] for ws in window_scores],
                    "Context": [ws.get("context", "unknown") for ws in window_scores],
                }
            )

            bars = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X(
                        "Window:N",
                        sort=None,
                        axis=alt.Axis(title="Window", labelAngle=-45),
                    ),
                    y=alt.Y(
                        "Score:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(title="Indicator Score", tickCount=5),
                    ),
                    color=alt.Color(
                        "Context:N",
                        scale=alt.Scale(
                            domain=["solitary_digital", "adversarial_social_digital_gaming", "neutral"],
                            range=["#2ca02c", "#d62728", "#b0b0b0"],
                        ),
                        legend=alt.Legend(title="Context"),
                    ),
                    tooltip=[
                        alt.Tooltip("Window:N", title="Window"),
                        alt.Tooltip("Score:Q", title="Score", format=".4f"),
                        alt.Tooltip("Context:N", title="Context"),
                    ],
                )
            )

            chart = bars.properties(height=250).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True)

            st.divider()

        # Use same window source as Step 5 (only windows with biomarker data)
        window_times = get_all_window_times(user_id, option.indicator_date)

        if not window_times:
            st.warning("No window data available for this day.")
            return

        # Window selector
        st.markdown("#### Window Detail Inspection")
        window_labels = [
            f"{start.strftime('%H:%M')} - {end.strftime('%H:%M')}"
            for start, end in window_times
        ]

        selected_window_label = st.selectbox(
            "Select Window",
            options=window_labels,
            index=0,
            help="Select a time window to view FASL computation",
            key=f"{session_prefix}step6_window_selector",
        )

        selected_idx = window_labels.index(selected_window_label)
        selected_window_start, _ = window_times[selected_idx]

        # Try to load persisted data (Story 6.17)
        details = None
        if analysis_run_id:
            details = get_window_indicator_details(
                analysis_run_id=analysis_run_id,
                indicator_name=option.indicator_name,
                window_start=selected_window_start,
            )

        if details and details.get("fasl_contributions"):
            # Render from persisted data
            _render_step_6_from_persisted(details, option, selected_window_label, session_prefix)
        else:
            # Fallback: recompute (pre-6.17 data)
            _render_step_6_from_recomputation(user_id, option, selected_window_start, selected_window_label, session_prefix)

        # Expandable formula reference
        show_formulas = st.toggle(
            "Show formula reference",
            value=False,
            key=f"{session_prefix}step6_show_formulas",
        )

        if show_formulas:
            st.markdown("#### Formula Reference")
            st.markdown("**Effective Context Weight:**")
            st.latex(
                r"w_{\text{eff\_ctx}} = 1.0 + (w_{\text{ctx}} - 1.0) \times \text{confidence}"
            )

            st.markdown("**Effective Weight:**")
            st.latex(
                r"w_{\text{eff}} = w_{\text{biomarker}} \times w_{\text{eff\_ctx}}"
            )

            st.markdown("**FASL Operator:**")
            st.latex(
                r"L = \frac{\sum_{i} w_{\text{eff},i} \times \mu_{\text{directed},i}}{\sum_{i} w_{\text{eff},i}}"
            )


def _render_step_6_from_persisted(
    details: dict,
    option: "DailyIndicatorOption",
    selected_window_label: str,
    session_prefix: str,
) -> None:
    """Render Step 6 from persisted computation_log data (Story 6.17)."""
    dominant_context = details.get("dominant_context", "neutral")
    context_confidence = details.get("context_confidence", 0.0)
    indicator_score = details.get("indicator_score", details.get("value", 0.0))
    contributions = [c for c in details.get("fasl_contributions", []) if c.get("biomarker_weight", 0) > 0]

    # Show context info
    st.markdown("#### Context for this Window")
    st.dataframe(
        [{"Dominant Context": dominant_context, "Confidence": f"{context_confidence:.4f}"}],
        use_container_width=True,
        hide_index=True,
    )

    # Show context weight configuration
    st.markdown("#### Context Weight Configuration")
    context_weights_config = get_context_weights_config()
    context_weights = context_weights_config.get(dominant_context, {})

    if context_weights:
        st.markdown(f"**Context: {dominant_context}**")
        weights_table = []
        for bio_name, weight in context_weights.items():
            effect = "Amplified" if weight > 1.0 else "Dampened" if weight < 1.0 else "Neutral"
            weights_table.append({"Biomarker": bio_name, "Context Weight": f"{weight:.2f}", "Effect": effect})
        st.dataframe(weights_table, use_container_width=True, hide_index=True)
    else:
        st.info(f"No specific weights defined for context '{dominant_context}'. Using neutral (1.0) for all biomarkers.")

    # Show contributions table
    st.markdown("#### FASL Contributions")
    contrib_table = []
    for c in contributions:
        if c.get("is_missing"):
            contrib_table.append({
                "Biomarker": c["biomarker"],
                "Directed Membership": "0.500 (neutral fill)",
                "Biomarker Weight": f"{c['biomarker_weight']:.2f}",
                "Context Weight": f"{c['context_weight']:.2f}",
                "Effective Weight": f"{c['effective_weight']:.4f}",
                "Contribution": f"{c['contribution']:.4f}",
            })
        else:
            contrib_table.append({
                "Biomarker": c["biomarker"],
                "Directed Membership": f"{c['directed_membership']:.4f}",
                "Biomarker Weight": f"{c['biomarker_weight']:.2f}",
                "Context Weight": f"{c['context_weight']:.2f}",
                "Effective Weight": f"{c['effective_weight']:.4f}",
                "Contribution": f"{c['contribution']:.4f}",
            })

    st.dataframe(contrib_table, use_container_width=True, hide_index=True)

    # Amplification badges
    amplified = [c["biomarker"] for c in contributions if c.get("context_weight", 1.0) > 1.0]
    dampened = [c["biomarker"] for c in contributions if c.get("context_weight", 1.0) < 1.0]

    if amplified:
        st.markdown(f"**Amplified in {dominant_context}:** {', '.join(amplified)}")
    if dampened:
        st.markdown(f"**Dampened in {dominant_context}:** {', '.join(dampened)}")
    if not amplified and not dampened:
        st.info("All biomarkers at neutral context weight (1.0).")

    # Result
    st.markdown("#### Step 6: Result")
    st.caption(
        "Step 5 returned the biomarker memberships per window. For every window, combine the biomarker memberships into the respective indicators using FASL. Output is a window-level likelihood per indicator. Note: There is only one result as the window and indicator were selected in step 0."
    )
    st.dataframe(
        [{"Indicator": option.indicator_name, "Window": selected_window_label, "Indicator Score": f"{indicator_score:.4f}"}],
        use_container_width=True,
        hide_index=True,
    )

    # Detailed computation toggle
    show_details = st.toggle("Show detailed computations", value=False, key=f"{session_prefix}step6_show_details")
    if show_details:
        st.markdown("#### Per-Biomarker FASL Computation Details")
        for c in contributions:
            with st.container():
                st.markdown(f"**{c['biomarker']}**")
                if c.get("is_missing"):
                    st.warning("Missing data - neutral fill (0.5) used")
                    st.divider()
                    continue

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Weights:**")
                    st.code(f"biomarker_weight = {c['biomarker_weight']:.4f}\ncontext_weight = {c['context_weight']:.4f}\nconfidence = {context_confidence:.4f}")

                with col2:
                    st.markdown("**Effective Context Weight:**")
                    eff_ctx = 1.0 + (c["context_weight"] - 1.0) * context_confidence
                    st.code(f"eff_ctx = 1.0 + (ctx_weight - 1.0) x confidence\neff_ctx = 1.0 + ({c['context_weight']:.4f} - 1.0) x {context_confidence:.4f}\neff_ctx = {eff_ctx:.4f}")

                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**Effective Weight:**")
                    st.code(f"eff_weight = bio_weight x eff_ctx\neff_weight = {c['biomarker_weight']:.4f} x {eff_ctx:.4f}\neff_weight = {c['effective_weight']:.4f}")

                with col4:
                    st.markdown("**Contribution:**")
                    st.code(f"contribution = directed_mem x eff_weight\ncontribution = {c['directed_membership']:.4f} x {c['effective_weight']:.4f}\ncontribution = {c['contribution']:.4f}")

                st.divider()

        # Final FASL aggregation
        st.markdown("#### Final FASL Aggregation")
        total_contribution = sum(c["contribution"] for c in contributions)
        total_eff_weight = sum(c["effective_weight"] for c in contributions)
        st.code(f"sum(contributions) = {total_contribution:.4f}\nsum(effective_weights) = {total_eff_weight:.4f}\n\nindicator_score = sum(contributions) / sum(effective_weights)\nindicator_score = {total_contribution:.4f} / {total_eff_weight:.4f}\nindicator_score = {indicator_score:.4f}")


def _render_step_6_from_recomputation(
    user_id: str,
    option: "DailyIndicatorOption",
    selected_window_start,
    selected_window_label: str,
    session_prefix: str,
) -> None:
    """Fallback: render Step 6 by recomputing FASL (pre-6.17 data)."""
    fasl = compute_fasl_for_display(
        user_id=user_id,
        target_date=option.indicator_date,
        indicator_name=option.indicator_name,
        window_start=selected_window_start,
    )

    if fasl is None:
        st.warning("Could not compute FASL for this window.")
        return

    # Delegate to persisted renderer using dict format
    details = {
        "dominant_context": fasl.dominant_context,
        "context_confidence": fasl.context_confidence,
        "indicator_score": fasl.indicator_score,
        "fasl_contributions": fasl.contributions,
    }
    _render_step_6_from_persisted(details, option, selected_window_label, session_prefix)


def render_step_7_daily_summary(summary: dict, session_prefix: str = "") -> None:
    """Render Step 7: Daily Summary Computation.

    Displays daily likelihood (simple mean of window FASL scores)
    and quality metrics. Optionally shows all individual window scores
    and the step-by-step mean computation.

    Args:
        summary: Daily indicator summary dict from computation_log
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    with st.expander("Step 7: Daily Summary Computation", expanded=False):
        quality = summary.get("quality", {})
        likelihood = summary.get("likelihood", summary.get("value", 0))

        # Show selected indicator and day context prominently
        indicator_name = st.session_state.get(
            f"{session_prefix}current_indicator_name", "Unknown"
        )
        indicator_date = st.session_state.get(f"{session_prefix}current_indicator_date")
        date_str = indicator_date.strftime("%Y-%m-%d") if indicator_date else "Unknown"
        weekday = indicator_date.strftime("%A") if indicator_date else ""

        st.markdown(
            f"""
<div style="
    text-align: center;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 0.75rem;
    background: linear-gradient(135deg, #3b82f615, #3b82f605);
    border: 1px solid #3b82f630;
">
    <div style="font-size: 0.85rem; color: #888; margin-bottom: 0.25rem;">Selected Indicator</div>
    <div style="font-size: 1.5rem; font-weight: 600; color: #3b82f6; margin-bottom: 0.5rem;">{indicator_name}</div>
    <div style="font-size: 0.85rem; color: #888;">Day</div>
    <div style="font-size: 1.1rem; font-weight: 500; color: #374151;">{date_str} <span style="color: #6b7280;">({weekday})</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

        # --- Daily Likelihood Result ---
        st.markdown("#### Daily Likelihood")
        st.caption(
            "Simple mean of all window-level FASL indicator scores for this day."
        )

        st.markdown(
            f"""
<div style="
    text-align: center;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 0.5rem;
    background: linear-gradient(135deg, #3b82f615, #3b82f605);
    border: 1px solid #3b82f630;
">
    <div style="font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">Daily Likelihood</div>
    <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">{likelihood:.3f}</div>
    <div style="font-size: 0.75rem; color: #888; margin-top: 0.25rem;">= mean(window FASL scores)</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Detailed Likelihood Computation ---
        show_likelihood_details = st.toggle(
            "Show detailed computation",
            value=False,
            key=f"{session_prefix}step7_likelihood_details",
        )

        if show_likelihood_details:
            window_scores = summary.get("window_scores", [])

            if window_scores:
                st.markdown("#### Window FASL Scores")
                st.caption(
                    f"All {len(window_scores)} window-level FASL scores for "
                    f"**{indicator_name}** on {date_str}."
                )

                # Build table data from stored window scores
                scores_table = []
                score_values = []
                for i, ws in enumerate(window_scores):
                    score = ws["score"]
                    score_values.append(score)
                    scores_table.append({
                        "#": i + 1,
                        "Window": ws["window_start"],
                        "FASL Score": f"{score:.4f}",
                        "Context": ws["context"],
                    })

                st.dataframe(scores_table, use_container_width=True, hide_index=True)

                # Show the actual mean computation
                st.markdown("#### Mean Computation")
                total = sum(score_values)
                n = len(score_values)
                computed_mean = total / n if n > 0 else 0.0

                # Build summation string showing all actual values
                values_str = " + ".join(f"{s:.4f}" for s in score_values)
                computation_str = (
                    f"sum = {values_str}\n"
                    f"sum = {total:.4f}\n\n"
                    f"n = {n}\n\n"
                    f"likelihood = sum / n\n"
                    f"likelihood = {total:.4f} / {n}\n"
                    f"likelihood = {computed_mean:.4f}"
                )

                st.code(computation_str)
            else:
                st.info("No window scores stored for this indicator/date. Re-run analysis to populate.")

        st.divider()

        # --- Quality Metrics ---
        st.markdown("#### Quality Metrics")
        st.markdown("*How complete and reliable is this data?*")

        expected_windows = quality.get("expected_windows", 96)
        total_windows = quality.get("total_windows", 0)
        data_coverage = quality.get("data_coverage", 0)
        completeness = quality.get("average_biomarker_completeness", 0)
        context_avail = quality.get("context_availability", 0)

        quality_table = [
            {
                "Metric": "Expected Windows",
                "Value": str(expected_windows),
                "Status": "-",
            },
            {
                "Metric": "Actual Windows",
                "Value": str(total_windows),
                "Status": "OK" if total_windows >= expected_windows * 0.7 else "Low",
            },
            {
                "Metric": "Data Coverage",
                "Value": f"{data_coverage:.1%}",
                "Status": "OK" if data_coverage >= 0.7 else "Low",
            },
            {
                "Metric": "Biomarker Completeness",
                "Value": f"{completeness:.1%}",
                "Status": "OK" if completeness >= 0.7 else "Low",
            },
            {
                "Metric": "Context Availability",
                "Value": f"{context_avail:.1%}",
                "Status": "OK" if context_avail >= 0.5 else "Low",
            },
        ]
        st.dataframe(quality_table, use_container_width=True, hide_index=True)

        # Show detailed computation for quality metrics
        show_quality_details = st.toggle(
            "Show quality computation details",
            value=False,
            key=f"{session_prefix}step7_quality_details",
        )

        if show_quality_details:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Data Coverage:**")
                st.code(
                    f"expected_windows = 24h / 15min = {expected_windows}\nactual_windows = {total_windows}\n\ndata_coverage = actual / expected\ndata_coverage = {total_windows} / {expected_windows}\ndata_coverage = {data_coverage:.4f} ({data_coverage:.1%})"
                )

            with col2:
                st.markdown("**Biomarker Completeness:**")
                st.code(
                    f"For each window:\n  completeness = biomarkers_present / biomarkers_expected\n\naverage_completeness = mean(window_completeness)\naverage_completeness = {completeness:.4f} ({completeness:.1%})"
                )

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Context Availability:**")
                st.code(
                    f"windows_with_context = count(context != 'neutral')\ntotal_windows = {total_windows}\n\ncontext_availability = windows_with_context / total\ncontext_availability = {context_avail:.4f} ({context_avail:.1%})"
                )

            with col4:
                st.markdown("**Quality Assessment:**")
                issues = []
                if data_coverage < 0.7:
                    issues.append(f"Low data coverage ({data_coverage:.0%})")
                if completeness < 0.7:
                    issues.append(f"Low biomarker completeness ({completeness:.0%})")
                if context_avail < 0.5:
                    issues.append(f"Limited context data ({context_avail:.0%})")

                if issues:
                    st.warning("**Issues:**\n- " + "\n- ".join(issues))
                else:
                    st.success("All quality metrics within acceptable range")

        st.divider()

        # --- Formula Reference ---
        st.markdown("#### Formula")
        st.latex(r"\text{likelihood}_{\text{daily}} = \frac{1}{n} \sum_{k=1}^{n} L_k")
        st.markdown("Where $L_k$ = window-level FASL indicator score from Step 6")


def render_config_summary(
    user_id: str,
    indicator_name: str,
    session_prefix: str = "",
) -> None:
    """Render Configuration Summary section.

    AC8: Configuration Summary Section
    - Collapsible section at bottom showing all configs used
    - Include indicator config from config/indicators.yaml
    - Include context weights from config/context_weights.yaml
    - Show baseline values (user-specific or population default)
    - Show daily summary parameters from config/analysis.yaml
    - Show window configuration

    Args:
        user_id: User ID for baseline lookup
        indicator_name: Indicator name for config lookup
        session_prefix: Prefix for session state keys for side-by-side mode
    """
    from src.dashboard.data.indicator_transparency import (
        get_baselines_for_user,
        get_biomarker_defaults,
        get_context_weights_config,
        get_indicator_config,
    )

    with st.expander("Configuration Summary", expanded=False):
        st.markdown(
            """
**All configurations used in this computation.**

This section shows the exact configuration values used for computing this indicator.
"""
        )

        # --- Indicator Configuration ---
        st.markdown("#### Indicator Configuration")
        st.caption(f"Source: `config/indicators.yaml` → `{indicator_name}`")

        indicator_config = get_indicator_config(indicator_name)
        if indicator_config:
            biomarkers = indicator_config.get("biomarkers", {})
            config_table = []
            for bio_name, bio_config in biomarkers.items():
                config_table.append(
                    {
                        "Biomarker": bio_name,
                        "Weight": f"{bio_config.get('weight', 0):.2f}",
                        "Direction": bio_config.get("direction", "N/A"),
                    }
                )
            st.dataframe(config_table, use_container_width=True, hide_index=True)
        else:
            st.warning(f"Indicator config not found for: {indicator_name}")

        st.divider()

        # --- Context Weights Configuration ---
        st.markdown("#### Context Weights Configuration")
        st.caption("Source: `config/context_weights.yaml`")

        context_weights = get_context_weights_config()
        if context_weights:
            for context_name, weights in context_weights.items():
                st.markdown(f"**{context_name}:**")
                weights_table = [
                    {"Biomarker": bio, "Weight": f"{w:.2f}"}
                    for bio, w in weights.items()
                ]
                st.dataframe(weights_table, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No context weights configured - all biomarkers use neutral (1.0) weight."
            )

        st.divider()

        # --- Baseline Values ---
        st.markdown("#### Baseline Values")
        st.caption(f"User: `{user_id}`")

        user_baselines = get_baselines_for_user(user_id)
        defaults = get_biomarker_defaults()

        if user_baselines:
            st.markdown("**User-Specific Baselines:**")
            baseline_table = []
            for bio_name, baseline in user_baselines.items():
                baseline_table.append(
                    {
                        "Biomarker": bio_name,
                        "Mean": f"{baseline['mean']:.4f}",
                        "Std": f"{baseline['std']:.4f}",
                        "Data Points": baseline.get("data_points", "N/A"),
                    }
                )
            st.dataframe(baseline_table, use_container_width=True, hide_index=True)
        else:
            st.info("No user-specific baselines found.")

        if defaults:
            st.markdown("**Population Defaults (fallback):**")
            st.caption("Source: `config/biomarker_defaults.yaml`")
            defaults_table = [
                {
                    "Biomarker": bio,
                    "Mean": f"{d.get('mean', 0.5):.4f}",
                    "Std": f"{d.get('std', 0.1):.4f}",
                }
                for bio, d in defaults.items()
            ]
            st.dataframe(defaults_table, use_container_width=True, hide_index=True)

        st.divider()

        # --- Window Configuration ---
        st.markdown("#### Window Configuration")
        st.caption("Source: `config/analysis.yaml`")

        st.code(
            """window_size_minutes: 15
aggregation_method: mean
min_readings_per_window: 1
expected_windows_per_day: 96

# FASL Settings
missing_biomarker_strategy: neutral_fill
neutral_membership: 0.5"""
        )

        st.divider()

        # --- Daily Summary Parameters ---
        st.markdown("#### Daily Summary Parameters")
        st.caption("Source: Simple mean of window FASL scores")

        st.code(
            """# Daily Likelihood
method: simple_mean  # mean of window-level FASL scores
# No tuning parameters required"""
        )
