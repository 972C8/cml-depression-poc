"""Comparison component for comparing analysis runs."""

import pandas as pd
import streamlit as st

from src.dashboard.components.filters import get_display_timezone
from src.dashboard.data.analysis import get_analysis_run_summary


def _format_timestamp(ts: pd.Timestamp) -> str:
    """Convert timestamp to display timezone and format."""
    tz = get_display_timezone()
    if ts.tzinfo is not None:
        local_ts = ts.tz_convert(tz)
    else:
        local_ts = ts.tz_localize(tz)
    return local_ts.strftime("%Y-%m-%d %H:%M")


def render_run_comparison(run_id_1: str, run_id_2: str) -> None:
    """Render side-by-side comparison of two analysis runs.

    .. deprecated:: Story 6.12
        This function is deprecated. Side-by-side comparison is now integrated
        directly into the results section using render_results_overview() with
        a comparison run selector. See Story 6.12 for details.

    Args:
        run_id_1: First analysis run ID (baseline)
        run_id_2: Second analysis run ID (experiment)
    """
    run1 = get_analysis_run_summary(run_id_1)
    run2 = get_analysis_run_summary(run_id_2)

    if run1 is None or run2 is None:
        st.error("Could not load one or both analysis runs")
        return

    st.markdown("### Analysis Run Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Baseline:** `{run_id_1[:8]}...`")
        st.caption(f"Created: {run1['created_at']}")

    with col2:
        st.markdown(f"**Experiment:** `{run_id_2[:8]}...`")
        st.caption(f"Created: {run2['created_at']}")

    # Extract data from pipeline traces
    trace1 = run1.get("pipeline_trace", {}) or {}
    trace2 = run2.get("pipeline_trace", {}) or {}

    def extract_step_outputs(trace: dict, step_name: str) -> dict:
        """Extract outputs from a specific pipeline step."""
        steps = trace.get("steps", [])
        for step in steps:
            if step.get("step_name") == step_name:
                return step.get("outputs", {}) or {}
        return {}

    # --- Episode Decision Comparison ---
    episode1 = extract_step_outputs(trace1, "Episode Decision")
    episode2 = extract_step_outputs(trace2, "Episode Decision")

    if episode1 and episode2:
        st.markdown("#### Episode Decision")

        col1, col2 = st.columns(2)

        with col1:
            likely1 = episode1.get("episode_likely", False)
            present1 = episode1.get("indicators_present", 0)
            required1 = episode1.get("min_indicators_required", 0)
            status1 = "🔴 Likely" if likely1 else "🟢 Not Likely"
            st.metric("Baseline", status1, f"{present1}/{required1} indicators")
            if episode1.get("rationale"):
                st.caption(episode1["rationale"])

        with col2:
            likely2 = episode2.get("episode_likely", False)
            present2 = episode2.get("indicators_present", 0)
            required2 = episode2.get("min_indicators_required", 0)
            status2 = "🔴 Likely" if likely2 else "🟢 Not Likely"
            st.metric("Experiment", status2, f"{present2}/{required2} indicators")
            if episode2.get("rationale"):
                st.caption(episode2["rationale"])

    # --- Context Comparison ---
    context1 = extract_step_outputs(trace1, "Evaluate Context")
    context2 = extract_step_outputs(trace2, "Evaluate Context")

    if context1 and context2:
        st.markdown("#### Context Detected")

        col1, col2 = st.columns(2)

        with col1:
            active1 = context1.get("active_context", "unknown")
            confidence1 = context1.get("confidence_scores", {}).get(active1, 0)
            st.metric("Baseline", active1, f"confidence: {confidence1:.2f}")

        with col2:
            active2 = context2.get("active_context", "unknown")
            confidence2 = context2.get("confidence_scores", {}).get(active2, 0)
            st.metric("Experiment", active2, f"confidence: {confidence2:.2f}")

    # --- DSM-Gate Results Comparison ---
    gate1 = extract_step_outputs(trace1, "Apply DSM-Gate")
    gate2 = extract_step_outputs(trace2, "Apply DSM-Gate")

    if gate1 and gate2:
        gate_results1 = gate1.get("gate_results", {})
        gate_results2 = gate2.get("gate_results", {})
        dsm1 = gate1.get("dsm_params", {})
        dsm2 = gate2.get("dsm_params", {})
        gate_need1 = dsm1.get("gate_need", dsm1.get("m_window", 14))
        gate_need2 = dsm2.get("gate_need", dsm2.get("m_window", 14))

        if gate_results1 and gate_results2:
            st.markdown("#### DSM-Gate Results")

            comparison_data = []
            all_indicators = set(gate_results1.keys()) | set(gate_results2.keys())

            for ind_name in sorted(all_indicators):
                r1 = gate_results1.get(ind_name, {})
                r2 = gate_results2.get(ind_name, {})

                present1 = "✓" if r1.get("presence_flag") else "✗"
                present2 = "✓" if r2.get("presence_flag") else "✗"
                days1 = r1.get("days_above_threshold", 0)
                days2 = r2.get("days_above_threshold", 0)
                eval1 = r1.get("days_evaluated", 0)
                eval2 = r2.get("days_evaluated", 0)
                threshold = r1.get("threshold", r2.get("threshold", 0))

                comparison_data.append(
                    {
                        "Indicator": ind_name,
                        "Baseline": f"{present1} ({days1}/{eval1} days, need ≥{gate_need1})",
                        "Experiment": f"{present2} ({days2}/{eval2} days, need ≥{gate_need2})",
                        "Threshold": f"{threshold:.2f}",
                    }
                )

            df = pd.DataFrame(comparison_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

    # --- Indicator Scores Comparison ---
    indicators1 = extract_step_outputs(trace1, "Compute Indicators").get(
        "indicator_scores", {}
    )
    indicators2 = extract_step_outputs(trace2, "Compute Indicators").get(
        "indicator_scores", {}
    )

    if indicators1 and indicators2:
        st.markdown("#### Indicator Scores")

        comparison_data = []
        all_indicators = set(indicators1.keys()) | set(indicators2.keys())

        for ind_name in sorted(all_indicators):
            score1 = indicators1.get(ind_name, {}).get("daily_likelihood", 0)
            score2 = indicators2.get(ind_name, {}).get("daily_likelihood", 0)
            delta = score2 - score1

            comparison_data.append(
                {
                    "Indicator": ind_name,
                    "Baseline": f"{score1:.3f}",
                    "Experiment": f"{score2:.3f}",
                    "Delta": f"{delta:+.3f}",
                }
            )

        df = pd.DataFrame(comparison_data)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Check if any comparison data was shown
    if not any(
        [episode1, episode2, context1, context2, gate1, gate2, indicators1, indicators2]
    ):
        st.info("No pipeline trace data available for comparison")

    # Config diff
    with st.expander("Configuration Differences", expanded=False):
        config1 = run1.get("config_snapshot", {}) or {}
        config2 = run2.get("config_snapshot", {}) or {}

        has_differences = False

        # Show baseline diff (Story 4.14 AC7)
        baseline1 = config1.get("baseline", {})
        baseline2 = config2.get("baseline", {})

        if baseline1 != baseline2:
            has_differences = True
            st.markdown("**Baseline:**")
            st.json({"baseline_run": baseline1, "experiment_run": baseline2})

        # Show DSM-gate config diff
        gate_config1 = config1.get("dsm_gate_defaults", {})
        gate_config2 = config2.get("dsm_gate_defaults", {})

        if gate_config1 != gate_config2:
            has_differences = True
            st.markdown("**DSM-Gate:**")
            st.json({"baseline": gate_config1, "experiment": gate_config2})

        # Show EMA diff
        ema1 = config1.get("ema", {})
        ema2 = config2.get("ema", {})

        if ema1 != ema2:
            has_differences = True
            st.markdown("**EMA:**")
            st.json({"baseline": ema1, "experiment": ema2})

        if not has_differences:
            st.info("No significant configuration differences detected")


def render_comparison_selector(user_id: str) -> tuple[str | None, str | None]:
    """Render selector for choosing runs to compare.

    .. deprecated:: Story 6.12
        This function is deprecated. Comparison run selection is now integrated
        directly below the primary run selector in the Analysis page results
        section. See Story 6.12 for details.

    Args:
        user_id: User ID to filter runs by

    Returns:
        Tuple of (run_id_1, run_id_2) or (None, None) if not selected
    """
    from src.dashboard.data.analysis import load_analysis_runs

    runs_df = load_analysis_runs(user_id=user_id, limit=20)

    if runs_df.empty or len(runs_df) < 2:
        st.info("Need at least 2 analysis runs to compare")
        return None, None

    # Build options
    options = {
        row[
            "run_id"
        ]: f"{row['run_id'][:8]}... ({_format_timestamp(row['created_at'])})"
        for _, row in runs_df.iterrows()
    }

    col1, col2 = st.columns(2)

    with col1:
        run_id_1 = st.selectbox(
            "Baseline Run",
            options=list(options.keys()),
            format_func=lambda x: options[x],
            key="comparison_baseline",
        )

    with col2:
        # Filter out the baseline from options
        remaining_options = {k: v for k, v in options.items() if k != run_id_1}
        if not remaining_options:
            st.warning("Only one run available - cannot compare")
            return None, None

        run_id_2 = st.selectbox(
            "Experiment Run",
            options=list(remaining_options.keys()),
            format_func=lambda x: remaining_options[x],
            key="comparison_experiment",
        )

    return run_id_1, run_id_2
