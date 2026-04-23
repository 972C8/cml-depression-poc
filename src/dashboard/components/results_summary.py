"""Results summary component for displaying analysis results.

Renders analysis results including:
- Windowed pipeline: likelihood metrics, daily summaries, data quality
- Legacy pipeline: episode determination with DSM-gate results
- Results overview: episode decision based on DSM-gate criteria
"""

import altair as alt
import pandas as pd
import streamlit as st

# Threshold for determining if indicator shows concern
CONCERN_THRESHOLD = 0.5

# Default DSM-gate parameters
DEFAULT_THETA = 0.5
DEFAULT_M_WINDOW = 14
DEFAULT_MIN_INDICATORS = 5
DEFAULT_CORE_INDICATORS = ("1_depressed_mood", "2_loss_of_interest")


def _extract_indicator_daily_likelihoods(
    pipeline_trace: dict | None,
) -> dict[str, list[dict]]:
    """Extract daily likelihood values per indicator from pipeline trace.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict mapping indicator name to list of daily summary dicts,
        each containing 'date' and 'likelihood' keys
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return {}

    indicator_dailies: dict[str, list[dict]] = {}

    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Daily Aggregation":
            outputs = step.get("outputs", {})
            indicator_name = outputs.get("indicator_name", "unknown")
            daily_summaries = outputs.get("daily_summaries", [])

            indicator_dailies[indicator_name] = daily_summaries

    return indicator_dailies


def _format_indicator_name(name: str) -> str:
    """Format indicator name for display (snake_case to Title Case)."""
    return name.replace("_", " ").title()


def _extract_stored_episode_decision(pipeline_trace: dict | None) -> dict | None:
    """Extract stored episode decision from pipeline trace.

    Looks for the "Episode Decision" step added in the windowed pipeline
    that stores pre-computed DSM-gate results.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict with episode decision data or None if not found
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return None

    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Episode Decision":
            outputs = step.get("outputs", {})
            # Validate required fields are present
            if "episode_likely" in outputs and "gate_results" in outputs:
                return {
                    "episode_likely": outputs.get("episode_likely", False),
                    "indicators_present": outputs.get("indicators_present", 0),
                    "min_indicators_required": outputs.get(
                        "min_indicators_required", 5
                    ),
                    "core_indicator_present": outputs.get(
                        "core_indicator_present", False
                    ),
                    "core_indicators_present": outputs.get(
                        "core_indicators_present", []
                    ),
                    "core_indicators_required": tuple(
                        outputs.get("core_indicators_required", DEFAULT_CORE_INDICATORS)
                    ),
                    "gate_results": outputs.get("gate_results", {}),
                    "dsm_params": outputs.get(
                        "dsm_params",
                        {
                            "theta": DEFAULT_THETA,
                            "m_window": DEFAULT_M_WINDOW,
                            "gate_need": DEFAULT_M_WINDOW,
                        },
                    ),
                    "decision_rationale": outputs.get("decision_rationale", ""),
                }

    return None


def render_results_overview(
    pipeline_trace: dict | None,
    session_prefix: str = "",
) -> None:
    """Render results overview with episode decision and indicator status.

    Displays a visually appealing overview including:
    - Episode decision status with prominent visual indicator
    - Core symptoms and criteria satisfaction status
    - Per-indicator breakdown with progress visualization
    - Mean daily likelihood metrics

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun.pipeline_trace
        session_prefix: Prefix for session state keys to support side-by-side
            comparison mode. Use "primary_" or "comparison_" when in
            side-by-side mode. Default "" for single-column mode.
    """
    if not _is_windowed_pipeline(pipeline_trace):
        # Skip overview for legacy pipeline - handled by render_episode_summary
        return

    decision = _extract_stored_episode_decision(pipeline_trace)
    if decision is None:
        st.warning("Episode decision not found in pipeline trace.")
        return

    episode_likely = decision["episode_likely"]
    indicators_present = decision["indicators_present"]
    min_required = decision["min_indicators_required"]
    core_satisfied = decision["core_indicator_present"]
    core_present = decision["core_indicators_present"]
    core_required = decision["core_indicators_required"]
    gate_results = decision["gate_results"]
    dsm_params = decision["dsm_params"]

    m_window = dsm_params["m_window"]
    theta = dsm_params["theta"]
    gate_need = dsm_params.get("gate_need", m_window)
    total_indicators = len(gate_results)

    # === EPISODE DECISION HERO CARD ===
    if episode_likely:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                        border-radius: 12px; padding: 20px; margin-bottom: 20px;
                        border-left: 5px solid #28a745;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 48px;">✓</span>
                    <div>
                        <div style="font-size: 24px; font-weight: 700; color: #155724;">
                            Episode Likely
                        </div>
                        <div style="font-size: 14px; color: #155724; opacity: 0.8;">
                            DSM-5 criteria satisfied
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Build reason text
        reasons = []
        if indicators_present < min_required:
            reasons.append(f"{indicators_present}/{min_required} indicators")
        if not core_satisfied:
            reasons.append("no core symptom")
        reason_text = " · ".join(reasons) if reasons else "Criteria not met"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                        border-radius: 12px; padding: 20px; margin-bottom: 20px;
                        border-left: 5px solid #6c757d;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 48px; opacity: 0.5;">○</span>
                    <div>
                        <div style="font-size: 24px; font-weight: 700; color: #495057;">
                            Episode Not Likely
                        </div>
                        <div style="font-size: 14px; color: #6c757d;">
                            {reason_text}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # === CONTEXT SOURCE INFO (Story 6.14 AC5) ===
    context_source_info = _extract_context_source(pipeline_trace)
    if context_source_info:
        source = context_source_info.get("context_source", "unknown")
        if source == "selected_run":
            run_id = context_source_info.get("context_evaluation_run_id", "")
            run_id_short = run_id[:8] if run_id else "N/A"
            coverage = context_source_info.get("coverage_ratio", 0.0)
            dates_missing = context_source_info.get("dates_missing", 0)

            if dates_missing > 0:
                coverage_text = f"{coverage:.0%} coverage · {dates_missing} date(s) use neutral weights"
                icon = "⚠️"
            else:
                coverage_text = f"{coverage:.0%} coverage"
                icon = "🎯"

            st.markdown(
                f"""
                <div style="
                    background: #e3f2fd;
                    border-radius: 8px;
                    padding: 10px 15px;
                    margin-bottom: 15px;
                    font-size: 13px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">
                    <span style="font-size: 16px;">{icon}</span>
                    <div>
                        <strong>Context Source:</strong> Selected Run ({run_id_short}...)
                        <span style="color: #666; margin-left: 8px;">{coverage_text}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Auto-generated context
            evals_added = context_source_info.get("evaluations_added", 0)
            evals_text = f"{evals_added} evaluations added" if evals_added > 0 else "using existing data"

            st.markdown(
                f"""
                <div style="
                    background: #f5f5f5;
                    border-radius: 8px;
                    padding: 10px 15px;
                    margin-bottom: 15px;
                    font-size: 13px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                ">
                    <span style="font-size: 16px;">🔄</span>
                    <div>
                        <strong>Context Source:</strong> Auto-generated
                        <span style="color: #666; margin-left: 8px;">{evals_text}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # === CRITERIA STATUS CARDS ===
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            indicators_met = indicators_present >= min_required
            status_icon = "✓" if indicators_met else "✗"
            status_color = "#28a745" if indicators_met else "#dc3545"

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 14px; color: #6c757d; margin-bottom: 4px;">
                        Indicators Present
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: {status_color};">
                        {indicators_present}/{total_indicators}
                    </div>
                    <div style="font-size: 12px; color: #6c757d;">
                        {status_icon} Need {min_required}+ for episode
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Visual progress bar
            progress = (
                indicators_present / total_indicators if total_indicators > 0 else 0
            )
            st.progress(progress)

    with col2:
        with st.container(border=True):
            status_icon = "✓" if core_satisfied else "✗"
            status_color = "#28a745" if core_satisfied else "#dc3545"
            core_text = (
                ", ".join(_format_indicator_name(c) for c in core_present)
                if core_present
                else "None"
            )

            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 14px; color: #6c757d; margin-bottom: 4px;">
                        Core Symptoms
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: {status_color};">
                        {"Yes" if core_satisfied else "No"}
                    </div>
                    <div style="font-size: 12px; color: #6c757d;">
                        {status_icon} {core_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Show which core indicators are required
            st.caption(
                f"Requires at least 1 of: {', '.join(_format_indicator_name(c) for c in core_required)}"
            )

    # === INDICATOR BREAKDOWN ===
    st.markdown("")  # Spacer
    st.markdown("#### Indicator Details")
    st.caption(
        f"An indicator is present if above **{theta:.0%}** threshold on "
        f"≥**{gate_need}** of the last **{m_window}** days. "
        f"Episode requires ≥{min_required} present indicators, "
        f"including at least one core indicator."
    )

    # Get daily likelihood data for line charts
    indicator_dailies = _extract_indicator_daily_likelihoods(pipeline_trace)

    # Build indicator data sorted by mean likelihood (highest first)
    indicator_data = []
    for indicator_name, result in gate_results.items():
        is_core = indicator_name in core_required

        # Extract daily likelihood values for the chart
        daily_summaries = indicator_dailies.get(indicator_name, [])
        sorted_summaries = sorted(daily_summaries, key=lambda x: x.get("date", ""))
        window_summaries = sorted_summaries[-m_window:]
        daily_values = [s.get("likelihood", 0.0) for s in window_summaries]
        daily_dates = [s.get("date", "")[-5:] for s in window_summaries]  # MM-DD format

        indicator_data.append(
            {
                "name": indicator_name,
                "display_name": _format_indicator_name(indicator_name),
                "is_core": is_core,
                "presence": result.get("presence_flag", False),
                "days_above": result["days_above_threshold"],
                "days_eval": result["days_evaluated"],
                "gate_need": gate_need,
                "mean_likelihood": result["mean_likelihood"],
                "insufficient_data": result["insufficient_data"],
                "daily_values": daily_values,
                "daily_dates": daily_dates,
            }
        )

    # Sort by indicator name (numeric prefix: 1_depressed_mood, 2_..., 9_suicidality)
    indicator_data.sort(key=lambda x: x["name"])

    # Display in a grid layout
    cols = st.columns(3)
    for idx, ind in enumerate(indicator_data):
        col = cols[idx % 3]

        with col:
            # Card styling based on presence
            if ind["presence"]:
                status_badge = "✓ Present"
                badge_color = "#28a745"
            else:
                status_badge = "○ Absent"
                badge_color = "#6c757d"

            # Core indicator badge
            core_badge = " ★" if ind["is_core"] else ""

            # Indicator is "present" if above threshold on >= gate_need days
            days_met = ind["presence"]

            with st.container(border=True):
                # Header with name and status
                st.markdown(
                    f"""
                    <div style="margin-bottom: 8px;">
                        <span style="font-weight: 600; font-size: 14px;">
                            {ind["display_name"]}{core_badge}
                        </span>
                        <span style="float: right; font-size: 12px; color: {badge_color};">
                            {status_badge}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Mean likelihood with line chart
                likelihood_color = (
                    "#28a745" if ind["mean_likelihood"] >= theta else "#ffc107"
                )
                st.markdown(
                    f"""
                    <div style="font-size: 12px; color: #6c757d; margin-bottom: 2px;">
                        Mean Likelihood
                    </div>
                    <div style="font-size: 20px; font-weight: 600; color: {likelihood_color};">
                        {ind["mean_likelihood"]:.1%}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Line chart showing daily likelihood values
                if ind["daily_values"]:
                    chart_data = pd.DataFrame(
                        {
                            "date": ind["daily_dates"],
                            "likelihood": ind["daily_values"],
                        }
                    )

                    # Line chart for daily values
                    line = (
                        alt.Chart(chart_data)
                        .mark_line(point=True, strokeWidth=2)
                        .encode(
                            x=alt.X("date:N", axis=alt.Axis(labels=False, title=None)),
                            y=alt.Y(
                                "likelihood:Q",
                                scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(title=None, tickCount=3),
                            ),
                        )
                    )

                    # Horizontal threshold line (theta from config)
                    threshold_line = (
                        alt.Chart(pd.DataFrame({"threshold": [theta]}))
                        .mark_rule(strokeDash=[4, 4], strokeWidth=1, color="#dc3545")
                        .encode(y="threshold:Q")
                    )

                    chart = (
                        (line + threshold_line)
                        .properties(height=80)
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(chart, use_container_width=True)

                # Days above threshold (N-of-M rule: need >= gate_need of m_window)
                days_color = "#28a745" if days_met else "#dc3545"
                insufficient_note = " *" if ind["insufficient_data"] else ""
                st.markdown(
                    f"""
                    <div style="font-size: 12px; color: #6c757d; margin-top: 8px;">
                        Days above threshold (need ≥{ind["gate_need"]})
                    </div>
                    <div style="font-size: 14px; color: {days_color};">
                        <strong>{ind["days_above"]}</strong> / {ind["days_eval"]} days{insufficient_note}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Footnotes
    st.markdown("")  # Spacer
    has_insufficient = any(d["insufficient_data"] for d in indicator_data)
    has_core = any(d["is_core"] for d in indicator_data)

    footnotes = []
    if has_core:
        footnotes.append("★ = Core symptom (at least one required)")
    if has_insufficient:
        footnotes.append(f"* = Insufficient data (fewer than {m_window} days)")

    if footnotes:
        st.caption(" · ".join(footnotes))


def _is_windowed_pipeline(pipeline_trace: dict | None) -> bool:
    """Check if this is a windowed analysis pipeline run.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        True if windowed pipeline, False otherwise
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return False

    step_names = {step.get("step_name") for step in pipeline_trace["steps"]}
    # Windowed pipeline has "Daily Aggregation" step
    return "Daily Aggregation" in step_names


def _extract_windowed_summary(pipeline_trace: dict | None) -> dict | None:
    """Extract summary data from windowed pipeline trace.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict with windowed pipeline summary or None
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return None

    summary = {
        "window_count": 0,
        "daily_summaries_count": 0,
        "indicators": {},
        "context_evaluations_added": 0,
        "biomarker_count": 0,
    }

    for step in pipeline_trace["steps"]:
        step_name = step.get("step_name")
        outputs = step.get("outputs", {})

        if step_name == "Context History Population":
            summary["context_evaluations_added"] = outputs.get("evaluations_added", 0)

        elif step_name == "Read Data":
            summary["biomarker_count"] = outputs.get("biomarker_count", 0)

        elif step_name == "Window Aggregation":
            summary["window_count"] = outputs.get("window_count", 0)

        elif step_name == "Daily Aggregation":
            indicator_name = outputs.get("indicator_name", "unknown")
            summary["indicators"][indicator_name] = {
                "daily_summaries_count": outputs.get("daily_summaries_count", 0),
                "dates_processed": outputs.get("dates_processed", []),
            }
            summary["daily_summaries_count"] += outputs.get("daily_summaries_count", 0)

    return summary if summary["window_count"] > 0 else None


def _extract_episode_decision(pipeline_trace: dict | None) -> dict | None:
    """Extract episode decision data from pipeline trace.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict with episode decision data or None if not found
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return None

    # Find "Episode Decision" step
    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Episode Decision":
            outputs = step.get("outputs", {})
            return {
                "episode_likely": outputs.get("episode_likely", False),
                "indicators_present": outputs.get("indicators_present", 0),
                "min_indicators_required": outputs.get("min_indicators_required", 5),
                "rationale": outputs.get("rationale", ""),
            }

    return None


def _extract_gate_results(pipeline_trace: dict | None) -> dict | None:
    """Extract DSM-gate results from pipeline trace.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict with gate results per indicator or None if not found
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return None

    # Find "Apply DSM-Gate" step
    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Apply DSM-Gate":
            outputs = step.get("outputs", {})
            return outputs.get("gate_results", {})

    return None


def _extract_context_result(pipeline_trace: dict | None) -> dict | None:
    """Extract context evaluation result from pipeline trace.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict with active_context and confidence_scores or None if not found
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return None

    # Find "Evaluate Context" step
    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Evaluate Context":
            outputs = step.get("outputs", {})
            return {
                "active_context": outputs.get("active_context", "unknown"),
                "confidence_scores": outputs.get("confidence_scores", {}),
            }

    return None


def _extract_context_source(pipeline_trace: dict | None) -> dict | None:
    """Extract context source information from pipeline trace.

    Story 6.14: Display context source in Results section for transparency.

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun

    Returns:
        Dict with context source info or None if not found:
        - context_source: 'selected_run' or 'auto_generated'
        - context_evaluation_run_id: UUID string (only for selected_run)
        - dates_covered: number of dates with context data
        - dates_missing: number of dates without context data
        - neutral_weight_dates: list of date strings using neutral weights
        - coverage_ratio: coverage as float 0-1
    """
    if not pipeline_trace or "steps" not in pipeline_trace:
        return None

    # Check for "Context History (Selected Run)" step (Story 6.14)
    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Context History (Selected Run)":
            outputs = step.get("outputs", {})
            return {
                "context_source": outputs.get("context_source", "selected_run"),
                "context_evaluation_run_id": outputs.get("context_evaluation_run_id"),
                "dates_covered": outputs.get("dates_covered", 0),
                "dates_missing": outputs.get("dates_missing", 0),
                "neutral_weight_dates": outputs.get("neutral_weight_dates", []),
                "coverage_ratio": outputs.get("coverage_ratio", 0.0),
            }

    # Check for "Context History Population" step (auto-generated)
    for step in pipeline_trace["steps"]:
        if step.get("step_name") == "Context History Population":
            outputs = step.get("outputs", {})
            return {
                "context_source": outputs.get("context_source", "auto_generated"),
                "context_evaluation_run_id": None,
                "evaluations_added": outputs.get("evaluations_added", 0),
                "gaps_found": outputs.get("gaps_found", 0),
            }

    return None


def _render_windowed_summary(
    pipeline_trace: dict,
    config_snapshot: dict | None = None,
) -> None:
    """Render summary for windowed analysis pipeline.

    Displays metrics from the windowed pipeline including:
    - Window count and daily summaries
    - Indicator likelihood metrics
    - Data quality metrics

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun.pipeline_trace
        config_snapshot: Optional config snapshot dict
    """
    summary = _extract_windowed_summary(pipeline_trace)

    if summary is None:
        st.warning("Could not extract windowed pipeline data.")
        return

    # Top-level metrics
    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric("Windows Processed", summary["window_count"])

    with metric_cols[1]:
        st.metric("Daily Summaries", summary["daily_summaries_count"])

    with metric_cols[2]:
        st.metric("Indicators", len(summary["indicators"]))

    with metric_cols[3]:
        st.metric("Biomarkers Read", summary["biomarker_count"])

    # Context history info
    if summary["context_evaluations_added"] > 0:
        st.info(
            f"Context history backfill: {summary['context_evaluations_added']} "
            "evaluations added"
        )

    # Indicator breakdown
    if summary["indicators"]:
        st.markdown("#### Indicators Processed")

        for indicator_name, indicator_data in sorted(summary["indicators"].items()):
            days_count = indicator_data["daily_summaries_count"]
            dates = indicator_data.get("dates_processed", [])

            with st.expander(f"**{indicator_name}** ({days_count} daily summaries)"):
                if dates:
                    st.caption(f"Dates: {', '.join(dates)}")
                else:
                    st.caption("Date details not available")

    # Configuration info
    if config_snapshot:
        window_config = config_snapshot.get("window", {})
        window_size = window_config.get("size_minutes", "N/A")
        aggregation = window_config.get("aggregation_method", "N/A")

        st.markdown("---")
        st.caption(f"Window size: {window_size} min | " f"Aggregation: {aggregation}")


def render_episode_summary(
    pipeline_trace: dict | None,
    config_snapshot: dict | None = None,
) -> None:
    """Render analysis summary based on pipeline type.

    Automatically detects whether this is a windowed or legacy pipeline
    and renders the appropriate summary view.

    For windowed pipeline (Story 6.6+):
    - Shows window count, daily summaries, indicator metrics
    - Displays data quality and coverage information

    For legacy pipeline:
    - Shows episode likelihood determination with DSM-gate results
    - Displays indicator presence status

    Args:
        pipeline_trace: Pipeline trace dict from AnalysisRun.pipeline_trace
        config_snapshot: Optional config snapshot dict
    """
    # Check if this is a windowed pipeline run
    if _is_windowed_pipeline(pipeline_trace):
        _render_windowed_summary(pipeline_trace, config_snapshot)
        return

    # Legacy pipeline handling below
    episode_data = _extract_episode_decision(pipeline_trace)
    gate_results = _extract_gate_results(pipeline_trace)
    context_result = _extract_context_result(pipeline_trace)

    if episode_data is None:
        st.warning("Analysis summary data not available for this run.")
        return

    episode_likely = episode_data["episode_likely"]
    indicators_present = episode_data["indicators_present"]
    min_required = episode_data["min_indicators_required"]
    rationale = episode_data["rationale"]

    if gate_results:
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Episode Likely", "Yes" if episode_likely else "No")
        with metric_cols[1]:
            indicator_status = (
                "MET" if indicators_present >= min_required else "NOT MET"
            )
            st.metric(
                "Indicators Present",
                f"{indicators_present} of {len(gate_results)}",
                delta=f">={min_required} required: {indicator_status}",
                delta_color="normal"
                if indicators_present >= min_required
                else "inverse",
            )
        with metric_cols[2]:
            if context_result:
                active_context = context_result["active_context"]
                confidence = context_result["confidence_scores"].get(
                    active_context, 0.0
                )
                st.metric(
                    "Detected Context",
                    active_context.replace("_", " ").title(),
                    f"confidence: {confidence:.2f}",
                )
            else:
                st.metric("Detected Context", "N/A")

        # Main status display with color coding
        if episode_likely:
            st.success(f"**Episode criteria MET** — {rationale}")
        else:
            st.error(f"**Episode criteria NOT MET** — {rationale}")

        # Detailed breakdown
        st.markdown("#### Indicator Presence")

        # Extract DSM-gate defaults from config_snapshot
        dsm_defaults = (config_snapshot or {}).get("dsm_gate_defaults", {})
        m_window = dsm_defaults.get("m_window", "M")
        theta = dsm_defaults.get("theta", "θ")
        st.caption(
            f"Episode requires **{m_window} consecutive days** with "
            f"≥5 indicators above **{theta}** threshold and at least one "
            f"core indicator persisting on every day across "
            f"**{m_window} consecutive days**."
        )

        # Split indicators into present and absent
        present_indicators = []
        absent_indicators = []

        for indicator_name, result in sorted(gate_results.items()):
            if result.get("presence_flag", False):
                present_indicators.append(indicator_name)
            else:
                absent_indicators.append(indicator_name)

        # Display in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Present ({len(present_indicators)}):**")
            if present_indicators:
                for ind in present_indicators:
                    result = gate_results[ind]
                    days_above = result.get("days_above_threshold", 0)
                    days_eval = result.get("days_evaluated", 0)
                    st.markdown(f"- :green[{ind}] ({days_above}/{days_eval} days)")
            else:
                st.caption("None")

        with col2:
            st.markdown(f"**Absent ({len(absent_indicators)}):**")
            if absent_indicators:
                for ind in absent_indicators:
                    result = gate_results[ind]
                    days_above = result.get("days_above_threshold", 0)
                    days_eval = result.get("days_evaluated", 0)
                    st.markdown(
                        f"- :red[{ind}] ({days_above}/{days_eval} evaluated days)"
                    )
            else:
                st.caption("None")

        # Summary metrics
        st.markdown("---")
    else:
        # Fallback: just show summary without detailed breakdown
        st.info(f"Indicators present: {indicators_present}/{min_required} required")
