"""Pipeline Transparency Viewer component.

Renders each of the 9 pipeline steps with educational content:
- What happened (with actual numbers from the run)
- Why (clinical/technical rationale)
- How (the algorithm/process)
- Formula (LaTeX mathematical formulas where applicable)

Each step is rendered as a collapsed expander by default.
"""

import streamlit as st

from src.core.pipeline import PipelineTrace


def _render_step_header(step_number: int, step_name: str, duration_ms: int) -> str:
    """Build step header string for expander."""
    return f"Step {step_number}: {step_name} ({duration_ms} ms)"


def _render_step_1_run_id(trace: PipelineTrace) -> None:
    """Render Step 1: Generate Analysis Run ID."""
    with st.expander("Step 1: Generate Analysis Run ID", expanded=False):
        st.markdown("#### What Happened")
        st.markdown(f"Generated unique identifier: `{trace.analysis_run_id}`")

        st.markdown("#### Why")
        st.markdown("""
Every analysis run needs a unique identifier for:
- **Traceability**: Linking all computed results to a single execution
- **Debugging**: Identifying which run produced which outputs
- **Audit trail**: Enabling clinical review of historical analyses
""")

        st.markdown("#### How")
        st.markdown("UUID v4 generation using Python's `uuid.uuid4()` - produces a 128-bit random identifier.")

        st.markdown("#### Formula")
        st.latex(r"\text{run\_id} = \text{UUID4}() \rightarrow \text{128-bit random identifier}")


def _render_step_2_context_history(step: dict) -> None:
    """Render Step 2: Context History Population."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    status = outputs.get("status", "unknown")
    gaps_found = outputs.get("gaps_found", 0)
    evaluations_added = outputs.get("evaluations_added", 0)
    message = outputs.get("message", "")

    with st.expander(_render_step_header(2, "Context History Population", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", status.replace("_", " ").title())
        with col2:
            st.metric("Gaps Found", gaps_found)
        with col3:
            st.metric("Evaluations Added", evaluations_added)

        if message:
            st.info(message)

        st.markdown("#### Why")
        st.markdown("""
Context changes throughout the day. A user might be at a **social gathering** in the morning
(where low speech activity is concerning) and **alone at home** in the evening (where low speech
activity is normal).

Without historical context, we cannot correctly weight biomarkers for their situational relevance.
The pipeline must know what context was active at each point in time.
""")

        st.markdown("#### How")
        st.markdown("""
1. **Gap Detection**: Divide analysis period into 15-minute intervals and check for existing context evaluations
2. **Backfill**: For each gap, read historical sensor data and evaluate context retroactively
3. **Forward-Fill**: When looking up context at timestamp T, return the most recent evaluation before T
4. **Staleness Handling**: If context is older than 2 hours, use neutral context (weight = 1.0)
""")

        st.markdown("#### Formula")
        st.markdown("**Expected evaluation timestamps:**")
        st.latex(r"T_{\text{expected}} = \left\{ t : t = t_{\text{start}} + k \cdot \Delta t, \; k \in \mathbb{Z}^+ \right\}")
        st.markdown("where $\\Delta t$ = evaluation interval (default 15 minutes)")

        st.markdown("**Gap identification:**")
        st.latex(r"\text{Gaps} = T_{\text{expected}} \setminus T_{\text{existing}}")


def _render_step_3_read_data(step: dict) -> None:
    """Render Step 3: Read Biomarker Data."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    biomarker_count = outputs.get("biomarker_count", 0)
    has_data = outputs.get("has_data", False)
    start_time = inputs.get("start_time", "")
    end_time = inputs.get("end_time", "")

    with st.expander(_render_step_header(3, "Read Biomarker Data", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Biomarker Records Retrieved", biomarker_count)
        with col2:
            st.metric("Data Found", "Yes" if has_data else "No")

        st.caption(f"Time range: {start_time} to {end_time}")

        st.markdown("#### Why")
        st.markdown("""
Raw data is stored compactly in PostgreSQL (one row = multiple biomarker values in JSON).
The pipeline needs individual records for window-level processing.

This step also:
- Normalizes timestamps to UTC
- Expands JSON values into individual `BiomarkerRecord` objects
- Groups data for efficient downstream access
""")

        st.markdown("#### How")
        st.markdown("""
1. **Query**: `SELECT * FROM biomarker_data WHERE user_id = ? AND timestamp BETWEEN ? AND ?`
2. **Expansion**: Each row with `{whispering: 0.3, prolonged_pauses: 0.4}` becomes two separate records
3. **Normalization**: Naive timestamps assumed UTC; aware timestamps converted via `astimezone(UTC)`
""")

        st.markdown("#### Data Structure")
        st.code("""
BiomarkerRecord(
    name="whispering",
    value=0.35,
    timestamp=2026-01-15T09:23:41Z,
    user_id="user123"
)""", language="python")


def _render_step_4_window_aggregation(step: dict) -> None:
    """Render Step 4: Window Aggregation."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    window_count = outputs.get("window_count", 0)
    biomarkers = outputs.get("biomarkers_aggregated", [])
    biomarker_stats = outputs.get("biomarker_stats", {})
    window_size = inputs.get("window_size_minutes", 15)
    aggregation_method = inputs.get("aggregation_method", "mean")
    biomarker_count = inputs.get("biomarker_count", 0)

    with st.expander(_render_step_header(4, "Window Aggregation", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Windows Created", window_count)
        with col2:
            st.metric("Window Size", f"{window_size} min")
        with col3:
            st.metric("Aggregation", aggregation_method)

        st.markdown(f"**Biomarkers aggregated:** {', '.join(biomarkers) if biomarkers else 'N/A'}")
        st.caption(f"From {biomarker_count} raw biomarker records")

        # Show actual per-biomarker stats
        if biomarker_stats:
            st.markdown("**Actual aggregation results per biomarker:**")
            for biomarker_name, stats in biomarker_stats.items():
                st.markdown(f"- **{biomarker_name}**: {stats.get('window_count', 0)} windows, "
                           f"values [{stats.get('value_min', 0):.4f} - {stats.get('value_max', 0):.4f}], "
                           f"mean={stats.get('value_mean', 0):.4f}")

        st.markdown("#### Why")
        st.markdown("""
**This is where temporal pattern preservation begins.**

The original pipeline averaged all readings to daily values, which destroyed intra-day patterns
and made co-occurrence detection impossible. If speech, voice, and movement are all low at the
**same time** (a withdrawal pattern), daily averaging masks this signal.

**Tumbling windows** (non-overlapping, fixed-size) ensure:
- **No double-counting**: Each reading contributes to exactly one window
- **Clean context binding**: Unambiguous boundaries for context lookup
- **Fixed daily window count**: 96 windows for 15-min intervals (comparable across days)
""")

        st.markdown("#### How")
        st.markdown("""
1. **Floor timestamp to window boundary**: Align to clock time, not data arrival
2. **Group readings by window**: All readings in `[window_start, window_end)` together
3. **Aggregate**: Compute mean (or median/max/min) of grouped readings
""")

        st.markdown("#### Formula")
        st.markdown("**Window boundary alignment:**")
        st.latex(r"\text{minutes\_since\_midnight} = \text{hour} \times 60 + \text{minute}")
        st.latex(r"\text{floored\_minutes} = \left\lfloor \frac{\text{minutes\_since\_midnight}}{\text{window\_size}} \right\rfloor \times \text{window\_size}")

        st.markdown("**Mean aggregation:**")
        st.latex(r"\bar{x}_{\text{window}} = \frac{1}{n} \sum_{i=1}^{n} x_i")

        st.markdown("**Example:**")
        st.code(f"""
floor_to_window(09:23:41, {window_size}) → 09:15:00
floor_to_window(09:15:00, {window_size}) → 09:15:00  (exactly on boundary)
floor_to_window(09:14:59, {window_size}) → 09:00:00  (just before boundary)
""")


def _render_step_5_membership(step: dict, all_steps: list[dict]) -> None:
    """Render Step 5: Membership Computation with Context."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    indicator_name = outputs.get("indicator_name", inputs.get("indicator_name", "unknown"))
    membership_count = outputs.get("membership_count", 0)
    biomarkers_processed = outputs.get("biomarkers_processed", [])
    membership_stats = outputs.get("membership_stats", {})
    context_weights_used = outputs.get("context_weights_used", {})
    context_strategy = inputs.get("context_strategy", "dominant")

    # Collect all membership steps for this indicator
    membership_steps = [s for s in all_steps if s.get("step_name") == "Membership Computation"]

    with st.expander(_render_step_header(5, "Membership Computation", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")

        # Show all indicators processed
        if len(membership_steps) > 1:
            st.markdown("**Indicators processed:**")
            for ms in membership_steps:
                ms_outputs = ms.get("outputs", {})
                ind_name = ms_outputs.get("indicator_name", "unknown")
                mem_count = ms_outputs.get("membership_count", 0)
                st.markdown(f"- **{ind_name}**: {mem_count} window memberships computed")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Indicator", indicator_name)
            with col2:
                st.metric("Memberships Computed", membership_count)

        st.markdown(f"**Context strategy:** `{context_strategy}`")
        if biomarkers_processed:
            st.markdown(f"**Biomarkers:** {', '.join(biomarkers_processed)}")

        # Show actual membership stats per biomarker
        if membership_stats:
            st.markdown("**Actual computation results per biomarker:**")
            for biomarker_name, stats in membership_stats.items():
                with st.container():
                    st.markdown(f"**{biomarker_name}** ({stats.get('count', 0)} windows)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"z-score: [{stats.get('z_score_min', 0):.2f}, {stats.get('z_score_max', 0):.2f}]")
                        st.caption(f"mean: {stats.get('z_score_mean', 0):.2f}")
                    with col2:
                        st.markdown(f"μ (raw): [{stats.get('membership_min', 0):.2f}, {stats.get('membership_max', 0):.2f}]")
                        st.caption(f"mean: {stats.get('membership_mean', 0):.2f}")
                    with col3:
                        st.markdown(f"μ (weighted): [{stats.get('weighted_membership_min', 0):.2f}, {stats.get('weighted_membership_max', 0):.2f}]")
                        st.caption(f"mean: {stats.get('weighted_membership_mean', 0):.2f}")

        # Show actual context weights used
        if context_weights_used:
            st.markdown("**Context weights applied:**")
            for context_name, biomarker_weights in context_weights_used.items():
                weights_str = ", ".join(f"{b}={w}" for b, w in biomarker_weights.items())
                st.markdown(f"- **{context_name}**: {weights_str}")

        st.markdown("#### Why")
        st.markdown("""
Raw biomarker values are **not directly comparable**. A whispering score of 0.3 means nothing
without knowing what's typical for this user.

**Z-score normalization** answers: "How unusual is this value compared to this user's baseline?"

**Context weighting** adjusts for situational relevance:
- Low speech at a **party** → more concerning (weight > 1.0)
- Low speech when **alone** → less concerning (weight < 1.0)
""")

        st.markdown("#### How")
        st.markdown("""
For each window:
1. **Fetch baseline** from `BaselineRepository` (user-specific or population default)
2. **Compute z-score**: How many standard deviations from baseline?
3. **Map to membership**: Sigmoid function produces [0, 1] value
4. **Get context**: Look up what context was active at window midpoint
5. **Apply weight**: Multiply membership by context weight
""")

        st.markdown("#### Formulas")

        st.markdown("**Z-Score:**")
        st.latex(r"z = \frac{x - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("| Z-Score | Meaning |")
            st.markdown("|---------|---------|")
            st.markdown("| z = 0 | At baseline |")
            st.markdown("| z = -1 | 1 std below |")
            st.markdown("| z = -2 | Unusual (low) |")
            st.markdown("| z = +2 | Unusual (high) |")

        st.markdown("**Sigmoid Membership Function:**")
        st.latex(r"\mu = \frac{1}{1 + e^{-z}}")

        st.markdown("Maps z-scores to bounded [0, 1] range:")
        st.latex(r"z = 0 \rightarrow \mu = 0.5 \quad \text{(baseline, neutral)}")
        st.latex(r"z = +2 \rightarrow \mu = 0.88 \quad \text{(elevated)}")
        st.latex(r"z = -2 \rightarrow \mu = 0.12 \quad \text{(depressed)}")

        st.markdown("**Context-Weighted Membership:**")
        st.latex(r"\mu_{\text{weighted}} = \mu \times w_{\text{context}}")


def _render_step_6_window_fasl(step: dict, all_steps: list[dict]) -> None:
    """Render Step 6: Window-Level FASL."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    indicator_name = outputs.get("indicator_name", inputs.get("indicator_name", "unknown"))
    window_indicator_count = outputs.get("window_indicator_count", 0)
    indicator_score_stats = outputs.get("indicator_score_stats", {})
    missing_strategy = inputs.get("missing_strategy", "neutral_fill")

    # Collect all FASL steps
    fasl_steps = [s for s in all_steps if s.get("step_name") == "Window FASL"]

    with st.expander(_render_step_header(6, "Window-Level FASL", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")

        if len(fasl_steps) > 1:
            st.markdown("**Window indicators computed per indicator:**")
            for fs in fasl_steps:
                fs_outputs = fs.get("outputs", {})
                ind_name = fs_outputs.get("indicator_name", "unknown")
                wi_count = fs_outputs.get("window_indicator_count", 0)
                fs_stats = fs_outputs.get("indicator_score_stats", {})
                st.markdown(f"- **{ind_name}**: {wi_count} windows")
                if fs_stats:
                    st.caption(f"  scores: [{fs_stats.get('score_min', 0):.4f} - {fs_stats.get('score_max', 0):.4f}], "
                              f"mean={fs_stats.get('score_mean', 0):.4f}, std={fs_stats.get('score_std', 0):.4f}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Indicator", indicator_name)
            with col2:
                st.metric("Window Indicators", window_indicator_count)

        st.markdown(f"**Missing biomarker strategy:** `{missing_strategy}`")

        # Show actual indicator score stats
        if indicator_score_stats:
            st.markdown("**Actual indicator score distribution:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Score", f"{indicator_score_stats.get('score_min', 0):.4f}")
            with col2:
                st.metric("Max Score", f"{indicator_score_stats.get('score_max', 0):.4f}")
            with col3:
                st.metric("Mean Score", f"{indicator_score_stats.get('score_mean', 0):.4f}")
            with col4:
                st.metric("Std Dev", f"{indicator_score_stats.get('score_std', 0):.4f}")

            st.markdown(f"**Biomarker completeness (mean):** {indicator_score_stats.get('completeness_mean', 0):.2%}")

            # Show peak window details
            peak_window_start = indicator_score_stats.get("peak_window_start", "")
            peak_context = indicator_score_stats.get("peak_window_context", "")
            peak_biomarkers = indicator_score_stats.get("peak_contributing_biomarkers", {})

            if peak_window_start:
                st.markdown(f"**Peak window:** `{peak_window_start}` (context: `{peak_context}`)")
                if peak_biomarkers:
                    st.markdown("**Peak window biomarker contributions:**")
                    contributions = ", ".join(f"{b}={v:.4f}" for b, v in peak_biomarkers.items())
                    st.code(contributions)

        st.markdown("#### Why")
        st.markdown("""
Individual biomarkers are building blocks; **indicators represent clinical symptoms**.

Each DSM-5 indicator (e.g., "1_depressed_mood") combines its mapped biomarkers (e.g., whispering,
prolonged_pauses, reduced_social_interaction, passive_media_binge).
FASL (Fuzzy-Aggregated Symptom Likelihood) produces a **window-level likelihood score** indicating
how likely the symptom is present **at this specific time**.

**Key insight:** FASL detects "all biomarkers concerning together" patterns. If biomarkers are
concerning at different times, daily averaging misses this. Window-level FASL catches the exact
moment when all biomarkers align.
""")

        st.markdown("#### How")
        st.markdown("""
For each window and each indicator:
1. **Gather** all biomarker memberships for this window
2. **Apply direction**: For `lower_is_worse`, compute `1 - μ`
3. **Weighted average**: Combine using indicator-specific weights
""")

        st.markdown("#### Formula")
        st.markdown("**FASL Operator:**")
        st.latex(r"L_k = \frac{\sum_{i} w_i \cdot \mu_i^{\text{directed}}}{\sum_{i} w_i}")

        st.markdown("Where:")
        st.markdown("- $L_k$ = indicator likelihood score [0, 1]")
        st.markdown("- $w_i$ = weight for biomarker $i$ (from indicator config)")
        st.markdown("- $\\mu_i^{\\text{directed}}$ = membership adjusted for direction")

        st.markdown("**Direction Handling:**")
        st.latex(r"\mu^{\text{directed}} = \begin{cases} \mu & \text{if higher\_is\_worse} \\ 1 - \mu & \text{if lower\_is\_worse} \end{cases}")


def _render_step_7_daily_aggregation(step: dict, all_steps: list[dict]) -> None:
    """Render Step 7: Daily Summary Computation."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    indicator_name = outputs.get("indicator_name", inputs.get("indicator_name", "unknown"))
    daily_summaries_count = outputs.get("daily_summaries_count", 0)
    dates_processed = outputs.get("dates_processed", [])
    daily_summaries = outputs.get("daily_summaries", [])

    # Collect all daily aggregation steps
    daily_steps = [s for s in all_steps if s.get("step_name") == "Daily Aggregation"]

    with st.expander(_render_step_header(7, "Daily Summary Computation", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")

        total_summaries = sum(s.get("outputs", {}).get("daily_summaries_count", 0) for s in daily_steps)
        st.metric("Total Daily Summaries", total_summaries)

        if len(daily_steps) > 1:
            st.markdown("**Per indicator:**")
            for ds in daily_steps:
                ds_outputs = ds.get("outputs", {})
                ind_name = ds_outputs.get("indicator_name", "unknown")
                count = ds_outputs.get("daily_summaries_count", 0)
                dates = ds_outputs.get("dates_processed", [])
                st.markdown(f"- **{ind_name}**: {count} summaries")
                if dates:
                    st.caption(f"  Dates: {', '.join(dates[:5])}{'...' if len(dates) > 5 else ''}")

        # Show actual daily summary values
        all_daily_summaries = []
        for ds in daily_steps:
            ds_outputs = ds.get("outputs", {})
            ind_name = ds_outputs.get("indicator_name", "unknown")
            summaries = ds_outputs.get("daily_summaries", [])
            for s in summaries:
                s["indicator_name"] = ind_name
                all_daily_summaries.append(s)

        if all_daily_summaries:
            st.markdown("---")
            st.markdown("**Actual Daily Summary Values:**")

            for summary in all_daily_summaries:
                date_str = summary.get("date", "unknown")
                ind_name = summary.get("indicator_name", "unknown")

                with st.container():
                    st.markdown(f"**{ind_name}** - {date_str}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Likelihood", f"{summary.get('likelihood', 0):.4f}")
                    with col2:
                        st.metric("Windows", summary.get("total_windows", 0))
                    with col3:
                        st.metric("Coverage", f"{summary.get('data_coverage', 0):.1%}")

                    st.caption(f"Completeness: {summary.get('average_biomarker_completeness', 0):.1%} | "
                              f"Context: {summary.get('context_availability', 0):.1%}")

                    st.markdown("---")

        st.markdown("#### Why")
        st.markdown("""
Window-level FASL scores (Step 6) already capture context-weighted, direction-corrected
indicator likelihoods for each time window. The daily likelihood is the **maximum mean
of any k consecutive windows** (consecutive-window peak mean).

This approach:
- **Captures intra-day episodes**: A sustained hour of elevated concern is detected even if the rest of the day is calm
- **Requires sustained elevation**: Isolated single-window spikes are diluted by their neighbors — only contiguous episodes produce high scores
- **Is clinically interpretable**: "How bad was the worst contiguous stretch of the day?"
- **k maps to real time**: With 15-min windows, k=4 means "worst contiguous hour"
""")

        st.markdown("#### How")
        st.markdown("""
Slide a window of k consecutive FASL scores across the day, compute the mean
of each position, and take the maximum:

| Output | Description |
|--------|-------------|
| **likelihood** | Max mean of k consecutive window FASL scores |
| **total_windows** | Number of windows with data |
| **data_coverage** | Ratio of actual to expected windows |
| **quality metrics** | Biomarker completeness, context availability |
""")

        st.markdown("#### Formula")
        st.latex(r"\text{likelihood}_{\text{daily}} = \max_{i} \; \frac{1}{k} \sum_{j=i}^{i+k-1} L_j")
        st.markdown("Where $L_j$ = window-level FASL indicator score from Step 6, "
                     "$k$ = `episode.peak_window_k`")


def _render_step_8_persist(step: dict) -> None:
    """Render Step 8: Persist Results."""
    outputs = step.get("outputs", {})
    inputs = step.get("inputs", {})

    summaries_saved = outputs.get("summaries_saved", 0)
    run_saved = outputs.get("run_saved", False)
    run_id = inputs.get("run_id", "")

    with st.expander(_render_step_header(8, "Persist Results", step.get("duration_ms", 0)), expanded=False):
        st.markdown("#### What Happened")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Summaries Saved", summaries_saved)
        with col2:
            st.metric("Run Metadata Saved", "Yes" if run_saved else "No")

        if run_id:
            st.caption(f"Run ID: `{run_id[:8]}...`")

        st.markdown("#### Why")
        st.markdown("""
Results must be stored for:
- **Clinical review**: Healthcare providers can examine historical analyses
- **Audit trail**: Every computation is logged with full transparency
- **Reproducibility**: Configuration snapshot enables re-running with same settings
""")

        st.markdown("#### How")
        st.markdown("""
| Data | Storage | Purpose |
|------|---------|---------|
| `DailyIndicatorSummary` | `indicator_summary` table (JSONB) | Clinical review, trending |
| Analysis run metadata | `analysis_run` table | Run identification |
| Config snapshot | `analysis_run.config_snapshot` | Reproducibility |
""")

        st.markdown("#### Database Operations")
        st.code("""
-- Save daily summaries
INSERT INTO indicator_summary (user_id, analysis_run_id, date, indicator_name, summary_data)
VALUES (?, ?, ?, ?, ?::jsonb)

-- Save analysis run
INSERT INTO analysis_run (id, user_id, start_time, end_time, config_snapshot, created_at)
VALUES (?, ?, ?, ?, ?::jsonb, NOW())
""", language="sql")


def _render_step_9_trace(trace: PipelineTrace) -> None:
    """Render Step 9: Save Pipeline Trace."""
    with st.expander("Step 9: Save Pipeline Trace", expanded=False):
        st.markdown("#### What Happened")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Steps Recorded", len(trace.steps))
        with col2:
            st.metric("Total Duration", f"{trace.total_duration_ms} ms")

        st.markdown("#### Why")
        st.markdown("""
**Full transparency** enables:
- **Debugging**: Identify where issues occurred in the pipeline
- **Performance optimization**: Find slow steps
- **Clinical audit**: Every decision traceable
- **Reproducibility**: Complete record of what happened
""")

        st.markdown("#### How")
        st.markdown("""
The `PipelineTracer` captures for each step:
- Step name and number
- Start/end timestamps
- Input parameters
- Output results
- Duration in milliseconds
- Optional metadata (warnings, config overrides)
""")

        st.markdown("#### Trace Structure")
        st.code(f"""
PipelineTrace(
    analysis_run_id="{trace.analysis_run_id[:8]}...",
    user_id="{trace.user_id}",
    steps=[{len(trace.steps)} PipelineStep objects],
    total_duration_ms={trace.total_duration_ms},
    started_at="{trace.started_at.isoformat()[:19]}",
    completed_at="{trace.completed_at.isoformat()[:19]}"
)
""", language="python")


def render_pipeline_transparency(trace: PipelineTrace) -> None:
    """Render the complete 9-step pipeline transparency view.

    Each step is displayed as a collapsed expander containing:
    - What happened (with actual numbers from the run)
    - Why (clinical/technical rationale)
    - How (the algorithm/process)
    - Formula (LaTeX mathematical formulas where applicable)

    Args:
        trace: PipelineTrace containing all step data from the analysis run
    """
    if not trace or not trace.steps:
        st.warning("No pipeline trace data available.")
        return

    st.markdown("#### Pipeline Steps (9-Step Windowed Analysis)")
    st.caption("Click each step to expand and see what happened, why, how, and the formulas used.")

    # Build step lookup
    steps_by_name: dict[str, list[dict]] = {}
    all_steps = [s.to_dict() for s in trace.steps]

    for step in all_steps:
        name = step.get("step_name", "unknown")
        if name not in steps_by_name:
            steps_by_name[name] = []
        steps_by_name[name].append(step)

    # Step 1: Generate Run ID (not explicitly traced, use trace metadata)
    _render_step_1_run_id(trace)

    # Step 2: Context History Population
    if "Context History Population" in steps_by_name:
        _render_step_2_context_history(steps_by_name["Context History Population"][0])

    # Step 3: Read Data
    if "Read Data" in steps_by_name:
        _render_step_3_read_data(steps_by_name["Read Data"][0])

    # Step 4: Window Aggregation
    if "Window Aggregation" in steps_by_name:
        _render_step_4_window_aggregation(steps_by_name["Window Aggregation"][0])

    # Step 5: Membership Computation (may have multiple for different indicators)
    if "Membership Computation" in steps_by_name:
        _render_step_5_membership(steps_by_name["Membership Computation"][0], all_steps)

    # Step 6: Window FASL (may have multiple for different indicators)
    if "Window FASL" in steps_by_name:
        _render_step_6_window_fasl(steps_by_name["Window FASL"][0], all_steps)

    # Step 7: Daily Aggregation (may have multiple for different indicators)
    if "Daily Aggregation" in steps_by_name:
        _render_step_7_daily_aggregation(steps_by_name["Daily Aggregation"][0], all_steps)

    # Step 8: Persist Results
    if "Persist Results" in steps_by_name:
        _render_step_8_persist(steps_by_name["Persist Results"][0])

    # Step 9: Pipeline Trace (use trace metadata)
    _render_step_9_trace(trace)
