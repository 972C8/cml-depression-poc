"""Generate Mock Data page - generate test data for scenarios."""

import sys
from pathlib import Path

import yaml

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

from src.dashboard.components.filters import (  # noqa: E402
    get_display_timezone,
    init_filter_session_state,
)
from src.dashboard.components.layout import (  # noqa: E402
    render_footer,
    render_page_header,
    render_sidebar_status,
)
from src.dashboard.data.scenarios import (  # noqa: E402
    GenerationConfig,
    generate_scenario_data,
    get_available_scenarios,
    get_scenario_info,
    reset_user_data,
)

# Initialize filter session state
init_filter_session_state()

# Page-specific session state
if "scenario_last_generation" not in st.session_state:
    st.session_state["scenario_last_generation"] = None
if "scenario_confirm_reset" not in st.session_state:
    st.session_state["scenario_confirm_reset"] = False

# Track current page for cross-page state management
st.session_state["_current_page"] = "generate_mock_data"

# Page content
render_page_header(
    "Generate Mock Data",
    "🧪",
    "Generate test data for predefined scenarios",
)

# --- Configuration Section ---
st.subheader("Test Configuration")

# Scenario selection (full width)
scenarios = get_available_scenarios()
selected_scenario = st.selectbox(
    "Scenario",
    options=scenarios,
    format_func=lambda x: get_scenario_info(x).name if get_scenario_info(x) else x,
    help="Select a predefined scenario to test",
)

scenario_info = get_scenario_info(selected_scenario)
if scenario_info:
    st.markdown(f"**Description:** {scenario_info.description}")
    st.markdown(f"**Expected Context:** `{scenario_info.expected_context}`")
    st.markdown(f"**Expected Behavior:** {scenario_info.expected_behavior}")

# Generator config sub-section
st.markdown("#### Generator Config")
st.caption("Config loaded from `config/mock_data/`")

# Show schedule info if scenario has one
scenario_config_path = _project_root / "config" / "mock_data" / "scenarios" / f"{selected_scenario}.yaml"
if scenario_config_path.exists():
    with open(scenario_config_path) as f:
        scenario_config = yaml.safe_load(f)
    schedule = scenario_config.get("schedule")
    if schedule:
        sched_col1, sched_col2 = st.columns(2)
        active_hours = schedule.get("active_hours", {})
        day_pattern = schedule.get("day_pattern", {})
        with sched_col1:
            if active_hours:
                st.metric("Active Hours", f"{active_hours.get('start', '?')}:00 – {active_hours.get('end', '?')}:00")
        with sched_col2:
            if day_pattern:
                days_on = day_pattern.get("days_on", "?")
                days_off = day_pattern.get("days_off", "?")
                offset = day_pattern.get("offset", "?")
                st.metric("Day Pattern", f"{days_on} on / {days_off} off (offset {offset})")
else:
    scenario_config = None

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    with st.expander("Scenario Config", expanded=False):
        if scenario_config:
            st.code(yaml.dump(scenario_config, default_flow_style=False, sort_keys=False), language="yaml")
        else:
            st.warning(f"Config file not found: {selected_scenario}.yaml")

with config_col2:
    with st.expander("Biomarker Config", expanded=False):
        biomarker_config_path = _project_root / "config" / "mock_data" / "neutral_biomarkers.yaml"
        if biomarker_config_path.exists():
            with open(biomarker_config_path) as f:
                biomarker_config = yaml.safe_load(f)
            st.code(yaml.dump(biomarker_config, default_flow_style=False, sort_keys=False), language="yaml")
        else:
            st.warning("Biomarker config file not found")

with config_col3:
    with st.expander("Context Markers Config", expanded=False):
        context_config_path = _project_root / "config" / "mock_data" / "neutral_context_markers.yaml"
        if context_config_path.exists():
            with open(context_config_path) as f:
                context_config = yaml.safe_load(f)
            st.code(yaml.dump(context_config, default_flow_style=False, sort_keys=False), language="yaml")
        else:
            st.warning("Context markers config file not found")

st.divider()

# --- Generation Settings ---
st.subheader("Generation Settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)

with settings_col1:
    user_id = st.text_input(
        "Test User ID",
        value="test-user",
        help="User ID for generated test data",
    )

with settings_col2:
    days = st.number_input(
        "Days of Data",
        min_value=1,
        max_value=30,
        value=14,
        help="Number of days of mock data to generate",
    )

with settings_col3:
    seed = st.number_input(
        "Seed Value",
        value=42,
        min_value=0,
        step=1,
        disabled=not st.session_state.get("seed_enabled", True),
        help="Seed for reproducible random generation",
    )
    seed_enabled = st.checkbox("Use Fixed Seed", value=True, key="seed_enabled")
    if not seed_enabled:
        seed = None

with st.expander("Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        biomarker_interval = st.number_input(
            "Biomarker Interval (minutes)",
            value=10,
            min_value=1,
            max_value=120,
            help="Minutes between biomarker samples",
        )

    with adv_col2:
        context_interval = st.number_input(
            "Context Interval (minutes)",
            value=20,
            min_value=1,
            max_value=240,
            help="Minutes between context samples",
        )

    # Modality ablation
    st.markdown("**Modality Selection (Ablation Testing)**")
    mod_col1, mod_col2 = st.columns(2)
    with mod_col1:
        include_speech = st.checkbox("Include Speech", value=True)
    with mod_col2:
        include_network = st.checkbox("Include Network", value=True)

    if not include_speech and not include_network:
        st.warning("At least one modality must be selected")
        include_speech = True  # Force at least one

    modalities = []
    if include_speech:
        modalities.append("speech")
    if include_network:
        modalities.append("network")

st.divider()

# --- Action Buttons ---
st.subheader("Actions")

action_col1, action_col2 = st.columns(2)

with action_col1:
    generate_clicked = st.button(
        "Generate Mock Data",
        type="primary",
        use_container_width=True,
        help="Generate biomarker and context data for the selected scenario",
    )

with action_col2:
    reset_clicked = st.button(
        "Reset User Data",
        type="secondary",
        use_container_width=True,
        help="Delete all data for the test user",
    )

# Handle Generate Mock Data
if generate_clicked:
    config = GenerationConfig(
        scenario=selected_scenario,
        user_id=user_id,
        days=days,
        seed=seed,
        biomarker_interval=biomarker_interval,
        context_interval=context_interval,
        modalities=modalities if len(modalities) < 2 else None,
    )

    with st.spinner(f"Generating {days} days of mock data..."):
        try:
            result = generate_scenario_data(config)
            st.session_state["scenario_last_generation"] = result
            st.success(
                f"Generated {result.biomarker_count} biomarkers and "
                f"{result.context_count} context markers for '{selected_scenario}'"
            )
        except Exception as e:
            st.error(f"Failed to generate data: {e}")

# Handle Reset User Data
if reset_clicked:
    st.session_state["scenario_confirm_reset"] = True

if st.session_state.get("scenario_confirm_reset"):
    st.warning(f"This will delete ALL data for user '{user_id}'")
    confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 3])
    with confirm_col1:
        if st.button("Confirm Delete", type="primary"):
            with st.spinner("Deleting data..."):
                try:
                    result = reset_user_data(user_id)
                    st.session_state["scenario_last_generation"] = None
                    st.session_state["scenario_confirm_reset"] = False
                    st.success(
                        f"Deleted: {result.biomarkers_deleted} biomarkers, "
                        f"{result.context_deleted} context, "
                        f"{result.indicators_deleted} indicators, "
                        f"{result.analysis_runs_deleted} analysis runs, "
                        f"{result.context_history_deleted} context history, "
                        f"{result.user_baselines_deleted} baselines, "
                        f"{result.context_evaluation_runs_deleted} context eval runs"
                    )
                except Exception as e:
                    st.error(f"Failed to reset data: {e}")
    with confirm_col2:
        if st.button("Cancel"):
            st.session_state["scenario_confirm_reset"] = False
            st.rerun()

st.divider()

# Generation result display
last_generation = st.session_state.get("scenario_last_generation")
if last_generation:
    with st.expander("Last Generation Details"):
        tz = get_display_timezone()
        start_local = last_generation.start_time.astimezone(tz) if last_generation.start_time.tzinfo else last_generation.start_time.replace(tzinfo=tz)
        end_local = last_generation.end_time.astimezone(tz) if last_generation.end_time.tzinfo else last_generation.end_time.replace(tzinfo=tz)
        st.markdown(
            f"""
- **Scenario:** {last_generation.scenario}
- **Biomarkers:** {last_generation.biomarker_count}
- **Context Markers:** {last_generation.context_count}
- **Time Range:** {start_local.strftime('%Y-%m-%d %H:%M')} to {end_local.strftime('%Y-%m-%d %H:%M')}
- **Modalities:** {', '.join(last_generation.modalities_generated)}
"""
        )

# System status at end of sidebar
render_sidebar_status()

render_footer()
