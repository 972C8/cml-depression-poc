"""Pipeline viewer component for rendering pipeline traces.

Extracted from Pipeline page to enable reuse on Analysis page.
"""

import streamlit as st

from src.core.pipeline import PipelineTrace
from src.dashboard.components.filters import get_display_timezone


def render_pipeline_flow(trace: PipelineTrace) -> None:
    """Render visual pipeline flow diagram.

    Displays a horizontal flow showing each step's number and name
    connected with arrows: [1] step_name -> [2] step_name -> ...

    Args:
        trace: PipelineTrace containing the steps to visualize
    """
    if not trace.steps:
        st.caption("No pipeline steps recorded")
        return

    flow_parts = []
    for step in trace.steps:
        flow_parts.append(f"[{step.step_number}] {step.step_name}")
    st.markdown(" → ".join(flow_parts))


def render_pipeline_steps(trace: PipelineTrace) -> None:
    """Render step-by-step pipeline trace with expandable details.

    For each step displays: step number, name, duration (ms).
    Collapsible tabs for: Inputs, Outputs, Metadata.

    Args:
        trace: PipelineTrace containing the steps to render
    """
    if not trace.steps:
        st.caption("No pipeline steps recorded")
        return

    for step in trace.steps:
        step_header = f"Step {step.step_number}: {step.step_name} ({step.duration_ms} ms)"

        with st.expander(step_header, expanded=False):
            # Key metrics inline
            info_cols = st.columns(3)
            with info_cols[0]:
                st.markdown(f"**Duration:** {step.duration_ms} ms")
            with info_cols[1]:
                tz = get_display_timezone()
                step_time = (
                    step.timestamp.astimezone(tz)
                    if step.timestamp.tzinfo
                    else step.timestamp.replace(tzinfo=tz)
                )
                st.markdown(f"**Timestamp:** {step_time.strftime('%H:%M:%S.%f')[:-3]}")
            with info_cols[2]:
                # Extract key scalar outputs for quick view
                scalar_outputs = {
                    k: v
                    for k, v in step.outputs.items()
                    if isinstance(v, int | float | str | bool)
                }
                if scalar_outputs:
                    st.markdown(f"**Key Outputs:** {len(scalar_outputs)} values")

            # Display key scalar values prominently
            if scalar_outputs:
                st.markdown("**Key Values:**")
                for key, value in list(scalar_outputs.items())[:5]:
                    if isinstance(value, float):
                        st.markdown(f"- `{key}`: {value:.4f}")
                    else:
                        st.markdown(f"- `{key}`: {value}")

            # Collapsible sections for full data
            tab_inputs, tab_outputs, tab_meta = st.tabs(["Inputs", "Outputs", "Metadata"])

            with tab_inputs:
                if step.inputs:
                    st.json(step.inputs)
                else:
                    st.caption("No inputs recorded")

            with tab_outputs:
                if step.outputs:
                    st.json(step.outputs)
                else:
                    st.caption("No outputs recorded")

            with tab_meta:
                if step.metadata:
                    st.json(step.metadata)
                else:
                    st.caption("No metadata recorded")


def render_config_snapshot(config_snapshot: dict | None, label: str = "View Configuration Snapshot") -> None:
    """Render config snapshot in a collapsible expander.

    Args:
        config_snapshot: Dictionary containing configuration data
        label: Label for the expander (default: "View Configuration Snapshot")
    """
    if config_snapshot:
        with st.expander(label, expanded=False):
            st.json(config_snapshot)
    else:
        st.caption("No configuration snapshot available for this run.")
