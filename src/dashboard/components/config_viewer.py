"""Configuration viewer components for displaying analysis config."""

import pandas as pd
import streamlit as st

from src.core.config import AnalysisConfig
from src.dashboard.data.config import config_to_yaml


def render_indicators_section(config: AnalysisConfig) -> None:
    """Render indicators configuration section.

    Args:
        config: AnalysisConfig instance
    """
    st.markdown("### Indicators")
    st.markdown(f"**{len(config.indicators)} indicator(s) configured**")

    for indicator_name, indicator_config in config.indicators.items():
        with st.expander(f"**{indicator_name}**", expanded=False):
            # Biomarker weights table
            biomarker_data = []
            for bio_name, bio_weight in indicator_config.biomarkers.items():
                biomarker_data.append(
                    {
                        "Biomarker": bio_name,
                        "Weight": f"{bio_weight.weight:.2f}",
                        "Direction": bio_weight.direction,
                    }
                )

            df = pd.DataFrame(biomarker_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

            # DSM-gate override if present
            if indicator_config.dsm_gate:
                st.markdown("**Custom DSM-Gate:**")
                gate = indicator_config.dsm_gate
                st.markdown(
                    f"- theta: `{gate.theta}` (only theta is per-indicator; M and N use global defaults)"
                )
            else:
                st.caption("Uses default DSM-gate parameters")

            # Min biomarkers
            st.caption(f"Min biomarkers required: {indicator_config.min_biomarkers}")


def render_context_weights_section(config: AnalysisConfig) -> None:
    """Render context weight adjustments section.

    Args:
        config: AnalysisConfig instance
    """
    st.markdown("### Context Weight Adjustments")

    if not config.context_weights:
        st.info("No context weight adjustments configured.")
        return

    st.markdown(f"**{len(config.context_weights)} context(s) configured**")
    st.caption(
        "Multipliers: < 1.0 reduces weight, > 1.0 increases weight, = 1.0 no change"
    )

    for context_name, biomarker_multipliers in config.context_weights.items():
        with st.expander(f"**{context_name}**", expanded=False):
            data = []
            for bio_name, multiplier in biomarker_multipliers.items():
                if multiplier > 1.0:
                    effect = "increase"
                elif multiplier < 1.0:
                    effect = "decrease"
                else:
                    effect = "unchanged"
                data.append(
                    {
                        "Biomarker": bio_name,
                        "Multiplier": f"{multiplier:.2f}",
                        "Effect": effect,
                    }
                )
            df = pd.DataFrame(data)
            st.dataframe(df, hide_index=True, use_container_width=True)


def render_dsm_gate_section(config: AnalysisConfig) -> None:
    """Render DSM-gate configuration section.

    Args:
        config: AnalysisConfig instance
    """
    st.markdown("### DSM-Gate Parameters")

    gate = config.dsm_gate_defaults
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Theta (θ)", f"{gate.theta:.2f}")
    with col2:
        st.metric("M Window (days)", gate.m_window)
    with col3:
        st.metric("Gate Need (N)", gate.gate_need)

    st.caption(
        f"An indicator is present if above {gate.theta} threshold on "
        f"≥{gate.gate_need} of the last {gate.m_window} days (N-of-M rule). "
        f"Episode requires ≥{config.episode.min_indicators} present indicators, "
        f"including at least one core indicator."
    )


def render_ema_section(config: AnalysisConfig) -> None:
    """Render EMA smoothing configuration section.

    Args:
        config: AnalysisConfig instance
    """
    st.markdown("### EMA Smoothing")

    # Story 6.13: EMA is now part of context_evaluation
    ema = config.context_evaluation.ema
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Alpha", f"{ema.alpha:.2f}")
    with col2:
        st.metric("Hysteresis", f"{ema.hysteresis:.2f}")
    with col3:
        st.metric("Dwell Time", ema.dwell_time)

    st.caption(
        "Alpha: smoothing factor (higher = less smoothing). "
        "Hysteresis: buffer to prevent jitter. "
        "Dwell time: periods before transition."
    )


def render_episode_section(config: AnalysisConfig) -> None:
    """Render episode decision configuration section.

    Args:
        config: AnalysisConfig instance
    """
    st.markdown("### Episode Decision (DSM-5)")

    episode = config.episode
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Min Indicators", f"{episode.min_indicators} of {len(config.indicators)}")
    with col2:
        st.metric("Core Indicators", len(episode.core_indicators))

    st.markdown("**Core indicators (at least one must be present):**")
    for core in episode.core_indicators:
        st.markdown(f"- `{core}`")


def render_global_settings_section(config: AnalysisConfig) -> None:
    """Render global settings configuration.

    Args:
        config: AnalysisConfig instance
    """
    st.markdown("### Global Settings")

    st.metric("Timezone", config.timezone)


def render_raw_config(config: AnalysisConfig) -> None:
    """Render raw YAML configuration.

    Args:
        config: AnalysisConfig instance
    """
    st.code(config_to_yaml(config), language="yaml")


def render_config_viewer(config: AnalysisConfig) -> None:
    """Render complete configuration viewer.

    Args:
        config: AnalysisConfig instance
    """
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Indicators",
            "Context Weights",
            "DSM-Gate",
            "EMA",
            "Episode",
            "Global",
        ]
    )

    with tab1:
        render_indicators_section(config)

    with tab2:
        render_context_weights_section(config)

    with tab3:
        render_dsm_gate_section(config)

    with tab4:
        render_ema_section(config)

    with tab5:
        render_episode_section(config)

    with tab6:
        render_global_settings_section(config)
