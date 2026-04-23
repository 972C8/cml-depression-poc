"""Experiment editor component for modifying analysis configuration."""

from collections.abc import Callable
from datetime import datetime

import streamlit as st

from src.core.config import AnalysisConfig, get_default_config
from src.dashboard.data.config import config_to_yaml

# Cache default config for showing reference values
_DEFAULT_CONFIG = None


def _get_default_config() -> AnalysisConfig:
    """Get cached default config for reference values."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = get_default_config()
    return _DEFAULT_CONFIG


def render_experiment_editor(
    config: AnalysisConfig, key_suffix: str = ""
) -> AnalysisConfig | None:
    """Render experiment editor with editable parameters.

    Args:
        config: Base AnalysisConfig to edit
        key_suffix: Optional suffix for widget keys (use to force re-render)

    Returns:
        Modified AnalysisConfig if valid, None if validation fails
    """
    st.markdown("### Edit Experiment Parameters")

    # Convert to mutable dict for editing
    config_dict = config.to_dict()
    validation_errors: list[str] = []

    # Get default config for reference values
    default_config = _get_default_config()

    # Create tabs for Context vs Analysis parameters
    context_tab, analysis_tab = st.tabs(["Context Evaluation", "Analysis"])

    # ==========================================================================
    # CONTEXT EVALUATION TAB
    # ==========================================================================
    with context_tab:
        st.caption(
            "Parameters for fuzzy logic context detection from sensor data. "
            "These settings affect how contexts (social, solitary, work, etc.) are identified."
        )

        # Initialize context_evaluation dict if not present
        if "context_evaluation" not in config_dict:
            config_dict["context_evaluation"] = {}

        # --- Marker Memberships ---
        with st.expander("**Marker Memberships**", expanded=False):
            st.caption(
                "Define fuzzy membership functions for each marker. "
                "Each marker has fuzzy sets (e.g., low, medium, high) with membership boundaries."
            )

            marker_memberships_dict = {}
            for (
                marker_name,
                marker_config,
            ) in config.context_evaluation.marker_memberships.items():
                st.markdown(f"##### {marker_name}")

                default_marker = (
                    default_config.context_evaluation.marker_memberships.get(
                        marker_name
                    )
                )
                sets_dict = {}

                for set_name, set_config in marker_config.sets.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.text(set_name)
                        st.caption(set_config.type)
                    with col2:
                        # Display params as editable
                        params_str = ", ".join(f"{p:.1f}" for p in set_config.params)
                        new_params_str = st.text_input(
                            f"Params ({set_config.type})",
                            value=params_str,
                            key=f"exp_marker_{marker_name}_{set_name}_params{key_suffix}",
                            help=f"Format: comma-separated floats. {set_config.type} needs {3 if set_config.type == 'triangular' else 4} values",
                        )
                        # Show default if different
                        if default_marker and set_name in default_marker.sets:
                            default_params = ", ".join(
                                f"{p:.1f}" for p in default_marker.sets[set_name].params
                            )
                            if default_params != new_params_str:
                                st.markdown(
                                    f"<span style='color: #888888; font-size: 0.85em;'>default: {default_params}</span>",
                                    unsafe_allow_html=True,
                                )

                        # Parse params
                        try:
                            new_params = [
                                float(p.strip()) for p in new_params_str.split(",")
                            ]
                            sets_dict[set_name] = {
                                "type": set_config.type,
                                "params": new_params,
                            }
                        except ValueError:
                            st.error(f"Invalid params for {marker_name}.{set_name}")
                            validation_errors.append(
                                f"Invalid params for {marker_name}.{set_name}"
                            )
                            sets_dict[set_name] = {
                                "type": set_config.type,
                                "params": list(set_config.params),
                            }

                marker_memberships_dict[marker_name] = {"sets": sets_dict}
                st.divider()

            config_dict["context_evaluation"][
                "marker_memberships"
            ] = marker_memberships_dict

        # --- Context Assumptions ---
        with st.expander("**Context Assumptions**", expanded=False):
            st.caption(
                "Define fuzzy rules for each context. "
                "Each context is detected based on weighted marker conditions."
            )

            context_assumptions_dict = {}
            for (
                ctx_name,
                ctx_config,
            ) in config.context_evaluation.context_assumptions.items():
                st.markdown(f"##### {ctx_name}")

                # Operator display (single valid value: WEIGHTED_MEAN)
                operator = "WEIGHTED_MEAN"
                st.text(f"Operator: {operator}")

                # Conditions editor
                conditions_list = []
                total_weight = 0.0

                for i, condition in enumerate(ctx_config.conditions):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.text(f"{condition.marker}")
                    with col2:
                        st.text(f"→ {condition.fuzzy_set}")
                    with col3:
                        new_weight = st.number_input(
                            "Weight",
                            min_value=0.0,
                            max_value=1.0,
                            value=condition.weight,
                            step=0.05,
                            key=f"exp_ctx_{ctx_name}_cond_{i}_weight{key_suffix}",
                        )
                        new_weight = round(new_weight, 2)

                    conditions_list.append(
                        {
                            "marker": condition.marker,
                            "set": condition.fuzzy_set,
                            "weight": new_weight,
                        }
                    )
                    total_weight += new_weight

                # Validate weights sum to 1.0
                if conditions_list and abs(total_weight - 1.0) > 0.01:
                    st.warning(f"Weights sum to {total_weight:.2f} (should be 1.0)")
                    validation_errors.append(
                        f"{ctx_name}: condition weights sum to {total_weight:.2f}"
                    )

                context_assumptions_dict[ctx_name] = {
                    "conditions": conditions_list,
                    "operator": operator,
                }
                st.divider()

            config_dict["context_evaluation"][
                "context_assumptions"
            ] = context_assumptions_dict

        # --- Neutral Threshold ---
        st.markdown("#### Neutral Threshold")
        neutral_threshold = st.slider(
            "Threshold for neutral context",
            min_value=0.0,
            max_value=1.0,
            value=config.context_evaluation.neutral_threshold,
            step=0.05,
            key=f"exp_neutral_threshold{key_suffix}",
            help="Contexts scoring below this threshold fall back to 'neutral'",
        )
        neutral_threshold = round(neutral_threshold, 2)
        st.markdown(
            f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.context_evaluation.neutral_threshold}</span>",
            unsafe_allow_html=True,
        )
        config_dict["context_evaluation"]["neutral_threshold"] = neutral_threshold

        # --- EMA Smoothing ---
        st.markdown("#### EMA Smoothing")
        st.caption("Controls context transition smoothing and stability")
        col1, col2, col3 = st.columns(3)

        with col1:
            alpha = st.slider(
                "Alpha",
                min_value=0.01,
                max_value=1.0,
                value=config.context_evaluation.ema.alpha,
                step=0.01,
                key=f"exp_alpha{key_suffix}",
                help="Smoothing factor (higher = less smoothing)",
            )
            alpha = round(alpha, 2)
            st.markdown(
                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.context_evaluation.ema.alpha}</span>",
                unsafe_allow_html=True,
            )
        with col2:
            hysteresis = st.slider(
                "Hysteresis",
                min_value=0.0,
                max_value=0.5,
                value=config.context_evaluation.ema.hysteresis,
                step=0.01,
                key=f"exp_hysteresis{key_suffix}",
                help="Buffer to prevent jitter",
            )
            hysteresis = round(hysteresis, 2)
            st.markdown(
                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.context_evaluation.ema.hysteresis}</span>",
                unsafe_allow_html=True,
            )
        with col3:
            dwell_time = st.number_input(
                "Dwell Time",
                min_value=1,
                max_value=10,
                value=config.context_evaluation.ema.dwell_time,
                key=f"exp_dwell_time{key_suffix}",
                help="Minimum periods before transition",
            )
            st.markdown(
                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.context_evaluation.ema.dwell_time}</span>",
                unsafe_allow_html=True,
            )

        config_dict["context_evaluation"]["ema"] = {
            "alpha": alpha,
            "hysteresis": hysteresis,
            "dwell_time": dwell_time,
        }

    # ==========================================================================
    # ANALYSIS TAB
    # ==========================================================================
    with analysis_tab:
        st.caption(
            "Parameters for indicator computation and DSM-gate analysis. "
            "These settings affect how indicators are calculated and weighted."
        )

        # --- DSM-Gate Parameters ---
        st.markdown("#### DSM-Gate Parameters")
        dsm_col1, dsm_col2, dsm_col3 = st.columns(3)

        with dsm_col1:
            theta = st.slider(
                "Theta (θ)",
                min_value=0.0,
                max_value=1.0,
                value=config.dsm_gate_defaults.theta,
                step=0.01,
                key=f"exp_theta{key_suffix}",
                help="Daily likelihood threshold. A day counts as positive if L(d) ≥ θ.",
            )
            theta = round(theta, 2)
            st.markdown(
                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.dsm_gate_defaults.theta}</span>",
                unsafe_allow_html=True,
            )
        with dsm_col2:
            m_window = st.number_input(
                "M Window (days)",
                min_value=1,
                max_value=60,
                value=config.dsm_gate_defaults.m_window,
                key=f"exp_m_window{key_suffix}",
                help="Sliding-window size in days (DSM-5: 2-week period).",
            )
            st.markdown(
                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.dsm_gate_defaults.m_window}</span>",
                unsafe_allow_html=True,
            )
        with dsm_col3:
            gate_need = st.number_input(
                "Gate Need (N)",
                min_value=1,
                max_value=m_window,
                value=min(config.dsm_gate_defaults.gate_need, m_window),
                key=f"exp_gate_need{key_suffix}",
                help="Required positive days within window for indicator presence (N-of-M rule).",
            )
            st.markdown(
                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_config.dsm_gate_defaults.gate_need}</span>",
                unsafe_allow_html=True,
            )

        config_dict["dsm_gate_defaults"] = {
            "theta": theta,
            "m_window": m_window,
            "gate_need": gate_need,
        }

        # --- Indicator Weights ---
        st.markdown("#### Indicator Weights")

        for indicator_name, indicator_config in config.indicators.items():
            with st.expander(f"**{indicator_name}**", expanded=False):
                st.caption("Weights must sum to 1.0")

                # Get default indicator config for reference
                default_indicator = default_config.indicators.get(indicator_name)

                weights = {}
                for bio_name, bio_weight in indicator_config.biomarkers.items():
                    ind_col1, ind_col2 = st.columns([3, 1])
                    with ind_col1:
                        new_weight = st.slider(
                            f"{bio_name}",
                            min_value=0.0,
                            max_value=1.0,
                            value=bio_weight.weight,
                            step=0.01,
                            key=f"exp_{indicator_name}_{bio_name}_weight{key_suffix}",
                        )
                        new_weight = round(new_weight, 2)
                        # Show default value if available
                        if (
                            default_indicator
                            and bio_name in default_indicator.biomarkers
                        ):
                            default_weight = default_indicator.biomarkers[
                                bio_name
                            ].weight
                            st.markdown(
                                f"<span style='color: #888888; font-size: 0.85em;'>default: {default_weight}</span>",
                                unsafe_allow_html=True,
                            )
                    with ind_col2:
                        direction_short = bio_weight.direction[:10]
                        st.caption(direction_short)

                    weights[bio_name] = {
                        "weight": new_weight,
                        "direction": bio_weight.direction,
                    }

                # Validate weights sum to 1.0
                total = sum(w["weight"] for w in weights.values())
                if abs(total - 1.0) > 0.001:
                    st.warning(f"Weights sum to {total:.2f} (should be 1.0)")
                    validation_errors.append(
                        f"{indicator_name}: weights sum to {total:.2f}, should be 1.0"
                    )

                config_dict["indicators"][indicator_name]["biomarkers"] = weights

        # --- Context Weights ---
        st.markdown("#### Context Weights")
        st.caption(
            "Multipliers: < 1.0 reduces weight, > 1.0 increases weight, = 1.0 no change"
        )

        if not config.context_weights:
            st.info("No context weights configured.")
        else:
            for context_name, biomarker_multipliers in config.context_weights.items():
                with st.expander(f"**{context_name}**", expanded=False):
                    # Get default context weights for reference
                    default_context = default_config.context_weights.get(
                        context_name, {}
                    )

                    updated_multipliers = {}
                    for bio_name, multiplier in biomarker_multipliers.items():
                        ctx_col1, ctx_col2 = st.columns([3, 1])
                        with ctx_col1:
                            new_multiplier = st.slider(
                                f"{bio_name}",
                                min_value=0.01,
                                max_value=3.0,
                                value=float(multiplier),
                                step=0.01,
                                key=f"exp_ctx_{context_name}_{bio_name}{key_suffix}",
                            )
                            new_multiplier = round(new_multiplier, 2)
                            # Show default value if available
                            if bio_name in default_context:
                                default_mult = default_context[bio_name]
                                st.markdown(
                                    f"<span style='color: #888888; font-size: 0.85em;'>default: {default_mult}</span>",
                                    unsafe_allow_html=True,
                                )
                        with ctx_col2:
                            if new_multiplier > 1.0:
                                effect = "↑ increase"
                            elif new_multiplier < 1.0:
                                effect = "↓ decrease"
                            else:
                                effect = "— unchanged"
                            st.caption(effect)

                        updated_multipliers[bio_name] = new_multiplier

                    config_dict["context_weights"][context_name] = updated_multipliers

    # Show validation summary (outside tabs)
    if validation_errors:
        st.error(f"Validation errors: {len(validation_errors)}")
        return None

    # Try to construct valid config
    try:
        return AnalysisConfig(**config_dict)
    except Exception as e:
        st.error(f"Invalid configuration: {e}")
        return None


def render_experiment_manager(
    config: AnalysisConfig,
    on_create: Callable | None = None,
) -> None:
    """Render experiment management UI.

    Args:
        config: Current AnalysisConfig
        on_create: Optional callback when experiment is created
    """
    st.markdown("### Experiment Management")

    # Create experiment form
    with st.form("create_experiment_form"):
        st.markdown("#### Create New Experiment")

        exp_name = st.text_input(
            "Experiment Name",
            placeholder="My Experiment",
            key="exp_name_input",
        )
        exp_description = st.text_area(
            "Description (optional)",
            placeholder="Describe what this experiment tests...",
            key="exp_desc_input",
        )

        submitted = st.form_submit_button("Create Experiment", type="primary")

        if submitted:
            if not exp_name:
                st.error("Experiment name is required")
            else:
                from src.dashboard.data.experiments import create_experiment

                try:
                    experiment = create_experiment(
                        name=exp_name,
                        config=config,
                        description=exp_description if exp_description else None,
                    )
                    st.success(f"Created experiment: {exp_name}")
                    if on_create:
                        on_create(experiment)
                except Exception as e:
                    st.error(f"Failed to create experiment: {e}")


def render_export_yaml(config: AnalysisConfig, name: str = "experiment") -> None:
    """Render export to YAML button.

    Args:
        config: AnalysisConfig to export
        name: Base name for the downloaded file
    """
    st.markdown("### Export Configuration")

    # Validate config before export
    try:
        # Re-validate by reconstructing
        AnalysisConfig(**config.to_dict())
        valid = True
    except Exception as e:
        st.error(f"Configuration is invalid and cannot be exported: {e}")
        valid = False

    if valid:
        yaml_content = config_to_yaml(config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.yaml"

        st.download_button(
            label="Export to YAML",
            data=yaml_content,
            file_name=filename,
            mime="text/yaml",
            use_container_width=True,
        )

        st.caption(f"Will download as: `{filename}`")

        with st.expander("Preview YAML", expanded=False):
            st.code(yaml_content, language="yaml")
