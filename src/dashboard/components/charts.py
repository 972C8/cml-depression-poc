"""Reusable Plotly chart components for timeline visualization."""

import pandas as pd
import plotly.graph_objects as go

# Color palettes — 29 biomarkers across speech and network modalities
BIOMARKER_COLORS = {
    # Speech modality
    "whispering": "#1f77b4",
    "prolonged_pauses": "#2ca02c",
    "monoquality": "#9467bd",
    "monopitch": "#ff7f0e",
    "pitch": "#d62728",
    "voice_breathiness_noise": "#8c564b",
    "speech_prosody": "#e377c2",
    "speech_quality": "#7f7f7f",
    "slow_speech": "#bcbd22",
    "voice_production": "#17becf",
    "vocalization": "#aec7e8",
    # Network modality
    "reduced_social_interaction": "#ffbb78",
    "passive_media_binge": "#98df8a",
    "shrinking_domain_diversity": "#ff9896",
    "reduced_interactive_engagement": "#c5b0d5",
    "food_ordering_pattern_shift": "#c49c94",
    "calorie_nutrition_information_seeking": "#f7b6d2",
    "shifted_sleep_timing": "#c7c7c7",
    "changed_sleep_duration": "#dbdb8d",
    "restless_device_switching": "#9edae5",
    "slowed_variable_typing_dynamics": "#393b79",
    "extended_daytime_inactivity": "#637939",
    "decline_effortful_interaction": "#8c6d31",
    "engagement_with_mental_health_content": "#843c39",
    "digital_self_withdrawal_data_purge_behavior": "#7b4173",
    "fragmented_focus": "#5254a3",
    "indecisive_information_seeking": "#6b6ecf",
    "crisis_oriented_help_seeking": "#b5cf6b",
    "self_harm_community_engagement": "#e7969c",
}

# 9 DSM-5 indicators
INDICATOR_COLORS = {
    "1_depressed_mood": "#1f77b4",  # Blue
    "2_loss_of_interest": "#ff7f0e",  # Orange
    "3_weight_changes": "#2ca02c",  # Green
    "4_insomnia_hypersomnia": "#d62728",  # Red
    "5_psychomotor_agitation_retardation": "#9467bd",  # Purple
    "6_fatigue_loss_of_energy": "#8c564b",  # Brown
    "7_worthlessness_guilt": "#e377c2",  # Pink
    "8_diminished_ability_to_think_concentrate": "#7f7f7f",  # Gray
    "9_suicidality": "#bcbd22",  # Yellow-green
}

CONTEXT_COLORS = {
    "solitary_digital": "rgba(44, 160, 44, 0.15)",  # Light green
    "adversarial_social_digital_gaming": "rgba(214, 39, 40, 0.15)",  # Light red
    "neutral": "rgba(127, 127, 127, 0.1)",  # Light gray
}


def render_biomarker_timeline_chart(
    df: pd.DataFrame,
    selected_names: list[str] | None = None,
    show_threshold: bool = False,
    threshold_value: float = 0.5,
    title: str = "Biomarker Values Over Time",
) -> go.Figure:
    """Render interactive line chart for biomarker values.

    Args:
        df: DataFrame with timestamp, name, value columns
        selected_names: Biomarker names to display (None = all)
        show_threshold: Whether to show threshold line
        threshold_value: Y-value for threshold line
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No biomarker data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16, "color": "gray"},
        )
        return _apply_chart_layout(fig, title)

    names = selected_names or df["name"].unique().tolist()

    for name in names:
        name_df = df[df["name"] == name].sort_values("timestamp")
        if name_df.empty:
            continue

        color = BIOMARKER_COLORS.get(name, "#7f7f7f")

        fig.add_trace(
            go.Scatter(
                x=name_df["timestamp"],
                y=name_df["value"],
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"size": 4},
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "Time: %{x}<br>"
                    "Value: %{y:.4f}<extra></extra>"
                ),
            )
        )

    if show_threshold:
        fig.add_hline(
            y=threshold_value,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text=f"Threshold ({threshold_value})",
            annotation_position="right",
        )

    return _apply_chart_layout(fig, title, yaxis_range=[0, 1.05])


def render_indicator_timeline_chart(
    df: pd.DataFrame,
    selected_types: list[str] | None = None,
    threshold_value: float = 0.5,
    title: str = "Indicator Likelihoods Over Time",
) -> go.Figure:
    """Render interactive line chart for indicator likelihoods.

    Args:
        df: DataFrame with timestamp, indicator_type, likelihood columns
        selected_types: Indicator types to display (None = all)
        threshold_value: Y-value for DSM-gate threshold line
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No indicator data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16, "color": "gray"},
        )
        return _apply_chart_layout(fig, title)

    types = selected_types or df["indicator_type"].unique().tolist()

    for ind_type in types:
        type_df = df[df["indicator_type"] == ind_type].sort_values("timestamp")
        if type_df.empty:
            continue

        color = INDICATOR_COLORS.get(ind_type, "#7f7f7f")

        fig.add_trace(
            go.Scatter(
                x=type_df["timestamp"],
                y=type_df["likelihood"],
                mode="lines+markers",
                name=ind_type,
                line={"color": color, "width": 2},
                marker={"size": 5},
                hovertemplate=(
                    f"<b>{ind_type}</b><br>"
                    "Time: %{x}<br>"
                    "Likelihood: %{y:.4f}<extra></extra>"
                ),
            )
        )

    # DSM-gate threshold line
    fig.add_hline(
        y=threshold_value,
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text=f"DSM Threshold ({threshold_value})",
        annotation_position="right",
    )

    return _apply_chart_layout(fig, title, yaxis_range=[0, 1.05])


def add_context_shading(
    fig: go.Figure,
    context_periods: pd.DataFrame,
) -> go.Figure:
    """Add context background shading to a figure.

    Args:
        fig: Existing Plotly figure
        context_periods: DataFrame with start_time, end_time, context columns

    Returns:
        Figure with context shading added
    """
    if context_periods.empty:
        return fig

    for _, row in context_periods.iterrows():
        color = CONTEXT_COLORS.get(row["context"], CONTEXT_COLORS["neutral"])
        fig.add_vrect(
            x0=row["start_time"],
            x1=row["end_time"],
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=row["context"][:10],
            annotation_position="top left",
            annotation_font_size=8,
            annotation_font_color="gray",
        )

    return fig


def render_combined_timeline_chart(
    bio_df: pd.DataFrame,
    ind_df: pd.DataFrame,
    context_periods: pd.DataFrame | None = None,
    selected_biomarkers: list[str] | None = None,
    selected_indicators: list[str] | None = None,
    threshold_value: float = 0.5,
    title: str = "Combined Timeline View",
) -> go.Figure:
    """Render combined chart with biomarkers, indicators, and context shading.

    Args:
        bio_df: Biomarker DataFrame
        ind_df: Indicator DataFrame
        context_periods: Context periods for shading
        selected_biomarkers: Biomarker names to display
        selected_indicators: Indicator types to display
        threshold_value: DSM-gate threshold
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Add context shading first (background layer)
    if context_periods is not None and not context_periods.empty:
        fig = add_context_shading(fig, context_periods)

    # Add biomarkers (dashed lines)
    if not bio_df.empty:
        names = selected_biomarkers or bio_df["name"].unique().tolist()
        for name in names:
            name_df = bio_df[bio_df["name"] == name].sort_values("timestamp")
            if name_df.empty:
                continue

            color = BIOMARKER_COLORS.get(name, "#7f7f7f")
            fig.add_trace(
                go.Scatter(
                    x=name_df["timestamp"],
                    y=name_df["value"],
                    mode="lines",
                    name=f"[B] {name}",
                    line={"color": color, "width": 1.5, "dash": "dash"},
                    hovertemplate=(
                        f"<b>Biomarker: {name}</b><br>"
                        "Time: %{x}<br>"
                        "Value: %{y:.4f}<extra></extra>"
                    ),
                    legendgroup="biomarkers",
                    legendgrouptitle_text="Biomarkers",
                )
            )

    # Add indicators (solid lines)
    if not ind_df.empty:
        types = selected_indicators or ind_df["indicator_type"].unique().tolist()
        for ind_type in types:
            type_df = ind_df[ind_df["indicator_type"] == ind_type].sort_values(
                "timestamp"
            )
            if type_df.empty:
                continue

            color = INDICATOR_COLORS.get(ind_type, "#7f7f7f")
            fig.add_trace(
                go.Scatter(
                    x=type_df["timestamp"],
                    y=type_df["likelihood"],
                    mode="lines+markers",
                    name=f"[I] {ind_type}",
                    line={"color": color, "width": 2},
                    marker={"size": 4},
                    hovertemplate=(
                        f"<b>Indicator: {ind_type}</b><br>"
                        "Time: %{x}<br>"
                        "Likelihood: %{y:.4f}<extra></extra>"
                    ),
                    legendgroup="indicators",
                    legendgrouptitle_text="Indicators",
                )
            )

    # DSM-gate threshold
    fig.add_hline(
        y=threshold_value,
        line_dash="dot",
        line_color="red",
        opacity=0.5,
        annotation_text="DSM Threshold",
        annotation_position="right",
    )

    return _apply_chart_layout(fig, title, yaxis_range=[0, 1.05])


def _apply_chart_layout(
    fig: go.Figure,
    title: str,
    yaxis_range: list[float] | None = None,
) -> go.Figure:
    """Apply common layout settings to a chart.

    Args:
        fig: Plotly figure
        title: Chart title
        yaxis_range: Optional y-axis range [min, max]

    Returns:
        Figure with layout applied
    """
    layout_kwargs = {
        "title": title,
        "xaxis_title": "Time",
        "yaxis_title": "Value",
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        "height": 450,
        "margin": {"l": 50, "r": 50, "t": 80, "b": 50},
    }

    if yaxis_range:
        layout_kwargs["yaxis"] = {"range": yaxis_range}

    fig.update_layout(**layout_kwargs)

    # Enable zoom and pan with time range presets
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1h", "step": "hour", "stepmode": "backward"},
                {"count": 6, "label": "6h", "step": "hour", "stepmode": "backward"},
                {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
                {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ]
        },
    )

    return fig
