"""Baseline selector component for analysis page.

Story 4.14: Baseline Configuration & Selection
Tasks 3.1-3.6: Baseline selection UI component.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src.core.baseline_config import (
    BaselineFile,
    list_available_baselines,
    load_baseline_file,
)

__all__ = [
    "get_baselines_directory",
    "init_baseline_session_state",
    "load_baseline_from_upload",
    "render_baseline_selector",
    "validate_baseline_for_analysis",
]

logger = logging.getLogger(__name__)


def get_baselines_directory() -> Path:
    """Get the path to the baselines directory.

    Returns:
        Path to config/baselines directory.
    """
    # Use relative path from working directory (consistent with other config paths)
    return Path("config/baselines")


def init_baseline_session_state() -> None:
    """Initialize baseline-related session state keys.

    Session state keys:
    - baseline_uploaded_content: parsed BaselineFile (for upload)
    """
    if "baseline_uploaded_content" not in st.session_state:
        st.session_state["baseline_uploaded_content"] = None


def load_baseline_from_upload(content: bytes) -> tuple[BaselineFile | None, str | None]:
    """Parse and validate uploaded baseline file content.

    AC2: File uploader with JSON validation.

    Args:
        content: Raw bytes from uploaded file.

    Returns:
        Tuple of (BaselineFile, None) on success, or (None, error_message) on failure.
    """
    try:
        data = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except UnicodeDecodeError as e:
        return None, f"Invalid file encoding: {e}"

    try:
        baseline_file = BaselineFile.model_validate(data)
        return baseline_file, None
    except ValidationError as e:
        # Extract user-friendly error message
        errors = e.errors()
        if errors:
            first_error = errors[0]
            loc = ".".join(str(x) for x in first_error.get("loc", []))
            msg = first_error.get("msg", "Validation failed")
            return None, f"Validation error at '{loc}': {msg}"
        return None, str(e)


def validate_baseline_for_analysis(
    baseline: BaselineFile, required_biomarkers: list[str]
) -> list[str]:
    """Validate baseline has required biomarkers.

    AC5: Warn if baseline doesn't include all biomarkers in config.

    Args:
        baseline: The baseline file to validate.
        required_biomarkers: List of biomarker names needed for analysis.

    Returns:
        List of warning messages (empty if all biomarkers present).
    """
    warnings = []
    for biomarker in required_biomarkers:
        if biomarker not in baseline.baselines:
            warnings.append(f"Missing baseline for biomarker: {biomarker}")
    return warnings


def render_baseline_details(baseline: BaselineFile, source: str) -> None:
    """Render baseline details table.

    AC4: Show baseline details button/toggle.

    Args:
        baseline: The baseline file to display.
        source: Source description (e.g., "predefined / population_default.json").
    """
    st.markdown(f"**Source:** {source}")

    # Build dataframe for table
    data = []
    for name, defn in baseline.baselines.items():
        data.append(
            {
                "Biomarker": name,
                "Mean": defn.mean,
                "Std": defn.std,
            }
        )

    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Show metadata if present
    if baseline.metadata:
        st.markdown("**Metadata:**")
        meta_items = []
        if baseline.metadata.name:
            meta_items.append(f"- Name: {baseline.metadata.name}")
        if baseline.metadata.description:
            meta_items.append(f"- Description: {baseline.metadata.description}")
        if baseline.metadata.created:
            meta_items.append(f"- Created: {baseline.metadata.created}")
        if baseline.metadata.version:
            meta_items.append(f"- Version: {baseline.metadata.version}")
        if meta_items:
            st.markdown("\n".join(meta_items))


def render_baseline_selector(key_prefix: str = "baseline") -> BaselineFile | None:
    """Render baseline strategy selection UI.

    AC1: Baseline Strategy UI with radio buttons.
    AC2: Upload custom baseline with validation.
    AC3: Select predefined baseline dropdown.
    AC4: Show baseline details toggle.

    Args:
        key_prefix: Unique key prefix for Streamlit widgets.

    Returns:
        Selected/uploaded BaselineFile or None if not configured.
    """
    init_baseline_session_state()

    # Strategy selection (AC1)
    # Initialize widget key if needed
    strategy_key = f"{key_prefix}_strategy"
    if strategy_key not in st.session_state:
        st.session_state[strategy_key] = "predefined"

    strategy = st.radio(
        "Select baseline source",
        options=["predefined", "upload"],
        format_func=lambda x: (
            "Select predefined baseline (Recommended)" if x == "predefined" else "Upload custom baseline"
        ),
        key=strategy_key,
        horizontal=True,
    )

    selected_baseline: BaselineFile | None = None

    if strategy == "upload":
        # Upload custom baseline (AC2)
        uploaded_file = st.file_uploader(
            "Upload baseline file",
            type=["json"],
            key=f"{key_prefix}_upload",
            help="Upload a JSON file containing baseline definitions",
        )

        if uploaded_file is not None:
            content = uploaded_file.read()
            baseline, error = load_baseline_from_upload(content)

            if error:
                st.error(f"Invalid baseline file: {error}")
                st.session_state["baseline_uploaded_content"] = None
            else:
                st.success(
                    f"Loaded baseline with {len(baseline.baselines)} biomarkers"
                )
                st.session_state["baseline_uploaded_content"] = baseline
                selected_baseline = baseline
        else:
            # Use previously uploaded content if available
            selected_baseline = st.session_state.get("baseline_uploaded_content")

    else:
        # Select predefined baseline (AC3)
        baselines_dir = get_baselines_directory()
        available = list_available_baselines(baselines_dir)

        if not available:
            st.error(
                f"No baseline files found in {baselines_dir}. "
                "Add .json files to this directory."
            )
        else:
            # Use a stable session state key (not widget key) to persist selection
            state_key = f"{key_prefix}_selected_value"
            widget_key = f"{key_prefix}_select"

            # Default to population_default
            default_baseline = "population_default" if "population_default" in available else available[0]

            # Initialize or validate stored selection
            if state_key not in st.session_state:
                st.session_state[state_key] = default_baseline
            elif st.session_state[state_key] not in available:
                st.session_state[state_key] = default_baseline

            # Calculate index from our stable state
            current_selection = st.session_state[state_key]
            current_index = available.index(current_selection) if current_selection in available else 0

            # Callback to update our stable state when user changes selection
            def on_baseline_change():
                st.session_state[state_key] = st.session_state[widget_key]

            selected_name = st.selectbox(
                "Select baseline file",
                options=available,
                index=current_index,
                key=widget_key,
                on_change=on_baseline_change,
            )

            # Load selected baseline
            try:
                baseline_path = baselines_dir / f"{selected_name}.json"
                selected_baseline = load_baseline_file(baseline_path)
            except Exception as e:
                st.error(f"Failed to load baseline: {e}")
                logger.error(f"Failed to load baseline {selected_name}", exc_info=True)
                selected_baseline = None

    # Show baseline details (AC4)
    if selected_baseline is not None:
        show_details = st.toggle(
            "Show baseline details",
            value=False,
            key=f"{key_prefix}_show_details",
        )
        if show_details:
            if strategy == "upload":
                source = "uploaded / custom"
            else:
                source = f"predefined / {st.session_state.get(f'{key_prefix}_select', 'unknown')}.json"
            render_baseline_details(selected_baseline, source)

    return selected_baseline
