"""Experiment page - create and edit experiments."""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

from src.dashboard.components.experiment_editor import (  # noqa: E402
    render_experiment_editor,
    render_export_yaml,
)
from src.dashboard.components.filters import (  # noqa: E402
    init_filter_session_state,
    render_user_sidebar,
)
from src.dashboard.components.layout import (  # noqa: E402
    render_footer,
    render_page_header,
    render_sidebar_status,
)
from src.dashboard.data.config import get_current_config  # noqa: E402
from src.dashboard.data.experiments import (  # noqa: E402
    create_experiment,
    delete_experiment,
    get_experiment_config,
    list_experiments,
    update_experiment,
)

# Initialize filter session state
init_filter_session_state()

# Track current page for cross-page state management
st.session_state["_current_page"] = "experiment"

# Render user selector in sidebar (no date filter needed for experiments)
user_id = render_user_sidebar()

# System status at end of sidebar
render_sidebar_status()

# Page content
render_page_header(
    "Experimentation",
    "🧪",
    "Create experiments to test alternative configurations and modify parameters.",
)

# Load current config for creating experiments
config = get_current_config()

exp_tab1, exp_tab2 = st.tabs(["View/Edit Experiment", "Create Experiment"])

with exp_tab1:
    st.markdown("View experiment, edit parameters and export to YAML.")

    # Select experiment to edit
    edit_experiments = list_experiments(limit=20)

    if edit_experiments.empty:
        st.info("No experiments to edit. Create one first.")
    else:
        edit_options = {
            row["id"]: f"{row['name']} ({row['id'][:8]}...)"
            for _, row in edit_experiments.iterrows()
        }

        col_select, col_reset = st.columns([3, 1])

        with col_select:
            selected_edit_id = st.selectbox(
                "Select Experiment",
                options=list(edit_options.keys()),
                format_func=lambda x: edit_options[x],
                key="edit_experiment_selector",
            )

        with col_reset:
            st.markdown("")  # Spacing to align with selectbox
            if st.button("Reset to Default", use_container_width=True):
                if selected_edit_id:
                    default_config = get_current_config()
                    update_experiment(selected_edit_id, default_config)
                    # Increment editor version to force new widget keys
                    st.session_state["_editor_version"] = (
                            st.session_state.get("_editor_version", 0) + 1
                    )
                    st.success("Reset to default config!")
                    st.rerun()

        if selected_edit_id:
            # Get editor version for widget key suffix
            editor_version = st.session_state.get("_editor_version", 0)
            key_suffix = f"_v{editor_version}"

            edit_config = get_experiment_config(selected_edit_id)
            if edit_config:
                st.markdown("---")
                modified_config = render_experiment_editor(edit_config, key_suffix)

                if modified_config:
                    st.markdown("---")
                    if st.button(
                            "Save Changes", type="primary", use_container_width=True
                    ):
                        update_experiment(selected_edit_id, modified_config)
                        st.success("Changes saved!")
                        st.rerun()

                    st.markdown("---")
                    render_export_yaml(modified_config, name="experiment")

with exp_tab2:
    st.markdown("Create a new experiment from the current configuration.")

    with st.form("create_experiment_form"):
        exp_name = st.text_input(
            "Experiment Name",
            placeholder="My Experiment",
        )
        exp_description = st.text_area(
            "Description (optional)",
            placeholder="Describe what this experiment tests...",
        )

        if st.form_submit_button("Create Experiment", type="primary"):
            if not exp_name:
                st.error("Experiment name is required")
            else:
                try:
                    experiment = create_experiment(
                        name=exp_name,
                        config=config,
                        description=exp_description if exp_description else None,
                    )
                    st.success(
                        f"Created experiment: {exp_name} ({str(experiment.id)[:8]}...)"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create experiment: {e}")

    # List existing experiments
    st.markdown("---")
    st.markdown("**Existing Experiments:**")
    experiments_list = list_experiments(limit=10)

    if experiments_list.empty:
        st.info("No experiments created yet.")
    else:
        for _, exp_row in experiments_list.iterrows():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{exp_row['name']}** (`{exp_row['id'][:8]}...`)")
                if exp_row["description"]:
                    st.caption(exp_row["description"])
            with col2:
                if st.button("Delete", key=f"del_{exp_row['id']}", type="secondary"):
                    if delete_experiment(exp_row["id"]):
                        st.success("Deleted!")
                        st.rerun()

render_footer()
