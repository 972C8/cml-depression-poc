"""MT_POC Dashboard - Main entry point with navigation.

Uses st.navigation() for custom page names and ordering.
"""

import sys
from pathlib import Path

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st  # noqa: E402

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MT_POC Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define pages with custom titles
pages = [
    st.Page("pages/0_home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_⚙️_Analysis.py", title="Analysis", icon="⚙️"),
    st.Page("pages/2_📋_Context.py", title="Evaluate Context", icon="📋"),
    st.Page("pages/3_⚗️_Experiment.py", title="Experiment", icon="⚗️"),
    st.Page("pages/4_🧪_Generate_Mock_Data.py", title="Generate Mock Data", icon="🧪"),
    st.Page("pages/5_📊_Data.py", title="Data", icon="📊"),
]

# Set up navigation
pg = st.navigation(pages)

# Sidebar title (status is rendered by each page at end of sidebar)
with st.sidebar:
    st.title("MT_POC")

# Run the selected page
pg.run()
