"""Tests for dashboard layout components and pages."""

from pathlib import Path
from unittest.mock import patch


def get_home_page_path() -> Path:
    """Get path to the Home page module."""
    return (
        Path(__file__).parent.parent.parent
        / "src"
        / "dashboard"
        / "pages"
        / "0_🏠_Home.py"
    )


class TestHomePage:
    """Tests for Home page functionality."""

    def test_home_page_exists(self):
        """Verify Home page file exists."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        home_page = pages_path / "0_🏠_Home.py"
        assert home_page.exists()

    def test_home_page_has_correct_prefix(self):
        """Verify Home page has prefix 0 for ordering."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        home_page = pages_path / "0_🏠_Home.py"
        assert home_page.name.startswith("0_")

    def test_home_page_has_required_functions(self):
        """Verify Home page defines required functions."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        home_page = pages_path / "0_🏠_Home.py"
        content = home_page.read_text()

        assert "def get_data_counts" in content
        assert "def get_last_analysis_run" in content
        assert "def get_available_users" in content
        assert "def trigger_analysis" in content

    def test_home_page_has_page_config(self):
        """Verify Home page sets page config."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        home_page = pages_path / "0_🏠_Home.py"
        content = home_page.read_text()

        assert "st.set_page_config" in content
        assert 'page_title="Home - MT_POC"' in content
        assert 'page_icon="🏠"' in content

    def test_home_page_has_metrics_section(self):
        """Verify Home page displays metrics."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        home_page = pages_path / "0_🏠_Home.py"
        content = home_page.read_text()

        assert "st.metric" in content
        assert '"Users"' in content
        assert '"Biomarkers"' in content

    def test_home_page_has_quick_actions(self):
        """Verify Home page has quick action buttons."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        home_page = pages_path / "0_🏠_Home.py"
        content = home_page.read_text()

        assert "Run Analysis" in content
        assert "st.button" in content or "st.page_link" in content


class TestLayoutHelpers:
    """Tests for layout helper functions."""

    def test_layout_module_exists(self):
        """Verify layout.py exists in components."""
        layout_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "components"
            / "layout.py"
        )
        assert layout_path.exists()

    def test_render_page_header_exists(self):
        """Verify render_page_header function is available."""
        from src.dashboard.components.layout import render_page_header

        assert callable(render_page_header)

    def test_render_page_header_returns_none(self):
        """Verify render_page_header doesn't return value."""
        from src.dashboard.components.layout import render_page_header

        # Mock streamlit to avoid actual rendering
        with patch("src.dashboard.components.layout.st"):
            result = render_page_header("Test", "🎯")
            assert result is None

    def test_render_sidebar_status_exists(self):
        """Verify render_sidebar_status function is available."""
        from src.dashboard.components.layout import render_sidebar_status

        assert callable(render_sidebar_status)

    def test_render_footer_exists(self):
        """Verify render_footer function is available."""
        from src.dashboard.components.layout import render_footer

        assert callable(render_footer)


class TestPageStructure:
    """Tests for page structure consistency."""

    def test_all_pages_exist(self):
        """Verify all 8 pages exist (Context and Context Evaluation merged into unified Context page)."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        page_files = list(pages_path.glob("*.py"))
        assert len(page_files) == 8

        page_names = [p.name for p in page_files]
        assert any("Home" in name for name in page_names)
        assert any("Biomarkers" in name for name in page_names)
        assert any("Context" in name for name in page_names)
        assert any("Indicators" in name for name in page_names)
        assert any("Timeline" in name for name in page_names)
        assert any("Analysis" in name for name in page_names)
        assert any("Pipeline" in name for name in page_names)
        assert any("Scenarios" in name for name in page_names)

    def test_pages_have_ordered_prefixes(self):
        """Verify pages have proper numeric prefixes for ordering."""
        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        page_files = sorted(pages_path.glob("*.py"))
        page_names = [p.name for p in page_files]

        # Check that each page starts with a number (supports subpages like 2b_)
        import re

        for name in page_names:
            assert re.match(r"^\d+[a-z]?_", name), f"Page {name} doesn't start with valid prefix"
