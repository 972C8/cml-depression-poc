"""Tests for dashboard application module."""

from unittest.mock import MagicMock, patch


class TestCheckDatabaseConnection:
    """Tests for check_database_connection function (now in layout module)."""

    def test_check_database_connection_returns_tuple(self):
        """Verify function returns a tuple of (bool, str)."""
        from src.dashboard.components.layout import check_database_connection

        result = check_database_connection()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_check_database_connection_success(self):
        """Verify function returns True when database is connected."""
        from src.dashboard.components.layout import check_database_connection

        is_connected, status_msg = check_database_connection()
        # Assuming database is running during tests
        assert is_connected is True
        assert status_msg == "Connected"

    @patch("src.dashboard.components.layout.SessionLocal")
    def test_check_database_connection_failure(self, mock_session_local):
        """Verify function returns False when database connection fails."""
        # Mock the session to raise an exception
        mock_session = MagicMock()
        mock_session.execute.side_effect = Exception("Connection refused")
        mock_session_local.return_value = mock_session

        from src.dashboard.components.layout import check_database_connection

        is_connected, status_msg = check_database_connection()
        assert is_connected is False
        assert "Connection refused" in status_msg

    @patch("src.dashboard.components.layout.SessionLocal")
    def test_check_database_connection_closes_session(self, mock_session_local):
        """Verify function closes the session after checking."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        from src.dashboard.components.layout import check_database_connection

        check_database_connection()
        mock_session.close.assert_called_once()


class TestDashboardConfiguration:
    """Tests for dashboard page configuration."""

    def test_streamlit_config_file_exists(self):
        """Verify .streamlit/config.toml exists."""
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / ".streamlit" / "config.toml"
        assert config_path.exists()

    def test_streamlit_config_has_correct_port(self):
        """Verify Streamlit config sets port 8501."""
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / ".streamlit" / "config.toml"
        content = config_path.read_text()
        assert "port = 8501" in content

    def test_streamlit_config_disables_usage_stats(self):
        """Verify Streamlit config disables usage statistics."""
        from pathlib import Path

        config_path = Path(__file__).parent.parent.parent / ".streamlit" / "config.toml"
        content = config_path.read_text()
        assert "gatherUsageStats = false" in content


class TestDashboardPages:
    """Tests for dashboard pages structure."""

    def test_pages_directory_exists(self):
        """Verify pages directory exists."""
        from pathlib import Path

        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        assert pages_path.exists()
        assert pages_path.is_dir()

    def test_pages_directory_has_no_init_file(self):
        """Verify pages directory has no __init__.py (Streamlit convention)."""
        from pathlib import Path

        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        init_file = pages_path / "__init__.py"
        assert not init_file.exists()

    def test_all_placeholder_pages_exist(self):
        """Verify all 8 pages exist (Context and Context Evaluation merged into unified Context page)."""
        from pathlib import Path

        pages_path = Path(__file__).parent.parent.parent / "src" / "dashboard" / "pages"
        page_files = list(pages_path.glob("*.py"))
        assert len(page_files) == 8

        # Check expected page names exist
        page_names = [p.name for p in page_files]
        assert any("Home" in name for name in page_names)
        assert any("Biomarkers" in name for name in page_names)
        assert any("Context" in name for name in page_names)
        assert any("Indicators" in name for name in page_names)
        assert any("Pipeline" in name for name in page_names)
        assert any("Timeline" in name for name in page_names)
        assert any("Analysis" in name for name in page_names)
        assert any("Scenarios" in name for name in page_names)
