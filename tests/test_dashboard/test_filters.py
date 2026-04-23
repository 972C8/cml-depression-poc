"""Tests for dashboard filter components."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestFiltersModule:
    """Tests for filters module existence and structure."""

    def test_filters_module_exists(self):
        """Verify filters.py exists in components."""
        filters_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "components"
            / "filters.py"
        )
        assert filters_path.exists()


class TestUserSelector:
    """Tests for user selection component."""

    def test_get_available_users_is_callable(self):
        """Verify get_available_users function is available."""
        from src.dashboard.components.filters import get_available_users

        assert callable(get_available_users)

    def test_user_selector_is_callable(self):
        """Verify user_selector function is available."""
        from src.dashboard.components.filters import user_selector

        assert callable(user_selector)

    def test_get_available_users_returns_list(self):
        """Verify get_available_users returns a list."""
        from src.dashboard.components.filters import get_available_users

        with patch("src.dashboard.components.filters.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                "user1",
                "user2",
            ]

            result = get_available_users()
            assert isinstance(result, list)
            assert result == ["user1", "user2"]

    def test_get_available_users_handles_empty_result(self):
        """Verify get_available_users handles empty database."""
        from src.dashboard.components.filters import get_available_users

        with patch("src.dashboard.components.filters.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = get_available_users()
            assert result == []

    def test_get_available_users_handles_exception(self):
        """Verify get_available_users handles database errors gracefully."""
        from src.dashboard.components.filters import get_available_users

        with patch("src.dashboard.components.filters.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = get_available_users()
            assert result == []


class TestTimeRangeSelector:
    """Tests for time range selection component."""

    def test_time_range_selector_is_callable(self):
        """Verify time_range_selector function is available."""
        from src.dashboard.components.filters import time_range_selector

        assert callable(time_range_selector)

    def test_get_preset_range_is_callable(self):
        """Verify get_preset_range helper is available."""
        from src.dashboard.components.filters import get_preset_range

        assert callable(get_preset_range)

    def test_get_preset_range_today(self):
        """Verify 'today' preset returns correct range."""
        from src.dashboard.components.filters import get_preset_range

        start, end = get_preset_range("today")
        assert start.date() == end.date()
        assert start.hour == 0
        assert start.minute == 0

    def test_get_preset_range_7d(self):
        """Verify '7d' preset returns 7 day range."""
        from src.dashboard.components.filters import get_preset_range

        start, end = get_preset_range("7d")
        diff = end - start
        assert diff.days == 7

    def test_get_preset_range_30d(self):
        """Verify '30d' preset returns 30 day range."""
        from src.dashboard.components.filters import get_preset_range

        start, end = get_preset_range("30d")
        diff = end - start
        assert diff.days == 30

    def test_get_preset_range_custom_defaults_to_7d(self):
        """Verify 'custom' preset defaults to 7 day range."""
        from src.dashboard.components.filters import get_preset_range

        start, end = get_preset_range("custom")
        diff = end - start
        assert diff.days == 7


class TestSessionStatePersistence:
    """Tests for session state initialization and persistence."""

    def test_init_filter_session_state_is_callable(self):
        """Verify init_filter_session_state is available."""
        from src.dashboard.components.filters import init_filter_session_state

        assert callable(init_filter_session_state)

    def test_init_filter_session_state_creates_keys(self):
        """Verify session state keys are initialized."""
        from src.dashboard.components.filters import init_filter_session_state

        mock_session_state = {}

        with patch(
            "src.dashboard.components.filters.st.session_state", mock_session_state
        ):
            init_filter_session_state()

            assert "selected_user_id" in mock_session_state
            assert "selected_start_date" in mock_session_state
            assert "selected_end_date" in mock_session_state

    def test_init_filter_session_state_preserves_existing(self):
        """Verify existing session state values are preserved."""
        from src.dashboard.components.filters import init_filter_session_state

        existing_date = datetime(2024, 1, 1, tzinfo=UTC)
        mock_session_state = {
            "selected_user_id": "existing_user",
            "selected_start_date": existing_date,
        }

        with patch(
            "src.dashboard.components.filters.st.session_state", mock_session_state
        ):
            init_filter_session_state()

            assert mock_session_state["selected_user_id"] == "existing_user"
            assert mock_session_state["selected_start_date"] == existing_date


class TestSelectionSummary:
    """Tests for selection summary component."""

    def test_get_selection_summary_is_callable(self):
        """Verify get_selection_summary is available."""
        from src.dashboard.components.filters import get_selection_summary

        assert callable(get_selection_summary)

    def test_get_selection_summary_returns_dict(self):
        """Verify get_selection_summary returns expected structure."""
        from src.dashboard.components.filters import get_selection_summary

        with patch("src.dashboard.components.filters.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalar.return_value = 5

            result = get_selection_summary(
                "user1",
                datetime.now(UTC) - timedelta(days=7),
                datetime.now(UTC),
            )

            assert isinstance(result, dict)
            assert "biomarkers" in result
            assert "contexts" in result
            assert "indicators" in result
            assert "has_data" in result

    def test_get_selection_summary_handles_exception(self):
        """Verify get_selection_summary handles database errors."""
        from src.dashboard.components.filters import get_selection_summary

        with patch("src.dashboard.components.filters.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = get_selection_summary(
                "user1",
                datetime.now(UTC) - timedelta(days=7),
                datetime.now(UTC),
            )

            assert result == {
                "biomarkers": 0,
                "contexts": 0,
                "indicators": 0,
                "has_data": False,
            }


class TestRenderFilterSidebar:
    """Tests for convenience render function."""

    def test_render_filter_sidebar_is_callable(self):
        """Verify render_filter_sidebar is available."""
        from src.dashboard.components.filters import render_filter_sidebar

        assert callable(render_filter_sidebar)
