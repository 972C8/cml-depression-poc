"""Tests for context data view page."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestContextModule:
    """Tests for context page module existence."""

    def test_context_page_exists(self):
        """Verify unified Context page exists."""
        context_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "pages"
            / "2_📊_Context.py"
        )
        assert context_path.exists()


class TestLoadContextMarkers:
    """Tests for context marker data loading function."""

    def test_load_context_markers_is_callable(self):
        """load_context_markers function exists and is callable."""
        from src.dashboard.data.context import load_context_markers

        assert callable(load_context_markers)

    def test_load_context_markers_returns_dataframe(self):
        """load_context_markers returns a pandas DataFrame."""
        from src.dashboard.data.context import load_context_markers

        with patch("src.dashboard.data.context.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_context_markers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert isinstance(result, pd.DataFrame)

    def test_load_context_markers_empty_results(self):
        """Empty query results return empty DataFrame with correct columns."""
        from src.dashboard.data.context import load_context_markers

        with patch("src.dashboard.data.context.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_context_markers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert list(result.columns) == [
                "timestamp",
                "type",
                "name",
                "value",
                "source",
            ]
            assert len(result) == 0

    def test_json_expansion_creates_multiple_rows(self):
        """JSON value field is expanded to individual rows."""
        from src.dashboard.data.context import load_context_markers

        # Create mock context with multiple values in JSON
        mock_context = MagicMock()
        mock_context.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_context.context_type = "environment"
        mock_context.value = {"people_in_room": 0.8, "ambient_noise": 0.6}
        mock_context.metadata_ = {"source": "sensor_v1"}

        with patch("src.dashboard.data.context.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_context
            ]

            result = load_context_markers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            # One DB row with 2 values should become 2 DataFrame rows
            assert len(result) == 2
            assert "people_in_room" in result["name"].values
            assert "ambient_noise" in result["name"].values

    def test_load_context_markers_handles_exception(self):
        """Database errors return empty DataFrame."""
        from src.dashboard.data.context import load_context_markers

        with patch("src.dashboard.data.context.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = load_context_markers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_load_context_markers_with_type_filter(self):
        """Context type filter restricts results."""
        from src.dashboard.data.context import load_context_markers

        mock_context = MagicMock()
        mock_context.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_context.context_type = "environment"
        mock_context.value = {"people_in_room": 0.8}
        mock_context.metadata_ = None

        with patch("src.dashboard.data.context.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_context
            ]

            result = load_context_markers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
                context_types=["environment"],
            )

            assert len(result) == 1
            assert result.iloc[0]["type"] == "environment"

    def test_load_context_markers_handles_null_metadata(self):
        """Null metadata is handled gracefully."""
        from src.dashboard.data.context import load_context_markers

        mock_context = MagicMock()
        mock_context.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_context.context_type = "environment"
        mock_context.value = {"temperature": 22.5}
        mock_context.metadata_ = None

        with patch("src.dashboard.data.context.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_context
            ]

            result = load_context_markers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert len(result) == 1
            assert result.iloc[0]["source"] == ""


class TestContextFilters:
    """Tests for context filter helpers."""

    def test_filter_by_marker_names(self):
        """Context marker name filter restricts DataFrame rows."""
        from src.dashboard.data.context import filter_by_names

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "type": ["environment", "environment", "environment"],
                "name": ["people_in_room", "ambient_noise", "network_activity"],
                "value": [0.8, 0.6, 0.4],
                "source": ["", "", ""],
            }
        )

        result = filter_by_names(df, ["people_in_room"])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "people_in_room"

    def test_filter_by_names_empty_list_returns_all(self):
        """Empty filter list returns all rows."""
        from src.dashboard.data.context import filter_by_names

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "type": ["environment", "environment", "environment"],
                "name": ["people_in_room", "ambient_noise", "network_activity"],
                "value": [0.8, 0.6, 0.4],
                "source": ["", "", ""],
            }
        )

        result = filter_by_names(df, [])
        assert len(result) == 3

    def test_filter_by_multiple_names(self):
        """Multiple name filter returns matching rows."""
        from src.dashboard.data.context import filter_by_names

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "type": ["environment"] * 3,
                "name": ["people_in_room", "ambient_noise", "network_activity"],
                "value": [0.8, 0.6, 0.4],
                "source": [""] * 3,
            }
        )

        result = filter_by_names(df, ["people_in_room", "ambient_noise"])
        assert len(result) == 2
        assert set(result["name"].tolist()) == {"people_in_room", "ambient_noise"}


class TestContextExport:
    """Tests for CSV export functionality."""

    def test_generate_csv_filename(self):
        """CSV filename follows expected pattern."""
        from src.dashboard.data.context import generate_csv_filename

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        filename = generate_csv_filename("user123", start, end)

        assert filename == "context_user123_20240101_20240131.csv"

    def test_dataframe_to_csv(self):
        """DataFrame exports to valid CSV string."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)],
                "type": ["environment"],
                "name": ["people_in_room"],
                "value": [0.8],
                "source": ["sensor_v1"],
            }
        )

        csv_data = df.to_csv(index=False)
        assert "timestamp" in csv_data
        assert "people_in_room" in csv_data
        assert "0.8" in csv_data


class TestContextSummaryStatistics:
    """Tests for summary statistics calculation."""

    def test_calculate_summary_stats(self):
        """Summary statistics are calculated per context marker name."""
        from src.dashboard.data.context import calculate_summary_stats

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 4,
                "type": ["environment"] * 4,
                "name": [
                    "people_in_room",
                    "people_in_room",
                    "ambient_noise",
                    "ambient_noise",
                ],
                "value": [0.5, 1.0, 0.3, 0.7],
                "source": [""] * 4,
            }
        )

        stats = calculate_summary_stats(df)

        assert len(stats) == 2  # Two unique marker names
        assert "people_in_room" in stats["Marker"].values
        assert "ambient_noise" in stats["Marker"].values

        # Check people_in_room stats
        pir_row = stats[stats["Marker"] == "people_in_room"].iloc[0]
        assert pir_row["Count"] == 2
        assert pir_row["Mean"] == 0.75  # (0.5 + 1.0) / 2

    def test_calculate_summary_stats_empty_dataframe(self):
        """Empty DataFrame returns empty stats."""
        from src.dashboard.data.context import calculate_summary_stats

        df = pd.DataFrame(columns=["timestamp", "type", "name", "value", "source"])

        stats = calculate_summary_stats(df)

        assert len(stats) == 0


class TestContextPagination:
    """Tests for pagination helpers."""

    def test_calculate_pagination_indices(self):
        """Pagination indices are calculated correctly."""
        from src.dashboard.data.context import calculate_page_indices

        # Page 1 of 100 rows with page size 25
        start, end = calculate_page_indices(
            current_page=1, page_size=25, total_rows=100
        )
        assert start == 0
        assert end == 25

        # Page 2
        start, end = calculate_page_indices(
            current_page=2, page_size=25, total_rows=100
        )
        assert start == 25
        assert end == 50

    def test_calculate_total_pages(self):
        """Total pages calculation handles edge cases."""
        from src.dashboard.data.context import calculate_total_pages

        assert calculate_total_pages(100, 25) == 4
        assert calculate_total_pages(101, 25) == 5
        assert calculate_total_pages(0, 25) == 1  # Minimum 1 page
        assert calculate_total_pages(24, 25) == 1

    def test_last_page_indices(self):
        """Last page correctly caps at total rows."""
        from src.dashboard.data.context import calculate_page_indices

        # Page 4 of 90 rows with page size 25 (should only have 15 rows)
        start, end = calculate_page_indices(current_page=4, page_size=25, total_rows=90)
        assert start == 75
        assert end == 90  # Capped at total_rows, not 100
