"""Tests for biomarker data view page."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestBiomarkersModule:
    """Tests for biomarkers page module existence."""

    def test_biomarkers_page_exists(self):
        """Verify Biomarkers page exists."""
        biomarkers_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "pages"
            / "1_📈_Biomarkers.py"
        )
        assert biomarkers_path.exists()


class TestLoadBiomarkers:
    """Tests for biomarker data loading function."""

    def test_load_biomarkers_is_callable(self):
        """load_biomarkers function exists and is callable."""
        from src.dashboard.data.biomarkers import load_biomarkers

        assert callable(load_biomarkers)

    def test_load_biomarkers_returns_dataframe(self):
        """load_biomarkers returns a pandas DataFrame."""
        from src.dashboard.data.biomarkers import load_biomarkers

        with patch("src.dashboard.data.biomarkers.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_biomarkers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert isinstance(result, pd.DataFrame)

    def test_load_biomarkers_empty_results(self):
        """Empty query results return empty DataFrame with correct columns."""
        from src.dashboard.data.biomarkers import load_biomarkers

        with patch("src.dashboard.data.biomarkers.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_biomarkers(
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
        from src.dashboard.data.biomarkers import load_biomarkers

        # Create mock biomarker with multiple values in JSON
        mock_biomarker = MagicMock()
        mock_biomarker.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_biomarker.biomarker_type = "speech"
        mock_biomarker.value = {"speech_activity": 0.75, "voice_energy": 0.60}
        mock_biomarker.metadata_ = {"source": "app_v1"}

        with patch("src.dashboard.data.biomarkers.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_biomarker
            ]

            result = load_biomarkers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            # One DB row with 2 values should become 2 DataFrame rows
            assert len(result) == 2
            assert "speech_activity" in result["name"].values
            assert "voice_energy" in result["name"].values

    def test_load_biomarkers_handles_exception(self):
        """Database errors return empty DataFrame."""
        from src.dashboard.data.biomarkers import load_biomarkers

        with patch("src.dashboard.data.biomarkers.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = load_biomarkers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_load_biomarkers_with_type_filter(self):
        """Biomarker type filter restricts results."""
        from src.dashboard.data.biomarkers import load_biomarkers

        mock_biomarker = MagicMock()
        mock_biomarker.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_biomarker.biomarker_type = "speech"
        mock_biomarker.value = {"speech_activity": 0.75}
        mock_biomarker.metadata_ = None

        with patch("src.dashboard.data.biomarkers.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_biomarker
            ]

            result = load_biomarkers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
                biomarker_types=["speech"],
            )

            assert len(result) == 1
            assert result.iloc[0]["type"] == "speech"

    def test_load_biomarkers_handles_null_metadata(self):
        """Null metadata is handled gracefully."""
        from src.dashboard.data.biomarkers import load_biomarkers

        mock_biomarker = MagicMock()
        mock_biomarker.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_biomarker.biomarker_type = "network"
        mock_biomarker.value = {"call_count": 5}
        mock_biomarker.metadata_ = None

        with patch("src.dashboard.data.biomarkers.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_biomarker
            ]

            result = load_biomarkers(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert len(result) == 1
            assert result.iloc[0]["source"] == ""


class TestBiomarkerFilters:
    """Tests for biomarker filter helpers."""

    def test_filter_by_biomarker_names(self):
        """Biomarker name filter restricts DataFrame rows."""
        from src.dashboard.data.biomarkers import filter_by_names

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "type": ["speech", "speech", "network"],
                "name": ["speech_activity", "voice_energy", "call_count"],
                "value": [0.75, 0.60, 5.0],
                "source": ["", "", ""],
            }
        )

        result = filter_by_names(df, ["speech_activity"])
        assert len(result) == 1
        assert result.iloc[0]["name"] == "speech_activity"

    def test_filter_by_names_empty_list_returns_all(self):
        """Empty filter list returns all rows."""
        from src.dashboard.data.biomarkers import filter_by_names

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "type": ["speech", "speech", "network"],
                "name": ["speech_activity", "voice_energy", "call_count"],
                "value": [0.75, 0.60, 5.0],
                "source": ["", "", ""],
            }
        )

        result = filter_by_names(df, [])
        assert len(result) == 3


class TestBiomarkerExport:
    """Tests for CSV export functionality."""

    def test_generate_csv_filename(self):
        """CSV filename follows expected pattern."""
        from src.dashboard.data.biomarkers import generate_csv_filename

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        filename = generate_csv_filename("user123", start, end)

        assert filename == "biomarkers_user123_20240101_20240131.csv"

    def test_dataframe_to_csv(self):
        """DataFrame exports to valid CSV string."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)],
                "type": ["speech"],
                "name": ["speech_activity"],
                "value": [0.75],
                "source": ["app_v1"],
            }
        )

        csv_data = df.to_csv(index=False)
        assert "timestamp" in csv_data
        assert "speech_activity" in csv_data
        assert "0.75" in csv_data


class TestSummaryStatistics:
    """Tests for summary statistics calculation."""

    def test_calculate_summary_stats(self):
        """Summary statistics are calculated per biomarker name."""
        from src.dashboard.data.biomarkers import calculate_summary_stats

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 4,
                "type": ["speech"] * 4,
                "name": [
                    "speech_activity",
                    "speech_activity",
                    "voice_energy",
                    "voice_energy",
                ],
                "value": [0.5, 1.0, 0.3, 0.7],
                "source": [""] * 4,
            }
        )

        stats = calculate_summary_stats(df)

        assert len(stats) == 2  # Two unique biomarker names
        assert "speech_activity" in stats["Biomarker"].values
        assert "voice_energy" in stats["Biomarker"].values

        # Check speech_activity stats
        sa_row = stats[stats["Biomarker"] == "speech_activity"].iloc[0]
        assert sa_row["Count"] == 2
        assert sa_row["Mean"] == 0.75  # (0.5 + 1.0) / 2

    def test_calculate_summary_stats_empty_dataframe(self):
        """Empty DataFrame returns empty stats."""
        from src.dashboard.data.biomarkers import calculate_summary_stats

        df = pd.DataFrame(columns=["timestamp", "type", "name", "value", "source"])

        stats = calculate_summary_stats(df)

        assert len(stats) == 0


class TestPagination:
    """Tests for pagination helpers."""

    def test_calculate_pagination_indices(self):
        """Pagination indices are calculated correctly."""
        from src.dashboard.data.biomarkers import calculate_page_indices

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
        from src.dashboard.data.biomarkers import calculate_total_pages

        assert calculate_total_pages(100, 25) == 4
        assert calculate_total_pages(101, 25) == 5
        assert calculate_total_pages(0, 25) == 1  # Minimum 1 page
        assert calculate_total_pages(24, 25) == 1
