"""Tests for indicator data view page."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestIndicatorsModule:
    """Tests for indicators page module existence."""

    def test_indicators_page_exists(self):
        """Verify Indicators page exists."""
        indicators_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "pages"
            / "3_🎯_Indicators.py"
        )
        assert indicators_path.exists()

    def test_indicators_data_module_exists(self):
        """Verify indicators data module exists."""
        indicators_data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "data"
            / "indicators.py"
        )
        assert indicators_data_path.exists()


class TestLoadIndicators:
    """Tests for indicator data loading function."""

    def test_load_indicators_is_callable(self):
        """load_indicators function exists and is callable."""
        from src.dashboard.data.indicators import load_indicators

        assert callable(load_indicators)

    def test_load_indicators_returns_dataframe(self):
        """load_indicators returns a pandas DataFrame."""
        from src.dashboard.data.indicators import load_indicators

        with patch("src.dashboard.data.indicators.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_indicators(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert isinstance(result, pd.DataFrame)

    def test_load_indicators_empty_results(self):
        """Empty query results return empty DataFrame with correct columns."""
        from src.dashboard.data.indicators import load_indicators

        with patch("src.dashboard.data.indicators.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_indicators(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert list(result.columns) == [
                "timestamp",
                "indicator_type",
                "likelihood",
                "presence_flag",
                "data_reliability_score",
                "context_used",
                "analysis_run_id",
            ]
            assert len(result) == 0

    def test_load_indicators_maps_columns_correctly(self):
        """Indicator fields are mapped to correct DataFrame columns."""
        from src.dashboard.data.indicators import load_indicators

        mock_indicator = MagicMock()
        mock_indicator.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_indicator.indicator_type = "social_withdrawal"
        mock_indicator.value = 0.75
        mock_indicator.presence_flag = True
        mock_indicator.data_reliability_score = 0.85
        mock_indicator.context_used = "solitary_digital"
        mock_indicator.analysis_run_id = uuid.uuid4()

        with patch("src.dashboard.data.indicators.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_indicator
            ]

            result = load_indicators(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert len(result) == 1
            row = result.iloc[0]
            assert row["indicator_type"] == "social_withdrawal"
            assert row["likelihood"] == 0.75
            assert row["presence_flag"] == True  # noqa: E712
            assert row["data_reliability_score"] == 0.85
            assert row["context_used"] == "solitary_digital"

    def test_load_indicators_handles_exception(self):
        """Database errors return empty DataFrame."""
        from src.dashboard.data.indicators import load_indicators

        with patch("src.dashboard.data.indicators.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = load_indicators(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_load_indicators_with_type_filter(self):
        """Indicator type filter restricts results."""
        from src.dashboard.data.indicators import load_indicators

        mock_indicator = MagicMock()
        mock_indicator.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_indicator.indicator_type = "social_withdrawal"
        mock_indicator.value = 0.75
        mock_indicator.presence_flag = True
        mock_indicator.data_reliability_score = 0.85
        mock_indicator.context_used = None
        mock_indicator.analysis_run_id = uuid.uuid4()

        with patch("src.dashboard.data.indicators.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_indicator
            ]

            result = load_indicators(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
                indicator_types=["social_withdrawal"],
            )

            assert len(result) == 1
            assert result.iloc[0]["indicator_type"] == "social_withdrawal"

    def test_load_indicators_handles_null_context(self):
        """Null context_used is replaced with 'N/A'."""
        from src.dashboard.data.indicators import load_indicators

        mock_indicator = MagicMock()
        mock_indicator.timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_indicator.indicator_type = "diminished_interest"
        mock_indicator.value = 0.60
        mock_indicator.presence_flag = False
        mock_indicator.data_reliability_score = 0.70
        mock_indicator.context_used = None
        mock_indicator.analysis_run_id = uuid.uuid4()

        with patch("src.dashboard.data.indicators.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_indicator
            ]

            result = load_indicators(
                user_id="test_user",
                start=datetime.now(UTC) - timedelta(days=7),
                end=datetime.now(UTC),
            )

            assert len(result) == 1
            assert result.iloc[0]["context_used"] == "N/A"


class TestIndicatorFilters:
    """Tests for indicator filter helpers."""

    def test_filter_by_types(self):
        """Indicator type filter restricts DataFrame rows."""
        from src.dashboard.data.indicators import filter_by_types

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "indicator_type": [
                    "social_withdrawal",
                    "diminished_interest",
                    "fatigue",
                ],
                "likelihood": [0.75, 0.60, 0.45],
                "presence_flag": [True, False, None],
                "data_reliability_score": [0.85, 0.70, 0.60],
                "context_used": ["solitary_digital", "N/A", "N/A"],
                "analysis_run_id": [str(uuid.uuid4()) for _ in range(3)],
            }
        )

        result = filter_by_types(df, ["social_withdrawal"])
        assert len(result) == 1
        assert result.iloc[0]["indicator_type"] == "social_withdrawal"

    def test_filter_by_types_empty_list_returns_all(self):
        """Empty filter list returns all rows."""
        from src.dashboard.data.indicators import filter_by_types

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 3,
                "indicator_type": [
                    "social_withdrawal",
                    "diminished_interest",
                    "fatigue",
                ],
                "likelihood": [0.75, 0.60, 0.45],
                "presence_flag": [True, False, None],
                "data_reliability_score": [0.85, 0.70, 0.60],
                "context_used": ["solitary_digital", "N/A", "N/A"],
                "analysis_run_id": [str(uuid.uuid4()) for _ in range(3)],
            }
        )

        result = filter_by_types(df, [])
        assert len(result) == 3


class TestIndicatorExport:
    """Tests for CSV export functionality."""

    def test_generate_csv_filename(self):
        """CSV filename follows expected pattern."""
        from src.dashboard.data.indicators import generate_csv_filename

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        filename = generate_csv_filename("user123", start, end)

        assert filename == "indicators_user123_20240101_20240131.csv"

    def test_dataframe_to_csv(self):
        """DataFrame exports to valid CSV string."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)],
                "indicator_type": ["social_withdrawal"],
                "likelihood": [0.75],
                "presence_flag": [True],
                "data_reliability_score": [0.85],
                "context_used": ["solitary_digital"],
                "analysis_run_id": [str(uuid.uuid4())],
            }
        )

        csv_data = df.to_csv(index=False)
        assert "timestamp" in csv_data
        assert "social_withdrawal" in csv_data
        assert "0.75" in csv_data


class TestSummaryStatistics:
    """Tests for summary statistics calculation."""

    def test_calculate_summary_stats(self):
        """Summary statistics are calculated per indicator type."""
        from src.dashboard.data.indicators import calculate_summary_stats

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 4,
                "indicator_type": [
                    "social_withdrawal",
                    "social_withdrawal",
                    "diminished_interest",
                    "diminished_interest",
                ],
                "likelihood": [0.6, 0.8, 0.4, 0.5],
                "presence_flag": [True, True, False, None],
                "data_reliability_score": [0.85, 0.90, 0.70, 0.75],
                "context_used": ["solitary_digital"] * 4,
                "analysis_run_id": [str(uuid.uuid4()) for _ in range(4)],
            }
        )

        stats = calculate_summary_stats(df)

        assert len(stats) == 2  # Two unique indicator types
        assert "social_withdrawal" in stats["Indicator"].values
        assert "diminished_interest" in stats["Indicator"].values

        # Check social_withdrawal stats
        sw_row = stats[stats["Indicator"] == "social_withdrawal"].iloc[0]
        assert sw_row["Count"] == 2
        assert sw_row["Mean Likelihood"] == 0.7  # (0.6 + 0.8) / 2
        assert sw_row["% Present"] == 100.0  # Both have presence_flag = True

    def test_calculate_percent_present(self):
        """Correctly calculates percentage present."""
        from src.dashboard.data.indicators import calculate_summary_stats

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC)] * 4,
                "indicator_type": ["social_withdrawal"] * 4,
                "likelihood": [0.5, 0.6, 0.7, 0.8],
                "presence_flag": [True, True, False, None],
                "data_reliability_score": [0.85] * 4,
                "context_used": ["solitary_digital"] * 4,
                "analysis_run_id": [str(uuid.uuid4()) for _ in range(4)],
            }
        )

        stats = calculate_summary_stats(df)
        assert stats.iloc[0]["% Present"] == 50.0  # 2 out of 4

    def test_calculate_summary_stats_empty_dataframe(self):
        """Empty DataFrame returns empty stats."""
        from src.dashboard.data.indicators import calculate_summary_stats

        df = pd.DataFrame(
            columns=[
                "timestamp",
                "indicator_type",
                "likelihood",
                "presence_flag",
                "data_reliability_score",
                "context_used",
                "analysis_run_id",
            ]
        )

        stats = calculate_summary_stats(df)

        assert len(stats) == 0


class TestPagination:
    """Tests for pagination helpers."""

    def test_calculate_pagination_indices(self):
        """Pagination indices are calculated correctly."""
        from src.dashboard.data.indicators import calculate_page_indices

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
        from src.dashboard.data.indicators import calculate_total_pages

        assert calculate_total_pages(100, 25) == 4
        assert calculate_total_pages(101, 25) == 5
        assert calculate_total_pages(0, 25) == 1  # Minimum 1 page
        assert calculate_total_pages(24, 25) == 1
