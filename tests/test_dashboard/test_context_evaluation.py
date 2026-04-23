"""Tests for context evaluation data loading from context history.

Story 6.9: Context Page Migration to Context History Service
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestContextEvaluationModule:
    """Tests for context evaluation page module existence."""

    def test_context_page_exists(self):
        """Verify unified Context page exists."""
        page_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "pages"
            / "4_📋_Context.py"
        )
        assert page_path.exists()

    def test_context_evaluation_data_module_exists(self):
        """Verify context evaluation data module exists."""
        data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "data"
            / "context_evaluation.py"
        )
        assert data_path.exists()


class TestLoadContextHistoryRecords:
    """Tests for loading context history records from database."""

    def test_function_exists_and_callable(self):
        """load_context_history_records function exists."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        assert callable(load_context_history_records)

    def test_returns_dataframe(self):
        """Returns pandas DataFrame."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_context_history_records(
                user_id="test",
                start=datetime.now(UTC) - timedelta(days=1),
                end=datetime.now(UTC),
            )
            assert isinstance(result, pd.DataFrame)

    def test_empty_records_returns_empty_df(self):
        """No records returns empty DataFrame with correct columns."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_context_history_records(
                user_id="test",
                start=datetime.now(UTC) - timedelta(days=1),
                end=datetime.now(UTC),
            )
            assert result.empty
            assert "dominant_context" in result.columns
            assert "confidence" in result.columns
            assert "evaluated_at" in result.columns
            assert "context_state" in result.columns

    def test_handles_exception(self):
        """Database errors return empty DataFrame."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = load_context_history_records(
                user_id="test",
                start=datetime.now(UTC) - timedelta(days=1),
                end=datetime.now(UTC),
            )
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_converts_records_to_dataframe(self):
        """Properly converts ContextHistoryRecord objects to DataFrame."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        # Create mock records
        mock_record = MagicMock()
        mock_record.evaluated_at = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        mock_record.dominant_context = "solitary_digital"
        mock_record.confidence = 0.8
        mock_record.context_state = {"solitary_digital": 0.8, "neutral": 0.2}
        mock_record.evaluation_trigger = "backfill"
        mock_record.sensors_used = ["people_in_room"]
        mock_record.sensor_snapshot = {"people_in_room": 3.0}

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_record
            ]

            result = load_context_history_records(
                user_id="test",
                start=datetime.now(UTC) - timedelta(days=1),
                end=datetime.now(UTC),
            )

            assert len(result) == 1
            assert result.iloc[0]["dominant_context"] == "solitary_digital"
            assert result.iloc[0]["confidence"] == 0.8


class TestGetContextHistoryStatus:
    """Tests for getting context history status with ensure call."""

    def test_function_exists_and_callable(self):
        """get_context_history_status function exists."""
        from src.dashboard.data.context_evaluation import get_context_history_status

        assert callable(get_context_history_status)

    def test_returns_result_dict(self):
        """Returns dict with status information."""
        from src.dashboard.data.context_evaluation import get_context_history_status

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db

            with patch(
                "src.dashboard.data.context_evaluation.ContextHistoryService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_service.return_value = mock_instance

                # Mock EnsureHistoryResult
                mock_result = MagicMock()
                mock_result.status.value = "already_populated"
                mock_result.gaps_found = 0
                mock_result.evaluations_added = 0
                mock_result.message = "Already populated"
                mock_instance.ensure_context_history_exists.return_value = mock_result

                result = get_context_history_status(
                    user_id="test",
                    start=datetime.now(UTC) - timedelta(days=1),
                    end=datetime.now(UTC),
                )

                assert isinstance(result, dict)
                assert "status" in result
                assert "gaps_found" in result
                assert "evaluations_added" in result
                assert "message" in result

    def test_calls_ensure_context_history(self):
        """Calls ContextHistoryService.ensure_context_history_exists."""
        from src.dashboard.data.context_evaluation import get_context_history_status

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db

            with patch(
                "src.dashboard.data.context_evaluation.ContextHistoryService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_service.return_value = mock_instance

                mock_result = MagicMock()
                mock_result.status.value = "evaluations_added"
                mock_result.gaps_found = 2
                mock_result.evaluations_added = 5
                mock_result.message = "Added 5 evaluations"
                mock_instance.ensure_context_history_exists.return_value = mock_result

                start = datetime.now(UTC) - timedelta(days=1)
                end = datetime.now(UTC)

                get_context_history_status(
                    user_id="test_user",
                    start=start,
                    end=end,
                )

                mock_instance.ensure_context_history_exists.assert_called_once_with(
                    "test_user", start, end
                )

    def test_handles_no_sensor_data(self):
        """Returns appropriate status when no sensor data available."""
        from src.dashboard.data.context_evaluation import get_context_history_status

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db

            with patch(
                "src.dashboard.data.context_evaluation.ContextHistoryService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_service.return_value = mock_instance

                mock_result = MagicMock()
                mock_result.status.value = "no_sensor_data"
                mock_result.gaps_found = 1
                mock_result.evaluations_added = 0
                mock_result.message = "No sensor data"
                mock_instance.ensure_context_history_exists.return_value = mock_result

                result = get_context_history_status(
                    user_id="test",
                    start=datetime.now(UTC) - timedelta(days=1),
                    end=datetime.now(UTC),
                )

                assert result["status"] == "no_sensor_data"


class TestDetectContextTransitions:
    """Tests for context transition detection using history records."""

    def test_detects_single_transition(self):
        """Correctly detects a single context transition."""
        from src.dashboard.data.context_evaluation import detect_context_transitions

        df = pd.DataFrame(
            {
                "evaluated_at": [
                    datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 15, tzinfo=UTC),
                ],
                "dominant_context": ["solitary_digital", "neutral"],
            }
        )

        transitions = detect_context_transitions(df)
        assert len(transitions) == 1
        assert transitions.iloc[0]["from_context"] == "solitary_digital"
        assert transitions.iloc[0]["to_context"] == "neutral"

    def test_no_transitions_empty_df(self):
        """No transitions returns empty DataFrame."""
        from src.dashboard.data.context_evaluation import detect_context_transitions

        df = pd.DataFrame(
            {
                "evaluated_at": [
                    datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 15, tzinfo=UTC),
                ],
                "dominant_context": ["solitary_digital", "solitary_digital"],
            }
        )

        transitions = detect_context_transitions(df)
        assert transitions.empty

    def test_empty_dataframe_returns_empty(self):
        """Empty input returns empty transitions."""
        from src.dashboard.data.context_evaluation import detect_context_transitions

        df = pd.DataFrame(columns=["evaluated_at", "dominant_context"])
        transitions = detect_context_transitions(df)
        assert transitions.empty

    def test_single_row_returns_empty(self):
        """Single row returns empty (no transitions possible)."""
        from src.dashboard.data.context_evaluation import detect_context_transitions

        df = pd.DataFrame(
            {
                "evaluated_at": [datetime(2024, 1, 1, 10, 0, tzinfo=UTC)],
                "dominant_context": ["solitary_digital"],
            }
        )
        transitions = detect_context_transitions(df)
        assert transitions.empty

    def test_multiple_transitions(self):
        """Correctly detects multiple transitions."""
        from src.dashboard.data.context_evaluation import detect_context_transitions

        df = pd.DataFrame(
            {
                "evaluated_at": [
                    datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 15, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 30, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 45, tzinfo=UTC),
                ],
                "dominant_context": [
                    "solitary_digital",
                    "neutral",
                    "solitary_digital",
                    "neutral",
                ],
            }
        )

        transitions = detect_context_transitions(df)
        assert len(transitions) == 3

    def test_duration_calculation(self):
        """Duration in previous context is calculated correctly."""
        from src.dashboard.data.context_evaluation import detect_context_transitions

        df = pd.DataFrame(
            {
                "evaluated_at": [
                    datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                    datetime(2024, 1, 1, 10, 30, tzinfo=UTC),  # 30 min later
                ],
                "dominant_context": ["solitary_digital", "neutral"],
            }
        )

        transitions = detect_context_transitions(df)
        assert transitions.iloc[0]["duration_minutes"] == 30.0


class TestCalculateContextDistribution:
    """Tests for context distribution calculation using history records."""

    def test_calculates_percentages(self):
        """Correctly calculates context time distribution."""
        from src.dashboard.data.context_evaluation import calculate_context_distribution

        df = pd.DataFrame(
            {
                "dominant_context": [
                    "solitary_digital",
                    "solitary_digital",
                    "neutral",
                    "neutral",
                ],
            }
        )

        dist = calculate_context_distribution(df)
        assert len(dist) == 2

        # solitary_digital should be 50% (2 out of 4)
        solitary_row = dist[dist["context"] == "solitary_digital"].iloc[0]
        assert solitary_row["percentage"] == 50.0
        assert solitary_row["count"] == 2

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty distribution."""
        from src.dashboard.data.context_evaluation import calculate_context_distribution

        df = pd.DataFrame(columns=["dominant_context"])
        dist = calculate_context_distribution(df)
        assert dist.empty

    def test_single_context(self):
        """Single context returns 100%."""
        from src.dashboard.data.context_evaluation import calculate_context_distribution

        df = pd.DataFrame({"dominant_context": ["neutral"] * 10})

        dist = calculate_context_distribution(df)
        assert len(dist) == 1
        assert dist.iloc[0]["percentage"] == 100.0


class TestContextEvaluationExport:
    """Tests for CSV export functionality."""

    def test_generate_csv_filename(self):
        """CSV filename follows expected pattern."""
        from src.dashboard.data.context_evaluation import (
            generate_evaluation_csv_filename,
        )

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        filename = generate_evaluation_csv_filename("user123", start, end)

        assert filename == "context_eval_user123_20240101_20240131.csv"


class TestContextEvaluationPagination:
    """Tests for pagination helpers."""

    def test_calculate_pagination_indices(self):
        """Pagination indices are calculated correctly."""
        from src.dashboard.data.context_evaluation import calculate_page_indices

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
        from src.dashboard.data.context_evaluation import calculate_total_pages

        assert calculate_total_pages(100, 25) == 4
        assert calculate_total_pages(101, 25) == 5
        assert calculate_total_pages(0, 25) == 1  # Minimum 1 page
        assert calculate_total_pages(24, 25) == 1

    def test_last_page_indices(self):
        """Last page correctly caps at total rows."""
        from src.dashboard.data.context_evaluation import calculate_page_indices

        # Page 4 of 90 rows with page size 25 (should only have 15 rows)
        start, end = calculate_page_indices(current_page=4, page_size=25, total_rows=90)
        assert start == 75
        assert end == 90


class TestOldFunctionsRemoved:
    """Tests to verify old functions have been removed."""

    def test_run_context_evaluation_history_removed(self):
        """run_context_evaluation_history should not exist (deleted)."""
        with pytest.raises(ImportError):
            from src.dashboard.data.context_evaluation import (  # noqa: F401
                run_context_evaluation_history,
            )

    def test_context_evaluation_point_removed(self):
        """ContextEvaluationPoint should not exist (deleted)."""
        with pytest.raises(ImportError):
            from src.dashboard.data.context_evaluation import (  # noqa: F401
                ContextEvaluationPoint,
            )

    def test_group_markers_by_time_removed(self):
        """_group_markers_by_time should not exist (deleted)."""
        with pytest.raises(ImportError):
            from src.dashboard.data.context_evaluation import (  # noqa: F401
                _group_markers_by_time,
            )

    def test_empty_evaluation_df_removed(self):
        """_empty_evaluation_df should not exist (deleted)."""
        with pytest.raises(ImportError):
            from src.dashboard.data.context_evaluation import (  # noqa: F401
                _empty_evaluation_df,
            )


class TestEvaluationTriggerTypes:
    """Tests for handling different evaluation_trigger types (AC #5)."""

    def test_loads_backfill_trigger_records(self):
        """Correctly loads records with backfill trigger."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        mock_record = MagicMock()
        mock_record.evaluated_at = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        mock_record.dominant_context = "neutral"
        mock_record.confidence = 0.6
        mock_record.context_state = {"neutral": 0.6}
        mock_record.evaluation_trigger = "backfill"
        mock_record.sensors_used = ["people_in_room"]
        mock_record.sensor_snapshot = {"people_in_room": 1.0}

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_record
            ]

            result = load_context_history_records(
                user_id="test",
                start=datetime.now(UTC) - timedelta(days=1),
                end=datetime.now(UTC),
            )

            assert result.iloc[0]["evaluation_trigger"] == "backfill"

    def test_loads_on_demand_trigger_records(self):
        """Correctly loads records with on_demand trigger."""
        from src.dashboard.data.context_evaluation import load_context_history_records

        mock_record = MagicMock()
        mock_record.evaluated_at = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        mock_record.dominant_context = "solitary_digital"
        mock_record.confidence = 0.8
        mock_record.context_state = {"solitary_digital": 0.8}
        mock_record.evaluation_trigger = "on_demand"
        mock_record.sensors_used = ["people_in_room", "ambient_noise"]
        mock_record.sensor_snapshot = {"people_in_room": 5.0, "ambient_noise": 60.0}

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_record
            ]

            result = load_context_history_records(
                user_id="test",
                start=datetime.now(UTC) - timedelta(days=1),
                end=datetime.now(UTC),
            )

            assert result.iloc[0]["evaluation_trigger"] == "on_demand"


class TestIntegrationEnsureHistoryExists:
    """Integration tests for ensure_context_history_exists call (AC #2)."""

    def test_commits_when_evaluations_added(self):
        """Calls db.commit() when evaluations were added."""
        from src.dashboard.data.context_evaluation import get_context_history_status

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db

            with patch(
                "src.dashboard.data.context_evaluation.ContextHistoryService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_service.return_value = mock_instance

                mock_result = MagicMock()
                mock_result.status.value = "evaluations_added"
                mock_result.gaps_found = 3
                mock_result.evaluations_added = 10
                mock_result.message = "Added 10 evaluations"
                mock_instance.ensure_context_history_exists.return_value = mock_result

                get_context_history_status(
                    user_id="test",
                    start=datetime.now(UTC) - timedelta(days=1),
                    end=datetime.now(UTC),
                )

                # Verify commit was called since evaluations were added
                mock_db.commit.assert_called_once()

    def test_no_commit_when_already_populated(self):
        """Does not commit when history was already populated."""
        from src.dashboard.data.context_evaluation import get_context_history_status

        with patch(
            "src.dashboard.data.context_evaluation.SessionLocal"
        ) as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db

            with patch(
                "src.dashboard.data.context_evaluation.ContextHistoryService"
            ) as mock_service:
                mock_instance = MagicMock()
                mock_service.return_value = mock_instance

                mock_result = MagicMock()
                mock_result.status.value = "already_populated"
                mock_result.gaps_found = 0
                mock_result.evaluations_added = 0
                mock_result.message = "Already populated"
                mock_instance.ensure_context_history_exists.return_value = mock_result

                get_context_history_status(
                    user_id="test",
                    start=datetime.now(UTC) - timedelta(days=1),
                    end=datetime.now(UTC),
                )

                # Verify commit was NOT called since no evaluations were added
                mock_db.commit.assert_not_called()
