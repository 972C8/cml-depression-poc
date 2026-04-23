"""Tests for analysis trigger UI page."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.baseline_config import BaselineDefinition, BaselineFile, BaselineMetadata


@pytest.fixture
def mock_baseline_config():
    """Create mock baseline config for testing."""
    return BaselineFile(
        metadata=BaselineMetadata(name="Test Baseline"),
        baselines={
            "speech_activity": BaselineDefinition(mean=0.5, std=0.15),
            "voice_energy": BaselineDefinition(mean=0.55, std=0.18),
            "connections": BaselineDefinition(mean=0.4, std=0.2),
            "bytes_out": BaselineDefinition(mean=0.45, std=0.25),
            "bytes_in": BaselineDefinition(mean=0.45, std=0.25),
            "speech_rate": BaselineDefinition(mean=0.5, std=0.2),
            "network_variety": BaselineDefinition(mean=0.35, std=0.2),
            "activity_level": BaselineDefinition(mean=0.5, std=0.2),
            "sleep_duration": BaselineDefinition(mean=0.5, std=0.15),
            "awakenings": BaselineDefinition(mean=0.3, std=0.15),
            "speech_activity_night": BaselineDefinition(mean=0.2, std=0.1),
        },
    )


class TestAnalysisPageModule:
    """Tests for analysis page module existence."""

    def test_analysis_page_exists(self):
        """Verify Analysis page exists."""
        analysis_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "pages"
            / "1_⚙️_Analysis.py"
        )
        assert analysis_path.exists()

    def test_analysis_data_module_exists(self):
        """Verify analysis data module exists."""
        analysis_data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "data"
            / "analysis.py"
        )
        assert analysis_data_path.exists()

    def test_analysis_actions_module_exists(self):
        """Verify analysis actions module exists."""
        analysis_actions_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "actions"
            / "analysis.py"
        )
        assert analysis_actions_path.exists()


class TestLoadAnalysisRuns:
    """Tests for analysis run loading."""

    def test_function_exists_and_callable(self):
        """load_analysis_runs function exists."""
        from src.dashboard.data.analysis import load_analysis_runs

        assert callable(load_analysis_runs)

    def test_returns_dataframe(self):
        """Returns pandas DataFrame."""
        from src.dashboard.data.analysis import load_analysis_runs

        with patch("src.dashboard.data.analysis.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_analysis_runs(user_id="test")
            assert isinstance(result, pd.DataFrame)

    def test_empty_result_has_correct_columns(self):
        """Empty result has expected columns."""
        from src.dashboard.data.analysis import load_analysis_runs

        with patch("src.dashboard.data.analysis.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = []

            result = load_analysis_runs(user_id="test")
            assert "run_id" in result.columns
            assert "user_id" in result.columns
            assert "start_time" in result.columns
            assert "end_time" in result.columns
            assert "created_at" in result.columns

    def test_maps_columns_correctly(self):
        """Analysis run fields are mapped to correct DataFrame columns."""
        from src.dashboard.data.analysis import load_analysis_runs

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.user_id = "test_user"
        mock_run.start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_run.end_time = datetime(2024, 1, 7, 12, 0, 0, tzinfo=UTC)
        mock_run.created_at = datetime(2024, 1, 7, 13, 0, 0, tzinfo=UTC)

        with patch("src.dashboard.data.analysis.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalars.return_value.all.return_value = [
                mock_run
            ]

            result = load_analysis_runs(user_id="test_user")

            assert len(result) == 1
            row = result.iloc[0]
            assert row["run_id"] == str(mock_run.id)
            assert row["user_id"] == "test_user"
            assert row["start_time"] == mock_run.start_time
            assert row["end_time"] == mock_run.end_time

    def test_handles_exception(self):
        """Database errors return empty DataFrame."""
        from src.dashboard.data.analysis import load_analysis_runs

        with patch("src.dashboard.data.analysis.SessionLocal") as mock_session:
            mock_session.return_value.__enter__.side_effect = Exception("DB Error")

            result = load_analysis_runs(user_id="test")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0


class TestGetAnalysisRunSummary:
    """Tests for analysis run summary retrieval."""

    def test_function_exists_and_callable(self):
        """get_analysis_run_summary function exists."""
        from src.dashboard.data.analysis import get_analysis_run_summary

        assert callable(get_analysis_run_summary)

    def test_returns_dict_for_valid_run(self):
        """Returns dict for existing run."""
        from src.dashboard.data.analysis import get_analysis_run_summary

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()
        mock_run.user_id = "test_user"
        mock_run.start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        mock_run.end_time = datetime(2024, 1, 7, 12, 0, 0, tzinfo=UTC)
        mock_run.created_at = datetime(2024, 1, 7, 13, 0, 0, tzinfo=UTC)
        mock_run.config_snapshot = {"version": "1.0"}
        mock_run.pipeline_trace = {"steps": []}

        with patch("src.dashboard.data.analysis.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalar_one_or_none.return_value = mock_run

            result = get_analysis_run_summary(str(mock_run.id))

            assert result is not None
            assert result["run_id"] == str(mock_run.id)
            assert result["user_id"] == "test_user"
            assert result["config_snapshot"] == {"version": "1.0"}

    def test_returns_none_for_invalid_id(self):
        """Returns None for non-existent run."""
        from src.dashboard.data.analysis import get_analysis_run_summary

        with patch("src.dashboard.data.analysis.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_db
            mock_db.execute.return_value.scalar_one_or_none.return_value = None

            result = get_analysis_run_summary("00000000-0000-0000-0000-000000000000")
            assert result is None


class TestTriggerAnalysis:
    """Tests for analysis trigger action."""

    def test_function_exists_and_callable(self):
        """trigger_analysis function exists."""
        from src.dashboard.actions.analysis import trigger_analysis

        assert callable(trigger_analysis)

    def test_returns_result_object(self, mock_baseline_config):
        """Returns AnalysisTriggerResult."""
        from src.dashboard.actions.analysis import (
            AnalysisTriggerResult,
            trigger_analysis,
        )

        with patch("src.core.analysis.run_analysis") as mock_run:
            # Mock WindowedAnalysisResult structure
            mock_summary = MagicMock()
            mock_summary.likelihood = 0.6
            mock_summary.average_biomarker_completeness = 0.8
            mock_summary.indicator_name = "social_withdrawal"
            mock_summary.episodes = []

            mock_result = MagicMock()
            mock_result.run_id = "test-uuid"
            mock_result.user_id = "test"
            mock_result.duration_ms = 1000
            mock_result.window_count = 10
            mock_result.context_evaluations_added = 0
            mock_result.daily_summaries = (mock_summary,)
            mock_result.start_date = datetime.now(UTC).date()
            mock_result.end_date = datetime.now(UTC).date()
            mock_result.config_snapshot = {}
            mock_run.return_value = mock_result

            result = trigger_analysis(
                user_id="test",
                start_time=datetime.now(UTC) - timedelta(days=1),
                end_time=datetime.now(UTC),
                baseline_config=mock_baseline_config,
            )
            assert isinstance(result, AnalysisTriggerResult)
            assert result.success is True
            assert result.run_id == "test-uuid"
            assert result.indicator_count == 1  # Derived from daily_summaries

    def test_success_maps_all_fields(self, mock_baseline_config):
        """Successful result maps all fields from WindowedAnalysisResult."""
        from src.dashboard.actions.analysis import trigger_analysis

        with patch("src.core.analysis.run_analysis") as mock_run:
            start_time = datetime.now(UTC) - timedelta(days=1)
            end_time = datetime.now(UTC)

            # Create mock daily summaries with episodes for context detection
            mock_episode = MagicMock()
            mock_episode.dominant_context = "solitary_digital"

            mock_summaries = []
            for i in range(3):
                s = MagicMock()
                s.likelihood = 0.7  # Above 0.5 threshold
                s.average_biomarker_completeness = 0.75
                s.indicator_name = f"indicator_{i}"
                s.episodes = [mock_episode]
                mock_summaries.append(s)

            mock_result = MagicMock()
            mock_result.run_id = "run-uuid-123"
            mock_result.user_id = "user123"
            mock_result.duration_ms = 1500
            mock_result.window_count = 20
            mock_result.context_evaluations_added = 5
            mock_result.daily_summaries = tuple(mock_summaries)
            mock_result.start_date = start_time.date()
            mock_result.end_date = end_time.date()
            mock_result.config_snapshot = {}
            mock_run.return_value = mock_result

            result = trigger_analysis(
                user_id="user123",
                start_time=start_time,
                end_time=end_time,
                baseline_config=mock_baseline_config,
            )

            assert result.success is True
            assert result.run_id == "run-uuid-123"
            assert result.user_id == "user123"
            # New windowed fields
            assert result.window_count == 20
            assert result.context_evaluations_added == 5
            assert result.daily_summaries_count == 3
            assert result.peak_likelihood == pytest.approx(0.7)
            assert result.mean_likelihood == pytest.approx(0.7)
            assert result.duration_ms == 1500
            # Derived legacy fields
            assert result.indicator_count == 3  # 3 unique indicator names
            assert result.avg_data_reliability == 0.75
            assert result.context_detected == "solitary_digital"
            assert result.episode_likely is True  # peak 0.7 >= 0.5
            assert result.indicators_present == 3  # All 3 have likelihood >= 0.5

    def test_handles_analysis_error(self, mock_baseline_config):
        """Handles AnalysisError gracefully."""
        from src.core.analysis import AnalysisError
        from src.dashboard.actions.analysis import trigger_analysis

        with patch("src.core.analysis.run_analysis") as mock_run:
            mock_run.side_effect = AnalysisError(
                "No data found",
                run_id=None,
                step="read_data",
            )

            result = trigger_analysis(
                user_id="test",
                start_time=datetime.now(UTC) - timedelta(days=1),
                end_time=datetime.now(UTC),
                baseline_config=mock_baseline_config,
            )
            assert result.success is False
            assert result.error_code == "ANALYSIS_ERROR"
            assert result.error_step == "read_data"
            assert "No data found" in result.error_message

    def test_handles_analysis_error_with_run_id(self, mock_baseline_config):
        """AnalysisError with run_id is preserved."""
        from src.core.analysis import AnalysisError
        from src.dashboard.actions.analysis import trigger_analysis

        run_id = uuid.uuid4()
        with patch("src.core.analysis.run_analysis") as mock_run:
            mock_run.side_effect = AnalysisError(
                "Pipeline failed",
                run_id=run_id,
                step="compute_indicators",
            )

            result = trigger_analysis(
                user_id="test",
                start_time=datetime.now(UTC) - timedelta(days=1),
                end_time=datetime.now(UTC),
                baseline_config=mock_baseline_config,
            )
            assert result.success is False
            assert result.run_id == str(run_id)
            assert result.error_step == "compute_indicators"

    def test_handles_unexpected_error(self, mock_baseline_config):
        """Handles unexpected exceptions gracefully."""
        from src.dashboard.actions.analysis import trigger_analysis

        with patch("src.core.analysis.run_analysis") as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected failure")

            result = trigger_analysis(
                user_id="test",
                start_time=datetime.now(UTC) - timedelta(days=1),
                end_time=datetime.now(UTC),
                baseline_config=mock_baseline_config,
            )
            assert result.success is False
            assert result.error_code == "UNEXPECTED_ERROR"
            assert "Unexpected" in result.error_message


class TestAnalysisTriggerResult:
    """Tests for AnalysisTriggerResult dataclass."""

    def test_dataclass_exists(self):
        """AnalysisTriggerResult class exists."""
        from src.dashboard.actions.analysis import AnalysisTriggerResult

        assert AnalysisTriggerResult is not None

    def test_default_values(self):
        """Dataclass has sensible defaults."""
        from src.dashboard.actions.analysis import AnalysisTriggerResult

        result = AnalysisTriggerResult(success=True)

        assert result.success is True
        assert result.run_id is None
        assert result.indicator_count == 0
        assert result.avg_data_reliability == 0.0
        assert result.duration_ms == 0
        assert result.episode_likely is False

    def test_error_result(self):
        """Can create error result."""
        from src.dashboard.actions.analysis import AnalysisTriggerResult

        result = AnalysisTriggerResult(
            success=False,
            error_code="TEST_ERROR",
            error_message="Test error message",
            error_step="test_step",
        )

        assert result.success is False
        assert result.error_code == "TEST_ERROR"
        assert result.error_message == "Test error message"
        assert result.error_step == "test_step"
