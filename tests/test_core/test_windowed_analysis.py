"""Integration tests for analysis pipeline.

Story 6.6: Pipeline Integration & Orchestration (AC6)

Tests the complete analysis pipeline including:
- WindowedAnalysisResult dataclass
- run_analysis orchestrator
- Error handling at each step
- Sparse data handling
"""

from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import pytest

from src.core.analysis import (
    AnalysisError,
    WindowedAnalysisResult,
    run_analysis,
)
from src.core.baseline_config import BaselineDefinition, BaselineFile, BaselineMetadata
from src.core.config import get_default_config
from src.core.context.history import ContextHistoryStatus, EnsureHistoryResult
from src.core.data_reader import BiomarkerRecord, DataReaderResult, DataStats
from src.core.models.daily_summary import DailyIndicatorSummary
from src.core.models.window_models import (
    WindowAggregate,
    WindowIndicator,
    WindowMembership,
)


class TestWindowedAnalysisResultDataclass:
    """Tests for WindowedAnalysisResult dataclass (AC1)."""

    def test_dataclass_fields_exist(self) -> None:
        """Verify all required fields are present."""
        result = WindowedAnalysisResult(
            run_id="test-run-123",
            user_id="user-456",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7),
            daily_summaries=(),
            window_count=10,
            context_evaluations_added=5,
            duration_ms=1234,
            config_snapshot={"key": "value"},
        )

        assert result.run_id == "test-run-123"
        assert result.user_id == "user-456"
        assert result.start_date == date(2025, 1, 1)
        assert result.end_date == date(2025, 1, 7)
        assert result.daily_summaries == ()
        assert result.window_count == 10
        assert result.context_evaluations_added == 5
        assert result.duration_ms == 1234
        assert result.config_snapshot == {"key": "value"}

    def test_dataclass_is_frozen(self) -> None:
        """Verify dataclass is immutable (frozen)."""
        result = WindowedAnalysisResult(
            run_id="test-run-123",
            user_id="user-456",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7),
            daily_summaries=(),
            window_count=10,
            context_evaluations_added=5,
            duration_ms=1234,
            config_snapshot={},
        )

        with pytest.raises(FrozenInstanceError):
            result.run_id = "new-id"  # type: ignore

    def test_daily_summaries_is_tuple(self) -> None:
        """Verify daily_summaries uses tuple for immutability."""
        result = WindowedAnalysisResult(
            run_id="test-run-123",
            user_id="user-456",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7),
            daily_summaries=(),
            window_count=10,
            context_evaluations_added=5,
            duration_ms=1234,
            config_snapshot={},
        )

        assert isinstance(result.daily_summaries, tuple)


class TestRunAnalysisFunction:
    """Tests for run_analysis function (AC2)."""

    def test_function_signature_accepts_required_params(self) -> None:
        """Verify function accepts user_id, start_time, end_time."""
        # Verify function exists and has correct signature
        import inspect

        sig = inspect.signature(run_analysis)
        params = list(sig.parameters.keys())

        assert "user_id" in params
        assert "start_time" in params
        assert "end_time" in params
        assert "config" in params
        assert "session" in params

    def test_function_returns_windowed_analysis_result(self) -> None:
        """Verify function returns WindowedAnalysisResult type."""
        import inspect

        sig = inspect.signature(run_analysis)
        # Return annotation should be WindowedAnalysisResult
        assert sig.return_annotation == WindowedAnalysisResult


class TestAnalysisErrorStructure:
    """Tests for AnalysisError class structure."""

    def test_analysis_error_has_run_id_and_step(self) -> None:
        """Verify AnalysisError includes run_id and step information."""
        error = AnalysisError(
            "Test error",
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            step="read_data",
        )

        assert error.run_id == UUID("12345678-1234-5678-1234-567812345678")
        assert error.step == "read_data"
        assert "Test error" in str(error)
        assert "read_data" in str(error)


class TestSaveDailySummariesPersistence:
    """Tests for save_daily_summaries function (AC3)."""

    def test_function_exists_and_is_importable(self) -> None:
        """Verify save_daily_summaries can be imported."""
        from src.core.persistence import save_daily_summaries

        assert callable(save_daily_summaries)

    def test_function_signature(self) -> None:
        """Verify save_daily_summaries has correct signature."""
        import inspect

        from src.core.persistence import save_daily_summaries

        sig = inspect.signature(save_daily_summaries)
        params = list(sig.parameters.keys())

        assert "daily_summaries" in params
        assert "user_id" in params
        assert "analysis_run_id" in params
        assert "session" in params

    def test_returns_count(self) -> None:
        """Verify save_daily_summaries returns int count."""
        import inspect

        from src.core.persistence import save_daily_summaries

        sig = inspect.signature(save_daily_summaries)
        # Return type should be int
        assert sig.return_annotation == int


class TestWindowedAnalysisConfigSnapshot:
    """Tests for config snapshot in windowed analysis results."""

    def test_result_includes_config_snapshot(self) -> None:
        """Verify WindowedAnalysisResult includes config_snapshot dict."""
        result = WindowedAnalysisResult(
            run_id="test-run-123",
            user_id="user-456",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7),
            daily_summaries=(),
            window_count=10,
            context_evaluations_added=5,
            duration_ms=1234,
            config_snapshot={"window": {"size_minutes": 15}},
        )

        assert "window" in result.config_snapshot
        assert result.config_snapshot["window"]["size_minutes"] == 15

    def test_config_snapshot_is_dict(self) -> None:
        """Verify config_snapshot is a dict type."""
        result = WindowedAnalysisResult(
            run_id="test-run-123",
            user_id="user-456",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 7),
            daily_summaries=(),
            window_count=10,
            context_evaluations_added=5,
            duration_ms=1234,
            config_snapshot={},
        )

        assert isinstance(result.config_snapshot, dict)


# =============================================================================
# Integration Tests - Full Pipeline (AC6: Test full pipeline end-to-end)
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session."""
    session = MagicMock()
    session.commit = Mock()
    session.rollback = Mock()
    session.flush = Mock()
    return session


@pytest.fixture
def mock_data_result_with_biomarkers():
    """Create mock data reader result with sample biomarker data."""
    timestamp = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    biomarker = BiomarkerRecord(
        id="bio-1",
        user_id="test_user",
        timestamp=timestamp,
        biomarker_type="speech",
        name="speech_activity",
        value=0.6,
        raw_value={"speech_activity": 0.6},
        metadata=None,
    )

    return DataReaderResult(
        biomarkers=(biomarker,),
        context_markers=(),
        biomarkers_by_type={"speech": (biomarker,)},
        biomarkers_by_name={"speech_activity": (biomarker,)},
        context_by_name={},
        stats=DataStats(
            biomarker_count=1,
            context_count=0,
            time_range_start=timestamp,
            time_range_end=timestamp,
            biomarker_types_found=frozenset(["speech"]),
            biomarker_names_found=frozenset(["speech_activity"]),
            context_names_found=frozenset(),
        ),
    )


@pytest.fixture
def mock_ensure_history_result():
    """Create mock context history result."""
    return EnsureHistoryResult(
        status=ContextHistoryStatus.ALREADY_POPULATED,
        gaps_found=0,
        evaluations_added=0,
        message="Context history already exists for the entire range",
    )


@pytest.fixture
def mock_window_aggregates():
    """Create mock window aggregates."""
    window_start = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    window_end = datetime(2025, 1, 15, 10, 15, 0, tzinfo=UTC)

    return {
        "speech_activity": [
            WindowAggregate(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.6,
                readings_count=1,
                readings_timestamps=(window_start,),
                aggregation_method="mean",
            )
        ]
    }


@pytest.fixture
def mock_window_memberships():
    """Create mock window memberships."""
    window_start = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    window_end = datetime(2025, 1, 15, 10, 15, 0, tzinfo=UTC)

    return {
        "speech_activity": [
            WindowMembership(
                biomarker_name="speech_activity",
                window_start=window_start,
                window_end=window_end,
                aggregated_value=0.6,
                z_score=0.5,
                membership=0.7,
                context_strategy="dominant",
                context_state={"neutral": 1.0},
                dominant_context="neutral",
                context_weight=1.0,
                context_confidence=0.0,
                weighted_membership=0.7,
                readings_count=1,
            )
        ]
    }


@pytest.fixture
def mock_window_indicators():
    """Create mock window indicators."""
    window_start = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
    window_end = datetime(2025, 1, 15, 10, 15, 0, tzinfo=UTC)

    return [
        WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.65,
            contributing_biomarkers={"speech_activity": 0.7},
            biomarkers_present=1,
            biomarkers_expected=3,
            biomarker_completeness=0.33,
            dominant_context="neutral",
        )
    ]


@pytest.fixture
def mock_daily_summary():
    """Create mock daily summary."""
    return DailyIndicatorSummary(
        indicator_name="social_withdrawal",
        date=date(2025, 1, 15),
        likelihood=0.65,
        window_scores=(),
        total_windows=1,
        expected_windows=96,
        data_coverage=0.01,
        average_biomarker_completeness=0.33,
        context_availability=1.0,
    )


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


class TestRunWindowedAnalysisIntegration:
    """Integration tests for run_analysis orchestrator."""

    @patch("src.core.analysis.ContextHistoryService")
    @patch("src.core.analysis.DataReader")
    @patch("src.core.analysis.aggregate_into_windows")
    @patch("src.core.analysis.compute_window_memberships")
    @patch("src.core.analysis.compute_window_indicators")
    @patch("src.core.analysis.compute_daily_summary")
    @patch("src.core.analysis.save_daily_summaries")
    @patch("src.core.analysis.save_analysis_run")
    @patch("src.core.analysis.save_pipeline_trace")
    def test_run_analysis_full_pipeline(
        self,
        mock_save_trace,
        mock_save_run,
        mock_save_summaries,
        mock_compute_daily,
        mock_compute_indicators,
        mock_compute_memberships,
        mock_aggregate,
        MockDataReader,
        MockContextHistory,
        mock_session,
        mock_data_result_with_biomarkers,
        mock_ensure_history_result,
        mock_window_aggregates,
        mock_window_memberships,
        mock_window_indicators,
        mock_daily_summary,
        mock_baseline_config,
    ):
        """Test complete windowed analysis pipeline with mocked components (AC6)."""
        # Setup mocks
        mock_context_service = MockContextHistory.return_value
        mock_context_service.ensure_context_history_exists.return_value = (
            mock_ensure_history_result
        )

        mock_data_reader = MockDataReader.return_value
        mock_data_reader.read_all.return_value = mock_data_result_with_biomarkers

        mock_aggregate.return_value = mock_window_aggregates
        mock_compute_memberships.return_value = mock_window_memberships
        mock_compute_indicators.return_value = mock_window_indicators
        mock_compute_daily.return_value = mock_daily_summary
        mock_save_summaries.return_value = 1

        # Run analysis
        user_id = "test_user"
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC)
        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC)
        config = get_default_config()

        result = run_analysis(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            baseline_config=mock_baseline_config,
            config=config,
            session=mock_session,
        )

        # Verify result type and structure
        assert isinstance(result, WindowedAnalysisResult)
        assert result.user_id == user_id
        assert result.start_date == start_time.date()
        assert result.end_date == end_time.date()
        assert result.window_count == 1
        assert result.context_evaluations_added == 0
        assert result.duration_ms >= 0
        assert isinstance(result.config_snapshot, dict)
        # Default config has 3 indicators, so we get 3 daily summaries (one per indicator)
        assert len(result.daily_summaries) >= 1

        # Verify all pipeline steps were called
        mock_context_service.ensure_context_history_exists.assert_called_once()
        mock_data_reader.read_all.assert_called_once()
        mock_aggregate.assert_called_once()
        mock_compute_memberships.assert_called()
        mock_compute_indicators.assert_called()
        mock_compute_daily.assert_called()
        mock_save_summaries.assert_called_once()
        mock_save_run.assert_called_once()
        mock_save_trace.assert_called_once()


class TestRunWindowedAnalysisErrorHandling:
    """Real error handling tests for run_analysis (AC6)."""

    @patch("src.core.analysis.ContextHistoryService")
    @patch("src.core.analysis.DataReader")
    def test_raises_analysis_error_when_no_data_found(
        self,
        MockDataReader,
        MockContextHistory,
        mock_session,
        mock_ensure_history_result,
        mock_baseline_config,
    ):
        """Verify AnalysisError is raised when no biomarker data found."""
        # Setup mocks - context service works but no data
        mock_context_service = MockContextHistory.return_value
        mock_context_service.ensure_context_history_exists.return_value = (
            mock_ensure_history_result
        )

        mock_data_reader = MockDataReader.return_value
        mock_data_reader.read_all.return_value = DataReaderResult(
            biomarkers=(),
            context_markers=(),
            biomarkers_by_type={},
            biomarkers_by_name={},
            context_by_name={},
            stats=DataStats(
                biomarker_count=0,
                context_count=0,
                time_range_start=None,
                time_range_end=None,
                biomarker_types_found=frozenset(),
                biomarker_names_found=frozenset(),
                context_names_found=frozenset(),
            ),
        )

        # Run and expect error
        user_id = "test_user"
        end_time = datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC)
        start_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC)

        with pytest.raises(AnalysisError) as exc_info:
            run_analysis(
                user_id=user_id,
                start_time=start_time,
                end_time=end_time,
                baseline_config=mock_baseline_config,
                config=get_default_config(),
                session=mock_session,
            )

        error = exc_info.value
        assert "No data found" in str(error)
        assert error.step == "read_data"
        assert error.run_id is not None

    @patch("src.core.analysis.ContextHistoryService")
    @patch("src.core.analysis.DataReader")
    @patch("src.core.analysis.aggregate_into_windows")
    def test_raises_analysis_error_when_no_windows_generated(
        self,
        mock_aggregate,
        MockDataReader,
        MockContextHistory,
        mock_session,
        mock_data_result_with_biomarkers,
        mock_ensure_history_result,
        mock_baseline_config,
    ):
        """Verify AnalysisError is raised when window aggregation produces no windows."""
        # Setup mocks
        mock_context_service = MockContextHistory.return_value
        mock_context_service.ensure_context_history_exists.return_value = (
            mock_ensure_history_result
        )

        mock_data_reader = MockDataReader.return_value
        mock_data_reader.read_all.return_value = mock_data_result_with_biomarkers

        # Window aggregation returns empty result
        mock_aggregate.return_value = {}

        # Run and expect error
        with pytest.raises(AnalysisError) as exc_info:
            run_analysis(
                user_id="test_user",
                start_time=datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC),
                end_time=datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC),
                baseline_config=mock_baseline_config,
                config=get_default_config(),
                session=mock_session,
            )

        error = exc_info.value
        assert "No windows generated" in str(error)
        assert error.step == "window_aggregation"


class TestWindowedAnalysisSparseData:
    """Tests for sparse data handling in windowed analysis (AC6)."""

    @patch("src.core.analysis.ContextHistoryService")
    @patch("src.core.analysis.DataReader")
    @patch("src.core.analysis.aggregate_into_windows")
    @patch("src.core.analysis.compute_window_memberships")
    @patch("src.core.analysis.compute_window_indicators")
    @patch("src.core.analysis.compute_daily_summary")
    @patch("src.core.analysis.save_daily_summaries")
    @patch("src.core.analysis.save_analysis_run")
    @patch("src.core.analysis.save_pipeline_trace")
    def test_handles_sparse_data_single_window(
        self,
        mock_save_trace,
        mock_save_run,
        mock_save_summaries,
        mock_compute_daily,
        mock_compute_indicators,
        mock_compute_memberships,
        mock_aggregate,
        MockDataReader,
        MockContextHistory,
        mock_session,
        mock_ensure_history_result,
        mock_baseline_config,
    ):
        """Test pipeline handles sparse data with only one window (AC6)."""
        # Setup minimal data - one biomarker reading
        timestamp = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        biomarker = BiomarkerRecord(
            id="bio-sparse",
            user_id="test_user",
            timestamp=timestamp,
            biomarker_type="speech",
            name="speech_activity",
            value=0.5,
            raw_value={"speech_activity": 0.5},
            metadata=None,
        )

        sparse_data_result = DataReaderResult(
            biomarkers=(biomarker,),
            context_markers=(),
            biomarkers_by_type={"speech": (biomarker,)},
            biomarkers_by_name={"speech_activity": (biomarker,)},
            context_by_name={},
            stats=DataStats(
                biomarker_count=1,
                context_count=0,
                time_range_start=timestamp,
                time_range_end=timestamp,
                biomarker_types_found=frozenset(["speech"]),
                biomarker_names_found=frozenset(["speech_activity"]),
                context_names_found=frozenset(),
            ),
        )

        # Configure mocks
        mock_context_service = MockContextHistory.return_value
        mock_context_service.ensure_context_history_exists.return_value = (
            mock_ensure_history_result
        )

        mock_data_reader = MockDataReader.return_value
        mock_data_reader.read_all.return_value = sparse_data_result

        # Single window from sparse data
        window_start = timestamp
        window_end = timestamp + timedelta(minutes=15)
        mock_aggregate.return_value = {
            "speech_activity": [
                WindowAggregate(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.5,
                    readings_count=1,
                    readings_timestamps=(timestamp,),
                    aggregation_method="mean",
                )
            ]
        }

        mock_compute_memberships.return_value = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.5,
                    z_score=0.0,
                    membership=0.5,
                    context_strategy="dominant",
                    context_state={"neutral": 1.0},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.5,
                    readings_count=1,
                )
            ]
        }

        sparse_indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.5,
            contributing_biomarkers={"speech_activity": 0.5},
            biomarkers_present=1,
            biomarkers_expected=3,
            biomarker_completeness=0.33,
            dominant_context="neutral",
        )
        mock_compute_indicators.return_value = [sparse_indicator]

        sparse_summary = DailyIndicatorSummary(
            indicator_name="social_withdrawal",
            date=timestamp.date(),
            likelihood=0.0,
            window_scores=(),
            total_windows=1,
            expected_windows=96,
            data_coverage=0.01,  # Very low coverage - sparse!
            average_biomarker_completeness=0.33,
            context_availability=1.0,
        )
        mock_compute_daily.return_value = sparse_summary
        mock_save_summaries.return_value = 1

        # Run analysis
        result = run_analysis(
            user_id="test_user",
            start_time=datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC),
            end_time=datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC),
            baseline_config=mock_baseline_config,
            config=get_default_config(),
            session=mock_session,
        )

        # Verify sparse data handling
        assert isinstance(result, WindowedAnalysisResult)
        assert result.window_count == 1  # Only one window from sparse data
        # Default config has 3 indicators, so we get summaries for each
        assert len(result.daily_summaries) >= 1
        # Data coverage should reflect sparse nature (check first summary)
        assert result.daily_summaries[0].data_coverage < 0.1


class TestSaveWindowIndicatorsPersistence:
    """Tests for save_window_indicators function (AC3)."""

    def test_save_window_indicators_exists_and_importable(self) -> None:
        """Verify save_window_indicators can be imported."""
        from src.core.persistence import save_window_indicators

        assert callable(save_window_indicators)

    def test_save_window_indicators_signature(self) -> None:
        """Verify save_window_indicators has correct signature."""
        import inspect

        from src.core.persistence import save_window_indicators

        sig = inspect.signature(save_window_indicators)
        params = list(sig.parameters.keys())

        assert "window_indicators" in params
        assert "user_id" in params
        assert "analysis_run_id" in params
        assert "session" in params

    def test_save_window_indicators_returns_count(self) -> None:
        """Verify save_window_indicators returns int count."""
        import inspect

        from src.core.persistence import save_window_indicators

        sig = inspect.signature(save_window_indicators)
        assert sig.return_annotation == int

    def test_save_window_indicators_empty_list(self, mock_session) -> None:
        """Verify save_window_indicators handles empty list."""
        from uuid import uuid4

        from src.core.persistence import save_window_indicators

        result = save_window_indicators(
            window_indicators=[],
            user_id="test_user",
            analysis_run_id=uuid4(),
            session=mock_session,
        )

        assert result == 0
        mock_session.add.assert_not_called()


class TestContextEvaluationRunIdParameter:
    """Tests for context_evaluation_run_id parameter in run_analysis (Story 6.14)."""

    def test_run_analysis_accepts_context_evaluation_run_id_param(self) -> None:
        """Verify run_analysis accepts context_evaluation_run_id parameter (AC1)."""
        import inspect

        sig = inspect.signature(run_analysis)
        params = list(sig.parameters.keys())

        assert "context_evaluation_run_id" in params

    def test_run_analysis_context_evaluation_run_id_is_optional(self) -> None:
        """Verify context_evaluation_run_id parameter is optional with None default."""
        import inspect

        sig = inspect.signature(run_analysis)
        param = sig.parameters["context_evaluation_run_id"]

        assert param.default is None

    @patch("src.core.analysis.ContextHistoryService")
    @patch("src.core.analysis.DataReader")
    @patch("src.core.analysis.aggregate_into_windows")
    @patch("src.core.analysis.compute_window_memberships")
    @patch("src.core.analysis.compute_window_indicators")
    @patch("src.core.analysis.compute_daily_summary")
    @patch("src.core.analysis.save_daily_summaries")
    @patch("src.core.analysis.save_analysis_run")
    @patch("src.core.analysis.save_pipeline_trace")
    def test_run_analysis_with_run_id_uses_selected_run_service(
        self,
        mock_save_trace,
        mock_save_run,
        mock_save_summaries,
        mock_compute_daily,
        mock_compute_indicators,
        mock_compute_memberships,
        mock_aggregate,
        MockDataReader,
        MockContextHistory,
        mock_session,
        mock_data_result_with_biomarkers,
        mock_ensure_history_result,
        mock_baseline_config,
    ):
        """Test that providing context_evaluation_run_id creates service with filter (AC1, AC3)."""
        from uuid import uuid4

        run_id = uuid4()

        # Mock the coverage check
        mock_coverage = MagicMock()
        mock_coverage.dates_covered = 1
        mock_coverage.missing_dates = []
        mock_coverage.coverage_ratio = 1.0

        mock_context_service = MockContextHistory.return_value
        mock_context_service.check_context_coverage.return_value = mock_coverage

        mock_data_reader = MockDataReader.return_value
        mock_data_reader.read_all.return_value = mock_data_result_with_biomarkers

        # Setup other mocks
        window_start = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 10, 15, 0, tzinfo=UTC)
        mock_aggregate.return_value = {
            "speech_activity": [
                WindowAggregate(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.6,
                    readings_count=1,
                    readings_timestamps=(window_start,),
                    aggregation_method="mean",
                )
            ]
        }
        mock_compute_memberships.return_value = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.6,
                    z_score=0.0,
                    membership=0.5,
                    context_strategy="dominant",
                    context_state={"neutral": 1.0},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.5,
                    readings_count=1,
                )
            ]
        }
        mock_indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.5,
            contributing_biomarkers={"speech_activity": 0.5},
            biomarkers_present=1,
            biomarkers_expected=3,
            biomarker_completeness=0.33,
            dominant_context="neutral",
        )
        mock_compute_indicators.return_value = [mock_indicator]
        mock_summary = DailyIndicatorSummary(
            indicator_name="social_withdrawal",
            date=window_start.date(),
            likelihood=0.0,
            window_scores=(),
            total_windows=1,
            expected_windows=96,
            data_coverage=0.01,
            average_biomarker_completeness=0.33,
            context_availability=1.0,
        )
        mock_compute_daily.return_value = mock_summary
        mock_save_summaries.return_value = 1

        # Run analysis with context_evaluation_run_id
        result = run_analysis(
            user_id="test_user",
            start_time=datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC),
            end_time=datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC),
            baseline_config=mock_baseline_config,
            config=get_default_config(),
            session=mock_session,
            context_evaluation_run_id=run_id,
        )

        # Verify ContextHistoryService was created with run_id
        MockContextHistory.assert_called()
        call_kwargs = MockContextHistory.call_args.kwargs
        assert "context_evaluation_run_id" in call_kwargs
        assert call_kwargs["context_evaluation_run_id"] == run_id

        # Verify check_context_coverage was called (instead of ensure_context_history_exists)
        mock_context_service.check_context_coverage.assert_called_once()
        # ensure_context_history_exists should NOT be called when run_id is provided
        mock_context_service.ensure_context_history_exists.assert_not_called()

        assert isinstance(result, WindowedAnalysisResult)

    @patch("src.core.analysis.ContextHistoryService")
    @patch("src.core.analysis.DataReader")
    @patch("src.core.analysis.aggregate_into_windows")
    @patch("src.core.analysis.compute_window_memberships")
    @patch("src.core.analysis.compute_window_indicators")
    @patch("src.core.analysis.compute_daily_summary")
    @patch("src.core.analysis.save_daily_summaries")
    @patch("src.core.analysis.save_analysis_run")
    @patch("src.core.analysis.save_pipeline_trace")
    def test_run_analysis_without_run_id_calls_ensure_history(
        self,
        mock_save_trace,
        mock_save_run,
        mock_save_summaries,
        mock_compute_daily,
        mock_compute_indicators,
        mock_compute_memberships,
        mock_aggregate,
        MockDataReader,
        MockContextHistory,
        mock_session,
        mock_data_result_with_biomarkers,
        mock_ensure_history_result,
        mock_baseline_config,
    ):
        """Test that without context_evaluation_run_id, ensure_context_history_exists is called."""
        mock_context_service = MockContextHistory.return_value
        mock_context_service.ensure_context_history_exists.return_value = mock_ensure_history_result

        mock_data_reader = MockDataReader.return_value
        mock_data_reader.read_all.return_value = mock_data_result_with_biomarkers

        # Setup other mocks
        window_start = datetime(2025, 1, 15, 10, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 10, 15, 0, tzinfo=UTC)
        mock_aggregate.return_value = {
            "speech_activity": [
                WindowAggregate(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.6,
                    readings_count=1,
                    readings_timestamps=(window_start,),
                    aggregation_method="mean",
                )
            ]
        }
        mock_compute_memberships.return_value = {
            "speech_activity": [
                WindowMembership(
                    biomarker_name="speech_activity",
                    window_start=window_start,
                    window_end=window_end,
                    aggregated_value=0.6,
                    z_score=0.0,
                    membership=0.5,
                    context_strategy="dominant",
                    context_state={"neutral": 1.0},
                    dominant_context="neutral",
                    context_weight=1.0,
                    context_confidence=0.0,
                    weighted_membership=0.5,
                    readings_count=1,
                )
            ]
        }
        mock_indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name="social_withdrawal",
            indicator_score=0.5,
            contributing_biomarkers={"speech_activity": 0.5},
            biomarkers_present=1,
            biomarkers_expected=3,
            biomarker_completeness=0.33,
            dominant_context="neutral",
        )
        mock_compute_indicators.return_value = [mock_indicator]
        mock_summary = DailyIndicatorSummary(
            indicator_name="social_withdrawal",
            date=window_start.date(),
            likelihood=0.0,
            window_scores=(),
            total_windows=1,
            expected_windows=96,
            data_coverage=0.01,
            average_biomarker_completeness=0.33,
            context_availability=1.0,
        )
        mock_compute_daily.return_value = mock_summary
        mock_save_summaries.return_value = 1

        # Run analysis WITHOUT context_evaluation_run_id
        result = run_analysis(
            user_id="test_user",
            start_time=datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC),
            end_time=datetime(2025, 1, 15, 23, 59, 59, tzinfo=UTC),
            baseline_config=mock_baseline_config,
            config=get_default_config(),
            session=mock_session,
            # context_evaluation_run_id not provided
        )

        # Verify ensure_context_history_exists was called
        mock_context_service.ensure_context_history_exists.assert_called_once()
        # check_context_coverage should NOT be called when no run_id
        mock_context_service.check_context_coverage.assert_not_called()

        assert isinstance(result, WindowedAnalysisResult)
