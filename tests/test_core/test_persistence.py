"""Tests for indicator persistence module.

Tests cover:
- _extract_modalities: mapping biomarkers to modalities
- _build_computation_log: creating JSON computation trace
- save_indicator: persisting single indicator
- save_all_indicators: batch persistence with transaction safety
"""

import uuid
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

from src.core.config import get_default_config
from src.core.dsm_gate import IndicatorGateResult
from src.core.indicator_computation import (
    BiomarkerContribution,
    DailyIndicatorScore,
    IndicatorScore,
)
from src.shared.models import Indicator


class TestExtractModalities:
    """Test _extract_modalities helper function."""

    def test_speech_biomarkers_return_speech_modality(self):
        """Speech biomarkers should map to 'speech' modality."""
        from src.core.persistence import _extract_modalities

        biomarkers = ("speech_activity", "voice_energy", "speech_rate")
        result = _extract_modalities(biomarkers)

        assert result == ["speech"]

    def test_network_biomarkers_return_network_modality(self):
        """Network biomarkers should map to 'network' modality."""
        from src.core.persistence import _extract_modalities

        biomarkers = ("connections", "bytes_out", "bytes_in")
        result = _extract_modalities(biomarkers)

        assert result == ["network"]

    def test_mixed_biomarkers_return_multiple_modalities(self):
        """Mixed biomarkers should return multiple unique modalities."""
        from src.core.persistence import _extract_modalities

        biomarkers = ("speech_activity", "connections", "sleep_duration")
        result = _extract_modalities(biomarkers)

        # Should be sorted alphabetically
        assert result == ["network", "sleep", "speech"]

    def test_unknown_biomarker_returns_unknown_modality(self):
        """Unknown biomarkers should map to 'unknown' modality."""
        from src.core.persistence import _extract_modalities

        biomarkers = ("unknown_biomarker",)
        result = _extract_modalities(biomarkers)

        assert result == ["unknown"]

    def test_empty_biomarkers_return_empty_list(self):
        """Empty input should return empty list."""
        from src.core.persistence import _extract_modalities

        result = _extract_modalities(())

        assert result == []

    def test_duplicates_are_removed(self):
        """Multiple biomarkers of same modality should deduplicate."""
        from src.core.persistence import _extract_modalities

        biomarkers = (
            "speech_activity",
            "voice_energy",
            "speech_rate",
            "speech_activity_night",
        )
        result = _extract_modalities(biomarkers)

        assert result == ["speech"]


class TestBuildComputationLog:
    """Test _build_computation_log helper function."""

    @pytest.fixture
    def sample_indicator_score(self) -> IndicatorScore:
        """Create sample IndicatorScore for testing."""
        return IndicatorScore(
            indicator_name="social_withdrawal",
            daily_likelihood=0.65,
            contributions={
                "speech_activity": BiomarkerContribution(
                    name="speech_activity",
                    membership=0.75,
                    direction="contra",
                    base_weight=0.30,
                    context_multiplier=1.5,
                    adjusted_weight=0.36,
                    contribution=0.09,
                ),
                "connections": BiomarkerContribution(
                    name="connections",
                    membership=0.40,
                    direction="contra",
                    base_weight=0.25,
                    context_multiplier=0.7,
                    adjusted_weight=0.16,
                    contribution=0.096,
                ),
            },
            biomarkers_used=("speech_activity", "connections"),
            biomarkers_missing=(),
            data_reliability_score=0.85,
            context_applied="solitary_digital",
            context_confidence=0.9,
            weights_before_context={"speech_activity": 0.30, "connections": 0.25},
            weights_after_context={"speech_activity": 0.36, "connections": 0.16},
            timestamp=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_gate_result(self) -> IndicatorGateResult:
        """Create sample IndicatorGateResult for testing."""
        return IndicatorGateResult(
            indicator_name="social_withdrawal",
            presence_flag=True,
            days_above_threshold=8,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True,) * 8 + (False,) * 6,
            insufficient_data=False,
        )

    def test_computation_log_structure(self, sample_indicator_score: IndicatorScore):
        """Computation log should have expected structure."""
        from src.core.persistence import _build_computation_log

        log = _build_computation_log(sample_indicator_score, None)

        assert "indicator_name" in log
        assert "daily_likelihood" in log
        assert "contributions" in log
        assert "context" in log
        assert "weights" in log
        assert "biomarkers_used" in log
        assert "biomarkers_missing" in log
        assert "data_reliability_score" in log

    def test_computation_log_contains_contributions(
        self, sample_indicator_score: IndicatorScore
    ):
        """Contributions should have per-biomarker breakdown."""
        from src.core.persistence import _build_computation_log

        log = _build_computation_log(sample_indicator_score, None)

        assert "speech_activity" in log["contributions"]
        assert "connections" in log["contributions"]

        speech_contrib = log["contributions"]["speech_activity"]
        assert speech_contrib["membership"] == 0.75
        assert speech_contrib["direction"] == "contra"
        assert speech_contrib["base_weight"] == 0.30

    def test_computation_log_with_gate_result(
        self,
        sample_indicator_score: IndicatorScore,
        sample_gate_result: IndicatorGateResult,
    ):
        """Computation log should include gate details when provided."""
        from src.core.persistence import _build_computation_log

        log = _build_computation_log(sample_indicator_score, sample_gate_result)

        assert "gate" in log
        assert log["gate"]["presence_flag"] is True
        assert log["gate"]["days_above_threshold"] == 8
        assert log["gate"]["threshold"] == 0.5

    def test_computation_log_without_gate_result(
        self, sample_indicator_score: IndicatorScore
    ):
        """Computation log should not have gate section when gate_result is None."""
        from src.core.persistence import _build_computation_log

        log = _build_computation_log(sample_indicator_score, None)

        assert "gate" not in log

    def test_computation_log_is_json_serializable(
        self,
        sample_indicator_score: IndicatorScore,
        sample_gate_result: IndicatorGateResult,
    ):
        """Computation log should be JSON-serializable."""
        import json

        from src.core.persistence import _build_computation_log

        log = _build_computation_log(sample_indicator_score, sample_gate_result)

        # Should not raise
        json_str = json.dumps(log)
        assert isinstance(json_str, str)


class TestSaveIndicator:
    """Test save_indicator function."""

    @pytest.fixture
    def sample_indicator_score(self) -> IndicatorScore:
        """Create sample IndicatorScore for testing."""
        return IndicatorScore(
            indicator_name="social_withdrawal",
            daily_likelihood=0.65,
            contributions={
                "speech_activity": BiomarkerContribution(
                    name="speech_activity",
                    membership=0.75,
                    direction="contra",
                    base_weight=0.30,
                    context_multiplier=1.5,
                    adjusted_weight=0.36,
                    contribution=0.09,
                ),
            },
            biomarkers_used=("speech_activity",),
            biomarkers_missing=(),
            data_reliability_score=0.85,
            context_applied="solitary_digital",
            context_confidence=0.9,
            weights_before_context={"speech_activity": 0.30},
            weights_after_context={"speech_activity": 0.36},
            timestamp=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_gate_result(self) -> IndicatorGateResult:
        """Create sample IndicatorGateResult for testing."""
        return IndicatorGateResult(
            indicator_name="social_withdrawal",
            presence_flag=True,
            days_above_threshold=8,
            days_evaluated=14,
            window_size=14,
            required_days=10,
            threshold=0.5,
            daily_flags=(True,) * 8 + (False,) * 6,
            insufficient_data=False,
        )

    def test_save_indicator_creates_orm_object(
        self,
        db_session,
        sample_indicator_score: IndicatorScore,
        sample_gate_result: IndicatorGateResult,
    ):
        """save_indicator should create correct ORM object."""
        from src.core.persistence import save_indicator

        user_id = "test-user-001"
        analysis_run_id = uuid.uuid4()

        indicator = save_indicator(
            indicator_score=sample_indicator_score,
            gate_result=sample_gate_result,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        assert indicator.user_id == user_id
        assert indicator.indicator_type == "social_withdrawal"
        assert indicator.value == 0.65
        assert indicator.data_reliability_score == 0.85
        assert indicator.presence_flag is True
        assert indicator.context_used == "solitary_digital"
        assert indicator.analysis_run_id == analysis_run_id

        # Cleanup
        db_session.rollback()

    def test_save_indicator_with_none_gate_result(
        self,
        db_session,
        sample_indicator_score: IndicatorScore,
    ):
        """save_indicator should handle None gate_result."""
        from src.core.persistence import save_indicator

        user_id = "test-user-002"
        analysis_run_id = uuid.uuid4()

        indicator = save_indicator(
            indicator_score=sample_indicator_score,
            gate_result=None,  # No gate result
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        assert indicator.presence_flag is None

        # Cleanup
        db_session.rollback()

    def test_save_indicator_adds_to_session(
        self,
        db_session,
        sample_indicator_score: IndicatorScore,
        sample_gate_result: IndicatorGateResult,
    ):
        """save_indicator should add ORM object to session."""
        from src.core.persistence import save_indicator

        user_id = "test-user-003"
        analysis_run_id = uuid.uuid4()

        indicator = save_indicator(
            indicator_score=sample_indicator_score,
            gate_result=sample_gate_result,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # Object should be pending in session
        assert indicator in db_session.new or indicator in db_session

        # Cleanup
        db_session.rollback()

    def test_modalities_used_populated(
        self,
        db_session,
        sample_indicator_score: IndicatorScore,
    ):
        """modalities_used should contain extracted modalities from biomarkers."""
        from src.core.persistence import save_indicator

        user_id = "test-user-004"
        analysis_run_id = uuid.uuid4()

        indicator = save_indicator(
            indicator_score=sample_indicator_score,
            gate_result=None,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        assert indicator.modalities_used is not None
        assert isinstance(indicator.modalities_used, list)
        assert "speech" in indicator.modalities_used

        # Cleanup
        db_session.rollback()

    def test_computation_log_contains_contributions(
        self,
        db_session,
        sample_indicator_score: IndicatorScore,
        sample_gate_result: IndicatorGateResult,
    ):
        """computation_log should contain contribution breakdown."""
        from src.core.persistence import save_indicator

        user_id = "test-user-005"
        analysis_run_id = uuid.uuid4()

        indicator = save_indicator(
            indicator_score=sample_indicator_score,
            gate_result=sample_gate_result,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        assert indicator.computation_log is not None
        assert "contributions" in indicator.computation_log
        assert "speech_activity" in indicator.computation_log["contributions"]

        # Cleanup
        db_session.rollback()


class TestSaveAllIndicators:
    """Test save_all_indicators batch function."""

    @pytest.fixture
    def sample_indicator_scores(self) -> dict[str, IndicatorScore]:
        """Create sample IndicatorScores for batch testing."""
        timestamp = datetime.now(UTC)
        return {
            "social_withdrawal": IndicatorScore(
                indicator_name="social_withdrawal",
                daily_likelihood=0.65,
                contributions={},
                biomarkers_used=("speech_activity",),
                biomarkers_missing=(),
                data_reliability_score=0.85,
                context_applied="solitary_digital",
                context_confidence=0.9,
                weights_before_context={},
                weights_after_context={},
                timestamp=timestamp,
            ),
            "sleep_disturbance": IndicatorScore(
                indicator_name="sleep_disturbance",
                daily_likelihood=0.45,
                contributions={},
                biomarkers_used=("sleep_duration",),
                biomarkers_missing=(),
                data_reliability_score=0.75,
                context_applied="solitary_digital",
                context_confidence=0.9,
                weights_before_context={},
                weights_after_context={},
                timestamp=timestamp,
            ),
        }

    @pytest.fixture
    def sample_gate_results(self) -> dict[str, IndicatorGateResult]:
        """Create sample IndicatorGateResults for batch testing."""
        return {
            "social_withdrawal": IndicatorGateResult(
                indicator_name="social_withdrawal",
                presence_flag=True,
                days_above_threshold=8,
                days_evaluated=14,
                window_size=14,
                    threshold=0.5,
                daily_flags=(True,) * 8 + (False,) * 6,
                insufficient_data=False,
            ),
            "sleep_disturbance": IndicatorGateResult(
                indicator_name="sleep_disturbance",
                presence_flag=False,
                days_above_threshold=3,
                days_evaluated=14,
                window_size=14,
                    threshold=0.5,
                daily_flags=(True,) * 3 + (False,) * 11,
                insufficient_data=False,
            ),
        }

    def test_batch_save_creates_all_indicators(
        self,
        db_session,
        sample_indicator_scores,
        sample_gate_results,
    ):
        """save_all_indicators should create all indicator records."""
        from src.core.persistence import save_all_indicators

        user_id = "test-user-batch-001"
        analysis_run_id = uuid.uuid4()

        indicators = save_all_indicators(
            indicator_scores=sample_indicator_scores,
            gate_results=sample_gate_results,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        assert len(indicators) == 2

        # Verify both indicators exist
        indicator_names = {ind.indicator_type for ind in indicators}
        assert "social_withdrawal" in indicator_names
        assert "sleep_disturbance" in indicator_names

        # Cleanup
        db_session.rollback()

    def test_batch_save_with_missing_gate_result(
        self,
        db_session,
        sample_indicator_scores,
    ):
        """save_all_indicators should handle missing gate results."""
        from src.core.persistence import save_all_indicators

        user_id = "test-user-batch-002"
        analysis_run_id = uuid.uuid4()

        # Only provide gate result for one indicator
        partial_gate_results = {
            "social_withdrawal": IndicatorGateResult(
                indicator_name="social_withdrawal",
                presence_flag=True,
                days_above_threshold=8,
                days_evaluated=14,
                window_size=14,
                    threshold=0.5,
                daily_flags=(True,) * 8 + (False,) * 6,
                insufficient_data=False,
            ),
        }

        indicators = save_all_indicators(
            indicator_scores=sample_indicator_scores,
            gate_results=partial_gate_results,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # Both should be saved, one with None presence_flag
        assert len(indicators) == 2

        # Find sleep_disturbance indicator
        sleep_indicator = next(
            (ind for ind in indicators if ind.indicator_type == "sleep_disturbance"),
            None,
        )
        assert sleep_indicator is not None
        assert sleep_indicator.presence_flag is None

        # Cleanup
        db_session.rollback()

    def test_batch_save_flushes_session(
        self,
        db_session,
        sample_indicator_scores,
        sample_gate_results,
    ):
        """save_all_indicators should flush session to validate."""
        from src.core.persistence import save_all_indicators

        user_id = "test-user-batch-003"
        analysis_run_id = uuid.uuid4()

        indicators = save_all_indicators(
            indicator_scores=sample_indicator_scores,
            gate_results=sample_gate_results,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # After flush, objects should have IDs assigned
        for indicator in indicators:
            assert indicator.id is not None

        # Cleanup
        db_session.rollback()

    def test_transaction_rollback_on_failure(
        self,
        db_session,
    ):
        """Transaction should rollback on failure."""
        from src.core.persistence import save_all_indicators

        user_id = "test-user-batch-004"
        analysis_run_id = uuid.uuid4()

        # Create indicator score with invalid data (will fail on flush)
        # Using an extremely long string that exceeds String(100) for context_used
        bad_scores = {
            "bad_indicator": IndicatorScore(
                indicator_name="bad_indicator",
                daily_likelihood=0.65,
                contributions={},
                biomarkers_used=(),
                biomarkers_missing=(),
                data_reliability_score=0.85,
                context_applied="x" * 200,  # Exceeds 100 char limit
                context_confidence=0.9,
                weights_before_context={},
                weights_after_context={},
                timestamp=datetime.now(UTC),
            ),
        }

        with pytest.raises((ValueError, Exception)):  # noqa: B017
            save_all_indicators(
                indicator_scores=bad_scores,
                gate_results={},
                user_id=user_id,
                analysis_run_id=analysis_run_id,
                session=db_session,
            )

        # Session should be in failed state - rollback should work
        db_session.rollback()

        # Verify no indicators were persisted
        result = db_session.execute(
            select(Indicator).where(Indicator.user_id == user_id)
        )
        assert result.scalars().all() == []


class TestSaveAnalysisRun:
    """Test save_analysis_run function."""

    def test_save_analysis_run_stores_metadata(self, db_session):
        """save_analysis_run should store run metadata."""
        from src.core.persistence import save_analysis_run

        user_id = "test-user-run-001"
        analysis_run_id = uuid.uuid4()
        start_time = datetime.now(UTC)
        end_time = start_time
        config = get_default_config()

        result = save_analysis_run(
            analysis_run_id=analysis_run_id,
            user_id=user_id,
            config=config,
            start_time=start_time,
            end_time=end_time,
            session=db_session,
        )

        # Result is an AnalysisRun ORM object
        assert result.id == analysis_run_id
        assert result.user_id == user_id
        assert result.config_snapshot is not None
        assert isinstance(result.config_snapshot, dict)

        # Cleanup
        db_session.rollback()


class TestSaveDailyIndicatorScores:
    """Test save_daily_indicator_scores function (Story 4.13)."""

    @pytest.fixture
    def sample_daily_indicator_scores(self) -> dict[str, list[DailyIndicatorScore]]:
        """Create sample DailyIndicatorScores for testing."""
        today = date.today()
        return {
            "social_withdrawal": [
                DailyIndicatorScore(
                    date=today,
                    indicator_name="social_withdrawal",
                    daily_likelihood=0.65,
                    biomarkers_used=("speech_activity",),
                    biomarkers_missing=(),
                    data_reliability_score=0.85,
                ),
            ],
            "sleep_disturbance": [
                DailyIndicatorScore(
                    date=today,
                    indicator_name="sleep_disturbance",
                    daily_likelihood=0.45,
                    biomarkers_used=("sleep_duration",),
                    biomarkers_missing=(),
                    data_reliability_score=0.75,
                ),
            ],
        }

    @pytest.fixture
    def sample_gate_results(self) -> dict[str, IndicatorGateResult]:
        """Create sample IndicatorGateResults for testing."""
        return {
            "social_withdrawal": IndicatorGateResult(
                indicator_name="social_withdrawal",
                presence_flag=True,
                days_above_threshold=8,
                days_evaluated=14,
                window_size=14,
                    threshold=0.5,
                daily_flags=(True,) * 8 + (False,) * 6,
                insufficient_data=False,
            ),
            "sleep_disturbance": IndicatorGateResult(
                indicator_name="sleep_disturbance",
                presence_flag=False,
                days_above_threshold=3,
                days_evaluated=14,
                window_size=14,
                    threshold=0.5,
                daily_flags=(True,) * 3 + (False,) * 11,
                insufficient_data=False,
            ),
        }

    def test_save_daily_scores_creates_all_rows(
        self,
        db_session,
        sample_daily_indicator_scores,
        sample_gate_results,
    ):
        """save_daily_indicator_scores should create one row per day per indicator."""
        from src.core.persistence import save_daily_indicator_scores

        user_id = "test-user-daily-001"
        analysis_run_id = uuid.uuid4()
        end_date = date.today()

        indicators = save_daily_indicator_scores(
            daily_indicator_scores=sample_daily_indicator_scores,
            gate_results=sample_gate_results,
            context_used="solitary_digital",
            end_date=end_date,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # Should have 2 rows (1 day x 2 indicators)
        assert len(indicators) == 2

        # Verify both indicators exist
        indicator_names = {ind.indicator_type for ind in indicators}
        assert "social_withdrawal" in indicator_names
        assert "sleep_disturbance" in indicator_names

        # Cleanup
        db_session.rollback()

    def test_save_daily_scores_end_date_gets_presence_flag(
        self,
        db_session,
        sample_daily_indicator_scores,
        sample_gate_results,
    ):
        """Only end_date row should have presence_flag populated."""
        from src.core.persistence import save_daily_indicator_scores

        user_id = "test-user-daily-002"
        analysis_run_id = uuid.uuid4()
        end_date = date.today()

        indicators = save_daily_indicator_scores(
            daily_indicator_scores=sample_daily_indicator_scores,
            gate_results=sample_gate_results,
            context_used="solitary_digital",
            end_date=end_date,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # Find social_withdrawal indicator (end_date row)
        sw_indicator = next(
            (ind for ind in indicators if ind.indicator_type == "social_withdrawal"),
            None,
        )
        assert sw_indicator is not None
        assert sw_indicator.presence_flag is True  # From gate result

        # Find sleep_disturbance indicator (end_date row)
        sd_indicator = next(
            (ind for ind in indicators if ind.indicator_type == "sleep_disturbance"),
            None,
        )
        assert sd_indicator is not None
        assert sd_indicator.presence_flag is False  # From gate result

        # Cleanup
        db_session.rollback()

    def test_save_daily_scores_all_rows_have_context_used(
        self,
        db_session,
        sample_daily_indicator_scores,
        sample_gate_results,
    ):
        """All daily rows should have context_used populated."""
        from src.core.persistence import save_daily_indicator_scores

        user_id = "test-user-daily-003"
        analysis_run_id = uuid.uuid4()
        end_date = date.today()

        indicators = save_daily_indicator_scores(
            daily_indicator_scores=sample_daily_indicator_scores,
            gate_results=sample_gate_results,
            context_used="solitary_digital",
            end_date=end_date,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # All rows should have context_used
        for ind in indicators:
            assert ind.context_used == "solitary_digital"

        # Cleanup
        db_session.rollback()

    def test_save_daily_scores_computation_log_populated(
        self,
        db_session,
        sample_daily_indicator_scores,
        sample_gate_results,
    ):
        """All daily rows should have computation_log populated."""
        from src.core.persistence import save_daily_indicator_scores

        user_id = "test-user-daily-004"
        analysis_run_id = uuid.uuid4()
        end_date = date.today()

        indicators = save_daily_indicator_scores(
            daily_indicator_scores=sample_daily_indicator_scores,
            gate_results=sample_gate_results,
            context_used="solitary_digital",
            end_date=end_date,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=db_session,
        )

        # All rows should have computation_log
        for ind in indicators:
            assert ind.computation_log is not None
            assert "indicator_name" in ind.computation_log
            assert "daily_likelihood" in ind.computation_log
            assert "biomarkers_used" in ind.computation_log

        # Cleanup
        db_session.rollback()
