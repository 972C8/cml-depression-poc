"""Tests for biomarker processor dataclasses and processor."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.data_reader import BiomarkerRecord
from src.core.processors.biomarker_processor import (
    BaselineStats,
    BiomarkerMembership,
    BiomarkerProcessor,
    DailyBiomarkerMembership,
)


class TestBaselineStats:
    """Test BaselineStats dataclass."""

    def test_baseline_stats_creation(self):
        """Test creating BaselineStats with all fields."""
        stats = BaselineStats(
            mean=0.75,
            std=0.15,
            percentile_25=0.65,
            percentile_75=0.85,
            data_points=30,
            source="user",
        )

        assert stats.mean == 0.75
        assert stats.std == 0.15
        assert stats.percentile_25 == 0.65
        assert stats.percentile_75 == 0.85
        assert stats.data_points == 30
        assert stats.source == "user"

    def test_baseline_stats_optional_percentiles(self):
        """Test BaselineStats with None percentiles."""
        stats = BaselineStats(
            mean=0.5,
            std=0.2,
            data_points=5,
            source="population",
        )

        assert stats.percentile_25 is None
        assert stats.percentile_75 is None

    def test_baseline_stats_is_population_baseline(self):
        """Test is_population_baseline property."""
        user_stats = BaselineStats(mean=0.5, std=0.2, source="user")
        population_stats = BaselineStats(mean=0.5, std=0.2, source="population")

        assert not user_stats.is_population_baseline
        assert population_stats.is_population_baseline

    def test_baseline_stats_frozen(self):
        """Test BaselineStats is immutable."""
        stats = BaselineStats(mean=0.5, std=0.2)

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            stats.mean = 0.6


class TestBiomarkerMembership:
    """Test BiomarkerMembership dataclass."""

    def test_biomarker_membership_creation(self):
        """Test creating BiomarkerMembership with all fields."""
        now = datetime.now(timezone.utc)
        baseline = BaselineStats(mean=0.5, std=0.2, data_points=30)
        membership = BiomarkerMembership(
            name="speech_activity",
            membership=0.75,
            z_score=1.5,
            raw_value=0.8,
            baseline=baseline,
            data_points_used=30,
            data_quality=1.0,
            membership_function_used="triangular",
            timestamp=now,
        )

        assert membership.name == "speech_activity"
        assert membership.membership == 0.75
        assert membership.z_score == 1.5
        assert membership.raw_value == 0.8
        assert membership.baseline == baseline
        assert membership.data_points_used == 30
        assert membership.data_quality == 1.0
        assert membership.membership_function_used == "triangular"
        assert membership.timestamp == now

    def test_biomarker_membership_unavailable(self):
        """Test BiomarkerMembership with None membership (unavailable)."""
        now = datetime.now(timezone.utc)
        membership = BiomarkerMembership(
            name="missing_biomarker",
            membership=None,
            z_score=None,
            raw_value=0.0,
            baseline=None,
            data_points_used=0,
            data_quality=0.0,
            membership_function_used="none",
            timestamp=now,
        )

        assert membership.membership is None
        assert membership.z_score is None
        assert membership.baseline is None
        assert membership.data_quality == 0.0

    def test_biomarker_membership_frozen(self):
        """Test BiomarkerMembership is immutable."""
        now = datetime.now(timezone.utc)
        baseline = BaselineStats(mean=0.5, std=0.2)
        membership = BiomarkerMembership(
            name="test",
            membership=0.5,
            z_score=0.0,
            raw_value=0.5,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.8,
            membership_function_used="sigmoid",
            timestamp=now,
        )

        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
            membership.membership = 0.6


class TestDailyBiomarkerMembership:
    """Test DailyBiomarkerMembership dataclass (Story 4.13)."""

    def test_daily_membership_creation(self):
        """Test creating DailyBiomarkerMembership with date field."""
        now = datetime.now(timezone.utc)
        test_date = date(2025, 1, 15)
        baseline = BaselineStats(mean=6.0, std=0.71, data_points=30)

        membership = DailyBiomarkerMembership(
            date=test_date,
            name="sleep_duration",
            membership=0.62,
            z_score=0.70,
            raw_value=6.5,
            baseline=baseline,
            data_points_used=3,
            data_quality=1.0,
            membership_function_used="sigmoid",
            timestamp=now,
        )

        assert membership.date == test_date
        assert membership.name == "sleep_duration"
        assert membership.membership == 0.62
        assert membership.z_score == 0.70
        assert membership.raw_value == 6.5
        assert membership.baseline == baseline
        assert membership.data_points_used == 3

    def test_daily_membership_frozen(self):
        """Test DailyBiomarkerMembership is immutable."""
        now = datetime.now(timezone.utc)
        membership = DailyBiomarkerMembership(
            date=date(2025, 1, 15),
            name="test",
            membership=0.5,
            z_score=0.0,
            raw_value=0.5,
            baseline=None,
            data_points_used=1,
            data_quality=0.8,
            membership_function_used="linear",
            timestamp=now,
        )

        with pytest.raises(Exception):
            membership.date = date(2025, 1, 16)


class TestAggregateByDay:
    """Test BiomarkerProcessor.aggregate_by_day method (Story 4.13 Task 1.1)."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock AnalysisConfig."""
        config = MagicMock()
        config.biomarker_processing.default_min_data_points = 7
        config.biomarker_processing.min_std_deviation = 0.01
        config.biomarker_processing.z_score_bounds.lower = -3.0
        config.biomarker_processing.z_score_bounds.range = 6.0
        config.biomarker_processing.z_score_warning_threshold = 4.0
        config.biomarker_processing.generic_baseline.mean = 0.5
        config.biomarker_processing.generic_baseline.std = 0.2
        config.biomarker_defaults = {}
        config.biomarker_membership = {}
        config.reliability.population_baseline_quality_penalty = 0.5
        return config

    @pytest.fixture
    def mock_session(self):
        """Create a mock SQLAlchemy session."""
        return MagicMock()

    @pytest.fixture
    def processor(self, mock_config, mock_session):
        """Create BiomarkerProcessor instance."""
        return BiomarkerProcessor(mock_config, mock_session)

    def test_aggregate_by_day_groups_correctly(self, processor):
        """Test that biomarkers are grouped by date and name."""
        # Create mock biomarker records across 3 days
        base_date = datetime(2025, 1, 13, 12, 0, 0, tzinfo=timezone.utc)

        records = [
            # Day 1 - Jan 13
            BiomarkerRecord(
                id="1", user_id="user1", timestamp=base_date,
                biomarker_type="sleep", name="sleep_duration", value=6.5,
                raw_value={"sleep_duration": 6.5}
            ),
            BiomarkerRecord(
                id="2", user_id="user1", timestamp=base_date + timedelta(hours=1),
                biomarker_type="sleep", name="sleep_duration", value=6.8,
                raw_value={"sleep_duration": 6.8}
            ),
            # Day 2 - Jan 14
            BiomarkerRecord(
                id="3", user_id="user1", timestamp=base_date + timedelta(days=1),
                biomarker_type="sleep", name="sleep_duration", value=5.5,
                raw_value={"sleep_duration": 5.5}
            ),
            # Day 3 - Jan 15
            BiomarkerRecord(
                id="4", user_id="user1", timestamp=base_date + timedelta(days=2),
                biomarker_type="sleep", name="sleep_duration", value=7.0,
                raw_value={"sleep_duration": 7.0}
            ),
        ]

        result = processor.aggregate_by_day(
            records,
            start_date=date(2025, 1, 13),
            end_date=date(2025, 1, 15),
        )

        # Check we have 3 days
        assert len(result) == 3

        # Check Day 1 has 2 records
        day1 = result[date(2025, 1, 13)]
        assert "sleep_duration" in day1
        assert len(day1["sleep_duration"]) == 2

        # Check Day 2 and 3 have 1 record each
        assert len(result[date(2025, 1, 14)]["sleep_duration"]) == 1
        assert len(result[date(2025, 1, 15)]["sleep_duration"]) == 1

    def test_aggregate_by_day_filters_date_range(self, processor):
        """Test that records outside date range are excluded."""
        base_date = datetime(2025, 1, 13, 12, 0, 0, tzinfo=timezone.utc)

        records = [
            # Before range - Jan 12
            BiomarkerRecord(
                id="1", user_id="user1", timestamp=base_date - timedelta(days=1),
                biomarker_type="sleep", name="sleep_duration", value=6.0,
                raw_value={"sleep_duration": 6.0}
            ),
            # In range - Jan 13
            BiomarkerRecord(
                id="2", user_id="user1", timestamp=base_date,
                biomarker_type="sleep", name="sleep_duration", value=6.5,
                raw_value={"sleep_duration": 6.5}
            ),
            # After range - Jan 16
            BiomarkerRecord(
                id="3", user_id="user1", timestamp=base_date + timedelta(days=3),
                biomarker_type="sleep", name="sleep_duration", value=7.0,
                raw_value={"sleep_duration": 7.0}
            ),
        ]

        result = processor.aggregate_by_day(
            records,
            start_date=date(2025, 1, 13),
            end_date=date(2025, 1, 14),
        )

        # Only Jan 13 should be included
        assert len(result) == 1
        assert date(2025, 1, 13) in result
        assert date(2025, 1, 12) not in result
        assert date(2025, 1, 16) not in result

    def test_aggregate_by_day_multiple_biomarkers(self, processor):
        """Test aggregation with multiple biomarker types."""
        base_date = datetime(2025, 1, 13, 12, 0, 0, tzinfo=timezone.utc)

        records = [
            BiomarkerRecord(
                id="1", user_id="user1", timestamp=base_date,
                biomarker_type="sleep", name="sleep_duration", value=6.5,
                raw_value={"sleep_duration": 6.5}
            ),
            BiomarkerRecord(
                id="2", user_id="user1", timestamp=base_date,
                biomarker_type="speech", name="speech_activity", value=0.8,
                raw_value={"speech_activity": 0.8}
            ),
        ]

        result = processor.aggregate_by_day(
            records,
            start_date=date(2025, 1, 13),
            end_date=date(2025, 1, 13),
        )

        day1 = result[date(2025, 1, 13)]
        assert "sleep_duration" in day1
        assert "speech_activity" in day1

    def test_aggregate_by_day_empty_input(self, processor):
        """Test aggregation with empty input."""
        result = processor.aggregate_by_day(
            [],
            start_date=date(2025, 1, 13),
            end_date=date(2025, 1, 15),
        )

        assert result == {}
