"""Tests for window aggregation module.

Story 6.2: Window Aggregation Module (AC7)
Tests floor_to_window, aggregate_into_windows, aggregation methods, and edge cases.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.core.data_reader import BiomarkerRecord
from src.core.models.window_models import WindowAggregate
from src.core.processors.window_aggregator import (
    aggregate_into_windows,
    floor_to_window,
)

# ============================================================================
# WindowAggregate Dataclass Tests (AC1)
# ============================================================================


class TestWindowAggregate:
    """Tests for WindowAggregate dataclass."""

    def test_create_window_aggregate(self):
        """Test creating a WindowAggregate instance."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        timestamps = (
            datetime(2025, 1, 15, 9, 5, 0, tzinfo=UTC),
            datetime(2025, 1, 15, 9, 10, 0, tzinfo=UTC),
        )

        agg = WindowAggregate(
            biomarker_name="speech_activity",
            window_start=window_start,
            window_end=window_end,
            aggregated_value=0.6,
            readings_count=2,
            readings_timestamps=timestamps,
            aggregation_method="mean",
        )

        assert agg.biomarker_name == "speech_activity"
        assert agg.window_start == window_start
        assert agg.window_end == window_end
        assert agg.aggregated_value == 0.6
        assert agg.readings_count == 2
        assert agg.readings_timestamps == timestamps
        assert agg.aggregation_method == "mean"

    def test_window_aggregate_is_frozen(self):
        """Test that WindowAggregate is immutable."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        agg = WindowAggregate(
            biomarker_name="speech_activity",
            window_start=window_start,
            window_end=window_end,
            aggregated_value=0.6,
            readings_count=1,
            readings_timestamps=(window_start,),
        )

        with pytest.raises(AttributeError):
            agg.aggregated_value = 0.9

    def test_window_aggregate_default_method(self):
        """Test that aggregation_method defaults to 'mean'."""
        window_start = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        window_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

        agg = WindowAggregate(
            biomarker_name="speech_activity",
            window_start=window_start,
            window_end=window_end,
            aggregated_value=0.6,
            readings_count=1,
            readings_timestamps=(window_start,),
        )

        assert agg.aggregation_method == "mean"


# ============================================================================
# floor_to_window Tests (AC2)
# ============================================================================


class TestFloorToWindow:
    """Tests for floor_to_window utility function."""

    @pytest.mark.parametrize(
        "timestamp,window_minutes,expected",
        [
            # 15-minute windows
            (
                datetime(2025, 1, 15, 9, 23, 41, tzinfo=UTC),
                15,
                datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC),
            ),
            (
                datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC),
                15,
                datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC),
            ),
            (
                datetime(2025, 1, 15, 9, 14, 59, tzinfo=UTC),
                15,
                datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC),
            ),
            (
                datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC),
                15,
                datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC),
            ),
            # 5-minute windows
            (
                datetime(2025, 1, 15, 9, 7, 30, tzinfo=UTC),
                5,
                datetime(2025, 1, 15, 9, 5, 0, tzinfo=UTC),
            ),
            # 10-minute windows
            (
                datetime(2025, 1, 15, 9, 25, 30, tzinfo=UTC),
                10,
                datetime(2025, 1, 15, 9, 20, 0, tzinfo=UTC),
            ),
            # 30-minute windows
            (
                datetime(2025, 1, 15, 9, 45, 0, tzinfo=UTC),
                30,
                datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC),
            ),
            # 60-minute windows
            (
                datetime(2025, 1, 15, 9, 59, 59, tzinfo=UTC),
                60,
                datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC),
            ),
        ],
    )
    def test_floor_to_window_various_sizes(self, timestamp, window_minutes, expected):
        """Test floor_to_window with various window sizes."""
        result = floor_to_window(timestamp, window_minutes)
        assert result == expected

    def test_floor_to_window_preserves_timezone(self):
        """Test that floor_to_window preserves timezone information."""
        zurich_tz = timezone(timedelta(hours=1))
        timestamp = datetime(2025, 1, 15, 10, 23, 41, tzinfo=zurich_tz)

        result = floor_to_window(timestamp, 15)

        assert result.tzinfo == zurich_tz
        assert result == datetime(2025, 1, 15, 10, 15, 0, tzinfo=zurich_tz)

    def test_floor_to_window_clears_seconds_microseconds(self):
        """Test that floor_to_window clears seconds and microseconds."""
        timestamp = datetime(2025, 1, 15, 9, 23, 41, 123456, tzinfo=UTC)

        result = floor_to_window(timestamp, 15)

        assert result.second == 0
        assert result.microsecond == 0

    def test_floor_to_window_invalid_window_size(self):
        """Test that invalid window sizes raise ValueError."""
        timestamp = datetime(2025, 1, 15, 9, 23, 41, tzinfo=UTC)

        with pytest.raises(ValueError):
            floor_to_window(timestamp, 7)  # Not in allowed sizes

        with pytest.raises(ValueError):
            floor_to_window(timestamp, 0)

        with pytest.raises(ValueError):
            floor_to_window(timestamp, -15)


# ============================================================================
# aggregate_into_windows Tests (AC3, AC4, AC5)
# ============================================================================


@pytest.fixture
def sample_biomarker_records():
    """Create sample BiomarkerRecord instances for testing."""
    base_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
    return [
        BiomarkerRecord(
            id="1",
            user_id="user1",
            timestamp=base_time,
            biomarker_type="speech",
            name="speech_activity",
            value=0.5,
            raw_value={"speech_activity": 0.5},
        ),
        BiomarkerRecord(
            id="2",
            user_id="user1",
            timestamp=base_time + timedelta(minutes=5),
            biomarker_type="speech",
            name="speech_activity",
            value=0.7,
            raw_value={"speech_activity": 0.7},
        ),
        BiomarkerRecord(
            id="3",
            user_id="user1",
            timestamp=base_time + timedelta(minutes=10),
            biomarker_type="speech",
            name="speech_activity",
            value=0.6,
            raw_value={"speech_activity": 0.6},
        ),
        # Second biomarker type
        BiomarkerRecord(
            id="4",
            user_id="user1",
            timestamp=base_time + timedelta(minutes=2),
            biomarker_type="network",
            name="bytes_in",
            value=1000.0,
            raw_value={"bytes_in": 1000.0},
        ),
        BiomarkerRecord(
            id="5",
            user_id="user1",
            timestamp=base_time + timedelta(minutes=7),
            biomarker_type="network",
            name="bytes_in",
            value=2000.0,
            raw_value={"bytes_in": 2000.0},
        ),
        # Reading in next window
        BiomarkerRecord(
            id="6",
            user_id="user1",
            timestamp=base_time + timedelta(minutes=20),
            biomarker_type="speech",
            name="speech_activity",
            value=0.8,
            raw_value={"speech_activity": 0.8},
        ),
    ]


class TestAggregateIntoWindows:
    """Tests for aggregate_into_windows function."""

    def test_aggregate_groups_by_biomarker_name(self, sample_biomarker_records):
        """Test that records are grouped by biomarker name."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        assert "speech_activity" in result
        assert "bytes_in" in result
        assert len(result) == 2

    def test_aggregate_groups_by_window(self, sample_biomarker_records):
        """Test that records are grouped into correct windows."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        # speech_activity should have 2 windows (09:00 and 09:15)
        speech_windows = result["speech_activity"]
        assert len(speech_windows) == 2

        # First window has 3 readings (at 09:00, 09:05, 09:10)
        window_09_00 = speech_windows[0]
        assert window_09_00.window_start == datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        assert window_09_00.readings_count == 3

        # Second window has 1 reading (at 09:20)
        window_09_15 = speech_windows[1]
        assert window_09_15.window_start == datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        assert window_09_15.readings_count == 1

    def test_aggregate_mean_method(self, sample_biomarker_records):
        """Test mean aggregation method."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        # speech_activity first window: mean of 0.5, 0.7, 0.6 = 0.6
        window = result["speech_activity"][0]
        assert window.aggregated_value == pytest.approx(0.6, rel=1e-6)
        assert window.aggregation_method == "mean"

    def test_aggregate_median_method(self, sample_biomarker_records):
        """Test median aggregation method."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="median",
        )

        # speech_activity first window: median of 0.5, 0.6, 0.7 = 0.6
        window = result["speech_activity"][0]
        assert window.aggregated_value == pytest.approx(0.6, rel=1e-6)
        assert window.aggregation_method == "median"

    def test_aggregate_max_method(self, sample_biomarker_records):
        """Test max aggregation method."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="max",
        )

        # speech_activity first window: max of 0.5, 0.7, 0.6 = 0.7
        window = result["speech_activity"][0]
        assert window.aggregated_value == pytest.approx(0.7, rel=1e-6)
        assert window.aggregation_method == "max"

    def test_aggregate_min_method(self, sample_biomarker_records):
        """Test min aggregation method."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="min",
        )

        # speech_activity first window: min of 0.5, 0.7, 0.6 = 0.5
        window = result["speech_activity"][0]
        assert window.aggregated_value == pytest.approx(0.5, rel=1e-6)
        assert window.aggregation_method == "min"

    def test_aggregate_preserves_timestamps(self, sample_biomarker_records):
        """Test that original timestamps are preserved."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        window = result["speech_activity"][0]
        assert len(window.readings_timestamps) == 3
        assert datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC) in window.readings_timestamps
        assert datetime(2025, 1, 15, 9, 5, 0, tzinfo=UTC) in window.readings_timestamps
        assert datetime(2025, 1, 15, 9, 10, 0, tzinfo=UTC) in window.readings_timestamps

    def test_aggregate_window_end_calculated(self, sample_biomarker_records):
        """Test that window_end is calculated correctly."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        window = result["speech_activity"][0]
        expected_end = datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)
        assert window.window_end == expected_end


# ============================================================================
# Edge Case Tests (AC5)
# ============================================================================


class TestAggregateEdgeCases:
    """Tests for edge cases in aggregation."""

    def test_empty_records_returns_empty_dict(self):
        """Test that empty input returns empty dict."""
        result = aggregate_into_windows(
            [],
            window_size_minutes=15,
            aggregation_method="mean",
        )

        assert result == {}

    def test_single_reading_window(self):
        """Test window with single reading."""
        records = [
            BiomarkerRecord(
                id="1",
                user_id="user1",
                timestamp=datetime(2025, 1, 15, 9, 5, 0, tzinfo=UTC),
                biomarker_type="speech",
                name="speech_activity",
                value=0.5,
                raw_value={"speech_activity": 0.5},
            ),
        ]

        result = aggregate_into_windows(
            records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        window = result["speech_activity"][0]
        assert window.aggregated_value == 0.5
        assert window.readings_count == 1

    def test_boundary_reading_inclusive_start(self):
        """Test that reading at exactly window_start belongs to that window."""
        # Reading at exactly 09:15:00.000 should belong to 09:15 window
        records = [
            BiomarkerRecord(
                id="1",
                user_id="user1",
                timestamp=datetime(2025, 1, 15, 9, 15, 0, 0, tzinfo=UTC),
                biomarker_type="speech",
                name="speech_activity",
                value=0.5,
                raw_value={"speech_activity": 0.5},
            ),
        ]

        result = aggregate_into_windows(
            records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        window = result["speech_activity"][0]
        assert window.window_start == datetime(2025, 1, 15, 9, 15, 0, tzinfo=UTC)

    def test_windows_sorted_by_time(self, sample_biomarker_records):
        """Test that windows are sorted by window_start."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
        )

        speech_windows = result["speech_activity"]
        for i in range(1, len(speech_windows)):
            assert speech_windows[i - 1].window_start < speech_windows[i].window_start

    def test_invalid_aggregation_method(self, sample_biomarker_records):
        """Test that invalid aggregation method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_into_windows(
                sample_biomarker_records,
                window_size_minutes=15,
                aggregation_method="invalid",
            )

    def test_min_readings_filters_windows(self):
        """Test that windows with fewer than min_readings are skipped (AC6)."""
        base_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        records = [
            # Window 09:00 - only 1 reading
            BiomarkerRecord(
                id="1",
                user_id="user1",
                timestamp=base_time,
                biomarker_type="speech",
                name="speech_activity",
                value=0.5,
                raw_value={"speech_activity": 0.5},
            ),
            # Window 09:15 - 3 readings (should pass min_readings=2)
            BiomarkerRecord(
                id="2",
                user_id="user1",
                timestamp=base_time + timedelta(minutes=15),
                biomarker_type="speech",
                name="speech_activity",
                value=0.6,
                raw_value={"speech_activity": 0.6},
            ),
            BiomarkerRecord(
                id="3",
                user_id="user1",
                timestamp=base_time + timedelta(minutes=20),
                biomarker_type="speech",
                name="speech_activity",
                value=0.7,
                raw_value={"speech_activity": 0.7},
            ),
            BiomarkerRecord(
                id="4",
                user_id="user1",
                timestamp=base_time + timedelta(minutes=25),
                biomarker_type="speech",
                name="speech_activity",
                value=0.8,
                raw_value={"speech_activity": 0.8},
            ),
        ]

        # With min_readings=2, the 09:00 window (1 reading) should be skipped
        result = aggregate_into_windows(
            records,
            window_size_minutes=15,
            aggregation_method="mean",
            min_readings=2,
        )

        speech_windows = result["speech_activity"]
        assert len(speech_windows) == 1  # Only 09:15 window passes
        assert speech_windows[0].window_start == datetime(
            2025, 1, 15, 9, 15, 0, tzinfo=UTC
        )
        assert speech_windows[0].readings_count == 3

    def test_min_readings_default_one(self, sample_biomarker_records):
        """Test that min_readings defaults to 1 (single reading windows included)."""
        result = aggregate_into_windows(
            sample_biomarker_records,
            window_size_minutes=15,
            aggregation_method="mean",
            # min_readings not specified, should default to 1
        )

        # speech_activity has a single-reading window at 09:15 (value 0.8)
        speech_windows = result["speech_activity"]
        # Should have 2 windows: 09:00 (3 readings) and 09:15 (1 reading)
        assert len(speech_windows) == 2

    def test_min_readings_invalid_zero_raises(self, sample_biomarker_records):
        """Test that min_readings < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_readings must be >= 1"):
            aggregate_into_windows(
                sample_biomarker_records,
                window_size_minutes=15,
                aggregation_method="mean",
                min_readings=0,
            )

    def test_min_readings_invalid_negative_raises(self, sample_biomarker_records):
        """Test that negative min_readings raises ValueError."""
        with pytest.raises(ValueError, match="min_readings must be >= 1"):
            aggregate_into_windows(
                sample_biomarker_records,
                window_size_minutes=15,
                aggregation_method="mean",
                min_readings=-1,
            )

    def test_min_readings_filters_all_windows_returns_empty(self):
        """Test that filtering all windows returns empty result for biomarker."""
        base_time = datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)
        records = [
            BiomarkerRecord(
                id="1",
                user_id="user1",
                timestamp=base_time,
                biomarker_type="speech",
                name="speech_activity",
                value=0.5,
                raw_value={"speech_activity": 0.5},
            ),
        ]

        # With min_readings=5, no windows pass
        result = aggregate_into_windows(
            records,
            window_size_minutes=15,
            aggregation_method="mean",
            min_readings=5,
        )

        # Biomarker still in result but with empty list
        assert result["speech_activity"] == []


# ============================================================================
# Performance Test (AC7 - O(N) characteristic)
# ============================================================================


class TestAggregatePerformance:
    """Tests for O(N) performance characteristic."""

    def test_linear_time_complexity(self):
        """Test that aggregation scales linearly with input size."""
        import time

        # Generate N records
        def generate_records(n: int) -> list[BiomarkerRecord]:
            base_time = datetime(2025, 1, 15, 0, 0, 0, tzinfo=UTC)
            return [
                BiomarkerRecord(
                    id=str(i),
                    user_id="user1",
                    timestamp=base_time + timedelta(minutes=i),
                    biomarker_type="speech",
                    name="speech_activity",
                    value=float(i % 100) / 100,
                    raw_value={"speech_activity": float(i % 100) / 100},
                )
                for i in range(n)
            ]

        # Time small input
        small_records = generate_records(1000)
        start = time.perf_counter()
        aggregate_into_windows(small_records, 15, "mean")
        small_time = time.perf_counter() - start

        # Time larger input (10x)
        large_records = generate_records(10000)
        start = time.perf_counter()
        aggregate_into_windows(large_records, 15, "mean")
        large_time = time.perf_counter() - start

        # Large should take roughly 10x time (allowing 20x for overhead)
        # This is a rough check for O(N) vs O(N^2)
        assert large_time < small_time * 20, (
            f"Performance not linear: small={small_time:.4f}s, "
            f"large={large_time:.4f}s (ratio: {large_time / small_time:.1f}x)"
        )
