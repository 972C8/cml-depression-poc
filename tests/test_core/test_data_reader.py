"""Unit tests for data_reader module."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.core.data_reader import (
    BiomarkerRecord,
    ContextRecord,
    DataReader,
    DataReaderResult,
    DataStats,
)
from src.shared.models import Biomarker, Context

# ============================================================================
# Test Dataclasses (Subtasks 13.1, 13.2)
# ============================================================================


class TestBiomarkerRecord:
    """Tests for BiomarkerRecord dataclass."""

    def test_creation(self):
        """Test creating a BiomarkerRecord."""
        now = datetime.now(UTC)
        record = BiomarkerRecord(
            id="123",
            user_id="user-001",
            timestamp=now,
            biomarker_type="speech",
            name="speech_activity",
            value=0.75,
            raw_value={"speech_activity": 0.75},
            metadata=None,
        )
        assert record.id == "123"
        assert record.user_id == "user-001"
        assert record.timestamp == now
        assert record.biomarker_type == "speech"
        assert record.name == "speech_activity"
        assert record.value == 0.75
        assert record.raw_value == {"speech_activity": 0.75}
        assert record.metadata is None

    def test_immutability(self):
        """Test that BiomarkerRecord is frozen (immutable)."""
        record = BiomarkerRecord(
            id="123",
            user_id="user-001",
            timestamp=datetime.now(UTC),
            biomarker_type="speech",
            name="speech_activity",
            value=0.75,
            raw_value={"speech_activity": 0.75},
            metadata=None,
        )
        with pytest.raises(AttributeError):
            record.value = 0.5  # Should fail - frozen


class TestContextRecord:
    """Tests for ContextRecord dataclass."""

    def test_creation(self):
        """Test creating a ContextRecord."""
        now = datetime.now(UTC)
        record = ContextRecord(
            id="456",
            user_id="user-001",
            timestamp=now,
            context_type="environment",
            name="people_in_room",
            value=3.0,
            raw_value={"people_in_room": 3.0},
            metadata=None,
        )
        assert record.id == "456"
        assert record.user_id == "user-001"
        assert record.timestamp == now
        assert record.context_type == "environment"
        assert record.name == "people_in_room"
        assert record.value == 3.0
        assert record.raw_value == {"people_in_room": 3.0}
        assert record.metadata is None

    def test_immutability(self):
        """Test that ContextRecord is frozen (immutable)."""
        record = ContextRecord(
            id="456",
            user_id="user-001",
            timestamp=datetime.now(UTC),
            context_type="environment",
            name="people_in_room",
            value=3.0,
            raw_value={"people_in_room": 3.0},
            metadata=None,
        )
        with pytest.raises(AttributeError):
            record.value = 5.0  # Should fail - frozen


class TestDataStats:
    """Tests for DataStats dataclass."""

    def test_has_data_true_with_biomarkers(self):
        """Test has_data returns True when biomarkers exist."""
        stats = DataStats(
            biomarker_count=5,
            context_count=0,
            time_range_start=None,
            time_range_end=None,
            biomarker_types_found=frozenset(["speech"]),
            biomarker_names_found=frozenset(["speech_activity"]),
            context_names_found=frozenset(),
        )
        assert stats.has_data is True

    def test_has_data_true_with_context(self):
        """Test has_data returns True when context markers exist."""
        stats = DataStats(
            biomarker_count=0,
            context_count=3,
            time_range_start=None,
            time_range_end=None,
            biomarker_types_found=frozenset(),
            biomarker_names_found=frozenset(),
            context_names_found=frozenset(["people_in_room"]),
        )
        assert stats.has_data is True

    def test_has_data_false_when_empty(self):
        """Test has_data returns False when no data exists."""
        stats = DataStats(
            biomarker_count=0,
            context_count=0,
            time_range_start=None,
            time_range_end=None,
            biomarker_types_found=frozenset(),
            biomarker_names_found=frozenset(),
            context_names_found=frozenset(),
        )
        assert stats.has_data is False

    def test_immutability(self):
        """Test that DataStats is frozen (immutable)."""
        stats = DataStats(
            biomarker_count=5,
            context_count=0,
            time_range_start=None,
            time_range_end=None,
            biomarker_types_found=frozenset(["speech"]),
            biomarker_names_found=frozenset(["speech_activity"]),
            context_names_found=frozenset(),
        )
        with pytest.raises(AttributeError):
            stats.biomarker_count = 10  # Should fail - frozen


class TestDataReaderResult:
    """Tests for DataReaderResult dataclass."""

    def test_immutability(self):
        """Test that DataReaderResult is frozen (immutable)."""
        stats = DataStats(
            biomarker_count=0,
            context_count=0,
            time_range_start=None,
            time_range_end=None,
            biomarker_types_found=frozenset(),
            biomarker_names_found=frozenset(),
            context_names_found=frozenset(),
        )
        result = DataReaderResult(
            biomarkers=(),
            context_markers=(),
            biomarkers_by_type={},
            biomarkers_by_name={},
            context_by_name={},
            stats=stats,
        )
        with pytest.raises(AttributeError):
            result.biomarkers = ()  # Should fail - frozen


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestTimezoneNormalization:
    """Tests for _normalize_to_utc (Subtask 13.5)."""

    def test_naive_datetime_assumed_utc(self, db_session):
        """Test naive datetime is assumed to be UTC."""
        reader = DataReader(db_session)
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        normalized = reader._normalize_to_utc(naive_dt)

        assert normalized.tzinfo == UTC
        assert normalized.year == 2024
        assert normalized.month == 1
        assert normalized.day == 1
        assert normalized.hour == 12

    def test_aware_datetime_converted_to_utc(self, db_session):
        """Test aware datetime is converted to UTC."""
        reader = DataReader(db_session)
        # Create datetime in EST (UTC-5)
        from datetime import timedelta
        from datetime import timezone as tz

        est = tz(timedelta(hours=-5))
        aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=est)
        normalized = reader._normalize_to_utc(aware_dt)

        assert normalized.tzinfo == UTC
        # 12:00 EST = 17:00 UTC
        assert normalized.hour == 17

    def test_already_utc_unchanged(self, db_session):
        """Test UTC datetime remains unchanged."""
        reader = DataReader(db_session)
        utc_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        normalized = reader._normalize_to_utc(utc_dt)

        assert normalized == utc_dt


class TestRowExpansion:
    """Tests for row expansion (Subtasks 13.3, 13.4)."""

    def test_expand_biomarker_row_single_value(self, db_session):
        """Test expanding a biomarker row with single value."""
        reader = DataReader(db_session)

        biomarker = Biomarker(
            id=uuid.uuid4(),
            user_id="user-001",
            timestamp=datetime.now(UTC),
            biomarker_type="speech",
            value={"speech_activity": 0.75},
            metadata_=None,
        )

        records = reader._expand_biomarker_row(biomarker)

        assert len(records) == 1
        assert records[0].name == "speech_activity"
        assert records[0].value == 0.75
        assert records[0].biomarker_type == "speech"

    def test_expand_biomarker_row_multiple_values(self, db_session):
        """Test expanding a biomarker row with multiple values."""
        reader = DataReader(db_session)

        biomarker = Biomarker(
            id=uuid.uuid4(),
            user_id="user-001",
            timestamp=datetime.now(UTC),
            biomarker_type="speech",
            value={"speech_activity": 0.75, "voice_energy": 0.60, "speech_rate": 0.55},
            metadata_=None,
        )

        records = reader._expand_biomarker_row(biomarker)

        assert len(records) == 3
        names = {r.name for r in records}
        assert names == {"speech_activity", "voice_energy", "speech_rate"}
        assert all(r.biomarker_type == "speech" for r in records)

    def test_expand_context_row_multiple_values(self, db_session):
        """Test expanding a context row with multiple values."""
        reader = DataReader(db_session)

        context = Context(
            id=uuid.uuid4(),
            user_id="user-001",
            timestamp=datetime.now(UTC),
            context_type="environment",
            value={"people_in_room": 3.0, "ambient_noise": 0.4},
            metadata_=None,
        )

        records = reader._expand_context_row(context)

        assert len(records) == 2
        names = {r.name for r in records}
        assert names == {"people_in_room", "ambient_noise"}
        assert all(r.context_type == "environment" for r in records)


# ============================================================================
# Test DataReader Methods
# ============================================================================


class TestReadBiomarkers:
    """Tests for read_biomarkers method."""

    def test_read_biomarkers_expands_json(self, db_session):
        """Test that read_biomarkers expands JSON values (Subtask 13.6)."""
        # Create test data
        user_id = "test-user-001"
        now = datetime.now(UTC)

        biomarker1 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            biomarker_type="speech",
            value={"speech_activity": 0.75, "voice_energy": 0.60},
            metadata_=None,
        )
        biomarker2 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now + timedelta(minutes=5),
            biomarker_type="network",
            value={"bytes_in": 1024.0, "bytes_out": 512.0},
            metadata_=None,
        )

        db_session.add(biomarker1)
        db_session.add(biomarker2)
        db_session.flush()

        # Read biomarkers
        reader = DataReader(db_session)
        records = reader.read_biomarkers(
            user_id=user_id,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        # Verify expansion: 2 DB rows -> 4 records
        assert len(records) == 4
        names = {r.name for r in records}
        assert names == {"speech_activity", "voice_energy", "bytes_in", "bytes_out"}

    def test_read_biomarkers_with_type_filter(self, db_session):
        """Test biomarker_types filter at DB level."""
        user_id = "test-user-002"
        now = datetime.now(UTC)

        biomarker1 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            biomarker_type="speech",
            value={"speech_activity": 0.75},
            metadata_=None,
        )
        biomarker2 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now + timedelta(minutes=5),
            biomarker_type="network",
            value={"bytes_in": 1024.0},
            metadata_=None,
        )

        db_session.add(biomarker1)
        db_session.add(biomarker2)
        db_session.flush()

        # Read only speech biomarkers
        reader = DataReader(db_session)
        records = reader.read_biomarkers(
            user_id=user_id,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
            biomarker_types=["speech"],
        )

        assert len(records) == 1
        assert records[0].biomarker_type == "speech"
        assert records[0].name == "speech_activity"

    def test_read_biomarkers_with_name_filter(self, db_session):
        """Test names filter post-expansion (Subtask 13.7)."""
        user_id = "test-user-003"
        now = datetime.now(UTC)

        biomarker = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            biomarker_type="speech",
            value={"speech_activity": 0.75, "voice_energy": 0.60, "speech_rate": 0.55},
            metadata_=None,
        )

        db_session.add(biomarker)
        db_session.flush()

        # Read only specific biomarker names
        reader = DataReader(db_session)
        records = reader.read_biomarkers(
            user_id=user_id,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
            names=["speech_activity", "speech_rate"],
        )

        assert len(records) == 2
        names = {r.name for r in records}
        assert names == {"speech_activity", "speech_rate"}

    def test_read_biomarkers_empty_returns_empty_list(self, db_session):
        """Test empty results return empty list (Subtask 13.8)."""
        reader = DataReader(db_session)
        records = reader.read_biomarkers(
            user_id="nonexistent-user",
            start_time=datetime.now(UTC) - timedelta(days=1),
            end_time=datetime.now(UTC),
        )

        assert records == []


class TestReadContextMarkers:
    """Tests for read_context_markers method."""

    def test_read_context_markers_basic(self, db_session):
        """Test basic context marker reading (Subtask 13.9)."""
        user_id = "test-user-004"
        now = datetime.now(UTC)

        context = Context(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            context_type="environment",
            value={"people_in_room": 3.0, "ambient_noise": 0.4},
            metadata_=None,
        )

        db_session.add(context)
        db_session.flush()

        # Read context markers
        reader = DataReader(db_session)
        records = reader.read_context_markers(
            user_id=user_id,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        assert len(records) == 2
        names = {r.name for r in records}
        assert names == {"people_in_room", "ambient_noise"}


class TestReadAll:
    """Tests for read_all method."""

    def test_read_all_returns_grouped_data(self, db_session):
        """Test read_all returns properly grouped data (Subtask 13.10)."""
        user_id = "test-user-005"
        now = datetime.now(UTC)

        biomarker1 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            biomarker_type="speech",
            value={"speech_activity": 0.75},
            metadata_=None,
        )
        biomarker2 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now + timedelta(minutes=5),
            biomarker_type="network",
            value={"bytes_in": 1024.0},
            metadata_=None,
        )
        context = Context(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            context_type="environment",
            value={"people_in_room": 3.0},
            metadata_=None,
        )

        db_session.add(biomarker1)
        db_session.add(biomarker2)
        db_session.add(context)
        db_session.flush()

        # Read all data
        reader = DataReader(db_session)
        result = reader.read_all(
            user_id=user_id,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        # Verify structure
        assert isinstance(result, DataReaderResult)
        assert len(result.biomarkers) == 2
        assert len(result.context_markers) == 1

        # Verify grouping by type
        assert "speech" in result.biomarkers_by_type
        assert "network" in result.biomarkers_by_type
        assert len(result.biomarkers_by_type["speech"]) == 1
        assert len(result.biomarkers_by_type["network"]) == 1

        # Verify grouping by name
        assert "speech_activity" in result.biomarkers_by_name
        assert "bytes_in" in result.biomarkers_by_name

        # Verify context grouping
        assert "people_in_room" in result.context_by_name
        assert len(result.context_by_name["people_in_room"]) == 1


class TestComputeStats:
    """Tests for _compute_stats and DataStats."""

    def test_compute_stats_calculations(self, db_session):
        """Test stats calculations are correct (Subtask 13.11)."""
        user_id = "test-user-006"
        now = datetime.now(UTC)

        biomarker1 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now,
            biomarker_type="speech",
            value={"speech_activity": 0.75, "voice_energy": 0.60},
            metadata_=None,
        )
        biomarker2 = Biomarker(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now + timedelta(minutes=10),
            biomarker_type="network",
            value={"bytes_in": 1024.0},
            metadata_=None,
        )
        context = Context(
            id=uuid.uuid4(),
            user_id=user_id,
            timestamp=now + timedelta(minutes=5),
            context_type="environment",
            value={"people_in_room": 3.0},
            metadata_=None,
        )

        db_session.add(biomarker1)
        db_session.add(biomarker2)
        db_session.add(context)
        db_session.flush()

        # Read all data
        reader = DataReader(db_session)
        result = reader.read_all(
            user_id=user_id,
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )

        stats = result.stats

        # Verify counts
        assert stats.biomarker_count == 3  # 2 from biomarker1, 1 from biomarker2
        assert stats.context_count == 1

        # Verify time range
        assert stats.time_range_start == now
        assert stats.time_range_end == now + timedelta(minutes=10)

        # Verify types and names
        assert stats.biomarker_types_found == frozenset(["speech", "network"])
        assert stats.biomarker_names_found == frozenset(
            ["speech_activity", "voice_energy", "bytes_in"]
        )
        assert stats.context_names_found == frozenset(["people_in_room"])

        # Verify has_data
        assert stats.has_data is True


class TestGroupingFunctions:
    """Tests for grouping functions (Subtask 13.12)."""

    def test_group_biomarkers_by_type(self, db_session):
        """Test grouping biomarkers by type."""
        reader = DataReader(db_session)

        records = [
            BiomarkerRecord(
                id="1",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                biomarker_type="speech",
                name="speech_activity",
                value=0.75,
                raw_value={},
                metadata=None,
            ),
            BiomarkerRecord(
                id="2",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                biomarker_type="speech",
                name="voice_energy",
                value=0.60,
                raw_value={},
                metadata=None,
            ),
            BiomarkerRecord(
                id="3",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                biomarker_type="network",
                name="bytes_in",
                value=1024.0,
                raw_value={},
                metadata=None,
            ),
        ]

        grouped = reader._group_biomarkers_by_type(records)

        assert "speech" in grouped
        assert "network" in grouped
        assert len(grouped["speech"]) == 2
        assert len(grouped["network"]) == 1

    def test_group_biomarkers_by_name(self, db_session):
        """Test grouping biomarkers by name."""
        reader = DataReader(db_session)

        records = [
            BiomarkerRecord(
                id="1",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                biomarker_type="speech",
                name="speech_activity",
                value=0.75,
                raw_value={},
                metadata=None,
            ),
            BiomarkerRecord(
                id="2",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                biomarker_type="speech",
                name="speech_activity",
                value=0.80,
                raw_value={},
                metadata=None,
            ),
        ]

        grouped = reader._group_biomarkers_by_name(records)

        assert "speech_activity" in grouped
        assert len(grouped["speech_activity"]) == 2

    def test_group_context_by_name(self, db_session):
        """Test grouping context by name."""
        reader = DataReader(db_session)

        records = [
            ContextRecord(
                id="1",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                context_type="environment",
                name="people_in_room",
                value=3.0,
                raw_value={},
                metadata=None,
            ),
            ContextRecord(
                id="2",
                user_id="user-001",
                timestamp=datetime.now(UTC),
                context_type="environment",
                name="people_in_room",
                value=5.0,
                raw_value={},
                metadata=None,
            ),
        ]

        grouped = reader._group_context_by_name(records)

        assert "people_in_room" in grouped
        assert len(grouped["people_in_room"]) == 2
