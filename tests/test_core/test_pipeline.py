"""Tests for pipeline transparency module.

Tests cover:
- PipelineStep creation and serialization (Task 2, AC3)
- PipelineTrace creation, serialization, deserialization (Task 3, AC2, AC15)
- PipelineTracer step tracking and context manager (Task 4, AC4-8)
- Serialization helpers for complex types (Task 5, AC16, AC17)
- Trace persistence and retrieval (Tasks 7-8, AC9-14)
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from src.core.pipeline import (
    PipelineStep,
    PipelineTrace,
    PipelineTracer,
    _serialize_value,
    get_pipeline_trace,
    save_pipeline_trace,
)
from src.shared.models import AnalysisRun


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_create_pipeline_step(self) -> None:
        """Test creating a PipelineStep with all fields."""
        now = datetime.now(UTC)
        step = PipelineStep(
            step_name="Read Data",
            step_number=1,
            inputs={"user_id": "test-user"},
            outputs={"count": 100},
            duration_ms=50,
            metadata={"source": "db"},
            timestamp=now,
        )

        assert step.step_name == "Read Data"
        assert step.step_number == 1
        assert step.inputs == {"user_id": "test-user"}
        assert step.outputs == {"count": 100}
        assert step.duration_ms == 50
        assert step.metadata == {"source": "db"}
        assert step.timestamp == now

    def test_pipeline_step_is_frozen(self) -> None:
        """Test that PipelineStep is immutable."""
        step = PipelineStep(
            step_name="Test",
            step_number=1,
            inputs={},
            outputs={},
            duration_ms=10,
            metadata={},
            timestamp=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            step.step_name = "Modified"  # type: ignore[misc]

    def test_pipeline_step_to_dict(self) -> None:
        """Test PipelineStep serialization to dict."""
        now = datetime.now(UTC)
        step = PipelineStep(
            step_name="Test Step",
            step_number=1,
            inputs={"key": "value"},
            outputs={"result": 42},
            duration_ms=100,
            metadata={"note": "test"},
            timestamp=now,
        )

        d = step.to_dict()

        assert d["step_name"] == "Test Step"
        assert d["step_number"] == 1
        assert d["inputs"] == {"key": "value"}
        assert d["outputs"] == {"result": 42}
        assert d["duration_ms"] == 100
        assert d["metadata"] == {"note": "test"}
        assert d["timestamp"] == now.isoformat()

    def test_pipeline_step_to_dict_handles_complex_inputs(self) -> None:
        """Test serialization with complex nested structures."""
        now = datetime.now(UTC)
        step = PipelineStep(
            step_name="Complex Step",
            step_number=2,
            inputs={
                "datetime_val": datetime(2025, 1, 1, tzinfo=UTC),
                "uuid_val": uuid4(),
                "nested": {"a": 1, "b": [1, 2, 3]},
            },
            outputs={},
            duration_ms=10,
            metadata={},
            timestamp=now,
        )

        d = step.to_dict()

        assert isinstance(d["inputs"]["datetime_val"], str)
        assert isinstance(d["inputs"]["uuid_val"], str)
        assert d["inputs"]["nested"] == {"a": 1, "b": [1, 2, 3]}


class TestPipelineTrace:
    """Tests for PipelineTrace dataclass."""

    @pytest.fixture
    def sample_steps(self) -> tuple[PipelineStep, ...]:
        """Create sample pipeline steps."""
        now = datetime.now(UTC)
        return (
            PipelineStep(
                step_name="Step 1",
                step_number=1,
                inputs={"x": 1},
                outputs={"y": 2},
                duration_ms=10,
                metadata={},
                timestamp=now,
            ),
            PipelineStep(
                step_name="Step 2",
                step_number=2,
                inputs={"a": "b"},
                outputs={"c": "d"},
                duration_ms=20,
                metadata={},
                timestamp=now + timedelta(milliseconds=10),
            ),
        )

    def test_create_pipeline_trace(
        self, sample_steps: tuple[PipelineStep, ...]
    ) -> None:
        """Test creating a PipelineTrace."""
        now = datetime.now(UTC)
        trace = PipelineTrace(
            analysis_run_id="run-123",
            user_id="user-456",
            steps=sample_steps,
            total_duration_ms=30,
            started_at=now,
            completed_at=now + timedelta(milliseconds=30),
        )

        assert trace.analysis_run_id == "run-123"
        assert trace.user_id == "user-456"
        assert len(trace.steps) == 2
        assert trace.total_duration_ms == 30

    def test_pipeline_trace_is_frozen(
        self, sample_steps: tuple[PipelineStep, ...]
    ) -> None:
        """Test that PipelineTrace is immutable."""
        now = datetime.now(UTC)
        trace = PipelineTrace(
            analysis_run_id="run-123",
            user_id="user-456",
            steps=sample_steps,
            total_duration_ms=30,
            started_at=now,
            completed_at=now + timedelta(milliseconds=30),
        )

        with pytest.raises(AttributeError):
            trace.user_id = "modified"  # type: ignore[misc]

    def test_pipeline_trace_to_dict(
        self, sample_steps: tuple[PipelineStep, ...]
    ) -> None:
        """Test PipelineTrace serialization."""
        now = datetime.now(UTC)
        trace = PipelineTrace(
            analysis_run_id="run-123",
            user_id="user-456",
            steps=sample_steps,
            total_duration_ms=30,
            started_at=now,
            completed_at=now + timedelta(milliseconds=30),
        )

        d = trace.to_dict()

        assert d["analysis_run_id"] == "run-123"
        assert d["user_id"] == "user-456"
        assert len(d["steps"]) == 2
        assert d["total_duration_ms"] == 30
        assert d["started_at"] == now.isoformat()

    def test_pipeline_trace_from_dict(
        self, sample_steps: tuple[PipelineStep, ...]
    ) -> None:
        """Test PipelineTrace deserialization."""
        now = datetime.now(UTC)
        trace = PipelineTrace(
            analysis_run_id="run-123",
            user_id="user-456",
            steps=sample_steps,
            total_duration_ms=30,
            started_at=now,
            completed_at=now + timedelta(milliseconds=30),
        )

        d = trace.to_dict()
        restored = PipelineTrace.from_dict(d)

        assert restored.analysis_run_id == trace.analysis_run_id
        assert restored.user_id == trace.user_id
        assert len(restored.steps) == len(trace.steps)
        assert restored.steps[0].step_name == trace.steps[0].step_name
        assert restored.total_duration_ms == trace.total_duration_ms

    def test_pipeline_trace_roundtrip(
        self, sample_steps: tuple[PipelineStep, ...]
    ) -> None:
        """Test that trace can be serialized and deserialized without loss."""
        now = datetime.now(UTC)
        original = PipelineTrace(
            analysis_run_id="roundtrip-test",
            user_id="user-xyz",
            steps=sample_steps,
            total_duration_ms=30,
            started_at=now,
            completed_at=now + timedelta(milliseconds=30),
        )

        # Serialize and deserialize
        d = original.to_dict()
        restored = PipelineTrace.from_dict(d)

        # Verify all fields match
        assert restored.analysis_run_id == original.analysis_run_id
        assert restored.user_id == original.user_id
        assert restored.total_duration_ms == original.total_duration_ms
        assert len(restored.steps) == len(original.steps)

        for i, (orig_step, rest_step) in enumerate(
            zip(original.steps, restored.steps, strict=True)
        ):
            assert rest_step.step_name == orig_step.step_name, f"Step {i} name mismatch"
            assert (
                rest_step.step_number == orig_step.step_number
            ), f"Step {i} number mismatch"
            assert rest_step.inputs == orig_step.inputs, f"Step {i} inputs mismatch"
            assert rest_step.outputs == orig_step.outputs, f"Step {i} outputs mismatch"
            assert (
                rest_step.duration_ms == orig_step.duration_ms
            ), f"Step {i} duration mismatch"

    def test_pipeline_trace_to_summary(
        self, sample_steps: tuple[PipelineStep, ...]
    ) -> None:
        """Test human-readable summary generation."""
        now = datetime.now(UTC)
        trace = PipelineTrace(
            analysis_run_id="run-123",
            user_id="user-456",
            steps=sample_steps,
            total_duration_ms=30,
            started_at=now,
            completed_at=now + timedelta(milliseconds=30),
        )

        summary = trace.to_summary()

        # Check key elements are present
        assert "run-123" in summary
        assert "user-456" in summary
        assert "30 ms" in summary
        assert "Steps: 2" in summary
        assert "[1] Step 1" in summary
        assert "[2] Step 2" in summary

    def test_pipeline_trace_to_summary_includes_scalar_outputs(self) -> None:
        """Test that summary includes scalar output values."""
        now = datetime.now(UTC)
        steps = (
            PipelineStep(
                step_name="Count Step",
                step_number=1,
                inputs={},
                outputs={"count": 42, "ratio": 0.95, "name": "test"},
                duration_ms=10,
                metadata={},
                timestamp=now,
            ),
        )
        trace = PipelineTrace(
            analysis_run_id="run-abc",
            user_id="user-def",
            steps=steps,
            total_duration_ms=10,
            started_at=now,
            completed_at=now + timedelta(milliseconds=10),
        )

        summary = trace.to_summary()

        assert "count: 42" in summary
        assert "ratio: 0.95" in summary
        assert "name: test" in summary


class TestPipelineTracer:
    """Tests for PipelineTracer class."""

    def test_tracer_initialization(self) -> None:
        """Test creating a PipelineTracer."""
        tracer = PipelineTracer("run-id", "user-id")

        assert tracer.analysis_run_id == "run-id"
        assert tracer.user_id == "user-id"

    def test_tracer_single_step(self) -> None:
        """Test tracking a single step."""
        tracer = PipelineTracer("run-1", "user-1")

        tracer.start_step("Step A", inputs={"in": 1})
        tracer.end_step(outputs={"out": 2})

        trace = tracer.get_trace()

        assert len(trace.steps) == 1
        assert trace.steps[0].step_name == "Step A"
        assert trace.steps[0].step_number == 1
        assert trace.steps[0].inputs == {"in": 1}
        assert trace.steps[0].outputs == {"out": 2}
        assert trace.steps[0].duration_ms >= 0

    def test_tracer_multiple_steps(self) -> None:
        """Test tracking multiple steps in sequence."""
        tracer = PipelineTracer("run-2", "user-2")

        tracer.start_step("Read", inputs={"path": "/data"})
        tracer.end_step(outputs={"records": 100})

        tracer.start_step("Process", inputs={"records": 100})
        tracer.end_step(outputs={"processed": 95})

        tracer.start_step("Write", inputs={"processed": 95})
        tracer.end_step(outputs={"written": 95})

        trace = tracer.get_trace()

        assert len(trace.steps) == 3
        assert trace.steps[0].step_name == "Read"
        assert trace.steps[0].step_number == 1
        assert trace.steps[1].step_name == "Process"
        assert trace.steps[1].step_number == 2
        assert trace.steps[2].step_name == "Write"
        assert trace.steps[2].step_number == 3

    def test_tracer_measures_duration(self) -> None:
        """Test that tracer measures step duration."""
        tracer = PipelineTracer("run-3", "user-3")

        tracer.start_step("Slow Step", inputs={})
        time.sleep(0.05)  # 50ms
        tracer.end_step(outputs={})

        trace = tracer.get_trace()

        # Allow some tolerance for timing
        assert trace.steps[0].duration_ms >= 45
        assert trace.steps[0].duration_ms < 200  # Generous upper bound

    def test_tracer_total_duration(self) -> None:
        """Test that total duration sums step durations."""
        tracer = PipelineTracer("run-4", "user-4")

        tracer.start_step("Step 1", inputs={})
        time.sleep(0.02)
        tracer.end_step(outputs={})

        tracer.start_step("Step 2", inputs={})
        time.sleep(0.02)
        tracer.end_step(outputs={})

        trace = tracer.get_trace()

        # Total should be sum of individual durations
        individual_sum = sum(s.duration_ms for s in trace.steps)
        assert trace.total_duration_ms == individual_sum

    def test_tracer_with_metadata(self) -> None:
        """Test that metadata is captured."""
        tracer = PipelineTracer("run-5", "user-5")

        tracer.start_step("Step", inputs={})
        tracer.end_step(outputs={}, metadata={"warning": "low data"})

        trace = tracer.get_trace()

        assert trace.steps[0].metadata == {"warning": "low data"}

    def test_tracer_context_manager(self) -> None:
        """Test using tracer as context manager."""
        with PipelineTracer("run-6", "user-6") as tracer:
            tracer.start_step("Step", inputs={})
            tracer.end_step(outputs={})

        trace = tracer.get_trace()
        assert len(trace.steps) == 1

    def test_tracer_end_step_without_start_raises(self) -> None:
        """Test that end_step without start_step raises error."""
        tracer = PipelineTracer("run-7", "user-7")

        with pytest.raises(RuntimeError, match="end_step called without start_step"):
            tracer.end_step(outputs={})

    def test_tracer_timestamps_are_utc(self) -> None:
        """Test that all timestamps use UTC."""
        tracer = PipelineTracer("run-8", "user-8")

        tracer.start_step("Step", inputs={})
        tracer.end_step(outputs={})

        trace = tracer.get_trace()

        assert trace.started_at.tzinfo is not None
        assert trace.completed_at.tzinfo is not None
        assert trace.steps[0].timestamp.tzinfo is not None


class TestSerializeValue:
    """Tests for _serialize_value helper."""

    def test_serialize_primitives(self) -> None:
        """Test serializing primitive types."""
        assert _serialize_value(None) is None
        assert _serialize_value("string") == "string"
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(True) is True

    def test_serialize_datetime(self) -> None:
        """Test serializing datetime to ISO 8601."""
        dt = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        result = _serialize_value(dt)
        assert result == "2025-06-15T10:30:00+00:00"

    def test_serialize_uuid(self) -> None:
        """Test serializing UUID to string."""
        u = UUID("12345678-1234-5678-1234-567812345678")
        result = _serialize_value(u)
        assert result == "12345678-1234-5678-1234-567812345678"

    def test_serialize_dict(self) -> None:
        """Test serializing nested dict."""
        d = {"a": 1, "b": {"c": datetime(2025, 1, 1, tzinfo=UTC)}}
        result = _serialize_value(d)
        assert result["a"] == 1
        assert result["b"]["c"] == "2025-01-01T00:00:00+00:00"

    def test_serialize_list(self) -> None:
        """Test serializing list."""
        lst = [1, "two", datetime(2025, 1, 1, tzinfo=UTC)]
        result = _serialize_value(lst)
        assert result[0] == 1
        assert result[1] == "two"
        assert result[2] == "2025-01-01T00:00:00+00:00"

    def test_serialize_tuple(self) -> None:
        """Test serializing tuple (converted to list)."""
        t = (1, 2, 3)
        result = _serialize_value(t)
        assert result == [1, 2, 3]

    def test_serialize_large_collection_truncates(self) -> None:
        """Test that large collections are truncated."""
        large_list = list(range(200))
        result = _serialize_value(large_list)

        assert result["_truncated"] is True
        assert result["_count"] == 200
        assert len(result["_sample"]) == 10

    def test_serialize_dataclass(self) -> None:
        """Test serializing a dataclass."""

        @dataclass
        class SampleData:
            name: str
            value: int

        obj = SampleData(name="test", value=42)
        result = _serialize_value(obj)

        assert result["name"] == "test"
        assert result["value"] == 42

    def test_serialize_unknown_type_to_string(self) -> None:
        """Test that unknown types are converted to string."""

        class CustomClass:
            def __str__(self) -> str:
                return "CustomClass()"

        obj = CustomClass()
        result = _serialize_value(obj)
        assert result == "CustomClass()"


class TestPipelinePersistence:
    """Tests for pipeline trace persistence and retrieval."""

    @pytest.fixture
    def sample_trace(self) -> PipelineTrace:
        """Create a sample pipeline trace for testing."""
        now = datetime.now(UTC)
        steps = (
            PipelineStep(
                step_name="Read Data",
                step_number=1,
                inputs={"user_id": "test-user"},
                outputs={"count": 100},
                duration_ms=25,
                metadata={},
                timestamp=now,
            ),
            PipelineStep(
                step_name="Process",
                step_number=2,
                inputs={"count": 100},
                outputs={"processed": 95},
                duration_ms=50,
                metadata={"note": "5 records skipped"},
                timestamp=now + timedelta(milliseconds=25),
            ),
        )
        return PipelineTrace(
            analysis_run_id="test-run-id",
            user_id="test-user",
            steps=steps,
            total_duration_ms=75,
            started_at=now,
            completed_at=now + timedelta(milliseconds=75),
        )

    def test_save_pipeline_trace(
        self, db_session: Session, sample_trace: PipelineTrace
    ) -> None:
        """Test saving a pipeline trace to the database."""
        run_id = uuid4()

        # Create the AnalysisRun first
        analysis_run = AnalysisRun(
            id=run_id,
            user_id="test-user",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=1),
            config_snapshot={"test": True},
        )
        db_session.add(analysis_run)
        db_session.flush()

        # Save the trace
        save_pipeline_trace(sample_trace, run_id, db_session)
        db_session.flush()

        # Verify the trace was saved
        db_session.refresh(analysis_run)
        assert analysis_run.pipeline_trace is not None
        assert analysis_run.pipeline_trace["analysis_run_id"] == "test-run-id"
        assert len(analysis_run.pipeline_trace["steps"]) == 2

    def test_save_pipeline_trace_no_run_logs_warning(
        self, db_session: Session, sample_trace: PipelineTrace, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that saving trace for non-existent run logs warning."""
        nonexistent_run_id = uuid4()

        # Should not raise, just log warning
        save_pipeline_trace(sample_trace, nonexistent_run_id, db_session)

        assert "not found" in caplog.text.lower()

    def test_get_pipeline_trace(
        self, db_session: Session, sample_trace: PipelineTrace
    ) -> None:
        """Test retrieving a pipeline trace from the database."""
        run_id = uuid4()

        # Create the AnalysisRun with trace
        analysis_run = AnalysisRun(
            id=run_id,
            user_id="test-user",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=1),
            config_snapshot={"test": True},
            pipeline_trace=sample_trace.to_dict(),
        )
        db_session.add(analysis_run)
        db_session.flush()

        # Retrieve the trace
        retrieved = get_pipeline_trace(run_id, db_session)

        assert retrieved is not None
        assert retrieved.analysis_run_id == sample_trace.analysis_run_id
        assert retrieved.user_id == sample_trace.user_id
        assert len(retrieved.steps) == len(sample_trace.steps)
        assert retrieved.steps[0].step_name == "Read Data"
        assert retrieved.steps[1].step_name == "Process"

    def test_get_pipeline_trace_not_found(self, db_session: Session) -> None:
        """Test that get_pipeline_trace returns None for non-existent run."""
        nonexistent_run_id = uuid4()

        result = get_pipeline_trace(nonexistent_run_id, db_session)

        assert result is None

    def test_get_pipeline_trace_no_trace(self, db_session: Session) -> None:
        """Test that get_pipeline_trace returns None when run exists but has no trace."""
        run_id = uuid4()

        # Create AnalysisRun without trace
        analysis_run = AnalysisRun(
            id=run_id,
            user_id="test-user",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=1),
            config_snapshot={"test": True},
            pipeline_trace=None,  # No trace
        )
        db_session.add(analysis_run)
        db_session.flush()

        result = get_pipeline_trace(run_id, db_session)

        assert result is None

    def test_pipeline_trace_roundtrip_via_db(
        self, db_session: Session, sample_trace: PipelineTrace
    ) -> None:
        """Test full roundtrip: create trace, save to DB, retrieve, verify."""
        run_id = uuid4()

        # Create AnalysisRun
        analysis_run = AnalysisRun(
            id=run_id,
            user_id="test-user",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=1),
            config_snapshot={"test": True},
        )
        db_session.add(analysis_run)
        db_session.flush()

        # Save trace
        save_pipeline_trace(sample_trace, run_id, db_session)
        db_session.flush()

        # Retrieve trace
        retrieved = get_pipeline_trace(run_id, db_session)

        # Verify full roundtrip
        assert retrieved is not None
        assert retrieved.analysis_run_id == sample_trace.analysis_run_id
        assert retrieved.total_duration_ms == sample_trace.total_duration_ms
        assert len(retrieved.steps) == len(sample_trace.steps)

        for orig, ret in zip(sample_trace.steps, retrieved.steps, strict=True):
            assert ret.step_name == orig.step_name
            assert ret.step_number == orig.step_number
            assert ret.inputs == orig.inputs
            assert ret.outputs == orig.outputs
            assert ret.duration_ms == orig.duration_ms
            assert ret.metadata == orig.metadata
