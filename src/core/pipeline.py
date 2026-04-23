"""Pipeline transparency module.

Provides introspection utilities to capture and store the full computation
pipeline for white-box explainability (FR-022, NFR9).

This module enables analysts to inspect exactly how indicators were computed,
including intermediate values at each pipeline stage. The trace is stored
with analysis results and can be retrieved for debugging or audit purposes.

Usage:
    tracer = PipelineTracer(run_id, user_id)

    tracer.start_step("Read Data", inputs={"user_id": user_id})
    # ... do work ...
    tracer.end_step(outputs={"count": 100})

    trace = tracer.get_trace()
    save_pipeline_trace(trace, run_id, session)
"""

import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineStep",
    "PipelineTrace",
    "PipelineTracer",
    "get_pipeline_trace",
    "save_pipeline_trace",
]


def _serialize_value(value: Any) -> Any:
    """Serialize complex values to JSON-compatible types.

    Handles datetime, UUID, dataclass, dict, list, and tuple types.
    Large collections (>100 items) are truncated to preserve storage.
    Unknown types are converted to their string representation.

    Args:
        value: Any value to serialize

    Returns:
        JSON-serializable representation of the value
    """
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        items = list(value)
        if len(items) > 100:
            return {
                "_truncated": True,
                "_count": len(items),
                "_sample": [_serialize_value(i) for i in items[:10]],
            }
        return [_serialize_value(i) for i in items]
    if hasattr(value, "__dataclass_fields__"):
        # Dataclass - use asdict
        return _serialize_value(asdict(value))
    # Fallback: convert to string
    return str(value)


@dataclass(frozen=True)
class PipelineStep:
    """Captures a single step in the analysis pipeline.

    Attributes:
        step_name: Human-readable step name (e.g., "Read Data")
        step_number: Sequential step number (1-7)
        inputs: Input values for this step
        outputs: Output values produced by this step
        duration_ms: Execution time in milliseconds
        metadata: Additional context (config used, warnings, etc.)
        timestamp: When this step completed
    """

    step_name: str
    step_number: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    duration_ms: int
    metadata: dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Returns:
            Dictionary with all fields serialized for JSON storage.
        """
        return {
            "step_name": self.step_name,
            "step_number": self.step_number,
            "inputs": _serialize_value(self.inputs),
            "outputs": _serialize_value(self.outputs),
            "duration_ms": self.duration_ms,
            "metadata": _serialize_value(self.metadata),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PipelineTrace:
    """Complete trace of an analysis pipeline run.

    Attributes:
        analysis_run_id: UUID of the analysis run (as string)
        user_id: User being analyzed
        steps: All pipeline steps in order
        total_duration_ms: Total pipeline execution time
        started_at: When pipeline started
        completed_at: When pipeline completed
    """

    analysis_run_id: str
    user_id: str
    steps: tuple[PipelineStep, ...]
    total_duration_ms: int
    started_at: datetime
    completed_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Returns:
            Dictionary ready for JSON storage.
        """
        return {
            "analysis_run_id": self.analysis_run_id,
            "user_id": self.user_id,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineTrace":
        """Deserialize from JSON dict.

        Args:
            data: Dictionary from JSON storage

        Returns:
            Reconstructed PipelineTrace instance
        """
        steps = tuple(
            PipelineStep(
                step_name=s["step_name"],
                step_number=s["step_number"],
                inputs=s["inputs"],
                outputs=s["outputs"],
                duration_ms=s["duration_ms"],
                metadata=s["metadata"],
                timestamp=datetime.fromisoformat(s["timestamp"]),
            )
            for s in data["steps"]
        )
        return cls(
            analysis_run_id=data["analysis_run_id"],
            user_id=data["user_id"],
            steps=steps,
            total_duration_ms=data["total_duration_ms"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
        )

    def to_summary(self) -> str:
        """Generate human-readable pipeline summary.

        Returns:
            Multi-line string with pipeline overview and key metrics per step.
        """
        lines = [
            f"Pipeline Trace: {self.analysis_run_id}",
            f"User: {self.user_id}",
            f"Total Duration: {self.total_duration_ms} ms",
            f"Steps: {len(self.steps)}",
            "",
        ]
        for step in self.steps:
            lines.append(
                f"  [{step.step_number}] {step.step_name}: " f"{step.duration_ms} ms"
            )
            # Add key outputs (only scalar values)
            for key, value in step.outputs.items():
                if isinstance(value, int | float | str | bool):
                    lines.append(f"      {key}: {value}")
        return "\n".join(lines)


class PipelineTracer:
    """Context manager for collecting pipeline trace.

    Tracks timing and captures inputs/outputs for each pipeline step.
    Thread-safe for use in concurrent environments.

    Usage:
        tracer = PipelineTracer(run_id, user_id)

        tracer.start_step("Read Data", inputs={"user_id": user_id})
        # ... do work ...
        tracer.end_step(outputs={"count": 100})

        trace = tracer.get_trace()

    As context manager:
        with PipelineTracer(run_id, user_id) as tracer:
            tracer.start_step("Step 1", inputs={})
            tracer.end_step(outputs={})

        trace = tracer.get_trace()
    """

    def __init__(self, analysis_run_id: str, user_id: str) -> None:
        """Initialize tracer for an analysis run.

        Args:
            analysis_run_id: UUID string for the analysis run
            user_id: User being analyzed
        """
        self.analysis_run_id = analysis_run_id
        self.user_id = user_id
        self._steps: list[PipelineStep] = []
        self._started_at = datetime.now(UTC)
        self._current_step_name: str | None = None
        self._current_step_number: int = 0
        self._current_step_start: float | None = None
        self._current_step_inputs: dict[str, Any] = {}

    def start_step(
        self,
        step_name: str,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Begin timing a pipeline step.

        Args:
            step_name: Human-readable name for this step
            inputs: Input values to record for this step
        """
        self._current_step_number += 1
        self._current_step_name = step_name
        self._current_step_start = time.perf_counter()
        self._current_step_inputs = inputs or {}
        logger.debug("Started step %d: %s", self._current_step_number, step_name)

    def end_step(
        self,
        outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Complete current step and record results.

        Args:
            outputs: Output values produced by this step
            metadata: Additional context (warnings, config, etc.)

        Raises:
            RuntimeError: If called without a corresponding start_step
        """
        if self._current_step_start is None:
            raise RuntimeError("end_step called without start_step")

        duration_ms = int((time.perf_counter() - self._current_step_start) * 1000)

        step = PipelineStep(
            step_name=self._current_step_name or "Unknown",
            step_number=self._current_step_number,
            inputs=self._current_step_inputs,
            outputs=outputs or {},
            duration_ms=duration_ms,
            metadata=metadata or {},
            timestamp=datetime.now(UTC),
        )
        self._steps.append(step)

        logger.debug(
            "Completed step %d: %s (%d ms)",
            step.step_number,
            step.step_name,
            duration_ms,
        )

        # Reset current step state
        self._current_step_name = None
        self._current_step_start = None
        self._current_step_inputs = {}

    def get_trace(self) -> PipelineTrace:
        """Build and return the complete pipeline trace.

        Returns:
            PipelineTrace with all recorded steps and timing.
        """
        completed_at = datetime.now(UTC)
        total_duration_ms = sum(s.duration_ms for s in self._steps)

        return PipelineTrace(
            analysis_run_id=self.analysis_run_id,
            user_id=self.user_id,
            steps=tuple(self._steps),
            total_duration_ms=total_duration_ms,
            started_at=self._started_at,
            completed_at=completed_at,
        )

    def __enter__(self) -> "PipelineTracer":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager. Trace available via get_trace()."""
        pass


def save_pipeline_trace(
    trace: PipelineTrace,
    analysis_run_id: UUID,
    session: Session,
) -> None:
    """Store pipeline trace with analysis results.

    Serializes the trace to JSON and updates the AnalysisRun record.
    The caller is responsible for committing the transaction.

    Args:
        trace: Complete pipeline trace to save
        analysis_run_id: UUID of the analysis run
        session: SQLAlchemy session (caller manages transaction)
    """
    # Import here to avoid circular dependency
    from src.shared.models import AnalysisRun

    trace_dict = trace.to_dict()

    stmt = select(AnalysisRun).where(AnalysisRun.id == analysis_run_id)
    result = session.execute(stmt)
    run = result.scalar_one_or_none()

    if run:
        run.pipeline_trace = trace_dict
        session.flush()
        logger.debug(
            "Saved pipeline trace for analysis run '%s' (%d steps)",
            analysis_run_id,
            len(trace.steps),
        )
    else:
        logger.warning(
            "Cannot save pipeline trace: AnalysisRun '%s' not found",
            analysis_run_id,
        )


def get_pipeline_trace(
    analysis_run_id: UUID,
    session: Session,
) -> PipelineTrace | None:
    """Retrieve stored pipeline trace.

    Args:
        analysis_run_id: UUID of the analysis run
        session: SQLAlchemy session

    Returns:
        PipelineTrace if found and trace exists, None otherwise
    """
    # Import here to avoid circular dependency
    from src.shared.models import AnalysisRun

    stmt = select(AnalysisRun).where(AnalysisRun.id == analysis_run_id)
    result = session.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        logger.debug("AnalysisRun '%s' not found", analysis_run_id)
        return None

    if run.pipeline_trace is None:
        logger.debug("AnalysisRun '%s' has no pipeline trace", analysis_run_id)
        return None

    trace = PipelineTrace.from_dict(run.pipeline_trace)
    logger.debug(
        "Retrieved pipeline trace for analysis run '%s' (%d steps)",
        analysis_run_id,
        len(trace.steps),
    )
    return trace
