import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.shared.database import Base


class BiomarkerType(str, Enum):
    """Types of biomarkers supported."""

    SPEECH = "speech"
    NETWORK = "network"


class Biomarker(Base):
    """Biomarker data from external sources."""

    __tablename__ = "biomarkers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    biomarker_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("ix_biomarkers_user_timestamp", "user_id", "timestamp"),)


class Context(Base):
    """Context data from external sources."""

    __tablename__ = "context"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    context_type: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("ix_context_user_timestamp", "user_id", "timestamp"),)


class Indicator(Base):
    """Computed indicators from analysis engine."""

    __tablename__ = "indicators"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    indicator_type: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    data_reliability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    # Note: config_snapshot removed - use AnalysisRun.config_snapshot instead
    # Story 4.10: Additional columns for full persistence
    presence_flag: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        doc="DSM-gate presence flag (True if indicator meets N-of-M criterion)",
    )
    context_used: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        doc="Active context during analysis (e.g., 'solitary_digital', 'neutral')",
    )
    modalities_used: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="List of modalities that contributed to this indicator",
    )
    computation_log: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Detailed computation trace for transparency/debugging",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("ix_indicators_user_timestamp", "user_id", "timestamp"),)


class UserBaseline(Base):
    """User-specific baseline statistics for biomarkers."""

    __tablename__ = "user_baselines"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    biomarker_name: Mapped[str] = mapped_column(String(100), nullable=False)
    mean: Mapped[float] = mapped_column(Float, nullable=False)
    std: Mapped[float] = mapped_column(Float, nullable=False)
    percentile_25: Mapped[float | None] = mapped_column(Float, nullable=True)
    percentile_75: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_points: Mapped[int] = mapped_column(nullable=False)
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_user_baselines_user_biomarker", "user_id", "biomarker_name", unique=True
        ),
    )


class AnalysisRun(Base):
    """Analysis run metadata and pipeline trace.

    Stores metadata for each analysis run including configuration used,
    time window, and the full pipeline trace for white-box explainability
    (Story 4.12, FR-022, NFR9).
    """

    __tablename__ = "analysis_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Start of analysis time window",
    )
    end_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="End of analysis time window",
    )
    config_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Analysis configuration used for this run",
    )
    # Story 4.12: Pipeline trace for transparency
    pipeline_trace: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Full pipeline trace for white-box explainability",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (Index("ix_analysis_runs_user_time", "user_id", "start_time"),)


class ContextHistoryRecord(Base):
    """Context evaluation history for temporal binding.

    Stores historical context evaluations so that biomarker windows can be
    bound to the correct context that was active at that time. Supports
    forward-fill queries and gap detection for context-aware analysis.

    Story 6.1: Context History Infrastructure
    """

    __tablename__ = "context_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        doc="Timestamp of context evaluation",
    )
    context_state: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="EMA-smoothed context confidence scores",
    )
    raw_scores: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Raw (pre-smoothing) context scores for transparency",
    )
    dominant_context: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="The highest-confidence context at evaluation time",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Confidence score for dominant context (0-1)",
    )
    evaluation_trigger: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Trigger type: 'on_demand', 'manual', 'backfill'",
    )
    sensors_used: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="List of sensor types that contributed to evaluation",
    )
    sensor_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Raw sensor values at evaluation time (optional)",
    )
    # Step 5: Stabilization transparency fields (hysteresis & dwell time)
    switch_blocked: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        doc="Whether stabilization prevented a context switch",
    )
    switch_blocked_reason: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Why switch was blocked: 'hysteresis' or 'dwell_time'",
    )
    candidate_context: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        doc="Context that would win without stabilization",
    )
    score_difference: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        doc="Score difference between candidate and current context",
    )
    dwell_progress: Mapped[int | None] = mapped_column(
        nullable=True,
        doc="Current dwell count for candidate context",
    )
    dwell_required: Mapped[int | None] = mapped_column(
        nullable=True,
        doc="Number of consecutive readings required to switch",
    )
    # Story 6.13: Link to context evaluation run for versioning
    context_evaluation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("context_evaluation_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Reference to ContextEvaluationRun that created this record",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Unique constraint: one evaluation per user per timestamp per run
        # Story 6.13: Include run_id to allow different runs to have evaluations
        # at the same timestamp. COALESCE handles NULL run_ids for legacy records.
        Index(
            "uq_context_history_user_time_run",
            "user_id",
            "evaluated_at",
            "context_evaluation_run_id",
            unique=True,
        ),
        # Index for fast range queries on evaluated_at
        Index("ix_context_history_evaluated_at", "evaluated_at"),
        # Index for context-specific queries
        Index(
            "ix_context_history_dominant_evaluated",
            "dominant_context",
            "evaluated_at",
        ),
        # Story 6.13: Index for filtering history by user and run
        Index(
            "ix_context_history_user_run",
            "user_id",
            "context_evaluation_run_id",
        ),
    )


class ContextEvaluationRun(Base):
    """Context evaluation run for versioned context evaluations.

    Stores metadata for each context evaluation run, enabling experimentation
    with different fuzzy logic settings and reproducible context evaluations.

    Story 6.13: Context Evaluation Experimentation
    """

    __tablename__ = "context_evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        doc="User ID for which context was evaluated",
    )
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Start of evaluated time period (None = all history)",
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="End of evaluated time period (None = all history)",
    )
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Full ExperimentContextEvalConfig including EMA parameters",
    )
    evaluation_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0,
        doc="Number of ContextHistoryRecord rows created by this run",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Index for finding runs by user and date range
        Index("ix_context_eval_runs_user_start", "user_id", "start_time"),
        # Index for listing recent runs by user
        Index("ix_context_eval_runs_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"<ContextEvaluationRun(id={self.id}, user_id='{self.user_id}', "
            f"evaluation_count={self.evaluation_count})>"
        )


class ConfigExperiment(Base):
    """Stored configuration experiment for A/B testing analysis parameters.

    Allows analysts to create, store, and test alternative configurations
    against the default configuration (Story 5.9 - FR-032).
    """

    __tablename__ = "config_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Human-readable name for the experiment",
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Optional description of the experiment",
    )
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        doc="Full analysis configuration as JSON",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<ConfigExperiment(id={self.id}, name='{self.name}')>"


def init_db() -> None:
    """Create all tables in the database."""
    from src.shared.database import engine

    Base.metadata.create_all(bind=engine)
