"""Pydantic schemas for indicator endpoints."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IndicatorResponse(BaseModel):
    """Response model for indicator data."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Indicator unique identifier")
    user_id: str = Field(..., description="User identifier")
    timestamp: datetime = Field(..., description="Timestamp of the indicator")
    indicator_type: str = Field(..., description="Type of indicator")
    value: float = Field(..., description="Computed indicator value")
    data_reliability_score: float | None = Field(None, description="Data reliability score (0-1)")
    analysis_run_id: UUID = Field(
        ..., description="Analysis run ID for traceability"
    )
    config_snapshot: dict | None = Field(
        None, description="Analysis configuration used"
    )
    created_at: datetime = Field(..., description="When the indicator was created")


class IndicatorQuery(BaseModel):
    """Query parameters for indicator retrieval."""

    model_config = ConfigDict()

    user_id: str = Field(..., description="User ID (required)")
    start_time: datetime | None = Field(
        None, description="Start of time range (ISO 8601)"
    )
    end_time: datetime | None = Field(None, description="End of time range (ISO 8601)")
    indicator_type: str | None = Field(
        None, description="Filter by indicator type"
    )

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def set_default_time_range(cls, v, info):
        """Set default time range to last 24 hours if not provided."""
        if info.field_name == "start_time" and v is None:
            return datetime.now(UTC) - timedelta(days=1)
        if info.field_name == "end_time" and v is None:
            return datetime.now(UTC)
        return v
