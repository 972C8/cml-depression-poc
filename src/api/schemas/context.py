"""Pydantic schemas for context endpoints."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContextCreate(BaseModel):
    """Schema for creating context data."""

    model_config = ConfigDict()

    user_id: str = Field(..., min_length=1, description="User identifier")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp")
    context_type: str = Field(
        ..., min_length=1, description="Type of context (location, activity, etc.)"
    )
    value: dict[str, Any] = Field(..., description="Context value data")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata"
    )


class ContextResponse(BaseModel):
    """Schema for context response."""

    model_config = ConfigDict(strict=True, from_attributes=True, populate_by_name=True)

    id: UUID
    user_id: str
    timestamp: datetime
    context_type: str
    value: dict[str, Any]
    metadata: dict[str, Any] | None = Field(validation_alias="metadata_")
    created_at: datetime


class ContextBatchCreate(BaseModel):
    """Schema for batch context creation."""

    model_config = ConfigDict()

    items: list[ContextCreate] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of context records to create (max 1000)",
    )


class ContextQuery(BaseModel):
    """Query parameters for context data retrieval."""

    model_config = ConfigDict()

    user_id: str = Field(..., description="User ID (required)")
    start_time: datetime | None = Field(
        None, description="Start of time range (ISO 8601)"
    )
    end_time: datetime | None = Field(None, description="End of time range (ISO 8601)")
    context_type: str | None = Field(
        None, description="Filter by context type (location, activity, environment, etc.)"
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
