"""Pydantic schemas for biomarker endpoints."""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BiomarkerCreate(BaseModel):
    """Schema for creating a biomarker."""

    model_config = ConfigDict()

    user_id: str = Field(..., min_length=1, description="User identifier")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of biomarker")
    biomarker_type: Literal["speech", "network"] = Field(
        ..., description="Type of biomarker"
    )
    value: dict[str, Any] = Field(..., description="Biomarker value data")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata"
    )


class BiomarkerResponse(BaseModel):
    """Schema for biomarker response."""

    model_config = ConfigDict(strict=True, from_attributes=True, populate_by_name=True)

    id: UUID
    user_id: str
    timestamp: datetime
    biomarker_type: str
    value: dict[str, Any]
    metadata: dict[str, Any] | None = Field(validation_alias="metadata_")
    created_at: datetime


class BiomarkerBatchCreate(BaseModel):
    """Schema for batch biomarker creation."""

    model_config = ConfigDict()

    items: list[BiomarkerCreate] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of biomarkers to create (max 1000)",
    )


class BatchResponse(BaseModel):
    """Schema for batch operation response."""

    model_config = ConfigDict(strict=True)

    created_count: int = Field(..., description="Number of records created")


class BiomarkerQuery(BaseModel):
    """Query parameters for biomarker retrieval."""

    user_id: str = Field(..., description="User ID (required)")
    start_time: datetime | None = Field(None, description="Start of time range (ISO 8601)")
    end_time: datetime | None = Field(None, description="End of time range (ISO 8601)")
    biomarker_type: Literal["speech", "network"] | None = Field(
        None, description="Filter by biomarker type"
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
