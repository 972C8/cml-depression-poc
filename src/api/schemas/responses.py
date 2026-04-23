from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(strict=True)

    status: str


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    model_config = ConfigDict(strict=True)

    data: T
    meta: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    """Error detail structure."""

    model_config = ConfigDict(strict=True)

    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(strict=True)

    error: ErrorDetail
