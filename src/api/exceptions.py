"""Custom exception handlers for API error responses."""

import uuid

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas.responses import ErrorDetail, ErrorResponse


def get_request_id(request: Request) -> str:
    """Get or generate request ID."""
    return getattr(request.state, "request_id", str(uuid.uuid4()))


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle HTTPException and unwrap detail if it's already formatted."""
    # If detail is already a dict with 'error' key, return it directly
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
        )

    # Otherwise, wrap in standard error format
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code="HTTP_ERROR",
                message=str(exc.detail),
                details={"request_id": get_request_id(request)},
            )
        ).model_dump(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = exc.errors()

    # Check if this is a JSON decode error (AC1)
    if errors and errors[0].get("type") == "json_invalid":
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="INVALID_JSON",
                    message="Request body contains invalid JSON",
                    details={
                        "message": errors[0].get("msg", "JSON decode error"),
                        "request_id": get_request_id(request),
                    },
                )
            ).model_dump(),
        )

    # Regular validation errors (AC2, AC3, AC4)
    formatted_errors = []
    for error in errors:
        formatted_errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message="Request validation failed",
                details={
                    "errors": formatted_errors,
                    "request_id": get_request_id(request),
                },
            )
        ).model_dump(),
    )
