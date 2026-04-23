"""API dependencies for authentication and authorization."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.schemas.responses import ErrorDetail, ErrorResponse
from src.shared.config import get_settings

# Define HTTPBearer security scheme for Swagger UI
security = HTTPBearer()


def create_error_response(
    code: str, message: str, request: Request, details: dict | None = None
) -> dict:
    """Create standardized error response with request_id.

    Args:
        code: Error code
        message: Error message
        request: FastAPI request object
        details: Optional additional details

    Returns:
        Error response dict
    """
    error_details = details or {}
    error_details["request_id"] = getattr(
        request.state, "request_id", "unknown"
    )

    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            details=error_details,
        )
    ).model_dump()


def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> None:
    """Verify API key from Authorization header.

    Args:
        credentials: HTTPBearer credentials (automatically extracts Bearer token)
        request: FastAPI request object for request_id

    Raises:
        HTTPException: 401 if key is invalid or missing

    Returns:
        None when key is valid (dependency satisfied)
    """
    settings = get_settings()

    token = credentials.credentials

    if not token or token not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=create_error_response(
                code="INVALID_API_KEY",
                message="Invalid API key",
                request=request,
            ),
        )

    # Key is valid - dependency satisfied
    return None
