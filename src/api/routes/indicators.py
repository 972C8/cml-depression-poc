"""Indicator retrieval routes."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import verify_api_key
from src.api.schemas.indicator import IndicatorQuery, IndicatorResponse
from src.api.schemas.responses import ApiResponse
from src.shared.database import get_db
from src.shared.models import Indicator

router = APIRouter(
    prefix="/indicators",
    tags=["indicators"],
    dependencies=[Depends(verify_api_key)],
)


@router.get(
    "",
    response_model=ApiResponse[list[IndicatorResponse]],
    status_code=status.HTTP_200_OK,
)
def get_indicators(
    query: IndicatorQuery = Depends(),
    db: Session = Depends(get_db),
) -> ApiResponse[list[IndicatorResponse]]:
    """Retrieve indicator records for a user within a time range.

    Filters:
    - user_id: Required - user identifier
    - start_time: Optional - start of time range (defaults to 24 hours ago)
    - end_time: Optional - end of time range (defaults to now)
    - indicator_type: Optional - filter by indicator type

    Results are ordered by timestamp ascending to preserve temporal order.
    Each indicator includes analysis_run_id for traceability.
    """
    # Build query using SQLAlchemy 2.0 select() style
    stmt = select(Indicator).where(Indicator.user_id == query.user_id)

    # Apply time range filters
    if query.start_time:
        stmt = stmt.where(Indicator.timestamp >= query.start_time)
    if query.end_time:
        stmt = stmt.where(Indicator.timestamp <= query.end_time)

    # Apply type filter if provided
    if query.indicator_type:
        stmt = stmt.where(Indicator.indicator_type == query.indicator_type)

    # Order by timestamp (NFR3: preserve temporal ordering)
    stmt = stmt.order_by(Indicator.timestamp.asc())

    # Execute query
    indicators = db.execute(stmt).scalars().all()

    # Convert to response models
    indicator_responses = [
        IndicatorResponse.model_validate(indicator) for indicator in indicators
    ]

    return ApiResponse(
        data=indicator_responses,
        meta={"count": len(indicator_responses)},
    )
