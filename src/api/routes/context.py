"""Context ingestion and retrieval routes."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import verify_api_key
from src.api.schemas.biomarker import BatchResponse
from src.api.schemas.context import (
    ContextBatchCreate,
    ContextCreate,
    ContextQuery,
    ContextResponse,
)
from src.api.schemas.responses import ApiResponse
from src.shared.database import get_db
from src.shared.models import Context

router = APIRouter(
    prefix="/context",
    tags=["context"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "",
    response_model=ApiResponse[ContextResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_context(
    context_in: ContextCreate,
    db: Session = Depends(get_db),
) -> ApiResponse[ContextResponse]:
    """Create a new context data record."""
    context = Context(
        user_id=context_in.user_id,
        timestamp=context_in.timestamp,
        context_type=context_in.context_type,
        value=context_in.value,
        metadata_=context_in.metadata,
    )
    db.add(context)
    db.commit()
    db.refresh(context)

    return ApiResponse(data=ContextResponse.model_validate(context))


@router.post(
    "/batch",
    response_model=ApiResponse[BatchResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_context_batch(
    batch_in: ContextBatchCreate,
    db: Session = Depends(get_db),
) -> ApiResponse[BatchResponse]:
    """Create multiple context records in a single transaction.

    All records are validated before any are saved. If any record
    fails validation, the entire batch is rejected.
    """
    contexts = [
        Context(
            user_id=item.user_id,
            timestamp=item.timestamp,
            context_type=item.context_type,
            value=item.value,
            metadata_=item.metadata,
        )
        for item in batch_in.items
    ]

    try:
        db.add_all(contexts)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return ApiResponse(data=BatchResponse(created_count=len(contexts)))


@router.get(
    "",
    response_model=ApiResponse[list[ContextResponse]],
    status_code=status.HTTP_200_OK,
)
def get_context(
    query: ContextQuery = Depends(),
    db: Session = Depends(get_db),
) -> ApiResponse[list[ContextResponse]]:
    """Retrieve context records for a user within a time range.

    Filters:
    - user_id: Required - user identifier
    - start_time: Optional - start of time range (defaults to 24 hours ago)
    - end_time: Optional - end of time range (defaults to now)
    - context_type: Optional - filter by context type (location, activity, etc.)

    Results are ordered by timestamp ascending to preserve temporal order.
    """
    # Build query using SQLAlchemy 2.0 select() style
    stmt = select(Context).where(Context.user_id == query.user_id)

    # Apply time range filters
    if query.start_time:
        stmt = stmt.where(Context.timestamp >= query.start_time)
    if query.end_time:
        stmt = stmt.where(Context.timestamp <= query.end_time)

    # Apply type filter if provided
    if query.context_type:
        stmt = stmt.where(Context.context_type == query.context_type)

    # Order by timestamp (NFR3: preserve temporal ordering)
    stmt = stmt.order_by(Context.timestamp.asc())

    # Execute query
    contexts = db.execute(stmt).scalars().all()

    # Convert to response models
    context_responses = [
        ContextResponse.model_validate(context) for context in contexts
    ]

    return ApiResponse(
        data=context_responses,
        meta={"count": len(context_responses)},
    )
