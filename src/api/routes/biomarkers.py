"""Biomarker ingestion and retrieval routes."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.dependencies import verify_api_key
from src.api.schemas.biomarker import (
    BatchResponse,
    BiomarkerBatchCreate,
    BiomarkerCreate,
    BiomarkerQuery,
    BiomarkerResponse,
)
from src.api.schemas.responses import ApiResponse
from src.shared.database import get_db
from src.shared.models import Biomarker

router = APIRouter(
    prefix="/biomarkers",
    tags=["biomarkers"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "",
    response_model=ApiResponse[BiomarkerResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_biomarker(
    biomarker_in: BiomarkerCreate,
    db: Session = Depends(get_db),
) -> ApiResponse[BiomarkerResponse]:
    """Create a new biomarker record."""
    biomarker = Biomarker(
        user_id=biomarker_in.user_id,
        timestamp=biomarker_in.timestamp,
        biomarker_type=biomarker_in.biomarker_type,
        value=biomarker_in.value,
        metadata_=biomarker_in.metadata,
    )
    db.add(biomarker)
    db.commit()
    db.refresh(biomarker)

    return ApiResponse(data=BiomarkerResponse.model_validate(biomarker))


@router.post(
    "/batch",
    response_model=ApiResponse[BatchResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_biomarkers_batch(
    batch_in: BiomarkerBatchCreate,
    db: Session = Depends(get_db),
) -> ApiResponse[BatchResponse]:
    """Create multiple biomarker records in a single transaction.

    All records are validated before any are saved. If any record
    fails validation, the entire batch is rejected.
    """
    biomarkers = [
        Biomarker(
            user_id=item.user_id,
            timestamp=item.timestamp,
            biomarker_type=item.biomarker_type,
            value=item.value,
            metadata_=item.metadata,
        )
        for item in batch_in.items
    ]

    try:
        db.add_all(biomarkers)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return ApiResponse(data=BatchResponse(created_count=len(biomarkers)))


@router.get(
    "",
    response_model=ApiResponse[list[BiomarkerResponse]],
    status_code=status.HTTP_200_OK,
)
def get_biomarkers(
    query: BiomarkerQuery = Depends(),
    db: Session = Depends(get_db),
) -> ApiResponse[list[BiomarkerResponse]]:
    """Retrieve biomarker records for a user within a time range.

    Filters:
    - user_id: Required - user identifier
    - start_time: Optional - start of time range (defaults to 24 hours ago)
    - end_time: Optional - end of time range (defaults to now)
    - biomarker_type: Optional - filter by type (speech or network)

    Results are ordered by timestamp ascending to preserve temporal order.
    """
    # Build query using SQLAlchemy 2.0 select() style
    stmt = select(Biomarker).where(Biomarker.user_id == query.user_id)

    # Apply time range filters
    if query.start_time:
        stmt = stmt.where(Biomarker.timestamp >= query.start_time)
    if query.end_time:
        stmt = stmt.where(Biomarker.timestamp <= query.end_time)

    # Apply type filter if provided
    if query.biomarker_type:
        stmt = stmt.where(Biomarker.biomarker_type == query.biomarker_type)

    # Order by timestamp (NFR3: preserve temporal ordering)
    stmt = stmt.order_by(Biomarker.timestamp.asc())

    # Execute query
    biomarkers = db.execute(stmt).scalars().all()

    # Convert to response models
    biomarker_responses = [
        BiomarkerResponse.model_validate(biomarker) for biomarker in biomarkers
    ]

    return ApiResponse(
        data=biomarker_responses,
        meta={"count": len(biomarker_responses)},
    )
