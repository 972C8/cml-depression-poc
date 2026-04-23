"""Data reader module for analysis engine.

Reads biomarker and context data from the database and expands JSON values
into individual records for analysis processing.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.shared.models import Biomarker, Context

__all__ = [
    # Dataclasses
    "BiomarkerRecord",
    "ContextRecord",
    "DataReaderResult",
    "DataStats",
    # Main class
    "DataReader",
]

# ============================================================================
# Dataclasses (Tasks 1-4)
# ============================================================================


@dataclass(frozen=True)
class BiomarkerRecord:
    """Single biomarker value extracted from a database row.

    One DB row with multiple biomarkers in its value JSON becomes
    multiple BiomarkerRecord instances.
    """

    id: str  # UUID as string
    user_id: str
    timestamp: datetime  # UTC normalized
    biomarker_type: str  # e.g., "speech", "network"
    name: str  # e.g., "speech_activity", "bytes_in"
    value: float  # The actual value (extracted from JSON)
    raw_value: dict[str, Any]  # Original JSON dict from DB row
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ContextRecord:
    """Single context marker value extracted from a database row."""

    id: str
    user_id: str
    timestamp: datetime  # UTC normalized
    context_type: str  # e.g., "environment"
    name: str  # e.g., "people_in_room", "ambient_noise"
    value: float
    raw_value: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class DataStats:
    """Statistics about retrieved data."""

    biomarker_count: int
    context_count: int
    time_range_start: datetime | None
    time_range_end: datetime | None
    biomarker_types_found: frozenset[str]  # Use frozenset for frozen dataclass
    biomarker_names_found: frozenset[str]
    context_names_found: frozenset[str]

    @property
    def has_data(self) -> bool:
        """Return True if any data was found."""
        return self.biomarker_count > 0 or self.context_count > 0


@dataclass(frozen=True)
class DataReaderResult:
    """Complete result from DataReader.read_all()."""

    biomarkers: tuple[BiomarkerRecord, ...]  # Use tuple for frozen dataclass
    context_markers: tuple[ContextRecord, ...]
    biomarkers_by_type: dict[str, tuple[BiomarkerRecord, ...]]
    biomarkers_by_name: dict[str, tuple[BiomarkerRecord, ...]]
    context_by_name: dict[str, tuple[ContextRecord, ...]]
    stats: DataStats


# ============================================================================
# DataReader Class (Tasks 5-12)
# ============================================================================


class DataReader:
    """Read and expand biomarker/context data from database.

    Provides domain-specific data access for the analysis engine.
    Returns dataclasses (not ORM objects) for decoupling.
    """

    def __init__(self, session: Session) -> None:
        """Initialize with database session.

        Args:
            session: SQLAlchemy session (via dependency injection)
        """
        self._session = session
        self._logger = logging.getLogger(__name__)

    # Task 6: Timezone normalization
    def _normalize_to_utc(self, dt: datetime) -> datetime:
        """Normalize datetime to UTC.

        Args:
            dt: Input datetime (may be naive or aware)

        Returns:
            UTC-aware datetime
        """
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            return dt.replace(tzinfo=UTC)
        else:
            # Aware datetime - convert to UTC
            return dt.astimezone(UTC)

    # Task 5: Row expansion logic
    def _expand_biomarker_row(self, orm_obj: Biomarker) -> list[BiomarkerRecord]:
        """Expand one ORM row into multiple BiomarkerRecord objects.

        Args:
            orm_obj: SQLAlchemy Biomarker model instance

        Returns:
            List of BiomarkerRecord, one per biomarker in the value JSON
        """
        records = []
        for name, val in orm_obj.value.items():
            record = BiomarkerRecord(
                id=str(orm_obj.id),
                user_id=orm_obj.user_id,
                timestamp=self._normalize_to_utc(orm_obj.timestamp),
                biomarker_type=orm_obj.biomarker_type,
                name=name,
                value=float(val),
                raw_value=orm_obj.value,
                metadata=orm_obj.metadata_,
            )
            records.append(record)
        return records

    def _expand_context_row(self, orm_obj: Context) -> list[ContextRecord]:
        """Expand one ORM row into multiple ContextRecord objects.

        Args:
            orm_obj: SQLAlchemy Context model instance

        Returns:
            List of ContextRecord, one per context marker in the value JSON
        """
        records = []
        for name, val in orm_obj.value.items():
            record = ContextRecord(
                id=str(orm_obj.id),
                user_id=orm_obj.user_id,
                timestamp=self._normalize_to_utc(orm_obj.timestamp),
                context_type=orm_obj.context_type,
                name=name,
                value=float(val),
                raw_value=orm_obj.value,
                metadata=orm_obj.metadata_,
            )
            records.append(record)
        return records

    # Task 10: Grouping functions
    def _group_biomarkers_by_type(
        self,
        records: list[BiomarkerRecord],
    ) -> dict[str, tuple[BiomarkerRecord, ...]]:
        """Group biomarker records by type."""
        grouped: dict[str, list[BiomarkerRecord]] = defaultdict(list)
        for record in records:
            grouped[record.biomarker_type].append(record)
        # Convert lists to tuples for frozen dataclass compatibility
        return {k: tuple(v) for k, v in grouped.items()}

    def _group_biomarkers_by_name(
        self,
        records: list[BiomarkerRecord],
    ) -> dict[str, tuple[BiomarkerRecord, ...]]:
        """Group biomarker records by name."""
        grouped: dict[str, list[BiomarkerRecord]] = defaultdict(list)
        for record in records:
            grouped[record.name].append(record)
        return {k: tuple(v) for k, v in grouped.items()}

    def _group_context_by_name(
        self,
        records: list[ContextRecord],
    ) -> dict[str, tuple[ContextRecord, ...]]:
        """Group context records by name."""
        grouped: dict[str, list[ContextRecord]] = defaultdict(list)
        for record in records:
            grouped[record.name].append(record)
        return {k: tuple(v) for k, v in grouped.items()}

    # Task 11: Compute stats function
    def _compute_stats(
        self,
        biomarkers: list[BiomarkerRecord],
        context_markers: list[ContextRecord],
    ) -> DataStats:
        """Compute statistics about retrieved data.

        Args:
            biomarkers: List of biomarker records
            context_markers: List of context records

        Returns:
            DataStats with counts, time ranges, and types/names found
        """
        # Calculate counts
        biomarker_count = len(biomarkers)
        context_count = len(context_markers)

        # Calculate time range
        all_timestamps = [r.timestamp for r in biomarkers] + [
            r.timestamp for r in context_markers
        ]
        time_range_start = min(all_timestamps) if all_timestamps else None
        time_range_end = max(all_timestamps) if all_timestamps else None

        # Collect unique types and names
        biomarker_types_found = frozenset(r.biomarker_type for r in biomarkers)
        biomarker_names_found = frozenset(r.name for r in biomarkers)
        context_names_found = frozenset(r.name for r in context_markers)

        return DataStats(
            biomarker_count=biomarker_count,
            context_count=context_count,
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            biomarker_types_found=biomarker_types_found,
            biomarker_names_found=biomarker_names_found,
            context_names_found=context_names_found,
        )

    # Task 8: Implement read_biomarkers method
    def read_biomarkers(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        biomarker_types: list[str] | None = None,
        names: list[str] | None = None,
    ) -> list[BiomarkerRecord]:
        """Read and expand biomarker data.

        Args:
            user_id: User identifier
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            biomarker_types: Filter by modality type (DB-level filter)
            names: Filter by biomarker name (post-expansion filter)

        Returns:
            List of expanded BiomarkerRecord objects
        """
        # Build query
        stmt = (
            select(Biomarker)
            .where(Biomarker.user_id == user_id)
            .where(Biomarker.timestamp >= start_time)
            .where(Biomarker.timestamp <= end_time)
        )

        # Apply type filter at DB level if provided
        if biomarker_types:
            stmt = stmt.where(Biomarker.biomarker_type.in_(biomarker_types))

        # Order by timestamp
        stmt = stmt.order_by(Biomarker.timestamp.asc())

        # Execute and expand
        orm_objects = self._session.execute(stmt).scalars().all()
        records = []
        for obj in orm_objects:
            records.extend(self._expand_biomarker_row(obj))

        # Apply name filter post-expansion if provided
        if names:
            names_set = set(names)
            records = [r for r in records if r.name in names_set]

        # Log statistics
        if not records:
            self._logger.warning(
                f"No biomarker data found for user={user_id}, "
                f"range=[{start_time}, {end_time}]"
            )
        else:
            types_found = {r.biomarker_type for r in records}
            names_found = {r.name for r in records}
            self._logger.info(
                f"Retrieved {len(records)} biomarker records: "
                f"types={types_found}, names={names_found}"
            )

        return records

    # Task 9: Implement read_context_markers method
    def read_context_markers(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        context_types: list[str] | None = None,
        names: list[str] | None = None,
    ) -> list[ContextRecord]:
        """Read and expand context marker data.

        Args:
            user_id: User identifier
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            context_types: Filter by context type (DB-level filter)
            names: Filter by context marker name (post-expansion filter)

        Returns:
            List of expanded ContextRecord objects
        """
        # Build query
        stmt = (
            select(Context)
            .where(Context.user_id == user_id)
            .where(Context.timestamp >= start_time)
            .where(Context.timestamp <= end_time)
        )

        # Apply type filter at DB level if provided
        if context_types:
            stmt = stmt.where(Context.context_type.in_(context_types))

        # Order by timestamp
        stmt = stmt.order_by(Context.timestamp.asc())

        # Execute and expand
        orm_objects = self._session.execute(stmt).scalars().all()
        records = []
        for obj in orm_objects:
            records.extend(self._expand_context_row(obj))

        # Apply name filter post-expansion if provided
        if names:
            names_set = set(names)
            records = [r for r in records if r.name in names_set]

        # Log statistics
        if not records:
            self._logger.warning(
                f"No context data found for user={user_id}, "
                f"range=[{start_time}, {end_time}]"
            )
        else:
            names_found = {r.name for r in records}
            self._logger.info(
                f"Retrieved {len(records)} context records: " f"names={names_found}"
            )

        return records

    # Task 12: Implement read_all method
    def read_all(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        biomarker_types: list[str] | None = None,
        biomarker_names: list[str] | None = None,
        context_types: list[str] | None = None,
        context_names: list[str] | None = None,
    ) -> DataReaderResult:
        """Read all data and return grouped results with statistics.

        Args:
            user_id: User identifier
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            biomarker_types: Filter by biomarker type (DB-level filter)
            biomarker_names: Filter by biomarker name (post-expansion filter)
            context_types: Filter by context type (DB-level filter)
            context_names: Filter by context name (post-expansion filter)

        Returns:
            DataReaderResult with all data, grouped views, and statistics
        """
        # Read biomarkers and context markers
        biomarkers = self.read_biomarkers(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            biomarker_types=biomarker_types,
            names=biomarker_names,
        )

        context_markers = self.read_context_markers(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            context_types=context_types,
            names=context_names,
        )

        # Group results
        biomarkers_by_type = self._group_biomarkers_by_type(biomarkers)
        biomarkers_by_name = self._group_biomarkers_by_name(biomarkers)
        context_by_name = self._group_context_by_name(context_markers)

        # Compute stats
        stats = self._compute_stats(biomarkers, context_markers)

        return DataReaderResult(
            biomarkers=tuple(biomarkers),
            context_markers=tuple(context_markers),
            biomarkers_by_type=biomarkers_by_type,
            biomarkers_by_name=biomarkers_by_name,
            context_by_name=context_by_name,
            stats=stats,
        )
