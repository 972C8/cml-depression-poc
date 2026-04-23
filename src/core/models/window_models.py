"""Window aggregation data models.

Story 6.2: Window Aggregation Module (AC1)
Story 6.3: Membership with Context Weighting (AC1)
Story 6.4: Window-Level FASL Aggregation (AC1)

Defines immutable data structures for aggregated biomarker readings
within time windows and their context-weighted membership values.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

__all__ = [
    "WindowAggregate",
    "WindowIndicator",
    "WindowMembership",
    "FASLContribution",
]


@dataclass(frozen=True)
class WindowAggregate:
    """Aggregated biomarker readings within a time window.

    Represents the result of aggregating multiple biomarker readings
    that fall within a single time window (tumbling window approach).

    Attributes:
        biomarker_name: Name of the biomarker (e.g., "speech_activity")
        window_start: Start of the time window (inclusive)
        window_end: End of the time window (exclusive)
        aggregated_value: Computed aggregate (mean/median/max/min)
        readings_count: Number of readings in this window
        readings_timestamps: Original timestamps of readings (for context lookup)
        aggregation_method: Method used for aggregation (default: "mean")
    """

    biomarker_name: str
    window_start: datetime
    window_end: datetime
    aggregated_value: float
    readings_count: int
    readings_timestamps: tuple[datetime, ...]  # Tuple for frozen dataclass
    aggregation_method: str = "mean"


@dataclass(frozen=True)
class WindowMembership:
    """Membership value for a biomarker window with context information.

    Represents the result of computing membership for a single time window,
    including context state for downstream FASL computation.

    Story 6.3: Membership with Context Weighting (AC1)
    Story 6.10: Context weights applied in FASL formula, not to membership

    Attributes:
        biomarker_name: Name of the biomarker (e.g., "speech_activity")
        window_start: Start of the time window (inclusive)
        window_end: End of the time window (exclusive)
        aggregated_value: Computed aggregate from WindowAggregate
        z_score: Normalized value relative to baseline ((value - mean) / std)
        membership: Raw membership value [0, 1] from sigmoid of z-score
        context_strategy: Strategy used ('dominant', 'time_weighted', 'reading_weighted')
        context_state: All context activations as dict (context_name -> confidence)
        dominant_context: The highest-confidence context name
        context_weight: Context weight multiplier (passed to FASL, NOT applied to membership)
        context_confidence: Raw float confidence in [0.0, 1.0] for context blending
        weighted_membership: Equals membership (Story 6.10: context weight NOT applied here)
        readings_count: Number of readings in this window

    Note:
        Story 6.10 moved context weight application from membership to FASL formula.
        The `weighted_membership` field is kept for backward compatibility but now
        equals `membership`. Context weights and confidence are passed through to
        FASL computation where they are applied using the formula:
        effective_context_weight = 1.0 + (context_weight - 1.0) × confidence
    """

    biomarker_name: str
    window_start: datetime
    window_end: datetime
    aggregated_value: float
    z_score: float
    membership: float
    context_strategy: Literal["dominant", "time_weighted", "reading_weighted"]
    context_state: dict[str, float]
    dominant_context: str
    context_weight: float
    context_confidence: float
    weighted_membership: float
    readings_count: int


@dataclass(frozen=True)
class FASLContribution:
    """Per-biomarker contribution details from FASL computation.

    Story 6.17: Captures the full FASL breakdown for each biomarker
    to enable faithful transparency display from persisted data.

    Attributes:
        biomarker: Biomarker name
        directed_membership: Membership after direction adjustment
        biomarker_weight: Base weight from indicator config
        context_weight: Context-specific weight multiplier
        effective_weight: Final weight (biomarker_weight * effective_context_weight)
        contribution: effective_weight * directed_membership
        is_missing: Whether this biomarker had no data (neutral fill)
    """

    biomarker: str
    directed_membership: float
    biomarker_weight: float
    context_weight: float
    effective_weight: float
    contribution: float
    is_missing: bool


@dataclass(frozen=True)
class WindowIndicator:
    """FASL indicator score for a single time window.

    Represents the result of applying FASL aggregation to biomarker
    memberships within a single time window, producing an indicator
    score that captures temporal co-occurrence patterns.

    Story 6.4: Window-Level FASL Aggregation (AC1)

    Attributes:
        window_start: Start of the time window (inclusive)
        window_end: End of the time window (exclusive)
        indicator_name: Name of the indicator (e.g., "social_withdrawal")
        indicator_score: FASL output value [0, 1]
        contributing_biomarkers: Mapping of biomarker name to weighted membership.
            Note: While this dataclass is frozen, the dict is mutable by Python's
            design. Callers should treat this as read-only for immutability.
        biomarkers_present: Number of biomarkers with data in this window
        biomarkers_expected: Total number of expected biomarkers for this indicator
        biomarker_completeness: Ratio of present to expected (present / expected)
        dominant_context: The highest-confidence context for this window
        context_confidence: Raw float confidence in [0.0, 1.0]
        fasl_contributions: Per-biomarker FASL breakdown for transparency
    """

    window_start: datetime
    window_end: datetime
    indicator_name: str
    indicator_score: float
    contributing_biomarkers: dict[str, float]
    biomarkers_present: int
    biomarkers_expected: int
    biomarker_completeness: float
    dominant_context: str
    context_confidence: float = 0.0
    fasl_contributions: tuple[FASLContribution, ...] = ()
