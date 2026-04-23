"""Window-level FASL aggregation module.

Story 6.4: Window-Level FASL Aggregation

Computes indicator scores at the window level by applying FASL
(Fuzzy-Aggregated Symptom Likelihood) to window memberships.
This preserves temporal co-occurrence detection that would be
lost with daily averaging.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from src.core.config import AnalysisConfig, IndicatorConfig
from src.core.models.window_models import (
    FASLContribution,
    WindowIndicator,
    WindowMembership,
)

__all__ = [
    "MembershipData",
    "apply_fasl_operator",
    "apply_missing_strategy",
    "compute_window_indicators",
]

_logger = logging.getLogger(__name__)

# Type alias for missing biomarker strategies
MissingBiomarkerStrategy = Literal["neutral_fill", "partial_fasl", "skip_window"]


@dataclass
class MembershipData:
    """Data container for membership values and context information.

    Story 6.10: Separates raw membership from context weights for FASL computation.
    """

    memberships: dict[str, float]
    context_weights: dict[str, float]
    confidences: dict[str, float]


def apply_missing_strategy(
    window_memberships: dict[str, WindowMembership],
    expected_biomarkers: list[str],
    strategy: MissingBiomarkerStrategy,
    neutral_value: float = 0.5,
) -> MembershipData | None:
    """Apply missing biomarker strategy to window memberships.

    Story 6.10: Returns raw memberships (not weighted by context) plus context
    weights and confidences for FASL computation.

    Handles incomplete windows where not all expected biomarkers have data.

    Args:
        window_memberships: Mapping of biomarker name to WindowMembership for this window
        expected_biomarkers: List of biomarker names expected for this indicator
        strategy: Missing biomarker handling strategy:
            - 'neutral_fill': Use neutral_value (default 0.5) for missing biomarkers
            - 'partial_fasl': Use only present biomarkers with renormalized weights
            - 'skip_window': Don't compute indicator for incomplete windows
        neutral_value: Value to use for missing biomarkers in neutral_fill (default: 0.5)

    Returns:
        MembershipData with raw memberships, context_weights, and confidences,
        or None if skip_window strategy and biomarkers are missing.
    """
    present = set(window_memberships.keys())
    expected = set(expected_biomarkers)
    missing = expected - present

    if strategy == "skip_window" and missing:
        return None

    # Extract raw memberships (not weighted by context)
    memberships = {name: wm.membership for name, wm in window_memberships.items()}
    context_weights = {
        name: wm.context_weight for name, wm in window_memberships.items()
    }
    confidences = {
        name: wm.context_confidence for name, wm in window_memberships.items()
    }

    if strategy == "neutral_fill":
        # Fill missing with neutral value and neutral context
        for name in missing:
            memberships[name] = neutral_value
            context_weights[name] = 1.0  # Neutral context weight
            confidences[name] = 0.0  # No confidence in context

    # partial_fasl and skip_window (when complete) just use present biomarkers

    return MembershipData(
        memberships=memberships,
        context_weights=context_weights,
        confidences=confidences,
    )


def apply_fasl_operator(
    memberships: dict[str, float],
    weights: dict[str, float],
    directions: dict[str, Literal["higher_is_worse", "lower_is_worse"]],
    context_weights: dict[str, float] | None = None,
    confidences: dict[str, float] | None = None,
) -> float:
    """Apply FASL operator to combine biomarker memberships into indicator score.

    Story 6.10: Context weights are applied in FASL (not in membership computation).

    FASL Formula with Context Weights:
        effective_context_weight_i = 1.0 + (context_weight_i - 1.0) × confidence_i
        effective_weight_i = biomarker_weight_i × effective_context_weight_i
        L_k = SUM[effective_weight_i × directed_mu_i] / SUM[effective_weight_i]

    Where:
        L_k = indicator likelihood score [0, 1]
        mu_i = raw membership value [0, 1] (NOT weighted by context)
        directed_mu_i = mu_i (higher_is_worse) or 1-mu_i (lower_is_worse)
        context_weight_i = context-specific weight multiplier for biomarker
        confidence_i = confidence in context detection [0, 1]
        effective_context_weight_i = blended context weight based on confidence
        effective_weight_i = final weight for biomarker contribution

    Context Weight Semantics:
        - 0: Exclude biomarker from FASL (numerator AND denominator)
        - 1: Neutral (default, no adjustment)
        - > 1: Amplify this biomarker's contribution

    Args:
        memberships: Mapping of biomarker name -> raw membership value [0, 1]
        weights: Mapping of biomarker name -> biomarker weight
        directions: Mapping of biomarker name -> direction
        context_weights: Optional mapping of biomarker name -> context weight.
            Defaults to 1.0 (neutral) for all biomarkers if not provided.
        confidences: Optional mapping of biomarker name -> confidence [0, 1].
            Defaults to 1.0 if not provided. When confidence=0, context_weight
            has no effect (effective_context_weight=1.0).

    Returns:
        FASL indicator score in [0, 1], or NaN if all effective weights are 0.
    """
    if not memberships:
        return 0.0

    # Default context weights and confidences if not provided
    if context_weights is None:
        context_weights = {}
    if confidences is None:
        confidences = {}

    numerator = 0.0
    denominator = 0.0

    for biomarker_name, membership in memberships.items():
        biomarker_weight = weights.get(biomarker_name, 0.0)
        direction = directions.get(biomarker_name, "higher_is_worse")

        # Get context weight and confidence (default to neutral)
        context_weight = context_weights.get(biomarker_name, 1.0)
        confidence = confidences.get(biomarker_name, 1.0)

        # Compute effective context weight (Story 6.10 AC3)
        # confidence=0 → effective_context_weight=1.0 (neutral)
        # confidence=1 → effective_context_weight=context_weight (full effect)
        effective_context_weight = 1.0 + (context_weight - 1.0) * confidence

        # Compute effective weight (Story 6.10 AC2)
        effective_weight = biomarker_weight * effective_context_weight

        # Skip biomarker if effective weight is 0 (Story 6.10 AC4)
        if effective_weight == 0.0:
            continue

        # Apply direction handling to raw membership
        if direction == "higher_is_worse":
            directed_mu = membership
        else:  # lower_is_worse
            directed_mu = 1.0 - membership

        # Clip to [0, 1] range (per paper: clip negative contributions to zero)
        directed_mu = max(0.0, min(1.0, directed_mu))

        contribution = effective_weight * directed_mu
        numerator += contribution
        denominator += effective_weight

    # Story 6.10 AC4: Return NaN if all effective weights are 0
    if denominator == 0:
        return math.nan

    return numerator / denominator


def _apply_fasl_with_contributions(
    memberships: dict[str, float],
    weights: dict[str, float],
    directions: dict[str, Literal["higher_is_worse", "lower_is_worse"]],
    context_weights: dict[str, float],
    confidences: dict[str, float],
    expected_biomarkers: list[str],
    present_biomarkers: set[str],
) -> tuple[float, list[FASLContribution]]:
    """Apply FASL and capture per-biomarker contribution details.

    Story 6.17: Wraps apply_fasl_operator to also capture the per-biomarker
    breakdown needed for transparency persistence.

    Returns:
        Tuple of (indicator_score, list of FASLContribution)
    """
    contributions: list[FASLContribution] = []
    numerator = 0.0
    denominator = 0.0

    for biomarker_name in memberships:
        membership = memberships[biomarker_name]
        biomarker_weight = weights.get(biomarker_name, 0.0)
        direction = directions.get(biomarker_name, "higher_is_worse")
        context_weight = context_weights.get(biomarker_name, 1.0)
        confidence = confidences.get(biomarker_name, 1.0)
        is_missing = biomarker_name not in present_biomarkers

        # Compute effective context weight
        effective_context_weight = 1.0 + (context_weight - 1.0) * confidence
        effective_weight = biomarker_weight * effective_context_weight

        # Apply direction
        if direction == "higher_is_worse":
            directed_mu = membership
        else:
            directed_mu = 1.0 - membership
        directed_mu = max(0.0, min(1.0, directed_mu))

        if effective_weight != 0.0:
            contribution = effective_weight * directed_mu
            numerator += contribution
            denominator += effective_weight
        else:
            contribution = 0.0

        contributions.append(FASLContribution(
            biomarker=biomarker_name,
            directed_membership=directed_mu,
            biomarker_weight=biomarker_weight,
            context_weight=context_weight,
            effective_weight=effective_weight,
            contribution=contribution,
            is_missing=is_missing,
        ))

    if denominator == 0:
        return math.nan, contributions

    return numerator / denominator, contributions


def compute_window_indicators(
    window_memberships: dict[str, list[WindowMembership]],
    indicator_name: str,
    indicator_config: IndicatorConfig,
    config: AnalysisConfig,
) -> list[WindowIndicator]:
    """Compute window-level indicator scores using FASL aggregation.

    Applies FASL to combine biomarker memberships within each time window,
    producing indicator scores that capture temporal co-occurrence patterns.

    Args:
        window_memberships: Mapping of biomarker name to list of WindowMembership
            (from Story 6.3 compute_window_memberships)
        indicator_name: Name of the indicator being computed
        indicator_config: Configuration for this indicator (biomarker mappings)
        config: Analysis configuration with FASL settings

    Returns:
        List of WindowIndicator, one per unique window across all biomarkers,
        sorted chronologically. Windows with incomplete data are handled per
        configured strategy.
    """
    if not window_memberships:
        return []

    # Get FASL configuration
    # Default strategy is 'neutral_fill' per AC3
    missing_strategy: MissingBiomarkerStrategy = getattr(
        config, "fasl_missing_strategy", "neutral_fill"
    )
    if hasattr(config, "fasl") and hasattr(config.fasl, "missing_biomarker_strategy"):
        missing_strategy = config.fasl.missing_biomarker_strategy

    neutral_value = 0.5
    if hasattr(config, "fasl") and hasattr(config.fasl, "neutral_membership"):
        neutral_value = config.fasl.neutral_membership

    # Extract expected biomarkers and their weights/directions from indicator config
    expected_biomarkers = list(indicator_config.biomarkers.keys())
    weights = {
        name: bio_config.weight
        for name, bio_config in indicator_config.biomarkers.items()
    }
    directions = {
        name: bio_config.direction
        for name, bio_config in indicator_config.biomarkers.items()
    }

    # Step 1: Collect all unique windows across all biomarkers
    # Use (window_start, window_end) as window key
    all_windows: set[tuple[datetime, datetime]] = set()
    for memberships_list in window_memberships.values():
        for wm in memberships_list:
            all_windows.add((wm.window_start, wm.window_end))

    # Step 2: For each window, gather memberships and compute indicator
    results: list[WindowIndicator] = []

    for window_start, window_end in sorted(all_windows):
        # Gather memberships for this window from all biomarkers
        window_bio_memberships: dict[str, WindowMembership] = {}
        for biomarker_name, memberships_list in window_memberships.items():
            for wm in memberships_list:
                if wm.window_start == window_start and wm.window_end == window_end:
                    window_bio_memberships[biomarker_name] = wm
                    break

        # Apply missing biomarker strategy
        membership_data = apply_missing_strategy(
            window_memberships=window_bio_memberships,
            expected_biomarkers=expected_biomarkers,
            strategy=missing_strategy,
            neutral_value=neutral_value,
        )

        if membership_data is None:
            # skip_window strategy and incomplete
            _logger.debug(
                "Skipping window %s-%s for indicator '%s' due to missing biomarkers",
                window_start,
                window_end,
                indicator_name,
            )
            continue

        # Step 3: Apply FASL operator with context weights (Story 6.10)
        # Also capture per-biomarker contributions for transparency (Story 6.17)
        indicator_score, fasl_contributions = _apply_fasl_with_contributions(
            memberships=membership_data.memberships,
            weights=weights,
            directions=directions,
            context_weights=membership_data.context_weights,
            confidences=membership_data.confidences,
            expected_biomarkers=expected_biomarkers,
            present_biomarkers=set(window_bio_memberships.keys()),
        )

        # Handle NaN score (all effective weights were 0)
        if math.isnan(indicator_score):
            _logger.debug(
                "Skipping window %s-%s for indicator '%s': all effective weights are 0",
                window_start,
                window_end,
                indicator_name,
            )
            continue

        # Determine dominant context and confidence (from first present biomarker)
        dominant_context = "neutral"
        context_confidence = 0.0
        for wm in window_bio_memberships.values():
            dominant_context = wm.dominant_context
            context_confidence = wm.context_confidence
            break

        # Track contributing biomarkers with their raw memberships (Story 6.10)
        contributing_biomarkers = {
            name: wm.membership for name, wm in window_bio_memberships.items()
        }

        # Calculate completeness (only count biomarkers expected for this indicator)
        biomarkers_present = len(set(window_bio_memberships.keys()) & set(expected_biomarkers))
        biomarkers_expected = len(expected_biomarkers)
        biomarker_completeness = (
            biomarkers_present / biomarkers_expected if biomarkers_expected > 0 else 0.0
        )

        # Build WindowIndicator
        window_indicator = WindowIndicator(
            window_start=window_start,
            window_end=window_end,
            indicator_name=indicator_name,
            indicator_score=indicator_score,
            contributing_biomarkers=contributing_biomarkers,
            biomarkers_present=biomarkers_present,
            biomarkers_expected=biomarkers_expected,
            biomarker_completeness=biomarker_completeness,
            dominant_context=dominant_context,
            context_confidence=context_confidence,
            fasl_contributions=tuple(fasl_contributions),
        )
        results.append(window_indicator)

        _logger.debug(
            "Window %s-%s: indicator=%s, score=%.3f, completeness=%.2f",
            window_start,
            window_end,
            indicator_name,
            indicator_score,
            biomarker_completeness,
        )

    _logger.info(
        "Computed %d window indicators for '%s' (strategy=%s)",
        len(results),
        indicator_name,
        missing_strategy,
    )

    return results
