"""Indicator score computation using FASL formula with context-aware weights.

This module computes indicator daily likelihood scores from biomarker membership values
using the Fuzzy-Aggregated Symptom Likelihood (FASL) formula, extended with
context-aware weight adjustments.

FASL = Fuzzy-Aggregated Symptom Likelihood (from paper Section 4.5)

Direction handling:
    - higher_is_worse: high mu (deviation above baseline) = concerning
      contribution = weight * mu
    - lower_is_worse: low mu (deviation below baseline) = concerning
      contribution = weight * (1 - mu)

This ensures both direction types contribute meaningfully to the indicator likelihood.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from src.core.config import AnalysisConfig, IndicatorConfig
from src.core.context.evaluator import ContextResult
from src.core.context.weights import AdjustedIndicatorWeights, adjust_biomarker_weights
from src.core.processors.biomarker_processor import (
    BiomarkerMembership,
    DailyBiomarkerMembership,
)

__all__ = [
    "BiomarkerContribution",
    "IndicatorComputer",
    "IndicatorScore",
    "compute_all_indicators",
    "compute_indicator",
]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiomarkerContribution:
    """Individual biomarker's contribution to an indicator score.

    Records how a single biomarker contributes to the indicator's daily likelihood,
    including weight adjustments from context.

    Attributes:
        name: Biomarker name (e.g., "speech_activity", "connections")
        membership: Membership value (mu) in [0,1] from Story 4.5
        direction: "pro" (higher_is_worse) or "contra" (lower_is_worse)
        base_weight: Original weight before context adjustment
        context_multiplier: Effective multiplier from context (from Story 4.7)
        adjusted_weight: Final weight after context adjustment (normalized)
        contribution: Final contribution to indicator likelihood
    """

    name: str
    membership: float
    direction: Literal["pro", "contra"]
    base_weight: float
    context_multiplier: float
    adjusted_weight: float
    contribution: float


@dataclass(frozen=True)
class IndicatorScore:
    """Result of indicator score computation using FASL formula.

    Contains the daily likelihood score and detailed breakdown of how each
    biomarker contributed, along with context and data reliability information.

    Attributes:
        indicator_name: Name of the indicator (e.g., "social_withdrawal")
        daily_likelihood: L_k in [0,1] - FASL output representing likelihood
        contributions: Per-biomarker breakdown (biomarker_name -> BiomarkerContribution)
        biomarkers_used: Names of biomarkers that were available for computation
        biomarkers_missing: Names of expected biomarkers that were not available
        data_reliability_score: Data reliability score based on data quality and coverage (0-1)
        context_applied: Name of the active context used for weight adjustment
        context_confidence: Confidence of the context detection (0-1)
        weights_before_context: Base weights before context adjustment
        weights_after_context: Normalized weights after context adjustment
        timestamp: When the computation was performed
    """

    indicator_name: str
    daily_likelihood: float
    contributions: dict[str, BiomarkerContribution]
    biomarkers_used: tuple[str, ...]
    biomarkers_missing: tuple[str, ...]
    data_reliability_score: float
    context_applied: str
    context_confidence: float
    weights_before_context: dict[str, float]
    weights_after_context: dict[str, float]
    timestamp: datetime


@dataclass(frozen=True)
class DailyIndicatorScore:
    """Indicator score for a specific day in the analysis window.

    Used for building time series of daily likelihoods for DSM-Gate.

    Attributes:
        date: The specific date this score represents
        indicator_name: Name of the indicator
        daily_likelihood: L_k(d) in [0,1] for this day
        biomarkers_used: Names of biomarkers available on this day
        biomarkers_missing: Names of expected biomarkers missing on this day
        data_reliability_score: Data quality score for this day
    """

    date: date
    indicator_name: str
    daily_likelihood: float
    biomarkers_used: tuple[str, ...]
    biomarkers_missing: tuple[str, ...]
    data_reliability_score: float


def _calculate_data_reliability(
    biomarkers_used: list[str],
    biomarkers_expected: list[str],
    biomarker_memberships: dict[str, BiomarkerMembership],
    coverage_weight: float = 0.6,
    quality_weight: float = 0.4,
) -> float:
    """Calculate data reliability score based on data availability and quality.

    Data Reliability = (coverage_factor * coverage_weight) + (quality_factor * quality_weight)

    Where:
    - coverage_factor = len(biomarkers_used) / len(biomarkers_expected)
    - quality_factor = mean(data_quality) for used biomarkers

    Args:
        biomarkers_used: List of biomarker names that were available
        biomarkers_expected: List of all biomarker names expected for this indicator
        biomarker_memberships: Dict of biomarker name -> BiomarkerMembership
        coverage_weight: Weight for coverage factor (default: 0.6)
        quality_weight: Weight for quality factor (default: 0.4)

    Returns:
        Data reliability score in [0, 1]
    """
    if not biomarkers_expected:
        return 0.0

    coverage = len(biomarkers_used) / len(biomarkers_expected)

    if not biomarkers_used:
        return coverage * coverage_weight  # No quality data available

    total_quality = sum(
        biomarker_memberships[name].data_quality for name in biomarkers_used
    )
    avg_quality = total_quality / len(biomarkers_used)

    data_reliability = (coverage * coverage_weight) + (avg_quality * quality_weight)
    # Clamp to [0, 1] in case data_quality values are out of expected range
    return max(0.0, min(1.0, data_reliability))


def compute_indicator(
    indicator_name: str,
    indicator_config: IndicatorConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    context_adjusted_weights: AdjustedIndicatorWeights,
    coverage_weight: float = 0.6,
    quality_weight: float = 0.4,
) -> IndicatorScore:
    """Compute indicator score using FASL formula with context weights.

    FASL Formula:
        L_k(t) = SUM [ w_tilde_{k,i} * max(0, s_{k,i} * mu_{k,i}) ] / SUM w_tilde_{k,i}

    Where:
        L_k(t)      = daily likelihood for indicator k at time t in [0,1]
        mu_{k,i}    = membership value for biomarker i (from Story 4.5) in [0,1]
        s_{k,i}     = direction sign: +1 (pro/higher_is_worse) or -1 (contra/lower_is_worse)
        w_tilde_{k,i} = context-adjusted weight from Story 4.7

    Args:
        indicator_name: Name of the indicator being computed
        indicator_config: Configuration for this indicator (biomarker mappings)
        biomarker_memberships: Membership values from Story 4.5
        context_adjusted_weights: Adjusted weights from Story 4.7
        coverage_weight: Weight for biomarker coverage in reliability calculation (default: 0.6)
        quality_weight: Weight for data quality in reliability calculation (default: 0.4)

    Returns:
        IndicatorScore with daily likelihood and breakdown
    """
    numerator = 0.0
    denominator = 0.0
    contributions: dict[str, BiomarkerContribution] = {}
    biomarkers_used: list[str] = []
    biomarkers_missing: list[str] = []
    biomarkers_expected = list(indicator_config.biomarkers.keys())

    for biomarker_name, bio_config in indicator_config.biomarkers.items():
        # Get membership value (mu) from Story 4.5
        membership_result = biomarker_memberships.get(biomarker_name)

        if membership_result is None or membership_result.membership is None:
            biomarkers_missing.append(biomarker_name)
            _logger.debug(
                "Biomarker '%s' missing for indicator '%s'",
                biomarker_name,
                indicator_name,
            )
            continue  # Biomarker dropout handling (FR-013)

        biomarkers_used.append(biomarker_name)
        mu = membership_result.membership  # in [0,1]

        # Determine direction label for transparency
        direction_str: Literal["pro", "contra"] = (
            "pro" if bio_config.direction == "higher_is_worse" else "contra"
        )

        # Get context-adjusted weight (from Story 4.7)
        w_adjusted = context_adjusted_weights.biomarker_weights.get(biomarker_name)
        if w_adjusted is None:
            _logger.warning(
                "Biomarker '%s' weight missing from context_adjusted_weights for "
                "indicator '%s', defaulting to 0.0",
                biomarker_name,
                indicator_name,
            )
            w_adjusted = 0.0

        # Compute directed contribution based on direction
        # - higher_is_worse: high mu (deviation above baseline) = concerning
        # - lower_is_worse: low mu (deviation below baseline) = concerning, so invert
        if bio_config.direction == "higher_is_worse":
            directed_mu = mu
        else:  # lower_is_worse
            directed_mu = 1.0 - mu

        contribution = w_adjusted * directed_mu

        numerator += contribution
        denominator += w_adjusted

        # Record contribution details
        contributions[biomarker_name] = BiomarkerContribution(
            name=biomarker_name,
            membership=mu,
            direction=direction_str,
            base_weight=bio_config.weight,
            context_multiplier=context_adjusted_weights.multipliers_applied.get(
                biomarker_name, 1.0
            ),
            adjusted_weight=w_adjusted,
            contribution=contribution,
        )

    # Check minimum biomarker requirement
    min_biomarkers = indicator_config.min_biomarkers
    if len(biomarkers_used) < min_biomarkers:
        _logger.warning(
            "Indicator '%s' has insufficient biomarkers: %d < %d required",
            indicator_name,
            len(biomarkers_used),
            min_biomarkers,
        )

    # Final likelihood (normalized)
    if denominator > 0:
        daily_likelihood = numerator / denominator
    else:
        daily_likelihood = 0.0
        _logger.warning(
            "No biomarkers available for indicator '%s' (denominator=0)",
            indicator_name,
        )

    # Calculate data reliability score using configured weights
    data_reliability_score = _calculate_data_reliability(
        biomarkers_used,
        biomarkers_expected,
        biomarker_memberships,
        coverage_weight=coverage_weight,
        quality_weight=quality_weight,
    )

    # Log computation summary
    _logger.info(
        "Indicator '%s': L_k=%.3f, used=%d/%d biomarkers, data_reliability=%.2f, context=%s",
        indicator_name,
        daily_likelihood,
        len(biomarkers_used),
        len(biomarkers_expected),
        data_reliability_score,
        context_adjusted_weights.context_applied,
    )

    # Debug-level per-biomarker detail
    for name, contrib in contributions.items():
        _logger.debug(
            "  %s: mu=%.3f, s=%s, w=%.3f, contribution=%.3f",
            name,
            contrib.membership,
            contrib.direction,
            contrib.adjusted_weight,
            contrib.contribution,
        )

    return IndicatorScore(
        indicator_name=indicator_name,
        daily_likelihood=daily_likelihood,
        contributions=contributions,
        biomarkers_used=tuple(biomarkers_used),
        biomarkers_missing=tuple(biomarkers_missing),
        data_reliability_score=data_reliability_score,
        context_applied=context_adjusted_weights.context_applied,
        context_confidence=context_adjusted_weights.confidence,
        weights_before_context=context_adjusted_weights.base_weights.copy(),
        weights_after_context=context_adjusted_weights.biomarker_weights.copy(),
        timestamp=context_adjusted_weights.timestamp,
    )


def compute_all_indicators(
    config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    context_result: ContextResult,
) -> dict[str, IndicatorScore]:
    """Compute scores for all indicators in the configuration.

    Iterates through all configured indicators, adjusts weights based on context,
    and computes indicator scores using the FASL formula.

    Args:
        config: Analysis configuration with indicator definitions
        biomarker_memberships: Membership values from Story 4.5
        context_result: Result from ContextEvaluator (active context + confidence)

    Returns:
        Dict mapping indicator_name -> IndicatorScore
    """
    results: dict[str, IndicatorScore] = {}

    # Get reliability weights from config
    coverage_weight = config.reliability.coverage_weight
    quality_weight = config.reliability.quality_weight

    for indicator_name, indicator_config in config.indicators.items():
        # Get context-adjusted weights (from Story 4.7)
        adjusted_weights = adjust_biomarker_weights(
            indicator_name=indicator_name,
            indicator_config=indicator_config,
            context_result=context_result,
            config=config,
        )

        # Compute indicator score with configured reliability weights
        indicator_score = compute_indicator(
            indicator_name=indicator_name,
            indicator_config=indicator_config,
            biomarker_memberships=biomarker_memberships,
            context_adjusted_weights=adjusted_weights,
            coverage_weight=coverage_weight,
            quality_weight=quality_weight,
        )

        results[indicator_name] = indicator_score

    _logger.info(
        "Computed %d indicator scores: %s",
        len(results),
        list(results.keys()),
    )

    return results


class IndicatorComputer:
    """Stateless class for computing indicator scores.

    Provides a clean interface for computing individual or all indicator scores
    based on configuration, biomarker memberships, and context.

    Attributes:
        config: Analysis configuration containing indicators and context weights
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the indicator computer.

        Args:
            config: Analysis configuration with indicators and context_weights
        """
        self.config = config
        self._logger = logging.getLogger(__name__)

    def compute_single(
        self,
        indicator_name: str,
        biomarker_memberships: dict[str, BiomarkerMembership],
        context_result: ContextResult,
    ) -> IndicatorScore:
        """Compute score for a single indicator.

        Args:
            indicator_name: Name of the indicator to compute
            biomarker_memberships: Membership values from Story 4.5
            context_result: Result from ContextEvaluator

        Returns:
            IndicatorScore for the specified indicator

        Raises:
            KeyError: If indicator_name not found in configuration
        """
        if indicator_name not in self.config.indicators:
            raise KeyError(f"Indicator '{indicator_name}' not found in configuration")

        indicator_config = self.config.indicators[indicator_name]

        # Get context-adjusted weights
        adjusted_weights = adjust_biomarker_weights(
            indicator_name=indicator_name,
            indicator_config=indicator_config,
            context_result=context_result,
            config=self.config,
        )

        # Compute and return with configured reliability weights
        return compute_indicator(
            indicator_name=indicator_name,
            indicator_config=indicator_config,
            biomarker_memberships=biomarker_memberships,
            context_adjusted_weights=adjusted_weights,
            coverage_weight=self.config.reliability.coverage_weight,
            quality_weight=self.config.reliability.quality_weight,
        )

    def compute_all(
        self,
        biomarker_memberships: dict[str, BiomarkerMembership],
        context_result: ContextResult,
    ) -> dict[str, IndicatorScore]:
        """Compute scores for all configured indicators.

        Args:
            biomarker_memberships: Membership values from Story 4.5
            context_result: Result from ContextEvaluator

        Returns:
            Dict mapping indicator_name -> IndicatorScore
        """
        return compute_all_indicators(
            config=self.config,
            biomarker_memberships=biomarker_memberships,
            context_result=context_result,
        )

    def compute_daily_series(
        self,
        daily_biomarker_memberships: dict[str, list[DailyBiomarkerMembership]],
        context_result: ContextResult,
    ) -> dict[str, list[DailyIndicatorScore]]:
        """Compute daily indicator scores for each day in the analysis window.

        Applies the FASL formula to each day's memberships independently,
        while keeping context weights constant across all days.

        Per thesis Algorithm 5.4:
        - Context is evaluated ONCE at start of analysis
        - For each day d: compute L_k(d) using that day's memberships
        - Return time series for DSM-Gate evaluation

        Args:
            daily_biomarker_memberships: Dict mapping biomarker_name to list of
                DailyBiomarkerMembership (one per day with data)
            context_result: Result from ContextEvaluator (evaluated once,
                applied to all days)

        Returns:
            Dict mapping indicator_name to list of DailyIndicatorScore,
            ordered by date. The list contains one score per day that had
            any biomarker data.

        """
        # Collect all unique dates across all biomarkers
        all_dates: set[date] = set()
        for memberships in daily_biomarker_memberships.values():
            for m in memberships:
                all_dates.add(m.date)

        if not all_dates:
            self._logger.warning("No daily memberships provided")
            return {}

        sorted_dates = sorted(all_dates)
        self._logger.info(
            f"Computing daily series for {len(sorted_dates)} days "
            f"[{sorted_dates[0]} to {sorted_dates[-1]}]"
        )

        # Pre-compute adjusted weights for all indicators (context applied once)
        adjusted_weights_by_indicator: dict[str, AdjustedIndicatorWeights] = {}
        for indicator_name, indicator_config in self.config.indicators.items():
            from src.core.context.weights import adjust_biomarker_weights

            adjusted_weights_by_indicator[indicator_name] = adjust_biomarker_weights(
                indicator_name=indicator_name,
                indicator_config=indicator_config,
                context_result=context_result,
                config=self.config,
            )

        # Get reliability weights from config
        coverage_weight = self.config.reliability.coverage_weight
        quality_weight = self.config.reliability.quality_weight

        # Initialize results
        results: dict[str, list[DailyIndicatorScore]] = defaultdict(list)

        # For each day, compute indicator scores
        for current_date in sorted_dates:
            # Build membership dict for this day (biomarker_name -> BiomarkerMembership)
            # We need to convert DailyBiomarkerMembership to BiomarkerMembership format
            day_memberships: dict[str, BiomarkerMembership] = {}
            for biomarker_name, memberships in daily_biomarker_memberships.items():
                # Find membership for this day
                for m in memberships:
                    if m.date == current_date:
                        # Convert to BiomarkerMembership (same fields except date)
                        day_memberships[biomarker_name] = BiomarkerMembership(
                            name=m.name,
                            membership=m.membership,
                            z_score=m.z_score,
                            raw_value=m.raw_value,
                            baseline=m.baseline,
                            data_points_used=m.data_points_used,
                            data_quality=m.data_quality,
                            membership_function_used=m.membership_function_used,
                            timestamp=m.timestamp,
                        )
                        break

            # Compute indicator scores for this day
            for indicator_name, indicator_config in self.config.indicators.items():
                adjusted_weights = adjusted_weights_by_indicator[indicator_name]
                biomarkers_expected = list(indicator_config.biomarkers.keys())

                # Compute using the existing compute_indicator function
                indicator_score = compute_indicator(
                    indicator_name=indicator_name,
                    indicator_config=indicator_config,
                    biomarker_memberships=day_memberships,
                    context_adjusted_weights=adjusted_weights,
                    coverage_weight=coverage_weight,
                    quality_weight=quality_weight,
                )

                daily_score = DailyIndicatorScore(
                    date=current_date,
                    indicator_name=indicator_name,
                    daily_likelihood=indicator_score.daily_likelihood,
                    biomarkers_used=indicator_score.biomarkers_used,
                    biomarkers_missing=indicator_score.biomarkers_missing,
                    data_reliability_score=indicator_score.data_reliability_score,
                )
                results[indicator_name].append(daily_score)

        # Sort results by date
        for indicator_name in results:
            results[indicator_name].sort(key=lambda s: s.date)

        self._logger.info(
            f"Computed daily series for {len(results)} indicators "
            f"across {len(sorted_dates)} days"
        )

        return dict(results)
