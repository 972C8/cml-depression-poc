"""Context-aware biomarker weight adjustment.

This module provides weight adjustment logic that modifies indicator biomarker
weights based on detected context (e.g., social vs solitary_digital). Weights
are adjusted using confidence-blended multipliers and always normalized to sum
to 1.0.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from src.core.config import AnalysisConfig, IndicatorConfig
from src.core.context.evaluator import ContextResult

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdjustedIndicatorWeights:
    """Result of context-aware weight adjustment for a single indicator.

    Contains both the adjusted biomarker weights and metadata about the
    adjustment process for transparency and debugging.

    Attributes:
        indicator_name: Name of the indicator (e.g., "social_withdrawal")
        biomarker_weights: Normalized adjusted weights (biomarker -> weight, sum=1.0)
        base_weights: Original weights before adjustment (biomarker -> weight)
        multipliers_applied: Effective multipliers applied to each biomarker
        context_applied: Name of the context used for adjustment
        confidence: Confidence score of the active context (0-1)
        timestamp: When the adjustment was performed
    """

    indicator_name: str
    biomarker_weights: dict[str, float]
    base_weights: dict[str, float]
    multipliers_applied: dict[str, float]
    context_applied: str
    confidence: float
    timestamp: datetime


def adjust_biomarker_weights(
    indicator_name: str,
    indicator_config: IndicatorConfig,
    context_result: ContextResult,
    config: AnalysisConfig,
) -> AdjustedIndicatorWeights:
    """Adjust biomarker weights based on detected context.

    Applies context-specific multipliers to base biomarker weights, blended
    with context confidence. Always normalizes result to sum to 1.0.

    Algorithm:
        1. Extract base weights from indicator_config
        2. Get context multipliers from config (fallback to 1.0 if missing)
        3. Blend multipliers with confidence: effective = 1.0 + (mult - 1.0) * conf
        4. Apply to base weights: raw_adjusted = base * effective_multiplier
        5. Normalize to sum to 1.0

    Args:
        indicator_name: Name of the indicator being computed
        indicator_config: Configuration for this indicator (base weights)
        context_result: Result from ContextEvaluator (active context + confidence)
        config: Analysis configuration (contains context_weights mapping)

    Returns:
        AdjustedIndicatorWeights with normalized adjusted weights

    Example:
        >>> config = AnalysisConfig(...)
        >>> indicator_config = config.indicators["social_withdrawal"]
        >>> context_result = ContextResult(active_context="solitary_digital", confidence_scores={"solitary_digital": 0.85}, ...)
        >>> adjusted = adjust_biomarker_weights("social_withdrawal", indicator_config, context_result, config)
        >>> sum(adjusted.biomarker_weights.values())  # Always 1.0
        1.0
    """
    context = context_result.active_context
    confidence = context_result.confidence_scores.get(context, 0.0)

    # Get context multipliers (fallback to empty dict if context unknown)
    context_multipliers = config.context_weights.get(context, {})

    # Warn if unknown context
    if context not in config.context_weights:
        _logger.warning(
            "Unknown context '%s', using neutral weights (multiplier=1.0)",
            context,
        )

    adjusted = {}
    base_weights = {}
    multipliers_applied = {}

    for biomarker_name, bio_config in indicator_config.biomarkers.items():
        base_weight = bio_config.weight
        base_weights[biomarker_name] = base_weight

        # Get multiplier (default: 1.0 = no change)
        multiplier = context_multipliers.get(biomarker_name, 1.0)

        # Blend based on context confidence (0.0 = no effect, 1.0 = full effect)
        # Formula: effective = 1.0 + (multiplier - 1.0) * confidence
        effective_multiplier = 1.0 + (multiplier - 1.0) * confidence
        multipliers_applied[biomarker_name] = effective_multiplier

        adjusted[biomarker_name] = base_weight * effective_multiplier

    # Normalize so weights sum to 1
    total = sum(adjusted.values())
    if total == 0:
        _logger.warning(
            "Total weight is zero for indicator '%s', using base weights",
            indicator_name,
        )
        normalized_weights = base_weights.copy()
    else:
        normalized_weights = {k: v / total for k, v in adjusted.items()}

    return AdjustedIndicatorWeights(
        indicator_name=indicator_name,
        biomarker_weights=normalized_weights,
        base_weights=base_weights,
        multipliers_applied=multipliers_applied,
        context_applied=context,
        confidence=confidence,
        timestamp=context_result.timestamp,
    )


class ContextWeightAdjuster:
    """Stateless adjuster for context-aware biomarker weight adjustment.

    Adjusts weights for all indicators in the analysis configuration based on
    the detected context. Provides logging for transparency and debugging.

    Attributes:
        config: Analysis configuration containing indicators and context weights
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the weight adjuster.

        Args:
            config: Analysis configuration with indicators and context_weights
        """
        self.config = config
        self._logger = logging.getLogger(__name__)

    def adjust_all_indicators(
        self, context_result: ContextResult
    ) -> dict[str, AdjustedIndicatorWeights]:
        """Adjust weights for all indicators based on context.

        Args:
            context_result: Result from ContextEvaluator with active context

        Returns:
            Dict mapping indicator names to their adjusted weights

        Example:
            >>> adjuster = ContextWeightAdjuster(config)
            >>> context_result = ContextResult(active_context="solitary_digital", ...)
            >>> adjusted_weights = adjuster.adjust_all_indicators(context_result)
            >>> adjusted_weights["social_withdrawal"].biomarker_weights
            {'speech_activity': 0.36, 'voice_energy': 0.21, ...}
        """
        adjusted_indicators = {}

        for indicator_name, indicator_config in self.config.indicators.items():
            adjusted = adjust_biomarker_weights(
                indicator_name=indicator_name,
                indicator_config=indicator_config,
                context_result=context_result,
                config=self.config,
            )
            adjusted_indicators[indicator_name] = adjusted

            # Count biomarkers with non-neutral multipliers (not equal to 1.0)
            adjusted_count = sum(
                1 for m in adjusted.multipliers_applied.values() if abs(m - 1.0) > 1e-6
            )

            # Log adjustment summary
            self._logger.info(
                "Weight adjustment for %s: context=%s (confidence=%.2f), "
                "adjusted=%d biomarkers",
                indicator_name,
                adjusted.context_applied,
                adjusted.confidence,
                adjusted_count,
            )

            # Debug-level detail
            self._logger.debug(
                "Weight details for %s: base=%s, multipliers=%s, adjusted=%s",
                indicator_name,
                adjusted.base_weights,
                adjusted.multipliers_applied,
                adjusted.biomarker_weights,
            )

        return adjusted_indicators
