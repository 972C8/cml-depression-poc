"""Window membership computation with context weighting.

Story 6.3: Membership with Context Weighting (AC5, AC6, AC8)
Story 4.14: Baseline Configuration & Selection (AC6 - file-based baseline)

This module computes membership values for aggregated window data,
applying context-appropriate weighting based on the active context
during each window.
"""

import logging
import math
import uuid

from sqlalchemy.orm import Session

from src.core.baseline_config import BaselineFile
from src.core.config import AnalysisConfig
from src.core.context.strategies import ContextStrategy, get_window_context
from src.core.models.window_models import WindowAggregate, WindowMembership


class BaselineError(Exception):
    """Raised when baseline configuration is missing or incomplete."""

    pass

__all__ = [
    "BaselineError",
    "compute_window_memberships",
    "z_score_to_membership",
]

logger = logging.getLogger(__name__)


def z_score_to_membership(z_score: float) -> float:
    """Map z-score to [0, 1] membership using sigmoid function.

    AC6: Z-Score to Membership Mapping
    Uses sigmoid: membership = 1 / (1 + exp(-z_score))

    The sigmoid provides natural mapping:
    - z = 0 → 0.5 (baseline, neutral)
    - z = +1 → ~0.73 (slightly elevated)
    - z = +2 → ~0.88 (moderately elevated)
    - z = +3 → ~0.95 (significantly elevated)
    - z = -3 → ~0.05 (significantly depressed)

    Args:
        z_score: Normalized value ((value - mean) / std)

    Returns:
        Membership value in [0, 1]
    """
    return 1.0 / (1.0 + math.exp(-z_score))


def compute_window_memberships(
    window_aggregates: dict[str, list[WindowAggregate]],
    user_id: str,
    indicator_name: str,
    session: Session,
    config: AnalysisConfig,
    baseline_config: BaselineFile,
    strategy: ContextStrategy | None = None,
    context_evaluation_run_id: uuid.UUID | None = None,
) -> dict[str, list[WindowMembership]]:
    """Compute context-weighted membership values for window aggregates.

    AC5: Membership Computation Function
    For each window aggregate:
    1. Fetch baseline from the provided baseline_config
    2. Compute z-score: z = (value - mean) / std
    3. Map z-score to membership [0, 1] using sigmoid
    4. Get context for window using configured strategy
    5. Get weight for target indicator from context
    6. Compute weighted membership: membership × context_weight

    Story 4.14: Baseline is required - no fallback chain.
    If a biomarker is missing from the baseline_config, a BaselineError is raised.

    Story 6.14: When context_evaluation_run_id is provided, context queries
    are filtered to only use records from that specific run. Timestamps not
    covered by the run use neutral weights.

    Args:
        window_aggregates: Dict mapping biomarker names to lists of WindowAggregate
                          (output from Story 6.2 aggregate_into_windows)
        user_id: User identifier
        indicator_name: Target indicator for context weight lookup
        session: SQLAlchemy database session
        config: Analysis configuration
        baseline_config: Required file-based baseline configuration
        strategy: Context strategy to use (overrides config if provided)
        context_evaluation_run_id: Optional UUID of a specific context evaluation
            run to use for context queries. When provided, only context records
            from that run are used.

    Returns:
        Dict mapping biomarker names to lists of WindowMembership

    Raises:
        BaselineError: If baseline_config is missing a required biomarker
    """
    # Get strategy from config if not provided
    if strategy is None:
        strategy = _get_configured_strategy(config)

    # Process each biomarker
    result: dict[str, list[WindowMembership]] = {}

    for biomarker_name, aggregates in window_aggregates.items():
        memberships = []

        # Get baseline stats from file-based config (required)
        mean, std = _get_baseline_stats(biomarker_name, baseline_config, config)

        logger.debug(
            f"Baseline for {biomarker_name}: mean={mean:.4f}, std={std:.4f}"
        )

        for aggregate in aggregates:
            membership = _compute_single_window_membership(
                aggregate=aggregate,
                user_id=user_id,
                indicator_name=indicator_name,
                mean=mean,
                std=std,
                strategy=strategy,
                session=session,
                config=config,
                context_evaluation_run_id=context_evaluation_run_id,
            )
            memberships.append(membership)

        result[biomarker_name] = memberships
        logger.debug(f"Computed {len(memberships)} memberships for {biomarker_name}")

    return result


def _compute_single_window_membership(
    aggregate: WindowAggregate,
    user_id: str,
    indicator_name: str,
    mean: float,
    std: float,
    strategy: ContextStrategy,
    session: Session,
    config: AnalysisConfig,
    context_evaluation_run_id: uuid.UUID | None = None,
) -> WindowMembership:
    """Compute membership for a single window aggregate.

    Args:
        aggregate: Window aggregate to process
        user_id: User identifier
        indicator_name: Target indicator for context weight lookup
        mean: Baseline mean for this biomarker
        std: Baseline standard deviation for this biomarker
        strategy: Context strategy to use
        session: Database session
        config: Analysis configuration
        context_evaluation_run_id: Optional UUID to filter context queries

    Returns:
        WindowMembership with computed values
    """
    # Compute z-score (AC8: handle zero/negative std)
    if std < 0:
        logger.error(
            f"Negative std ({std}) for {aggregate.biomarker_name} indicates data corruption, using z_score=0"
        )
        z_score = 0.0
    elif std == 0:
        logger.warning(f"Zero std for {aggregate.biomarker_name}, using z_score=0")
        z_score = 0.0
    else:
        z_score = (aggregate.aggregated_value - mean) / std

    # Map z-score to membership using sigmoid (AC6)
    membership = z_score_to_membership(z_score)

    # Get context for this window using specified strategy
    context_result = get_window_context(
        user_id=user_id,
        window_start=aggregate.window_start,
        window_end=aggregate.window_end,
        indicator_name=indicator_name,
        session=session,
        config=config,
        strategy=strategy,
        readings_timestamps=aggregate.readings_timestamps,
        context_evaluation_run_id=context_evaluation_run_id,
    )

    # Story 6.10: Context weight is NOT applied to membership here.
    # Context weights are applied in FASL formula (window_fasl.py) to avoid
    # counterintuitive behavior for lower_is_worse biomarkers.
    # weighted_membership is kept for backward compatibility but equals membership.
    weighted_membership = membership

    return WindowMembership(
        biomarker_name=aggregate.biomarker_name,
        window_start=aggregate.window_start,
        window_end=aggregate.window_end,
        aggregated_value=aggregate.aggregated_value,
        z_score=z_score,
        membership=membership,
        context_strategy=strategy,
        context_state=context_result.context_state,
        dominant_context=context_result.dominant_context,
        context_weight=context_result.context_weight,
        context_confidence=context_result.confidence,
        weighted_membership=weighted_membership,
        readings_count=aggregate.readings_count,
    )


def _get_baseline_stats(
    biomarker_name: str,
    baseline_config: BaselineFile,
    config: AnalysisConfig,
) -> tuple[float, float]:
    """Get baseline mean and std for a biomarker from the baseline file.

    Story 4.14: Baseline is required - no fallback chain.
    Story 6.3 AC8: Edge Case Handling for std validation.

    Args:
        biomarker_name: Name of the biomarker
        baseline_config: File-based baseline config (required)
        config: Analysis configuration with min_std setting

    Returns:
        Tuple of (mean, std)

    Raises:
        BaselineError: If biomarker is not found in baseline_config
    """
    if biomarker_name not in baseline_config.baselines:
        raise BaselineError(
            f"Baseline missing for biomarker '{biomarker_name}'. "
            f"The selected baseline file must include all biomarkers used in analysis. "
            f"Available biomarkers in baseline: {list(baseline_config.baselines.keys())}"
        )

    file_baseline = baseline_config.baselines[biomarker_name]
    mean = file_baseline.mean
    std = file_baseline.std

    # Apply minimum std floor to prevent division by zero
    min_std = config.biomarker_processing.min_std_deviation
    if std < min_std:
        logger.warning(
            f"Baseline std {std} for {biomarker_name} below minimum, "
            f"using min_std={min_std}"
        )
        std = min_std

    logger.info(f"Using baseline for {biomarker_name}: mean={mean}, std={std}")
    return mean, std


def _get_configured_strategy(config: AnalysisConfig) -> ContextStrategy:
    """Get the configured context strategy.

    AC7: Get strategy from config.context.strategy setting.

    Args:
        config: Analysis configuration

    Returns:
        Context strategy name (default: 'dominant')
    """
    return config.context.strategy
