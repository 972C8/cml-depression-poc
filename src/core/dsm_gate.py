"""DSM-Gate Logic & Episode Decision Module.

This module implements DSM-5 based gate logic for determining episode presence
for Major Depressive Disorder (MDD) detection.

The DSM-5 defines a Major Depressive Episode as:
    "Five (or more) of the following symptoms have been present during the
    same 2-week period and represent a change from previous functioning;
    at least one of the symptoms is either (1) depressed mood or
    (2) loss of interest or pleasure."

The gate operates in two stages:

Stage 1 — Per-indicator presence (N-of-M rule):
    I_k(d) = 1 if L_k(d) >= theta else 0
    present(k) = true if SUM(I_k(d), d=t-M+1..t) >= N

Stage 2 — Episode decision (aggregation):
    episode(t) = 1 if SUM(present_k) >= 5  AND  EXISTS c in C : present(c)
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

from src.core.config import AnalysisConfig, DSMGateConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DSMGate",
    "EpisodeDecision",
    "IndicatorGateResult",
    "apply_indicator_gate",
    "compute_episode_decision",
]


@dataclass(frozen=True)
class IndicatorGateResult:
    """Result of applying DSM-gate to a single indicator.

    Stage 1 of the DSM-5 Gate converts the continuous daily likelihood
    into a binary presence flag using the N-of-M rule: the indicator is
    considered present if at least gate_need (N) of the last m_window (M)
    days have a likelihood >= theta.

    Attributes:
        indicator_name: Name of the indicator being evaluated
        presence_flag: True if the indicator meets the N-of-M criterion
            (days_above_threshold >= required_days) and has sufficient data
            (>= m_window days). With insufficient data, always False.
        days_above_threshold: Count of days with likelihood >= theta
        days_evaluated: Number of days actually evaluated (may be < window_size)
        window_size: Size of the evaluation window (M)
        required_days: Required positive days for presence (N / gate_need)
        threshold: Likelihood threshold for daily evaluation (theta)
        daily_flags: Per-day threshold evaluation results for the window
        insufficient_data: True if less than M days available
    """

    indicator_name: str
    presence_flag: bool
    days_above_threshold: int
    days_evaluated: int
    window_size: int
    required_days: int
    threshold: float
    daily_flags: tuple[bool, ...]
    insufficient_data: bool


@dataclass(frozen=True)
class EpisodeDecision:
    """Episode-level decision combining multiple indicator gate results.

    Stage 2 of the DSM-5 Gate aggregates the per-indicator presence flags
    into a single episode assessment. An episode is flagged when at least
    min_indicators (default 5) of the nine indicators are present and at
    least one of them belongs to the set of core indicators (depressed
    mood or loss of interest).

    Attributes:
        episode_likely: True if episode criteria are met
        indicators_present: Count of indicators with presence_flag=True
        core_indicator_present: True if any core indicator has presence_flag=True
        core_indicators_present: Names of core indicators that are present
        all_indicator_results: Complete gate results for all indicators (immutable)
        decision_rationale: Human-readable explanation of the decision
        min_indicators_required: Configured threshold (default 5)
        timestamp: Decision timestamp for audit trail
    """

    episode_likely: bool
    indicators_present: int
    core_indicator_present: bool
    core_indicators_present: tuple[str, ...]
    all_indicator_results: Mapping[str, IndicatorGateResult]
    decision_rationale: str
    min_indicators_required: int
    timestamp: datetime


def _get_gate_config(indicator_name: str, config: AnalysisConfig) -> DSMGateConfig:
    """Get gate config for indicator, checking for per-indicator theta override.

    Lookup order:
    1. Check if indicator has inline dsm_gate override in IndicatorConfig
       -> Use its theta, but always use global m_window and gate_need
    2. Fall back to global dsm_gate_defaults

    Args:
        indicator_name: Name of the indicator
        config: Analysis configuration

    Returns:
        DSMGateConfig with appropriate theta and global m_window / gate_need
    """
    global_defaults = config.dsm_gate_defaults
    indicator_config = config.indicators.get(indicator_name)

    if indicator_config and indicator_config.dsm_gate:
        # Only take theta from per-indicator override; m_window and gate_need are global
        return DSMGateConfig(
            theta=indicator_config.dsm_gate.theta,
            m_window=global_defaults.m_window,
            gate_need=global_defaults.gate_need,
        )

    return global_defaults


def apply_indicator_gate(
    indicator_name: str,
    daily_likelihoods: Sequence[float],
    config: AnalysisConfig,
) -> IndicatorGateResult:
    """Apply Stage 1 of the DSM-5 Gate to a single indicator.

    Converts daily likelihoods into a binary presence flag using the
    N-of-M rule: the indicator is present if at least gate_need (N) of
    the last m_window (M) days have L_k(d) >= theta.

    Args:
        indicator_name: Name of the indicator
        daily_likelihoods: Sequence of daily likelihood values L_k(d).
            Values should be in [0, 1] range. NaN/Infinity values are
            treated as invalid and logged as warnings (treated as 0.0).
        config: Analysis configuration with gate parameters

    Returns:
        IndicatorGateResult with daily flags and presence determination

    Note:
        Edge case handling:
        - Empty input: Returns insufficient_data=True, presence_flag=False
        - NaN values: Treated as 0.0 (below any threshold), logged as warning
        - Infinity values: Treated as 0.0 (invalid), logged as warning
        - Negative values: Treated as below threshold (valid comparison)
        - Values > 1.0: Treated as above threshold (valid comparison)
    """
    gate_config = _get_gate_config(indicator_name, config)
    theta = gate_config.theta
    m_window = gate_config.m_window
    gate_need = gate_config.gate_need

    # Handle empty input
    if not daily_likelihoods:
        logger.warning(
            f"No daily likelihoods provided for indicator '{indicator_name}'"
        )
        return IndicatorGateResult(
            indicator_name=indicator_name,
            presence_flag=False,
            days_above_threshold=0,
            days_evaluated=0,
            window_size=m_window,
            required_days=gate_need,
            threshold=theta,
            daily_flags=(),
            insufficient_data=True,
        )

    # Handle partial data - fewer than M days available
    if len(daily_likelihoods) < m_window:
        partial_window = list(daily_likelihoods)

        # Validate partial data for NaN/Infinity
        for i, val in enumerate(partial_window):
            if math.isnan(val) or math.isinf(val):
                partial_window[i] = 0.0

        partial_flags = tuple(likelihood >= theta for likelihood in partial_window)
        partial_days_above = sum(partial_flags)

        logger.debug(
            f"Partial data for indicator '{indicator_name}': "
            f"{len(daily_likelihoods)} days < {m_window} window, "
            f"{partial_days_above} days above threshold"
        )
        return IndicatorGateResult(
            indicator_name=indicator_name,
            presence_flag=False,
            days_above_threshold=partial_days_above,
            days_evaluated=len(daily_likelihoods),
            window_size=m_window,
            required_days=gate_need,
            threshold=theta,
            daily_flags=partial_flags,
            insufficient_data=True,
        )

    # Extract last M days
    window = list(daily_likelihoods[-m_window:])

    # Validate and sanitize values (NaN/Infinity -> 0.0 with warning)
    invalid_count = 0
    for i, val in enumerate(window):
        if math.isnan(val) or math.isinf(val):
            invalid_count += 1
            window[i] = 0.0  # Treat as below threshold
    if invalid_count > 0:
        logger.warning(
            f"Indicator '{indicator_name}': {invalid_count} NaN/Infinity values "
            f"in window, treated as 0.0"
        )

    # Compute daily flags: I_k(d) = 1 if L_k(d) >= theta else 0
    daily_flags = tuple(likelihood >= theta for likelihood in window)

    # Count days above threshold
    days_above = sum(daily_flags)

    # N-of-M presence rule: present(k) = true if days_above >= gate_need
    presence = days_above >= gate_need

    logger.info(
        f"Gate '{indicator_name}': days_above={days_above}/{m_window}, "
        f"gate_need={gate_need}, present={presence}, theta={theta}"
    )

    return IndicatorGateResult(
        indicator_name=indicator_name,
        presence_flag=presence,
        days_above_threshold=days_above,
        days_evaluated=m_window,
        window_size=m_window,
        required_days=gate_need,
        threshold=theta,
        daily_flags=daily_flags,
        insufficient_data=False,
    )


def _build_rationale(
    episode_likely: bool,
    indicators_present: int,
    min_indicators: int,
    core_present: list[str],
    core_indicators: tuple[str, ...],
) -> str:
    """Build human-readable decision rationale.

    Args:
        episode_likely: Whether episode criteria are met
        indicators_present: Count of present indicators
        min_indicators: Required minimum indicators
        core_present: List of core indicators that are present
        core_indicators: All core indicators defined in config

    Returns:
        Human-readable rationale string
    """
    parts = []

    # Indicator count check
    if indicators_present >= min_indicators:
        parts.append(
            f"{indicators_present} of 9 indicators present "
            f"(>={min_indicators} required: MET)"
        )
    else:
        parts.append(
            f"{indicators_present} of 9 indicators present "
            f"(>={min_indicators} required: NOT MET)"
        )

    # Core indicator check
    if core_present:
        parts.append(
            f"Core indicator(s) present: {', '.join(core_present)}"
        )
    else:
        parts.append(
            f"No core indicator present "
            f"(need at least one of: {', '.join(core_indicators)})"
        )

    # Final decision
    if episode_likely:
        parts.append("=> Episode criteria MET")
    else:
        parts.append("=> Episode criteria NOT MET")

    return "; ".join(parts)


def compute_episode_decision(
    indicator_gate_results: dict[str, IndicatorGateResult],
    config: AnalysisConfig,
) -> EpisodeDecision:
    """Apply Stage 2 of the DSM-5 Gate: episode-level aggregation.

    Aggregates per-indicator presence flags into a single episode
    assessment. An episode is indicated when:
        episode(t) = 1  if  SUM(present_k) >= min_indicators
                         AND  EXISTS c in core_indicators : present(c)

    Args:
        indicator_gate_results: Gate results per indicator (from Stage 1)
        config: Analysis configuration with episode parameters

    Returns:
        EpisodeDecision with overall assessment
    """
    min_indicators = config.episode.min_indicators
    core_indicators = config.episode.core_indicators

    # Count present indicators (those that passed N-of-M in Stage 1)
    present_indicators = [
        name for name, result in indicator_gate_results.items()
        if result.presence_flag
    ]
    indicators_present = len(present_indicators)

    # Check whether at least one core indicator is present
    core_present = [
        name for name in present_indicators
        if name in core_indicators
    ]

    # Episode decision: >= min_indicators present AND at least one core present
    episode_likely = (
        indicators_present >= min_indicators and len(core_present) > 0
    )

    # Build rationale
    rationale = _build_rationale(
        episode_likely,
        indicators_present,
        min_indicators,
        core_present,
        core_indicators,
    )

    logger.info(
        f"Episode decision: likely={episode_likely}, "
        f"indicators={indicators_present}/{min_indicators}, "
        f"core_present={core_present}"
    )
    logger.debug(f"Episode rationale: {rationale}")

    # Create immutable mapping for all_indicator_results
    immutable_results = MappingProxyType(dict(indicator_gate_results))

    return EpisodeDecision(
        episode_likely=episode_likely,
        indicators_present=indicators_present,
        core_indicator_present=len(core_present) > 0,
        core_indicators_present=tuple(core_present),
        all_indicator_results=immutable_results,
        decision_rationale=rationale,
        min_indicators_required=min_indicators,
        timestamp=datetime.now(UTC),
    )


class DSMGate:
    """DSM-gate processor for stateless gate operations.

    Provides a class-based interface for applying DSM-gate logic to indicators
    and computing episode-level decisions. Ensures deterministic computation
    (same inputs always produce same outputs).

    Example:
        config = load_config("config/analysis.yaml")
        gate = DSMGate(config)

        # Apply gate to single indicator
        result = gate.apply_gate("social_withdrawal", daily_scores)

        # Apply to all indicators
        all_results = gate.apply_all_gates(indicator_scores_by_day)

        # Compute episode decision
        episode = gate.compute_episode(all_results)
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize DSMGate with analysis configuration.

        Args:
            config: Analysis configuration containing gate parameters
        """
        self._config = config
        self._logger = logging.getLogger(__name__)

    def apply_gate(
        self,
        indicator_name: str,
        daily_likelihoods: list[float],
    ) -> IndicatorGateResult:
        """Apply DSM-gate to compute daily flags and presence for an indicator.

        Args:
            indicator_name: Name of the indicator
            daily_likelihoods: List of daily likelihood values

        Returns:
            IndicatorGateResult with daily flags and presence determination
        """
        return apply_indicator_gate(indicator_name, daily_likelihoods, self._config)

    def apply_all_gates(
        self,
        indicator_scores_by_day: dict[str, list[float]],
    ) -> dict[str, IndicatorGateResult]:
        """Apply DSM-gate to all indicators.

        Args:
            indicator_scores_by_day: Dict mapping indicator names to their
                daily likelihood scores

        Returns:
            Dict mapping indicator names to their gate results
        """
        results = {}
        for indicator_name, daily_scores in indicator_scores_by_day.items():
            results[indicator_name] = self.apply_gate(indicator_name, daily_scores)
        return results

    def compute_episode(
        self,
        indicator_gate_results: dict[str, IndicatorGateResult],
    ) -> EpisodeDecision:
        """Compute episode-level decision from indicator gates.

        Args:
            indicator_gate_results: Gate results per indicator

        Returns:
            EpisodeDecision with overall assessment
        """
        return compute_episode_decision(indicator_gate_results, self._config)
