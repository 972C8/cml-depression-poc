"""EMA (Exponential Moving Average) smoother for context evaluation.

This module provides smoothing with hysteresis and dwell time
to prevent rapid context oscillations.
"""

import logging
from dataclasses import dataclass

__all__ = ["EMASmoother", "SwitchDecision"]


@dataclass(frozen=True)
class SwitchDecision:
    """Result of context switch decision with transparency info.

    Attributes:
        should_switch: Whether the context switch is allowed
        blocked_reason: Why switch was blocked ("hysteresis" | "dwell_time" | None)
        score_difference: Difference between candidate and current scores
        dwell_progress: Current dwell counter for candidate context
        dwell_required: Number of consecutive readings required
        candidate_context: Context that would win without stabilization
        current_context: Currently active context
    """

    should_switch: bool
    blocked_reason: str | None = None
    score_difference: float = 0.0
    dwell_progress: int = 0
    dwell_required: int = 0
    candidate_context: str | None = None
    current_context: str | None = None

logger = logging.getLogger(__name__)


class EMASmoother:
    """Exponential Moving Average smoother with hysteresis and dwell time.

    Provides stable context transitions by:
    - Smoothing raw values with EMA
    - Requiring hysteresis threshold to switch contexts
    - Requiring dwell time (consecutive readings) before switching

    Attributes:
        alpha: Smoothing factor (0 < alpha <= 1). Higher = less smoothing.
        hysteresis: Minimum difference required to switch contexts.
        dwell_time: Number of consecutive readings above threshold before switch.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        hysteresis: float = 0.1,
        dwell_time: int = 2,
    ) -> None:
        """Initialize EMA smoother.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
            hysteresis: Minimum difference to switch contexts
            dwell_time: Consecutive readings required before switch
        """
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if hysteresis < 0:
            raise ValueError(f"hysteresis must be >= 0, got {hysteresis}")
        if dwell_time < 0:
            raise ValueError(f"dwell_time must be >= 0, got {dwell_time}")

        self.alpha = alpha
        self.hysteresis = hysteresis
        self.dwell_time = dwell_time

        # State storage
        self._previous_values: dict[str, float] = {}
        self._dwell_counters: dict[str, int] = {}
        self._active_context: str | None = None
        self._logger = logging.getLogger(__name__)

    def smooth(self, context_name: str, raw_value: float) -> float:
        """Apply EMA smoothing to a raw context confidence value.

        Args:
            context_name: Name of the context being smoothed
            raw_value: Raw confidence score (0-1)

        Returns:
            Smoothed confidence value
        """
        if context_name not in self._previous_values:
            # First value - no previous, return as-is
            self._previous_values[context_name] = raw_value
            return raw_value

        previous = self._previous_values[context_name]
        smoothed = self.alpha * raw_value + (1 - self.alpha) * previous
        self._previous_values[context_name] = smoothed
        return smoothed

    def should_switch_context(
        self,
        candidate_context: str,
        candidate_score: float,
        current_context: str | None,
        current_score: float,
    ) -> SwitchDecision:
        """Determine if context should switch based on hysteresis and dwell time.

        Args:
            candidate_context: Context with highest current score
            candidate_score: Score of candidate context
            current_context: Currently active context (None if first evaluation)
            current_score: Score of current context

        Returns:
            SwitchDecision with should_switch and transparency info
        """
        difference = candidate_score - current_score

        # First evaluation - always accept
        if current_context is None:
            self._active_context = candidate_context
            self._dwell_counters[candidate_context] = self.dwell_time
            return SwitchDecision(
                should_switch=True,
                blocked_reason=None,
                score_difference=difference,
                dwell_progress=self.dwell_time,
                dwell_required=self.dwell_time,
                candidate_context=candidate_context,
                current_context=current_context,
            )

        # Same context - no switch needed
        if candidate_context == current_context:
            self._dwell_counters[candidate_context] = self.dwell_time
            return SwitchDecision(
                should_switch=False,
                blocked_reason=None,  # Not blocked, just no switch needed
                score_difference=0.0,
                dwell_progress=self.dwell_time,
                dwell_required=self.dwell_time,
                candidate_context=candidate_context,
                current_context=current_context,
            )

        # Check hysteresis - difference must exceed threshold
        if difference <= self.hysteresis:
            # Not enough difference, reset candidate's dwell counter
            self._dwell_counters[candidate_context] = 0
            return SwitchDecision(
                should_switch=False,
                blocked_reason="hysteresis",
                score_difference=difference,
                dwell_progress=0,
                dwell_required=self.dwell_time,
                candidate_context=candidate_context,
                current_context=current_context,
            )

        # Check dwell time - need consecutive readings above threshold
        if candidate_context not in self._dwell_counters:
            self._dwell_counters[candidate_context] = 0

        self._dwell_counters[candidate_context] += 1
        current_dwell = self._dwell_counters[candidate_context]

        if current_dwell >= self.dwell_time:
            # Dwell time satisfied - allow switch
            self._active_context = candidate_context
            self._logger.debug(
                f"Context switch: {current_context} -> {candidate_context} "
                f"(diff={difference:.3f}, dwell={current_dwell})"
            )
            return SwitchDecision(
                should_switch=True,
                blocked_reason=None,
                score_difference=difference,
                dwell_progress=current_dwell,
                dwell_required=self.dwell_time,
                candidate_context=candidate_context,
                current_context=current_context,
            )

        # Not enough consecutive readings yet
        return SwitchDecision(
            should_switch=False,
            blocked_reason="dwell_time",
            score_difference=difference,
            dwell_progress=current_dwell,
            dwell_required=self.dwell_time,
            candidate_context=candidate_context,
            current_context=current_context,
        )

    def get_active_context(self) -> str | None:
        """Get the currently active context.

        Returns:
            Active context name or None if not set
        """
        return self._active_context

    def reset(self) -> None:
        """Reset all smoother state.

        Clears previous values, dwell counters, and active context.
        """
        self._previous_values.clear()
        self._dwell_counters.clear()
        self._active_context = None
        self._logger.debug("EMASmoother state reset")
