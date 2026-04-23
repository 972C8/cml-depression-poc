"""Context evaluator for multimodal analysis engine.

This module provides the main ContextEvaluator class that uses fuzzy logic
and EMA smoothing to determine active context from context markers.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.config import (
    AnalysisConfig,
    EMAConfig,
    ExperimentContextEvalConfig,
    MembershipFunction,
)
from src.core.context.membership import LinguisticVariable, MembershipCalculator
from src.core.context.smoother import EMASmoother, SwitchDecision
from src.core.data_reader import ContextRecord

__all__ = [
    "ContextAssumption",
    "ContextAssumptionConfig",
    "ContextCondition",
    "ContextEvaluationConfig",
    "ContextEvaluator",
    "ContextResult",
    "MarkerMembershipConfig",
    "get_default_context_config",
    "load_context_config",
]

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes (Task 2)
# ============================================================================


@dataclass(frozen=True)
class ContextResult:
    """Result of context evaluation.

    Attributes:
        active_context: Name of the highest confidence context
        confidence_scores: Dict of context name to smoothed confidence (0-1)
        raw_scores: Dict of context name to pre-smoothing scores
        smoothed: Whether EMA smoothing was applied
        markers_used: Tuple of marker names that contributed to evaluation
        markers_missing: Tuple of marker names that were unavailable
        timestamp: Timestamp of the evaluation
        switch_blocked: Whether stabilization prevented a context switch
        switch_blocked_reason: Why switch was blocked ("hysteresis" | "dwell_time")
        candidate_context: Context that would win without stabilization
        score_difference: Score diff between candidate and current context
        dwell_progress: Current dwell count for candidate context
        dwell_required: Number of consecutive readings required to switch
    """

    active_context: str
    confidence_scores: dict[str, float]
    raw_scores: dict[str, float]
    smoothed: bool
    markers_used: tuple[str, ...]
    markers_missing: tuple[str, ...]
    timestamp: datetime
    # Stabilization transparency fields
    switch_blocked: bool = False
    switch_blocked_reason: str | None = None
    candidate_context: str | None = None
    score_difference: float | None = None
    dwell_progress: int | None = None
    dwell_required: int | None = None


# ============================================================================
# Configuration Models (Task 8)
# ============================================================================


class MarkerMembershipConfig(BaseModel):
    """Membership function configuration for a single marker.

    Allows defining any fuzzy sets (e.g., low, medium, high, very_low, etc.)
    dynamically without hardcoding field names.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    # Common presets for convenience (optional)
    low: MembershipFunction | None = None
    medium: MembershipFunction | None = None
    high: MembershipFunction | None = None
    quiet: MembershipFunction | None = None
    moderate: MembershipFunction | None = None
    loud: MembershipFunction | None = None

    def get_membership_fns(self) -> dict[str, MembershipFunction]:
        """Get all defined membership functions as a dict.

        Includes both preset fields and any dynamically added fields.
        """
        result = {}
        # Check preset fields
        for name in ["low", "medium", "high", "quiet", "moderate", "loud"]:
            fn = getattr(self, name, None)
            if fn is not None:
                result[name] = fn

        # Check extra fields (dynamically defined sets like 'very_low', 'very_high')
        if self.model_extra:
            for name, value in self.model_extra.items():
                if isinstance(value, MembershipFunction):
                    result[name] = value
                elif isinstance(value, dict):
                    # Convert dict to MembershipFunction
                    result[name] = MembershipFunction(**value)

        return result


class ContextCondition(BaseModel):
    """Condition specification for a context assumption.

    Attributes:
        fuzzy_set: Name of the fuzzy set (e.g., "high", "low")
        weight: Contribution weight for this condition
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    fuzzy_set: str = Field(alias="set")
    weight: float = Field(ge=0.0, le=1.0)


class ContextAssumptionConfig(BaseModel):
    """Configuration for a context assumption.

    Attributes:
        conditions: Dict mapping marker name to condition spec
        operator: Aggregation operator (WEIGHTED_MEAN computes FASL weighted mean)
        threshold: Optional fallback activation threshold
    """

    model_config = ConfigDict(frozen=True)

    conditions: dict[str, ContextCondition] = Field(default_factory=dict)
    operator: Literal["WEIGHTED_MEAN"] = "WEIGHTED_MEAN"
    threshold: float | None = None

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> Self:
        """Validate that condition weights sum to 1.0 within tolerance.

        Skips validation if no conditions (threshold-only assumption).
        """
        if not self.conditions:
            return self

        total = sum(c.weight for c in self.conditions.values())
        if not (0.999 <= total <= 1.001):  # Tolerance for floating point
            weights_detail = [(k, v.weight) for k, v in self.conditions.items()]
            raise ValueError(
                f"Condition weights must sum to 1.0, got {total:.4f}. "
                f"Weights: {weights_detail}"
            )
        return self


class ContextEvaluationConfig(BaseModel):
    """Complete context evaluation configuration.

    Attributes:
        marker_memberships: Dict of marker name to membership config
        context_assumptions: Dict of context name to assumption config
    """

    model_config = ConfigDict(frozen=True)

    marker_memberships: dict[str, MarkerMembershipConfig]
    context_assumptions: dict[str, ContextAssumptionConfig]


# ============================================================================
# Context Assumption (Task 6)
# ============================================================================


@dataclass
class ContextAssumption:
    """A context assumption with weighted conditions on markers.

    Example:
        social context: people_in_room=high (0.6), ambient_noise=moderate (0.4)
    """

    name: str
    conditions: dict[str, tuple[str, float]]  # marker -> (set_name, weight)
    operator: Literal["WEIGHTED_MEAN"] = "WEIGHTED_MEAN"
    threshold: float | None = None

    def evaluate(
        self,
        marker_memberships: dict[str, dict[str, float]],
    ) -> tuple[float, list[str], list[str]]:
        """Evaluate assumption against marker membership values.

        Computes the FASL weighted mean: score = Σ(weight_i × membership_i) / Σ(weight_i).
        Missing markers are excluded and remaining weights are renormalised.

        Args:
            marker_memberships: Dict of marker name -> {set_name: membership_degree}

        Returns:
            Tuple of (score, markers_used, markers_missing)
        """
        scores = []
        weights = []
        markers_used = []
        markers_missing = []

        for marker_name, (set_name, weight) in self.conditions.items():
            if marker_name not in marker_memberships:
                markers_missing.append(marker_name)
                continue

            marker_sets = marker_memberships[marker_name]
            if set_name not in marker_sets:
                markers_missing.append(marker_name)
                continue

            membership_value = marker_sets[set_name]
            scores.append(membership_value)
            weights.append(weight)
            markers_used.append(marker_name)

        if not scores:
            # No markers available - use threshold as fallback if defined
            if self.threshold is not None:
                return self.threshold, [], list(self.conditions.keys())
            return 0.0, [], list(self.conditions.keys())

        # FASL weighted mean aggregation
        weighted_sum = sum(w * s for w, s in zip(weights, scores, strict=True))
        total_weight = sum(weights)
        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return score, markers_used, markers_missing


# ============================================================================
# Config Loading (Task 8)
# ============================================================================


def load_context_config(path: str | Path) -> ContextEvaluationConfig:
    """Load context evaluation configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        ContextEvaluationConfig instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Context config not found: {file_path}")

    with open(file_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty context config file: {file_path}")

    return ContextEvaluationConfig(**data)


def get_default_context_config() -> ContextEvaluationConfig:
    """Get default context evaluation configuration.

    Returns:
        ContextEvaluationConfig with sensible defaults
    """
    return ContextEvaluationConfig(
        marker_memberships={
            "people_in_room": MarkerMembershipConfig(
                low=MembershipFunction(type="triangular", params=[0, 0, 2]),
                medium=MembershipFunction(type="triangular", params=[1, 3, 5]),
                high=MembershipFunction(type="trapezoidal", params=[4, 6, 10, 10]),
            ),
            "ambient_noise": MarkerMembershipConfig(
                quiet=MembershipFunction(type="triangular", params=[0, 0, 0.3]),
                moderate=MembershipFunction(type="triangular", params=[0.2, 0.5, 0.8]),
                loud=MembershipFunction(type="triangular", params=[0.7, 1.0, 1.0]),
            ),
            "network_activity_level": MarkerMembershipConfig(
                low=MembershipFunction(type="triangular", params=[0, 0, 0.4]),
                medium=MembershipFunction(type="triangular", params=[0.2, 0.5, 0.8]),
                high=MembershipFunction(type="triangular", params=[0.6, 1.0, 1.0]),
            ),
        },
        context_assumptions={
            "solitary_digital": ContextAssumptionConfig(
                conditions={
                    "people_in_room": ContextCondition(set="low", weight=0.4),
                    "network_activity_level": ContextCondition(set="high", weight=0.6),
                },
                operator="WEIGHTED_MEAN",
            ),
            "neutral": ContextAssumptionConfig(
                conditions={},
                threshold=0.3,
            ),
        },
    )


# ============================================================================
# Context Evaluator (Task 7)
# ============================================================================


class ContextEvaluator:
    """Main context evaluator using fuzzy logic and EMA smoothing.

    Uses configurable membership functions and context assumptions
    to evaluate context from markers.
    """

    def __init__(
        self,
        analysis_config: AnalysisConfig | None = None,
        context_config: ContextEvaluationConfig | None = None,
    ) -> None:
        """Initialize context evaluator.

        Args:
            analysis_config: Optional AnalysisConfig for EMA settings
            context_config: Optional context-specific configuration
        """
        self._logger = logging.getLogger(__name__)

        # Get EMA config from analysis config or load from YAML
        # Story 6.13: EMA is now nested under context_evaluation
        if analysis_config is not None:
            ema_config = analysis_config.context_evaluation.ema
        else:
            # Load from YAML (authoritative source) as fallback
            ema_path = (
                Path(__file__).parent.parent.parent.parent / "config" / "ema.yaml"
            )
            with open(ema_path) as f:
                ema_data = yaml.safe_load(f)
            ema_config = EMAConfig(**ema_data)

        # Initialize smoother
        self._smoother = EMASmoother(
            alpha=ema_config.alpha,
            hysteresis=ema_config.hysteresis,
            dwell_time=ema_config.dwell_time,
        )

        # Initialize membership calculator
        self._calculator = MembershipCalculator()

        # Load context config
        self._context_config = context_config or get_default_context_config()

        # Build linguistic variables from config
        self._linguistic_vars: dict[str, LinguisticVariable] = {}
        for (
            marker_name,
            membership_config,
        ) in self._context_config.marker_memberships.items():
            self._linguistic_vars[marker_name] = LinguisticVariable(
                name=marker_name,
                membership_fns=membership_config.get_membership_fns(),
            )

        # Build context assumptions from config
        self._assumptions: dict[str, ContextAssumption] = {}
        for (
            ctx_name,
            assumption_config,
        ) in self._context_config.context_assumptions.items():
            conditions = {}
            for marker_name, condition in assumption_config.conditions.items():
                conditions[marker_name] = (condition.fuzzy_set, condition.weight)

            self._assumptions[ctx_name] = ContextAssumption(
                name=ctx_name,
                conditions=conditions,
                operator=assumption_config.operator,
                threshold=assumption_config.threshold,
            )

        self._logger.info(
            f"ContextEvaluator initialized with {len(self._linguistic_vars)} markers, "
            f"{len(self._assumptions)} contexts"
        )

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentContextEvalConfig,
    ) -> "ContextEvaluator":
        """Create a ContextEvaluator from an ExperimentContextEvalConfig.

        This factory method converts the experiment-storable config format
        (from config.py) to the internal format used by ContextEvaluator.

        Story 6.13: Context Evaluation Experimentation

        Args:
            config: ExperimentContextEvalConfig containing marker memberships,
                   context assumptions, neutral threshold, and EMA settings

        Returns:
            Initialized ContextEvaluator using the provided configuration
        """
        # Convert ExperimentContextEvalConfig to internal ContextEvaluationConfig format
        marker_memberships_internal: dict[str, MarkerMembershipConfig] = {}
        for marker_name, marker_mm in config.marker_memberships.items():
            # Build kwargs for MarkerMembershipConfig with fuzzy sets as dynamic fields
            sets_kwargs = {}
            for set_name, set_def in marker_mm.sets.items():
                # Convert MarkerMembershipSet to MembershipFunction
                sets_kwargs[set_name] = MembershipFunction(
                    type=set_def.type,
                    params=list(set_def.params),
                )
            marker_memberships_internal[marker_name] = MarkerMembershipConfig(
                **sets_kwargs
            )

        context_assumptions_internal: dict[str, ContextAssumptionConfig] = {}
        for ctx_name, ctx_def in config.context_assumptions.items():
            conditions_internal: dict[str, ContextCondition] = {}
            for cond in ctx_def.conditions:
                conditions_internal[cond.marker] = ContextCondition(
                    fuzzy_set=cond.fuzzy_set,
                    weight=cond.weight,
                )
            context_assumptions_internal[ctx_name] = ContextAssumptionConfig(
                conditions=conditions_internal,
                operator=ctx_def.operator,
                threshold=None,  # Non-neutral contexts don't use threshold
            )

        # Add neutral context with threshold
        context_assumptions_internal["neutral"] = ContextAssumptionConfig(
            conditions={},
            operator="WEIGHTED_MEAN",
            threshold=config.neutral_threshold,
        )

        context_config = ContextEvaluationConfig(
            marker_memberships=marker_memberships_internal,
            context_assumptions=context_assumptions_internal,
            neutral_threshold=config.neutral_threshold,
        )

        # Create instance with converted config
        instance = cls.__new__(cls)
        instance._logger = logging.getLogger(__name__)

        # Initialize smoother from config's EMA settings
        instance._smoother = EMASmoother(
            alpha=config.ema.alpha,
            hysteresis=config.ema.hysteresis,
            dwell_time=config.ema.dwell_time,
        )

        # Initialize membership calculator
        instance._calculator = MembershipCalculator()

        # Store context config
        instance._context_config = context_config

        # Build linguistic variables from config
        instance._linguistic_vars = {}
        for marker_name, membership_config in context_config.marker_memberships.items():
            instance._linguistic_vars[marker_name] = LinguisticVariable(
                name=marker_name,
                membership_fns=membership_config.get_membership_fns(),
            )

        # Build context assumptions from config
        instance._assumptions = {}
        for ctx_name, assumption_config in context_config.context_assumptions.items():
            conditions = {}
            for marker_name, condition in assumption_config.conditions.items():
                conditions[marker_name] = (condition.fuzzy_set, condition.weight)

            instance._assumptions[ctx_name] = ContextAssumption(
                name=ctx_name,
                conditions=conditions,
                operator=assumption_config.operator,
                threshold=assumption_config.threshold,
            )

        instance._logger.info(
            f"ContextEvaluator.from_experiment_config: initialized with "
            f"{len(instance._linguistic_vars)} markers, {len(instance._assumptions)} contexts"
        )

        return instance

    def _extract_latest_values(
        self,
        context_markers: list[ContextRecord],
    ) -> dict[str, float]:
        """Extract the latest value for each marker name.

        Args:
            context_markers: List of context records

        Returns:
            Dict mapping marker name to latest value
        """
        latest: dict[str, tuple[datetime, float]] = {}

        for record in context_markers:
            if record.name not in latest or record.timestamp > latest[record.name][0]:
                latest[record.name] = (record.timestamp, record.value)

        return {name: value for name, (_, value) in latest.items()}

    def _compute_marker_memberships(
        self,
        marker_values: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """Compute fuzzy membership for each marker value.

        Args:
            marker_values: Dict of marker name to value

        Returns:
            Dict of marker name to {set_name: membership_degree}
        """
        result = {}
        for marker_name, value in marker_values.items():
            if marker_name in self._linguistic_vars:
                result[marker_name] = self._linguistic_vars[marker_name].evaluate(value)
        return result

    def evaluate(
        self,
        context_markers: list[ContextRecord],
        apply_smoothing: bool = True,
    ) -> ContextResult:
        """Evaluate context from markers.

        Args:
            context_markers: List of context records from DataReader
            apply_smoothing: Whether to apply EMA smoothing

        Returns:
            ContextResult with active context and confidence scores
        """
        # Determine timestamp
        if context_markers:
            timestamp = max(r.timestamp for r in context_markers)
        else:
            timestamp = datetime.now(UTC)

        # Extract latest marker values
        marker_values = self._extract_latest_values(context_markers)

        # Handle no markers case
        if not marker_values:
            self._logger.warning("No context markers available for evaluation")
            all_markers = list(self._linguistic_vars.keys())
            return ContextResult(
                active_context="neutral",
                confidence_scores={"neutral": 1.0},
                raw_scores={},
                smoothed=False,
                markers_used=(),
                markers_missing=tuple(all_markers),
                timestamp=timestamp,
            )

        # Compute fuzzy memberships for available markers
        marker_memberships = self._compute_marker_memberships(marker_values)

        # Evaluate each context assumption
        raw_scores: dict[str, float] = {}
        all_markers_used: set[str] = set()
        all_markers_missing: set[str] = set()

        for ctx_name, assumption in self._assumptions.items():
            score, used, missing = assumption.evaluate(marker_memberships)
            raw_scores[ctx_name] = score
            all_markers_used.update(used)
            all_markers_missing.update(missing)

        # Apply EMA smoothing if requested
        confidence_scores: dict[str, float] = {}
        if apply_smoothing:
            for ctx_name, raw_score in raw_scores.items():
                confidence_scores[ctx_name] = self._smoother.smooth(ctx_name, raw_score)
        else:
            confidence_scores = raw_scores.copy()

        # Determine active context (highest confidence)
        switch_decision: SwitchDecision | None = None
        if confidence_scores:
            active_context = max(confidence_scores, key=lambda k: confidence_scores[k])

            # Apply hysteresis and dwell time for context switching
            if apply_smoothing:
                current_context = self._smoother.get_active_context()
                current_score = (
                    confidence_scores.get(current_context, 0.0)
                    if current_context
                    else 0.0
                )
                switch_decision = self._smoother.should_switch_context(
                    active_context,
                    confidence_scores[active_context],
                    current_context,
                    current_score,
                )
                if not switch_decision.should_switch:
                    # Keep current context if switch not allowed
                    if current_context:
                        active_context = current_context
        else:
            active_context = "neutral"

        # Remove markers from missing that were actually used
        final_missing = all_markers_missing - all_markers_used

        # Build result with stabilization transparency info
        result = ContextResult(
            active_context=active_context,
            confidence_scores=confidence_scores,
            raw_scores=raw_scores,
            smoothed=apply_smoothing,
            markers_used=tuple(sorted(all_markers_used)),
            markers_missing=tuple(sorted(final_missing)),
            timestamp=timestamp,
            # Stabilization fields from switch decision
            switch_blocked=(
                switch_decision.blocked_reason is not None if switch_decision else False
            ),
            switch_blocked_reason=(
                switch_decision.blocked_reason if switch_decision else None
            ),
            candidate_context=(
                switch_decision.candidate_context if switch_decision else None
            ),
            score_difference=(
                switch_decision.score_difference if switch_decision else None
            ),
            dwell_progress=(
                switch_decision.dwell_progress if switch_decision else None
            ),
            dwell_required=(
                switch_decision.dwell_required if switch_decision else None
            ),
        )

        self._logger.debug(
            f"Context evaluation: active={active_context}, "
            f"scores={confidence_scores}, "
            f"used={all_markers_used}, missing={final_missing}"
        )

        return result

    def evaluate_batch(
        self,
        markers_list: list[list[ContextRecord]],
        apply_smoothing: bool = True,
    ) -> list[ContextResult]:
        """Evaluate multiple sets of markers, maintaining EMA state.

        Args:
            markers_list: List of marker lists, one per timestamp
            apply_smoothing: Whether to apply EMA smoothing

        Returns:
            List of ContextResult, one per input set
        """
        results = []
        for markers in markers_list:
            result = self.evaluate(markers, apply_smoothing=apply_smoothing)
            results.append(result)
        return results

    def reset(self) -> None:
        """Reset evaluator state (clears EMA smoother)."""
        self._smoother.reset()
        self._logger.debug("ContextEvaluator state reset")

    def initialize_state(
        self,
        previous_values: dict[str, float],
        active_context: str | None = None,
    ) -> None:
        """Initialize smoother state for continuity during backfill.

        Call this before processing a batch of historical evaluations to
        ensure EMA smoothing continues from the last known state.

        Args:
            previous_values: Dict of context name to last smoothed score
            active_context: The last active context (for hysteresis/dwell)
        """
        self._smoother._previous_values = previous_values.copy()
        if active_context:
            self._smoother._active_context = active_context
            # Initialize dwell counter for active context
            self._smoother._dwell_counters[active_context] = self._smoother.dwell_time
        self._logger.debug(
            f"ContextEvaluator state initialized: active={active_context}, "
            f"contexts={list(previous_values.keys())}"
        )
