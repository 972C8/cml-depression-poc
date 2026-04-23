"""Context evaluation package for multimodal analysis engine.

This package provides fuzzy logic-based context evaluation with EMA smoothing
and context-aware biomarker weight adjustment.
"""

from src.core.context.evaluator import (
    ContextAssumption,
    ContextAssumptionConfig,
    ContextCondition,
    ContextEvaluationConfig,
    ContextEvaluator,
    ContextResult,
    MarkerMembershipConfig,
    get_default_context_config,
    load_context_config,
)
from src.core.context.membership import (
    LinguisticVariable,
    MembershipCalculator,
)
from src.core.context.smoother import EMASmoother
from src.core.context.weights import (
    AdjustedIndicatorWeights,
    ContextWeightAdjuster,
    adjust_biomarker_weights,
)
from src.core.context.history import (
    ContextHistoryService,
    ContextHistoryStatus,
    ContextSegment,
    ContextState,
    EnsureHistoryResult,
    SensorSnapshot,
)
from src.core.context.strategies import (
    ContextStrategyResult,
    get_window_context,
    get_window_context_dominant,
    get_window_context_reading_weighted,
    get_window_context_time_weighted,
)

__all__ = [
    # Evaluator
    "ContextAssumption",
    "ContextAssumptionConfig",
    "ContextCondition",
    "ContextEvaluationConfig",
    "ContextEvaluator",
    "ContextResult",
    "MarkerMembershipConfig",
    "get_default_context_config",
    "load_context_config",
    # Membership
    "LinguisticVariable",
    "MembershipCalculator",
    # Smoother
    "EMASmoother",
    # Weights
    "AdjustedIndicatorWeights",
    "ContextWeightAdjuster",
    "adjust_biomarker_weights",
    # History (Story 6.1)
    "ContextHistoryService",
    "ContextHistoryStatus",
    "ContextSegment",
    "ContextState",
    "EnsureHistoryResult",
    "SensorSnapshot",
    # Strategies (Story 6.3)
    "ContextStrategyResult",
    "get_window_context",
    "get_window_context_dominant",
    "get_window_context_reading_weighted",
    "get_window_context_time_weighted",
]
