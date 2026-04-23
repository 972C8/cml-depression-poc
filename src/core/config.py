"""Analysis Configuration Module.

This module defines configuration models for the multimodal analysis engine.
Configuration includes indicator definitions, biomarker weights, DSM-gate parameters,
and context-aware adjustments.
"""

import json
import logging
from pathlib import Path
from typing import Literal, Self
from zoneinfo import ZoneInfo

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisConfig",
    "BaselineDefaults",
    "BiomarkerMembershipFunction",
    "BiomarkerProcessingConfig",
    "BiomarkerWeight",
    "ConfigurationError",
    "ContextAssumptionDef",
    "ContextConditionDef",
    "ContextConfig",
    "ContextHistoryConfig",
    "DSMGateConfig",
    "EMAConfig",
    "EpisodeConfig",
    "ExperimentContextEvalConfig",
    "FaslConfig",
    "GenericBaselineConfig",
    "IndicatorConfig",
    "MarkerMembership",
    "MarkerMembershipSet",
    "MembershipFunction",
    "ReliabilityConfig",
    "WindowConfig",
    "ZScoreBoundsConfig",
    "get_default_config",
    "load_config",
]


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class BaselineDefaults(BaseModel):
    """Population baseline defaults for cold-start scenarios.

    Used when a user has fewer than min_data_points of history
    for a specific biomarker. Provides sensible defaults based
    on population statistics.

    Attributes:
        mean: Population mean for this biomarker
        std: Population standard deviation (must be > 0)
        min_data_points: Switch to user-specific baseline after this many data points
    """

    model_config = ConfigDict(frozen=True)

    mean: float = Field(description="Population mean for this biomarker")
    std: float = Field(gt=0, description="Population std deviation (must be > 0)")
    min_data_points: int = Field(
        default=7,
        ge=1,
        description="Switch to user-specific baseline after this many data points",
    )


class ZScoreBoundsConfig(BaseModel):
    """Z-score normalization bounds configuration.

    When no membership function is configured for a biomarker, z-scores are
    linearly mapped to [0,1] using these bounds. Values outside these bounds
    are clipped to the [0,1] range.

    Attributes:
        lower: Lower bound for z-score normalization (default: -3.0)
        upper: Upper bound for z-score normalization (default: 3.0)
    """

    model_config = ConfigDict(frozen=True)

    lower: float = Field(
        default=-3.0,
        description="Lower bound for z-score normalization (typically -3.0)",
    )
    upper: float = Field(
        default=3.0,
        description="Upper bound for z-score normalization (typically 3.0)",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        """Validate that lower < upper."""
        if self.lower >= self.upper:
            raise ValueError(
                f"z_score_bounds.lower ({self.lower}) must be less than "
                f"z_score_bounds.upper ({self.upper})"
            )
        return self

    @property
    def range(self) -> float:
        """Get the range between upper and lower bounds."""
        return self.upper - self.lower


class GenericBaselineConfig(BaseModel):
    """Generic baseline fallback configuration.

    Used when a biomarker has no population default configured AND not enough
    user data for a personal baseline. These are conservative defaults that
    assume a normalized [0,1] range with moderate variance.

    Attributes:
        mean: Generic fallback mean value (default: 0.5)
        std: Generic fallback standard deviation (default: 0.2)
    """

    model_config = ConfigDict(frozen=True)

    mean: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Generic fallback mean (0.5 = center of normalized range)",
    )
    std: float = Field(
        default=0.2,
        gt=0.0,
        description="Generic fallback standard deviation",
    )


class ReliabilityConfig(BaseModel):
    """Data reliability scoring configuration.

    Controls how data quality and coverage are weighted when computing
    the reliability score for indicator computations.

    Formula: data_reliability = (coverage * coverage_weight) + (quality * quality_weight)

    Attributes:
        coverage_weight: Weight for biomarker coverage (0-1, default: 0.6)
        quality_weight: Weight for data quality (0-1, default: 0.4)
        population_baseline_quality_penalty: Quality multiplier for population baselines (0-1, default: 0.5)
    """

    model_config = ConfigDict(frozen=True)

    coverage_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for biomarker availability in reliability calculation",
    )
    quality_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for data quality in reliability calculation",
    )
    population_baseline_quality_penalty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quality multiplier when using population baseline (caps quality score)",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> Self:
        """Validate that weights sum to approximately 1.0."""
        total = self.coverage_weight + self.quality_weight
        if not (0.99 <= total <= 1.01):
            logger.warning(
                f"Reliability weights sum to {total:.2f}, expected ~1.0. "
                "Results may not be normalized."
            )
        return self


class BiomarkerProcessingConfig(BaseModel):
    """Global biomarker processing configuration.

    Contains all global parameters that control how biomarkers are processed,
    including z-score bounds, minimum thresholds, and fallback defaults.

    Attributes:
        z_score_bounds: Bounds for z-score to membership normalization
        z_score_warning_threshold: Threshold for logging extreme z-score warnings
        default_min_data_points: Fallback min_data_points when biomarker not configured
        min_std_deviation: Floor for standard deviation to prevent division by zero
        generic_baseline: Fallback baseline when no population default exists
    """

    model_config = ConfigDict(frozen=True)

    z_score_bounds: ZScoreBoundsConfig = Field(default_factory=ZScoreBoundsConfig)
    z_score_warning_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="Z-scores exceeding this absolute value trigger warnings",
    )
    default_min_data_points: int = Field(
        default=7,
        ge=1,
        description="Fallback min_data_points for unconfigured biomarkers",
    )
    min_std_deviation: float = Field(
        default=0.1,
        gt=0.0,
        description="Minimum std deviation floor to prevent division by zero",
    )
    generic_baseline: GenericBaselineConfig = Field(
        default_factory=GenericBaselineConfig
    )


class BiomarkerWeight(BaseModel):
    """Weight and direction configuration for a biomarker within an indicator.

    Attributes:
        weight: Contribution weight (0-1) for this biomarker within the indicator
        direction: Interpretation direction for z-scores:
            - "higher_is_worse": Positive z-scores indicate concern
            - "lower_is_worse": Negative z-scores indicate concern (inverted)
    """

    model_config = ConfigDict(frozen=True)

    weight: float = Field(ge=0.0, le=1.0)
    direction: Literal["higher_is_worse", "lower_is_worse"]


class BiomarkerMembershipFunction(BaseModel):
    """Membership function configuration for biomarker normalization.

    Defines the function type and parameters used to map z-scores to
    membership values in [0, 1]. Each biomarker can use a different
    membership function based on its expected distribution.

    Attributes:
        type: Function type (triangular, sigmoid, exponential_ramp, gaussian)
        params: Dictionary of function-specific parameters:
            - triangular: l (left), m (peak), h (right)
            - sigmoid: x0 (center), k (steepness)
            - exponential_ramp: tau (threshold), lam (rate)
            - gaussian: c (center), sigma (width)

    """

    model_config = ConfigDict(frozen=True)

    type: Literal["triangular", "sigmoid", "exponential_ramp", "gaussian"] = Field(
        description="Membership function type"
    )
    params: dict[str, float] = Field(
        description="Function-specific parameters as key-value pairs"
    )

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        """Validate that required parameters exist for the function type."""
        required_params = {
            "triangular": {"l", "m", "h"},
            "sigmoid": {"x0", "k"},
            "exponential_ramp": {"tau", "lam"},
            "gaussian": {"c", "sigma"},
        }

        required = required_params[self.type]
        provided = set(self.params.keys())

        if not required.issubset(provided):
            missing = required - provided
            raise ValueError(f"Missing required parameters for {self.type}: {missing}")

        return self


class DSMGateConfig(BaseModel):
    """DSM-gate parameters for per-indicator presence evaluation.

    The DSM-gate implements the clinical counting rule from DSM-5.
    For each indicator, a daily likelihood L_k(d) is compared against
    theta to produce a binary flag. The indicator is considered
    "present" if at least gate_need of the last m_window days are
    positive (N-of-M rule), operationalising the DSM-5 phrase
    "nearly every day" over the two-week criterion period.

    Attributes:
        theta: Threshold for daily likelihood (0-1). Day counts as "active" if above this value
        m_window: Sliding-window size in days (default: 14 = 2 weeks per DSM-5)
        gate_need: Required positive days within the window for indicator presence (default: 10)
    """

    model_config = ConfigDict(frozen=True)

    theta: float = 0.6
    m_window: int = 14
    gate_need: int = 10


class EMAConfig(BaseModel):
    """Exponential Moving Average (EMA) smoothing parameters.

    EMA smoothing reduces noise in indicator scores and prevents rapid oscillations
    in context transitions.

    Attributes:
        alpha: Smoothing factor (0 < alpha <= 1). Higher values = less smoothing
        hysteresis: Threshold buffer to prevent jitter in state transitions
        dwell_time: Minimum number of periods before allowing context transition
    """

    model_config = ConfigDict(frozen=True)

    alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    hysteresis: float = 0.1
    dwell_time: int = 2


def _load_ema_config_from_yaml() -> EMAConfig:
    """Load EMA config from YAML file (authoritative source).

    This function ensures config/ema.yaml is always the source of truth
    for EMA settings, rather than class defaults.

    Returns:
        EMAConfig loaded from config/ema.yaml
    """
    ema_path = Path(__file__).parent.parent.parent / "config" / "ema.yaml"
    try:
        with open(ema_path) as f:
            ema_data = yaml.safe_load(f)
        return EMAConfig(**ema_data)
    except FileNotFoundError:
        logger.warning(f"EMA config not found at {ema_path}, using defaults")
        return EMAConfig()


# ============================================================================
# Experiment Context Evaluation Configuration (Story 6.13)
# ============================================================================


class MarkerMembershipSet(BaseModel):
    """Single fuzzy set definition for a marker.

    Defines the shape and parameters of a fuzzy membership function
    used to convert raw marker values to membership degrees.

    Attributes:
        type: Function type - "triangular" (3 params) or "trapezoidal" (4 params)
        params: Function parameters as list of floats
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["triangular", "trapezoidal"]
    params: list[float]

    @model_validator(mode="after")
    def validate_params_length(self) -> Self:
        """Validate that params length matches the function type."""
        expected_length = 3 if self.type == "triangular" else 4
        if len(self.params) != expected_length:
            raise ValueError(
                f"{self.type} membership function requires {expected_length} "
                f"parameters, got {len(self.params)}"
            )
        return self


class MarkerMembership(BaseModel):
    """All fuzzy sets for a single marker.

    Contains named fuzzy sets (e.g., low, medium, high) for a marker
    like people_in_room or ambient_noise.

    Attributes:
        sets: Dict mapping set name to fuzzy set definition
    """

    model_config = ConfigDict(frozen=True)

    sets: dict[str, MarkerMembershipSet]


class ContextConditionDef(BaseModel):
    """Single condition in a context assumption.

    Defines which fuzzy set a marker must match and how much weight
    that condition contributes to the context assumption.

    Attributes:
        marker: Name of the marker (e.g., "people_in_room")
        fuzzy_set: Name of the fuzzy set to match (e.g., "high")
        weight: Contribution weight (0-1) for this condition
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    marker: str
    fuzzy_set: str = Field(alias="set")
    weight: float = Field(ge=0.0, le=1.0)


class ContextAssumptionDef(BaseModel):
    """Definition of a context assumption.

    A context assumption defines the fuzzy logic rules for detecting
    a specific context (e.g., solitary_digital).

    Attributes:
        conditions: List of marker conditions that define this context
        operator: Aggregation operator (WEIGHTED_MEAN computes FASL weighted mean)
    """

    model_config = ConfigDict(frozen=True)

    conditions: list[ContextConditionDef]
    operator: Literal["WEIGHTED_MEAN"] = "WEIGHTED_MEAN"

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> Self:
        """Validate that condition weights sum to 1.0 within tolerance."""
        if not self.conditions:
            return self

        total = sum(c.weight for c in self.conditions)
        if not (0.999 <= total <= 1.001):
            weights_detail = [(c.marker, c.weight) for c in self.conditions]
            raise ValueError(
                f"Condition weights must sum to 1.0, got {total:.4f}. "
                f"Weights: {weights_detail}"
            )
        return self


class ExperimentContextEvalConfig(BaseModel):
    """Unified context evaluation config for experiments.

    Contains ALL parameters that affect context evaluation:
    - Fuzzy logic (marker memberships, assumptions, threshold)
    - EMA smoothing (moved from AnalysisConfig.ema)

    This is the experiment-storable version of context evaluation parameters.
    Use ContextEvaluator.from_experiment_config() to convert to the internal
    format used by the evaluator.

    Note: Named differently from ContextEvaluationConfig in evaluator.py
    to avoid import conflicts.

    Story 6.13: Context Evaluation Experimentation

    Attributes:
        marker_memberships: Dict mapping marker name to its fuzzy sets
        context_assumptions: Dict mapping context name to its definition
        neutral_threshold: Threshold below which contexts fall back to neutral
        ema: EMA smoothing parameters (alpha, hysteresis, dwell_time)
    """

    model_config = ConfigDict(frozen=True)

    marker_memberships: dict[str, MarkerMembership]
    context_assumptions: dict[str, ContextAssumptionDef]
    neutral_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    ema: EMAConfig


class ContextHistoryConfig(BaseModel):
    """Context history configuration for temporal context binding.

    Controls how context evaluations are stored and retrieved for
    accurate context-aware weighting of biomarker data.

    Story 6.1: Context History Infrastructure (AC6)

    Attributes:
        evaluation_interval_minutes: How often to evaluate context (default: 15)
        staleness_hours: Max age before fallback to neutral context (default: 2)
        neutral_weight: Default weight when context is unknown (default: 1.0)
        clock_skew_tolerance_minutes: Tolerance for slight future timestamps (default: 5)
        high_confidence_ratio: Ratio of staleness_hours for 'high' confidence (default: 0.5)
    """

    model_config = ConfigDict(frozen=True)

    evaluation_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="How often to evaluate context (minutes)",
    )
    staleness_hours: float = Field(
        default=2.0,
        gt=0.0,
        le=24.0,
        description="Max age before context is considered stale (hours)",
    )
    neutral_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Default weight when context is unknown",
    )
    clock_skew_tolerance_minutes: int = Field(
        default=5,
        ge=0,
        le=30,
        description="Tolerance for slight future timestamps (minutes)",
    )
    high_confidence_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Ratio of staleness_hours below which confidence is 'high' (default: 0.5 = half of staleness)",
    )


class WindowConfig(BaseModel):
    """Window aggregation configuration.

    Controls how biomarker readings are aggregated into time windows
    for context binding and downstream analysis.

    Story 6.2: Window Aggregation Module (AC6)

    Attributes:
        size_minutes: Window size in minutes (5, 10, 15, 30, or 60)
        aggregation_method: Method for aggregating values ("mean", "median", "max", "min")
        min_readings: Minimum readings required for valid window (default: 1)
    """

    model_config = ConfigDict(frozen=True)

    size_minutes: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Window size in minutes (5, 10, 15, 30, or 60)",
    )
    aggregation_method: Literal["mean", "median", "max", "min"] = Field(
        default="mean",
        description="Aggregation method for values within window",
    )
    min_readings: int = Field(
        default=1,
        ge=1,
        description="Minimum readings required for valid window",
    )

    @field_validator("size_minutes")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        """Validate that window size is one of the allowed values."""
        valid_sizes = {5, 10, 15, 30, 60}
        if v not in valid_sizes:
            raise ValueError(
                f"Window size must be one of {sorted(valid_sizes)}, got {v}"
            )
        return v


class ContextConfig(BaseModel):
    """Context strategy configuration for window membership computation.

    Story 6.3: Membership with Context Weighting (AC7)

    Controls how context state is determined for each window when computing
    context-weighted membership values.

    Attributes:
        strategy: Context strategy for window analysis:
            - 'dominant': Use context at window midpoint (default, simplest)
            - 'time_weighted': Blend weights by time proportion within window
            - 'reading_weighted': Average weights across individual readings
    """

    model_config = ConfigDict(frozen=True)

    strategy: Literal["dominant", "time_weighted", "reading_weighted"] = Field(
        default="dominant",
        description="Context strategy for window analysis",
    )


class FaslConfig(BaseModel):
    """FASL (Fuzzy-Aggregated Symptom Likelihood) configuration.

    Story 6.4: Window-Level FASL Aggregation (AC5)

    Controls how window-level FASL is computed, including handling
    of missing biomarker data in windows.

    Attributes:
        missing_biomarker_strategy: Strategy for handling missing biomarkers:
            - 'neutral_fill': Use neutral_membership for missing biomarkers (default)
            - 'partial_fasl': Use only present biomarkers with renormalized weights
            - 'skip_window': Don't compute indicator for incomplete windows
        neutral_membership: Value to use for missing biomarkers in neutral_fill (default: 0.5)
    """

    model_config = ConfigDict(frozen=True)

    missing_biomarker_strategy: Literal[
        "neutral_fill", "partial_fasl", "skip_window"
    ] = Field(
        default="neutral_fill",
        description="Strategy for handling missing biomarkers in windows",
    )
    neutral_membership: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Value to use for missing biomarkers in neutral_fill strategy",
    )


class EpisodeConfig(BaseModel):
    """Episode decision configuration per DSM-5 MDD criteria.

    DSM-5 defines Major Depressive Disorder episode requiring:
    - At least 5 of 9 criteria present
    - At least one of the two core criteria present (depressed mood or loss of interest)

    Attributes:
        min_indicators: Minimum indicators required for episode (DSM-5: >=5 of 9)
        core_indicators: List of core symptom indicators (at least one must be present)
        peak_window_k: Number of consecutive windows for daily likelihood peak
            detection. Daily likelihood = max mean of any k consecutive windows.
            With 15-min windows, k=4 means "worst contiguous hour of the day".
    """

    model_config = ConfigDict(frozen=True)

    min_indicators: int = Field(default=5, ge=1, le=9)
    core_indicators: tuple[str, ...] = Field(
        default=("1_depressed_mood", "2_loss_of_interest")
    )
    peak_window_k: int = Field(
        default=4,
        ge=1,
        description="Number of consecutive windows for daily likelihood peak detection",
    )

    @field_validator("core_indicators", mode="before")
    @classmethod
    def convert_list_to_tuple(cls, v):
        """Convert list to tuple for immutability."""
        if isinstance(v, list):
            return tuple(v)
        return v


class MembershipFunction(BaseModel):
    """Fuzzy logic membership function configuration.

    Defines the shape of membership functions used in fuzzy logic computations.

    Attributes:
        type: Function type - "triangular" (3 params) or "trapezoidal" (4 params)
        params: Function parameters:
            - Triangular [a, b, c]: a=left foot, b=peak, c=right foot
            - Trapezoidal [a, b, c, d]: a=left foot, b=left shoulder,
              c=right shoulder, d=right foot
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["triangular", "trapezoidal"]
    params: list[float]

    @model_validator(mode="after")
    def validate_params_length(self) -> Self:
        """Validate that params length matches the function type."""
        expected_length = 3 if self.type == "triangular" else 4
        if len(self.params) != expected_length:
            raise ValueError(
                f"{self.type} membership function requires {expected_length} "
                f"parameters, got {len(self.params)}"
            )
        return self


class IndicatorConfig(BaseModel):
    """Configuration for a single indicator.

    An indicator aggregates multiple biomarkers with specified weights to compute
    a clinical measure (e.g., social_withdrawal, diminished_interest).

    Attributes:
        biomarkers: Dict mapping biomarker names to their weight and direction config
        dsm_gate: Optional per-indicator DSM-gate parameters (overrides global defaults)
        min_biomarkers: Minimum number of biomarkers required for valid computation
    """

    model_config = ConfigDict(frozen=True)

    biomarkers: dict[str, BiomarkerWeight]
    dsm_gate: DSMGateConfig | None = None
    min_biomarkers: int = 1

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> Self:
        """Validate that biomarker weights sum to 1.0 within tolerance."""
        total = sum(bw.weight for bw in self.biomarkers.values())
        if not (0.999 <= total <= 1.001):  # Tolerance for floating point
            weights_detail = [(k, v.weight) for k, v in self.biomarkers.items()]
            raise ValueError(
                f"Biomarker weights must sum to 1.0, got {total:.4f}. "
                f"Weights: {weights_detail}"
            )
        return self


class AnalysisConfig(BaseModel):
    """Complete analysis configuration.

    This is the top-level configuration model containing all parameters for
    multimodal analysis: indicators, context weights, DSM-gate, and context evaluation.

    Story 6.13: EMA parameters moved from top-level `ema` field into
    `context_evaluation.ema` for unified context evaluation configuration.

    Attributes:
        indicators: Dict of indicator names to their configurations
        context_weights: Context-specific biomarker multipliers (context -> biomarker -> multiplier)
            Missing biomarkers default to multiplier of 1.0
        timezone: IANA timezone string for day boundary calculation (default UTC)
        biomarker_defaults: Population baseline defaults for cold-start scenarios
            (biomarker name -> BaselineDefaults)
        biomarker_processing: Global biomarker processing parameters (z-score bounds, etc.)
        reliability: Data reliability scoring configuration
        membership_functions: Optional fuzzy logic membership function definitions
        dsm_gate_defaults: Default DSM-gate parameters (can be overridden per-indicator)
        context_evaluation: Unified context evaluation config (includes EMA, Story 6.13)
        episode: Episode decision configuration (min indicators, core indicators)
    """

    model_config = ConfigDict(frozen=True)

    indicators: dict[str, IndicatorConfig]
    context_weights: dict[str, dict[str, float]]
    timezone: str = "Europe/Zurich"
    biomarker_defaults: dict[str, BaselineDefaults] = Field(default_factory=dict)
    biomarker_processing: BiomarkerProcessingConfig = Field(
        default_factory=BiomarkerProcessingConfig,
        description="Global biomarker processing parameters",
    )
    reliability: ReliabilityConfig = Field(
        default_factory=ReliabilityConfig,
        description="Data reliability scoring configuration",
    )
    biomarker_membership: dict[str, BiomarkerMembershipFunction] = Field(
        default_factory=dict,
        description="Membership function configurations for biomarker normalization",
    )
    membership_functions: dict[str, MembershipFunction] | None = None
    dsm_gate_defaults: DSMGateConfig = Field(default_factory=DSMGateConfig)
    context_evaluation: "ExperimentContextEvalConfig" = Field(
        default_factory=lambda: _load_default_context_eval_config(),
        description="Unified context evaluation config including EMA (Story 6.13)",
    )
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    context_history: ContextHistoryConfig = Field(
        default_factory=ContextHistoryConfig,
        description="Context history configuration for temporal binding (Story 6.1)",
    )
    window: WindowConfig = Field(
        default_factory=WindowConfig,
        description="Window aggregation configuration (Story 6.2)",
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context strategy configuration for window membership (Story 6.3)",
    )
    fasl: FaslConfig = Field(
        default_factory=FaslConfig,
        description="FASL configuration for window-level indicator computation (Story 6.4)",
    )

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Validate timezone is a valid IANA timezone string."""
        try:
            ZoneInfo(v)
            return v
        except Exception as err:
            raise ValueError(
                f"Invalid timezone '{v}'. Must be valid IANA timezone "
                f"(e.g., 'UTC', 'Europe/Zurich', 'America/New_York')"
            ) from err

    @model_validator(mode="after")
    def validate_context_weights_positive(self) -> Self:
        """Validate that all context weight multipliers are positive."""
        for context_name, biomarker_weights in self.context_weights.items():
            for biomarker_name, multiplier in biomarker_weights.items():
                if multiplier <= 0:
                    raise ValueError(
                        f"Context weight multipliers must be positive, got "
                        f"{multiplier} for biomarker '{biomarker_name}' in "
                        f"context '{context_name}'"
                    )
        return self

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization/storage.

        Returns:
            Dictionary representation suitable for JSON/YAML storage
        """
        return self.model_dump(mode="json")


def load_config(path: str | Path) -> AnalysisConfig:
    """Load analysis configuration from YAML or JSON file.

    Args:
        path: Path to configuration file (.yaml, .yml, or .json)

    Returns:
        Validated AnalysisConfig instance

    Raises:
        ConfigurationError: If file not found, parse error, or validation fails
    """
    file_path = Path(path)

    # Check file existence
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")

    # Load file based on extension
    try:
        with open(file_path) as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ConfigurationError(
                    f"Unsupported file extension: {file_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Failed to parse configuration file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error reading configuration file: {e}") from e

    # Validate data is not empty
    if data is None:
        raise ConfigurationError(
            f"Configuration file is empty or contains only comments: {file_path}"
        )
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file must contain a YAML/JSON mapping, got {type(data).__name__}: {file_path}"
        )

    # Validate and construct AnalysisConfig
    try:
        config = AnalysisConfig(**data)
        indicator_count = len(config.indicators)
        logger.info(
            f"Successfully loaded configuration from {file_path} "
            f"with {indicator_count} indicator(s)"
        )
        return config
    except ValidationError as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


def _get_config_dir() -> Path:
    """Get the config directory path.

    Returns:
        Path to the config directory (project_root/config)
    """
    # Navigate from src/core/config.py to project root
    return Path(__file__).parent.parent.parent / "config"


def _load_yaml_file(filename: str) -> dict:
    """Load a YAML file from the config directory.

    Args:
        filename: Name of the YAML file (e.g., "indicators.yaml")

    Returns:
        Parsed YAML content as dict

    Raises:
        ConfigurationError: If file not found or parse error
    """
    config_dir = _get_config_dir()
    file_path = config_dir / filename

    if not file_path.exists():
        raise ConfigurationError(f"Default config file not found: {file_path}")

    try:
        with open(file_path) as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse {filename}: {e}") from e


def _load_default_context_eval_config() -> "ExperimentContextEvalConfig":
    """Load default ExperimentContextEvalConfig from YAML files.

    Loads and merges configuration from:
    - config/context_evaluation.yaml: marker memberships, context assumptions, neutral threshold
    - config/ema.yaml: EMA smoothing parameters

    Story 6.13: Context Evaluation Experimentation

    Returns:
        ExperimentContextEvalConfig with defaults from YAML files

    Raises:
        ConfigurationError: If required config files are missing or invalid
    """
    context_eval_data = _load_yaml_file("context_evaluation.yaml")
    ema_data = _load_yaml_file("ema.yaml")

    # Parse marker memberships
    marker_memberships: dict[str, MarkerMembership] = {}
    for marker_name, sets_data in context_eval_data.get(
        "marker_memberships", {}
    ).items():
        sets: dict[str, MarkerMembershipSet] = {}
        for set_name, set_def in sets_data.items():
            sets[set_name] = MarkerMembershipSet(
                type=set_def["type"],
                params=set_def["params"],
            )
        marker_memberships[marker_name] = MarkerMembership(sets=sets)

    # Parse context assumptions
    context_assumptions: dict[str, ContextAssumptionDef] = {}
    for ctx_name, ctx_data in context_eval_data.get("context_assumptions", {}).items():
        # Skip 'neutral' which uses threshold-only definition
        if ctx_name == "neutral":
            continue

        conditions: list[ContextConditionDef] = []
        for marker_name, cond_data in ctx_data.get("conditions", {}).items():
            conditions.append(
                ContextConditionDef(
                    marker=marker_name,
                    fuzzy_set=cond_data["set"],
                    weight=cond_data["weight"],
                )
            )
        context_assumptions[ctx_name] = ContextAssumptionDef(
            conditions=conditions,
            operator=ctx_data.get("operator", "WEIGHTED_MEAN"),
        )

    # Get neutral threshold from context_assumptions.neutral.threshold
    neutral_threshold = 0.3
    if "neutral" in context_eval_data.get("context_assumptions", {}):
        neutral_threshold = context_eval_data["context_assumptions"]["neutral"].get(
            "threshold", 0.3
        )

    return ExperimentContextEvalConfig(
        marker_memberships=marker_memberships,
        context_assumptions=context_assumptions,
        neutral_threshold=neutral_threshold,
        ema=EMAConfig(**ema_data),
    )


def get_default_config() -> AnalysisConfig:
    """Get default MDD-focused analysis configuration from YAML files.

    Loads configuration from separate YAML files in the config/ directory:
    - indicators.yaml: Indicator definitions with biomarker weights
    - context_weights.yaml: Context-specific biomarker multipliers
    - reliability.yaml: Data reliability scoring configuration
    - dsm_gate.yaml: DSM-gate parameters
    - context_evaluation.yaml + ema.yaml: Unified context evaluation config (Story 6.13)
    - episode.yaml: Episode decision configuration

    Direction handling in indicator_computation.py:
        - higher_is_worse: contribution = weight * mu
          Use when HIGH raw values indicate concern (e.g., many awakenings)
        - lower_is_worse: contribution = weight * (1 - mu)
          Use when LOW raw values indicate concern (e.g., reduced speech)

    Returns:
        AnalysisConfig with MDD-focused defaults loaded from YAML files

    Raises:
        ConfigurationError: If any required config file is missing or invalid
    """
    # Load individual config files
    indicators_data = _load_yaml_file("indicators.yaml")
    context_weights_data = _load_yaml_file("context_weights.yaml")
    reliability_data = _load_yaml_file("reliability.yaml")
    dsm_gate_data = _load_yaml_file("dsm_gate.yaml")
    episode_data = _load_yaml_file("episode.yaml")

    # Load unified context evaluation config (Story 6.13)
    context_evaluation = _load_default_context_eval_config()

    # Build indicators dict
    indicators = {}
    for name, config in indicators_data.items():
        biomarkers = {
            bio_name: BiomarkerWeight(**bio_config)
            for bio_name, bio_config in config.get("biomarkers", {}).items()
        }
        indicators[name] = IndicatorConfig(biomarkers=biomarkers)

    # Build reliability config
    reliability = ReliabilityConfig(**reliability_data)

    return AnalysisConfig(
        indicators=indicators,
        context_weights=context_weights_data,
        reliability=reliability,
        dsm_gate_defaults=DSMGateConfig(**dsm_gate_data),
        context_evaluation=context_evaluation,
        episode=EpisodeConfig(**episode_data),
    )
