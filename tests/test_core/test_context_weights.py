"""Tests for context-aware biomarker weight adjustment."""

from datetime import UTC, datetime

import pytest

from src.core.config import AnalysisConfig, BiomarkerWeight, IndicatorConfig
from src.core.context.evaluator import ContextResult
from src.core.context.weights import (
    AdjustedIndicatorWeights,
    ContextWeightAdjuster,
    adjust_biomarker_weights,
)


@pytest.fixture
def base_indicator_config() -> IndicatorConfig:
    """Create a test indicator configuration with balanced weights."""
    return IndicatorConfig(
        biomarkers={
            "speech_activity": BiomarkerWeight(
                weight=0.30, direction="higher_is_worse"
            ),
            "voice_energy": BiomarkerWeight(weight=0.20, direction="higher_is_worse"),
            "connections": BiomarkerWeight(weight=0.25, direction="lower_is_worse"),
            "bytes_out": BiomarkerWeight(weight=0.25, direction="lower_is_worse"),
        }
    )


@pytest.fixture
def analysis_config(base_indicator_config: IndicatorConfig) -> AnalysisConfig:
    """Create a test analysis configuration with context weights."""
    return AnalysisConfig(
        indicators={
            "social_withdrawal": base_indicator_config,
        },
        context_weights={
            "solitary_digital": {
                "speech_activity": 0.7,
                "voice_energy": 0.8,
                "connections": 1.5,
                "bytes_out": 1.4,
            },
        },
    )


@pytest.fixture
def social_context_result() -> ContextResult:
    """Create a solitary_digital context result with high confidence."""
    return ContextResult(
        active_context="solitary_digital",
        confidence_scores={"solitary_digital": 0.85, "neutral": 0.15},
        raw_scores={"solitary_digital": 0.82, "neutral": 0.18},
        smoothed=True,
        markers_used=("speech_activity", "voice_energy", "connections", "bytes_out"),
        markers_missing=(),
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def solitary_context_result() -> ContextResult:
    """Create a solitary_digital context result with high confidence."""
    return ContextResult(
        active_context="solitary_digital",
        confidence_scores={"solitary_digital": 0.90, "neutral": 0.10},
        raw_scores={"solitary_digital": 0.92, "neutral": 0.08},
        smoothed=True,
        markers_used=("speech_activity", "voice_energy", "connections", "bytes_out"),
        markers_missing=(),
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def neutral_context_result() -> ContextResult:
    """Create a neutral (unknown) context result."""
    return ContextResult(
        active_context="neutral",
        confidence_scores={"neutral": 0.50},
        raw_scores={"neutral": 0.50},
        smoothed=False,
        markers_used=(),
        markers_missing=(),
        timestamp=datetime.now(UTC),
    )


@pytest.fixture
def zero_confidence_social_result() -> ContextResult:
    """Create a solitary_digital context result with zero confidence."""
    return ContextResult(
        active_context="solitary_digital",
        confidence_scores={"solitary_digital": 0.0},
        raw_scores={"solitary_digital": 0.0},
        smoothed=False,
        markers_used=(),
        markers_missing=("speech_activity", "voice_energy"),
        timestamp=datetime.now(UTC),
    )


# ============================================================================
# Test AdjustedIndicatorWeights Dataclass
# ============================================================================


def test_adjusted_indicator_weights_creation() -> None:
    """Test creation of AdjustedIndicatorWeights dataclass."""
    timestamp = datetime.now(UTC)
    adjusted = AdjustedIndicatorWeights(
        indicator_name="social_withdrawal",
        biomarker_weights={"speech_activity": 0.4, "connections": 0.6},
        base_weights={"speech_activity": 0.5, "connections": 0.5},
        multipliers_applied={"speech_activity": 1.2, "connections": 0.8},
        context_applied="solitary_digital",
        confidence=0.85,
        timestamp=timestamp,
    )

    assert adjusted.indicator_name == "social_withdrawal"
    assert adjusted.biomarker_weights == {"speech_activity": 0.4, "connections": 0.6}
    assert adjusted.base_weights == {"speech_activity": 0.5, "connections": 0.5}
    assert adjusted.context_applied == "solitary_digital"
    assert adjusted.confidence == 0.85
    assert adjusted.timestamp == timestamp


def test_adjusted_indicator_weights_immutability() -> None:
    """Test that AdjustedIndicatorWeights is immutable (frozen)."""
    adjusted = AdjustedIndicatorWeights(
        indicator_name="test",
        biomarker_weights={},
        base_weights={},
        multipliers_applied={},
        context_applied="solitary_digital",
        confidence=0.5,
        timestamp=datetime.now(UTC),
    )

    with pytest.raises(AttributeError):
        adjusted.indicator_name = "modified"  # type: ignore


# ============================================================================
# Test Base Weight Extraction
# ============================================================================


def test_base_weight_extraction(
    base_indicator_config: IndicatorConfig,
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test that base weights are correctly extracted from IndicatorConfig."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    assert adjusted.base_weights == {
        "speech_activity": 0.30,
        "voice_energy": 0.20,
        "connections": 0.25,
        "bytes_out": 0.25,
    }


# ============================================================================
# Test Context Multiplier Lookup
# ============================================================================


def test_context_multiplier_lookup_existing_context(
    base_indicator_config: IndicatorConfig,
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test multiplier lookup when context exists in config."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    # With confidence=0.85, effective multiplier formula: 1.0 + (mult - 1.0) * 0.85
    # speech_activity: 1.0 + (0.7 - 1.0) * 0.85 = 0.745
    # voice_energy: 1.0 + (0.8 - 1.0) * 0.85 = 0.83
    # connections: 1.0 + (1.5 - 1.0) * 0.85 = 1.425
    # bytes_out: 1.0 + (1.4 - 1.0) * 0.85 = 1.34

    assert adjusted.multipliers_applied["speech_activity"] == pytest.approx(
        0.745, abs=1e-6
    )
    assert adjusted.multipliers_applied["voice_energy"] == pytest.approx(
        0.83, abs=1e-6
    )
    assert adjusted.multipliers_applied["connections"] == pytest.approx(1.425, abs=1e-6)
    assert adjusted.multipliers_applied["bytes_out"] == pytest.approx(1.34, abs=1e-6)


def test_fallback_when_context_not_in_mapping(
    base_indicator_config: IndicatorConfig,
    neutral_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test fallback to multiplier=1.0 when context is unknown."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=neutral_context_result,
        config=analysis_config,
    )

    # All multipliers should be 1.0 (neutral)
    for multiplier in adjusted.multipliers_applied.values():
        assert multiplier == pytest.approx(1.0, abs=1e-6)

    # Weights should remain unchanged (normalized base weights)
    assert adjusted.biomarker_weights == adjusted.base_weights


def test_fallback_when_biomarker_not_in_context_mapping(
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test fallback to multiplier=1.0 when biomarker not in context mapping."""
    # Create indicator with a biomarker not in context_weights
    indicator_config = IndicatorConfig(
        biomarkers={
            "speech_activity": BiomarkerWeight(
                weight=0.40, direction="higher_is_worse"
            ),
            "unknown_biomarker": BiomarkerWeight(
                weight=0.60, direction="higher_is_worse"
            ),
        }
    )

    adjusted = adjust_biomarker_weights(
        indicator_name="test_indicator",
        indicator_config=indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    # speech_activity should have adjusted multiplier
    assert adjusted.multipliers_applied["speech_activity"] == pytest.approx(
        0.745, abs=1e-6
    )

    # unknown_biomarker should default to 1.0
    assert adjusted.multipliers_applied["unknown_biomarker"] == pytest.approx(
        1.0, abs=1e-6
    )


# ============================================================================
# Test Confidence Blending
# ============================================================================


def test_confidence_blending_zero_confidence(
    base_indicator_config: IndicatorConfig,
    zero_confidence_social_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test confidence blending with confidence=0.0 (no effect)."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=zero_confidence_social_result,
        config=analysis_config,
    )

    # All effective multipliers should be 1.0 (no adjustment)
    for multiplier in adjusted.multipliers_applied.values():
        assert multiplier == pytest.approx(1.0, abs=1e-6)


def test_confidence_blending_full_confidence(
    base_indicator_config: IndicatorConfig,
    analysis_config: AnalysisConfig,
) -> None:
    """Test confidence blending with confidence=1.0 (full effect)."""
    full_confidence_result = ContextResult(
        active_context="solitary_digital",
        confidence_scores={"solitary_digital": 1.0},
        raw_scores={"solitary_digital": 1.0},
        smoothed=True,
        markers_used=("speech_activity", "voice_energy"),
        markers_missing=(),
        timestamp=datetime.now(UTC),
    )

    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=full_confidence_result,
        config=analysis_config,
    )

    # Effective multipliers should equal raw multipliers from config
    assert adjusted.multipliers_applied["speech_activity"] == pytest.approx(
        0.7, abs=1e-6
    )
    assert adjusted.multipliers_applied["voice_energy"] == pytest.approx(0.8, abs=1e-6)
    assert adjusted.multipliers_applied["connections"] == pytest.approx(1.5, abs=1e-6)
    assert adjusted.multipliers_applied["bytes_out"] == pytest.approx(1.4, abs=1e-6)


def test_confidence_blending_intermediate_values(
    base_indicator_config: IndicatorConfig,
    analysis_config: AnalysisConfig,
) -> None:
    """Test confidence blending with intermediate confidence values."""
    half_confidence_result = ContextResult(
        active_context="solitary_digital",
        confidence_scores={"solitary_digital": 0.5},
        raw_scores={"solitary_digital": 0.5},
        smoothed=True,
        markers_used=("speech_activity",),
        markers_missing=(),
        timestamp=datetime.now(UTC),
    )

    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=half_confidence_result,
        config=analysis_config,
    )

    # With confidence=0.5, effective multiplier: 1.0 + (mult - 1.0) * 0.5
    # speech_activity: 1.0 + (0.7 - 1.0) * 0.5 = 0.85
    # voice_energy: 1.0 + (0.8 - 1.0) * 0.5 = 0.9
    # connections: 1.0 + (1.5 - 1.0) * 0.5 = 1.25
    # bytes_out: 1.0 + (1.4 - 1.0) * 0.5 = 1.2

    assert adjusted.multipliers_applied["speech_activity"] == pytest.approx(
        0.85, abs=1e-6
    )
    assert adjusted.multipliers_applied["voice_energy"] == pytest.approx(0.9, abs=1e-6)
    assert adjusted.multipliers_applied["connections"] == pytest.approx(1.25, abs=1e-6)
    assert adjusted.multipliers_applied["bytes_out"] == pytest.approx(1.2, abs=1e-6)


# ============================================================================
# Test Weight Normalization
# ============================================================================


def test_weight_normalization_always_sums_to_one(
    base_indicator_config: IndicatorConfig,
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test that adjusted weights always sum to 1.0."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    total_weight = sum(adjusted.biomarker_weights.values())
    assert total_weight == pytest.approx(1.0, abs=1e-6)


def test_weight_normalization_multiple_scenarios(
    analysis_config: AnalysisConfig,
) -> None:
    """Test normalization across different scenarios."""
    scenarios = [
        # Different base weight distributions
        {
            "speech_activity": BiomarkerWeight(weight=0.7, direction="higher_is_worse"),
            "connections": BiomarkerWeight(weight=0.3, direction="lower_is_worse"),
        },
        {
            "speech_activity": BiomarkerWeight(
                weight=0.25, direction="higher_is_worse"
            ),
            "voice_energy": BiomarkerWeight(weight=0.25, direction="higher_is_worse"),
            "connections": BiomarkerWeight(weight=0.25, direction="lower_is_worse"),
            "bytes_out": BiomarkerWeight(weight=0.25, direction="lower_is_worse"),
        },
    ]

    context_result = ContextResult(
        active_context="solitary_digital",
        confidence_scores={"solitary_digital": 0.75},
        raw_scores={"solitary_digital": 0.75},
        smoothed=True,
        markers_used=("speech_activity",),
        markers_missing=(),
        timestamp=datetime.now(UTC),
    )

    for biomarkers in scenarios:
        indicator_config = IndicatorConfig(biomarkers=biomarkers)
        adjusted = adjust_biomarker_weights(
            indicator_name="test",
            indicator_config=indicator_config,
            context_result=context_result,
            config=analysis_config,
        )

        total_weight = sum(adjusted.biomarker_weights.values())
        assert total_weight == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# Test Context-Specific Weight Adjustments
# ============================================================================


def test_solitary_digital_context_increases_network_weights_via_social_fixture(
    base_indicator_config: IndicatorConfig,
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test that solitary_digital context increases network biomarker weights."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    # Network biomarkers should have higher weight than speech biomarkers
    speech_weight = (
        adjusted.biomarker_weights["speech_activity"]
        + adjusted.biomarker_weights["voice_energy"]
    )
    network_weight = (
        adjusted.biomarker_weights["connections"]
        + adjusted.biomarker_weights["bytes_out"]
    )

    assert network_weight > speech_weight
    assert network_weight > 0.5  # Should be more than half


def test_solitary_digital_context_increases_network_weights(
    base_indicator_config: IndicatorConfig,
    solitary_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test that solitary_digital context increases network biomarker weights."""
    adjusted = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=solitary_context_result,
        config=analysis_config,
    )

    # Network biomarkers should have higher weight than speech biomarkers
    speech_weight = (
        adjusted.biomarker_weights["speech_activity"]
        + adjusted.biomarker_weights["voice_energy"]
    )
    network_weight = (
        adjusted.biomarker_weights["connections"]
        + adjusted.biomarker_weights["bytes_out"]
    )

    assert network_weight > speech_weight
    assert network_weight > 0.5  # Should be more than half


# ============================================================================
# Test ContextWeightAdjuster Class
# ============================================================================


def test_context_weight_adjuster_initialization(
    analysis_config: AnalysisConfig,
) -> None:
    """Test ContextWeightAdjuster initialization."""
    adjuster = ContextWeightAdjuster(analysis_config)

    assert adjuster.config == analysis_config
    assert adjuster._logger is not None


def test_context_weight_adjuster_adjust_all_indicators(
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test adjust_all_indicators method."""
    adjuster = ContextWeightAdjuster(analysis_config)
    adjusted_indicators = adjuster.adjust_all_indicators(social_context_result)

    # Should have one result for each indicator
    assert len(adjusted_indicators) == len(analysis_config.indicators)
    assert "social_withdrawal" in adjusted_indicators

    # Each result should be properly adjusted
    adjusted = adjusted_indicators["social_withdrawal"]
    assert isinstance(adjusted, AdjustedIndicatorWeights)
    assert adjusted.context_applied == "solitary_digital"
    assert sum(adjusted.biomarker_weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_context_weight_adjuster_multiple_indicators(
    social_context_result: ContextResult,
) -> None:
    """Test adjuster with multiple indicators."""
    config = AnalysisConfig(
        indicators={
            "social_withdrawal": IndicatorConfig(
                biomarkers={
                    "speech_activity": BiomarkerWeight(
                        weight=0.5, direction="higher_is_worse"
                    ),
                    "connections": BiomarkerWeight(
                        weight=0.5, direction="lower_is_worse"
                    ),
                }
            ),
            "diminished_interest": IndicatorConfig(
                biomarkers={
                    "voice_energy": BiomarkerWeight(
                        weight=0.4, direction="higher_is_worse"
                    ),
                    "bytes_out": BiomarkerWeight(
                        weight=0.6, direction="lower_is_worse"
                    ),
                }
            ),
        },
        context_weights={
            "solitary_digital": {
                "speech_activity": 1.5,
                "voice_energy": 1.3,
                "connections": 0.7,
                "bytes_out": 0.8,
            }
        },
    )

    adjuster = ContextWeightAdjuster(config)
    adjusted_indicators = adjuster.adjust_all_indicators(social_context_result)

    assert len(adjusted_indicators) == 2
    assert "social_withdrawal" in adjusted_indicators
    assert "diminished_interest" in adjusted_indicators

    # Both should be properly normalized
    for adjusted in adjusted_indicators.values():
        total = sum(adjusted.biomarker_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# Test Determinism
# ============================================================================


def test_determinism_same_inputs_same_outputs(
    base_indicator_config: IndicatorConfig,
    social_context_result: ContextResult,
    analysis_config: AnalysisConfig,
) -> None:
    """Test that same inputs produce same outputs (determinism)."""
    # Run adjustment twice with identical inputs
    adjusted1 = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    adjusted2 = adjust_biomarker_weights(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        context_result=social_context_result,
        config=analysis_config,
    )

    # Results should be identical
    assert adjusted1.indicator_name == adjusted2.indicator_name
    assert adjusted1.context_applied == adjusted2.context_applied
    assert adjusted1.confidence == adjusted2.confidence

    for biomarker in adjusted1.biomarker_weights:
        assert adjusted1.biomarker_weights[biomarker] == pytest.approx(
            adjusted2.biomarker_weights[biomarker], abs=1e-9
        )
        assert adjusted1.base_weights[biomarker] == pytest.approx(
            adjusted2.base_weights[biomarker], abs=1e-9
        )
        assert adjusted1.multipliers_applied[biomarker] == pytest.approx(
            adjusted2.multipliers_applied[biomarker], abs=1e-9
        )
