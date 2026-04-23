"""Tests for indicator score computation using FASL formula with context weights."""

from datetime import UTC, datetime

import pytest

from src.core.config import AnalysisConfig, BiomarkerWeight, IndicatorConfig
from src.core.context.evaluator import ContextResult
from src.core.context.weights import AdjustedIndicatorWeights
from src.core.indicator_computation import (
    BiomarkerContribution,
    DailyIndicatorScore,
    IndicatorComputer,
    IndicatorScore,
    _calculate_data_reliability,
    compute_all_indicators,
    compute_indicator,
)
from src.core.processors.biomarker_processor import (
    BaselineStats,
    BiomarkerMembership,
    DailyBiomarkerMembership,
)


# ============================================================================
# Fixtures
# ============================================================================


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
        },
        min_biomarkers=2,
    )


@pytest.fixture
def analysis_config(base_indicator_config: IndicatorConfig) -> AnalysisConfig:
    """Create a test analysis configuration with context weights."""
    return AnalysisConfig(
        indicators={
            "social_withdrawal": base_indicator_config,
            "diminished_interest": IndicatorConfig(
                biomarkers={
                    "speech_rate": BiomarkerWeight(
                        weight=0.40, direction="lower_is_worse"
                    ),
                    "activity_level": BiomarkerWeight(
                        weight=0.60, direction="lower_is_worse"
                    ),
                }
            ),
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
def baseline_stats() -> BaselineStats:
    """Create test baseline stats."""
    return BaselineStats(
        mean=0.5,
        std=0.2,
        data_points=30,
        source="user",
    )


@pytest.fixture
def biomarker_memberships(baseline_stats: BaselineStats) -> dict[str, BiomarkerMembership]:
    """Create test biomarker memberships with varied values."""
    timestamp = datetime.now(UTC)
    return {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=0.75,
            z_score=1.25,
            raw_value=0.75,
            baseline=baseline_stats,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "voice_energy": BiomarkerMembership(
            name="voice_energy",
            membership=0.60,
            z_score=0.5,
            raw_value=0.60,
            baseline=baseline_stats,
            data_points_used=10,
            data_quality=0.85,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "connections": BiomarkerMembership(
            name="connections",
            membership=0.80,
            z_score=1.5,
            raw_value=0.80,
            baseline=baseline_stats,
            data_points_used=10,
            data_quality=0.95,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "bytes_out": BiomarkerMembership(
            name="bytes_out",
            membership=0.40,
            z_score=-0.5,
            raw_value=0.40,
            baseline=baseline_stats,
            data_points_used=10,
            data_quality=0.80,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
    }


@pytest.fixture
def adjusted_weights() -> AdjustedIndicatorWeights:
    """Create test adjusted weights for social context."""
    return AdjustedIndicatorWeights(
        indicator_name="social_withdrawal",
        biomarker_weights={
            "speech_activity": 0.36,
            "voice_energy": 0.21,
            "connections": 0.16,
            "bytes_out": 0.27,
        },
        base_weights={
            "speech_activity": 0.30,
            "voice_energy": 0.20,
            "connections": 0.25,
            "bytes_out": 0.25,
        },
        multipliers_applied={
            "speech_activity": 1.5,
            "voice_energy": 1.3,
            "connections": 0.7,
            "bytes_out": 0.8,
        },
        context_applied="solitary_digital",
        confidence=0.85,
        timestamp=datetime.now(UTC),
    )


# ============================================================================
# Test BiomarkerContribution Dataclass (AC4)
# ============================================================================


def test_biomarker_contribution_creation() -> None:
    """Test creation of BiomarkerContribution dataclass."""
    contrib = BiomarkerContribution(
        name="speech_activity",
        membership=0.75,
        direction="pro",
        base_weight=0.30,
        context_multiplier=1.5,
        adjusted_weight=0.36,
        contribution=0.27,
    )

    assert contrib.name == "speech_activity"
    assert contrib.membership == 0.75
    assert contrib.direction == "pro"
    assert contrib.base_weight == 0.30
    assert contrib.context_multiplier == 1.5
    assert contrib.adjusted_weight == 0.36
    assert contrib.contribution == 0.27


def test_biomarker_contribution_immutability() -> None:
    """Test that BiomarkerContribution is immutable (frozen)."""
    contrib = BiomarkerContribution(
        name="test",
        membership=0.5,
        direction="pro",
        base_weight=0.5,
        context_multiplier=1.0,
        adjusted_weight=0.5,
        contribution=0.25,
    )

    with pytest.raises(AttributeError):
        contrib.name = "modified"  # type: ignore


def test_biomarker_contribution_direction_values() -> None:
    """Test valid direction values for BiomarkerContribution."""
    pro_contrib = BiomarkerContribution(
        name="test",
        membership=0.5,
        direction="pro",
        base_weight=0.5,
        context_multiplier=1.0,
        adjusted_weight=0.5,
        contribution=0.25,
    )
    assert pro_contrib.direction == "pro"

    contra_contrib = BiomarkerContribution(
        name="test",
        membership=0.5,
        direction="contra",
        base_weight=0.5,
        context_multiplier=1.0,
        adjusted_weight=0.5,
        contribution=0.25,
    )
    assert contra_contrib.direction == "contra"


# ============================================================================
# Test IndicatorScore Dataclass (AC3)
# ============================================================================


def test_indicator_score_creation() -> None:
    """Test creation of IndicatorScore dataclass."""
    timestamp = datetime.now(UTC)
    contrib = BiomarkerContribution(
        name="speech_activity",
        membership=0.75,
        direction="pro",
        base_weight=0.30,
        context_multiplier=1.5,
        adjusted_weight=0.36,
        contribution=0.27,
    )

    score = IndicatorScore(
        indicator_name="social_withdrawal",
        daily_likelihood=0.45,
        contributions={"speech_activity": contrib},
        biomarkers_used=("speech_activity",),
        biomarkers_missing=("voice_energy",),
        data_reliability_score=0.8,
        context_applied="solitary_digital",
        context_confidence=0.85,
        weights_before_context={"speech_activity": 0.30},
        weights_after_context={"speech_activity": 0.36},
        timestamp=timestamp,
    )

    assert score.indicator_name == "social_withdrawal"
    assert score.daily_likelihood == 0.45
    assert len(score.contributions) == 1
    assert score.biomarkers_used == ("speech_activity",)
    assert score.biomarkers_missing == ("voice_energy",)
    assert score.data_reliability_score == 0.8
    assert score.context_applied == "solitary_digital"
    assert score.context_confidence == 0.85
    assert score.timestamp == timestamp


def test_indicator_score_immutability() -> None:
    """Test that IndicatorScore is immutable (frozen)."""
    score = IndicatorScore(
        indicator_name="test",
        daily_likelihood=0.5,
        contributions={},
        biomarkers_used=(),
        biomarkers_missing=(),
        data_reliability_score=0.5,
        context_applied="neutral",
        context_confidence=0.5,
        weights_before_context={},
        weights_after_context={},
        timestamp=datetime.now(UTC),
    )

    with pytest.raises(AttributeError):
        score.indicator_name = "modified"  # type: ignore


def test_indicator_score_daily_likelihood_bounds() -> None:
    """Test that daily_likelihood is expected to be in [0,1]."""
    timestamp = datetime.now(UTC)

    # Valid likelihood values
    for likelihood in [0.0, 0.5, 1.0]:
        score = IndicatorScore(
            indicator_name="test",
            daily_likelihood=likelihood,
            contributions={},
            biomarkers_used=(),
            biomarkers_missing=(),
            data_reliability_score=0.5,
            context_applied="neutral",
            context_confidence=0.5,
            weights_before_context={},
            weights_after_context={},
            timestamp=timestamp,
        )
        assert score.daily_likelihood == likelihood


# ============================================================================
# Test Direction Modifier (AC6, AC7)
# ============================================================================


def test_direction_modifier_higher_is_worse(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test direction modifier: +1 for higher_is_worse (pro) biomarkers."""
    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    # Only higher_is_worse biomarkers
    memberships = {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=0.8,  # High membership
            z_score=1.5,
            raw_value=0.8,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "voice_energy": BiomarkerMembership(
            name="voice_energy",
            membership=0.6,
            z_score=0.5,
            raw_value=0.6,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.85,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
    }

    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # Higher_is_worse: high mu should contribute positively
    # speech_activity contribution = 0.36 * max(0, +1 * 0.8) = 0.288
    # voice_energy contribution = 0.21 * max(0, +1 * 0.6) = 0.126
    assert result.contributions["speech_activity"].direction == "pro"
    assert result.contributions["voice_energy"].direction == "pro"
    assert result.contributions["speech_activity"].contribution > 0
    assert result.contributions["voice_energy"].contribution > 0


def test_direction_modifier_lower_is_worse(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test direction modifier for lower_is_worse (contra) biomarkers.

    For lower_is_worse: contribution = weight * (1 - mu)
    This means high mu (not concerning) -> low contribution
    And low mu (concerning) -> high contribution
    """
    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    # Only lower_is_worse biomarkers
    memberships = {
        "connections": BiomarkerMembership(
            name="connections",
            membership=0.8,  # High membership = NOT concerning for lower_is_worse
            z_score=1.5,
            raw_value=0.8,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "bytes_out": BiomarkerMembership(
            name="bytes_out",
            membership=0.6,
            z_score=0.5,
            raw_value=0.6,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.85,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
    }

    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # Lower_is_worse biomarkers use (1 - mu) for directed contribution
    assert result.contributions["connections"].direction == "contra"
    assert result.contributions["bytes_out"].direction == "contra"

    # connections: w=0.16, mu=0.8 -> contribution = 0.16 * (1 - 0.8) = 0.16 * 0.2 = 0.032
    # bytes_out: w=0.27, mu=0.6 -> contribution = 0.27 * (1 - 0.6) = 0.27 * 0.4 = 0.108
    assert result.contributions["connections"].contribution == pytest.approx(0.032, abs=0.01)
    assert result.contributions["bytes_out"].contribution == pytest.approx(0.108, abs=0.01)


def test_clipping_negative_contributions(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
    biomarker_memberships: dict[str, BiomarkerMembership],
) -> None:
    """Test that negative contributions are clipped to zero (AC7)."""
    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=biomarker_memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # All contributions should be >= 0 due to clipping
    for contrib in result.contributions.values():
        assert contrib.contribution >= 0.0


# ============================================================================
# Test FASL Formula (AC2)
# ============================================================================


def test_fasl_formula_all_biomarkers_present(
    base_indicator_config: IndicatorConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test FASL formula with all biomarkers present."""
    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=biomarker_memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # All biomarkers should be used
    assert len(result.biomarkers_used) == 4
    assert len(result.biomarkers_missing) == 0

    # Daily likelihood should be in [0, 1]
    assert 0.0 <= result.daily_likelihood <= 1.0

    # Verify FASL formula manually:
    # higher_is_worse (pro): speech_activity, voice_energy -> contribution = w * mu
    # lower_is_worse (contra): connections, bytes_out -> contribution = w * (1 - mu)
    #
    # speech_activity: w=0.36, mu=0.75 -> contribution = 0.36 * 0.75 = 0.27
    # voice_energy: w=0.21, mu=0.60 -> contribution = 0.21 * 0.60 = 0.126
    # connections: w=0.16, mu=0.80 -> contribution = 0.16 * (1 - 0.80) = 0.16 * 0.20 = 0.032
    # bytes_out: w=0.27, mu=0.40 -> contribution = 0.27 * (1 - 0.40) = 0.27 * 0.60 = 0.162
    #
    # numerator = 0.27 + 0.126 + 0.032 + 0.162 = 0.59
    # denominator = 0.36 + 0.21 + 0.16 + 0.27 = 1.0
    # L_k = 0.59 / 1.0 = 0.59

    expected_likelihood = pytest.approx(0.59, abs=0.01)
    assert result.daily_likelihood == expected_likelihood


def test_fasl_formula_zero_denominator() -> None:
    """Test FASL formula handles zero denominator (no weights)."""
    # Create indicator config with zero weights (edge case)
    indicator_config = IndicatorConfig(
        biomarkers={
            "test_bio": BiomarkerWeight(weight=1.0, direction="higher_is_worse"),
        }
    )

    # Adjusted weights with zero weight
    adjusted = AdjustedIndicatorWeights(
        indicator_name="test",
        biomarker_weights={"test_bio": 0.0},  # Zero weight
        base_weights={"test_bio": 1.0},
        multipliers_applied={"test_bio": 0.0},
        context_applied="neutral",
        confidence=0.0,
        timestamp=datetime.now(UTC),
    )

    memberships = {
        "test_bio": BiomarkerMembership(
            name="test_bio",
            membership=0.8,
            z_score=1.5,
            raw_value=0.8,
            baseline=BaselineStats(mean=0.5, std=0.2, data_points=10, source="user"),
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=datetime.now(UTC),
        ),
    }

    result = compute_indicator(
        indicator_name="test",
        indicator_config=indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted,
    )

    # Should handle zero denominator gracefully
    assert result.daily_likelihood == 0.0


# ============================================================================
# Test Biomarker Dropout (AC8, AC9)
# ============================================================================


def test_biomarker_dropout_some_missing(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test biomarker dropout: computation continues with available biomarkers (AC8)."""
    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    # Only 2 of 4 biomarkers available
    partial_memberships = {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=0.75,
            z_score=1.25,
            raw_value=0.75,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "voice_energy": BiomarkerMembership(
            name="voice_energy",
            membership=0.60,
            z_score=0.5,
            raw_value=0.60,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.85,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        # connections and bytes_out missing
    }

    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=partial_memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # Should use available biomarkers
    assert len(result.biomarkers_used) == 2
    assert "speech_activity" in result.biomarkers_used
    assert "voice_energy" in result.biomarkers_used

    # Should track missing biomarkers
    assert len(result.biomarkers_missing) == 2
    assert "connections" in result.biomarkers_missing
    assert "bytes_out" in result.biomarkers_missing

    # Should still compute valid likelihood
    assert 0.0 <= result.daily_likelihood <= 1.0


def test_biomarker_dropout_none_membership_value(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test biomarker dropout when membership value is None."""
    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    memberships = {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=0.75,
            z_score=1.25,
            raw_value=0.75,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "voice_energy": BiomarkerMembership(
            name="voice_energy",
            membership=None,  # Unavailable
            z_score=None,
            raw_value=0.0,
            baseline=None,
            data_points_used=0,
            data_quality=0.0,
            membership_function_used="none",
            timestamp=timestamp,
        ),
    }

    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # voice_energy should be in missing
    assert "voice_energy" in result.biomarkers_missing
    assert "speech_activity" in result.biomarkers_used


def test_graceful_degradation_all_missing(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test graceful degradation when all biomarkers are missing (AC9)."""
    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships={},  # No biomarkers
        context_adjusted_weights=adjusted_weights,
    )

    # Should return valid result with zero likelihood
    assert result.daily_likelihood == 0.0
    assert len(result.biomarkers_used) == 0
    assert len(result.biomarkers_missing) == 4
    assert result.data_reliability_score == 0.0


# ============================================================================
# Test Minimum Biomarker Requirement (AC10)
# ============================================================================


def test_minimum_biomarker_requirement(
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test minimum biomarker requirement warning (AC10)."""
    # Indicator requiring 3 biomarkers
    indicator_config = IndicatorConfig(
        biomarkers={
            "speech_activity": BiomarkerWeight(
                weight=0.5, direction="higher_is_worse"
            ),
            "voice_energy": BiomarkerWeight(weight=0.5, direction="higher_is_worse"),
        },
        min_biomarkers=3,  # Requires 3 but only 2 defined
    )

    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    memberships = {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=0.75,
            z_score=1.25,
            raw_value=0.75,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
    }

    # Should still compute but with warning (logged)
    result = compute_indicator(
        indicator_name="test",
        indicator_config=indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=AdjustedIndicatorWeights(
            indicator_name="test",
            biomarker_weights={"speech_activity": 0.5, "voice_energy": 0.5},
            base_weights={"speech_activity": 0.5, "voice_energy": 0.5},
            multipliers_applied={"speech_activity": 1.0, "voice_energy": 1.0},
            context_applied="neutral",
            confidence=0.5,
            timestamp=timestamp,
        ),
    )

    assert len(result.biomarkers_used) == 1  # Less than min_biomarkers


# ============================================================================
# Test Data Reliability Calculation (AC11)
# ============================================================================


def test_data_reliability_calculation_full_coverage(
    biomarker_memberships: dict[str, BiomarkerMembership],
) -> None:
    """Test data reliability calculation with full biomarker coverage."""
    biomarkers_used = list(biomarker_memberships.keys())
    biomarkers_expected = biomarkers_used.copy()

    data_reliability = _calculate_data_reliability(
        biomarkers_used, biomarkers_expected, biomarker_memberships
    )

    # Full coverage (1.0) * 0.6 + avg_quality * 0.4
    # avg_quality = (0.9 + 0.85 + 0.95 + 0.80) / 4 = 0.875
    # data_reliability = 1.0 * 0.6 + 0.875 * 0.4 = 0.6 + 0.35 = 0.95
    assert data_reliability == pytest.approx(0.95, abs=0.01)


def test_data_reliability_calculation_partial_coverage(
    biomarker_memberships: dict[str, BiomarkerMembership],
) -> None:
    """Test data reliability calculation with partial biomarker coverage."""
    biomarkers_used = ["speech_activity", "voice_energy"]
    biomarkers_expected = ["speech_activity", "voice_energy", "connections", "bytes_out"]

    data_reliability = _calculate_data_reliability(
        biomarkers_used, biomarkers_expected, biomarker_memberships
    )

    # Coverage = 2/4 = 0.5
    # avg_quality = (0.9 + 0.85) / 2 = 0.875
    # data_reliability = 0.5 * 0.6 + 0.875 * 0.4 = 0.3 + 0.35 = 0.65
    assert data_reliability == pytest.approx(0.65, abs=0.01)


def test_data_reliability_calculation_no_coverage() -> None:
    """Test data reliability calculation with no biomarkers."""
    data_reliability = _calculate_data_reliability([], ["a", "b", "c"], {})

    # Coverage = 0/3 = 0
    # data_reliability = 0 * 0.6 = 0
    assert data_reliability == 0.0


def test_data_reliability_calculation_no_expected() -> None:
    """Test data reliability calculation with no expected biomarkers."""
    data_reliability = _calculate_data_reliability([], [], {})
    assert data_reliability == 0.0


# ============================================================================
# Test compute_all_indicators (AC5)
# ============================================================================


def test_compute_all_indicators(
    analysis_config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    social_context_result: ContextResult,
) -> None:
    """Test compute_all_indicators returns scores for all indicators."""
    results = compute_all_indicators(
        config=analysis_config,
        biomarker_memberships=biomarker_memberships,
        context_result=social_context_result,
    )

    # Should have results for all configured indicators
    assert len(results) == 2
    assert "social_withdrawal" in results
    assert "diminished_interest" in results

    # Each result should be an IndicatorScore
    for indicator_name, score in results.items():
        assert isinstance(score, IndicatorScore)
        assert score.indicator_name == indicator_name
        assert 0.0 <= score.daily_likelihood <= 1.0


def test_compute_all_indicators_context_integration(
    analysis_config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    social_context_result: ContextResult,
) -> None:
    """Test that compute_all_indicators properly integrates context."""
    results = compute_all_indicators(
        config=analysis_config,
        biomarker_memberships=biomarker_memberships,
        context_result=social_context_result,
    )

    # All results should have the same context
    for score in results.values():
        assert score.context_applied == "solitary_digital"


# ============================================================================
# Test IndicatorComputer Class (AC1, AC12, AC13)
# ============================================================================


def test_indicator_computer_initialization(
    analysis_config: AnalysisConfig,
) -> None:
    """Test IndicatorComputer initialization."""
    computer = IndicatorComputer(analysis_config)

    assert computer.config == analysis_config
    assert computer._logger is not None


def test_indicator_computer_compute_single(
    analysis_config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    social_context_result: ContextResult,
) -> None:
    """Test IndicatorComputer.compute_single method."""
    computer = IndicatorComputer(analysis_config)

    result = computer.compute_single(
        indicator_name="social_withdrawal",
        biomarker_memberships=biomarker_memberships,
        context_result=social_context_result,
    )

    assert isinstance(result, IndicatorScore)
    assert result.indicator_name == "social_withdrawal"
    assert 0.0 <= result.daily_likelihood <= 1.0


def test_indicator_computer_compute_single_unknown_indicator(
    analysis_config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    social_context_result: ContextResult,
) -> None:
    """Test IndicatorComputer.compute_single with unknown indicator raises KeyError."""
    computer = IndicatorComputer(analysis_config)

    with pytest.raises(KeyError, match="unknown_indicator"):
        computer.compute_single(
            indicator_name="unknown_indicator",
            biomarker_memberships=biomarker_memberships,
            context_result=social_context_result,
        )


def test_indicator_computer_compute_all(
    analysis_config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    social_context_result: ContextResult,
) -> None:
    """Test IndicatorComputer.compute_all method."""
    computer = IndicatorComputer(analysis_config)

    results = computer.compute_all(
        biomarker_memberships=biomarker_memberships,
        context_result=social_context_result,
    )

    assert len(results) == 2
    assert "social_withdrawal" in results
    assert "diminished_interest" in results


# ============================================================================
# Test Determinism (AC13)
# ============================================================================


def test_determinism_same_inputs_same_outputs(
    base_indicator_config: IndicatorConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test that same inputs produce same outputs (determinism)."""
    result1 = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=biomarker_memberships,
        context_adjusted_weights=adjusted_weights,
    )

    result2 = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=biomarker_memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # Results should be identical
    assert result1.indicator_name == result2.indicator_name
    assert result1.daily_likelihood == result2.daily_likelihood
    assert result1.biomarkers_used == result2.biomarkers_used
    assert result1.biomarkers_missing == result2.biomarkers_missing
    assert result1.data_reliability_score == result2.data_reliability_score
    assert result1.context_applied == result2.context_applied

    # Contributions should be identical
    for name in result1.contributions:
        assert result1.contributions[name].contribution == pytest.approx(
            result2.contributions[name].contribution, abs=1e-9
        )


def test_determinism_indicator_computer(
    analysis_config: AnalysisConfig,
    biomarker_memberships: dict[str, BiomarkerMembership],
    social_context_result: ContextResult,
) -> None:
    """Test IndicatorComputer determinism."""
    computer = IndicatorComputer(analysis_config)

    results1 = computer.compute_all(biomarker_memberships, social_context_result)
    results2 = computer.compute_all(biomarker_memberships, social_context_result)

    for indicator in results1:
        assert results1[indicator].daily_likelihood == pytest.approx(
            results2[indicator].daily_likelihood, abs=1e-9
        )


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_edge_case_zero_membership_values(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test handling of zero membership values."""
    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    memberships = {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=0.0,  # Zero membership
            z_score=0.0,
            raw_value=0.0,
            baseline=baseline,
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
    }

    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # Zero membership should result in zero contribution
    assert result.contributions["speech_activity"].contribution == 0.0


def test_edge_case_perfect_membership_values(
    base_indicator_config: IndicatorConfig,
    adjusted_weights: AdjustedIndicatorWeights,
) -> None:
    """Test handling of perfect (1.0) membership values."""
    timestamp = datetime.now(UTC)
    baseline = BaselineStats(mean=0.5, std=0.2, data_points=30, source="user")

    memberships = {
        "speech_activity": BiomarkerMembership(
            name="speech_activity",
            membership=1.0,  # Perfect membership
            z_score=3.0,
            raw_value=1.0,
            baseline=baseline,
            data_points_used=10,
            data_quality=1.0,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
        "voice_energy": BiomarkerMembership(
            name="voice_energy",
            membership=1.0,
            z_score=3.0,
            raw_value=1.0,
            baseline=baseline,
            data_points_used=10,
            data_quality=1.0,
            membership_function_used="sigmoid",
            timestamp=timestamp,
        ),
    }

    result = compute_indicator(
        indicator_name="social_withdrawal",
        indicator_config=base_indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted_weights,
    )

    # With higher_is_worse biomarkers at mu=1.0
    # contribution = weight * max(0, +1 * 1.0) = weight
    assert result.contributions["speech_activity"].contribution == pytest.approx(
        0.36, abs=0.01  # adjusted weight
    )


def test_edge_case_single_biomarker() -> None:
    """Test indicator with single biomarker."""
    indicator_config = IndicatorConfig(
        biomarkers={
            "single_bio": BiomarkerWeight(weight=1.0, direction="higher_is_worse"),
        }
    )

    adjusted = AdjustedIndicatorWeights(
        indicator_name="test",
        biomarker_weights={"single_bio": 1.0},
        base_weights={"single_bio": 1.0},
        multipliers_applied={"single_bio": 1.0},
        context_applied="neutral",
        confidence=0.5,
        timestamp=datetime.now(UTC),
    )

    memberships = {
        "single_bio": BiomarkerMembership(
            name="single_bio",
            membership=0.6,
            z_score=0.5,
            raw_value=0.6,
            baseline=BaselineStats(mean=0.5, std=0.2, data_points=10, source="user"),
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=datetime.now(UTC),
        ),
    }

    result = compute_indicator(
        indicator_name="test",
        indicator_config=indicator_config,
        biomarker_memberships=memberships,
        context_adjusted_weights=adjusted,
    )

    # Single biomarker: L_k = (1.0 * 0.6) / 1.0 = 0.6
    assert result.daily_likelihood == pytest.approx(0.6, abs=0.01)
    assert len(result.biomarkers_used) == 1


# ============================================================================
# Test Default Config Compatibility
# ============================================================================


def test_default_config_direction_semantics() -> None:
    """Test that default config uses semantically correct directions.

    Both higher_is_worse and lower_is_worse directions work correctly:
    - higher_is_worse: contribution = weight * mu
    - lower_is_worse: contribution = weight * (1 - mu)

    This test validates that the default config directions match clinical semantics.
    """
    from src.core.config import get_default_config

    config = get_default_config()

    # Verify config has expected indicators
    assert "social_withdrawal" in config.indicators
    assert "diminished_interest" in config.indicators
    assert "sleep_disturbance" in config.indicators

    # Verify each indicator has biomarkers with valid directions
    for indicator_name, indicator_config in config.indicators.items():
        for bio_name, bio_config in indicator_config.biomarkers.items():
            assert bio_config.direction in ("higher_is_worse", "lower_is_worse"), (
                f"Indicator '{indicator_name}' biomarker '{bio_name}' has "
                f"invalid direction '{bio_config.direction}'"
            )


def test_lower_is_worse_contributes_correctly() -> None:
    """Test that lower_is_worse biomarkers contribute positively to likelihood.

    With the (1 - mu) formula, low mu values (concerning for lower_is_worse)
    produce high contributions, enabling proper clinical interpretation.
    """
    # Create an indicator with only lower_is_worse biomarkers
    indicator_config = IndicatorConfig(
        biomarkers={
            "low_bio": BiomarkerWeight(weight=1.0, direction="lower_is_worse"),
        }
    )

    adjusted = AdjustedIndicatorWeights(
        indicator_name="test",
        biomarker_weights={"low_bio": 1.0},
        base_weights={"low_bio": 1.0},
        multipliers_applied={"low_bio": 1.0},
        context_applied="neutral",
        confidence=0.5,
        timestamp=datetime.now(UTC),
    )

    # Low mu = concerning for lower_is_worse -> should give HIGH contribution
    memberships_low_mu = {
        "low_bio": BiomarkerMembership(
            name="low_bio",
            membership=0.2,  # Low mu = concerning
            z_score=-1.5,
            raw_value=0.2,
            baseline=BaselineStats(mean=0.5, std=0.2, data_points=10, source="user"),
            data_points_used=10,
            data_quality=0.9,
            membership_function_used="sigmoid",
            timestamp=datetime.now(UTC),
        ),
    }

    result = compute_indicator(
        indicator_name="test",
        indicator_config=indicator_config,
        biomarker_memberships=memberships_low_mu,
        context_adjusted_weights=adjusted,
    )

    # contribution = 1.0 * (1 - 0.2) = 0.8, likelihood = 0.8 / 1.0 = 0.8
    assert result.daily_likelihood == pytest.approx(0.8, abs=0.01)
    assert result.contributions["low_bio"].contribution == pytest.approx(0.8, abs=0.01)


# ============================================================================
# Test DailyIndicatorScore Dataclass (Story 4.13)
# ============================================================================


def test_daily_indicator_score_creation() -> None:
    """Test creation of DailyIndicatorScore dataclass."""
    from datetime import date

    score = DailyIndicatorScore(
        date=date(2025, 1, 15),
        indicator_name="social_withdrawal",
        daily_likelihood=0.65,
        biomarkers_used=("speech_activity", "voice_energy"),
        biomarkers_missing=("connections",),
        data_reliability_score=0.85,
    )

    assert score.date == date(2025, 1, 15)
    assert score.indicator_name == "social_withdrawal"
    assert score.daily_likelihood == 0.65
    assert score.biomarkers_used == ("speech_activity", "voice_energy")
    assert score.biomarkers_missing == ("connections",)
    assert score.data_reliability_score == 0.85


def test_daily_indicator_score_immutability() -> None:
    """Test DailyIndicatorScore is immutable."""
    from datetime import date

    score = DailyIndicatorScore(
        date=date(2025, 1, 15),
        indicator_name="test",
        daily_likelihood=0.5,
        biomarkers_used=(),
        biomarkers_missing=(),
        data_reliability_score=0.5,
    )

    with pytest.raises(AttributeError):
        score.date = date(2025, 1, 16)  # type: ignore


# ============================================================================
# Test compute_daily_series (Story 4.13 Task 2)
# ============================================================================


@pytest.fixture
def daily_biomarker_memberships(baseline_stats: BaselineStats) -> dict[str, list[DailyBiomarkerMembership]]:
    """Create daily biomarker memberships over 3 days."""
    from datetime import date

    timestamp = datetime.now(UTC)

    return {
        "speech_activity": [
            DailyBiomarkerMembership(
                date=date(2025, 1, 13),
                name="speech_activity",
                membership=0.65,
                z_score=0.75,
                raw_value=0.65,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.9,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
            DailyBiomarkerMembership(
                date=date(2025, 1, 14),
                name="speech_activity",
                membership=0.75,
                z_score=1.25,
                raw_value=0.75,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.9,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
            DailyBiomarkerMembership(
                date=date(2025, 1, 15),
                name="speech_activity",
                membership=0.80,
                z_score=1.5,
                raw_value=0.80,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.9,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
        ],
        "voice_energy": [
            DailyBiomarkerMembership(
                date=date(2025, 1, 13),
                name="voice_energy",
                membership=0.55,
                z_score=0.25,
                raw_value=0.55,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.85,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
            DailyBiomarkerMembership(
                date=date(2025, 1, 14),
                name="voice_energy",
                membership=0.60,
                z_score=0.5,
                raw_value=0.60,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.85,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
            DailyBiomarkerMembership(
                date=date(2025, 1, 15),
                name="voice_energy",
                membership=0.70,
                z_score=1.0,
                raw_value=0.70,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.85,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
        ],
    }


def test_compute_daily_series_returns_series(
    analysis_config: AnalysisConfig,
    daily_biomarker_memberships: dict[str, list[DailyBiomarkerMembership]],
    social_context_result: ContextResult,
) -> None:
    """Test compute_daily_series returns time series for each indicator."""
    from datetime import date

    computer = IndicatorComputer(analysis_config)

    results = computer.compute_daily_series(
        daily_biomarker_memberships=daily_biomarker_memberships,
        context_result=social_context_result,
    )

    # Should have results for configured indicators
    assert "social_withdrawal" in results
    assert "diminished_interest" in results

    # social_withdrawal should have 3 daily scores (one per day)
    assert len(results["social_withdrawal"]) == 3

    # Verify dates are in order
    dates = [s.date for s in results["social_withdrawal"]]
    assert dates == [date(2025, 1, 13), date(2025, 1, 14), date(2025, 1, 15)]


def test_compute_daily_series_values_vary_by_day(
    analysis_config: AnalysisConfig,
    daily_biomarker_memberships: dict[str, list[DailyBiomarkerMembership]],
    social_context_result: ContextResult,
) -> None:
    """Test that daily likelihoods vary based on that day's memberships."""
    computer = IndicatorComputer(analysis_config)

    results = computer.compute_daily_series(
        daily_biomarker_memberships=daily_biomarker_memberships,
        context_result=social_context_result,
    )

    social_scores = results["social_withdrawal"]

    # Likelihoods should be different for each day
    # Day 1: lower memberships -> lower likelihood
    # Day 3: higher memberships -> higher likelihood
    day1_likelihood = social_scores[0].daily_likelihood
    day3_likelihood = social_scores[2].daily_likelihood

    # With increasing memberships, likelihood should increase
    # (for higher_is_worse biomarkers)
    assert day3_likelihood > day1_likelihood


def test_compute_daily_series_context_constant(
    analysis_config: AnalysisConfig,
    daily_biomarker_memberships: dict[str, list[DailyBiomarkerMembership]],
    social_context_result: ContextResult,
) -> None:
    """Test that context weights are applied consistently across all days."""
    computer = IndicatorComputer(analysis_config)

    results = computer.compute_daily_series(
        daily_biomarker_memberships=daily_biomarker_memberships,
        context_result=social_context_result,
    )

    # All scores should be computed with the same context
    # This is implicit in the implementation - weights are pre-computed once
    for score in results["social_withdrawal"]:
        assert isinstance(score, DailyIndicatorScore)
        assert 0.0 <= score.daily_likelihood <= 1.0


def test_compute_daily_series_empty_input(
    analysis_config: AnalysisConfig,
    social_context_result: ContextResult,
) -> None:
    """Test compute_daily_series handles empty input."""
    computer = IndicatorComputer(analysis_config)

    results = computer.compute_daily_series(
        daily_biomarker_memberships={},
        context_result=social_context_result,
    )

    assert results == {}


def test_compute_daily_series_missing_biomarkers_some_days(
    analysis_config: AnalysisConfig,
    baseline_stats: BaselineStats,
    social_context_result: ContextResult,
) -> None:
    """Test that days with partial biomarker data are handled."""
    from datetime import date

    timestamp = datetime.now(UTC)

    # Only speech_activity on day 1, both on days 2-3
    daily_memberships = {
        "speech_activity": [
            DailyBiomarkerMembership(
                date=date(2025, 1, 13),
                name="speech_activity",
                membership=0.7,
                z_score=1.0,
                raw_value=0.7,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.9,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
            DailyBiomarkerMembership(
                date=date(2025, 1, 14),
                name="speech_activity",
                membership=0.75,
                z_score=1.25,
                raw_value=0.75,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.9,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
        ],
        "voice_energy": [
            # Missing on day 1, present on day 2
            DailyBiomarkerMembership(
                date=date(2025, 1, 14),
                name="voice_energy",
                membership=0.65,
                z_score=0.75,
                raw_value=0.65,
                baseline=baseline_stats,
                data_points_used=5,
                data_quality=0.85,
                membership_function_used="sigmoid",
                timestamp=timestamp,
            ),
        ],
    }

    computer = IndicatorComputer(analysis_config)

    results = computer.compute_daily_series(
        daily_biomarker_memberships=daily_memberships,
        context_result=social_context_result,
    )

    social_scores = results["social_withdrawal"]

    # Should have scores for both days
    assert len(social_scores) == 2

    # Day 1 should have voice_energy missing
    day1 = next(s for s in social_scores if s.date == date(2025, 1, 13))
    assert "voice_energy" in day1.biomarkers_missing or "voice_energy" not in day1.biomarkers_used

    # Day 2 should have both biomarkers
    day2 = next(s for s in social_scores if s.date == date(2025, 1, 14))
    assert "speech_activity" in day2.biomarkers_used


def test_compute_daily_series_output_for_dsm_gate(
    analysis_config: AnalysisConfig,
    daily_biomarker_memberships: dict[str, list[DailyBiomarkerMembership]],
    social_context_result: ContextResult,
) -> None:
    """Test that output can be converted to DSM-Gate input format."""
    computer = IndicatorComputer(analysis_config)

    results = computer.compute_daily_series(
        daily_biomarker_memberships=daily_biomarker_memberships,
        context_result=social_context_result,
    )

    # Convert to DSM-Gate format: {indicator_name: [L_k(d1), L_k(d2), ...]}
    dsm_gate_input = {
        name: [score.daily_likelihood for score in scores]
        for name, scores in results.items()
    }

    # Should be a dict of lists of floats
    assert "social_withdrawal" in dsm_gate_input
    assert isinstance(dsm_gate_input["social_withdrawal"], list)
    assert all(isinstance(v, float) for v in dsm_gate_input["social_withdrawal"])
    assert len(dsm_gate_input["social_withdrawal"]) == 3
