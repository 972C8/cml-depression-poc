"""Indicator persistence module.

Saves computed indicators to the database with full metadata for
reproducibility and transparency audit trail.

This module provides functions to persist analysis results including:
- Individual indicator scores with computation logs
- Daily indicator scores for M-day series (Story 4.13)
- Batch saving of all indicators from an analysis run
- Analysis run metadata for traceability
"""

import logging
import uuid
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy.orm import Session

from src.core.config import AnalysisConfig
from src.core.dsm_gate import IndicatorGateResult
from src.core.indicator_computation import DailyIndicatorScore, IndicatorScore
from src.core.models.daily_summary import DailyIndicatorSummary
from src.core.models.window_models import WindowIndicator
from src.shared.models import AnalysisRun, Indicator

logger = logging.getLogger(__name__)

__all__ = [
    "save_indicator",
    "save_all_indicators",
    "save_daily_indicator_scores",
    "save_daily_summaries",
    "save_window_indicators",
    "save_analysis_run",
]


# Biomarker to modality mapping
BIOMARKER_MODALITY_MAP: dict[str, str] = {
    # Speech modality
    "speech_activity": "speech",
    "voice_energy": "speech",
    "speech_rate": "speech",
    "speech_activity_night": "speech",
    # Network modality
    "connections": "network",
    "bytes_out": "network",
    "bytes_in": "network",
    "network_variety": "network",
    # Activity modality
    "activity_level": "activity",
    # Sleep modality
    "sleep_duration": "sleep",
    "awakenings": "sleep",
}


def _extract_modalities(biomarkers_used: tuple[str, ...]) -> list[str]:
    """Extract unique modalities from biomarker names.

    Maps biomarker names to their corresponding modality types using
    BIOMARKER_MODALITY_MAP. Unknown biomarkers are mapped to "unknown".

    Args:
        biomarkers_used: Tuple of biomarker names that contributed

    Returns:
        Sorted, deduplicated list of modality names
    """
    modalities: set[str] = set()
    for biomarker in biomarkers_used:
        modality = BIOMARKER_MODALITY_MAP.get(biomarker, "unknown")
        modalities.add(modality)
    return sorted(modalities)


def _build_computation_log(
    indicator_score: IndicatorScore,
    gate_result: IndicatorGateResult | None,
) -> dict[str, Any]:
    """Build computation log for transparency audit trail.

    Creates a JSON-serializable dictionary containing the full computation
    details including per-biomarker contributions, context adjustments,
    and optional gate results.

    Args:
        indicator_score: Result from indicator computation (Story 4.8)
        gate_result: Result from DSM-gate (Story 4.9), or None

    Returns:
        JSON-serializable dict with full computation details
    """
    log: dict[str, Any] = {
        "indicator_name": indicator_score.indicator_name,
        "daily_likelihood": indicator_score.daily_likelihood,
        "contributions": {
            name: {
                "membership": contrib.membership,
                "direction": contrib.direction,
                "base_weight": contrib.base_weight,
                "context_multiplier": contrib.context_multiplier,
                "adjusted_weight": contrib.adjusted_weight,
                "contribution": contrib.contribution,
            }
            for name, contrib in indicator_score.contributions.items()
        },
        "context": {
            "applied": indicator_score.context_applied,
            "confidence": indicator_score.context_confidence,
        },
        "weights": {
            "before_context": indicator_score.weights_before_context,
            "after_context": indicator_score.weights_after_context,
        },
        "biomarkers_used": list(indicator_score.biomarkers_used),
        "biomarkers_missing": list(indicator_score.biomarkers_missing),
        "data_reliability_score": indicator_score.data_reliability_score,
    }

    if gate_result:
        log["gate"] = {
            "presence_flag": gate_result.presence_flag,
            "days_above_threshold": gate_result.days_above_threshold,
            "days_evaluated": gate_result.days_evaluated,
            "window_size": gate_result.window_size,
            "required_days": gate_result.required_days,
            "threshold": gate_result.threshold,
            "daily_flags": list(gate_result.daily_flags),
            "insufficient_data": gate_result.insufficient_data,
        }

    return log


def _build_daily_computation_log(
    daily_score: DailyIndicatorScore,
    context_used: str,
) -> dict[str, Any]:
    """Build computation log for a daily indicator score.

    Creates a simplified JSON-serializable dictionary for daily scores.
    Less detailed than the full IndicatorScore log, but captures key info.

    Args:
        daily_score: Daily indicator score from Story 4.13
        context_used: Active context applied during computation

    Returns:
        JSON-serializable dict with daily computation details
    """
    return {
        "indicator_name": daily_score.indicator_name,
        "date": daily_score.date.isoformat(),
        "daily_likelihood": daily_score.daily_likelihood,
        "biomarkers_used": list(daily_score.biomarkers_used),
        "biomarkers_missing": list(daily_score.biomarkers_missing),
        "data_reliability_score": daily_score.data_reliability_score,
        "context_used": context_used,
    }


def save_indicator(
    indicator_score: IndicatorScore,
    gate_result: IndicatorGateResult | None,
    user_id: str,
    analysis_run_id: uuid.UUID,
    session: Session,
) -> Indicator:
    """Save a single indicator to the database.

    Creates an Indicator ORM object with all computed values and metadata.
    The object is added to the session but not committed - the caller is
    responsible for managing the transaction.

    Note: config_snapshot is stored on AnalysisRun, not on each Indicator.

    Args:
        indicator_score: Result from indicator computation (Story 4.8)
        gate_result: Result from DSM-gate (Story 4.9), or None
        user_id: User ID for this indicator
        analysis_run_id: UUID linking to analysis run
        session: SQLAlchemy session (caller manages transaction)

    Returns:
        Created Indicator ORM object (not yet committed)
    """
    # Extract modalities from biomarkers
    modalities = _extract_modalities(indicator_score.biomarkers_used)

    # Build computation log
    computation_log = _build_computation_log(indicator_score, gate_result)

    # Create Indicator ORM object
    indicator = Indicator(
        user_id=user_id,
        timestamp=indicator_score.timestamp,
        indicator_type=indicator_score.indicator_name,
        value=indicator_score.daily_likelihood,
        data_reliability_score=indicator_score.data_reliability_score,
        analysis_run_id=analysis_run_id,
        presence_flag=gate_result.presence_flag if gate_result else None,
        context_used=indicator_score.context_applied,
        modalities_used=modalities,
        computation_log=computation_log,
    )

    session.add(indicator)

    logger.debug(
        "Saved indicator '%s' for user '%s': L_k=%.3f, presence=%s",
        indicator_score.indicator_name,
        user_id,
        indicator_score.daily_likelihood,
        gate_result.presence_flag if gate_result else "N/A",
    )

    return indicator


def save_all_indicators(
    indicator_scores: dict[str, IndicatorScore],
    gate_results: dict[str, IndicatorGateResult],
    user_id: str,
    analysis_run_id: uuid.UUID,
    session: Session,
) -> list[Indicator]:
    """Save all indicators from an analysis run (end_date summary only).

    Persists indicator summary rows (one per indicator). For daily scores,
    use save_daily_indicator_scores instead.

    Args:
        indicator_scores: Dict of indicator name -> IndicatorScore
        gate_results: Dict of indicator name -> IndicatorGateResult
        user_id: User ID for these indicators
        analysis_run_id: UUID linking to analysis run
        session: SQLAlchemy session (caller manages transaction)

    Returns:
        List of created Indicator ORM objects

    Note:
        If gate_results is missing entries for some indicators, those
        indicators will be saved with presence_flag=None and a warning
        will be logged.
    """
    # Log warning for missing gate results
    missing_gates = set(indicator_scores.keys()) - set(gate_results.keys())
    if missing_gates:
        logger.warning(
            "Gate results missing for indicators: %s (will save with presence_flag=None)",
            missing_gates,
        )

    indicators: list[Indicator] = []
    for indicator_name, score in indicator_scores.items():
        gate_result = gate_results.get(indicator_name)
        indicator = save_indicator(
            indicator_score=score,
            gate_result=gate_result,
            user_id=user_id,
            analysis_run_id=analysis_run_id,
            session=session,
        )
        indicators.append(indicator)

    # Flush to validate (but don't commit - caller handles)
    session.flush()

    logger.info(
        "Saved %d indicators for user '%s' in analysis run '%s'",
        len(indicators),
        user_id,
        analysis_run_id,
    )

    return indicators


def save_daily_indicator_scores(
    daily_indicator_scores: dict[str, list[DailyIndicatorScore]],
    gate_results: dict[str, IndicatorGateResult],
    context_used: str,
    end_date: date,
    user_id: str,
    analysis_run_id: uuid.UUID,
    session: Session,
) -> list[Indicator]:
    """Save all daily indicator scores from an analysis run (Story 4.13).

    Persists one Indicator row per day per indicator. The end_date row
    gets the presence_flag from DSM-gate; other rows have presence_flag=NULL.

    Args:
        daily_indicator_scores: Dict of indicator name -> list of DailyIndicatorScore
        gate_results: Dict of indicator name -> IndicatorGateResult
        context_used: Active context during analysis (same for all days)
        end_date: The end date of analysis window (gets presence_flag)
        user_id: User ID for these indicators
        analysis_run_id: UUID linking to analysis run
        session: SQLAlchemy session (caller manages transaction)

    Returns:
        List of created Indicator ORM objects (M rows per indicator)
    """
    indicators: list[Indicator] = []
    total_days = 0

    for indicator_name, daily_scores in daily_indicator_scores.items():
        gate_result = gate_results.get(indicator_name)

        for daily_score in daily_scores:
            # Extract modalities from biomarkers used on this day
            modalities = _extract_modalities(daily_score.biomarkers_used)

            # Build computation log for this day
            computation_log = _build_daily_computation_log(daily_score, context_used)

            # Only end_date row gets presence_flag
            is_end_date = daily_score.date == end_date
            presence_flag = gate_result.presence_flag if (gate_result and is_end_date) else None

            # If this is the end_date and we have gate_result, add gate info to log
            if is_end_date and gate_result:
                computation_log["gate"] = {
                    "presence_flag": gate_result.presence_flag,
                    "days_above_threshold": gate_result.days_above_threshold,
                    "days_evaluated": gate_result.days_evaluated,
                    "window_size": gate_result.window_size,
                    "required_days": gate_result.required_days,
                    "threshold": gate_result.threshold,
                    "daily_flags": list(gate_result.daily_flags),
                    "insufficient_data": gate_result.insufficient_data,
                }

            # Create timestamp from date (use midnight UTC)
            timestamp = datetime.combine(
                daily_score.date,
                datetime.min.time(),
                tzinfo=UTC,
            )

            indicator = Indicator(
                user_id=user_id,
                timestamp=timestamp,
                indicator_type=indicator_name,
                value=daily_score.daily_likelihood,
                data_reliability_score=daily_score.data_reliability_score,
                analysis_run_id=analysis_run_id,
                presence_flag=presence_flag,
                context_used=context_used,
                modalities_used=modalities,
                computation_log=computation_log,
            )

            session.add(indicator)
            indicators.append(indicator)
            total_days += 1

    # Flush to validate (but don't commit - caller handles)
    session.flush()

    logger.info(
        "Saved %d daily indicator scores (%d indicators × %d days avg) "
        "for user '%s' in analysis run '%s'",
        total_days,
        len(daily_indicator_scores),
        total_days // max(len(daily_indicator_scores), 1),
        user_id,
        analysis_run_id,
    )

    return indicators


def save_analysis_run(
    analysis_run_id: uuid.UUID,
    user_id: str,
    config: AnalysisConfig,
    start_time: datetime,
    end_time: datetime,
    session: Session,
) -> AnalysisRun:
    """Save analysis run metadata to the database.

    Creates an AnalysisRun record with the run configuration and time window.
    The pipeline trace can be added later via save_pipeline_trace().
    The caller is responsible for committing the transaction.

    Args:
        analysis_run_id: UUID for this analysis run
        user_id: User ID for this analysis
        config: Analysis configuration used
        start_time: Start of analysis time window
        end_time: End of analysis time window
        session: SQLAlchemy session (caller manages transaction)

    Returns:
        Created AnalysisRun ORM object (not yet committed)
    """
    analysis_run = AnalysisRun(
        id=analysis_run_id,
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        config_snapshot=config.to_dict(),
        pipeline_trace=None,  # Will be set by save_pipeline_trace()
    )

    session.add(analysis_run)
    session.flush()

    logger.info(
        "Saved analysis run '%s' for user '%s': %s to %s",
        analysis_run_id,
        user_id,
        start_time,
        end_time,
    )

    return analysis_run


def save_daily_summaries(
    daily_summaries: list[DailyIndicatorSummary],
    user_id: str,
    analysis_run_id: uuid.UUID,
    session: Session,
) -> int:
    """Save daily indicator summaries from windowed analysis (Story 6.6 AC3).

    Persists DailyIndicatorSummary records as Indicator rows in the database.
    Each summary is stored with its multi-dimensional metrics in the
    computation_log JSONB column for flexibility during validation phase.

    Args:
        daily_summaries: List of DailyIndicatorSummary from compute_daily_summary
        user_id: User ID for these summaries
        analysis_run_id: UUID linking to analysis run
        session: SQLAlchemy session (caller manages transaction)

    Returns:
        Count of summaries saved
    """
    if not daily_summaries:
        logger.info("No daily summaries to save for user '%s'", user_id)
        return 0

    saved_count = 0

    for summary in daily_summaries:
        # Build computation log with daily summary metrics
        computation_log: dict[str, Any] = {
            "source": "windowed_analysis",
            "indicator_name": summary.indicator_name,
            "date": summary.date.isoformat(),
            "likelihood": summary.likelihood,
            "window_scores": list(summary.window_scores),
            "quality": {
                "total_windows": summary.total_windows,
                "expected_windows": summary.expected_windows,
                "data_coverage": summary.data_coverage,
                "average_biomarker_completeness": summary.average_biomarker_completeness,
                "context_availability": summary.context_availability,
            },
        }

        # Create timestamp from date (use midnight UTC)
        timestamp = datetime.combine(
            summary.date,
            datetime.min.time(),
            tzinfo=UTC,
        )

        # Create Indicator ORM object
        indicator = Indicator(
            user_id=user_id,
            timestamp=timestamp,
            indicator_type=summary.indicator_name,
            value=summary.likelihood,  # Use composite likelihood as primary value
            data_reliability_score=summary.data_coverage,  # Use coverage as reliability proxy
            analysis_run_id=analysis_run_id,
            presence_flag=None,  # Windowed analysis doesn't use DSM-gate
            context_used=None,  # Context varies per window
            modalities_used=None,  # Not tracked at daily level
            computation_log=computation_log,
        )

        session.add(indicator)
        saved_count += 1

    # Flush to validate (but don't commit - caller handles)
    session.flush()

    logger.info(
        "Saved %d daily summaries for user '%s' in analysis run '%s'",
        saved_count,
        user_id,
        analysis_run_id,
    )

    return saved_count


def save_window_indicators(
    window_indicators: list[WindowIndicator],
    user_id: str,
    analysis_run_id: uuid.UUID,
    session: Session,
) -> int:
    """Save window-level indicator scores (Story 6.6 AC3 - optional).

    Persists WindowIndicator records as Indicator rows in the database.
    This is an optional persistence layer for fine-grained window data,
    useful for debugging and detailed analysis but not required for
    the main pipeline.

    Each window indicator is stored with its FASL score and contributing
    biomarkers in the computation_log JSONB column.

    Args:
        window_indicators: List of WindowIndicator from compute_window_indicators
        user_id: User ID for these indicators
        analysis_run_id: UUID linking to analysis run
        session: SQLAlchemy session (caller manages transaction)

    Returns:
        Count of window indicators saved
    """
    if not window_indicators:
        logger.info("No window indicators to save for user '%s'", user_id)
        return 0

    saved_count = 0

    for wi in window_indicators:
        # Build computation log with window indicator details
        computation_log: dict[str, Any] = {
            "source": "window_indicator",
            "indicator_name": wi.indicator_name,
            "window_start": wi.window_start.isoformat(),
            "window_end": wi.window_end.isoformat(),
            "indicator_score": wi.indicator_score,
            "contributing_biomarkers": wi.contributing_biomarkers,
            "biomarkers_present": wi.biomarkers_present,
            "biomarkers_expected": wi.biomarkers_expected,
            "biomarker_completeness": wi.biomarker_completeness,
            "dominant_context": wi.dominant_context,
            "context_confidence": wi.context_confidence,
            "fasl_contributions": [
                {
                    "biomarker": fc.biomarker,
                    "directed_membership": fc.directed_membership,
                    "biomarker_weight": fc.biomarker_weight,
                    "context_weight": fc.context_weight,
                    "effective_weight": fc.effective_weight,
                    "contribution": fc.contribution,
                    "is_missing": fc.is_missing,
                }
                for fc in wi.fasl_contributions
            ],
        }

        # Create Indicator ORM object
        indicator = Indicator(
            user_id=user_id,
            timestamp=wi.window_start,
            indicator_type=f"{wi.indicator_name}_window",
            value=wi.indicator_score,
            data_reliability_score=wi.biomarker_completeness,
            analysis_run_id=analysis_run_id,
            presence_flag=None,
            context_used=wi.dominant_context,
            modalities_used=None,
            computation_log=computation_log,
        )

        session.add(indicator)
        saved_count += 1

    # Flush to validate (but don't commit - caller handles)
    session.flush()

    logger.info(
        "Saved %d window indicators for user '%s' in analysis run '%s'",
        saved_count,
        user_id,
        analysis_run_id,
    )

    return saved_count
