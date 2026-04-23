"""Analysis trigger action module."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time

from src.core.baseline_config import BaselineFile
from src.shared.logging import get_logger

# Type alias for progress callback: (step_number, total_steps, step_description)
ProgressCallback = Callable[[int, int, str], None] | None

logger = get_logger(__name__)

# Thresholds for legacy field derivations
EPISODE_THRESHOLD = 0.5  # Peak likelihood threshold for episode_likely
PRESENCE_THRESHOLD = 0.5  # Likelihood threshold for indicators_present count


@dataclass
class AnalysisTriggerResult:
    """Result of analysis trigger action.

    Contains both new windowed pipeline fields and derived legacy fields
    for backward compatibility with existing UI components.
    """

    success: bool
    run_id: str | None = None
    user_id: str | None = None

    # New windowed pipeline fields (direct from WindowedAnalysisResult)
    window_count: int = 0
    context_evaluations_added: int = 0
    daily_summaries_count: int = 0
    peak_likelihood: float = 0.0
    mean_likelihood: float = 0.0
    duration_ms: int = 0

    # Legacy fields (derived from daily_summaries for backward compatibility)
    indicator_count: int = 0
    avg_data_reliability: float = 0.0
    context_detected: str = ""
    episode_likely: bool = False
    indicators_present: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Error fields
    error_code: str | None = None
    error_message: str | None = None
    error_step: str | None = None


def trigger_analysis(
    user_id: str,
    start_time: datetime,
    end_time: datetime,
    baseline_config: BaselineFile,
    experiment_id: str | None = None,
    context_evaluation_run_id: uuid.UUID | None = None,
    progress_callback: ProgressCallback = None,
) -> AnalysisTriggerResult:
    """Trigger analysis pipeline and return result.

    Wraps run_analysis() with error handling appropriate for UI display.

    Story 6.14: When context_evaluation_run_id is provided, the analysis
    uses context evaluations from that specific run instead of auto-generating.

    Args:
        user_id: User to analyze
        start_time: Start of analysis window
        end_time: End of analysis window
        baseline_config: Required baseline configuration for biomarker normalization
        experiment_id: Optional experiment ID to use instead of default config
        context_evaluation_run_id: Optional UUID of a specific context evaluation
            run to use. When provided, skips auto-generation and uses context
            records from that run.

    Returns:
        AnalysisTriggerResult with success/error details
    """
    from src.core.analysis import AnalysisError, run_analysis

    baseline_name = baseline_config.metadata.name if baseline_config.metadata else "unnamed"
    context_run_str = str(context_evaluation_run_id)[:8] if context_evaluation_run_id else "auto"
    logger.info(
        "Triggering analysis for user '%s' (%s to %s), experiment=%s, baseline=%s, context_run=%s",
        user_id,
        start_time,
        end_time,
        experiment_id,
        baseline_name,
        context_run_str,
    )

    # Get config - either from experiment or default
    config = None
    if experiment_id:
        from src.dashboard.data.experiments import get_experiment_config

        config = get_experiment_config(experiment_id)
        if config is None:
            return AnalysisTriggerResult(
                success=False,
                error_code="EXPERIMENT_NOT_FOUND",
                error_message=f"Experiment {experiment_id} not found",
            )
        logger.info("Using experiment config: %s", experiment_id)

    try:
        result = run_analysis(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            baseline_config=baseline_config,  # Story 4.14: required baseline
            config=config,  # Use experiment config or default
            session=None,  # Create own session
            context_evaluation_run_id=context_evaluation_run_id,  # Story 6.14: selected context run
            progress_callback=progress_callback,
        )

        logger.info("Analysis completed: run_id=%s", result.run_id)

        # Map WindowedAnalysisResult to AnalysisTriggerResult
        summaries = result.daily_summaries

        # Direct mappings
        window_count = result.window_count
        context_evaluations_added = result.context_evaluations_added
        daily_summaries_count = len(summaries)

        # Compute peak and mean likelihood from daily summaries
        likelihoods = [s.likelihood for s in summaries]
        peak_likelihood = max(likelihoods) if likelihoods else 0.0
        mean_likelihood = sum(likelihoods) / len(likelihoods) if likelihoods else 0.0

        # Legacy field derivations from daily_summaries
        indicator_count = len({s.indicator_name for s in summaries})
        avg_data_reliability = (
            sum(s.average_biomarker_completeness for s in summaries) / len(summaries)
            if summaries
            else 0.0
        )
        episode_likely = peak_likelihood >= EPISODE_THRESHOLD
        indicators_present = sum(
            1 for s in summaries if s.likelihood >= PRESENCE_THRESHOLD
        )

        context_detected = "unknown"

        # Date to datetime conversion
        result_start_time = datetime.combine(result.start_date, time.min)
        result_end_time = datetime.combine(result.end_date, time.max)

        return AnalysisTriggerResult(
            success=True,
            run_id=result.run_id,
            user_id=result.user_id,
            # New windowed pipeline fields
            window_count=window_count,
            context_evaluations_added=context_evaluations_added,
            daily_summaries_count=daily_summaries_count,
            peak_likelihood=peak_likelihood,
            mean_likelihood=mean_likelihood,
            duration_ms=result.duration_ms,
            # Legacy fields (derived)
            indicator_count=indicator_count,
            avg_data_reliability=avg_data_reliability,
            context_detected=context_detected,
            episode_likely=episode_likely,
            indicators_present=indicators_present,
            start_time=result_start_time,
            end_time=result_end_time,
        )

    except AnalysisError as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        return AnalysisTriggerResult(
            success=False,
            error_code="ANALYSIS_ERROR",
            error_message=str(e),
            error_step=e.step,
            run_id=str(e.run_id) if e.run_id else None,
        )

    except Exception as e:
        logger.error("Unexpected analysis error: %s", e, exc_info=True)
        return AnalysisTriggerResult(
            success=False,
            error_code="UNEXPECTED_ERROR",
            error_message=f"Unexpected error: {e}",
        )
