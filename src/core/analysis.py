"""Analysis orchestration module.

Provides the main entry point for running the windowed analysis pipeline
on demand. Orchestrates context history population, data reading, window
aggregation, membership computation, FASL computation, daily summary
generation, and persistence.

The windowed pipeline preserves temporal patterns through window-level
processing with context binding.
"""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Type alias for progress callback: (step_number, total_steps, step_description)
ProgressCallback = Callable[[int, int, str], None] | None

from sqlalchemy.orm import Session

from src.core.baseline_config import BaselineFile
from src.core.config import (
    AnalysisConfig,
    ConfigurationError,
    get_default_config,
    load_config,
)
from src.core.context.history import ContextCoverageResult, ContextHistoryService
from src.core.data_reader import DataReader
from src.core.models.daily_summary import DailyIndicatorSummary
from src.core.dsm_gate import DSMGate
from src.core.persistence import (
    save_analysis_run,
    save_daily_summaries,
    save_window_indicators,
)
from src.core.pipeline import PipelineTracer, save_pipeline_trace
from src.core.processors.daily_aggregator import compute_daily_summary
from src.core.processors.window_aggregator import aggregate_into_windows
from src.core.processors.window_fasl import compute_window_indicators
from src.core.processors.window_membership import compute_window_memberships
from src.shared.database import get_db

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisError",
    "WindowedAnalysisResult",
    "run_analysis",
]

CONFIG_PATH = Path("config/analysis.yaml")
CONTEXT_CONFIG_PATH = Path("config/context_evaluation.yaml")


class AnalysisError(Exception):
    """Raised when analysis pipeline fails.

    Attributes:
        run_id: Analysis run UUID for traceability
        step: Pipeline step where failure occurred
        message: Error description
    """

    def __init__(
        self,
        message: str,
        run_id: uuid.UUID | None = None,
        step: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.step = step
        super().__init__(message)

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.run_id:
            parts.append(f"run_id={self.run_id}")
        if self.step:
            parts.append(f"step={self.step}")
        return " | ".join(parts)


@dataclass(frozen=True)
class WindowedAnalysisResult:
    """Result of windowed analysis pipeline execution.

    Story 6.6: Pipeline Integration & Orchestration (AC1)

    Contains results from the windowed analysis pipeline that preserves
    temporal patterns through window-level processing with context binding.

    Attributes:
        run_id: Unique identifier for this analysis run (UUID as string)
        user_id: User for whom analysis was performed
        start_date: Start date of analysis window
        end_date: End date of analysis window
        daily_summaries: Tuple of DailyIndicatorSummary for each day/indicator
        window_count: Total number of windows processed across all biomarkers
        context_evaluations_added: Number of context evaluations added during backfill
        duration_ms: Total execution time in milliseconds
        config_snapshot: Configuration snapshot for reproducibility
    """

    run_id: str
    user_id: str
    start_date: date
    end_date: date
    daily_summaries: tuple[DailyIndicatorSummary, ...]
    window_count: int
    context_evaluations_added: int
    duration_ms: int
    config_snapshot: dict[str, Any]


def _load_or_default_config(config: AnalysisConfig | None) -> AnalysisConfig:
    """Load configuration from file or use provided/default.

    Priority:
    1. Provided config object
    2. Config file at CONFIG_PATH
    3. Default config

    Args:
        config: Optional pre-loaded configuration

    Returns:
        AnalysisConfig to use for analysis
    """
    if config is not None:
        logger.info("Using provided configuration")
        return config

    if CONFIG_PATH.exists():
        try:
            loaded = load_config(CONFIG_PATH)
            logger.info("Loaded configuration from %s", CONFIG_PATH)
            return loaded
        except ConfigurationError as e:
            logger.warning(
                "Failed to load config from %s: %s. Using defaults.",
                CONFIG_PATH,
                e,
            )

    logger.info("Using default configuration")
    return get_default_config()


def run_analysis(
    user_id: str,
    start_time: datetime,
    end_time: datetime,
    baseline_config: BaselineFile,
    config: AnalysisConfig | None = None,
    session: Session | None = None,
    context_evaluation_run_id: uuid.UUID | None = None,
    progress_callback: ProgressCallback = None,
) -> WindowedAnalysisResult:
    """Run analysis pipeline for a user.

    Orchestrates the analysis pipeline (10 steps):
    1. Generate unique analysis_run_id
    2. Ensure context history exists for analysis period (or use selected run)
    3. Read biomarker records
    4. Window aggregation
    5. Membership computation with context
    6. Window-level FASL
    7. Daily summary computation
    8. Episode decision (DSM-gate)
    9. Persist results
    10. Build pipeline trace

    This pipeline preserves temporal patterns through window-level processing
    and binds correct context to each time window.

    Story 6.14: When context_evaluation_run_id is provided, the pipeline uses
    context evaluations from that specific run instead of auto-generating.
    Dates not covered by the selected run use neutral weights (1.0).

    Args:
        user_id: User identifier to analyze
        start_time: Start of analysis time window (inclusive)
        end_time: End of analysis time window (inclusive)
        baseline_config: Required baseline configuration for biomarker normalization.
            Must contain all biomarkers used in the analysis.
        config: Optional analysis configuration. If None, loads from
            config/analysis.yaml or uses defaults.
        session: Optional SQLAlchemy session. If None, creates a new
            session and manages its lifecycle.
        context_evaluation_run_id: Optional UUID of a specific context evaluation
            run to use. When provided, skips auto-generation and uses context
            records from that run. Dates not covered use neutral weights.

    Returns:
        WindowedAnalysisResult with run summary and daily summaries

    Raises:
        AnalysisError: If pipeline fails at any step, including missing baseline
    """
    run_id = uuid.uuid4()
    start_perf = time.perf_counter()
    owns_session = session is None
    session_gen = None

    logger.info(
        "Starting analysis run '%s' for user '%s' (%s to %s)",
        run_id,
        user_id,
        start_time,
        end_time,
    )

    # Load configuration
    analysis_config = _load_or_default_config(config)

    # Get or create session
    if owns_session:
        session_gen = get_db()
        session = next(session_gen)

    # Initialize pipeline tracer for transparency (Story 4.12)
    tracer = PipelineTracer(str(run_id), user_id)

    try:
        # Step 1: Generate analysis_run_id (already done above)
        logger.info("[Step 1/10] Generated analysis run ID: %s", run_id)
        if progress_callback:
            progress_callback(1, 10, "Initializing analysis run")

        # Step 2: Context history - either use selected run or ensure exists
        # Story 6.14: When context_evaluation_run_id is provided, use that run's
        # context evaluations instead of auto-generating
        context_evaluations_added = 0
        context_source: str = "auto_generated"
        neutral_weight_dates: list[str] = []

        if context_evaluation_run_id is not None:
            # Using a pre-selected context evaluation run
            logger.info(
                "[Step 2/10] Using selected context evaluation run: %s",
                context_evaluation_run_id,
            )
            if progress_callback:
                progress_callback(2, 10, "Loading selected context run")
            tracer.start_step(
                "Context History (Selected Run)",
                inputs={
                    "user_id": user_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "context_evaluation_run_id": str(context_evaluation_run_id),
                },
            )

            # Create service filtered by the selected run
            context_history_service = ContextHistoryService(
                session=session,
                config=analysis_config,
                context_evaluation_run_id=context_evaluation_run_id,
            )

            # Check coverage for the analysis date range
            coverage_result = context_history_service.check_context_coverage(
                user_id=user_id,
                start=start_time,
                end=end_time,
            )

            context_source = "selected_run"
            neutral_weight_dates = [d.isoformat() for d in coverage_result.missing_dates]

            tracer.end_step(
                outputs={
                    "context_source": context_source,
                    "context_evaluation_run_id": str(context_evaluation_run_id),
                    "dates_covered": coverage_result.dates_covered,
                    "dates_missing": len(coverage_result.missing_dates),
                    "neutral_weight_dates": neutral_weight_dates,
                    "coverage_ratio": coverage_result.coverage_ratio,
                }
            )

            logger.info(
                "Using context run %s: %d/%d dates covered (%.1f%%)",
                str(context_evaluation_run_id)[:8],
                coverage_result.dates_covered,
                coverage_result.dates_covered + len(coverage_result.missing_dates),
                coverage_result.coverage_ratio * 100,
            )
        else:
            # Auto-generate context history (existing behavior)
            logger.info("[Step 2/10] Ensuring context history exists")
            if progress_callback:
                progress_callback(2, 10, "Ensuring context history exists")
            tracer.start_step(
                "Context History Population",
                inputs={
                    "user_id": user_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            )

            context_history_service = ContextHistoryService(
                session=session,
                config=analysis_config,
            )
            history_result = context_history_service.ensure_context_history_exists(
                user_id=user_id,
                start=start_time,
                end=end_time,
            )
            context_evaluations_added = history_result.evaluations_added

            tracer.end_step(
                outputs={
                    "context_source": context_source,
                    "status": history_result.status.value,
                    "gaps_found": history_result.gaps_found,
                    "evaluations_added": context_evaluations_added,
                    "message": history_result.message,
                }
            )

            logger.info(
                "Context history: %s (added %d evaluations)",
                history_result.status.value,
                context_evaluations_added,
            )

        # Step 3: Read biomarker data
        logger.info("[Step 3/10] Reading biomarker data")
        if progress_callback:
            progress_callback(3, 10, "Reading biomarker data")
        tracer.start_step(
            "Read Data",
            inputs={
                "user_id": user_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

        data_reader = DataReader(session)
        data_result = data_reader.read_all(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
        )

        if not data_result.stats.has_data:
            tracer.end_step(
                outputs={"has_data": False, "biomarker_count": 0}
            )
            raise AnalysisError(
                f"No data found for user '{user_id}' in time range "
                f"{start_time} to {end_time}",
                run_id=run_id,
                step="read_data",
            )

        biomarker_records = list(data_result.biomarkers)

        tracer.end_step(
            outputs={
                "biomarker_count": data_result.stats.biomarker_count,
                "context_count": data_result.stats.context_count,
                "has_data": data_result.stats.has_data,
            }
        )

        logger.info(
            "Retrieved %d biomarker records",
            data_result.stats.biomarker_count,
        )

        # Step 4: Window aggregation
        logger.info("[Step 4/10] Aggregating biomarkers into windows")
        if progress_callback:
            progress_callback(4, 10, "Aggregating into windows")
        tracer.start_step(
            "Window Aggregation",
            inputs={
                "biomarker_count": len(biomarker_records),
                "window_size_minutes": analysis_config.window.size_minutes,
                "aggregation_method": analysis_config.window.aggregation_method,
            },
        )

        window_aggregates = aggregate_into_windows(
            records=biomarker_records,
            window_size_minutes=analysis_config.window.size_minutes,
            aggregation_method=analysis_config.window.aggregation_method,
            min_readings=analysis_config.window.min_readings,
        )

        total_windows = sum(len(windows) for windows in window_aggregates.values())

        # Compute per-biomarker aggregation stats for trace
        biomarker_stats = {}
        for biomarker_name, windows in window_aggregates.items():
            if windows:
                values = [w.aggregated_value for w in windows]
                biomarker_stats[biomarker_name] = {
                    "window_count": len(windows),
                    "value_min": round(min(values), 4),
                    "value_max": round(max(values), 4),
                    "value_mean": round(sum(values) / len(values), 4),
                }

        tracer.end_step(
            outputs={
                "window_count": total_windows,
                "biomarkers_aggregated": list(window_aggregates.keys()),
                "biomarker_stats": biomarker_stats,
            }
        )

        logger.info(
            "Aggregated into %d windows across %d biomarkers",
            total_windows,
            len(window_aggregates),
        )

        if total_windows == 0:
            raise AnalysisError(
                f"No windows generated from biomarker data for user '{user_id}'",
                run_id=run_id,
                step="window_aggregation",
            )

        # Step 5 & 6 & 7: Process each indicator
        logger.info("[Step 5-7/10] Computing indicators and daily summaries")

        all_daily_summaries: list[DailyIndicatorSummary] = []
        all_window_indicators = []
        total_window_indicators = 0
        indicator_names = list(analysis_config.indicators.keys())
        total_indicators = len(indicator_names)

        for idx, (indicator_name, indicator_config) in enumerate(analysis_config.indicators.items()):
            if progress_callback:
                progress_callback(5, 10, f"Processing indicator {idx + 1}/{total_indicators}: {indicator_name}")
            # Step 5: Membership computation with context
            tracer.start_step(
                "Membership Computation",
                inputs={
                    "indicator_name": indicator_name,
                    "context_strategy": analysis_config.context.strategy,
                },
            )

            window_memberships = compute_window_memberships(
                window_aggregates=window_aggregates,
                user_id=user_id,
                indicator_name=indicator_name,
                session=session,
                config=analysis_config,
                baseline_config=baseline_config,
                context_evaluation_run_id=context_evaluation_run_id,
            )

            membership_count = sum(len(m) for m in window_memberships.values())

            # Compute per-biomarker membership stats for trace
            membership_stats = {}
            context_weights_used = {}
            for biomarker_name, memberships in window_memberships.items():
                if memberships:
                    z_scores = [m.z_score for m in memberships]
                    raw_memberships = [m.membership for m in memberships]
                    weighted_memberships = [m.weighted_membership for m in memberships]
                    weights = [m.context_weight for m in memberships]
                    contexts = [m.dominant_context for m in memberships]

                    membership_stats[biomarker_name] = {
                        "count": len(memberships),
                        "z_score_min": round(min(z_scores), 4),
                        "z_score_max": round(max(z_scores), 4),
                        "z_score_mean": round(sum(z_scores) / len(z_scores), 4),
                        "membership_min": round(min(raw_memberships), 4),
                        "membership_max": round(max(raw_memberships), 4),
                        "membership_mean": round(sum(raw_memberships) / len(raw_memberships), 4),
                        "weighted_membership_min": round(min(weighted_memberships), 4),
                        "weighted_membership_max": round(max(weighted_memberships), 4),
                        "weighted_membership_mean": round(sum(weighted_memberships) / len(weighted_memberships), 4),
                        "context_weight_min": round(min(weights), 4),
                        "context_weight_max": round(max(weights), 4),
                    }

                    # Track unique context weights used
                    for ctx, weight in zip(contexts, weights):
                        if ctx not in context_weights_used:
                            context_weights_used[ctx] = {}
                        if biomarker_name not in context_weights_used[ctx]:
                            context_weights_used[ctx][biomarker_name] = weight

            tracer.end_step(
                outputs={
                    "indicator_name": indicator_name,
                    "membership_count": membership_count,
                    "biomarkers_processed": list(window_memberships.keys()),
                    "membership_stats": membership_stats,
                    "context_weights_used": context_weights_used,
                }
            )

            # Step 6: Window-level FASL
            tracer.start_step(
                "Window FASL",
                inputs={
                    "indicator_name": indicator_name,
                    "missing_strategy": analysis_config.fasl.missing_biomarker_strategy,
                },
            )

            window_indicators = compute_window_indicators(
                window_memberships=window_memberships,
                indicator_name=indicator_name,
                indicator_config=indicator_config,
                config=analysis_config,
            )

            total_window_indicators += len(window_indicators)
            all_window_indicators.extend(window_indicators)

            # Compute indicator score distribution for trace
            indicator_score_stats = {}
            if window_indicators:
                scores = [wi.indicator_score for wi in window_indicators]
                completeness_values = [wi.biomarker_completeness for wi in window_indicators]

                # Find peak window
                peak_idx = scores.index(max(scores))
                peak_window = window_indicators[peak_idx]

                # Compute std deviation
                mean_score = sum(scores) / len(scores)
                variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
                std_score = variance ** 0.5

                indicator_score_stats = {
                    "score_min": round(min(scores), 4),
                    "score_max": round(max(scores), 4),
                    "score_mean": round(mean_score, 4),
                    "score_std": round(std_score, 4),
                    "completeness_mean": round(sum(completeness_values) / len(completeness_values), 4),
                    "peak_window_start": peak_window.window_start.isoformat(),
                    "peak_window_context": peak_window.dominant_context,
                    "peak_contributing_biomarkers": {
                        k: round(v, 4) for k, v in peak_window.contributing_biomarkers.items()
                    },
                }

            tracer.end_step(
                outputs={
                    "indicator_name": indicator_name,
                    "window_indicator_count": len(window_indicators),
                    "indicator_score_stats": indicator_score_stats,
                }
            )

            # Step 7: Daily summary computation
            tracer.start_step(
                "Daily Aggregation",
                inputs={
                    "indicator_name": indicator_name,
                    "window_indicator_count": len(window_indicators),
                },
            )

            # Get unique dates from window indicators
            dates_with_data = {wi.window_start.date() for wi in window_indicators}

            indicator_summaries = []
            for target_date in sorted(dates_with_data):
                summary = compute_daily_summary(
                    window_indicators=window_indicators,
                    config=analysis_config,
                    target_date=target_date,
                )
                if summary:
                    indicator_summaries.append(summary)
                    all_daily_summaries.append(summary)

            # Extract daily summary values for trace
            daily_summary_data = []
            for summary in indicator_summaries:
                daily_summary_data.append({
                    "date": summary.date.isoformat(),
                    "likelihood": round(summary.likelihood, 4),
                    "total_windows": summary.total_windows,
                    "data_coverage": round(summary.data_coverage, 4),
                    "average_biomarker_completeness": round(summary.average_biomarker_completeness, 4),
                    "context_availability": round(summary.context_availability, 4),
                })

            tracer.end_step(
                outputs={
                    "indicator_name": indicator_name,
                    "daily_summaries_count": len(indicator_summaries),
                    "dates_processed": [d.isoformat() for d in sorted(dates_with_data)],
                    "daily_summaries": daily_summary_data,
                }
            )

        logger.info(
            "Computed %d window indicators and %d daily summaries",
            total_window_indicators,
            len(all_daily_summaries),
        )

        # Step 8: Episode Decision (DSM-gate)
        logger.info("[Step 8/10] Computing episode decision")
        if progress_callback:
            progress_callback(8, 10, "Computing episode decision")
        tracer.start_step(
            "Episode Decision",
            inputs={
                "daily_summaries_count": len(all_daily_summaries),
                "indicators": list(analysis_config.indicators.keys()),
            },
        )

        # Collect daily likelihoods per indicator from daily summaries
        indicator_daily_likelihoods: dict[str, list[float]] = {}
        for summary in all_daily_summaries:
            if summary.indicator_name not in indicator_daily_likelihoods:
                indicator_daily_likelihoods[summary.indicator_name] = []
            indicator_daily_likelihoods[summary.indicator_name].append(summary.likelihood)

        # Apply DSM gate to all indicators
        dsm_gate = DSMGate(analysis_config)
        indicator_gate_results = dsm_gate.apply_all_gates(indicator_daily_likelihoods)

        # Compute episode decision
        episode_decision = dsm_gate.compute_episode(indicator_gate_results)

        # Serialize gate results for trace storage
        gate_results_serialized = {}
        for ind_name, gate_result in indicator_gate_results.items():
            gate_results_serialized[ind_name] = {
                "presence_flag": gate_result.presence_flag,
                "days_above_threshold": gate_result.days_above_threshold,
                "days_evaluated": gate_result.days_evaluated,
                "window_size": gate_result.window_size,
                "threshold": gate_result.threshold,
                "insufficient_data": gate_result.insufficient_data,
                "mean_likelihood": (
                    round(sum(indicator_daily_likelihoods[ind_name]) / len(indicator_daily_likelihoods[ind_name]), 4)
                    if indicator_daily_likelihoods.get(ind_name)
                    else 0.0
                ),
            }

        tracer.end_step(
            outputs={
                "episode_likely": episode_decision.episode_likely,
                "indicators_present": episode_decision.indicators_present,
                "min_indicators_required": episode_decision.min_indicators_required,
                "core_indicator_present": episode_decision.core_indicator_present,
                "core_indicators_present": list(episode_decision.core_indicators_present),
                "core_indicators_required": list(analysis_config.episode.core_indicators),
                "decision_rationale": episode_decision.decision_rationale,
                "gate_results": gate_results_serialized,
                "dsm_params": {
                    "theta": analysis_config.dsm_gate_defaults.theta,
                    "m_window": analysis_config.dsm_gate_defaults.m_window,
                    "gate_need": analysis_config.dsm_gate_defaults.gate_need,
                },
            }
        )

        logger.info(
            "Episode decision: likely=%s, indicators=%d/%d, core=%s",
            episode_decision.episode_likely,
            episode_decision.indicators_present,
            episode_decision.min_indicators_required,
            episode_decision.core_indicator_present,
        )

        # Step 9: Persist results
        logger.info("[Step 9/10] Persisting results")
        if progress_callback:
            progress_callback(9, 10, "Persisting results")
        tracer.start_step(
            "Persist Results",
            inputs={
                "run_id": str(run_id),
                "daily_summaries_count": len(all_daily_summaries),
            },
        )

        # Save window indicators (Story 6.17: persist for Step 6 transparency)
        window_indicators_saved = save_window_indicators(
            window_indicators=all_window_indicators,
            user_id=user_id,
            analysis_run_id=run_id,
            session=session,
        )

        # Save daily summaries
        summaries_saved = save_daily_summaries(
            daily_summaries=all_daily_summaries,
            user_id=user_id,
            analysis_run_id=run_id,
            session=session,
        )

        # Save analysis run metadata
        save_analysis_run(
            analysis_run_id=run_id,
            user_id=user_id,
            config=analysis_config,
            start_time=start_time,
            end_time=end_time,
            session=session,
        )

        tracer.end_step(
            outputs={
                "window_indicators_saved": window_indicators_saved,
                "summaries_saved": summaries_saved,
                "run_saved": True,
            }
        )

        # Step 10: Build and save pipeline trace
        logger.info("[Step 10/10] Saving pipeline trace")
        if progress_callback:
            progress_callback(10, 10, "Saving pipeline trace")
        trace = tracer.get_trace()
        save_pipeline_trace(trace, run_id, session)

        # Commit if we own the session
        if owns_session:
            session.commit()

        logger.info("Results persisted to database")

        # Calculate duration
        duration_ms = int((time.perf_counter() - start_perf) * 1000)

        # Build config snapshot with baseline info (AC6)
        config_snapshot = analysis_config.to_dict()
        config_snapshot["baseline"] = {
            "strategy": "file",
            "source": baseline_config.metadata.name if baseline_config.metadata else "uploaded",
            "biomarkers": list(baseline_config.baselines.keys()),
        }

        # Build result
        result = WindowedAnalysisResult(
            run_id=str(run_id),
            user_id=user_id,
            start_date=start_time.date(),
            end_date=end_time.date(),
            daily_summaries=tuple(all_daily_summaries),
            window_count=total_windows,
            context_evaluations_added=context_evaluations_added,
            duration_ms=duration_ms,
            config_snapshot=config_snapshot,
        )

        logger.info(
            "Analysis run '%s' completed in %d ms: "
            "%d windows, %d daily summaries",
            run_id,
            duration_ms,
            total_windows,
            len(all_daily_summaries),
        )

        return result

    except AnalysisError:
        # Re-raise AnalysisError as-is
        if owns_session:
            session.rollback()
        raise

    except Exception as e:
        if owns_session:
            session.rollback()
        logger.error(
            "Analysis run '%s' failed: %s",
            run_id,
            e,
            exc_info=True,
        )
        raise AnalysisError(
            f"Analysis failed: {e}",
            run_id=run_id,
            step="unknown",
        ) from e

    finally:
        if owns_session and session_gen is not None:
            try:
                next(session_gen, None)  # Close generator
            except Exception:
                pass
