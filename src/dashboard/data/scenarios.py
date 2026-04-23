"""Scenario data operations for testing the analysis pipeline.

Provides functions for:
- Listing and describing predefined test scenarios
- Generating mock data for scenarios
- Resetting user test data
- Running analysis on scenario data
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, func, select

from src.core.analysis import WindowedAnalysisResult, run_analysis
from src.core.mock_data import (
    MockDataOrchestrator,
    load_mock_config,
    save_biomarkers,
    save_context,
)
from src.dashboard.components.filters import get_display_timezone
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import (
    AnalysisRun,
    Biomarker,
    Context,
    ContextEvaluationRun,
    ContextHistoryRecord,
    Indicator,
    UserBaseline,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ScenarioInfo:
    """Information about a scenario preset."""

    name: str
    description: str
    expected_context: str
    expected_behavior: str
    config_file: str


# Scenario definitions with expected behaviors
SCENARIOS: dict[str, ScenarioInfo] = {
    "neutral": ScenarioInfo(
        name="Neutral",
        description="Neutral scenario without scenario-specific overrides — pure baseline behavior.",
        expected_context="neutral",
        expected_behavior="No scenario overrides applied. All biomarkers and context markers use default baseline values.",
        config_file="config/mock_data/scenarios/neutral.yaml",
    ),
    "solitary_digital": ScenarioInfo(
        name="Solitary Digital Immersion",
        description="Young adult alone in bed browsing social media — high network activity, no social interaction, quiet environment. Active 20:00–24:00, two nights on / one night off.",
        expected_context="solitary_digital",
        expected_behavior="Network biomarkers weighted higher (1.5×); speech biomarkers weighted lower (0.5×). Scheduled: only active during evening hours, two nights on / one night off.",
        config_file="config/mock_data/scenarios/solitary_digital.yaml",
    ),
    "adversarial_social_digital_gaming": ScenarioInfo(
        name="Adversarial Social Digital Gaming",
        description="Young adult alone in room playing games online with friends — high network activity, high social interaction, loud environment. Active 20:00–24:00 daily.",
        expected_context="adversarial_social_digital_gaming",
        expected_behavior="Network biomarkers weighted higher (1.5×); speech biomarkers weighted lower (0.5×). Loud ambient noise differentiates from solitary_digital.",
        config_file="config/mock_data/scenarios/adversarial_social_digital_gaming.yaml",
    ),
    "adversarial_social_digital_gaming_good": ScenarioInfo(
        name="Adversarial Social Digital Gaming (with audio_source_digital)",
        description="Same conditions as adversarial scenario, but with audio_source_digital parameter indicating noise originates from digital sources. Expected: solitary_digital detected.",
        expected_context="solitary_digital",
        expected_behavior="Additional audio_source_digital marker (high) enables the system to recognise digital audio sources, compensating for high ambient noise. Result: solitary_digital context correctly detected.",
        config_file="config/mock_data/scenarios/adversarial_social_digital_gaming_good.yaml",
    ),
}


def get_available_scenarios() -> list[str]:
    """Get list of available scenario names.

    Returns:
        List of scenario key names
    """
    return list(SCENARIOS.keys())


def get_scenario_info(scenario_name: str) -> ScenarioInfo | None:
    """Get details for a specific scenario.

    Args:
        scenario_name: Key name of the scenario

    Returns:
        ScenarioInfo or None if not found
    """
    return SCENARIOS.get(scenario_name)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for mock data generation."""

    scenario: str
    user_id: str
    days: int
    seed: int | None
    biomarker_interval: int
    context_interval: int
    modalities: list[str] | None  # None = all


@dataclass(frozen=True)
class GenerationResult:
    """Result of mock data generation."""

    biomarker_count: int
    context_count: int
    start_time: datetime
    end_time: datetime
    scenario: str
    modalities_generated: list[str]


def generate_scenario_data(config: GenerationConfig) -> GenerationResult:
    """Generate mock data for a scenario.

    Args:
        config: Generation configuration

    Returns:
        GenerationResult with counts and metadata
    """
    mock_config = load_mock_config()

    orchestrator = MockDataOrchestrator(
        config=mock_config,
        seed=config.seed,
        scenario=config.scenario,
    )

    tz = get_display_timezone()
    now = datetime.now(tz)
    end_time = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_time = end_time - timedelta(days=config.days)

    biomarkers, context_markers = orchestrator.generate_all(
        user_id=config.user_id,
        start_time=start_time,
        end_time=end_time,
        biomarker_interval=config.biomarker_interval,
        context_interval=config.context_interval,
        modalities=config.modalities,
    )

    # Determine which modalities were generated
    if config.modalities:
        modalities_generated = config.modalities
    else:
        modalities_generated = list(mock_config.biomarkers.modalities.keys())

    # Save to database
    with SessionLocal() as session:
        save_biomarkers(biomarkers, session)
        save_context(context_markers, session)

    logger.info(
        "Generated %d biomarkers, %d context markers for scenario '%s'",
        len(biomarkers),
        len(context_markers),
        config.scenario,
    )

    return GenerationResult(
        biomarker_count=len(biomarkers),
        context_count=len(context_markers),
        start_time=start_time,
        end_time=end_time,
        scenario=config.scenario,
        modalities_generated=modalities_generated,
    )


@dataclass(frozen=True)
class ResetResult:
    """Result of data reset operation."""

    biomarkers_deleted: int
    context_deleted: int
    indicators_deleted: int
    analysis_runs_deleted: int
    context_history_deleted: int = 0
    user_baselines_deleted: int = 0
    context_evaluation_runs_deleted: int = 0


def reset_user_data(user_id: str) -> ResetResult:
    """Delete all data for a user.

    Deletes all user data from all tables:
    - biomarkers
    - context
    - indicators
    - analysis_runs
    - context_history
    - user_baselines

    Uses transaction for atomicity.

    Args:
        user_id: User ID to delete data for

    Returns:
        ResetResult with counts of deleted records
    """
    with SessionLocal() as session:
        # Count before deletion
        bio_count = session.execute(
            select(func.count())
            .select_from(Biomarker)
            .where(Biomarker.user_id == user_id)
        ).scalar_one()

        ctx_count = session.execute(
            select(func.count()).select_from(Context).where(Context.user_id == user_id)
        ).scalar_one()

        ind_count = session.execute(
            select(func.count())
            .select_from(Indicator)
            .where(Indicator.user_id == user_id)
        ).scalar_one()

        run_count = session.execute(
            select(func.count())
            .select_from(AnalysisRun)
            .where(AnalysisRun.user_id == user_id)
        ).scalar_one()

        ctx_history_count = session.execute(
            select(func.count())
            .select_from(ContextHistoryRecord)
            .where(ContextHistoryRecord.user_id == user_id)
        ).scalar_one()

        baseline_count = session.execute(
            select(func.count())
            .select_from(UserBaseline)
            .where(UserBaseline.user_id == user_id)
        ).scalar_one()

        context_eval_run_count = session.execute(
            select(func.count())
            .select_from(ContextEvaluationRun)
            .where(ContextEvaluationRun.user_id == user_id)
        ).scalar_one()

        # Delete in order (respect foreign key constraints)
        # 1. Indicators reference analysis_runs
        session.execute(delete(Indicator).where(Indicator.user_id == user_id))
        # 2. Analysis runs
        session.execute(delete(AnalysisRun).where(AnalysisRun.user_id == user_id))
        # 3. Context history records (may reference context_evaluation_runs)
        session.execute(
            delete(ContextHistoryRecord).where(ContextHistoryRecord.user_id == user_id)
        )
        # 4. Context evaluation runs (after context history due to FK)
        session.execute(
            delete(ContextEvaluationRun).where(ContextEvaluationRun.user_id == user_id)
        )
        # 5. User baselines
        session.execute(delete(UserBaseline).where(UserBaseline.user_id == user_id))
        # 6. Context markers
        session.execute(delete(Context).where(Context.user_id == user_id))
        # 7. Biomarkers
        session.execute(delete(Biomarker).where(Biomarker.user_id == user_id))

        session.commit()

    logger.info(
        "Reset data for user '%s': %d biomarkers, %d context, %d indicators, "
        "%d runs, %d context_history, %d baselines, %d context_eval_runs",
        user_id,
        bio_count,
        ctx_count,
        ind_count,
        run_count,
        ctx_history_count,
        baseline_count,
        context_eval_run_count,
    )

    return ResetResult(
        biomarkers_deleted=bio_count,
        context_deleted=ctx_count,
        indicators_deleted=ind_count,
        analysis_runs_deleted=run_count,
        context_history_deleted=ctx_history_count,
        user_baselines_deleted=baseline_count,
        context_evaluation_runs_deleted=context_eval_run_count,
    )


def check_user_has_data(user_id: str) -> bool:
    """Check if user has any biomarker data.

    Args:
        user_id: User ID to check

    Returns:
        True if user has biomarker records
    """
    with SessionLocal() as session:
        count = session.execute(
            select(func.count())
            .select_from(Biomarker)
            .where(Biomarker.user_id == user_id)
        ).scalar_one()
        return count > 0


def get_user_data_time_range(user_id: str) -> tuple[datetime, datetime] | None:
    """Get time range of user's biomarker data.

    Args:
        user_id: User ID to check

    Returns:
        Tuple of (min_timestamp, max_timestamp) or None if no data
    """
    with SessionLocal() as session:
        result = session.execute(
            select(
                func.min(Biomarker.timestamp),
                func.max(Biomarker.timestamp),
            ).where(Biomarker.user_id == user_id)
        ).one()

        if result[0] is None:
            return None

        return (result[0], result[1])


def run_scenario_analysis(user_id: str) -> WindowedAnalysisResult | None:
    """Run analysis on scenario data.

    Gets time range from existing data and runs analysis.

    Args:
        user_id: User ID to analyze

    Returns:
        WindowedAnalysisResult or None if no data
    """
    time_range = get_user_data_time_range(user_id)
    if time_range is None:
        return None

    start_time, end_time = time_range

    return run_analysis(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
    )
