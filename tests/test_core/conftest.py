"""Pytest fixtures for core module tests."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete

from src.core.mock_data import (
    MockDataConfig,
    MockDataOrchestrator,
    load_mock_config,
    save_biomarkers,
    save_context,
)
from src.shared.models import Biomarker, Context


@pytest.fixture
def mock_data_config() -> MockDataConfig:
    """Load mock data configuration.

    Returns:
        MockDataConfig instance with loaded configuration
    """
    return load_mock_config()


@pytest.fixture
def mock_orchestrator(mock_data_config: MockDataConfig) -> MockDataOrchestrator:
    """Create orchestrator with fixed seed for reproducibility.

    Args:
        mock_data_config: Loaded mock data configuration

    Returns:
        MockDataOrchestrator with fixed seed
    """
    return MockDataOrchestrator(mock_data_config, seed=42)


@pytest.fixture
def mock_biomarkers(db_session, mock_orchestrator: MockDataOrchestrator):
    """Generate and save mock biomarker data.

    Args:
        db_session: Database session fixture
        mock_orchestrator: Mock data orchestrator fixture

    Yields:
        Dict with user_id, start_time, end_time, count
    """
    user_id = "test-user-001"
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=7)

    biomarkers = mock_orchestrator.generate_biomarkers(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=60,
    )

    count = save_biomarkers(biomarkers, db_session)

    yield {
        "user_id": user_id,
        "start_time": start_time,
        "end_time": end_time,
        "count": count,
    }

    # Cleanup: delete generated records (SQLAlchemy 2.0 style)
    db_session.execute(delete(Biomarker).where(Biomarker.user_id == user_id))
    db_session.commit()


@pytest.fixture
def mock_context(db_session, mock_orchestrator: MockDataOrchestrator):
    """Generate and save mock context data.

    Args:
        db_session: Database session fixture
        mock_orchestrator: Mock data orchestrator fixture

    Yields:
        Dict with user_id, start_time, end_time, count
    """
    user_id = "test-user-002"
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=7)

    context = mock_orchestrator.generate_context(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=60,
    )

    count = save_context(context, db_session)

    yield {
        "user_id": user_id,
        "start_time": start_time,
        "end_time": end_time,
        "count": count,
    }

    # Cleanup: delete generated records (SQLAlchemy 2.0 style)
    db_session.execute(delete(Context).where(Context.user_id == user_id))
    db_session.commit()


@pytest.fixture
def mock_scenario(db_session, mock_data_config: MockDataConfig, request):
    """Generate scenario-specific mock data.

    Usage:
        @pytest.mark.parametrize("mock_scenario", ["solitary_digital"], indirect=True)
        def test_social_context(mock_scenario):
            ...

    Args:
        db_session: Database session fixture
        mock_data_config: Mock data configuration fixture
        request: Pytest request object with scenario name as param

    Yields:
        Dict with user_id, start_time, end_time, scenario_name, bio_count, ctx_count
    """
    scenario_name = request.param
    user_id = f"test-scenario-{scenario_name}"
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=3)

    orchestrator = MockDataOrchestrator(
        mock_data_config, seed=42, scenario=scenario_name
    )

    biomarkers, context = orchestrator.generate_all(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        biomarker_interval=30,
        context_interval=60,
    )

    bio_count = save_biomarkers(biomarkers, db_session)
    ctx_count = save_context(context, db_session)

    yield {
        "user_id": user_id,
        "start_time": start_time,
        "end_time": end_time,
        "scenario_name": scenario_name,
        "bio_count": bio_count,
        "ctx_count": ctx_count,
    }

    # Cleanup (SQLAlchemy 2.0 style)
    db_session.execute(delete(Biomarker).where(Biomarker.user_id == user_id))
    db_session.execute(delete(Context).where(Context.user_id == user_id))
    db_session.commit()
