"""Experiment data loading and management functions."""

import uuid

import pandas as pd
from sqlalchemy import select

from src.core.config import AnalysisConfig
from src.shared.database import SessionLocal
from src.shared.logging import get_logger
from src.shared.models import ConfigExperiment

logger = get_logger(__name__)


def create_experiment(
    name: str,
    config: AnalysisConfig,
    description: str | None = None,
) -> ConfigExperiment:
    """Create a new configuration experiment.

    Args:
        name: Experiment name
        config: AnalysisConfig to store as snapshot
        description: Optional description

    Returns:
        Created ConfigExperiment instance
    """
    with SessionLocal() as session:
        experiment = ConfigExperiment(
            name=name,
            description=description,
            config_snapshot=config.to_dict(),
        )
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        logger.info("Created experiment: %s (%s)", name, experiment.id)
        # Expunge to use outside session
        session.expunge(experiment)
        return experiment


def list_experiments(limit: int = 20) -> pd.DataFrame:
    """List all configuration experiments.

    Args:
        limit: Maximum number of experiments to return

    Returns:
        DataFrame with experiment metadata
    """
    try:
        with SessionLocal() as session:
            stmt = (
                select(ConfigExperiment)
                .order_by(ConfigExperiment.created_at.desc())
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()

        if not results:
            return pd.DataFrame(columns=["id", "name", "description", "created_at"])

        data = [
            {
                "id": str(exp.id),
                "name": exp.name,
                "description": exp.description or "",
                "created_at": exp.created_at,
            }
            for exp in results
        ]
        return pd.DataFrame(data)

    except Exception:
        logger.error("Failed to list experiments", exc_info=True)
        return pd.DataFrame(columns=["id", "name", "description", "created_at"])


def get_experiment(experiment_id: str) -> ConfigExperiment | None:
    """Get experiment by ID.

    Args:
        experiment_id: UUID string

    Returns:
        ConfigExperiment or None if not found
    """
    try:
        with SessionLocal() as session:
            stmt = select(ConfigExperiment).where(
                ConfigExperiment.id == uuid.UUID(experiment_id)
            )
            experiment = session.execute(stmt).scalar_one_or_none()
            if experiment:
                session.expunge(experiment)
            return experiment

    except Exception:
        logger.error("Failed to get experiment %s", experiment_id, exc_info=True)
        return None


def get_experiment_config(experiment_id: str) -> AnalysisConfig | None:
    """Get AnalysisConfig from experiment.

    Args:
        experiment_id: UUID string

    Returns:
        AnalysisConfig instance or None
    """
    experiment = get_experiment(experiment_id)
    if experiment is None:
        return None
    try:
        return AnalysisConfig(**experiment.config_snapshot)
    except Exception:
        logger.error(
            "Failed to parse config for experiment %s", experiment_id, exc_info=True
        )
        return None


def update_experiment(
    experiment_id: str,
    config: AnalysisConfig,
    name: str | None = None,
    description: str | None = None,
) -> ConfigExperiment | None:
    """Update experiment configuration.

    Args:
        experiment_id: UUID string
        config: Updated AnalysisConfig
        name: Optional new name
        description: Optional new description

    Returns:
        Updated ConfigExperiment or None
    """
    try:
        with SessionLocal() as session:
            stmt = select(ConfigExperiment).where(
                ConfigExperiment.id == uuid.UUID(experiment_id)
            )
            experiment = session.execute(stmt).scalar_one_or_none()

            if experiment is None:
                return None

            experiment.config_snapshot = config.to_dict()
            if name is not None:
                experiment.name = name
            if description is not None:
                experiment.description = description

            session.commit()
            session.refresh(experiment)
            logger.info("Updated experiment: %s", experiment_id)
            session.expunge(experiment)
            return experiment

    except Exception:
        logger.error("Failed to update experiment %s", experiment_id, exc_info=True)
        return None


def delete_experiment(experiment_id: str) -> bool:
    """Delete experiment by ID.

    Args:
        experiment_id: UUID string

    Returns:
        True if deleted, False if not found
    """
    try:
        with SessionLocal() as session:
            stmt = select(ConfigExperiment).where(
                ConfigExperiment.id == uuid.UUID(experiment_id)
            )
            experiment = session.execute(stmt).scalar_one_or_none()

            if experiment is None:
                return False

            session.delete(experiment)
            session.commit()
            logger.info("Deleted experiment: %s", experiment_id)
            return True

    except Exception:
        logger.error("Failed to delete experiment %s", experiment_id, exc_info=True)
        return False
