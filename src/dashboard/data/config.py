"""Configuration data loading functions for the dashboard."""

import streamlit as st
import yaml

from src.core.config import AnalysisConfig, get_default_config
from src.shared.logging import get_logger

logger = get_logger(__name__)


@st.cache_resource
def get_current_config() -> AnalysisConfig:
    """Get the currently active analysis configuration.

    Currently returns the hardcoded default config.
    Future: Could load from config/analysis.yaml if it exists.

    Returns:
        AnalysisConfig instance
    """
    logger.info("Loading analysis configuration")
    return get_default_config()


def reload_config() -> AnalysisConfig:
    """Force reload of configuration, clearing cache.

    Returns:
        Fresh AnalysisConfig instance
    """
    get_current_config.clear()
    logger.info("Configuration cache cleared, reloading")
    return get_current_config()


def config_to_yaml(config: AnalysisConfig) -> str:
    """Convert config to YAML string for display.

    Args:
        config: AnalysisConfig instance

    Returns:
        YAML formatted string
    """
    return yaml.dump(config.to_dict(), default_flow_style=False, sort_keys=False)
