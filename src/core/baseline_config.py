"""Baseline configuration models and file loading.

Story 4.14: Baseline Configuration & Selection
Tasks 2.1-2.6: Pydantic models and utility functions for baseline files.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BaselineDefinition",
    "BaselineFile",
    "BaselineMetadata",
    "list_available_baselines",
    "load_baseline_file",
]

logger = logging.getLogger(__name__)


class BaselineDefinition(BaseModel):
    """Definition of baseline statistics for a single biomarker.

    AC5: Validate required fields (mean, std) and types.
    """

    model_config = ConfigDict(strict=True)

    mean: float
    std: float = Field(gt=0, description="Standard deviation (must be > 0)")


class BaselineMetadata(BaseModel):
    """Optional metadata about a baseline file."""

    model_config = ConfigDict(strict=True)

    name: str | None = None
    description: str | None = None
    created: str | None = None
    version: str | None = None


class BaselineFile(BaseModel):
    """A baseline file containing metadata and baseline definitions.

    AC2, AC5: Schema for baseline files.
    """

    model_config = ConfigDict(strict=True)

    metadata: BaselineMetadata | None = None
    baselines: dict[str, BaselineDefinition]


def load_baseline_file(path: Path) -> BaselineFile:
    """Load and validate a baseline file from disk.

    AC2: Load baseline file with validation.

    Args:
        path: Path to the baseline JSON file.

    Returns:
        Validated BaselineFile instance.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
        ValidationError: If file doesn't match schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    logger.debug(f"Loading baseline file: {path}")
    content = path.read_text()
    data = json.loads(content)

    baseline_file = BaselineFile.model_validate(data)
    logger.info(
        f"Loaded baseline file: {path.name} with {len(baseline_file.baselines)} biomarkers"
    )
    return baseline_file


def list_available_baselines(baselines_dir: Path) -> list[str]:
    """List available baseline files in a directory.

    AC3: List .json files for dropdown selection.

    Args:
        baselines_dir: Path to directory containing baseline files.

    Returns:
        Sorted list of baseline names (filenames without .json extension).
    """
    if not baselines_dir.exists():
        logger.warning(f"Baselines directory does not exist: {baselines_dir}")
        return []

    json_files = list(baselines_dir.glob("*.json"))
    names = [f.stem for f in json_files]
    names.sort()

    logger.debug(f"Found {len(names)} baseline files in {baselines_dir}")
    return names
