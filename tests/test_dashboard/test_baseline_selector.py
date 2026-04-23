"""Tests for baseline selector component.

Story 4.14: Baseline Configuration & Selection
Task 3.7: Unit tests for component logic (validation, file loading, state management).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.baseline_config import BaselineDefinition, BaselineFile, BaselineMetadata
from src.dashboard.components.baseline_selector import (
    get_baselines_directory,
    init_baseline_session_state,
    load_baseline_from_upload,
    validate_baseline_for_analysis,
)


class TestGetBaselinesDirectory:
    """Tests for get_baselines_directory function."""

    def test_returns_path_object(self):
        """Returns a Path object."""
        result = get_baselines_directory()
        assert isinstance(result, Path)

    def test_returns_config_baselines_path(self):
        """Path ends with config/baselines."""
        result = get_baselines_directory()
        assert result.name == "baselines"
        assert result.parent.name == "config"


class TestInitBaselineSessionState:
    """Tests for init_baseline_session_state function."""

    @patch("src.dashboard.components.baseline_selector.st")
    def test_initializes_uploaded_content_key(self, mock_st):
        """Initializes baseline_uploaded_content session state key."""
        mock_st.session_state = {}
        init_baseline_session_state()
        assert "baseline_uploaded_content" in mock_st.session_state
        assert mock_st.session_state["baseline_uploaded_content"] is None

    @patch("src.dashboard.components.baseline_selector.st")
    def test_does_not_overwrite_existing_uploaded_content(self, mock_st):
        """Does not overwrite existing baseline_uploaded_content value."""
        existing_baseline = BaselineFile(baselines={})
        mock_st.session_state = {
            "baseline_uploaded_content": existing_baseline,
        }
        init_baseline_session_state()
        assert mock_st.session_state["baseline_uploaded_content"] is existing_baseline


class TestLoadBaselineFromUpload:
    """Tests for load_baseline_from_upload function."""

    def test_valid_json_content(self):
        """AC2: Successfully parse valid JSON baseline content."""
        content = json.dumps(
            {
                "metadata": {"name": "Test"},
                "baselines": {"test_biomarker": {"mean": 0.5, "std": 0.15}},
            }
        ).encode()

        result, error = load_baseline_from_upload(content)
        assert error is None
        assert result is not None
        assert result.metadata.name == "Test"
        assert "test_biomarker" in result.baselines

    def test_invalid_json(self):
        """AC2: Return error for invalid JSON."""
        content = b"not valid json {"

        result, error = load_baseline_from_upload(content)
        assert result is None
        assert error is not None
        assert "Invalid JSON" in error

    def test_missing_baselines_field(self):
        """AC2: Return error when baselines field missing."""
        content = json.dumps({"metadata": {"name": "Test"}}).encode()

        result, error = load_baseline_from_upload(content)
        assert result is None
        assert error is not None
        assert "baselines" in error.lower()

    def test_invalid_std_zero(self):
        """AC2, AC5: Return error when std is zero."""
        content = json.dumps(
            {"baselines": {"test": {"mean": 0.5, "std": 0}}}
        ).encode()

        result, error = load_baseline_from_upload(content)
        assert result is None
        assert error is not None
        assert "std" in error.lower()

    def test_invalid_std_negative(self):
        """AC2, AC5: Return error when std is negative."""
        content = json.dumps(
            {"baselines": {"test": {"mean": 0.5, "std": -0.1}}}
        ).encode()

        result, error = load_baseline_from_upload(content)
        assert result is None
        assert error is not None
        assert "std" in error.lower()

    def test_missing_mean(self):
        """AC2, AC5: Return error when mean is missing."""
        content = json.dumps({"baselines": {"test": {"std": 0.15}}}).encode()

        result, error = load_baseline_from_upload(content)
        assert result is None
        assert error is not None


class TestValidateBaselineForAnalysis:
    """Tests for validate_baseline_for_analysis function."""

    def test_valid_baseline_with_all_biomarkers(self):
        """AC5: Valid baseline passes validation."""
        baseline = BaselineFile(
            baselines={
                "sleep_duration": BaselineDefinition(mean=7.0, std=1.0),
                "step_count": BaselineDefinition(mean=8000, std=2000),
            }
        )
        required_biomarkers = ["sleep_duration", "step_count"]

        warnings = validate_baseline_for_analysis(baseline, required_biomarkers)
        assert len(warnings) == 0

    def test_missing_biomarker_warning(self):
        """AC5: Warn when baseline doesn't include required biomarker."""
        baseline = BaselineFile(
            baselines={"sleep_duration": BaselineDefinition(mean=7.0, std=1.0)}
        )
        required_biomarkers = ["sleep_duration", "step_count", "heart_rate"]

        warnings = validate_baseline_for_analysis(baseline, required_biomarkers)
        assert len(warnings) == 2  # step_count and heart_rate missing
        assert any("step_count" in w for w in warnings)
        assert any("heart_rate" in w for w in warnings)

    def test_empty_required_biomarkers(self):
        """No warnings when required list is empty."""
        baseline = BaselineFile(baselines={})
        warnings = validate_baseline_for_analysis(baseline, [])
        assert len(warnings) == 0

    def test_extra_biomarkers_no_warning(self):
        """Extra biomarkers in baseline don't produce warnings."""
        baseline = BaselineFile(
            baselines={
                "sleep_duration": BaselineDefinition(mean=7.0, std=1.0),
                "extra_biomarker": BaselineDefinition(mean=0.5, std=0.1),
            }
        )
        required_biomarkers = ["sleep_duration"]

        warnings = validate_baseline_for_analysis(baseline, required_biomarkers)
        assert len(warnings) == 0
