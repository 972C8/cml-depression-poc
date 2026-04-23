"""Tests for baseline configuration models and file loading.

Story 4.14: Baseline Configuration & Selection
Tasks 1.4, 2.7: Unit tests for baseline file format validation and model validation.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.baseline_config import (
    BaselineDefinition,
    BaselineFile,
    BaselineMetadata,
    list_available_baselines,
    load_baseline_file,
)


class TestBaselineDefinition:
    """Tests for BaselineDefinition model."""

    def test_valid_baseline_definition(self):
        """AC5: Valid baseline with mean and std."""
        baseline = BaselineDefinition(mean=0.5, std=0.15)
        assert baseline.mean == 0.5
        assert baseline.std == 0.15

    def test_std_must_be_positive(self):
        """AC5: Validate std > 0 (prevent division by zero)."""
        with pytest.raises(ValidationError) as exc_info:
            BaselineDefinition(mean=0.5, std=0.0)
        assert "std" in str(exc_info.value).lower()

    def test_negative_std_rejected(self):
        """AC5: Negative std should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            BaselineDefinition(mean=0.5, std=-0.1)
        assert "std" in str(exc_info.value).lower()

    def test_mean_can_be_negative(self):
        """Mean can be any number including negative."""
        baseline = BaselineDefinition(mean=-1.5, std=0.5)
        assert baseline.mean == -1.5

    def test_mean_can_be_zero(self):
        """Mean can be zero."""
        baseline = BaselineDefinition(mean=0.0, std=0.5)
        assert baseline.mean == 0.0


class TestBaselineMetadata:
    """Tests for BaselineMetadata model."""

    def test_valid_metadata(self):
        """All metadata fields can be set."""
        metadata = BaselineMetadata(
            name="Test Baseline",
            description="Test description",
            created="2026-01-29",
            version="1.0",
        )
        assert metadata.name == "Test Baseline"
        assert metadata.description == "Test description"
        assert metadata.created == "2026-01-29"
        assert metadata.version == "1.0"

    def test_metadata_all_optional(self):
        """All metadata fields are optional."""
        metadata = BaselineMetadata()
        assert metadata.name is None
        assert metadata.description is None
        assert metadata.created is None
        assert metadata.version is None


class TestBaselineFile:
    """Tests for BaselineFile model."""

    def test_valid_baseline_file(self):
        """AC5: Valid baseline file with metadata and baselines."""
        baseline_file = BaselineFile(
            metadata=BaselineMetadata(name="Test"),
            baselines={
                "sleep_duration": BaselineDefinition(mean=7.2, std=0.8),
                "step_count": BaselineDefinition(mean=8500, std=2100),
            },
        )
        assert baseline_file.metadata.name == "Test"
        assert len(baseline_file.baselines) == 2
        assert baseline_file.baselines["sleep_duration"].mean == 7.2

    def test_baselines_required(self):
        """baselines field is required."""
        with pytest.raises(ValidationError):
            BaselineFile(metadata=BaselineMetadata())

    def test_empty_baselines_allowed(self):
        """Empty baselines dict is valid (though not useful)."""
        baseline_file = BaselineFile(baselines={})
        assert baseline_file.baselines == {}

    def test_metadata_optional(self):
        """metadata field is optional."""
        baseline_file = BaselineFile(
            baselines={"test": BaselineDefinition(mean=0.5, std=0.1)}
        )
        assert baseline_file.metadata is None


class TestLoadBaselineFile:
    """Tests for load_baseline_file function."""

    def test_load_valid_file(self, tmp_path: Path):
        """AC2: Load valid baseline file."""
        file_path = tmp_path / "test_baseline.json"
        content = {
            "metadata": {"name": "Test", "version": "1.0"},
            "baselines": {
                "sleep_duration": {"mean": 7.2, "std": 0.8}
            },
        }
        file_path.write_text(json.dumps(content))

        result = load_baseline_file(file_path)
        assert result.metadata.name == "Test"
        assert result.baselines["sleep_duration"].mean == 7.2

    def test_load_file_without_metadata(self, tmp_path: Path):
        """File without metadata section is valid."""
        file_path = tmp_path / "no_meta.json"
        content = {"baselines": {"test": {"mean": 0.5, "std": 0.1}}}
        file_path.write_text(json.dumps(content))

        result = load_baseline_file(file_path)
        assert result.metadata is None
        assert result.baselines["test"].mean == 0.5

    def test_load_nonexistent_file(self, tmp_path: Path):
        """AC2: Error for nonexistent file."""
        file_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_baseline_file(file_path)

    def test_load_invalid_json(self, tmp_path: Path):
        """AC2: Error for invalid JSON."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json {")
        with pytest.raises(json.JSONDecodeError):
            load_baseline_file(file_path)

    def test_load_file_missing_baselines(self, tmp_path: Path):
        """AC2: Error when baselines field is missing."""
        file_path = tmp_path / "no_baselines.json"
        content = {"metadata": {"name": "Test"}}
        file_path.write_text(json.dumps(content))

        with pytest.raises(ValidationError):
            load_baseline_file(file_path)

    def test_load_file_invalid_std(self, tmp_path: Path):
        """AC2, AC5: Error when std <= 0."""
        file_path = tmp_path / "bad_std.json"
        content = {"baselines": {"test": {"mean": 0.5, "std": 0}}}
        file_path.write_text(json.dumps(content))

        with pytest.raises(ValidationError) as exc_info:
            load_baseline_file(file_path)
        assert "std" in str(exc_info.value).lower()

    def test_load_file_missing_mean(self, tmp_path: Path):
        """AC5: Error when mean is missing."""
        file_path = tmp_path / "no_mean.json"
        content = {"baselines": {"test": {"std": 0.1}}}
        file_path.write_text(json.dumps(content))

        with pytest.raises(ValidationError):
            load_baseline_file(file_path)

    def test_load_file_missing_std(self, tmp_path: Path):
        """AC5: Error when std is missing."""
        file_path = tmp_path / "no_std.json"
        content = {"baselines": {"test": {"mean": 0.5}}}
        file_path.write_text(json.dumps(content))

        with pytest.raises(ValidationError):
            load_baseline_file(file_path)


class TestListAvailableBaselines:
    """Tests for list_available_baselines function."""

    def test_list_empty_directory(self, tmp_path: Path):
        """AC3: Empty directory returns empty list."""
        result = list_available_baselines(tmp_path)
        assert result == []

    def test_list_json_files_only(self, tmp_path: Path):
        """AC3: Only .json files are listed."""
        (tmp_path / "baseline1.json").write_text("{}")
        (tmp_path / "baseline2.json").write_text("{}")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "config.yaml").write_text("key: value")

        result = list_available_baselines(tmp_path)
        assert len(result) == 2
        assert "baseline1" in result
        assert "baseline2" in result
        assert "readme" not in result

    def test_list_returns_names_without_extension(self, tmp_path: Path):
        """AC3: Return filenames without .json extension."""
        (tmp_path / "population_default.json").write_text("{}")

        result = list_available_baselines(tmp_path)
        assert result == ["population_default"]

    def test_list_sorted_alphabetically(self, tmp_path: Path):
        """AC3: Results are sorted alphabetically."""
        (tmp_path / "zebra.json").write_text("{}")
        (tmp_path / "alpha.json").write_text("{}")
        (tmp_path / "middle.json").write_text("{}")

        result = list_available_baselines(tmp_path)
        assert result == ["alpha", "middle", "zebra"]

    def test_list_nonexistent_directory(self, tmp_path: Path):
        """AC3: Nonexistent directory returns empty list."""
        nonexistent = tmp_path / "does_not_exist"
        result = list_available_baselines(nonexistent)
        assert result == []


class TestPopulationDefaultFile:
    """Test the shipped population_default.json file."""

    def test_population_default_loads_successfully(self):
        """AC8: population_default.json loads without errors."""
        baselines_dir = Path(__file__).parent.parent.parent / "config" / "baselines"
        default_file = baselines_dir / "population_default.json"

        if default_file.exists():
            result = load_baseline_file(default_file)
            assert result.metadata is not None
            assert result.metadata.name == "Population Default Baseline"
            assert len(result.baselines) > 0

    def test_population_default_has_required_biomarkers(self):
        """AC8: population_default.json has sensible defaults for all biomarkers."""
        baselines_dir = Path(__file__).parent.parent.parent / "config" / "baselines"
        default_file = baselines_dir / "population_default.json"

        if default_file.exists():
            result = load_baseline_file(default_file)
            required_biomarkers = [
                "speech_activity",
                "voice_energy",
                "connections",
                "bytes_in",
                "bytes_out",
                "speech_rate",
                "network_variety",
                "activity_level",
                "sleep_duration",
                "awakenings",
                "speech_activity_night",
            ]
            for biomarker in required_biomarkers:
                assert biomarker in result.baselines, f"Missing {biomarker}"
                assert result.baselines[biomarker].std > 0
