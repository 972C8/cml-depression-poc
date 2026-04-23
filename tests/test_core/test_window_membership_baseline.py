"""Tests for window membership baseline resolution (strict mode).

Story 4.14: Baseline Configuration & Selection
Task 6.5: Unit tests for baseline resolution - no fallback, baseline is required.
"""

import pytest

from src.core.baseline_config import BaselineDefinition, BaselineFile, BaselineMetadata
from src.core.config import AnalysisConfig, get_default_config
from src.core.processors.window_membership import BaselineError, _get_baseline_stats


class TestBaselineResolutionStrict:
    """Tests for _get_baseline_stats strict mode (no fallback)."""

    @pytest.fixture
    def config(self) -> AnalysisConfig:
        """Get default analysis config."""
        return get_default_config()

    @pytest.fixture
    def file_baseline(self) -> BaselineFile:
        """Sample file-based baseline."""
        return BaselineFile(
            metadata=BaselineMetadata(name="Test Baseline"),
            baselines={
                "sleep_duration": BaselineDefinition(mean=7.5, std=0.9),
                "step_count": BaselineDefinition(mean=9000, std=2500),
            },
        )

    def test_returns_baseline_from_file(
        self, config: AnalysisConfig, file_baseline: BaselineFile
    ):
        """Returns mean and std from the baseline file."""
        mean, std = _get_baseline_stats(
            biomarker_name="sleep_duration",
            baseline_config=file_baseline,
            config=config,
        )

        assert mean == 7.5
        assert std == 0.9

    def test_raises_error_when_biomarker_missing(
        self, config: AnalysisConfig, file_baseline: BaselineFile
    ):
        """Raises BaselineError when biomarker not in baseline file."""
        with pytest.raises(BaselineError) as exc_info:
            _get_baseline_stats(
                biomarker_name="unknown_biomarker",
                baseline_config=file_baseline,
                config=config,
            )

        assert "unknown_biomarker" in str(exc_info.value)
        assert "missing" in str(exc_info.value).lower()

    def test_error_message_includes_available_biomarkers(
        self, config: AnalysisConfig, file_baseline: BaselineFile
    ):
        """Error message lists available biomarkers for debugging."""
        with pytest.raises(BaselineError) as exc_info:
            _get_baseline_stats(
                biomarker_name="missing",
                baseline_config=file_baseline,
                config=config,
            )

        error_msg = str(exc_info.value)
        assert "sleep_duration" in error_msg or "step_count" in error_msg

    def test_applies_minimum_std_floor(self, config: AnalysisConfig):
        """Applies minimum std floor when file std is too small."""
        file_baseline = BaselineFile(
            baselines={
                "test_biomarker": BaselineDefinition(mean=0.5, std=0.0001),
            }
        )

        mean, std = _get_baseline_stats(
            biomarker_name="test_biomarker",
            baseline_config=file_baseline,
            config=config,
        )

        assert mean == 0.5
        assert std >= config.biomarker_processing.min_std_deviation

    def test_all_biomarkers_must_be_in_file(self, config: AnalysisConfig):
        """Each biomarker used in analysis must be in the baseline file."""
        # File only has one biomarker
        file_baseline = BaselineFile(
            baselines={
                "only_one_biomarker": BaselineDefinition(mean=0.5, std=0.1),
            }
        )

        # First biomarker works
        mean, std = _get_baseline_stats(
            biomarker_name="only_one_biomarker",
            baseline_config=file_baseline,
            config=config,
        )
        assert mean == 0.5

        # Other biomarkers fail
        with pytest.raises(BaselineError):
            _get_baseline_stats(
                biomarker_name="speech_activity",
                baseline_config=file_baseline,
                config=config,
            )


class TestBaselineResolutionLogging:
    """Tests for baseline resolution logging."""

    @pytest.fixture
    def config(self) -> AnalysisConfig:
        return get_default_config()

    def test_logs_baseline_values(self, config: AnalysisConfig, caplog):
        """Logs when baseline is used."""
        import logging

        caplog.set_level(logging.INFO)

        file_baseline = BaselineFile(
            baselines={"test": BaselineDefinition(mean=0.5, std=0.1)}
        )

        _get_baseline_stats("test", file_baseline, config)

        assert "baseline" in caplog.text.lower()
        assert "test" in caplog.text

    def test_logs_warning_for_low_std(self, config: AnalysisConfig, caplog):
        """Logs warning when std is below minimum."""
        import logging

        caplog.set_level(logging.WARNING)

        file_baseline = BaselineFile(
            baselines={"test": BaselineDefinition(mean=0.5, std=0.00001)}
        )

        _get_baseline_stats("test", file_baseline, config)

        assert "below minimum" in caplog.text.lower() or "min_std" in caplog.text


class TestBaselineErrorMessage:
    """Tests for BaselineError message formatting."""

    def test_error_is_descriptive(self):
        """BaselineError provides actionable error message."""
        file_baseline = BaselineFile(
            baselines={
                "biomarker_a": BaselineDefinition(mean=0.5, std=0.1),
                "biomarker_b": BaselineDefinition(mean=0.6, std=0.2),
            }
        )
        config = get_default_config()

        with pytest.raises(BaselineError) as exc_info:
            _get_baseline_stats("biomarker_c", file_baseline, config)

        error_msg = str(exc_info.value)
        # Should mention the missing biomarker
        assert "biomarker_c" in error_msg
        # Should indicate what to do
        assert "baseline" in error_msg.lower()
        # Should list available options
        assert "biomarker_a" in error_msg or "biomarker_b" in error_msg
