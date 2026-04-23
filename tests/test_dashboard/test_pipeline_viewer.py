"""Tests for pipeline_viewer component."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.core.pipeline import PipelineStep, PipelineTrace
from src.dashboard.components.pipeline_viewer import (
    render_config_snapshot,
    render_pipeline_flow,
    render_pipeline_steps,
)


@pytest.fixture
def sample_trace():
    """Create a sample PipelineTrace for testing."""
    return PipelineTrace(
        analysis_run_id="test-run-123",
        user_id="test-user",
        started_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2024, 1, 15, 10, 0, 1, tzinfo=UTC),
        total_duration_ms=1000,
        steps=(
            PipelineStep(
                step_name="data_retrieval",
                step_number=1,
                inputs={"user_id": "test-user"},
                outputs={"record_count": 100, "success": True},
                duration_ms=45,
                metadata={"source": "database"},
                timestamp=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            ),
            PipelineStep(
                step_name="context_evaluation",
                step_number=2,
                inputs={"records": 100},
                outputs={"context": "normal", "confidence": 0.85},
                duration_ms=23,
                metadata={},
                timestamp=datetime(2024, 1, 15, 10, 0, 0, 45000, tzinfo=UTC),
            ),
            PipelineStep(
                step_name="biomarker_processing",
                step_number=3,
                inputs={},
                outputs={},
                duration_ms=67,
                metadata={},
                timestamp=datetime(2024, 1, 15, 10, 0, 0, 68000, tzinfo=UTC),
            ),
        ),
    )


@pytest.fixture
def empty_trace():
    """Create a PipelineTrace with no steps."""
    return PipelineTrace(
        analysis_run_id="empty-run",
        user_id="test-user",
        started_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
        total_duration_ms=0,
        steps=(),
    )


class TestRenderPipelineFlow:
    """Tests for render_pipeline_flow function."""

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_renders_flow_diagram(self, mock_st, sample_trace):
        """Test that flow diagram is rendered with correct format."""
        render_pipeline_flow(sample_trace)

        mock_st.markdown.assert_called_once()
        call_args = mock_st.markdown.call_args[0][0]
        assert "[1] data_retrieval" in call_args
        assert "[2] context_evaluation" in call_args
        assert "[3] biomarker_processing" in call_args
        assert " → " in call_args

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_handles_empty_trace(self, mock_st, empty_trace):
        """Test that empty trace shows appropriate message."""
        render_pipeline_flow(empty_trace)

        mock_st.caption.assert_called_once_with("No pipeline steps recorded")
        mock_st.markdown.assert_not_called()


class TestRenderPipelineSteps:
    """Tests for render_pipeline_steps function."""

    @patch("src.dashboard.components.pipeline_viewer.get_display_timezone")
    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_renders_all_steps(self, mock_st, mock_tz, sample_trace):
        """Test that all steps are rendered with expanders."""
        mock_tz.return_value = UTC
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]

        render_pipeline_steps(sample_trace)

        # Should create expander for each step
        assert mock_st.expander.call_count == 3

        # Check expander titles include step info
        expander_calls = [call[0][0] for call in mock_st.expander.call_args_list]
        assert "Step 1: data_retrieval (45 ms)" in expander_calls
        assert "Step 2: context_evaluation (23 ms)" in expander_calls
        assert "Step 3: biomarker_processing (67 ms)" in expander_calls

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_handles_empty_trace(self, mock_st, empty_trace):
        """Test that empty trace shows appropriate message."""
        render_pipeline_steps(empty_trace)

        mock_st.caption.assert_called_once_with("No pipeline steps recorded")
        mock_st.expander.assert_not_called()


class TestRenderConfigSnapshot:
    """Tests for render_config_snapshot function."""

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_renders_config_with_default_label(self, mock_st):
        """Test that config is rendered with default label."""
        config = {"key": "value", "nested": {"a": 1}}

        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        render_config_snapshot(config)

        mock_st.expander.assert_called_once_with("View Configuration Snapshot", expanded=False)

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_renders_config_with_custom_label(self, mock_st):
        """Test that config is rendered with custom label."""
        config = {"key": "value"}

        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        render_config_snapshot(config, label="Custom Label")

        mock_st.expander.assert_called_once_with("Custom Label", expanded=False)

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_handles_none_config(self, mock_st):
        """Test that None config shows appropriate message."""
        render_config_snapshot(None)

        mock_st.caption.assert_called_once_with("No configuration snapshot available for this run.")
        mock_st.expander.assert_not_called()

    @patch("src.dashboard.components.pipeline_viewer.st")
    def test_handles_empty_config(self, mock_st):
        """Test that empty dict config is still rendered."""
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)

        render_config_snapshot({})

        # Empty dict is falsy but should still render
        mock_st.caption.assert_called_once()
