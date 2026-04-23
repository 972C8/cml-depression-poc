"""Tests for results_summary component."""

from unittest.mock import MagicMock, patch

import pytest

from src.dashboard.components.results_summary import (
    _extract_episode_decision,
    _extract_gate_results,
    render_episode_summary,
)


@pytest.fixture
def sample_pipeline_trace():
    """Create a sample pipeline trace for testing."""
    return {
        "analysis_run_id": "test-run-123",
        "user_id": "test-user",
        "steps": [
            {
                "step_name": "Read Data",
                "step_number": 1,
                "inputs": {"user_id": "test-user"},
                "outputs": {"biomarker_count": 100},
                "duration_ms": 45,
            },
            {
                "step_name": "Apply DSM-Gate",
                "step_number": 5,
                "inputs": {"indicator_count": 9},
                "outputs": {
                    "indicators_present": 6,
                    "indicators_total": 9,
                    "gate_results": {
                        "depressed_mood": {
                            "presence_flag": True,
                            "days_above_threshold": 5,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "anhedonia": {
                            "presence_flag": True,
                            "days_above_threshold": 6,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "appetite_change": {
                            "presence_flag": True,
                            "days_above_threshold": 4,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "sleep_disturbance": {
                            "presence_flag": True,
                            "days_above_threshold": 7,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "psychomotor_change": {
                            "presence_flag": False,
                            "days_above_threshold": 2,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "fatigue": {
                            "presence_flag": True,
                            "days_above_threshold": 5,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "worthlessness": {
                            "presence_flag": False,
                            "days_above_threshold": 1,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "concentration_issues": {
                            "presence_flag": True,
                            "days_above_threshold": 4,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                        "suicidal_ideation": {
                            "presence_flag": False,
                            "days_above_threshold": 0,
                            "days_evaluated": 7,
                            "threshold": 0.5,
                        },
                    },
                },
                "duration_ms": 23,
            },
            {
                "step_name": "Episode Decision",
                "step_number": 6,
                "inputs": {"gate_result_count": 9},
                "outputs": {
                    "episode_likely": True,
                    "indicators_present": 6,
                    "min_indicators_required": 5,
                    "rationale": "6 of 9 indicators present; (>=5 required: MET); Core indicator 'depressed_mood' present; => Episode criteria MET",
                },
                "duration_ms": 5,
            },
        ],
    }


@pytest.fixture
def not_met_pipeline_trace():
    """Create a pipeline trace for NOT MET case."""
    return {
        "steps": [
            {
                "step_name": "Apply DSM-Gate",
                "step_number": 5,
                "outputs": {
                    "indicators_present": 3,
                    "indicators_total": 9,
                    "gate_results": {
                        "depressed_mood": {
                            "presence_flag": False,
                            "days_above_threshold": 2,
                            "days_evaluated": 7,
                        },
                        "anhedonia": {
                            "presence_flag": True,
                            "days_above_threshold": 4,
                            "days_evaluated": 7,
                        },
                        "fatigue": {
                            "presence_flag": True,
                            "days_above_threshold": 5,
                            "days_evaluated": 7,
                        },
                        "sleep_disturbance": {
                            "presence_flag": True,
                            "days_above_threshold": 4,
                            "days_evaluated": 7,
                        },
                    },
                },
            },
            {
                "step_name": "Episode Decision",
                "step_number": 6,
                "outputs": {
                    "episode_likely": False,
                    "indicators_present": 3,
                    "min_indicators_required": 5,
                    "rationale": "3 of 9 indicators present; (>=5 required: NOT MET); No core indicator present; => Episode criteria NOT MET",
                },
            },
        ],
    }


class TestExtractEpisodeDecision:
    """Tests for _extract_episode_decision function."""

    def test_extracts_decision_from_trace(self, sample_pipeline_trace):
        """Test that episode decision is extracted correctly."""
        result = _extract_episode_decision(sample_pipeline_trace)

        assert result is not None
        assert result["episode_likely"] is True
        assert result["indicators_present"] == 6
        assert result["min_indicators_required"] == 5
        assert "MET" in result["rationale"]

    def test_returns_none_for_empty_trace(self):
        """Test returns None for empty trace."""
        assert _extract_episode_decision(None) is None
        assert _extract_episode_decision({}) is None
        assert _extract_episode_decision({"steps": []}) is None

    def test_returns_none_if_step_missing(self):
        """Test returns None if Episode Decision step is missing."""
        trace = {"steps": [{"step_name": "Other Step", "outputs": {}}]}
        assert _extract_episode_decision(trace) is None


class TestExtractGateResults:
    """Tests for _extract_gate_results function."""

    def test_extracts_gate_results(self, sample_pipeline_trace):
        """Test that gate results are extracted correctly."""
        result = _extract_gate_results(sample_pipeline_trace)

        assert result is not None
        assert "depressed_mood" in result
        assert result["depressed_mood"]["presence_flag"] is True
        assert result["psychomotor_change"]["presence_flag"] is False

    def test_returns_none_for_empty_trace(self):
        """Test returns None for empty trace."""
        assert _extract_gate_results(None) is None
        assert _extract_gate_results({}) is None


class TestRenderEpisodeSummary:
    """Tests for render_episode_summary function."""

    @patch("src.dashboard.components.results_summary.st")
    def test_renders_met_status_with_success(self, mock_st, sample_pipeline_trace):
        """Test that MET status uses st.success."""
        # Mock columns to return appropriate number of items for each call
        mock_st.columns.side_effect = [
            [MagicMock(), MagicMock(), MagicMock()],  # First call: 3 columns for metrics
            [MagicMock(), MagicMock()],  # Second call: 2 columns for indicators
        ]

        render_episode_summary(sample_pipeline_trace)

        mock_st.success.assert_called_once()
        call_args = mock_st.success.call_args[0][0]
        assert "Episode criteria MET" in call_args

    @patch("src.dashboard.components.results_summary.st")
    def test_renders_not_met_status_with_error(self, mock_st, not_met_pipeline_trace):
        """Test that NOT MET status uses st.error."""
        mock_st.columns.side_effect = [
            [MagicMock(), MagicMock(), MagicMock()],
            [MagicMock(), MagicMock()],
        ]

        render_episode_summary(not_met_pipeline_trace)

        mock_st.error.assert_called_once()
        call_args = mock_st.error.call_args[0][0]
        assert "Episode criteria NOT MET" in call_args

    @patch("src.dashboard.components.results_summary.st")
    def test_shows_warning_for_none_trace(self, mock_st):
        """Test that None trace shows warning."""
        render_episode_summary(None)

        mock_st.warning.assert_called_once_with(
            "Analysis summary data not available for this run."
        )

    @patch("src.dashboard.components.results_summary.st")
    def test_renders_indicator_columns(self, mock_st, sample_pipeline_trace):
        """Test that indicator presence is shown in columns."""
        mock_st.columns.side_effect = [
            [MagicMock(), MagicMock(), MagicMock()],
            [MagicMock(), MagicMock()],
        ]

        render_episode_summary(sample_pipeline_trace)

        # Verify columns were created for indicator display
        assert mock_st.columns.call_count >= 1
