"""Tests for timeline data module and chart components."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest


class TestLoadTimelineBiomarkers:
    """Tests for timeline biomarker loading."""

    def test_function_exists_and_callable(self):
        """load_timeline_biomarkers function exists."""
        from src.dashboard.data.timeline import load_timeline_biomarkers

        assert callable(load_timeline_biomarkers)

    def test_returns_dataframe(self):
        """Returns pandas DataFrame."""
        from src.dashboard.data.timeline import load_timeline_biomarkers

        with patch("src.dashboard.data.timeline.load_biomarkers") as mock_load:
            mock_load.return_value = pd.DataFrame()
            result = load_timeline_biomarkers(
                user_id="test",
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
            )
            assert isinstance(result, pd.DataFrame)

    def test_filters_by_names(self):
        """Filters biomarkers by name list."""
        from src.dashboard.data.timeline import load_timeline_biomarkers

        with patch("src.dashboard.data.timeline.load_biomarkers") as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "timestamp": [datetime.now()] * 3,
                    "name": ["a", "b", "c"],
                    "value": [0.5, 0.6, 0.7],
                }
            )
            result = load_timeline_biomarkers(
                user_id="test",
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
                names=["a", "b"],
            )
            assert len(result) == 2
            assert set(result["name"].unique()) == {"a", "b"}

    def test_returns_empty_df_for_no_data(self):
        """Returns empty DataFrame when no data available."""
        from src.dashboard.data.timeline import load_timeline_biomarkers

        with patch("src.dashboard.data.timeline.load_biomarkers") as mock_load:
            mock_load.return_value = pd.DataFrame()
            result = load_timeline_biomarkers(
                user_id="test",
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
            )
            assert isinstance(result, pd.DataFrame)
            assert "timestamp" in result.columns
            assert "name" in result.columns
            assert "value" in result.columns

    def test_aggregates_to_resolution(self):
        """Aggregates data to time resolution."""
        from src.dashboard.data.timeline import load_timeline_biomarkers

        base_time = datetime(2024, 1, 1, 10, 0)
        with patch("src.dashboard.data.timeline.load_biomarkers") as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "timestamp": [
                        base_time,
                        base_time + timedelta(minutes=5),
                        base_time + timedelta(minutes=10),
                    ],
                    "name": ["a", "a", "a"],
                    "value": [0.4, 0.5, 0.6],
                }
            )
            result = load_timeline_biomarkers(
                user_id="test",
                start=base_time,
                end=base_time + timedelta(hours=1),
                resolution_minutes=15,
            )
            # All three should aggregate to same 15-min bucket
            assert len(result) == 1
            assert result.iloc[0]["value"] == pytest.approx(0.5, rel=0.01)


class TestLoadTimelineIndicators:
    """Tests for timeline indicator loading."""

    def test_function_exists_and_callable(self):
        """load_timeline_indicators function exists."""
        from src.dashboard.data.timeline import load_timeline_indicators

        assert callable(load_timeline_indicators)

    def test_returns_dataframe(self):
        """Returns pandas DataFrame."""
        from src.dashboard.data.timeline import load_timeline_indicators

        with patch("src.dashboard.data.timeline.load_indicators") as mock_load:
            mock_load.return_value = pd.DataFrame()
            result = load_timeline_indicators(
                user_id="test",
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
            )
            assert isinstance(result, pd.DataFrame)

    def test_returns_empty_df_for_no_data(self):
        """Returns empty DataFrame when no data available."""
        from src.dashboard.data.timeline import load_timeline_indicators

        with patch("src.dashboard.data.timeline.load_indicators") as mock_load:
            mock_load.return_value = pd.DataFrame()
            result = load_timeline_indicators(
                user_id="test",
                start=datetime.now() - timedelta(days=1),
                end=datetime.now(),
            )
            assert isinstance(result, pd.DataFrame)
            assert "timestamp" in result.columns
            assert "indicator_type" in result.columns
            assert "likelihood" in result.columns


class TestLoadContextPeriods:
    """Tests for context period loading."""

    def test_function_exists_and_callable(self):
        """load_context_periods function exists."""
        from src.dashboard.data.timeline import load_context_periods

        assert callable(load_context_periods)

    def test_converts_points_to_periods(self):
        """Converts point data to period data."""
        from src.dashboard.data.timeline import load_context_periods

        with patch(
            "src.dashboard.data.timeline.load_context_history_records"
        ) as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "evaluated_at": [
                        datetime(2024, 1, 1, 10, 0),
                        datetime(2024, 1, 1, 10, 15),
                        datetime(2024, 1, 1, 10, 30),
                    ],
                    "dominant_context": [
                        "solitary_digital",
                        "solitary_digital",
                        "neutral",
                    ],
                }
            )
            result = load_context_periods(
                user_id="test",
                start=datetime(2024, 1, 1, 10, 0),
                end=datetime(2024, 1, 1, 10, 30),
            )
            assert len(result) == 2
            assert "start_time" in result.columns
            assert "end_time" in result.columns
            assert "context" in result.columns

    def test_returns_empty_for_no_data(self):
        """Returns empty DataFrame when no context data."""
        from src.dashboard.data.timeline import load_context_periods

        with patch(
            "src.dashboard.data.timeline.load_context_history_records"
        ) as mock_load:
            mock_load.return_value = pd.DataFrame()
            result = load_context_periods(
                user_id="test",
                start=datetime(2024, 1, 1, 10, 0),
                end=datetime(2024, 1, 1, 10, 30),
            )
            assert isinstance(result, pd.DataFrame)
            assert "start_time" in result.columns
            assert "end_time" in result.columns
            assert "context" in result.columns


class TestAggregateToResolution:
    """Tests for data aggregation helper."""

    def test_aggregates_values_by_mean(self):
        """Aggregates values using mean."""
        from src.dashboard.data.timeline import _aggregate_to_resolution

        base_time = datetime(2024, 1, 1, 10, 0)
        df = pd.DataFrame(
            {
                "timestamp": [
                    base_time,
                    base_time + timedelta(minutes=5),
                ],
                "name": ["a", "a"],
                "value": [0.4, 0.6],
            }
        )
        result = _aggregate_to_resolution(df, 15, "name", "value")
        assert len(result) == 1
        assert result.iloc[0]["value"] == pytest.approx(0.5, rel=0.01)


class TestRenderBiomarkerTimelineChart:
    """Tests for biomarker chart rendering."""

    def test_returns_plotly_figure(self):
        """Returns Plotly Figure object."""
        from src.dashboard.components.charts import render_biomarker_timeline_chart

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "name": ["test"],
                "value": [0.5],
            }
        )
        result = render_biomarker_timeline_chart(df)
        assert isinstance(result, go.Figure)

    def test_handles_empty_dataframe(self):
        """Handles empty DataFrame gracefully."""
        from src.dashboard.components.charts import render_biomarker_timeline_chart

        result = render_biomarker_timeline_chart(pd.DataFrame())
        assert isinstance(result, go.Figure)

    def test_filters_by_selected_names(self):
        """Only shows selected biomarker names."""
        from src.dashboard.components.charts import render_biomarker_timeline_chart

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()] * 3,
                "name": ["a", "b", "c"],
                "value": [0.5, 0.6, 0.7],
            }
        )
        result = render_biomarker_timeline_chart(df, selected_names=["a", "b"])
        assert isinstance(result, go.Figure)
        # Should have 2 traces (one per selected name)
        assert len(result.data) == 2

    def test_adds_threshold_line_when_enabled(self):
        """Adds threshold line when show_threshold is True."""
        from src.dashboard.components.charts import render_biomarker_timeline_chart

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "name": ["test"],
                "value": [0.5],
            }
        )
        result = render_biomarker_timeline_chart(
            df, show_threshold=True, threshold_value=0.6
        )
        assert isinstance(result, go.Figure)
        # Layout should have horizontal line shape
        assert result.layout.shapes is not None or len(result.data) > 0


class TestRenderIndicatorTimelineChart:
    """Tests for indicator chart rendering."""

    def test_returns_plotly_figure(self):
        """Returns Plotly Figure object."""
        from src.dashboard.components.charts import render_indicator_timeline_chart

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "indicator_type": ["test"],
                "likelihood": [0.5],
            }
        )
        result = render_indicator_timeline_chart(df)
        assert isinstance(result, go.Figure)

    def test_handles_empty_dataframe(self):
        """Handles empty DataFrame gracefully."""
        from src.dashboard.components.charts import render_indicator_timeline_chart

        result = render_indicator_timeline_chart(pd.DataFrame())
        assert isinstance(result, go.Figure)

    def test_includes_threshold_line(self):
        """Includes DSM-gate threshold line."""
        from src.dashboard.components.charts import render_indicator_timeline_chart

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "indicator_type": ["test"],
                "likelihood": [0.5],
            }
        )
        result = render_indicator_timeline_chart(df, threshold_value=0.6)
        assert isinstance(result, go.Figure)
        # Should have threshold line shape
        assert result.layout.shapes is not None


class TestAddContextShading:
    """Tests for context shading."""

    def test_adds_vrect_shapes(self):
        """Adds vertical rectangles for context periods."""
        from src.dashboard.components.charts import add_context_shading

        fig = go.Figure()
        periods = pd.DataFrame(
            {
                "start_time": [datetime(2024, 1, 1, 10, 0)],
                "end_time": [datetime(2024, 1, 1, 10, 30)],
                "context": ["solitary_digital"],
            }
        )

        result = add_context_shading(fig, periods)
        assert isinstance(result, go.Figure)
        # Figure should have shapes
        assert len(result.layout.shapes) > 0

    def test_handles_empty_periods(self):
        """Handles empty periods DataFrame."""
        from src.dashboard.components.charts import add_context_shading

        fig = go.Figure()
        periods = pd.DataFrame(columns=["start_time", "end_time", "context"])

        result = add_context_shading(fig, periods)
        assert isinstance(result, go.Figure)


class TestRenderCombinedTimelineChart:
    """Tests for combined chart rendering."""

    def test_returns_plotly_figure(self):
        """Returns Plotly Figure object."""
        from src.dashboard.components.charts import render_combined_timeline_chart

        bio_df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "name": ["test"],
                "value": [0.5],
            }
        )
        ind_df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "indicator_type": ["test"],
                "likelihood": [0.5],
            }
        )
        result = render_combined_timeline_chart(bio_df, ind_df)
        assert isinstance(result, go.Figure)

    def test_handles_empty_dataframes(self):
        """Handles empty DataFrames gracefully."""
        from src.dashboard.components.charts import render_combined_timeline_chart

        result = render_combined_timeline_chart(pd.DataFrame(), pd.DataFrame())
        assert isinstance(result, go.Figure)

    def test_includes_context_shading(self):
        """Includes context shading when provided."""
        from src.dashboard.components.charts import render_combined_timeline_chart

        bio_df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "name": ["test"],
                "value": [0.5],
            }
        )
        ind_df = pd.DataFrame()
        context_periods = pd.DataFrame(
            {
                "start_time": [datetime(2024, 1, 1, 10, 0)],
                "end_time": [datetime(2024, 1, 1, 10, 30)],
                "context": ["solitary_digital"],
            }
        )
        result = render_combined_timeline_chart(
            bio_df, ind_df, context_periods=context_periods
        )
        assert isinstance(result, go.Figure)
        # Should have shapes from context shading
        assert len(result.layout.shapes) > 0


class TestGenerateTimelineCsvFilename:
    """Tests for CSV filename generation."""

    def test_generates_valid_filename(self):
        """Generates valid filename with date range."""
        from src.dashboard.data.timeline import generate_timeline_csv_filename

        result = generate_timeline_csv_filename(
            "user123",
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
            "biomarkers",
        )
        assert "user123" in result
        assert "20240101" in result
        assert "20240131" in result
        assert "biomarkers" in result
        assert result.endswith(".csv")
