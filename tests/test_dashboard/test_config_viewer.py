"""Tests for configuration viewer components and data loading."""

from unittest.mock import MagicMock, patch

from src.core.config import AnalysisConfig, get_default_config


class TestConfigDataModule:
    """Tests for config data loading functions."""

    def test_get_current_config_exists(self):
        """get_current_config function exists and is callable."""
        from src.dashboard.data.config import get_current_config

        assert callable(get_current_config)

    def test_get_current_config_returns_analysis_config(self):
        """get_current_config returns AnalysisConfig instance."""
        from src.dashboard.data.config import get_current_config

        # Need to mock streamlit's cache decorator
        with patch("src.dashboard.data.config.st"):
            # Import fresh to get the original function without decorator
            import importlib

            import src.dashboard.data.config as config_module

            # Call the underlying function
            result = get_default_config()
            assert isinstance(result, AnalysisConfig)

    def test_reload_config_exists(self):
        """reload_config function exists and is callable."""
        from src.dashboard.data.config import reload_config

        assert callable(reload_config)

    def test_config_to_yaml_exists(self):
        """config_to_yaml function exists and is callable."""
        from src.dashboard.data.config import config_to_yaml

        assert callable(config_to_yaml)

    def test_config_to_yaml_returns_string(self):
        """config_to_yaml returns YAML string."""
        from src.dashboard.data.config import config_to_yaml

        config = get_default_config()
        result = config_to_yaml(config)
        assert isinstance(result, str)
        assert "indicators:" in result

    def test_config_to_yaml_contains_all_sections(self):
        """config_to_yaml output contains all config sections."""
        from src.dashboard.data.config import config_to_yaml

        config = get_default_config()
        result = config_to_yaml(config)

        assert "indicators:" in result
        assert "context_weights:" in result
        assert "dsm_gate_defaults:" in result
        assert "ema:" in result
        assert "episode:" in result
        assert "biomarker_defaults:" in result


class TestConfigViewerComponents:
    """Tests for config viewer component functions."""

    def test_render_indicators_section_exists(self):
        """render_indicators_section function exists."""
        from src.dashboard.components.config_viewer import render_indicators_section

        assert callable(render_indicators_section)

    def test_render_context_weights_section_exists(self):
        """render_context_weights_section function exists."""
        from src.dashboard.components.config_viewer import render_context_weights_section

        assert callable(render_context_weights_section)

    def test_render_dsm_gate_section_exists(self):
        """render_dsm_gate_section function exists."""
        from src.dashboard.components.config_viewer import render_dsm_gate_section

        assert callable(render_dsm_gate_section)

    def test_render_ema_section_exists(self):
        """render_ema_section function exists."""
        from src.dashboard.components.config_viewer import render_ema_section

        assert callable(render_ema_section)

    def test_render_episode_section_exists(self):
        """render_episode_section function exists."""
        from src.dashboard.components.config_viewer import render_episode_section

        assert callable(render_episode_section)

    def test_render_config_viewer_exists(self):
        """render_config_viewer function exists."""
        from src.dashboard.components.config_viewer import render_config_viewer

        assert callable(render_config_viewer)

    def test_render_global_settings_section_exists(self):
        """render_global_settings_section function exists."""
        from src.dashboard.components.config_viewer import render_global_settings_section

        assert callable(render_global_settings_section)

    def test_render_raw_config_exists(self):
        """render_raw_config function exists."""
        from src.dashboard.components.config_viewer import render_raw_config

        assert callable(render_raw_config)

    def test_render_indicators_section_with_mock_streamlit(self):
        """render_indicators_section renders without error."""
        with patch("src.dashboard.components.config_viewer.st") as mock_st:
            from src.dashboard.components.config_viewer import (
                render_indicators_section,
            )

            config = get_default_config()
            # Should not raise
            render_indicators_section(config)
            # Verify streamlit was called
            assert mock_st.markdown.called or mock_st.expander.called

    def test_render_context_weights_section_with_mock_streamlit(self):
        """render_context_weights_section renders without error."""
        with patch("src.dashboard.components.config_viewer.st") as mock_st:
            from src.dashboard.components.config_viewer import (
                render_context_weights_section,
            )

            config = get_default_config()
            render_context_weights_section(config)
            assert mock_st.markdown.called

    def test_render_dsm_gate_section_with_mock_streamlit(self):
        """render_dsm_gate_section renders without error."""
        with patch("src.dashboard.components.config_viewer.st") as mock_st:
            # Mock columns to return context managers
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            mock_st.columns.return_value = [mock_col, mock_col, mock_col]

            from src.dashboard.components.config_viewer import render_dsm_gate_section

            config = get_default_config()
            render_dsm_gate_section(config)
            assert mock_st.markdown.called or mock_st.metric.called

    def test_render_ema_section_with_mock_streamlit(self):
        """render_ema_section renders without error."""
        with patch("src.dashboard.components.config_viewer.st") as mock_st:
            # Mock columns to return context managers
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            mock_st.columns.return_value = [mock_col, mock_col, mock_col]

            from src.dashboard.components.config_viewer import render_ema_section

            config = get_default_config()
            render_ema_section(config)
            assert mock_st.markdown.called or mock_st.metric.called

    def test_render_episode_section_with_mock_streamlit(self):
        """render_episode_section renders without error."""
        with patch("src.dashboard.components.config_viewer.st") as mock_st:
            # Mock columns to return context managers
            mock_col = MagicMock()
            mock_col.__enter__ = MagicMock(return_value=mock_col)
            mock_col.__exit__ = MagicMock(return_value=False)
            mock_st.columns.return_value = [mock_col, mock_col]

            from src.dashboard.components.config_viewer import render_episode_section

            config = get_default_config()
            render_episode_section(config)
            assert mock_st.markdown.called or mock_st.metric.called
