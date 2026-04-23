"""Tests for experiment data module and CRUD operations."""

from unittest.mock import MagicMock, patch
import pandas as pd

from src.core.config import get_default_config


class TestExperimentEditorComponent:
    """Tests for experiment editor component."""

    def test_render_experiment_editor_exists(self):
        """render_experiment_editor function exists."""
        from src.dashboard.components.experiment_editor import render_experiment_editor

        assert callable(render_experiment_editor)

    def test_render_experiment_manager_exists(self):
        """render_experiment_manager function exists."""
        from src.dashboard.components.experiment_editor import render_experiment_manager

        assert callable(render_experiment_manager)

    def test_render_export_yaml_exists(self):
        """render_export_yaml function exists."""
        from src.dashboard.components.experiment_editor import render_export_yaml

        assert callable(render_export_yaml)


class TestComparisonComponent:
    """Tests for comparison component."""

    def test_render_run_comparison_exists(self):
        """render_run_comparison function exists."""
        from src.dashboard.components.comparison import render_run_comparison

        assert callable(render_run_comparison)

    def test_render_comparison_selector_exists(self):
        """render_comparison_selector function exists."""
        from src.dashboard.components.comparison import render_comparison_selector

        assert callable(render_comparison_selector)


class TestExperimentDataModule:
    """Tests for experiment data functions."""

    def test_create_experiment_exists(self):
        """create_experiment function exists and is callable."""
        from src.dashboard.data.experiments import create_experiment

        assert callable(create_experiment)

    def test_list_experiments_exists(self):
        """list_experiments function exists and is callable."""
        from src.dashboard.data.experiments import list_experiments

        assert callable(list_experiments)

    def test_get_experiment_exists(self):
        """get_experiment function exists and is callable."""
        from src.dashboard.data.experiments import get_experiment

        assert callable(get_experiment)

    def test_get_experiment_config_exists(self):
        """get_experiment_config function exists and is callable."""
        from src.dashboard.data.experiments import get_experiment_config

        assert callable(get_experiment_config)

    def test_update_experiment_exists(self):
        """update_experiment function exists and is callable."""
        from src.dashboard.data.experiments import update_experiment

        assert callable(update_experiment)

    def test_delete_experiment_exists(self):
        """delete_experiment function exists and is callable."""
        from src.dashboard.data.experiments import delete_experiment

        assert callable(delete_experiment)

    def test_list_experiments_returns_dataframe(self):
        """list_experiments returns a DataFrame."""
        with patch("src.dashboard.data.experiments.SessionLocal") as mock_session:
            mock_context = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_context.execute.return_value.scalars.return_value.all.return_value = []

            from src.dashboard.data.experiments import list_experiments

            result = list_experiments()
            assert isinstance(result, pd.DataFrame)
            assert "id" in result.columns
            assert "name" in result.columns
            assert "description" in result.columns
            assert "created_at" in result.columns

    def test_get_experiment_returns_none_for_invalid_id(self):
        """get_experiment returns None for invalid ID."""
        with patch("src.dashboard.data.experiments.SessionLocal") as mock_session:
            mock_context = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_context.execute.return_value.scalar_one_or_none.return_value = None

            from src.dashboard.data.experiments import get_experiment

            result = get_experiment("00000000-0000-0000-0000-000000000000")
            assert result is None

    def test_get_experiment_config_returns_none_for_invalid_id(self):
        """get_experiment_config returns None for invalid ID."""
        with patch("src.dashboard.data.experiments.get_experiment") as mock_get:
            mock_get.return_value = None

            from src.dashboard.data.experiments import get_experiment_config

            result = get_experiment_config("00000000-0000-0000-0000-000000000000")
            assert result is None

    def test_delete_experiment_returns_false_for_not_found(self):
        """delete_experiment returns False when experiment not found."""
        with patch("src.dashboard.data.experiments.SessionLocal") as mock_session:
            mock_context = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_context.execute.return_value.scalar_one_or_none.return_value = None

            from src.dashboard.data.experiments import delete_experiment

            result = delete_experiment("00000000-0000-0000-0000-000000000000")
            assert result is False

    def test_update_experiment_returns_none_for_not_found(self):
        """update_experiment returns None when experiment not found."""
        with patch("src.dashboard.data.experiments.SessionLocal") as mock_session:
            mock_context = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_context)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_context.execute.return_value.scalar_one_or_none.return_value = None

            from src.dashboard.data.experiments import update_experiment

            config = get_default_config()
            result = update_experiment(
                "00000000-0000-0000-0000-000000000000", config, name="Test"
            )
            assert result is None


class TestConfigExperimentModel:
    """Tests for ConfigExperiment database model."""

    def test_model_exists(self):
        """ConfigExperiment model exists."""
        from src.shared.models import ConfigExperiment

        assert ConfigExperiment is not None

    def test_model_has_required_fields(self):
        """ConfigExperiment has all required fields."""
        from src.shared.models import ConfigExperiment

        # Check table columns
        columns = ConfigExperiment.__table__.columns
        column_names = [c.name for c in columns]

        assert "id" in column_names
        assert "name" in column_names
        assert "description" in column_names
        assert "config_snapshot" in column_names
        assert "created_at" in column_names
        assert "updated_at" in column_names

    def test_model_tablename(self):
        """ConfigExperiment uses correct table name."""
        from src.shared.models import ConfigExperiment

        assert ConfigExperiment.__tablename__ == "config_experiments"

    def test_model_repr(self):
        """ConfigExperiment has __repr__ method."""
        from src.shared.models import ConfigExperiment

        # Check that __repr__ is defined (not just inherited)
        assert hasattr(ConfigExperiment, "__repr__")
        assert "__repr__" in ConfigExperiment.__dict__
