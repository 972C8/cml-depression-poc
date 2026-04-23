"""Tests for scenario testing page."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestScenariosPageModule:
    """Tests for scenarios page module existence."""

    def test_scenarios_page_exists(self):
        """Verify Generate Mock Data page exists."""
        scenarios_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "pages"
            / "3_🧪_Generate_Mock_Data.py"
        )
        assert scenarios_path.exists()

    def test_scenarios_data_module_exists(self):
        """Verify scenarios data module exists."""
        scenarios_data_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "dashboard"
            / "data"
            / "scenarios.py"
        )
        assert scenarios_data_path.exists()


class TestGetAvailableScenarios:
    """Tests for scenario listing."""

    def test_function_exists_and_callable(self):
        """get_available_scenarios function exists."""
        from src.dashboard.data.scenarios import get_available_scenarios

        assert callable(get_available_scenarios)

    def test_returns_list_of_strings(self):
        """Returns list of scenario names."""
        from src.dashboard.data.scenarios import get_available_scenarios

        result = get_available_scenarios()
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_contains_expected_scenarios(self):
        """Contains the three predefined scenarios."""
        from src.dashboard.data.scenarios import get_available_scenarios

        result = get_available_scenarios()
        assert "solitary_digital" in result
        assert "adversarial" in result


class TestGetScenarioInfo:
    """Tests for scenario info retrieval."""

    def test_function_exists_and_callable(self):
        """get_scenario_info function exists."""
        from src.dashboard.data.scenarios import get_scenario_info

        assert callable(get_scenario_info)

    def test_returns_solitary_digital_info(self):
        """Returns ScenarioInfo for solitary_digital."""
        from src.dashboard.data.scenarios import ScenarioInfo, get_scenario_info

        result = get_scenario_info("solitary_digital")
        assert isinstance(result, ScenarioInfo)
        assert result.name == "Solitary Digital Immersion"
        assert result.expected_context == "solitary_digital"

    def test_returns_adversarial_info(self):
        """Returns ScenarioInfo for adversarial."""
        from src.dashboard.data.scenarios import get_scenario_info

        result = get_scenario_info("adversarial")
        assert result is not None
        assert result.name == "Adversarial (TV Voices)"
        assert "ambiguous" in result.expected_context

    def test_returns_none_for_invalid_name(self):
        """Returns None for unknown scenario."""
        from src.dashboard.data.scenarios import get_scenario_info

        result = get_scenario_info("nonexistent_scenario")
        assert result is None


class TestGenerateScenarioData:
    """Tests for mock data generation."""

    def test_function_exists_and_callable(self):
        """generate_scenario_data function exists."""
        from src.dashboard.data.scenarios import generate_scenario_data

        assert callable(generate_scenario_data)

    def test_returns_generation_result(self):
        """Returns GenerationResult dataclass."""
        from src.dashboard.data.scenarios import (
            GenerationConfig,
            GenerationResult,
            generate_scenario_data,
        )

        with patch("src.dashboard.data.scenarios.load_mock_config") as mock_config:
            with patch(
                "src.dashboard.data.scenarios.MockDataOrchestrator"
            ) as mock_orch:
                with patch("src.dashboard.data.scenarios.SessionLocal"):
                    with patch("src.dashboard.data.scenarios.save_biomarkers"):
                        with patch("src.dashboard.data.scenarios.save_context"):
                            mock_config.return_value = MagicMock()
                            mock_config.return_value.biomarkers.modalities.keys.return_value = [
                                "speech",
                                "network",
                            ]
                            mock_orch_instance = MagicMock()
                            mock_orch_instance.generate_all.return_value = ([], [])
                            mock_orch.return_value = mock_orch_instance

                            config = GenerationConfig(
                                scenario="solitary_digital",
                                user_id="test",
                                days=1,
                                seed=42,
                                biomarker_interval=15,
                                context_interval=60,
                                modalities=None,
                            )
                            result = generate_scenario_data(config)
                            assert isinstance(result, GenerationResult)

    def test_generation_result_has_expected_fields(self):
        """GenerationResult has all expected fields."""
        from src.dashboard.data.scenarios import (
            GenerationConfig,
            generate_scenario_data,
        )

        with patch("src.dashboard.data.scenarios.load_mock_config") as mock_config:
            with patch(
                "src.dashboard.data.scenarios.MockDataOrchestrator"
            ) as mock_orch:
                with patch("src.dashboard.data.scenarios.SessionLocal"):
                    with patch("src.dashboard.data.scenarios.save_biomarkers"):
                        with patch("src.dashboard.data.scenarios.save_context"):
                            mock_config.return_value = MagicMock()
                            mock_config.return_value.biomarkers.modalities.keys.return_value = [
                                "speech",
                                "network",
                            ]
                            mock_orch_instance = MagicMock()
                            mock_orch_instance.generate_all.return_value = (
                                [{"test": 1}] * 10,
                                [{"test": 1}] * 5,
                            )
                            mock_orch.return_value = mock_orch_instance

                            config = GenerationConfig(
                                scenario="solitary_digital",
                                user_id="test",
                                days=1,
                                seed=42,
                                biomarker_interval=15,
                                context_interval=60,
                                modalities=None,
                            )
                            result = generate_scenario_data(config)

                            assert result.biomarker_count == 10
                            assert result.context_count == 5
                            assert result.scenario == "solitary_digital"
                            assert isinstance(result.start_time, datetime)
                            assert isinstance(result.end_time, datetime)
                            assert result.modalities_generated == ["speech", "network"]


class TestResetUserData:
    """Tests for data reset functionality."""

    def test_function_exists_and_callable(self):
        """reset_user_data function exists."""
        from src.dashboard.data.scenarios import reset_user_data

        assert callable(reset_user_data)

    def test_returns_reset_result(self):
        """Returns ResetResult with counts."""
        from src.dashboard.data.scenarios import ResetResult, reset_user_data

        with patch("src.dashboard.data.scenarios.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            # Mock scalar_one() to return 0 for all counts
            mock_session.execute.return_value.scalar_one.return_value = 0
            mock_session_cls.return_value = mock_session

            result = reset_user_data("test-user")
            assert isinstance(result, ResetResult)
            assert result.biomarkers_deleted == 0

    def test_reset_result_has_all_fields(self):
        """ResetResult includes all table counts."""
        from src.dashboard.data.scenarios import reset_user_data

        with patch("src.dashboard.data.scenarios.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.execute.return_value.scalar_one.side_effect = [5, 3, 2, 1]
            mock_session_cls.return_value = mock_session

            result = reset_user_data("test-user")

            assert result.biomarkers_deleted == 5
            assert result.context_deleted == 3
            assert result.indicators_deleted == 2
            assert result.analysis_runs_deleted == 1


class TestCheckUserHasData:
    """Tests for data existence check."""

    def test_function_exists_and_callable(self):
        """check_user_has_data function exists."""
        from src.dashboard.data.scenarios import check_user_has_data

        assert callable(check_user_has_data)

    def test_returns_bool(self):
        """Returns boolean indicating data presence."""
        from src.dashboard.data.scenarios import check_user_has_data

        with patch("src.dashboard.data.scenarios.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.execute.return_value.scalar_one.return_value = 0
            mock_session_cls.return_value = mock_session

            result = check_user_has_data("test-user")
            assert isinstance(result, bool)
            assert result is False

    def test_returns_true_when_data_exists(self):
        """Returns True when user has data."""
        from src.dashboard.data.scenarios import check_user_has_data

        with patch("src.dashboard.data.scenarios.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.execute.return_value.scalar_one.return_value = 10
            mock_session_cls.return_value = mock_session

            result = check_user_has_data("test-user")
            assert result is True


class TestGetUserDataTimeRange:
    """Tests for time range retrieval."""

    def test_function_exists_and_callable(self):
        """get_user_data_time_range function exists."""
        from src.dashboard.data.scenarios import get_user_data_time_range

        assert callable(get_user_data_time_range)

    def test_returns_none_when_no_data(self):
        """Returns None when no data exists."""
        from src.dashboard.data.scenarios import get_user_data_time_range

        with patch("src.dashboard.data.scenarios.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.execute.return_value.one.return_value = (None, None)
            mock_session_cls.return_value = mock_session

            result = get_user_data_time_range("test-user")
            assert result is None

    def test_returns_tuple_when_data_exists(self):
        """Returns tuple of (start, end) when data exists."""
        from src.dashboard.data.scenarios import get_user_data_time_range

        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 7, 23, 59, 59, tzinfo=UTC)

        with patch("src.dashboard.data.scenarios.SessionLocal") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.execute.return_value.one.return_value = (start_time, end_time)
            mock_session_cls.return_value = mock_session

            result = get_user_data_time_range("test-user")

            assert result is not None
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == start_time
            assert result[1] == end_time


class TestRunScenarioAnalysis:
    """Tests for analysis execution."""

    def test_function_exists_and_callable(self):
        """run_scenario_analysis function exists."""
        from src.dashboard.data.scenarios import run_scenario_analysis

        assert callable(run_scenario_analysis)

    def test_returns_none_when_no_data(self):
        """Returns None when user has no data."""
        from src.dashboard.data.scenarios import run_scenario_analysis

        with patch(
            "src.dashboard.data.scenarios.get_user_data_time_range"
        ) as mock_range:
            mock_range.return_value = None
            result = run_scenario_analysis("test-user")
            assert result is None

    def test_calls_run_analysis_with_time_range(self):
        """Calls run_analysis with correct time range."""
        from src.dashboard.data.scenarios import run_scenario_analysis

        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 1, 7, 23, 59, 59, tzinfo=UTC)

        with patch(
            "src.dashboard.data.scenarios.get_user_data_time_range"
        ) as mock_range:
            with patch("src.dashboard.data.scenarios.run_analysis") as mock_analysis:
                mock_range.return_value = (start_time, end_time)
                mock_result = MagicMock()
                mock_analysis.return_value = mock_result

                result = run_scenario_analysis("test-user")

                mock_analysis.assert_called_once_with(
                    user_id="test-user",
                    start_time=start_time,
                    end_time=end_time,
                )
                assert result is mock_result


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_dataclass_exists(self):
        """GenerationConfig dataclass exists."""
        from src.dashboard.data.scenarios import GenerationConfig

        assert GenerationConfig is not None

    def test_is_frozen(self):
        """GenerationConfig is immutable."""
        from src.dashboard.data.scenarios import GenerationConfig

        config = GenerationConfig(
            scenario="solitary_digital",
            user_id="test",
            days=7,
            seed=42,
            biomarker_interval=15,
            context_interval=60,
            modalities=None,
        )

        try:
            config.days = 14  # type: ignore
            raise AssertionError("Should raise FrozenInstanceError")
        except AttributeError:
            pass  # Expected

    def test_all_fields_accessible(self):
        """All fields are accessible."""
        from src.dashboard.data.scenarios import GenerationConfig

        config = GenerationConfig(
            scenario="solitary_digital",
            user_id="test-user",
            days=7,
            seed=42,
            biomarker_interval=15,
            context_interval=60,
            modalities=["speech"],
        )

        assert config.scenario == "solitary_digital"
        assert config.user_id == "test-user"
        assert config.days == 7
        assert config.seed == 42
        assert config.biomarker_interval == 15
        assert config.context_interval == 60
        assert config.modalities == ["speech"]


class TestScenarioInfo:
    """Tests for ScenarioInfo dataclass."""

    def test_dataclass_exists(self):
        """ScenarioInfo dataclass exists."""
        from src.dashboard.data.scenarios import ScenarioInfo

        assert ScenarioInfo is not None

    def test_all_fields_present(self):
        """ScenarioInfo has all required fields."""
        from src.dashboard.data.scenarios import ScenarioInfo

        info = ScenarioInfo(
            name="Test",
            description="Test description",
            expected_context="test",
            expected_behavior="Test behavior",
            config_file="test.yaml",
        )

        assert info.name == "Test"
        assert info.description == "Test description"
        assert info.expected_context == "test"
        assert info.expected_behavior == "Test behavior"
        assert info.config_file == "test.yaml"
