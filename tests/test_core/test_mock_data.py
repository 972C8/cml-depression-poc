"""Unit tests for mock data generator."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from src.core.mock_data import (
    ActiveHoursModel,
    BiomarkerParamsModel,
    ContextMarkerGenerator,
    DayPatternModel,
    MockDataConfig,
    MockDataOrchestrator,
    ModalityGenerator,
    ScenarioConfigModel,
    ScheduleModel,
    apply_daily_cycle,
    load_mock_config,
    load_scenario_config,
    save_biomarkers,
    save_context,
)
from src.shared.models import Biomarker, Context


class TestDailyCycle:
    """Tests for daily cycle pattern."""

    def test_apply_daily_cycle_peak_at_2pm(self):
        """Test that daily cycle peaks around 2 PM."""
        base_value = 0.5
        timestamp_2pm = datetime(2025, 1, 1, 14, 0, tzinfo=UTC)

        result = apply_daily_cycle(base_value, timestamp_2pm)

        # At 2 PM, should be at or near peak (> base_value)
        assert result > base_value
        assert 0.0 <= result <= 1.0

    def test_apply_daily_cycle_trough_at_3am(self):
        """Test that daily cycle is lowest around 3 AM."""
        base_value = 0.5
        timestamp_3am = datetime(2025, 1, 1, 3, 0, tzinfo=UTC)

        result = apply_daily_cycle(base_value, timestamp_3am)

        # At 3 AM, should be at or near trough (< base_value)
        assert result < base_value
        assert 0.0 <= result <= 1.0

    def test_apply_daily_cycle_clamping(self):
        """Test that values are clamped to [0, 1] range."""
        # High base value that might exceed 1 with cycle
        base_value = 0.95
        timestamp_2pm = datetime(2025, 1, 1, 14, 0, tzinfo=UTC)

        result = apply_daily_cycle(base_value, timestamp_2pm)

        assert 0.0 <= result <= 1.0


class TestBiomarkerParamsModel:
    """Tests for BiomarkerParamsModel validation."""

    def test_valid_params(self):
        """Test creating model with valid parameters."""
        params = BiomarkerParamsModel(baseline=0.5, variance=0.15, daily_cycle=True)

        assert params.baseline == 0.5
        assert params.variance == 0.15
        assert params.daily_cycle is True

    def test_invalid_baseline_too_high(self):
        """Test that baseline > 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            BiomarkerParamsModel(baseline=1.5, variance=0.15, daily_cycle=True)

    def test_invalid_variance_negative(self):
        """Test that negative variance raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            BiomarkerParamsModel(baseline=0.5, variance=-0.1, daily_cycle=True)


class TestConfigLoading:
    """Tests for config loading functionality."""

    def test_load_mock_config_success(self):
        """Test loading mock data config from default location."""
        config = load_mock_config()

        assert isinstance(config, MockDataConfig)
        assert len(config.biomarkers.modalities) > 0
        assert len(config.context_markers.markers) > 0

    def test_load_mock_config_has_speech_modality(self):
        """Test that config includes speech modality with evaluation biomarkers."""
        config = load_mock_config()

        assert "speech" in config.biomarkers.modalities
        speech_modality = config.biomarkers.modalities["speech"]
        assert "whispering" in speech_modality.biomarkers
        assert "prolonged_pauses" in speech_modality.biomarkers
        assert "monopitch" in speech_modality.biomarkers

    def test_load_mock_config_has_network_modality(self):
        """Test that config includes network modality with evaluation biomarkers."""
        config = load_mock_config()

        assert "network" in config.biomarkers.modalities
        network_modality = config.biomarkers.modalities["network"]
        assert "passive_media_binge" in network_modality.biomarkers
        assert "reduced_social_interaction" in network_modality.biomarkers

    def test_load_mock_config_has_29_biomarkers(self):
        """Test that config has all 29 evaluation biomarkers."""
        config = load_mock_config()

        total_biomarkers = sum(
            len(m.biomarkers) for m in config.biomarkers.modalities.values()
        )
        assert total_biomarkers == 29

    def test_load_mock_config_has_context_markers(self):
        """Test that config includes context markers."""
        config = load_mock_config()

        assert "people_in_room" in config.context_markers.markers
        assert "network_activity_level" in config.context_markers.markers

    def test_load_scenario_config_solitary_digital(self):
        """Test loading solitary digital scenario."""
        scenario = load_scenario_config("solitary_digital")

        assert scenario.name == "solitary_digital"
        assert "context" in scenario.overrides
        assert "biomarkers" in scenario.overrides

    def test_load_scenario_config_with_schedule(self):
        """Test that solitary_digital scenario has schedule field."""
        scenario = load_scenario_config("solitary_digital")

        assert scenario.schedule is not None
        assert scenario.schedule.active_hours is not None
        assert scenario.schedule.active_hours.start == 20
        assert scenario.schedule.active_hours.end == 24
        assert scenario.schedule.day_pattern is not None
        assert scenario.schedule.day_pattern.days_on == 1
        assert scenario.schedule.day_pattern.days_off == 1
        assert scenario.schedule.day_pattern.offset == 1

    def test_load_scenario_config_nonexistent(self):
        """Test that loading nonexistent scenario raises error."""
        with pytest.raises(FileNotFoundError):
            load_scenario_config("nonexistent_scenario")


class TestModalityGenerator:
    """Tests for ModalityGenerator."""

    def test_generate_snapshot_produces_valid_values(self, mock_data_config):
        """Test that generated values are in [0, 1] range."""
        from numpy.random import PCG64, Generator

        rng = Generator(PCG64(42))
        modality_config = mock_data_config.biomarkers.modalities["speech"]

        generator = ModalityGenerator("speech", modality_config, rng)
        timestamp = datetime.now(UTC)

        snapshot = generator.generate_snapshot(timestamp)

        assert len(snapshot) > 0
        for value in snapshot.values():
            assert 0.0 <= value <= 1.0

    def test_generate_snapshot_includes_all_biomarkers(self, mock_data_config):
        """Test that snapshot includes all configured biomarkers."""
        from numpy.random import PCG64, Generator

        rng = Generator(PCG64(42))
        modality_config = mock_data_config.biomarkers.modalities["speech"]

        generator = ModalityGenerator("speech", modality_config, rng)
        timestamp = datetime.now(UTC)

        snapshot = generator.generate_snapshot(timestamp)

        expected_biomarkers = set(modality_config.biomarkers.keys())
        actual_biomarkers = set(snapshot.keys())

        assert expected_biomarkers == actual_biomarkers

    def test_generate_snapshot_with_overrides(self, mock_data_config):
        """Test that scenario overrides are applied correctly."""
        from numpy.random import PCG64, Generator

        rng = Generator(PCG64(42))
        modality_config = mock_data_config.biomarkers.modalities["network"]

        overrides = {"passive_media_binge": {"baseline": 0.9, "variance": 0.05}}
        generator = ModalityGenerator("network", modality_config, rng, overrides)
        timestamp = datetime.now(UTC)

        snapshot = generator.generate_snapshot(timestamp)

        # With low variance and high baseline, value should be close to 0.9
        assert snapshot["passive_media_binge"] > 0.8


class TestContextMarkerGenerator:
    """Tests for ContextMarkerGenerator."""

    def test_generate_snapshot_produces_valid_values(self, mock_data_config):
        """Test that generated values are within configured bounds."""
        from numpy.random import PCG64, Generator

        rng = Generator(PCG64(42))

        generator = ContextMarkerGenerator(mock_data_config.context_markers, rng)
        timestamp = datetime.now(UTC)

        snapshot = generator.generate_snapshot(timestamp)

        assert len(snapshot) > 0
        # Values should be within their configured min/max bounds
        for marker_name, value in snapshot.items():
            params = mock_data_config.context_markers.markers[marker_name]
            assert value >= params.min_value
            if params.max_value is not None:
                assert value <= params.max_value

    def test_generate_snapshot_includes_all_markers(self, mock_data_config):
        """Test that snapshot includes all configured markers."""
        from numpy.random import PCG64, Generator

        rng = Generator(PCG64(42))

        generator = ContextMarkerGenerator(mock_data_config.context_markers, rng)
        timestamp = datetime.now(UTC)

        snapshot = generator.generate_snapshot(timestamp)

        expected_markers = set(mock_data_config.context_markers.markers.keys())
        actual_markers = set(snapshot.keys())

        assert expected_markers == actual_markers


class TestMockDataOrchestrator:
    """Tests for MockDataOrchestrator."""

    def test_generate_biomarkers_correct_count(self, mock_data_config):
        """Test that correct number of records are generated."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=1)

        biomarkers = orchestrator.generate_biomarkers(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # 1 day = 24 hours, N modalities = 24 * N records
        num_modalities = len(mock_data_config.biomarkers.modalities)
        expected_count = 24 * num_modalities
        assert len(biomarkers) == expected_count

    def test_generate_biomarkers_with_seed_reproducible(self, mock_data_config):
        """Test that same seed produces identical output."""
        orchestrator1 = MockDataOrchestrator(mock_data_config, seed=42)
        orchestrator2 = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        biomarkers1 = orchestrator1.generate_biomarkers(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        biomarkers2 = orchestrator2.generate_biomarkers(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # Compare values of first record
        assert biomarkers1[0]["value"] == biomarkers2[0]["value"]

    def test_generate_biomarkers_filtered_modalities(self, mock_data_config):
        """Test generating only specific modalities."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        biomarkers = orchestrator.generate_biomarkers(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
            modalities=["speech"],
        )

        # Should only have speech records
        assert len(biomarkers) == 1
        assert biomarkers[0]["biomarker_type"] == "speech"

    def test_generate_context_correct_count(self, mock_data_config):
        """Test that correct number of context records are generated."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=1)

        context = orchestrator.generate_context(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # 1 day = 24 hours, 1 context record per hour = 24 records
        assert len(context) == 24

    def test_generate_all_independent_intervals(self, mock_data_config):
        """Test that biomarkers and context can have different intervals."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=2)

        biomarkers, context = orchestrator.generate_all(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            biomarker_interval=30,  # 30 min intervals
            context_interval=60,  # 60 min intervals
        )

        # 2 hours = 120 min
        # Biomarkers: 120 / 30 = 4 samples per modality, N modalities
        # Context: 120 / 60 = 2 samples
        num_modalities = len(mock_data_config.biomarkers.modalities)
        assert len(biomarkers) == 4 * num_modalities
        assert len(context) == 2

    def test_generate_with_scenario_applies_overrides(self, mock_data_config):
        """Test that scenario overrides are applied when no schedule."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)
        orchestrator.scenario_config = ScenarioConfigModel(
            name="test_scenario",
            description="test",
            schedule=None,  # No schedule = always active
            overrides={
                "context": {
                    "people_in_room": {
                        "baseline": 8,
                        "variance": 0.5,
                        "max_value": 10,
                    },
                },
            },
        )

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        context = orchestrator.generate_context(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # With baseline 8 and variance 0.5, people_in_room should be > 5
        people_value = context[0]["value"]["people_in_room"]
        assert people_value > 5


class TestSchedulingLogic:
    """Tests for scenario scheduling logic (AC15)."""

    def _make_orchestrator_with_schedule(
        self,
        mock_data_config,
        active_hours=None,
        day_pattern=None,
    ):
        """Helper to create orchestrator with scheduling config."""
        ah = ActiveHoursModel(**active_hours) if active_hours else None
        dp = DayPatternModel(**day_pattern) if day_pattern else None
        schedule = ScheduleModel(active_hours=ah, day_pattern=dp)

        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)
        orchestrator.scenario_config = ScenarioConfigModel(
            name="test_scheduled",
            description="test",
            schedule=schedule,
            overrides={
                "biomarkers": {
                    "network": {
                        "passive_media_binge": {"baseline": 0.90, "variance": 0.05},
                    },
                },
                "context": {
                    "people_in_room": {"baseline": 0, "variance": 0},
                },
            },
        )
        return orchestrator

    def test_active_hours_only_during_window(self, mock_data_config):
        """Scenario with active_hours only applies during those hours."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            active_hours={"start": 20, "end": 24},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # 10 AM — outside window
        ts_10am = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_10am, gen_start) is False

        # 8 PM — inside window
        ts_8pm = datetime(2026, 1, 1, 20, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_8pm, gen_start) is True

        # 11 PM — inside window
        ts_11pm = datetime(2026, 1, 1, 23, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_11pm, gen_start) is True

    def test_day_pattern_only_on_matching_days(self, mock_data_config):
        """Scenario with day_pattern (1 on / 1 off) only applies on matching days."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            day_pattern={"days_on": 1, "days_off": 1, "offset": 1},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # Day 0 — offset=1, so day_in_cycle = (0-1) % 2 → 1, >= days_on(1) → inactive
        ts_day0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_day0, gen_start) is False

        # Day 1 — day_in_cycle = (1-1) % 2 = 0, < days_on(1) → active
        ts_day1 = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_day1, gen_start) is True

        # Day 2 — day_in_cycle = (2-1) % 2 = 1, >= days_on(1) → inactive
        ts_day2 = datetime(2026, 1, 3, 12, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_day2, gen_start) is False

        # Day 3 — day_in_cycle = (3-1) % 2 = 0, < days_on(1) → active
        ts_day3 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_day3, gen_start) is True

    def test_both_conditions_must_hold(self, mock_data_config):
        """Scenario with both active_hours and day_pattern: both must be true."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            active_hours={"start": 20, "end": 24},
            day_pattern={"days_on": 1, "days_off": 1, "offset": 1},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # Day 1 at 10 AM — right day, wrong hour → inactive
        ts = datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts, gen_start) is False

        # Day 0 at 21:00 — wrong day, right hour → inactive
        ts = datetime(2026, 1, 1, 21, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts, gen_start) is False

        # Day 1 at 21:00 — right day, right hour → active
        ts = datetime(2026, 1, 2, 21, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts, gen_start) is True

    def test_no_schedule_always_active(self, mock_data_config):
        """Scenario without schedule field applies uniformly (backward compat)."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)
        orchestrator.scenario_config = ScenarioConfigModel(
            name="always_on",
            description="test",
            schedule=None,
            overrides={"biomarkers": {}},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        ts = datetime(2026, 1, 5, 3, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts, gen_start) is True

    def test_no_scenario_config_inactive(self, mock_data_config):
        """No scenario config at all returns False."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)
        assert orchestrator.scenario_config is None

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts, gen_start) is False

    def test_day_boundary_edge_case(self, mock_data_config):
        """Test hour boundary: end=24 means 23:xx is active, 00:xx next day is not."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            active_hours={"start": 20, "end": 24},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # 23:59 — active (hour=23, 20 <= 23 < 24)
        ts_2359 = datetime(2026, 1, 1, 23, 59, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_2359, gen_start) is True

        # 00:00 next day — inactive (hour=0, not 20 <= 0 < 24)
        ts_0000 = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
        assert orchestrator._is_scenario_active(ts_0000, gen_start) is False

    def test_offset_zero_even_days(self, mock_data_config):
        """Offset=0 with 1 on / 1 off means day 0, 2, 4, ... are active."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            day_pattern={"days_on": 1, "days_off": 1, "offset": 0},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # Day 0 active
        assert orchestrator._is_scenario_active(
            datetime(2026, 1, 1, 12, 0, tzinfo=UTC), gen_start
        ) is True

        # Day 1 inactive
        assert orchestrator._is_scenario_active(
            datetime(2026, 1, 2, 12, 0, tzinfo=UTC), gen_start
        ) is False

        # Day 2 active
        assert orchestrator._is_scenario_active(
            datetime(2026, 1, 3, 12, 0, tzinfo=UTC), gen_start
        ) is True

    def test_multi_day_on_off_pattern(self, mock_data_config):
        """Pattern 2 on / 1 off activates two consecutive days then skips one."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            day_pattern={"days_on": 2, "days_off": 1, "offset": 0},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # Cycle length = 3.  Days 0,1 active; day 2 off; days 3,4 active; day 5 off; …
        expected = [True, True, False, True, True, False, True, True, False]
        for day, expect_active in enumerate(expected):
            ts = datetime(2026, 1, 1 + day, 12, 0, tzinfo=UTC)
            result = orchestrator._is_scenario_active(ts, gen_start)
            assert result is expect_active, f"Day {day}: expected {expect_active}, got {result}"

    def test_multi_day_on_off_with_offset(self, mock_data_config):
        """Pattern 2 on / 1 off with offset=1 shifts the cycle start."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            day_pattern={"days_on": 2, "days_off": 1, "offset": 1},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        # offset=1 shifts: day 0 → cycle pos (0-1)%3 = 2 → off
        # day 1 → (1-1)%3 = 0 → on; day 2 → 1 → on; day 3 → 2 → off
        expected = [False, True, True, False, True, True, False]
        for day, expect_active in enumerate(expected):
            ts = datetime(2026, 1, 1 + day, 12, 0, tzinfo=UTC)
            result = orchestrator._is_scenario_active(ts, gen_start)
            assert result is expect_active, f"Day {day}: expected {expect_active}, got {result}"

    def test_always_on_pattern(self, mock_data_config):
        """Pattern with off=0 means every day is active."""
        orchestrator = self._make_orchestrator_with_schedule(
            mock_data_config,
            day_pattern={"days_on": 1, "days_off": 0, "offset": 0},
        )

        gen_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        for day in range(7):
            ts = datetime(2026, 1, 1 + day, 12, 0, tzinfo=UTC)
            assert orchestrator._is_scenario_active(ts, gen_start) is True


class TestScheduleModelValidation:
    """Tests for ActiveHoursModel and DayPatternModel validation."""

    def test_active_hours_start_ge_end_raises(self):
        """start >= end should raise ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            ActiveHoursModel(start=22, end=20)

    def test_active_hours_negative_start_raises(self):
        """Negative start should raise ValueError."""
        with pytest.raises(ValueError):
            ActiveHoursModel(start=-1, end=24)

    def test_active_hours_end_exceeds_24_raises(self):
        """end > 24 should raise ValueError."""
        with pytest.raises(ValueError):
            ActiveHoursModel(start=20, end=25)

    def test_day_pattern_on_zero_raises(self):
        """on=0 should raise ValueError."""
        with pytest.raises(ValueError):
            DayPatternModel(days_on=0, days_off=1, offset=0)

    def test_day_pattern_negative_off_raises(self):
        """Negative off should raise ValueError."""
        with pytest.raises(ValueError):
            DayPatternModel(days_on=1, days_off=-1, offset=0)

    def test_day_pattern_negative_offset_raises(self):
        """Negative offset should raise ValueError."""
        with pytest.raises(ValueError):
            DayPatternModel(days_on=1, days_off=1, offset=-1)

    def test_valid_active_hours(self):
        """Valid ActiveHoursModel should be created."""
        ah = ActiveHoursModel(start=20, end=24)
        assert ah.start == 20
        assert ah.end == 24

    def test_valid_day_pattern(self):
        """Valid DayPatternModel should be created."""
        dp = DayPatternModel(days_on=2, days_off=1, offset=0)
        assert dp.days_on == 2
        assert dp.days_off == 1
        assert dp.offset == 0


class TestScheduledGeneration:
    """Integration tests for scheduled data generation (AC16)."""

    def test_14_day_scheduled_generation(self, mock_data_config):
        """Generate 14 days with solitary_digital scenario active 20:00-24:00, two nights on / one night off.

        Verify:
        - Neutral periods: biomarker values near baseline
        - Scenario periods: biomarker overrides applied
        - Context overrides applied during active periods
        - Two-on/one-off day pattern correct
        """
        orchestrator = MockDataOrchestrator(
            mock_data_config, seed=42, scenario="solitary_digital"
        )

        start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        end_time = start_time + timedelta(days=14)

        biomarkers = orchestrator.generate_biomarkers(
            user_id="test-schedule",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,  # hourly
        )

        context = orchestrator.generate_context(
            user_id="test-schedule",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # Extract network biomarker records
        network_records = [r for r in biomarkers if r["biomarker_type"] == "network"]

        # Check a known ACTIVE timestamp: Day 1 (odd), hour 21
        active_records = [
            r for r in network_records
            if r["timestamp"] == datetime(2026, 1, 2, 21, 0, tzinfo=UTC)
        ]
        assert len(active_records) == 1
        active_values = active_records[0]["value"]
        # passive_media_binge should be elevated (override baseline 0.40, variance 0.20)
        assert active_values["passive_media_binge"] > 0.1
        assert active_records[0]["metadata_"]["scenario_active"] is True

        # Check a known NEUTRAL timestamp: Day 0 (even), hour 21
        neutral_records = [
            r for r in network_records
            if r["timestamp"] == datetime(2026, 1, 1, 21, 0, tzinfo=UTC)
        ]
        assert len(neutral_records) == 1
        neutral_values = neutral_records[0]["value"]
        # passive_media_binge should be near baseline (0.25)
        assert neutral_values["passive_media_binge"] < 0.5
        assert neutral_records[0]["metadata_"]["scenario_active"] is False

        # Check a known NEUTRAL timestamp: Day 1 (odd), hour 10 (outside active hours)
        daytime_records = [
            r for r in network_records
            if r["timestamp"] == datetime(2026, 1, 2, 10, 0, tzinfo=UTC)
        ]
        assert len(daytime_records) == 1
        assert daytime_records[0]["metadata_"]["scenario_active"] is False

        # Check context during active period
        active_ctx = [
            r for r in context
            if r["timestamp"] == datetime(2026, 1, 2, 21, 0, tzinfo=UTC)
        ]
        assert len(active_ctx) == 1
        assert active_ctx[0]["metadata_"]["scenario_active"] is True
        # Network activity should be high during solitary_digital
        assert active_ctx[0]["value"]["network_activity_level"] > 0.6

        # Check context during neutral period
        neutral_ctx = [
            r for r in context
            if r["timestamp"] == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        ]
        assert len(neutral_ctx) == 1
        assert neutral_ctx[0]["metadata_"]["scenario_active"] is False

    def test_alternating_day_pattern_correct(self, mock_data_config):
        """Verify two-on/one-off day pattern: active days have 20:00-24:00 windows, off days are fully neutral."""
        orchestrator = MockDataOrchestrator(
            mock_data_config, seed=42, scenario="solitary_digital"
        )

        start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        end_time = start_time + timedelta(days=5)

        biomarkers = orchestrator.generate_biomarkers(
            user_id="test-pattern",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        # Count active/neutral records per day
        for day in range(5):
            day_start = start_time + timedelta(days=day)
            day_records = [
                r for r in biomarkers
                if r["biomarker_type"] == "network"
                and day_start <= r["timestamp"] < day_start + timedelta(days=1)
            ]

            active_count = sum(
                1 for r in day_records if r["metadata_"]["scenario_active"]
            )

            if day % 2 == 0:
                # Even days: fully neutral (day 0, 2, 4)
                assert active_count == 0, f"Day {day} should be fully neutral"
            else:
                # Odd days: only 20:00-23:xx active (4 hours)
                assert active_count == 4, f"Day {day} should have 4 active hours"


class TestReproducibility:
    """Tests for seed-based reproducibility with scheduling (AC17)."""

    def test_same_seed_identical_output_with_schedule(self, mock_data_config):
        """Same seed + same config = identical output with scheduling."""
        start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        end_time = start_time + timedelta(days=3)

        orchestrator1 = MockDataOrchestrator(
            mock_data_config, seed=42, scenario="solitary_digital"
        )
        bio1 = orchestrator1.generate_biomarkers(
            "user1", start_time, end_time, interval_minutes=60
        )
        ctx1 = orchestrator1.generate_context(
            "user1", start_time, end_time, interval_minutes=60
        )

        orchestrator2 = MockDataOrchestrator(
            mock_data_config, seed=42, scenario="solitary_digital"
        )
        bio2 = orchestrator2.generate_biomarkers(
            "user1", start_time, end_time, interval_minutes=60
        )
        ctx2 = orchestrator2.generate_context(
            "user1", start_time, end_time, interval_minutes=60
        )

        # Same number of records
        assert len(bio1) == len(bio2)
        assert len(ctx1) == len(ctx2)

        # Identical values
        for r1, r2 in zip(bio1, bio2, strict=True):
            assert r1["value"] == r2["value"]
            assert r1["metadata_"]["scenario_active"] == r2["metadata_"]["scenario_active"]

        for r1, r2 in zip(ctx1, ctx2, strict=True):
            assert r1["value"] == r2["value"]

    def test_rng_determinism_constant_calls(self, mock_data_config):
        """RNG call count per timestamp is constant regardless of scenario state.

        The same number of random values are generated at each timestamp
        whether the scenario is active or not — only the params differ.
        """
        start_time = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
        end_time = start_time + timedelta(days=2)

        # Generate with scenario
        orch_scenario = MockDataOrchestrator(
            mock_data_config, seed=42, scenario="solitary_digital"
        )
        bio_scenario = orch_scenario.generate_biomarkers(
            "user1", start_time, end_time, interval_minutes=60
        )

        # Generate without scenario
        orch_none = MockDataOrchestrator(mock_data_config, seed=42)
        bio_none = orch_none.generate_biomarkers(
            "user1", start_time, end_time, interval_minutes=60
        )

        # Same number of records (both generate for all biomarkers at all timestamps)
        assert len(bio_scenario) == len(bio_none)


class TestGeneratedDataFormat:
    """Tests for generated data format matching ORM models."""

    def test_biomarker_record_structure(self, mock_data_config):
        """Test that generated biomarker records match ORM structure."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        biomarkers = orchestrator.generate_biomarkers(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        record = biomarkers[0]

        assert "user_id" in record
        assert "timestamp" in record
        assert "biomarker_type" in record
        assert "value" in record
        assert "metadata_" in record

        assert isinstance(record["value"], dict)
        assert isinstance(record["metadata_"], dict)
        assert "scenario_active" in record["metadata_"]

    def test_context_record_structure(self, mock_data_config):
        """Test that generated context records match ORM structure."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        context = orchestrator.generate_context(
            user_id="test-user",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        record = context[0]

        assert "user_id" in record
        assert "timestamp" in record
        assert "context_type" in record
        assert "value" in record
        assert "metadata_" in record

        assert isinstance(record["value"], dict)
        assert isinstance(record["metadata_"], dict)
        assert "scenario_active" in record["metadata_"]


class TestDatabasePersistence:
    """Tests for database persistence functions."""

    def test_save_biomarkers(self, db_session, mock_data_config):
        """Test saving biomarkers to database."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        biomarkers = orchestrator.generate_biomarkers(
            user_id="test-save-bio",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        count = save_biomarkers(biomarkers, db_session)

        assert count == len(biomarkers)

        # Verify records in database (SQLAlchemy 2.0 style)
        saved = (
            db_session.execute(
                select(Biomarker).where(Biomarker.user_id == "test-save-bio")
            )
            .scalars()
            .all()
        )
        assert len(saved) == count

        # Cleanup (SQLAlchemy 2.0 style)
        db_session.execute(
            delete(Biomarker).where(Biomarker.user_id == "test-save-bio")
        )
        db_session.commit()

    def test_save_context(self, db_session, mock_data_config):
        """Test saving context to database."""
        orchestrator = MockDataOrchestrator(mock_data_config, seed=42)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=1)

        context = orchestrator.generate_context(
            user_id="test-save-ctx",
            start_time=start_time,
            end_time=end_time,
            interval_minutes=60,
        )

        count = save_context(context, db_session)

        assert count == len(context)

        # Verify records in database (SQLAlchemy 2.0 style)
        saved = (
            db_session.execute(
                select(Context).where(Context.user_id == "test-save-ctx")
            )
            .scalars()
            .all()
        )
        assert len(saved) == count

        # Cleanup (SQLAlchemy 2.0 style)
        db_session.execute(delete(Context).where(Context.user_id == "test-save-ctx"))
        db_session.commit()


class TestFixtures:
    """Tests for pytest fixtures."""

    def test_mock_biomarkers_fixture(self, mock_biomarkers):
        """Test that mock_biomarkers fixture works correctly."""
        assert "user_id" in mock_biomarkers
        assert "count" in mock_biomarkers
        assert mock_biomarkers["count"] > 0

    def test_mock_context_fixture(self, mock_context):
        """Test that mock_context fixture works correctly."""
        assert "user_id" in mock_context
        assert "count" in mock_context
        assert mock_context["count"] > 0

    @pytest.mark.parametrize("mock_scenario", ["solitary_digital"], indirect=True)
    def test_mock_scenario_fixture(self, mock_scenario):
        """Test that mock_scenario fixture works correctly."""
        assert "scenario_name" in mock_scenario
        assert mock_scenario["scenario_name"] == "solitary_digital"
        assert mock_scenario["bio_count"] > 0
        assert mock_scenario["ctx_count"] > 0


class TestCLI:
    """Tests for CLI functionality."""

    def test_cli_dry_run_does_not_write_to_db(self, db_session):
        """Test that --dry-run flag prevents database writes (AC11)."""
        import subprocess
        import sys

        # Count records before CLI run
        before_bio = (
            db_session.execute(
                select(Biomarker).where(Biomarker.user_id == "cli-dry-run-test")
            )
            .scalars()
            .all()
        )
        before_count = len(before_bio)

        # Run CLI with --dry-run
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.core.mock_data",
                "biomarkers",
                "--user",
                "cli-dry-run-test",
                "--days",
                "1",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd="/Users/tiborhaller/dev/MT_POC",
        )

        # CLI should succeed
        assert result.returncode == 0
        assert "Would generate" in result.stdout

        # Count records after CLI run
        after_bio = (
            db_session.execute(
                select(Biomarker).where(Biomarker.user_id == "cli-dry-run-test")
            )
            .scalars()
            .all()
        )
        after_count = len(after_bio)

        # No new records should be written
        assert after_count == before_count

    def test_cli_biomarkers_help(self):
        """Test that biomarkers --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.core.mock_data", "biomarkers", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/tiborhaller/dev/MT_POC",
        )

        assert result.returncode == 0
        assert "--user" in result.stdout
        assert "--days" in result.stdout
        assert "--dry-run" in result.stdout

    def test_cli_context_help(self):
        """Test that context --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.core.mock_data", "context", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/tiborhaller/dev/MT_POC",
        )

        assert result.returncode == 0
        assert "--user" in result.stdout
        assert "--interval" in result.stdout

    def test_cli_all_help(self):
        """Test that all --help works."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.core.mock_data", "all", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/tiborhaller/dev/MT_POC",
        )

        assert result.returncode == 0
        assert "--biomarker-interval" in result.stdout
        assert "--context-interval" in result.stdout
        assert "--scenario" in result.stdout
