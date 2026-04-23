"""Mock data generator for biomarker and context data.

This module provides config-driven generation of realistic mock data for testing
the analysis pipeline. All biomarkers and context markers are defined in YAML
config files, making the system fully extensible without code changes.
"""

import argparse
import logging
import math
import sys
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from numpy.random import PCG64, Generator
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from src.shared.database import SessionLocal
from src.shared.models import Biomarker, Context

logger = logging.getLogger(__name__)

__all__ = [
    # Config models
    "ActiveHoursModel",
    "BiomarkerParamsModel",
    "BiomarkersConfigModel",
    "ContextMarkerParamsModel",
    "ContextMarkersConfigModel",
    "DayPatternModel",
    "MockDataConfig",
    "ModalityConfigModel",
    "ScenarioConfigModel",
    "ScenarioOverrideModel",
    "ScheduleModel",
    # Generators
    "BaseGenerator",
    "ContextMarkerGenerator",
    "ModalityGenerator",
    "MockDataOrchestrator",
    # Functions
    "apply_daily_cycle",
    "load_mock_config",
    "load_scenario_config",
    "save_biomarkers",
    "save_context",
]


# =============================================================================
# Config Models
# =============================================================================


class BiomarkerParamsModel(BaseModel):
    """Parameters for a single biomarker."""

    model_config = ConfigDict(frozen=True)

    baseline: float
    variance: float
    daily_cycle: bool

    @field_validator("baseline", "variance")
    @classmethod
    def validate_range(cls, v: float) -> float:
        """Ensure values are in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0 and 1, got {v}")
        return v


class ModalityConfigModel(BaseModel):
    """Configuration for a single modality with its biomarkers."""

    model_config = ConfigDict(frozen=True)

    biomarkers: dict[str, BiomarkerParamsModel]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModalityConfigModel":
        """Create from raw dict, converting nested dicts to models."""
        biomarkers = {
            name: BiomarkerParamsModel(**params) for name, params in data.items()
        }
        return cls(biomarkers=biomarkers)


class BiomarkersConfigModel(BaseModel):
    """Top-level configuration for all biomarker modalities."""

    model_config = ConfigDict(frozen=True)

    modalities: dict[str, ModalityConfigModel]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BiomarkersConfigModel":
        """Create from raw dict."""
        modalities_data = data.get("modalities", {})
        modalities = {
            name: ModalityConfigModel.from_dict(biomarkers)
            for name, biomarkers in modalities_data.items()
        }
        return cls(modalities=modalities)


class ContextMarkerParamsModel(BaseModel):
    """Parameters for a single context marker.

    Context markers can use any scale (e.g., 0-10 for people count, 0-1 for ratios).
    The membership functions in context evaluation handle the scale conversion.
    """

    model_config = ConfigDict(frozen=True)

    baseline: float
    variance: float
    min_value: float = 0.0  # Optional floor for generated values
    max_value: float | None = None  # Optional ceiling for generated values
    enforce_integer_value: bool = False  # Round generated values to whole numbers

    @field_validator("baseline", "variance")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Ensure values are non-negative."""
        if v < 0.0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return v


class ContextMarkersConfigModel(BaseModel):
    """Configuration for context markers."""

    model_config = ConfigDict(frozen=True)

    context_type: str
    markers: dict[str, ContextMarkerParamsModel]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextMarkersConfigModel":
        """Create from raw dict."""
        markers = {
            name: ContextMarkerParamsModel(**params)
            for name, params in data.get("markers", {}).items()
        }
        return cls(
            context_type=data.get("context_type", "environment"), markers=markers
        )


class ScenarioOverrideModel(BaseModel):
    """Override parameters for a scenario."""

    model_config = ConfigDict(frozen=True, extra="allow")

    baseline: float | None = None
    variance: float | None = None


class ActiveHoursModel(BaseModel):
    """Time-of-day window for scenario activation."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0, le=23)  # 0-23
    end: int = Field(ge=1, le=24)  # 1-24

    @model_validator(mode="after")
    def validate_start_before_end(self) -> "ActiveHoursModel":
        """Validate that start < end (midnight crossing not supported)."""
        if self.start >= self.end:
            raise ValueError(
                f"active_hours.start ({self.start}) must be less than "
                f"active_hours.end ({self.end})"
            )
        return self


class DayPatternModel(BaseModel):
    """Cycle-based day pattern for scenario activation.

    Defines a repeating cycle of ``on`` active days followed by ``off``
    inactive days.  An optional ``offset`` shifts the start of the first
    cycle relative to day 0.

    Examples:
        - days_on=1, days_off=1, offset=0  →  active on days 0, 2, 4, …
        - days_on=2, days_off=1, offset=0  →  active on days 0-1, 3-4, 6-7, …
        - days_on=1, days_off=1, offset=1  →  active on days 1, 3, 5, …
    """

    model_config = ConfigDict(frozen=True)

    days_on: int = Field(ge=1)  # consecutive active days per cycle
    days_off: int = Field(ge=0)  # consecutive inactive days per cycle
    offset: int = Field(ge=0)  # day offset before first cycle starts


class ScheduleModel(BaseModel):
    """Schedule for when scenario overrides apply."""

    model_config = ConfigDict(frozen=True)

    active_hours: ActiveHoursModel | None = None
    day_pattern: DayPatternModel | None = None


class ScenarioConfigModel(BaseModel):
    """Configuration for a scenario preset."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    schedule: ScheduleModel | None = None
    overrides: dict[str, Any]  # Nested dict structure

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioConfigModel":
        """Create from raw dict."""
        schedule_data = data.get("schedule")
        schedule = None
        if schedule_data is not None:
            active_hours = None
            if "active_hours" in schedule_data:
                active_hours = ActiveHoursModel(**schedule_data["active_hours"])
            day_pattern = None
            if "day_pattern" in schedule_data:
                day_pattern = DayPatternModel(**schedule_data["day_pattern"])
            schedule = ScheduleModel(
                active_hours=active_hours, day_pattern=day_pattern
            )

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            schedule=schedule,
            overrides=data.get("overrides", {}),
        )


class MockDataConfig:
    """Container for all mock data configuration."""

    def __init__(
        self,
        biomarkers: BiomarkersConfigModel,
        context_markers: ContextMarkersConfigModel,
    ):
        self.biomarkers = biomarkers
        self.context_markers = context_markers


# =============================================================================
# Config Loaders
# =============================================================================


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML data

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty config file: {path}")

    return data


def load_mock_config(config_dir: Path | None = None) -> MockDataConfig:
    """Load mock data configuration from YAML files.

    Args:
        config_dir: Path to config directory (default: config/mock_data)

    Returns:
        MockDataConfig with all configuration loaded
    """
    if config_dir is None:
        # Default to config/mock_data relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config" / "mock_data"

    biomarkers_path = config_dir / "neutral_biomarkers.yaml"
    context_markers_path = config_dir / "neutral_context_markers.yaml"

    biomarkers_data = load_yaml_file(biomarkers_path)
    context_markers_data = load_yaml_file(context_markers_path)

    biomarkers_config = BiomarkersConfigModel.from_dict(biomarkers_data)
    context_markers_config = ContextMarkersConfigModel.from_dict(context_markers_data)

    logger.info(
        f"Loaded mock data config: {len(biomarkers_config.modalities)} modalities, "
        f"{len(context_markers_config.markers)} context markers"
    )

    return MockDataConfig(
        biomarkers=biomarkers_config, context_markers=context_markers_config
    )


def load_scenario_config(
    scenario_name: str, config_dir: Path | None = None
) -> ScenarioConfigModel:
    """Load a scenario configuration from YAML file.

    Args:
        scenario_name: Name of the scenario (e.g., 'solitary_digital')
        config_dir: Path to config directory (default: config/mock_data)

    Returns:
        ScenarioConfigModel with scenario configuration
    """
    if config_dir is None:
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "config" / "mock_data"

    scenario_path = config_dir / "scenarios" / f"{scenario_name}.yaml"
    scenario_data = load_yaml_file(scenario_path)

    return ScenarioConfigModel.from_dict(scenario_data)


# =============================================================================
# Generator Classes
# =============================================================================


def apply_daily_cycle(base_value: float, timestamp: datetime) -> float:
    """Apply daily cycle pattern to a base value.

    Peak activity at 14:00 (2 PM), lowest at 03:00 (3 AM).
    Uses sine wave with period of 24 hours.

    Args:
        base_value: The baseline value (0-1)
        timestamp: The timestamp for this data point

    Returns:
        Adjusted value with daily cycle applied (0-1)
    """
    hour = timestamp.hour + timestamp.minute / 60.0

    # Shift so peak is at 14:00 (2 PM) and trough at 02:00 (2 AM)
    phase_shift = 8  # hours before noon for trough

    cycle_factor = math.sin((hour - phase_shift) * math.pi / 12)

    # Modulate baseline by +/- 20% based on time of day
    modulation = 0.2 * cycle_factor

    adjusted = base_value * (1 + modulation)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, adjusted))


class BaseGenerator(ABC):
    """Abstract base class for data generators."""

    def __init__(self, rng: Generator):
        """Initialize generator with random number generator.

        Args:
            rng: Numpy random generator for reproducibility
        """
        self._rng = rng

    def generate_value(
        self,
        baseline: float,
        variance: float,
        timestamp: datetime,
        daily_cycle: bool = False,
    ) -> float:
        """Generate a single value with optional daily cycle.

        Args:
            baseline: Base value (0-1)
            variance: Random variance to apply
            timestamp: Timestamp for this value
            daily_cycle: Whether to apply daily cycle pattern

        Returns:
            Generated value clamped to [0, 1]
        """
        # Add random noise
        noise = self._rng.uniform(-variance, variance)
        value = baseline + noise

        # Apply daily cycle if enabled
        if daily_cycle:
            value = apply_daily_cycle(value, timestamp)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, value))

    @abstractmethod
    def generate_snapshot(self, timestamp: datetime) -> dict[str, float]:
        """Generate a snapshot of values at a given timestamp.

        Args:
            timestamp: Timestamp for this snapshot

        Returns:
            Dict mapping field names to values
        """
        pass


class ModalityGenerator(BaseGenerator):
    """Generator for a single modality's biomarkers."""

    def __init__(
        self,
        modality_name: str,
        modality_config: ModalityConfigModel,
        rng: Generator,
        overrides: dict[str, Any] | None = None,
    ):
        """Initialize modality generator.

        Args:
            modality_name: Name of the modality (e.g., 'speech')
            modality_config: Configuration for this modality
            rng: Random number generator
            overrides: Optional scenario overrides for biomarker params
        """
        super().__init__(rng)
        self.modality_name = modality_name
        self.modality_config = modality_config
        self.overrides = overrides or {}

    def generate_snapshot(self, timestamp: datetime) -> dict[str, float]:
        """Generate biomarker values for this modality at a timestamp.

        Args:
            timestamp: Timestamp for this snapshot

        Returns:
            Dict mapping biomarker names to values
        """
        snapshot = {}

        for biomarker_name, params in self.modality_config.biomarkers.items():
            # Check for scenario overrides
            baseline = params.baseline
            variance = params.variance
            daily_cycle = params.daily_cycle

            if biomarker_name in self.overrides:
                override = self.overrides[biomarker_name]
                baseline = override.get("baseline", baseline)
                variance = override.get("variance", variance)

            value = self.generate_value(baseline, variance, timestamp, daily_cycle)
            snapshot[biomarker_name] = value

        return snapshot


class ContextMarkerGenerator(BaseGenerator):
    """Generator for context markers."""

    def __init__(
        self,
        context_config: ContextMarkersConfigModel,
        rng: Generator,
        overrides: dict[str, Any] | None = None,
    ):
        """Initialize context marker generator.

        Args:
            context_config: Configuration for context markers
            rng: Random number generator
            overrides: Optional scenario overrides for marker params
        """
        super().__init__(rng)
        self.context_config = context_config
        self.overrides = overrides or {}

    def generate_snapshot(self, timestamp: datetime) -> dict[str, float]:
        """Generate context marker values at a timestamp.

        Args:
            timestamp: Timestamp for this snapshot

        Returns:
            Dict mapping marker names to values
        """
        snapshot = {}

        for marker_name, params in self.context_config.markers.items():
            # Check for scenario overrides
            baseline = params.baseline
            variance = params.variance
            min_value = params.min_value
            max_value = params.max_value
            enforce_integer_value = params.enforce_integer_value

            if marker_name in self.overrides:
                override = self.overrides[marker_name]
                baseline = override.get("baseline", baseline)
                variance = override.get("variance", variance)
                min_value = override.get("min_value", min_value)
                max_value = override.get("max_value", max_value)
                enforce_integer_value = override.get(
                    "enforce_integer_value", enforce_integer_value
                )

            # Generate value with noise
            noise = self._rng.uniform(-variance, variance)
            value = baseline + noise

            # Clamp to marker's scale (not 0-1)
            value = max(min_value, value)
            if max_value is not None:
                value = min(max_value, value)

            # Round to integer if configured
            if enforce_integer_value:
                value = round(value)

            snapshot[marker_name] = value

        return snapshot


class MockDataOrchestrator:
    """Orchestrates generation of biomarker and context data."""

    def __init__(
        self,
        config: MockDataConfig,
        seed: int | None = None,
        scenario: str | None = None,
    ):
        """Initialize orchestrator.

        Args:
            config: Mock data configuration
            seed: Random seed for reproducibility
            scenario: Optional scenario name to apply overrides
        """
        self.config = config
        self._rng = Generator(PCG64(seed)) if seed is not None else Generator(PCG64())

        # Load scenario overrides if specified
        self.scenario_config = None
        if scenario:
            self.scenario_config = load_scenario_config(scenario)
            logger.info(f"Applied scenario: {self.scenario_config.name}")

    def _get_biomarker_overrides(self, modality_name: str) -> dict[str, Any]:
        """Get scenario overrides for a specific modality.

        Args:
            modality_name: Name of the modality

        Returns:
            Dict of overrides for this modality
        """
        if not self.scenario_config:
            return {}

        overrides = self.scenario_config.overrides.get("biomarkers", {})
        return overrides.get(modality_name, {})

    def _get_context_overrides(self) -> dict[str, Any]:
        """Get scenario overrides for context markers.

        Returns:
            Dict of overrides for context markers
        """
        if not self.scenario_config:
            return {}

        return self.scenario_config.overrides.get("context", {})

    def _is_scenario_active(
        self, timestamp: datetime, generation_start: datetime
    ) -> bool:
        """Check if scenario overrides should be applied at this timestamp.

        Args:
            timestamp: Current generation timestamp
            generation_start: Start of the generation run (for day index calculation)

        Returns:
            True if scenario overrides should be applied
        """
        if not self.scenario_config:
            return False

        schedule = self.scenario_config.schedule
        if schedule is None:
            return True  # No schedule = always active (backward compat)

        # Check active hours
        if schedule.active_hours is not None:
            if not (schedule.active_hours.start <= timestamp.hour < schedule.active_hours.end):
                return False

        # Check day pattern (cycle-based: on/off)
        if schedule.day_pattern is not None:
            day_index = (timestamp.date() - generation_start.date()).days
            dp = schedule.day_pattern
            cycle_length = dp.days_on + dp.days_off
            day_in_cycle = (day_index - dp.offset) % cycle_length
            if day_in_cycle < 0:
                day_in_cycle += cycle_length
            if day_in_cycle >= dp.days_on:
                return False

        return True

    def generate_biomarkers(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
        modalities: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate biomarker data for specified time range.

        Args:
            user_id: User ID for the data
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (exclusive)
            interval_minutes: Minutes between data points
            modalities: List of modality names to generate (default: all)

        Returns:
            List of biomarker records ready for DB insertion
        """
        records = []

        # Determine which modalities to generate
        available_modalities = self.config.biomarkers.modalities.keys()
        if modalities is None:
            modalities_to_generate = list(available_modalities)
        else:
            modalities_to_generate = [
                m for m in modalities if m in available_modalities
            ]

        # Generate timestamps
        current_time = start_time
        interval_delta = timedelta(minutes=interval_minutes)

        while current_time < end_time:
            scenario_active = self._is_scenario_active(current_time, start_time)

            # Generate for each modality
            for modality_name in modalities_to_generate:
                modality_config = self.config.biomarkers.modalities[modality_name]
                overrides = (
                    self._get_biomarker_overrides(modality_name)
                    if scenario_active
                    else {}
                )

                generator = ModalityGenerator(
                    modality_name=modality_name,
                    modality_config=modality_config,
                    rng=self._rng,
                    overrides=overrides,
                )

                snapshot = generator.generate_snapshot(current_time)

                record = {
                    "user_id": user_id,
                    "timestamp": current_time,
                    "biomarker_type": modality_name,
                    "value": snapshot,
                    "metadata_": {
                        "generator": "mock_data",
                        "version": "1.0",
                        "source": f"mock_{modality_name}_sensor",
                        "scenario_active": scenario_active,
                    },
                }
                records.append(record)

            current_time += interval_delta

        return records

    def generate_context(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int,
    ) -> list[dict[str, Any]]:
        """Generate context data for specified time range.

        Args:
            user_id: User ID for the data
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (exclusive)
            interval_minutes: Minutes between data points

        Returns:
            List of context records ready for DB insertion
        """
        records = []

        # Generate timestamps
        current_time = start_time
        interval_delta = timedelta(minutes=interval_minutes)

        while current_time < end_time:
            scenario_active = self._is_scenario_active(current_time, start_time)
            overrides = self._get_context_overrides() if scenario_active else {}

            generator = ContextMarkerGenerator(
                context_config=self.config.context_markers,
                rng=self._rng,
                overrides=overrides,
            )

            snapshot = generator.generate_snapshot(current_time)

            record = {
                "user_id": user_id,
                "timestamp": current_time,
                "context_type": self.config.context_markers.context_type,
                "value": snapshot,
                "metadata_": {
                    "generator": "mock_data",
                    "version": "1.0",
                    "source": f"mock_{self.config.context_markers.context_type}_sensor",
                    "scenario_active": scenario_active,
                },
            }
            records.append(record)

            current_time += interval_delta

        return records

    def generate_all(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        biomarker_interval: int,
        context_interval: int,
        modalities: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Generate both biomarker and context data with independent intervals.

        Args:
            user_id: User ID for the data
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (exclusive)
            biomarker_interval: Minutes between biomarker samples
            context_interval: Minutes between context samples
            modalities: List of modality names to generate (default: all)

        Returns:
            Tuple of (biomarker_records, context_records)
        """
        biomarkers = self.generate_biomarkers(
            user_id, start_time, end_time, biomarker_interval, modalities
        )
        context = self.generate_context(user_id, start_time, end_time, context_interval)

        return biomarkers, context


# =============================================================================
# Database Persistence
# =============================================================================


def save_biomarkers(biomarkers: list[dict[str, Any]], session: Session) -> int:
    """Batch save biomarker records to database.

    Args:
        biomarkers: List of biomarker dicts matching ORM structure
        session: SQLAlchemy session

    Returns:
        Number of records saved
    """
    records = [Biomarker(**b) for b in biomarkers]
    session.add_all(records)
    session.commit()

    logger.info(f"Saved {len(records)} biomarker records to database")
    return len(records)


def save_context(markers: list[dict[str, Any]], session: Session) -> int:
    """Batch save context records to database.

    Args:
        markers: List of context dicts matching ORM structure
        session: SQLAlchemy session

    Returns:
        Number of records saved
    """
    records = [Context(**m) for m in markers]
    session.add_all(records)
    session.commit()

    logger.info(f"Saved {len(records)} context records to database")
    return len(records)


# =============================================================================
# CLI Interface
# =============================================================================


def parse_modalities(modalities_str: str | None) -> list[str] | None:
    """Parse comma-separated modalities string.

    Args:
        modalities_str: Comma-separated modality names or None

    Returns:
        List of modality names or None for all
    """
    if modalities_str is None:
        return None
    return [m.strip() for m in modalities_str.split(",")]


def calculate_time_range(days: int) -> tuple[datetime, datetime]:
    """Calculate start and end times for data generation.

    Args:
        days: Number of days to generate

    Returns:
        Tuple of (start_time, end_time) in UTC
    """
    now = datetime.now(UTC)
    end_time = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_time = end_time - timedelta(days=days)
    return start_time, end_time


def handle_biomarkers_command(args: argparse.Namespace) -> int:
    """Handle the 'biomarkers' subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    config = load_mock_config()
    start_time, end_time = calculate_time_range(args.days)

    orchestrator = MockDataOrchestrator(config=config, seed=args.seed)

    modalities = parse_modalities(args.modalities)
    biomarkers = orchestrator.generate_biomarkers(
        user_id=args.user,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=args.interval,
        modalities=modalities,
    )

    if args.dry_run:
        print(f"Would generate {len(biomarkers)} biomarker records")
        print(f"User: {args.user}")
        print(f"Time range: {start_time} to {end_time}")
        print(f"Interval: {args.interval} minutes")
        if modalities:
            print(f"Modalities: {', '.join(modalities)}")
    else:
        session = SessionLocal()
        try:
            count = save_biomarkers(biomarkers, session)
            print(f"Generated and saved {count} biomarker records")
        finally:
            session.close()

    return 0


def handle_context_command(args: argparse.Namespace) -> int:
    """Handle the 'context' subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    config = load_mock_config()
    start_time, end_time = calculate_time_range(args.days)

    orchestrator = MockDataOrchestrator(config=config, seed=args.seed)

    context = orchestrator.generate_context(
        user_id=args.user,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=args.interval,
    )

    if args.dry_run:
        print(f"Would generate {len(context)} context records")
        print(f"User: {args.user}")
        print(f"Time range: {start_time} to {end_time}")
        print(f"Interval: {args.interval} minutes")
    else:
        session = SessionLocal()
        try:
            count = save_context(context, session)
            print(f"Generated and saved {count} context records")
        finally:
            session.close()

    return 0


def handle_all_command(args: argparse.Namespace) -> int:
    """Handle the 'all' subcommand.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success)
    """
    config = load_mock_config()
    start_time, end_time = calculate_time_range(args.days)

    orchestrator = MockDataOrchestrator(
        config=config, seed=args.seed, scenario=args.scenario
    )

    modalities = parse_modalities(args.modalities)
    biomarkers, context = orchestrator.generate_all(
        user_id=args.user,
        start_time=start_time,
        end_time=end_time,
        biomarker_interval=args.biomarker_interval,
        context_interval=args.context_interval,
        modalities=modalities,
    )

    if args.dry_run:
        print(f"Would generate {len(biomarkers)} biomarker records")
        print(f"Would generate {len(context)} context records")
        print(f"User: {args.user}")
        print(f"Time range: {start_time} to {end_time}")
        print(f"Biomarker interval: {args.biomarker_interval} minutes")
        print(f"Context interval: {args.context_interval} minutes")
        if args.scenario:
            print(f"Scenario: {args.scenario}")
        if modalities:
            print(f"Modalities: {', '.join(modalities)}")
    else:
        session = SessionLocal()
        try:
            bio_count = save_biomarkers(biomarkers, session)
            ctx_count = save_context(context, session)
            print(f"Generated and saved {bio_count} biomarker records")
            print(f"Generated and saved {ctx_count} context records")
        finally:
            session.close()

    return 0


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python -m src.core.mock_data",
        description="Generate mock biomarker and context data for testing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments function
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--user", required=True, help="User ID")
        p.add_argument("--days", type=int, default=14, help="Days of data (default: 14)")
        p.add_argument("--seed", type=int, help="Random seed for reproducibility")
        p.add_argument(
            "--dry-run", action="store_true", help="Print count without DB write"
        )

    # biomarkers subcommand
    bio_parser = subparsers.add_parser(
        "biomarkers", help="Generate biomarker data only"
    )
    add_common_args(bio_parser)
    bio_parser.add_argument(
        "--interval", type=int, default=15, help="Minutes between samples (default: 15)"
    )
    bio_parser.add_argument(
        "--modalities", help="Comma-separated modalities (default: all)"
    )
    bio_parser.set_defaults(func=handle_biomarkers_command)

    # context subcommand
    ctx_parser = subparsers.add_parser("context", help="Generate context data only")
    add_common_args(ctx_parser)
    ctx_parser.add_argument(
        "--interval", type=int, default=60, help="Minutes between samples (default: 60)"
    )
    ctx_parser.set_defaults(func=handle_context_command)

    # all subcommand
    all_parser = subparsers.add_parser(
        "all", help="Generate both biomarker and context data"
    )
    add_common_args(all_parser)
    all_parser.add_argument(
        "--biomarker-interval",
        type=int,
        default=15,
        help="Minutes between biomarker samples (default: 15)",
    )
    all_parser.add_argument(
        "--context-interval",
        type=int,
        default=60,
        help="Minutes between context samples (default: 60)",
    )
    all_parser.add_argument("--scenario", help="Scenario preset name")
    all_parser.add_argument(
        "--modalities", help="Comma-separated modalities (default: all)"
    )
    all_parser.set_defaults(func=handle_all_command)

    # Parse and execute
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
