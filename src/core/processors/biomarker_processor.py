"""Biomarker processor for normalization and membership computation."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from src.core.config import AnalysisConfig
from src.core.processors.baseline_repository import BaselineRepository
from src.core.processors.membership import BiomarkerMembershipCalculator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineStats:
    """Statistical baseline for biomarker normalization."""

    mean: float
    std: float
    percentile_25: float | None = None
    percentile_75: float | None = None
    data_points: int = 0
    source: str = "user"  # "user" or "population"

    @property
    def is_population_baseline(self) -> bool:
        """Check if this is a population baseline."""
        return self.source == "population"


@dataclass(frozen=True)
class BiomarkerMembership:
    """Biomarker membership computation result."""

    name: str
    membership: float | None  # None if unavailable
    z_score: float | None
    raw_value: float
    baseline: BaselineStats | None
    data_points_used: int
    data_quality: float  # 0-1
    membership_function_used: str
    timestamp: datetime


@dataclass(frozen=True)
class DailyBiomarkerMembership:
    """Biomarker membership for a specific day.

    Extends BiomarkerMembership with an explicit date field for daily series.
    """

    date: date
    name: str
    membership: float | None
    z_score: float | None
    raw_value: float
    baseline: BaselineStats | None
    data_points_used: int
    data_quality: float
    membership_function_used: str
    timestamp: datetime


class BiomarkerProcessor:
    """Process biomarkers to compute normalized membership values."""

    def __init__(self, config: AnalysisConfig, session):
        """Initialize processor.

        Args:
            config: Analysis configuration
            session: SQLAlchemy session

        """
        self._config = config
        self._session = session
        self._repository = BaselineRepository(session)
        self._calculator = BiomarkerMembershipCalculator()
        self._logger = logging.getLogger(__name__)

    def process_biomarkers(
        self,
        biomarkers: list,
        user_id: str,
        analysis_date: date,
    ) -> dict[str, BiomarkerMembership]:
        """Process biomarkers and compute membership values.

        Args:
            biomarkers: List of BiomarkerRecord instances
            user_id: User ID for baseline retrieval
            analysis_date: Date of analysis

        Returns:
            Dictionary mapping biomarker name to BiomarkerMembership

        """
        self._logger.info(
            f"Processing {len(biomarkers)} biomarker records for user {user_id}"
        )

        # Group biomarkers by name
        grouped = defaultdict(list)
        for record in biomarkers:
            grouped[record.name].append(record)

        results = {}
        analysis_timestamp = datetime.now(UTC)

        for biomarker_name, records in grouped.items():
            try:
                membership = self._process_single_biomarker(
                    biomarker_name, records, user_id, analysis_date, analysis_timestamp
                )
                results[biomarker_name] = membership
            except Exception as e:
                self._logger.error(
                    f"Error processing biomarker {biomarker_name}: {e}",
                    exc_info=True,
                )
                # Return unavailable membership on error
                results[biomarker_name] = self._create_unavailable_membership(
                    biomarker_name, analysis_timestamp
                )

        self._logger.info(
            f"Processed {len(results)} biomarkers: {list(results.keys())}"
        )
        return results

    def process_biomarkers_daily(
        self,
        biomarkers: list,
        user_id: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, list[DailyBiomarkerMembership]]:
        """Process biomarkers and compute daily membership values.

        Computes baseline ONCE from all records, then computes membership
        values for each day separately using the shared baseline. This
        implements the algorithm from MA-Stephan-Nef Algorithm 5.4.

        Args:
            biomarkers: List of BiomarkerRecord instances
            user_id: User ID for baseline retrieval
            start_date: Start date of analysis window (inclusive)
            end_date: End date of analysis window (inclusive)

        Returns:
            Dictionary mapping biomarker name to list of DailyBiomarkerMembership,
            ordered by date. Days with no data for a biomarker are skipped.

        """
        self._logger.info(
            f"Processing {len(biomarkers)} biomarker records for daily series "
            f"[{start_date} to {end_date}] for user {user_id}"
        )

        # Group biomarkers by name for baseline computation
        grouped_by_name = defaultdict(list)
        for record in biomarkers:
            grouped_by_name[record.name].append(record)

        # Step 1: Compute baseline ONCE per biomarker from ALL records
        baselines: dict[str, BaselineStats] = {}
        for biomarker_name, records in grouped_by_name.items():
            baselines[biomarker_name] = self._get_or_compute_baseline(
                user_id, biomarker_name, records
            )
        self._logger.debug(
            f"Computed baselines for {len(baselines)} biomarkers (computed once)"
        )

        # Step 2: Aggregate biomarkers by day
        daily_records = self.aggregate_by_day(biomarkers, start_date, end_date)

        # Step 3: For each day, compute membership for each biomarker
        results: dict[str, list[DailyBiomarkerMembership]] = defaultdict(list)
        analysis_timestamp = datetime.now(UTC)

        # Iterate through days in order
        for current_date in sorted(daily_records.keys()):
            day_biomarkers = daily_records[current_date]

            for biomarker_name, records in day_biomarkers.items():
                baseline = baselines.get(biomarker_name)
                if baseline is None:
                    self._logger.warning(
                        f"No baseline for {biomarker_name} on {current_date}"
                    )
                    continue

                # Aggregate values for this day
                aggregated_value = float(np.mean([r.value for r in records]))

                # Compute z-score using shared baseline
                z_score = self._compute_z_score(aggregated_value, baseline)

                # Apply membership function
                membership_fn = self._config.biomarker_membership.get(biomarker_name)
                if membership_fn:
                    membership_value = self._calculator.calculate(
                        z_score,
                        {"type": membership_fn.type, "params": membership_fn.params},
                    )
                    fn_used = membership_fn.type
                else:
                    z_bounds = self._config.biomarker_processing.z_score_bounds
                    membership_value = max(
                        0.0, min(1.0, (z_score - z_bounds.lower) / z_bounds.range)
                    )
                    fn_used = "linear_default"

                # Compute data quality
                if biomarker_name in self._config.biomarker_defaults:
                    min_data_points = self._config.biomarker_defaults[
                        biomarker_name
                    ].min_data_points
                else:
                    min_data_points = (
                        self._config.biomarker_processing.default_min_data_points
                    )

                data_quality = self._compute_data_quality(
                    baseline.data_points,
                    min_data_points,
                    baseline.is_population_baseline,
                )

                daily_membership = DailyBiomarkerMembership(
                    date=current_date,
                    name=biomarker_name,
                    membership=membership_value,
                    z_score=z_score,
                    raw_value=aggregated_value,
                    baseline=baseline,
                    data_points_used=len(records),
                    data_quality=data_quality,
                    membership_function_used=fn_used,
                    timestamp=analysis_timestamp,
                )
                results[biomarker_name].append(daily_membership)

        # Sort each biomarker's list by date
        for biomarker_name in results:
            results[biomarker_name].sort(key=lambda m: m.date)

        total_days = len(daily_records)
        self._logger.info(
            f"Computed daily memberships: {len(results)} biomarkers across "
            f"{total_days} days"
        )

        return dict(results)

    def _process_single_biomarker(
        self,
        biomarker_name: str,
        records: list,
        user_id: str,
        analysis_date: date,
        analysis_timestamp: datetime,
    ) -> BiomarkerMembership:
        """Process a single biomarker.

        Args:
            biomarker_name: Name of biomarker
            records: List of records for this biomarker
            user_id: User ID
            analysis_date: Analysis date
            analysis_timestamp: Analysis timestamp

        Returns:
            BiomarkerMembership result

        """
        if not records:
            self._logger.warning(f"No data for biomarker {biomarker_name}")
            return self._create_unavailable_membership(
                biomarker_name, analysis_timestamp
            )

        # Get or compute baseline
        baseline = self._get_or_compute_baseline(user_id, biomarker_name, records)

        # Aggregate values for target date
        aggregated_value = self._aggregate_biomarker_values(records, analysis_date)

        # Compute z-score
        z_score = self._compute_z_score(aggregated_value, baseline)

        # Apply membership function
        membership_fn = self._config.biomarker_membership.get(biomarker_name)
        if membership_fn:
            membership_value = self._calculator.calculate(
                z_score, {"type": membership_fn.type, "params": membership_fn.params}
            )
            fn_used = membership_fn.type
        else:
            # No membership function configured, use z-score linearly mapped to [0,1]
            # using configured bounds (default: [-3.0, 3.0])
            z_bounds = self._config.biomarker_processing.z_score_bounds
            membership_value = max(
                0.0, min(1.0, (z_score - z_bounds.lower) / z_bounds.range)
            )
            fn_used = "linear_default"
            self._logger.warning(
                f"No membership function configured for {biomarker_name}, using default"
            )

        # Compute data quality
        # Get min_data_points from biomarker config or use default from processing config
        if biomarker_name in self._config.biomarker_defaults:
            min_data_points = self._config.biomarker_defaults[
                biomarker_name
            ].min_data_points
        else:
            min_data_points = self._config.biomarker_processing.default_min_data_points

        data_quality = self._compute_data_quality(
            baseline.data_points,
            min_data_points,
            baseline.is_population_baseline,
        )

        return BiomarkerMembership(
            name=biomarker_name,
            membership=membership_value,
            z_score=z_score,
            raw_value=aggregated_value,
            baseline=baseline,
            data_points_used=len(records),
            data_quality=data_quality,
            membership_function_used=fn_used,
            timestamp=analysis_timestamp,
        )

    def _get_or_compute_baseline(
        self,
        user_id: str,
        biomarker_name: str,
        historical_records: list,
    ) -> BaselineStats:
        """Get or compute baseline for biomarker.

        Args:
            user_id: User ID
            biomarker_name: Name of biomarker
            historical_records: Historical records for baseline computation

        Returns:
            BaselineStats (user-specific or population)

        """
        min_data_points = (
            self._config.biomarker_defaults.get(biomarker_name).min_data_points
            if biomarker_name in self._config.biomarker_defaults
            else self._config.biomarker_processing.default_min_data_points
        )

        # Check if user has enough data
        if len(historical_records) >= min_data_points:
            # Compute user-specific baseline
            values = [r.value for r in historical_records]
            mean = float(np.mean(values))
            std = float(np.std(values))
            p25 = float(np.percentile(values, 25))
            p75 = float(np.percentile(values, 75))

            self._logger.info(
                f"Using user-specific baseline for {biomarker_name}: "
                f"mean={mean:.3f}, std={std:.3f}, n={len(values)}"
            )

            # Use configured minimum std floor to avoid division by zero
            min_std = self._config.biomarker_processing.min_std_deviation
            return BaselineStats(
                mean=mean,
                std=std if std > min_std else min_std,
                percentile_25=p25,
                percentile_75=p75,
                data_points=len(values),
                source="user",
            )

        # Fall back to population defaults
        population_default = self._config.biomarker_defaults.get(biomarker_name)
        if population_default is None:
            # Use configured generic baseline fallback
            generic = self._config.biomarker_processing.generic_baseline
            self._logger.warning(
                f"No population default for {biomarker_name}, using generic defaults "
                f"(mean={generic.mean}, std={generic.std})"
            )
            return BaselineStats(
                mean=generic.mean, std=generic.std, data_points=0, source="generic"
            )

        self._logger.info(
            f"Using population baseline for {biomarker_name}: "
            f"mean={population_default.mean:.3f}, std={population_default.std:.3f} "
            f"(user has {len(historical_records)} < {min_data_points} required)"
        )

        return BaselineStats(
            mean=population_default.mean,
            std=population_default.std,
            data_points=len(historical_records),
            source="population",
        )

    def _aggregate_biomarker_values(self, records: list, target_date: date) -> float:
        """Aggregate biomarker values for target date.

        Args:
            records: List of biomarker records
            target_date: Target date

        Returns:
            Aggregated value (mean of values on target date)

        """
        # Filter records for target date
        target_records = [r for r in records if r.timestamp.date() == target_date]

        if not target_records:
            # Use most recent value if no data for target date
            sorted_records = sorted(records, key=lambda r: r.timestamp, reverse=True)
            return sorted_records[0].value

        # Return mean of values on target date
        return float(np.mean([r.value for r in target_records]))

    def aggregate_by_day(
        self, biomarkers: list, start_date: date, end_date: date
    ) -> dict[date, dict[str, list]]:
        """Aggregate biomarker records by day and by name.

        Groups all biomarker records by date and biomarker name, returning
        a structure suitable for daily membership computation.

        Args:
            biomarkers: List of BiomarkerRecord instances
            start_date: Start date of the analysis window (inclusive)
            end_date: End date of the analysis window (inclusive)

        Returns:
            Dictionary mapping date -> {biomarker_name -> list[BiomarkerRecord]}
            Only includes days that have at least one biomarker record.

        """
        # Initialize structure for all days in range
        result: dict[date, dict[str, list]] = {}

        # Group biomarkers by date and name
        for record in biomarkers:
            record_date = record.timestamp.date()
            # Only include records within the date range
            if start_date <= record_date <= end_date:
                if record_date not in result:
                    result[record_date] = defaultdict(list)
                result[record_date][record.name].append(record)

        # Convert defaultdicts to regular dicts and log
        result = {d: dict(by_name) for d, by_name in result.items()}

        self._logger.debug(
            f"Aggregated biomarkers by day: {len(result)} days with data "
            f"in range [{start_date}, {end_date}]"
        )

        return result

    def _compute_z_score(self, value: float, baseline: BaselineStats) -> float:
        """Compute z-score from value and baseline.

        Args:
            value: Raw value
            baseline: Baseline statistics

        Returns:
            Z-score

        """
        if baseline.std == 0:
            self._logger.warning("Zero std deviation, using z=0.0")
            return 0.0

        z_score = (value - baseline.mean) / baseline.std

        # Use configured warning threshold for extreme z-scores
        warning_threshold = self._config.biomarker_processing.z_score_warning_threshold
        if abs(z_score) > warning_threshold:
            self._logger.warning(
                f"Extreme z-score: z={z_score:.2f} (threshold: {warning_threshold})"
            )

        return z_score

    def _compute_data_quality(
        self, data_points: int, min_required: int, is_population: bool
    ) -> float:
        """Compute data quality score.

        Args:
            data_points: Number of data points used
            min_required: Minimum required data points
            is_population: Whether using population baseline

        Returns:
            Quality score in [0, 1]

        """
        quality = min(1.0, data_points / min_required)

        # Population baselines get penalized quality (configurable, default 50%)
        if is_population:
            penalty = self._config.reliability.population_baseline_quality_penalty
            quality *= penalty

        return quality

    def _create_unavailable_membership(
        self, biomarker_name: str, timestamp: datetime
    ) -> BiomarkerMembership:
        """Create unavailable membership result.

        Args:
            biomarker_name: Name of biomarker
            timestamp: Timestamp

        Returns:
            BiomarkerMembership with None values

        """
        return BiomarkerMembership(
            name=biomarker_name,
            membership=None,
            z_score=None,
            raw_value=0.0,
            baseline=None,
            data_points_used=0,
            data_quality=0.0,
            membership_function_used="none",
            timestamp=timestamp,
        )
