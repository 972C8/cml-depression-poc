"""Scenario 4: All-Indicators Degradation Heatmap — Sensor Robustness Evaluation.

Generates a heatmap visualising how all 9 DSM-5 indicator scores change under
systematic sensor degradation conditions, compared to a full-data baseline.

Each cell represents the mean absolute deviation across 14 days for one
indicator under one degradation condition.

Degradation conditions (heatmap rows) use positional biomarker selection:
    1. Drop 1st speech biomarker        — first speech biomarker removed
    2. Drop 1st network biomarker       — first network biomarker removed
    3. Drop 1st speech + 1st network    — first of each modality removed
    4. Drop speech modality             — all speech biomarkers removed
    5. Drop network modality            — all network biomarkers removed
    6. Faulty sensor (speech, stuck@1)  — first speech biomarker overridden

Special handling for indicator 7 (network-only, no speech biomarkers):
    - Speech dropout rows naturally produce 0.0 deviation.
    - Row 6 faults the first network biomarker instead.

Uses the same mock data generation infrastructure as Scenarios 1–3
(solitary_digital context, 14 days, seed 42) and the existing FASL
indicator computation pipeline.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so that src.* imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.mock_data import MockDataOrchestrator, load_mock_config
from src.core.processors.window_fasl import apply_fasl_operator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
DAYS = 14
USER_ID = "eval_scenario4_all"
SCENARIO = "solitary_digital"
BIOMARKER_INTERVAL_MIN = 15  # minutes between samples

INDICATORS_CONFIG_PATH = PROJECT_ROOT / "config" / "indicators.yaml"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "thesis"
    / "MA_Tibor_Haller"
    / "content"
    / "5_evaluation"
    / "scenario4_degradation_heatmap_all_indicators.png"
)

# Biomarker processing defaults (matching src/core/config.py defaults)
Z_LOWER = -3.0
Z_UPPER = 3.0
Z_RANGE = Z_UPPER - Z_LOWER
MIN_STD = 0.1

# Degradation condition labels (generic across indicators)
CONDITION_LABELS = [
    "Drop 1 speech biomarker",
    "Drop 1 network biomarker",
    "Drop 1 speech +\n1 network biomarker",
    "Drop speech modality",
    "Drop network modality",
    "Faulty sensor\n(speech, stuck at 1.0)",
]


# ---------------------------------------------------------------------------
# Indicator configuration loading
# ---------------------------------------------------------------------------


def load_indicators_config(
    config_path: Path,
) -> dict[str, dict[str, dict]]:
    """Load all indicator definitions from indicators.yaml.

    Returns:
        Dict mapping indicator name to its biomarker configuration.
        Each biomarker config includes 'type', 'weight', and 'direction'.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    indicators = {}
    for indicator_name, indicator_data in raw.items():
        indicators[indicator_name] = indicator_data["biomarkers"]
    return indicators


def get_modality_biomarkers(
    biomarkers: dict[str, dict],
    modality: str,
) -> list[str]:
    """Return biomarker names for the given modality, preserving YAML order."""
    return [name for name, cfg in biomarkers.items() if cfg["type"] == modality]


# ---------------------------------------------------------------------------
# Degradation condition generation per indicator
# ---------------------------------------------------------------------------


def build_conditions_for_indicator(
    biomarkers: dict[str, dict],
) -> list[dict]:
    """Build the 6 degradation conditions for a given indicator.

    Uses positional biomarker selection: 'first speech biomarker' is the
    first biomarker with type='speech' in the indicator's config ordering.

    Special handling for indicators with no speech biomarkers (e.g. indicator 7):
    - Drop conditions that reference speech produce empty drop lists
      (score equals baseline, deviation = 0).
    - Row 6 (faulty sensor) falls back to the first network biomarker.
    """
    speech = get_modality_biomarkers(biomarkers, "speech")
    network = get_modality_biomarkers(biomarkers, "network")

    conditions = []

    # Row 1: Drop first speech biomarker
    drop_1 = [speech[0]] if speech else []
    conditions.append({"drop": drop_1, "fault": {}})

    # Row 2: Drop first network biomarker
    drop_2 = [network[0]] if network else []
    conditions.append({"drop": drop_2, "fault": {}})

    # Row 3: Drop first speech + first network
    drop_3 = drop_1 + drop_2
    conditions.append({"drop": drop_3, "fault": {}})

    # Row 4: Drop entire speech modality
    conditions.append({"drop": list(speech), "fault": {}})

    # Row 5: Drop entire network modality
    conditions.append({"drop": list(network), "fault": {}})

    # Row 6: Faulty sensor — first speech biomarker stuck at 1.0
    # Fallback to first network biomarker if no speech biomarkers exist
    if speech:
        fault_target = speech[0]
    else:
        fault_target = network[0]
    conditions.append({"drop": [], "fault": {fault_target: 1.0}})

    return conditions


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_mock_data(seed: int, days: int) -> list[dict]:
    """Generate mock biomarker data using the existing infrastructure.

    Returns the raw biomarker records (list of dicts) without persisting
    to the database.
    """
    config = load_mock_config()
    orchestrator = MockDataOrchestrator(config=config, seed=seed, scenario=SCENARIO)

    now = datetime.now(UTC)
    end_time = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    start_time = end_time - timedelta(days=days)

    records = orchestrator.generate_biomarkers(
        user_id=USER_ID,
        start_time=start_time,
        end_time=end_time,
        interval_minutes=BIOMARKER_INTERVAL_MIN,
    )
    return records


# ---------------------------------------------------------------------------
# Biomarker extraction and daily aggregation
# ---------------------------------------------------------------------------


def extract_daily_values(
    records: list[dict],
    biomarkers: dict[str, dict],
) -> dict[str, dict[date, list[float]]]:
    """Extract per-biomarker, per-day raw values from mock data records.

    Only extracts the biomarkers specified in the indicator's config.
    """
    result: dict[str, dict[date, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        ts: datetime = record["timestamp"]
        day = ts.date()
        value_dict: dict[str, float] = record["value"]
        modality: str = record["biomarker_type"]

        for bio_name, cfg in biomarkers.items():
            if cfg["type"] == modality and bio_name in value_dict:
                result[bio_name][day].append(value_dict[bio_name])

    return dict(result)


def compute_daily_means(
    daily_values: dict[str, dict[date, list[float]]],
) -> dict[str, dict[date, float]]:
    """Aggregate raw values to daily means per biomarker."""
    means: dict[str, dict[date, float]] = {}
    for bio, day_map in daily_values.items():
        means[bio] = {day: float(np.mean(vals)) for day, vals in day_map.items()}
    return means


# ---------------------------------------------------------------------------
# Membership computation (mirrors BiomarkerProcessor logic without DB)
# ---------------------------------------------------------------------------


def compute_baseline(values: list[float]) -> tuple[float, float]:
    """Compute baseline mean and std from all data points.

    Applies the same minimum std floor as BiomarkerProcessor.
    """
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < MIN_STD:
        std = MIN_STD
    return mean, std


def compute_membership(value: float, mean: float, std: float) -> float:
    """Compute membership value from a raw value using linear z-score mapping.

    Replicates the default membership function from BiomarkerProcessor:
        z = (value - mean) / std
        mu = clamp((z - Z_LOWER) / Z_RANGE, 0, 1)
    """
    z = (value - mean) / std
    mu = (z - Z_LOWER) / Z_RANGE
    return max(0.0, min(1.0, mu))


def compute_daily_memberships(
    daily_means: dict[str, dict[date, float]],
    daily_values: dict[str, dict[date, list[float]]],
    biomarker_names: list[str],
) -> tuple[dict[str, dict[date, float]], list[date]]:
    """Compute daily membership values for each biomarker.

    Baseline is computed once per biomarker from ALL raw values across all
    days (matching the BiomarkerProcessor.process_biomarkers_daily algorithm).

    Returns:
        Tuple of (biomarker -> {date -> membership}, sorted_dates)
    """
    memberships: dict[str, dict[date, float]] = {}

    for bio_name in biomarker_names:
        # Flatten all raw values for baseline computation
        all_values = []
        for day_vals in daily_values[bio_name].values():
            all_values.extend(day_vals)
        mean, std = compute_baseline(all_values)

        # Compute membership per day
        memberships[bio_name] = {}
        for day, daily_mean in daily_means[bio_name].items():
            memberships[bio_name][day] = compute_membership(daily_mean, mean, std)

    # Determine sorted list of dates (intersection across all biomarkers)
    date_sets = [set(m.keys()) for m in memberships.values()]
    common_dates = sorted(set.intersection(*date_sets))

    # Trim to exactly DAYS dates (take the most recent ones)
    if len(common_dates) > DAYS:
        common_dates = common_dates[-DAYS:]

    return memberships, common_dates


# ---------------------------------------------------------------------------
# Indicator score computation using FASL
# ---------------------------------------------------------------------------


def compute_indicator_score(
    memberships_for_day: dict[str, float],
    biomarkers_config: dict[str, dict],
) -> float:
    """Compute indicator score for a single day using apply_fasl_operator.

    Uses the existing window-level FASL function with neutral context
    weights (no context adjustment).
    """
    weights: dict[str, float] = {
        name: cfg["weight"]
        for name, cfg in biomarkers_config.items()
        if name in memberships_for_day
    }
    directions: dict[str, str] = {
        name: cfg["direction"]
        for name, cfg in biomarkers_config.items()
        if name in memberships_for_day
    }

    return apply_fasl_operator(
        memberships=memberships_for_day,
        weights=weights,
        directions=directions,
        context_weights=None,
        confidences=None,
    )


def compute_daily_scores(
    memberships: dict[str, dict[date, float]],
    dates: list[date],
    biomarkers_config: dict[str, dict],
    drop: list[str] | None = None,
    fault: dict[str, float] | None = None,
) -> list[float]:
    """Compute daily indicator scores, optionally with degradation.

    Args:
        memberships: Full membership values per biomarker per day.
        dates: Ordered list of dates to compute.
        biomarkers_config: Biomarker configuration for this indicator.
        drop: Biomarker names to remove (dropout simulation).
        fault: Mapping of biomarker name to constant override value
               (faulty sensor simulation).

    Returns:
        List of daily indicator scores aligned with `dates`.
    """
    drop = drop or []
    fault = fault or {}
    scores = []

    for day in dates:
        day_memberships: dict[str, float] = {}
        for bio_name in biomarkers_config:
            if bio_name in drop:
                continue
            mu = fault.get(bio_name, memberships[bio_name][day])
            day_memberships[bio_name] = mu
        scores.append(compute_indicator_score(day_memberships, biomarkers_config))

    return scores


# ---------------------------------------------------------------------------
# Data reliability computation
# ---------------------------------------------------------------------------


def compute_data_reliability(
    total_biomarkers: int,
    present_biomarkers: int,
) -> float:
    """Compute coverage-based data reliability score."""
    if total_biomarkers == 0:
        return 0.0
    return present_biomarkers / total_biomarkers


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------


def generate_heatmap(
    deviations: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_path: Path,
) -> None:
    """Generate and save the all-indicators degradation heatmap figure."""
    fig, ax = plt.subplots(figsize=(16, 5))

    sns.heatmap(
        deviations,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Mean absolute deviation from baseline"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Indicator")
    ax.set_ylabel("Degradation condition")
    ax.set_title("Sensor Degradation Impact Across All Indicators")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")


# ---------------------------------------------------------------------------
# Short indicator labels for the heatmap x-axis
# ---------------------------------------------------------------------------


def short_label(indicator_name: str) -> str:
    """Convert indicator config key to a readable short label.

    Example: '1_depressed_mood' -> '1 Depressed mood'
    """
    parts = indicator_name.split("_", 1)
    number = parts[0]
    name = parts[1].replace("_", " ").capitalize()
    return f"{number} {name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the full all-indicators degradation heatmap pipeline."""
    print("=" * 70)
    print("Scenario 4: All-Indicators Degradation Heatmap")
    print(f"Seed: {SEED}, Days: {DAYS}, Scenario: {SCENARIO}")
    print("=" * 70)

    # Step 1: Load indicator configuration
    print("\n[1/6] Loading indicator configuration...")
    indicators = load_indicators_config(INDICATORS_CONFIG_PATH)
    indicator_names = list(indicators.keys())
    n_indicators = len(indicator_names)
    print(f"  Loaded {n_indicators} indicators")

    for name, biomarkers in indicators.items():
        speech = get_modality_biomarkers(biomarkers, "speech")
        network = get_modality_biomarkers(biomarkers, "network")
        print(f"  {name}: {len(speech)} speech, {len(network)} network")

    # Step 2: Generate mock data
    print("\n[2/6] Generating mock data...")
    records = generate_mock_data(seed=SEED, days=DAYS)
    print(f"  Generated {len(records)} biomarker records")

    # Step 3: Process each indicator
    n_conditions = len(CONDITION_LABELS)
    mean_deviations = np.zeros((n_conditions, n_indicators))
    reliability_scores = np.zeros((n_conditions, n_indicators))

    for ind_idx, indicator_name in enumerate(indicator_names):
        biomarkers_config = indicators[indicator_name]
        bio_names = list(biomarkers_config.keys())

        print(f"\n[3/6] Processing indicator: {indicator_name}")

        # Extract and aggregate daily values for this indicator
        daily_values = extract_daily_values(records, biomarkers_config)
        daily_means = compute_daily_means(daily_values)

        for bio_name in bio_names:
            n_days = len(daily_means.get(bio_name, {}))
            print(f"  {bio_name}: {n_days} days of data")

        # Compute memberships
        memberships, dates = compute_daily_memberships(
            daily_means, daily_values, bio_names
        )
        print(f"  {len(dates)} days with complete data")

        # Compute baseline scores
        baseline_scores = compute_daily_scores(
            memberships, dates, biomarkers_config
        )
        print(f"  Baseline scores:")
        for day, score in zip(dates, baseline_scores):
            print(f"    {day}: {score:.4f}")

        # Build degradation conditions for this indicator
        conditions = build_conditions_for_indicator(biomarkers_config)

        # Compute degraded scores and mean deviations
        for cond_idx, condition in enumerate(conditions):
            drop = condition["drop"]
            fault = condition["fault"]

            degraded_scores = compute_daily_scores(
                memberships, dates, biomarkers_config, drop=drop, fault=fault
            )

            # Compute data reliability
            present = len(biomarkers_config) - len(drop)
            reliability = compute_data_reliability(len(biomarkers_config), present)
            reliability_scores[cond_idx, ind_idx] = reliability

            # Compute per-day deviations and mean
            daily_devs = [
                abs(deg - base)
                for deg, base in zip(degraded_scores, baseline_scores)
            ]
            mean_dev = float(np.mean(daily_devs))
            mean_deviations[cond_idx, ind_idx] = mean_dev

            print(f"  Condition {cond_idx + 1} ({CONDITION_LABELS[cond_idx].replace(chr(10), ' ')}): "
                  f"mean_dev={mean_dev:.4f}, reliability={reliability:.2f}")
            if drop:
                print(f"    Dropped: {drop}")
            if fault:
                print(f"    Faulted: {fault}")

    # Step 4: Print summary tables
    print("\n" + "=" * 70)
    print("Summary: Mean Absolute Deviations")
    print("=" * 70)
    col_labels = [short_label(name) for name in indicator_names]
    header = f"{'Condition':<35} " + " ".join(f"{lbl:>20}" for lbl in col_labels)
    print(header)
    print("-" * len(header))
    for cond_idx in range(n_conditions):
        label = CONDITION_LABELS[cond_idx].replace("\n", " ")
        vals = " ".join(f"{mean_deviations[cond_idx, j]:>20.4f}" for j in range(n_indicators))
        print(f"{label:<35} {vals}")

    print("\n" + "=" * 70)
    print("Summary: Data Reliability Scores")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for cond_idx in range(n_conditions):
        label = CONDITION_LABELS[cond_idx].replace("\n", " ")
        vals = " ".join(f"{reliability_scores[cond_idx, j]:>20.2f}" for j in range(n_indicators))
        print(f"{label:<35} {vals}")

    # Step 5: Generate heatmap
    print("\n[6/6] Generating heatmap...")
    generate_heatmap(mean_deviations, CONDITION_LABELS, col_labels, OUTPUT_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
