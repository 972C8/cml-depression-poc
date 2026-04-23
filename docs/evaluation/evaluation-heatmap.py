"""Scenario 4: Degradation Heatmap — Sensor Robustness Evaluation.

Generates a heatmap visualising how indicator scores for `1_depressed_mood`
change under systematic sensor degradation conditions, compared to a
full-data baseline.

Degradation conditions (heatmap rows):
    1. Drop `whispering`                                  — 1 speech biomarker removed
    2. Drop `reduced_social_interaction`                  — 1 network biomarker removed
    3. Drop `whispering` + `reduced_social_interaction`   — 1 speech + 1 network removed
    4. Drop speech modality                               — both speech biomarkers removed
    5. Drop network modality                              — both network biomarkers removed
    6. `whispering` stuck at 1.0                          — faulty sensor (all 4 present)

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
USER_ID = "eval_scenario4"
SCENARIO = "solitary_digital"
BIOMARKER_INTERVAL_MIN = 15  # minutes between samples
INDICATOR = "1_depressed_mood"

# Biomarkers for 1_depressed_mood
BIOMARKERS = {
    "whispering": {"modality": "speech", "weight": 0.25, "direction": "higher_is_worse"},
    "prolonged_pauses": {"modality": "speech", "weight": 0.25, "direction": "higher_is_worse"},
    "reduced_social_interaction": {"modality": "network", "weight": 0.25, "direction": "higher_is_worse"},
    "passive_media_binge": {"modality": "network", "weight": 0.25, "direction": "higher_is_worse"},
}

# Degradation conditions
DEGRADATION_CONDITIONS = [
    {
        "label": "Drop whispering",
        "drop": ["whispering"],
        "fault": {},
    },
    {
        "label": "Drop reduced_social_interaction",
        "drop": ["reduced_social_interaction"],
        "fault": {},
    },
    {
        "label": "Drop whispering +\nreduced_social_interaction",
        "drop": ["whispering", "reduced_social_interaction"],
        "fault": {},
    },
    {
        "label": "Drop speech modality",
        "drop": ["whispering", "prolonged_pauses"],
        "fault": {},
    },
    {
        "label": "Drop network modality",
        "drop": ["reduced_social_interaction", "passive_media_binge"],
        "fault": {},
    },
    {
        "label": "whispering stuck at 1.0",
        "drop": [],
        "fault": {"whispering": 1.0},
    },
]

# Biomarker processing defaults (matching src/core/config.py defaults)
Z_LOWER = -3.0
Z_UPPER = 3.0
Z_RANGE = Z_UPPER - Z_LOWER
MIN_STD = 0.1

# Output path
OUTPUT_PATH = (
    PROJECT_ROOT
    / "thesis"
    / "MA_Tibor_Haller"
    / "content"
    / "5_evaluation"
    / "scenario4_degradation_heatmap.png"
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_mock_data(
    seed: int, days: int
) -> list[dict]:
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
) -> dict[str, dict[date, list[float]]]:
    """Extract per-biomarker, per-day raw values from mock data records.

    Each record contains a `value` dict mapping biomarker names to floats
    and a `biomarker_type` field indicating the modality.  Only the four
    biomarkers relevant to `1_depressed_mood` are extracted.

    Returns:
        Nested dict: biomarker_name -> {date -> [raw_values]}
    """
    result: dict[str, dict[date, list[float]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        ts: datetime = record["timestamp"]
        day = ts.date()
        value_dict: dict[str, float] = record["value"]
        modality: str = record["biomarker_type"]

        for biomarker_name, cfg in BIOMARKERS.items():
            if cfg["modality"] == modality and biomarker_name in value_dict:
                result[biomarker_name][day].append(value_dict[biomarker_name])

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
) -> tuple[dict[str, dict[date, float]], list[date]]:
    """Compute daily membership values for each biomarker.

    Baseline is computed once per biomarker from ALL raw values across all
    days (matching the BiomarkerProcessor.process_biomarkers_daily algorithm).

    Returns:
        Tuple of (biomarker -> {date -> membership}, sorted_dates)
    """
    memberships: dict[str, dict[date, float]] = {}

    for bio_name in BIOMARKERS:
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

    # Trim to exactly DAYS dates (take the most recent ones) to avoid
    # off-by-one from timedelta spanning an extra calendar day boundary
    if len(common_dates) > DAYS:
        common_dates = common_dates[-DAYS:]

    return memberships, common_dates


# ---------------------------------------------------------------------------
# Indicator score computation using FASL
# ---------------------------------------------------------------------------


def compute_indicator_score(
    memberships_for_day: dict[str, float],
) -> float:
    """Compute indicator score for a single day using apply_fasl_operator.

    Uses the existing window-level FASL function with neutral context
    weights (no context adjustment).
    """
    weights: dict[str, float] = {
        name: cfg["weight"] for name, cfg in BIOMARKERS.items() if name in memberships_for_day
    }
    directions: dict[str, str] = {
        name: cfg["direction"] for name, cfg in BIOMARKERS.items() if name in memberships_for_day
    }

    return apply_fasl_operator(
        memberships=memberships_for_day,
        weights=weights,
        directions=directions,
        context_weights=None,  # neutral
        confidences=None,
    )


def compute_daily_scores(
    memberships: dict[str, dict[date, float]],
    dates: list[date],
    drop: list[str] | None = None,
    fault: dict[str, float] | None = None,
) -> list[float]:
    """Compute daily indicator scores, optionally with degradation.

    Args:
        memberships: Full membership values per biomarker per day.
        dates: Ordered list of dates to compute.
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
        for bio_name in BIOMARKERS:
            if bio_name in drop:
                continue  # biomarker dropped
            mu = fault.get(bio_name, memberships[bio_name][day])
            day_memberships[bio_name] = mu
        scores.append(compute_indicator_score(day_memberships))

    return scores


# ---------------------------------------------------------------------------
# Data reliability computation
# ---------------------------------------------------------------------------


def compute_data_reliability(
    total_biomarkers: int,
    present_biomarkers: int,
) -> float:
    """Compute coverage-based data reliability score.

    Simplified to coverage factor only (matching the spec's expected values).
    """
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
    """Generate and save the degradation heatmap figure."""
    fig, ax = plt.subplots(figsize=(14, 5))

    sns.heatmap(
        deviations,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Absolute deviation from baseline"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Degradation condition")
    ax.set_title(f"Sensor Degradation Impact on {INDICATOR}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Execute the full scenario 4 evaluation pipeline."""
    print("=" * 70)
    print("Scenario 4: Degradation Heatmap")
    print(f"Indicator: {INDICATOR}")
    print(f"Seed: {SEED}, Days: {DAYS}, Scenario: {SCENARIO}")
    print("=" * 70)

    # Step 1: Generate mock data
    print("\n[1/5] Generating mock data...")
    records = generate_mock_data(seed=SEED, days=DAYS)
    print(f"  Generated {len(records)} biomarker records")

    # Step 2: Extract and aggregate daily values
    print("[2/5] Extracting daily biomarker values...")
    daily_values = extract_daily_values(records)
    daily_means = compute_daily_means(daily_values)

    for bio_name in BIOMARKERS:
        n_days = len(daily_means.get(bio_name, {}))
        print(f"  {bio_name}: {n_days} days of data")

    # Step 3: Compute memberships
    print("[3/5] Computing biomarker memberships...")
    memberships, dates = compute_daily_memberships(daily_means, daily_values)
    print(f"  {len(dates)} days with complete data")

    # Step 4: Compute baseline and degraded scores
    print("[4/5] Computing indicator scores...")

    baseline_scores = compute_daily_scores(memberships, dates)
    print(f"\n  Baseline scores (full data):")
    for day, score in zip(dates, baseline_scores):
        print(f"    {day}: {score:.4f}")

    # Compute degraded scores and deviations
    n_conditions = len(DEGRADATION_CONDITIONS)
    deviations = np.zeros((n_conditions, len(dates)))
    row_labels = []

    for i, condition in enumerate(DEGRADATION_CONDITIONS):
        label = condition["label"]
        drop = condition["drop"]
        fault = condition["fault"]
        row_labels.append(label)

        degraded_scores = compute_daily_scores(memberships, dates, drop=drop, fault=fault)

        # Compute data reliability
        present = len(BIOMARKERS) - len(drop)
        reliability = compute_data_reliability(len(BIOMARKERS), present)

        print(f"\n  Condition {i + 1}: {label}")
        print(f"    Data reliability: {reliability:.2f}")
        print(f"    Degraded scores:")
        for j, (day, deg, base) in enumerate(zip(dates, degraded_scores, baseline_scores)):
            dev = abs(deg - base)
            deviations[i, j] = dev
            print(f"      {day}: {deg:.4f} (dev={dev:.4f})")

    # Step 5: Generate heatmap
    print("\n[5/5] Generating heatmap...")
    col_labels = [str(i + 1) for i in range(len(dates))]
    generate_heatmap(deviations, row_labels, col_labels, OUTPUT_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
