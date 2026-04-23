# Baseline Files

This directory contains baseline definition files for biomarker normalization.

## Purpose

Baseline files define reference values (mean and standard deviation) for each biomarker. These values establish what "normal" looks like for a user, enabling the analysis pipeline to detect deviations that may indicate symptomatic behavior.

## How Baselines Are Used in Analysis

### Step 1: Z-Score Normalization

For each biomarker reading, the pipeline computes a z-score:

```
z-score = (observed_value - baseline_mean) / baseline_std
```

The z-score represents how many standard deviations the reading is from the baseline mean:
- `z = 0` → Reading equals baseline mean (neutral)
- `z = +2` → Reading is 2 std above mean (elevated)
- `z = -2` → Reading is 2 std below mean (depressed)

### Step 2: Membership Conversion (Sigmoid)

Z-scores are mapped to membership values in [0, 1] using a sigmoid function:

```
membership = 1 / (1 + exp(-z_score))
```

| Z-Score | Membership | Interpretation |
|---------|------------|----------------|
| -3 | ~0.05 | Significantly below baseline |
| -2 | ~0.12 | Moderately below baseline |
| -1 | ~0.27 | Slightly below baseline |
| 0 | 0.50 | At baseline (neutral) |
| +1 | ~0.73 | Slightly above baseline |
| +2 | ~0.88 | Moderately above baseline |
| +3 | ~0.95 | Significantly above baseline |

### Step 3: Indicator Scoring

Membership values feed into indicator scoring via the FASL (Fuzzy Aggregation with Soft Logic) formula. Higher membership values for relevant biomarkers increase the indicator score, which represents the likelihood of that symptom being present.

### Step 4: Episode Detection

Daily indicator scores are evaluated against DSM-5 criteria to determine if an episode is likely. The baseline quality directly impacts detection accuracy.

## File Format

Baseline files are JSON with the following structure:

```json
{
  "metadata": {
    "name": "Descriptive Name",
    "description": "Description of baseline source",
    "created": "YYYY-MM-DD",
    "version": "1.0"
  },
  "baselines": {
    "biomarker_name": {
      "mean": 0.5,
      "std": 0.15
    }
  }
}
```

### Required Fields

Each biomarker entry requires:
- `mean` (number): The baseline mean value
- `std` (number): Standard deviation (must be > 0)

### Optional Fields

- `metadata`: Descriptive information about the baseline (optional but recommended)

### Metadata Fields

The `metadata` object helps track where a baseline came from and is displayed in the UI:

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Short identifier shown in UI and logs | "User123 Calibration" |
| `description` | Explains the baseline source/purpose | "From 30-day healthy period" |
| `created` | Date the baseline was created | "2026-01-29" |
| `version` | Version for tracking changes | "1.0" |

**Why use metadata?**
- **Reproducibility**: Analysis results include the baseline name, so you can trace which reference values were used
- **Documentation**: Description explains the baseline's origin (calibration period, study cohort, etc.)
- **Version control**: Track updates to baseline values over time

### Field Meanings

| Field | Description | Impact on Analysis |
|-------|-------------|-------------------|
| `mean` | Expected "normal" value for the biomarker | Center point for z-score calculation |
| `std` | Expected variation around the mean | Sensitivity of deviation detection |

**Choosing `std` values:**
- Smaller `std` → More sensitive to small deviations
- Larger `std` → More tolerant of variation, only flags large deviations
- If `std` is too small, normal variation triggers false positives
- If `std` is too large, real symptoms may be missed

## Adding Custom Baselines

1. Create a new `.json` file in this directory
2. Follow the schema above
3. Include all biomarkers used in your analysis configuration
4. The file will appear in the "Select predefined baseline" dropdown on the Analysis page

### Recommended Approach

For best results, create baselines from a known healthy/stable period:

1. **Calibration Period**: Have the user collect data for 2-4 weeks during a non-symptomatic period
2. **Compute Statistics**: Calculate mean and std for each biomarker from this period
3. **Create Baseline File**: Export these values to a JSON file
4. **Use for Analysis**: Select this file when analyzing potentially symptomatic periods

## Included Files

- `population_default.json` - Default reference values derived from population averages for all configured biomarkers

## Validation

Files are validated on load:
- `mean` must be a number
- `std` must be a positive number (> 0)
- If `std` is 0 or negative, validation fails with an error message

**Important:** The baseline file must include all biomarkers used in the analysis. If a required biomarker is missing from the selected baseline, the analysis will fail with an error. Ensure your baseline file is complete before running analysis.

## Example: Creating a Custom Baseline

```json
{
  "metadata": {
    "name": "User123 Healthy Calibration",
    "description": "Baseline from 30-day calibration period (Jan 2026)",
    "created": "2026-02-01",
    "version": "1.0"
  },
  "baselines": {
    "sleep_duration": {
      "mean": 7.8,
      "std": 0.6
    },
    "activity_level": {
      "mean": 0.65,
      "std": 0.12
    },
    "speech_activity": {
      "mean": 0.55,
      "std": 0.18
    }
  }
}
```

This baseline indicates:
- User typically sleeps 7.8 hours (±0.6 hours is normal variation)
- User's activity level is typically 0.65 (on 0-1 scale)
- User's speech activity is typically 0.55

A reading of 5 hours sleep would yield: `z = (5 - 7.8) / 0.6 = -4.67`, indicating a very significant deviation.

The metadata will appear in:
- The "Show baseline details" panel on the Analysis page
- Analysis logs: `"Triggering analysis... baseline=User123 Healthy Calibration"`
- The config snapshot stored with each analysis run for reproducibility
