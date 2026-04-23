"""
Verify biomarker determinism per row (window-indicator pair) for Claude and ChatGPT.

Checks how deterministic Claude and ChatGPT are when predicting biomarker likelihoods
across 5 repeated runs.

1. Parses markdown comparison tables (2.1_claude_biomarkers_comparison_runs1-5.md and
   2.1_chatgpt_biomarkers_comparison_runs1-5.md) that contain likelihood percentages
   from 5 runs for each window-indicator pair.

2. Computes two agreement metrics per row:
   - Exact unanimous: all 5 runs produced the identical likelihood value.
   - Presence unanimous: all 5 runs agree on whether the biomarker is present (>=60%)
     or absent (<60%), even if the exact values differ.

3. Prints a summary for each LLM showing how many rows achieved exact and presence
   agreement out of the total.

The purpose is to evaluate reproducibility: if you ask the same LLM the same question
5 times, how often does it give the same answer? Exact agreement measures strict
consistency, while presence agreement measures whether the clinical conclusion
(present vs. absent) stays stable.
"""

import os

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
THRESHOLD = 60.0

FILES = {
    "Claude": os.path.join(OUTPUTS_DIR, "2.1_claude_biomarkers_comparison_runs1-5.md"),
    "ChatGPT": os.path.join(OUTPUTS_DIR, "2.1_chatgpt_biomarkers_comparison_runs1-5.md"),
}


def parse_rows(filepath):
    """Parse comparison file and return list of (window, indicator, [likelihoods])."""
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:  # skip header and separator
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        window = parts[1]
        indicator = parts[2]
        try:
            likelihoods = [float(v.replace("%", "").strip()) for v in parts[3:8]]
        except (ValueError, IndexError):
            continue
        rows.append((window, indicator, likelihoods))
    return rows


def compute_stats(rows):
    """Compute exact and presence agreement stats per row."""
    total = len(rows)
    exact_unanimous = 0
    presence_unanimous = 0

    for window, indicator, likelihoods in rows:
        # Exact: all 5 likelihood values are identical
        if len(set(likelihoods)) == 1:
            exact_unanimous += 1

        # Presence: all 5 runs agree on present (>=60%) or absent (<60%)
        presences = [l >= THRESHOLD for l in likelihoods]
        if len(set(presences)) == 1:
            presence_unanimous += 1

    return total, exact_unanimous, presence_unanimous


def main():
    print(f"Threshold for presence: {THRESHOLD}%")
    print(f"Each row (window-indicator pair) is evaluated independently.\n")

    for name, filepath in FILES.items():
        rows = parse_rows(filepath)
        total, exact, presence = compute_stats(rows)

        exact_pct = exact / total * 100 if total else 0
        presence_pct = presence / total * 100 if total else 0

        print(f"=== {name} ===")
        print(f"Total rows: {total}")
        print(f"Exact likelihood unanimous: {exact}/{total} ({exact_pct:.1f}%)")
        print(f"Presence unanimous:         {presence}/{total} ({presence_pct:.1f}%)")
        print()


if __name__ == "__main__":
    main()
