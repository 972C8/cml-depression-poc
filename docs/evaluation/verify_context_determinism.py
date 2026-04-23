"""
Verify context determinism per window for Claude and ChatGPT.

Checks how deterministic Claude and ChatGPT are when predicting context labels
across 5 repeated runs.

1. Parses markdown comparison tables (2.1_claude_context_comparison_runs1-5.md and
   2.1_chatgpt_context_comparison_runs1-5.md) that contain context labels from 5 runs
   for each window.

2. Computes unanimous agreement: all 5 runs produced the identical context label
   for a given window.

3. Prints a summary for each LLM showing how many windows achieved unanimous
   agreement out of the total.

The purpose is to evaluate reproducibility: if you ask the same LLM the same question
5 times, how often does it assign the same context label?
"""

import os

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")

FILES = {
    "Claude": os.path.join(OUTPUTS_DIR, "2.1_claude_context_comparison_runs1-5.md"),
    "ChatGPT": os.path.join(OUTPUTS_DIR, "2.1_chatgpt_context_comparison_runs1-5.md"),
}


def parse_rows(filepath):
    """Parse comparison file and return list of (window, [contexts])."""
    rows = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines[2:]:  # skip header and separator
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue
        window = parts[1]
        contexts = [parts[i] for i in range(2, 7)]
        rows.append((window, contexts))
    return rows


def compute_stats(rows):
    """Compute unanimous agreement stats per window."""
    total = len(rows)
    unanimous = 0

    for window, contexts in rows:
        if len(set(contexts)) == 1:
            unanimous += 1

    return total, unanimous


def main():
    print("Each window is evaluated independently.\n")

    for name, filepath in FILES.items():
        rows = parse_rows(filepath)
        total, unanimous = compute_stats(rows)

        unanimous_pct = unanimous / total * 100 if total else 0
        disagreements = total - unanimous

        print(f"=== {name} ===")
        print(f"Total windows: {total}")
        print(f"Unanimous: {unanimous}/{total} ({unanimous_pct:.2f}%)")
        print(f"Disagreements: {disagreements}")
        print()


if __name__ == "__main__":
    main()
