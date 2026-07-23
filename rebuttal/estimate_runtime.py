from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rebuttal.common import atomic_json, load_jsonl, project_root


def estimate(
    inputs: list[Path],
    problems: int,
    attempts: int,
) -> list[dict[str, Any]]:
    by_method: dict[str, list[float]] = defaultdict(list)
    for path in inputs:
        for row in load_jsonl(path):
            by_method[str(row["method"])].append(
                float(row.get("allocated_elapsed_s", 0.0))
            )

    estimates = []
    for method, durations in sorted(by_method.items()):
        positive = [duration for duration in durations if duration > 0]
        mean = sum(positive) / len(positive) if positive else 0.0
        total_seconds = mean * problems * attempts
        estimates.append(
            {
                "method": method,
                "pilot_rows": len(durations),
                "mean_seconds_per_problem_attempt": mean,
                "projected_problem_attempts": problems * attempts,
                "projected_hours": total_seconds / 3600,
                "warning": (
                    "Two-problem smoke estimates are only a launch sanity check; "
                    "use an N=1 full-test pilot for scheduling."
                ),
            }
        )
    return estimates


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--problems", type=int, default=244)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/rebuttal/runtime_estimate.json",
    )
    args = parser.parse_args()
    estimates = estimate(args.inputs, args.problems, args.attempts)
    if not estimates:
        raise SystemExit("No runtime rows found.")
    atomic_json(args.output, estimates)
    print(json.dumps(estimates, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
