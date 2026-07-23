from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from collections import defaultdict
from pathlib import Path
from typing import Any

from rebuttal.common import CHECKPOINTS, atomic_json, load_jsonl, project_root


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denom
    return center - margin, center + margin


def summarize_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[str(row["problem"])].append(row)
    for problem_rows in by_problem.values():
        problem_rows.sort(key=lambda row: int(row["attempt"]))

    output = []
    for checkpoint in CHECKPOINTS:
        eligible = {
            problem: problem_rows
            for problem, problem_rows in by_problem.items()
            if max(int(row["attempt"]) for row in problem_rows) >= checkpoint
        }
        success_map = {
            problem: any(
                bool(row["success"]) and int(row["attempt"]) <= checkpoint
                for row in problem_rows
            )
            for problem, problem_rows in eligible.items()
        }
        total = len(success_map)
        successes = sum(success_map.values())
        low, high = wilson(successes, total)
        per_problem_times = [
            sum(
                float(row.get("allocated_elapsed_s", 0.0))
                for row in problem_rows
                if int(row["attempt"]) <= checkpoint
            )
            for problem_rows in eligible.values()
        ]
        llm_calls = [
            sum(
                int(row.get("llm_forward_calls", 0))
                for row in problem_rows
                if int(row["attempt"]) <= checkpoint
            )
            for problem_rows in eligible.values()
        ]
        output.append(
            {
                "method": rows[0]["method"],
                "k": checkpoint,
                "solved": successes,
                "total": total,
                "pass_at_k": successes / total if total else 0.0,
                "ci_low": low,
                "ci_high": high,
                "avg_wall_seconds_per_problem": (
                    sum(per_problem_times) / len(per_problem_times)
                    if per_problem_times
                    else 0.0
                ),
                "avg_llm_calls_per_problem": (
                    sum(llm_calls) / len(llm_calls) if llm_calls else 0.0
                ),
                "peak_vram_bytes": max(
                    (
                        int(
                            row.get(
                                "peak_vram_bytes",
                                row.get("peak_vram_bytes_round", 0),
                            )
                        )
                        for row in rows
                    ),
                    default=0,
                ),
            }
        )
    return output


def exact_mcnemar_p(method_a_only: int, method_b_only: int) -> float:
    """Two-sided exact McNemar/binomial p-value for paired binary outcomes."""
    discordant = method_a_only + method_b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(0, min(method_a_only, method_b_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def solved_by_problem(
    rows: list[dict[str, Any]],
    checkpoint: int,
) -> dict[str, bool]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["problem"])].append(row)
    return {
        problem: any(
            bool(row["success"]) and int(row["attempt"]) <= checkpoint
            for row in problem_rows
        )
        for problem, problem_rows in grouped.items()
        if max(int(row["attempt"]) for row in problem_rows) >= checkpoint
    }


def paired_comparisons(
    methods: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for method_a, method_b in combinations(sorted(methods), 2):
        for checkpoint in CHECKPOINTS:
            solved_a = solved_by_problem(methods[method_a], checkpoint)
            solved_b = solved_by_problem(methods[method_b], checkpoint)
            shared = sorted(set(solved_a) & set(solved_b))
            if not shared:
                continue
            a_only = sum(solved_a[name] and not solved_b[name] for name in shared)
            b_only = sum(solved_b[name] and not solved_a[name] for name in shared)
            both = sum(solved_a[name] and solved_b[name] for name in shared)
            neither = len(shared) - a_only - b_only - both
            output.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "k": checkpoint,
                    "paired_total": len(shared),
                    "both_solve": both,
                    "a_only": a_only,
                    "b_only": b_only,
                    "neither": neither,
                    "pass_delta_a_minus_b": (
                        (sum(solved_a[name] for name in shared)
                         - sum(solved_b[name] for name in shared))
                        / len(shared)
                    ),
                    "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
                }
            )
    return output


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            root / "results/rebuttal/native/results.jsonl",
            root / "results/rebuttal/hlp/results.jsonl",
        ],
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/rebuttal/summary"
    )
    args = parser.parse_args()

    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in args.inputs:
        if not path.exists():
            print(f"Skipping missing input: {path}")
            continue
        for row in load_jsonl(path):
            methods[str(row["method"])].append(row)
    if not methods:
        raise SystemExit("No result rows found.")

    summary = [
        record
        for method_rows in methods.values()
        for record in summarize_method(method_rows)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "frontier.json", summary)
    with (args.output_dir / "frontier.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    comparisons = paired_comparisons(methods)
    atomic_json(args.output_dir / "paired_comparisons.json", comparisons)
    if comparisons:
        with (args.output_dir / "paired_comparisons.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)

    try:
        import matplotlib.pyplot as plt

        for method in sorted(methods):
            points = [row for row in summary if row["method"] == method]
            plt.plot(
                [row["avg_wall_seconds_per_problem"] for row in points],
                [100 * row["pass_at_k"] for row in points],
                marker="o",
                label=method,
            )
            for row in points:
                plt.annotate(
                    f"k={row['k']}",
                    (
                        row["avg_wall_seconds_per_problem"],
                        100 * row["pass_at_k"],
                    ),
                    fontsize=8,
                )
        plt.xlabel("Average cumulative wall-clock seconds / problem")
        plt.ylabel("Pass@k (%)")
        plt.title("Matched-hardware accuracy–time frontier")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "frontier.png", dpi=220)
        plt.close()
    except Exception as exc:
        print(f"Plot skipped: {exc}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
