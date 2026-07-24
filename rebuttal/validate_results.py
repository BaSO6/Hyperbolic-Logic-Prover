"""Strict completeness validation for one unsharded rebuttal result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rebuttal.common import atomic_json, load_jsonl


def validate_results(
    results_path: Path,
    expected_count: int,
    max_attempts: int,
) -> dict[str, Any]:
    manifest_path = results_path.parent / "manifest.json"
    if not manifest_path.is_file() or not results_path.is_file():
        raise ValueError(
            f"Expected both {manifest_path} and {results_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest["total_problem_count"]) != expected_count:
        raise ValueError("Manifest problem count does not match expectation")
    if int(manifest["max_attempts"]) != max_attempts:
        raise ValueError("Manifest max_attempts does not match expectation")
    if int(manifest.get("num_shards", 1)) != 1:
        raise ValueError("Use rebuttal.merge_shards for sharded results")
    declared_indices = [int(value) for value in manifest["problem_indices"]]
    declared_names = [str(value) for value in manifest["problem_names"]]
    if len(declared_indices) != expected_count or len(declared_names) != expected_count:
        raise ValueError("Manifest does not declare exactly the expected problems")
    declared_problem_map = dict(zip(declared_indices, declared_names))
    if len(declared_problem_map) != expected_count:
        raise ValueError("Manifest contains duplicate problem indices")

    rows = load_jsonl(results_path)
    expected_rows = expected_count * max_attempts
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, found {len(rows)}"
        )
    seen: set[tuple[str, int]] = set()
    indices: dict[str, int] = {}
    for row in rows:
        problem = str(row["problem"])
        problem_index = int(row["problem_index"])
        attempt = int(row["attempt"])
        if str(row["method"]) != str(manifest["method"]):
            raise ValueError(f"Wrong method for {problem}")
        if str(row["split"]) != str(manifest["split"]):
            raise ValueError(f"Wrong split for {problem}")
        if declared_problem_map.get(problem_index) != problem:
            raise ValueError(
                f"Problem name/index mismatch: {problem_index} is {problem}"
            )
        if attempt < 1 or attempt > max_attempts:
            raise ValueError(f"Invalid attempt for {problem}: {attempt}")
        expected_seed = int(manifest["seed"]) + attempt * 1_000_003 + problem_index
        if int(row["attempt_seed"]) != expected_seed:
            raise ValueError(f"Wrong attempt seed for {problem}: {attempt}")
        if (problem, attempt) in seen:
            raise ValueError(f"Duplicate result: {problem}, attempt {attempt}")
        seen.add((problem, attempt))
        previous = indices.setdefault(problem, problem_index)
        if previous != problem_index:
            raise ValueError(f"Inconsistent index for {problem}")
    if set(indices.values()) != set(range(expected_count)):
        raise ValueError("Problem indices do not exactly cover expected range")

    report = {
        "valid": True,
        "method": manifest["method"],
        "results": str(results_path.resolve()),
        "problem_count": expected_count,
        "max_attempts": max_attempts,
        "row_count": len(rows),
    }
    atomic_json(results_path.parent / "validation.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, required=True)
    args = parser.parse_args()
    report = validate_results(
        args.results.expanduser().resolve(),
        args.expected_count,
        args.max_attempts,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
