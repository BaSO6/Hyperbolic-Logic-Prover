from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rebuttal.common import atomic_json, atomic_jsonl, load_jsonl


CONSISTENT_MANIFEST_FIELDS = (
    "method",
    "dataset_sha256",
    "split",
    "total_problem_count",
    "expected_count",
    "max_attempts",
    "seed",
    "temperature",
    "top_p",
    "max_tokens",
    "max_new_tokens",
    "max_steps",
    "max_expansions",
    "deepseek_commit",
    "mathlib_commit",
    "model_config_sha256",
    "checkpoint_sha256",
    "graph_embeddings_sha256",
)


def _comparable_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        field: manifest.get(field)
        for field in CONSISTENT_MANIFEST_FIELDS
        if field in manifest
    }


def merge_shards(
    input_dir: Path,
    output: Path,
    num_shards: int,
    expected_count: int | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    if num_shards < 2:
        raise ValueError("Shard merging requires --num-shards >= 2")

    manifests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    seen_problem_indices: dict[str, int] = {}
    reference: dict[str, Any] | None = None

    for shard_index in range(num_shards):
        shard_dir = input_dir / f"shard-{shard_index:02d}-of-{num_shards:02d}"
        manifest_path = shard_dir / "manifest.json"
        results_path = shard_dir / "results.jsonl"
        if not manifest_path.is_file() or not results_path.is_file():
            raise ValueError(
                f"Missing completed shard {shard_index}: expected "
                f"{manifest_path} and {results_path}"
            )

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("num_shards", -1)) != num_shards:
            raise ValueError(f"Wrong num_shards in {manifest_path}")
        if int(manifest.get("shard_index", -1)) != shard_index:
            raise ValueError(f"Wrong shard_index in {manifest_path}")

        comparable = _comparable_manifest(manifest)
        if reference is None:
            reference = comparable
        elif comparable != reference:
            differences = sorted(
                field
                for field in set(reference) | set(comparable)
                if reference.get(field) != comparable.get(field)
            )
            raise ValueError(
                f"Inconsistent manifest for shard {shard_index}: "
                + ", ".join(differences)
            )

        if len(manifest["problem_indices"]) != len(manifest["problem_names"]):
            raise ValueError(f"Problem index/name length mismatch in {manifest_path}")
        declared_indices = {int(value) for value in manifest["problem_indices"]}
        declared_problem_map = {
            int(index): str(name)
            for index, name in zip(
                manifest["problem_indices"], manifest["problem_names"]
            )
        }
        if len(declared_indices) != int(manifest["shard_problem_count"]):
            raise ValueError(f"Duplicate declared problem index in {manifest_path}")
        for problem_index in declared_indices:
            if problem_index % num_shards != shard_index:
                raise ValueError(
                    f"Problem index {problem_index} belongs to another shard"
                )

        shard_rows = load_jsonl(results_path)
        for row in shard_rows:
            problem_index = int(row["problem_index"])
            problem = str(row["problem"])
            attempt = int(row["attempt"])
            method = str(row["method"])
            if problem_index not in declared_indices:
                raise ValueError(
                    f"Undeclared problem index {problem_index} in {results_path}"
                )
            if problem_index % num_shards != shard_index:
                raise ValueError(
                    f"Mis-sharded problem index {problem_index} in {results_path}"
                )
            if method != manifest["method"]:
                raise ValueError(f"Wrong method in {results_path}: {method}")
            if str(row.get("split")) != str(manifest["split"]):
                raise ValueError(f"Wrong split in {results_path}")
            if declared_problem_map[problem_index] != problem:
                raise ValueError(
                    f"Problem name/index mismatch in {results_path}: "
                    f"{problem_index} is {problem}"
                )
            if attempt < 1 or attempt > int(manifest["max_attempts"]):
                raise ValueError(f"Invalid attempt {attempt} in {results_path}")
            expected_seed = (
                int(manifest["seed"]) + attempt * 1_000_003 + problem_index
            )
            if int(row.get("attempt_seed", -1)) != expected_seed:
                raise ValueError(
                    f"Wrong attempt seed for {problem}, attempt {attempt}"
                )
            previous_index = seen_problem_indices.setdefault(problem, problem_index)
            if previous_index != problem_index:
                raise ValueError(f"Problem {problem} has inconsistent global indices")
            key = (method, problem, attempt)
            if key in seen:
                raise ValueError(f"Duplicate result row: {key}")
            seen.add(key)
            rows.append(row)
        manifests.append(manifest)

    assert reference is not None
    total_problem_count = int(reference["total_problem_count"])
    merged_max_attempts = int(reference["max_attempts"])
    if expected_count is not None and total_problem_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} problems, manifests declare "
            f"{total_problem_count}"
        )
    if max_attempts is not None and merged_max_attempts != max_attempts:
        raise ValueError(
            f"Expected {max_attempts} attempts, manifests declare "
            f"{merged_max_attempts}"
        )

    declared_all = {
        int(index)
        for manifest in manifests
        for index in manifest["problem_indices"]
    }
    expected_indices = set(range(total_problem_count))
    if declared_all != expected_indices:
        missing = sorted(expected_indices - declared_all)
        extra = sorted(declared_all - expected_indices)
        raise ValueError(
            f"Shard manifests do not exactly cover the dataset; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    expected_rows = total_problem_count * merged_max_attempts
    if len(rows) != expected_rows:
        raise ValueError(
            f"Incomplete results: expected {expected_rows} rows, got {len(rows)}"
        )
    for problem, problem_index in seen_problem_indices.items():
        attempts = {
            attempt
            for method, row_problem, attempt in seen
            if row_problem == problem and method == reference["method"]
        }
        expected_attempts = set(range(1, merged_max_attempts + 1))
        if attempts != expected_attempts:
            raise ValueError(
                f"Incomplete attempts for {problem} (index {problem_index})"
            )
    if len(seen_problem_indices) != total_problem_count:
        raise ValueError(
            f"Expected {total_problem_count} distinct problems, got "
            f"{len(seen_problem_indices)}"
        )

    rows.sort(key=lambda row: (int(row["problem_index"]), int(row["attempt"])))
    atomic_jsonl(output, rows)
    merged_manifest = {
        **reference,
        "problem_count": total_problem_count,
        "shard_problem_count": total_problem_count,
        "num_shards": num_shards,
        "shard_index": None,
        "merged": True,
        "merged_shards": list(range(num_shards)),
        "result_rows": len(rows),
        "source_manifests": [
            str(
                input_dir
                / f"shard-{index:02d}-of-{num_shards:02d}"
                / "manifest.json"
            )
            for index in range(num_shards)
        ],
    }
    atomic_json(output.parent / "manifest.json", merged_manifest)
    return merged_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly validate and merge deterministic rebuttal shards."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--max-attempts", type=int)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else input_dir / "results.jsonl"
    )
    manifest = merge_shards(
        input_dir=input_dir,
        output=output,
        num_shards=args.num_shards,
        expected_count=args.expected_count,
        max_attempts=args.max_attempts,
    )
    print(
        f"Merged {manifest['result_rows']} rows from "
        f"{manifest['num_shards']} shards into {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
