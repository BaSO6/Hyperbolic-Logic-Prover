import json
import tempfile
import unittest
from pathlib import Path

from rebuttal.aggregate import exact_mcnemar_p, paired_comparisons, summarize_method
from rebuttal.common import (
    atomic_json,
    indexed_problem_shard,
    selected_problems,
    sharded_output_path,
    validate_attempt_bound,
    validate_resume_manifest,
    validate_shard,
)
from rebuttal.estimate_runtime import estimate
from rebuttal.merge_shards import merge_shards


class RebuttalCommonTests(unittest.TestCase):
    def test_selected_problems_is_sorted_and_split_filtered(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "data.jsonl"
            rows = [
                {"name": "b", "split": "test"},
                {"name": "x", "split": "valid"},
                {"name": "a", "split": "test"},
            ]
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                [row["name"] for row in selected_problems(dataset, "test")],
                ["a", "b"],
            )

    def test_summary_uses_cumulative_attempt_success(self):
        rows = []
        for problem, successes in {"a": {2}, "b": set()}.items():
            for attempt in range(1, 33):
                rows.append(
                    {
                        "method": "m",
                        "problem": problem,
                        "attempt": attempt,
                        "success": attempt in successes,
                        "allocated_elapsed_s": 1.0,
                        "llm_forward_calls": 1,
                    }
                )
        summary = {row["k"]: row for row in summarize_method(rows)}
        self.assertEqual(summary[1]["solved"], 0)
        self.assertEqual(summary[2]["solved"], 1)
        self.assertEqual(summary[32]["solved"], 1)
        self.assertEqual(summary[8]["avg_wall_seconds_per_problem"], 8.0)

    def test_attempt_bound(self):
        validate_attempt_bound(1)
        validate_attempt_bound(32)
        for invalid in (0, 33):
            with self.assertRaises(ValueError):
                validate_attempt_bound(invalid)

    def test_shards_are_disjoint_complete_and_keep_global_indices(self):
        problems = [{"name": f"p{index}"} for index in range(9)]
        shards = [
            indexed_problem_shard(problems, 4, shard_index)
            for shard_index in range(4)
        ]
        flattened = [pair for shard in shards for pair in shard]
        self.assertEqual(sorted(index for index, _ in flattened), list(range(9)))
        self.assertEqual([index for index, _ in shards[2]], [2, 6])
        self.assertEqual(
            sharded_output_path(Path("/tmp/run/results.jsonl"), 4, 2),
            Path("/tmp/run/shard-02-of-04/results.jsonl"),
        )
        self.assertEqual(
            sharded_output_path(Path("/tmp/run/results.jsonl"), 1, 0),
            Path("/tmp/run/results.jsonl"),
        )
        with self.assertRaises(ValueError):
            validate_shard(4, 4)

    def _write_fake_shards(
        self,
        root: Path,
        *,
        duplicate: bool = False,
        inconsistent_seed: bool = False,
    ) -> None:
        for shard_index in range(2):
            shard_dir = root / f"shard-{shard_index:02d}-of-02"
            indices = list(range(shard_index, 4, 2))
            names = [f"p{index}" for index in indices]
            atomic_json(
                shard_dir / "manifest.json",
                {
                    "method": "m",
                    "dataset_sha256": "abc",
                    "split": "test",
                    "problem_count": len(indices),
                    "total_problem_count": 4,
                    "shard_problem_count": len(indices),
                    "num_shards": 2,
                    "shard_index": shard_index,
                    "problem_indices": indices,
                    "problem_names": names,
                    "expected_count": 4,
                    "max_attempts": 2,
                    "seed": 8 if inconsistent_seed and shard_index else 7,
                },
            )
            rows = [
                {
                    "method": "m",
                    "problem": name,
                    "problem_index": index,
                    "attempt": attempt,
                    "attempt_seed": 7 + attempt * 1_000_003 + index,
                    "split": "test",
                    "success": False,
                }
                for index, name in zip(indices, names)
                for attempt in (1, 2)
            ]
            if duplicate and shard_index == 0:
                rows.append(rows[0])
            (shard_dir / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

    def test_merge_shards_requires_exact_complete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fake_shards(root)
            output = root / "results.jsonl"
            manifest = merge_shards(root, output, 2, expected_count=4, max_attempts=2)
            self.assertEqual(manifest["result_rows"], 8)
            merged = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(row["problem_index"], row["attempt"]) for row in merged],
                [
                    (0, 1),
                    (0, 2),
                    (1, 1),
                    (1, 2),
                    (2, 1),
                    (2, 2),
                    (3, 1),
                    (3, 2),
                ],
            )

    def test_merge_shards_rejects_duplicate_or_config_mismatch(self):
        for failure_mode in ("duplicate", "inconsistent"):
            with self.subTest(failure_mode=failure_mode):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    self._write_fake_shards(
                        root,
                        duplicate=failure_mode == "duplicate",
                        inconsistent_seed=failure_mode == "inconsistent",
                    )
                    with self.assertRaises(ValueError):
                        merge_shards(root, root / "results.jsonl", 2)

    def test_resume_manifest_rejects_changed_seed_but_allows_more_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            output.write_text("{}\n", encoding="utf-8")
            atomic_json(
                output.parent / "manifest.json",
                {"method": "m", "seed": 7, "max_attempts": 1},
            )
            validate_resume_manifest(
                output,
                {"method": "m", "seed": 7, "max_attempts": 32},
                ("method", "seed"),
            )
            with self.assertRaises(ValueError):
                validate_resume_manifest(
                    output,
                    {"method": "m", "seed": 8, "max_attempts": 32},
                    ("method", "seed"),
                )

    def test_runtime_estimate_scales_problem_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "method": "m",
                                "allocated_elapsed_s": 2.0,
                            }
                        ),
                        json.dumps(
                            {
                                "method": "m",
                                "allocated_elapsed_s": 4.0,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            row = estimate([path], problems=10, attempts=2)[0]
            self.assertEqual(row["mean_seconds_per_problem_attempt"], 3.0)
            self.assertAlmostEqual(row["projected_hours"], 60 / 3600)

    def test_paired_comparison_counts_discordant_solutions(self):
        methods = {
            "a": [
                {"method": "a", "problem": "x", "attempt": 1, "success": True},
                {"method": "a", "problem": "y", "attempt": 1, "success": False},
            ],
            "b": [
                {"method": "b", "problem": "x", "attempt": 1, "success": False},
                {"method": "b", "problem": "y", "attempt": 1, "success": True},
            ],
        }
        row = paired_comparisons(methods)[0]
        self.assertEqual(row["a_only"], 1)
        self.assertEqual(row["b_only"], 1)
        self.assertEqual(row["mcnemar_exact_p"], 1.0)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
