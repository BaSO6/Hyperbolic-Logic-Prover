import json
import tempfile
import unittest
from pathlib import Path

from rebuttal.aggregate import exact_mcnemar_p, paired_comparisons, summarize_method
from rebuttal.common import selected_problems, validate_attempt_bound
from rebuttal.estimate_runtime import estimate


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
