from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import sys
import time
from pathlib import Path

from rebuttal.common import (
    append_jsonl,
    atomic_json,
    completed_attempts,
    indexed_problem_shard,
    project_root,
    runtime_manifest,
    selected_problems,
    sha256,
    sharded_output_path,
    validate_attempt_bound,
    validate_resume_manifest,
    validate_shard,
)


RECOVERED_METHOD = "recovered_hlp_astar_stepwise"
NO_RETRIEVAL_METHOD = "recovered_hlp_no_retrieval"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the recovered HLP code with auditable N<=32 attempts."
    )
    root = project_root()
    parser.add_argument(
        "--dataset", type=Path, default=root / "rebuttal/datasets/minif2f.jsonl"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=244,
        help="Required full-split size when --limit=0; use 186 for ProofNet-test.",
    )
    parser.add_argument("--model", default=str(root / "models/DeepSeek-Prover-V1.5-RL"))
    parser.add_argument("--checkpoint", default=str(root / "data/hgcn_final.pth"))
    parser.add_argument(
        "--mode",
        choices=("recovered_hlp", "no_retrieval"),
        default="recovered_hlp",
    )
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-expansions", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--output", type=Path, default=root / "results/rebuttal/hlp/results.jsonl"
    )
    parser.add_argument("--save-failure-traces", action="store_true")
    args = parser.parse_args()
    validate_attempt_bound(args.max_attempts)
    validate_shard(args.num_shards, args.shard_index)

    method = (
        RECOVERED_METHOD
        if args.mode == "recovered_hlp"
        else NO_RETRIEVAL_METHOD
    )
    model_path = Path(args.model).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = sharded_output_path(
        args.output.expanduser().resolve(), args.num_shards, args.shard_index
    )
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    all_problems = selected_problems(dataset_path, args.split, args.limit)
    if not all_problems:
        raise SystemExit("No problems selected.")
    if args.limit == 0 and len(all_problems) != args.expected_count:
        raise SystemExit(
            f"Refusing full run: expected {args.expected_count} {args.split} "
            f"problems, got {len(all_problems)}"
        )
    problems = indexed_problem_shard(
        all_problems, args.num_shards, args.shard_index
    )
    if not problems:
        raise SystemExit(
            f"Shard {args.shard_index}/{args.num_shards} contains no problems."
        )

    os.environ["HLP_TEMPERATURE"] = str(args.temperature)
    os.environ["HLP_TOP_P"] = str(args.top_p)
    os.environ["HLP_SEED"] = str(args.seed)
    os.environ["HLP_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(root))

    import torch
    from src.system2.lean_interaction import LeanEnv
    from src.system2.lie_search import RiemannSearchAgent

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = output_path.parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    done = completed_attempts(output_path, method)
    metadata = {
        **runtime_manifest(),
        "method": method,
        "paper_claim_compatible": False,
        "compatibility_warning": (
            "Recovered code uses A* and distance top-k retrieval, does not "
            "load a trained Lie policy checkpoint, and is not Algorithm 1."
        ),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "graph_embeddings_sha256": sha256(root / "data/node_embeddings.pt"),
        "model_config_sha256": (
            sha256(model_path / "config.json")
            if (model_path / "config.json").is_file()
            else None
        ),
        "split": args.split,
        "problem_count": len(problems),
        "total_problem_count": len(all_problems),
        "shard_problem_count": len(problems),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "problem_indices": [problem_index for problem_index, _ in problems],
        "problem_names": [problem["name"] for _, problem in problems],
        "expected_count": args.expected_count,
        "max_attempts": args.max_attempts,
        "max_steps": args.max_steps,
        "max_expansions": args.max_expansions,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    validate_resume_manifest(
        output_path,
        metadata,
        (
            "method",
            "dataset_sha256",
            "checkpoint_sha256",
            "graph_embeddings_sha256",
            "model_config_sha256",
            "split",
            "total_problem_count",
            "num_shards",
            "shard_index",
            "problem_indices",
            "problem_names",
            "max_steps",
            "max_expansions",
            "temperature",
            "top_p",
            "max_new_tokens",
            "seed",
        ),
    )
    atomic_json(output_path.parent / "manifest.json", metadata)

    agent = RiemannSearchAgent(str(checkpoint), str(model_path), device="cuda")
    if args.mode == "no_retrieval":
        agent.graph_emb = None
        agent.retrieval_mode = "none"
        agent.idx_to_name = {}

    rounds_path = output_path.parent / "round_metrics.jsonl"
    for attempt in range(1, args.max_attempts + 1):
        pending = [
            (problem_index, problem)
            for problem_index, problem in problems
            if (problem["name"], attempt) not in done
        ]
        if not pending:
            print(f"Attempt {attempt}: already complete")
            continue
        round_started = time.perf_counter()
        round_rows = []
        for problem_index, problem in pending:
            attempt_seed = args.seed + attempt * 1_000_003 + problem_index
            agent.state_visits.clear()
            agent.llm.set_sampling(
                seed=attempt_seed,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            agent.llm.reset_metrics()
            LeanEnv.reset_global_metrics()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                result = agent.search(
                    problem["formal_statement"],
                    max_steps=args.max_steps,
                    max_expansions=args.max_expansions,
                    initial_goal_text=problem.get("goal"),
                )
                status = str(result.get("status", "Unknown"))
                error = result.get("error")
            except Exception as exc:
                result = {"status": "ScriptCrash", "proof": [], "trace": []}
                status = "ScriptCrash"
                error = repr(exc)
            elapsed = time.perf_counter() - started
            metrics = agent.llm.metrics_snapshot()
            lean_metrics = LeanEnv.global_metrics()
            peak_vram = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )
            proof = result.get("proof", [])
            trace = result.get("trace", [])
            row = {
                "schema_version": 1,
                "method": method,
                "problem": problem["name"],
                "problem_index": problem_index,
                "split": problem["split"],
                "attempt": attempt,
                "attempt_seed": attempt_seed,
                "success": status == "Success",
                "status": status,
                "proof": proof,
                "proof_steps": len(proof),
                "expanded_nodes": max(
                    [int(step.get("step", 0)) for step in trace if isinstance(step, dict)]
                    or [0]
                ),
                "allocated_elapsed_s": elapsed,
                "peak_vram_bytes": peak_vram,
                "error": error,
                **metrics,
                **lean_metrics,
            }
            round_rows.append(row)
            append_jsonl(output_path, [row])

            if row["success"] or args.save_failure_traces:
                trace_path = trace_dir / f"{problem['name']}__a{attempt:02d}.json.gz"
                with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
                    json.dump(
                        {"problem": problem, "result": result, "metrics": row},
                        handle,
                        ensure_ascii=False,
                    )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        round_seconds = time.perf_counter() - round_started
        append_jsonl(
            rounds_path,
            [
                {
                    "method": method,
                    "attempt": attempt,
                    "pending_problems": len(pending),
                    "round_seconds": round_seconds,
                    "peak_vram_bytes": max(
                        (row["peak_vram_bytes"] for row in round_rows), default=0
                    ),
                }
            ],
        )
        print(
            f"Attempt {attempt:02d}/{args.max_attempts}: "
            f"{sum(row['success'] for row in round_rows)}/{len(round_rows)} successful, "
            f"{round_seconds:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
