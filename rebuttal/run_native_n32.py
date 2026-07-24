from __future__ import annotations

import argparse
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


METHOD = "native_deepseek_v1.5_rl_wholeproof"


def chunks(values, size):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Official-prompt native DeepSeek whole-proof pass@N, N<=32."
    )
    root = project_root()
    parser.add_argument(
        "--deepseek-root",
        type=Path,
        default=root / "third_party/DeepSeek-Prover-V1.5",
    )
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
    parser.add_argument("--max-attempts", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-problems", type=int, default=32)
    parser.add_argument("--lean-workers", type=int, default=16)
    parser.add_argument("--lean-timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--output", type=Path, default=root / "results/rebuttal/native/results.jsonl"
    )
    args = parser.parse_args()
    validate_attempt_bound(args.max_attempts)
    validate_shard(args.num_shards, args.shard_index)

    deepseek_root = args.deepseek_root.resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = sharded_output_path(
        args.output.expanduser().resolve(), args.num_shards, args.shard_index
    )
    if not (deepseek_root / "prover/lean/verifier.py").exists():
        raise SystemExit(f"Official DeepSeek checkout not found: {deepseek_root}")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    sys.path.insert(0, str(deepseek_root))
    os.chdir(deepseek_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from vllm import LLM, SamplingParams
    from prover.lean.verifier import Lean4ServerScheduler
    from prover.utils import MODEL_FORMAT

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_attempts(output_path, METHOD)
    metadata = {
        **runtime_manifest(),
        "method": METHOD,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
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
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "deepseek_commit": os.popen("git rev-parse HEAD").read().strip(),
        "mathlib_commit": os.popen("git -C mathlib4 rev-parse HEAD").read().strip(),
        "model_config_sha256": (
            sha256(model_path / "config.json")
            if (model_path / "config.json").is_file()
            else None
        ),
    }
    validate_resume_manifest(
        output_path,
        metadata,
        (
            "method",
            "dataset_sha256",
            "split",
            "total_problem_count",
            "num_shards",
            "shard_index",
            "problem_indices",
            "problem_names",
            "seed",
            "temperature",
            "top_p",
            "max_tokens",
            "deepseek_commit",
            "mathlib_commit",
            "model_config_sha256",
        ),
    )
    atomic_json(output_path.parent / "manifest.json", metadata)

    model = LLM(
        model=str(model_path),
        max_num_batched_tokens=8192,
        seed=args.seed,
        trust_remote_code=True,
    )
    prompt_func = MODEL_FORMAT["cot"]["prompt"]
    output_func = MODEL_FORMAT["cot"]["output"]
    verifier = Lean4ServerScheduler(
        max_concurrent_requests=args.lean_workers,
        timeout=args.lean_timeout,
        memory_limit=10,
        name="verifier",
    )

    rounds_path = output_path.parent / "round_metrics.jsonl"
    try:
        for attempt in range(1, args.max_attempts + 1):
            pending = [
                (problem_index, problem)
                for problem_index, problem in problems
                if (problem["name"], attempt) not in done
            ]
            if not pending:
                print(f"Attempt {attempt}: already complete")
                continue

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            round_started = time.perf_counter()
            generated = []
            generation_seconds = 0.0

            for batch in chunks(pending, args.batch_problems):
                prompts = [prompt_func(problem) for _, problem in batch]
                attempt_seeds = [
                    args.seed + attempt * 1_000_003 + problem_index
                    for problem_index, _ in batch
                ]
                sampling = [
                    SamplingParams(
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        n=1,
                        seed=attempt_seed,
                    )
                    for attempt_seed in attempt_seeds
                ]
                started = time.perf_counter()
                outputs = model.generate(prompts, sampling, use_tqdm=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generation_seconds += time.perf_counter() - started
                for (
                    (problem_index, problem),
                    attempt_seed,
                    request_output,
                ) in zip(batch, attempt_seeds, outputs):
                    completion = request_output.outputs[0]
                    proof_code = output_func(completion.text)
                    generated.append(
                        {
                            "problem_index": problem_index,
                            "problem": problem,
                            "attempt_seed": attempt_seed,
                            "proof_code": proof_code,
                            "prompt_tokens": len(request_output.prompt_token_ids),
                            "completion_tokens": len(completion.token_ids),
                        }
                    )

            verification_started = time.perf_counter()
            request_ids = verifier.submit_all_request(
                [
                    "".join(
                        [
                            item["problem"].get("header", ""),
                            item["problem"]["formal_statement"],
                            item["proof_code"],
                            item["problem"].get("tailer", ""),
                        ]
                    )
                    for item in generated
                ]
            )
            verification_outputs = verifier.get_all_request_outputs(request_ids)
            verification_seconds = time.perf_counter() - verification_started
            round_seconds = time.perf_counter() - round_started
            allocated_seconds = round_seconds / max(1, len(generated))
            peak_vram = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            )

            rows = []
            for item, verification in zip(generated, verification_outputs):
                rows.append(
                    {
                        "schema_version": 1,
                        "method": METHOD,
                        "problem": item["problem"]["name"],
                        "problem_index": item["problem_index"],
                        "split": item["problem"]["split"],
                        "attempt": attempt,
                        "attempt_seed": item["attempt_seed"],
                        "success": bool(verification.get("complete", False)),
                        "proof_code": item["proof_code"],
                        "prompt_tokens": item["prompt_tokens"],
                        "completion_tokens": item["completion_tokens"],
                        "llm_forward_calls": 1,
                        "lean_calls": 1,
                        "allocated_elapsed_s": allocated_seconds,
                        "verification_elapsed_s": float(
                            verification.get("verify_time", 0.0)
                        ),
                        "peak_vram_bytes_round": peak_vram,
                        "system_error": verification.get("system_errors"),
                    }
                )
            append_jsonl(output_path, rows)
            append_jsonl(
                rounds_path,
                [
                    {
                        "method": METHOD,
                        "attempt": attempt,
                        "pending_problems": len(pending),
                        "generation_seconds": generation_seconds,
                        "verification_seconds": verification_seconds,
                        "round_seconds": round_seconds,
                        "peak_vram_bytes": peak_vram,
                    }
                ],
            )
            print(
                f"Attempt {attempt:02d}/{args.max_attempts}: "
                f"{sum(row['success'] for row in rows)}/{len(rows)} successful, "
                f"{round_seconds:.1f}s"
            )
    finally:
        verifier.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
