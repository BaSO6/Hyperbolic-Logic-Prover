"""Four-arm retrieval audit for the reconstructed corrected cone model."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rebuttal.common import atomic_json, project_root, runtime_manifest, sha256


ARMS = (
    "distance",
    "paper_origin_forward",
    "corrected_apex_forward",
    "corrected_inverse",
)


def _merge_topk(
    best_scores: Any,
    best_indices: Any,
    chunk_scores: Any,
    chunk_offset: int,
    top_k: int,
) -> tuple[Any, Any]:
    import torch

    batch_size, chunk_size = chunk_scores.shape
    chunk_indices = torch.arange(
        chunk_offset,
        chunk_offset + chunk_size,
        device=chunk_scores.device,
    ).expand(batch_size, -1)
    if best_scores is None:
        combined_scores = chunk_scores
        combined_indices = chunk_indices
    else:
        combined_scores = torch.cat([best_scores, chunk_scores], dim=1)
        combined_indices = torch.cat([best_indices, chunk_indices], dim=1)
    keep = min(top_k, combined_scores.shape[1])
    values, positions = torch.topk(
        combined_scores, k=keep, dim=1, largest=False
    )
    indices = torch.gather(combined_indices, 1, positions)
    return values, indices


def rank_candidates(
    queries: Any,
    candidates: Any,
    query_indices: Any,
    arm: str,
    top_k: int,
    candidate_chunk_size: int,
    cone_k: float,
    epsilon: float,
) -> tuple[Any, Any, Any]:
    import torch

    from src.system1.entailment_cones import (
        cone_energy,
        cone_rank_scores,
        origin_cone_energy,
    )
    from src.system1.manifold_math import PoincareBall

    manifold = PoincareBall(c=1.0)
    best_scores = None
    best_indices = None
    containment_counts = torch.zeros(
        queries.shape[0], dtype=torch.long, device=queries.device
    )
    for offset in range(0, candidates.shape[0], candidate_chunk_size):
        chunk = candidates[offset : offset + candidate_chunk_size]
        query_grid = queries[:, None, :]
        candidate_grid = chunk[None, :, :]
        distance = manifold.dist(query_grid, candidate_grid)
        if arm == "distance":
            energy = torch.zeros_like(distance)
            scores = distance
        elif arm == "paper_origin_forward":
            energy = origin_cone_energy(
                query_grid,
                candidate_grid,
                cone_k=cone_k,
                epsilon=epsilon,
            )
            scores, _ = cone_rank_scores(energy, distance)
        elif arm == "corrected_apex_forward":
            energy = cone_energy(
                query_grid,
                candidate_grid,
                cone_k=cone_k,
                epsilon=epsilon,
            )
            scores, _ = cone_rank_scores(energy, distance)
        elif arm == "corrected_inverse":
            energy = cone_energy(
                candidate_grid,
                query_grid,
                cone_k=cone_k,
                epsilon=epsilon,
            )
            scores, _ = cone_rank_scores(energy, distance)
        else:
            raise ValueError(f"Unknown arm: {arm}")

        local_query_indices = query_indices - offset
        in_chunk = (local_query_indices >= 0) & (
            local_query_indices < chunk.shape[0]
        )
        membership = energy <= 1e-7
        if torch.any(in_chunk):
            rows = torch.nonzero(in_chunk, as_tuple=False).flatten()
            scores[rows, local_query_indices[rows]] = float("inf")
            membership[rows, local_query_indices[rows]] = False
        if arm != "distance":
            containment_counts += membership.sum(dim=1)
        best_scores, best_indices = _merge_topk(
            best_scores,
            best_indices,
            scores,
            offset,
            top_k,
        )
    return best_scores, best_indices, containment_counts


def pair_diagnostics(
    embeddings: Any,
    test_edges: Any,
    cone_k: float,
    epsilon: float,
) -> dict[str, Any]:
    import torch

    from src.system1.entailment_cones import cone_energy, origin_cone_energy

    premise, theorem = test_edges[:, 0], test_edges[:, 1]
    with torch.no_grad():
        inverse = cone_energy(
            embeddings[premise],
            embeddings[theorem],
            cone_k=cone_k,
            epsilon=epsilon,
        )
        apex_forward = cone_energy(
            embeddings[theorem],
            embeddings[premise],
            cone_k=cone_k,
            epsilon=epsilon,
        )
        origin_forward = origin_cone_energy(
            embeddings[theorem],
            embeddings[premise],
            cone_k=cone_k,
            epsilon=epsilon,
        )
    return {
        "test_edge_count": int(test_edges.shape[0]),
        "corrected_inverse_containment_rate": float(
            (inverse <= 1e-7).float().mean().cpu()
        ),
        "corrected_apex_forward_containment_rate": float(
            (apex_forward <= 1e-7).float().mean().cpu()
        ),
        "paper_origin_forward_containment_rate": float(
            (origin_forward <= 1e-7).float().mean().cpu()
        ),
        "mean_corrected_inverse_energy": float(inverse.mean().cpu()),
        "mean_corrected_apex_forward_energy": float(apex_forward.mean().cpu()),
        "mean_paper_origin_forward_energy": float(origin_forward.mean().cpu()),
        "premise_radius_mean": float(
            embeddings[premise].norm(dim=-1).mean().cpu()
        ),
        "theorem_radius_mean": float(
            embeddings[theorem].norm(dim=-1).mean().cpu()
        ),
        "radial_order_rate": float(
            (
                embeddings[premise].norm(dim=-1)
                <= embeddings[theorem].norm(dim=-1)
            )
            .float()
            .mean()
            .cpu()
        ),
    }


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Compare distance and three cone-direction retrieval arms."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "results/rebuttal/corrected_cone",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--candidate-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/rebuttal/corrected_cone/diagnostics",
    )
    args = parser.parse_args()

    import torch

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_manifest = json.loads(
        (model_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    cone_k = float(training_manifest["cone_k"])
    epsilon = float(training_manifest["epsilon"])
    embeddings = torch.load(
        model_dir / "node_embeddings.pt", map_location=args.device
    )
    test_edges = torch.load(
        model_dir / "test_edges.pt", map_location=args.device
    )
    with gzip.open(
        model_dir / "node_names.json.gz", "rt", encoding="utf-8"
    ) as handle:
        node_names = json.load(handle)
    if len(node_names) != embeddings.shape[0]:
        raise ValueError("Node-name and embedding counts differ")

    truth: dict[int, set[int]] = defaultdict(set)
    for premise, theorem in test_edges.detach().cpu().tolist():
        truth[int(theorem)].add(int(premise))
    query_indices_list = sorted(truth)
    if args.max_queries:
        query_indices_list = query_indices_list[: args.max_queries]
    if not query_indices_list:
        raise ValueError("No held-out theorem queries are available")
    if args.top_k < 1 or args.query_batch_size < 1 or args.candidate_chunk_size < 1:
        raise ValueError("top-k and batch/chunk sizes must be positive")
    query_indices = torch.tensor(
        query_indices_list, dtype=torch.long, device=args.device
    )

    rows: list[dict[str, Any]] = []
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = []
        for offset in range(0, len(query_indices_list), args.query_batch_size):
            batch_indices = query_indices[
                offset : offset + args.query_batch_size
            ]
            queries = embeddings[batch_indices]
            with torch.no_grad():
                scores, ranked, containment_counts = rank_candidates(
                    queries=queries,
                    candidates=embeddings,
                    query_indices=batch_indices,
                    arm=arm,
                    top_k=args.top_k,
                    candidate_chunk_size=args.candidate_chunk_size,
                    cone_k=cone_k,
                    epsilon=epsilon,
                )
            for row_offset, theorem_index in enumerate(
                batch_indices.detach().cpu().tolist()
            ):
                ranked_indices = ranked[row_offset].detach().cpu().tolist()
                relevant = truth[int(theorem_index)]
                hit_positions = [
                    position + 1
                    for position, candidate in enumerate(ranked_indices)
                    if candidate in relevant
                ]
                hit_count = len(hit_positions)
                row = {
                    "arm": arm,
                    "theorem_index": int(theorem_index),
                    "theorem": node_names[int(theorem_index)],
                    "heldout_premise_count": len(relevant),
                    "hits_at_k": hit_count,
                    "recall_at_k": hit_count / len(relevant),
                    "hit_at_k": hit_count > 0,
                    "reciprocal_rank": (
                        1.0 / min(hit_positions) if hit_positions else 0.0
                    ),
                    "contained_candidate_count": (
                        None
                        if arm == "distance"
                        else int(containment_counts[row_offset].cpu())
                    ),
                    "top_score": float(scores[row_offset, 0].cpu()),
                }
                rows.append(row)
                arm_rows.append(row)
        arm_summaries[arm] = {
            "queries": len(arm_rows),
            "mean_recall_at_k": sum(
                row["recall_at_k"] for row in arm_rows
            )
            / max(1, len(arm_rows)),
            "hit_rate_at_k": sum(row["hit_at_k"] for row in arm_rows)
            / max(1, len(arm_rows)),
            "mean_reciprocal_rank": sum(
                row["reciprocal_rank"] for row in arm_rows
            )
            / max(1, len(arm_rows)),
            "mean_contained_candidates": (
                None
                if arm == "distance"
                else sum(
                    int(row["contained_candidate_count"]) for row in arm_rows
                )
                / max(1, len(arm_rows))
            ),
        }
        print(json.dumps({arm: arm_summaries[arm]}, indent=2))

    with (output_dir / "per_query.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        **runtime_manifest(),
        "implementation": "reconstructed_corrected_entailment_cones",
        "model_dir": str(model_dir),
        "training_manifest_sha256": sha256(
            model_dir / "training_manifest.json"
        ),
        "embeddings_sha256": sha256(model_dir / "node_embeddings.pt"),
        "top_k": args.top_k,
        "query_count": len(query_indices_list),
        "arms": arm_summaries,
        "pair_diagnostics": pair_diagnostics(
            embeddings, test_edges, cone_k, epsilon
        ),
    }
    atomic_json(output_dir / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
