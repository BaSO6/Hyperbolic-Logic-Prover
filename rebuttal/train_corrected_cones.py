"""Train a reconstructed, direction-sensitive Poincaré entailment-cone model.

This is a corrected implementation reconstructed from the paper definition.  It
is intentionally labelled separately from the unavailable original training
run and writes a complete manifest for rebuttal auditing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from rebuttal.common import (
    append_jsonl,
    atomic_json,
    atomic_jsonl,
    load_jsonl,
    project_root,
    runtime_manifest,
    sha256,
)


def stable_bucket(premise: str, theorem: str, buckets: int = 100) -> int:
    payload = f"{premise}\0{theorem}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % buckets


def load_benchmark_names(paths: list[Path], split: str) -> set[str]:
    names: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == split:
                    names.add(str(row["name"]))
    return names


def matched_declarations(
    declarations: set[str],
    benchmark_names: set[str],
) -> set[str]:
    return {
        declaration
        for declaration in declarations
        if any(
            declaration == name or declaration.endswith("." + name)
            for name in benchmark_names
        )
    }


def load_directed_graph(
    proof_dependencies: Path,
    benchmark_paths: list[Path],
    benchmark_split: str,
) -> tuple[
    dict[str, Any],
    list[str],
    list[tuple[int, int]],
    dict[str, Any],
]:
    with proof_dependencies.open("r", encoding="utf-8") as handle:
        proof_data = json.load(handle)
    declaration_names = set(proof_data)
    referenced_premise_names = {
        str(premise)
        for info in proof_data.values()
        for premise in info.get("used_lemmas", [])
    }
    graph_names = declaration_names | referenced_premise_names
    benchmark_names = load_benchmark_names(benchmark_paths, benchmark_split)
    excluded = matched_declarations(graph_names, benchmark_names)
    node_names = sorted(graph_names - excluded)
    node_to_index = {name: index for index, name in enumerate(node_names)}

    # Positive entailment orientation: premise -> theorem.
    directed_edges: set[tuple[int, int]] = set()
    raw_dependency_references = 0
    excluded_incident_edges = 0
    for theorem, info in proof_data.items():
        for premise in info.get("used_lemmas", []):
            raw_dependency_references += 1
            if theorem in excluded or premise in excluded:
                excluded_incident_edges += 1
                continue
            directed_edges.add(
                (node_to_index[premise], node_to_index[theorem])
            )

    report = {
        "benchmark_split": benchmark_split,
        "benchmark_files": [str(path.resolve()) for path in benchmark_paths],
        "benchmark_file_sha256": {
            str(path.resolve()): sha256(path) for path in benchmark_paths
        },
        "benchmark_problem_names": len(benchmark_names),
        "raw_declarations": len(declaration_names),
        "premise_only_nodes": len(referenced_premise_names - declaration_names),
        "raw_graph_nodes": len(graph_names),
        "excluded_graph_nodes": len(excluded),
        "excluded_declaration_nodes": len(excluded & declaration_names),
        "excluded_premise_only_nodes": len(
            excluded - declaration_names
        ),
        "excluded_graph_node_names": sorted(excluded),
        "retained_graph_nodes": len(node_names),
        "raw_dependency_references": raw_dependency_references,
        "excluded_incident_edges": excluded_incident_edges,
        "retained_directional_edges": len(directed_edges),
        "edge_semantics": "premise_to_theorem",
    }
    return proof_data, node_names, sorted(directed_edges), report


def split_edges(
    node_names: list[str],
    edges: list[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    output = {"train": [], "valid": [], "test": []}
    for premise_index, theorem_index in edges:
        bucket = stable_bucket(
            node_names[premise_index], node_names[theorem_index]
        )
        split = "train" if bucket < 90 else ("valid" if bucket < 95 else "test")
        output[split].append((premise_index, theorem_index))
    return output


def node_names_digest(node_names: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(node_names) + "\n").encode("utf-8")
    ).hexdigest()


def atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_or_load_features(
    proof_data: dict[str, Any],
    node_names: list[str],
    model_path: Path,
    output_dir: Path,
    batch_size: int,
    device: str,
) -> Any:
    import torch

    features_path = output_dir / "node_features.pt"
    feature_manifest_path = output_dir / "feature_manifest.json"
    names_hash = node_names_digest(node_names)
    model_config = model_path / "config.json"
    expected_manifest = {
        "node_count": len(node_names),
        "node_names_sha256": names_hash,
        "model_config_sha256": sha256(model_config),
        "normalized": True,
    }
    if features_path.is_file() and feature_manifest_path.is_file():
        previous = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        if previous != expected_manifest:
            raise ValueError(
                "Existing feature cache has a different provenance; use a new "
                "--output-dir instead of mixing experiments."
            )
        features = torch.load(features_path, map_location="cpu")
        if features.ndim != 2 or features.shape[0] != len(node_names):
            raise ValueError("Cached node feature tensor has an invalid shape")
        return features

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(model_path), device=device)
    encoder.eval()
    feature_batches = []
    for offset in range(0, len(node_names), batch_size):
        batch_names = node_names[offset : offset + batch_size]
        texts = []
        for name in batch_names:
            info = proof_data.get(name, {})
            type_signature = info.get("type", "")
            heads = " ".join(info.get("head_symbols", []))
            text = f"{name} : {type_signature}" if type_signature else name
            if heads:
                text += f" | HEADS: {heads}"
            texts.append(text)
        with torch.no_grad():
            feature_batches.append(
                encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).cpu()
            )
        print(
            f"Features: {min(offset + batch_size, len(node_names))}/"
            f"{len(node_names)}"
        )
    features = torch.cat(feature_batches, dim=0)
    atomic_torch_save(features, features_path)
    atomic_json(feature_manifest_path, expected_manifest)
    return features


def make_message_edges(train_edges: Any, device: str) -> Any:
    import torch

    reverse = train_edges[:, [1, 0]]
    undirected = torch.cat([train_edges, reverse], dim=0)
    return undirected.t().contiguous().to(device)


def evaluate_edge_losses(
    embeddings: Any,
    edges: Any,
    cone_k: float,
    epsilon: float,
) -> dict[str, float]:
    import torch

    from src.system1.entailment_cones import cone_energy

    if edges.numel() == 0:
        return {"count": 0, "containment_rate": 0.0, "mean_energy": 0.0}
    premise, theorem = edges[:, 0], edges[:, 1]
    with torch.no_grad():
        energy = cone_energy(
            embeddings[premise],
            embeddings[theorem],
            cone_k=cone_k,
            epsilon=epsilon,
        )
    return {
        "count": int(edges.shape[0]),
        "containment_rate": float((energy <= 1e-7).float().mean().cpu()),
        "mean_energy": float(energy.mean().cpu()),
    }


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train the reconstructed corrected entailment-cone model."
    )
    parser.add_argument(
        "--proof-dependencies",
        type=Path,
        default=root / "data/proof_local_deps.json",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        action="append",
        default=None,
        help="JSONL benchmark whose selected split must be excluded.",
    )
    parser.add_argument("--exclude-split", default="test")
    parser.add_argument(
        "--sentence-model",
        type=Path,
        default=root / "models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/rebuttal/corrected_cone",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--edge-batch-size", type=int, default=32768)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cone-k", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-radius", type=float, default=0.95)
    parser.add_argument("--negative-margin", type=float, default=0.2)
    parser.add_argument("--radial-margin", type=float, default=0.01)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--radial-weight", type=float, default=0.2)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.edge_batch_size < 1 or args.feature_batch_size < 1:
        parser.error("batch sizes must be at least 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")

    import torch
    import torch.nn.functional as F

    from src.system1.corrected_cone_model import CorrectedConeEncoder
    from src.system1.entailment_cones import (
        cone_energy,
        validate_cone_parameters,
    )
    from src.system1.manifold_math import PoincareBall

    validate_cone_parameters(args.cone_k, args.epsilon)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_paths = args.exclude_dataset or [
        root / "rebuttal/datasets/minif2f.jsonl",
        root / "rebuttal/datasets/proofnet.jsonl",
    ]
    benchmark_paths = [path.expanduser().resolve() for path in benchmark_paths]
    proof_path = args.proof_dependencies.expanduser().resolve()
    sentence_model = args.sentence_model.expanduser().resolve()
    if not sentence_model.is_dir():
        raise SystemExit(f"Sentence model not found: {sentence_model}")
    if not (sentence_model / "config.json").is_file():
        raise SystemExit(f"Sentence model config not found: {sentence_model}")

    proof_data, node_names, directed_edges, exclusion_report = load_directed_graph(
        proof_path,
        benchmark_paths,
        args.exclude_split,
    )
    edge_splits = split_edges(node_names, directed_edges)
    names_hash = node_names_digest(node_names)
    with gzip.open(output_dir / "node_names.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(node_names, handle, ensure_ascii=False)
    atomic_json(output_dir / "exclusion_report.json", exclusion_report)

    features = generate_or_load_features(
        proof_data=proof_data,
        node_names=node_names,
        model_path=sentence_model,
        output_dir=output_dir,
        batch_size=args.feature_batch_size,
        device=str(device),
    )
    split_tensors = {
        split: torch.tensor(pairs, dtype=torch.long)
        for split, pairs in edge_splits.items()
    }
    for split, tensor in split_tensors.items():
        atomic_torch_save(tensor, output_dir / f"{split}_edges.pt")
    message_edges = make_message_edges(split_tensors["train"], str(device))
    train_edges = split_tensors["train"].to(device)
    valid_edges = split_tensors["valid"].to(device)
    test_edges = split_tensors["test"].to(device)
    features = features.to(device)

    config = {
        "schema_version": 1,
        "implementation": "reconstructed_corrected_entailment_cones",
        "original_training_artifact_recovered": False,
        "proof_dependencies": str(proof_path),
        "proof_dependencies_sha256": sha256(proof_path),
        "node_count": len(node_names),
        "node_names_sha256": names_hash,
        "edge_counts": {
            split: len(pairs) for split, pairs in edge_splits.items()
        },
        "edge_semantics": "premise_to_theorem",
        "message_graph": "bidirectional_train_edges_only",
        "heldout_edge_split": "sha256_pair_90_5_5",
        "sentence_model": str(sentence_model),
        "sentence_model_config_sha256": sha256(sentence_model / "config.json"),
        "input_dim": int(features.shape[1]),
        "output_dim": args.output_dim,
        "cone_k": args.cone_k,
        "epsilon": args.epsilon,
        "max_radius": args.max_radius,
        "negative_margin": args.negative_margin,
        "radial_margin": args.radial_margin,
        "negative_weight": args.negative_weight,
        "radial_weight": args.radial_weight,
        "alignment_weight": args.alignment_weight,
        "learning_rate": args.learning_rate,
        "edge_batch_size": args.edge_batch_size,
        "seed": args.seed,
        "exclude_split": args.exclude_split,
        "exclusion_report": exclusion_report,
    }
    atomic_json(output_dir / "training_config.json", config)

    model = CorrectedConeEncoder(
        input_dim=int(features.shape[1]),
        output_dim=args.output_dim,
        epsilon=args.epsilon,
        max_radius=args.max_radius,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = output_dir / "training_checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint["config"] != config:
            raise ValueError(
                "Existing corrected-cone checkpoint has different configuration; "
                "use a new --output-dir."
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resuming after epoch {start_epoch}")
    if start_epoch >= args.epochs:
        print(f"Training already complete through epoch {start_epoch}")

    manifold = PoincareBall(c=1.0)
    metrics_path = output_dir / "training_metrics.jsonl"
    if metrics_path.is_file():
        retained_metrics = [
            row
            for row in load_jsonl(metrics_path)
            if int(row.get("epoch", -1)) <= start_epoch
        ]
        atomic_jsonl(metrics_path, retained_metrics)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_started = time.perf_counter()
        sample_count = min(args.edge_batch_size, train_edges.shape[0])
        selected = torch.randperm(train_edges.shape[0], device=device)[:sample_count]
        batch = train_edges[selected]
        premise, theorem = batch[:, 0], batch[:, 1]
        negative_premise = torch.randint(
            0, features.shape[0], (sample_count,), device=device
        )

        optimizer.zero_grad(set_to_none=True)
        embeddings = model(features, message_edges)
        positive_energy = cone_energy(
            embeddings[premise],
            embeddings[theorem],
            cone_k=args.cone_k,
            epsilon=args.epsilon,
        )
        negative_energy = cone_energy(
            embeddings[negative_premise],
            embeddings[theorem],
            cone_k=args.cone_k,
            epsilon=args.epsilon,
        )
        loss_positive = positive_energy.mean()
        loss_negative = F.relu(
            args.negative_margin - negative_energy
        ).mean()
        premise_radius = embeddings[premise].norm(dim=-1)
        theorem_radius = embeddings[theorem].norm(dim=-1)
        loss_radial = F.relu(
            premise_radius - theorem_radius + args.radial_margin
        ).mean()

        alignment_count = min(4096, features.shape[0])
        alignment_nodes = torch.randperm(
            features.shape[0], device=device
        )[:alignment_count]
        query_embeddings = model.encode_queries(features[alignment_nodes])
        loss_alignment = manifold.dist(
            query_embeddings, embeddings[alignment_nodes]
        ).mean()
        loss = (
            loss_positive
            + args.negative_weight * loss_negative
            + args.radial_weight * loss_radial
            + args.alignment_weight * loss_alignment
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "positive_cone_loss": float(loss_positive.detach().cpu()),
            "negative_margin_loss": float(loss_negative.detach().cpu()),
            "radial_order_loss": float(loss_radial.detach().cpu()),
            "query_alignment_loss": float(loss_alignment.detach().cpu()),
            "sampled_positive_containment": float(
                (positive_energy.detach() <= 1e-7).float().mean().cpu()
            ),
            "sampled_negative_containment": float(
                (negative_energy.detach() <= 1e-7).float().mean().cpu()
            ),
            "mean_radius": float(embeddings.detach().norm(dim=-1).mean().cpu()),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        append_jsonl(metrics_path, [row])
        print(
            f"Epoch {epoch:03d}/{args.epochs}: loss={row['loss']:.5f}, "
            f"pos-in={row['sampled_positive_containment']:.3f}, "
            f"neg-in={row['sampled_negative_containment']:.3f}, "
            f"{row['epoch_seconds']:.1f}s"
        )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint = {
                "epoch": epoch,
                "config": config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }
            atomic_torch_save(checkpoint, checkpoint_path)

    model.eval()
    with torch.no_grad():
        final_embeddings = model(features, message_edges)
    model_checkpoint_path = output_dir / "model_checkpoint.pt"
    atomic_torch_save(
        {
            "config": config,
            "model_state": model.state_dict(),
        },
        model_checkpoint_path,
    )
    atomic_torch_save(final_embeddings.cpu(), output_dir / "node_embeddings.pt")
    final_report = {
        **runtime_manifest(),
        **config,
        "completed_epochs": max(start_epoch, args.epochs),
        "train_edge_metrics": evaluate_edge_losses(
            final_embeddings, train_edges, args.cone_k, args.epsilon
        ),
        "valid_edge_metrics": evaluate_edge_losses(
            final_embeddings, valid_edges, args.cone_k, args.epsilon
        ),
        "test_edge_metrics": evaluate_edge_losses(
            final_embeddings, test_edges, args.cone_k, args.epsilon
        ),
        "training_checkpoint_sha256": sha256(checkpoint_path),
        "model_checkpoint_sha256": sha256(model_checkpoint_path),
        "embeddings_sha256": sha256(output_dir / "node_embeddings.pt"),
    }
    atomic_json(output_dir / "training_manifest.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
