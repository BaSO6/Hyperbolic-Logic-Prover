"""Static/data audit for the reconstructed corrected-cone experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rebuttal.common import atomic_json, project_root, sha256
from rebuttal.train_corrected_cones import load_benchmark_names, matched_declarations


def audit(root: Path) -> dict[str, Any]:
    proof_path = root / "data/proof_local_deps.json"
    with proof_path.open("r", encoding="utf-8") as handle:
        proof_data = json.load(handle)
    declarations = set(proof_data)
    referenced_premises = {
        str(premise)
        for info in proof_data.values()
        for premise in info.get("used_lemmas", [])
    }
    graph_nodes = declarations | referenced_premises
    raw_dependency_references = sum(
        len(info.get("used_lemmas", [])) for info in proof_data.values()
    )
    benchmark_reports = {}
    for relative in (
        "rebuttal/datasets/minif2f.jsonl",
        "rebuttal/datasets/proofnet.jsonl",
    ):
        path = root / relative
        names = load_benchmark_names([path], "test")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        test_rows = [row for row in rows if row.get("split") == "test"]
        matches = matched_declarations(graph_nodes, names)
        benchmark_reports[relative] = {
            "test_row_count": len(test_rows),
            "unique_test_theorem_names": len(names),
            "duplicate_name_rows": len(test_rows) - len(names),
            "exact_or_namespace_suffix_overlap": len(matches),
            "matched_graph_nodes": sorted(matches),
            "sha256": sha256(path),
        }

    cone_path = root / "src/system1/entailment_cones.py"
    train_path = root / "rebuttal/train_corrected_cones.py"
    agent_path = root / "src/system2/lie_search.py"
    cone_text = cone_path.read_text(encoding="utf-8")
    train_text = train_path.read_text(encoding="utf-8")
    agent_text = agent_path.read_text(encoding="utf-8")
    checks = {
        "apex_angle_implemented": "def apex_angle" in cone_text,
        "inner_radius_validated": "validate_cone_parameters" in cone_text,
        "inverse_cone_implemented": "def inverse_cone_energy" in cone_text,
        "premise_to_theorem_training": (
            '"edge_semantics": "premise_to_theorem"' in train_text
        ),
        "directional_cone_loss": "positive_energy = cone_energy" in train_text,
        "inverse_cone_agent_arm": (
            'self.retrieval_strategy == "corrected_apex_forward"' in agent_text
            and "self.graph_emb,\n                        q," in agent_text
        ),
    }
    artifact_dir = root / "results/rebuttal/corrected_cone"
    artifacts = {}
    for name in (
        "training_manifest.json",
        "model_checkpoint.pt",
        "node_embeddings.pt",
        "diagnostics/summary.json",
    ):
        path = artifact_dir / name
        artifacts[name] = {
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return {
        "implementation": "reconstructed_corrected_entailment_cones",
        "original_training_artifact_recovered": False,
        "proof_dependencies_sha256": sha256(proof_path),
        "declaration_count": len(declarations),
        "premise_only_node_count": len(referenced_premises - declarations),
        "graph_node_count": len(graph_nodes),
        "raw_dependency_references": raw_dependency_references,
        "benchmark_overlap": benchmark_reports,
        "source_checks": checks,
        "source_ready": all(checks.values()),
        "trained_artifacts_ready": all(
            entry["exists"] for entry in artifacts.values()
        ),
        "artifacts": artifacts,
    }


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/rebuttal/corrected_cone/audit.json",
    )
    args = parser.parse_args()
    report = audit(args.root.expanduser().resolve())
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["source_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
