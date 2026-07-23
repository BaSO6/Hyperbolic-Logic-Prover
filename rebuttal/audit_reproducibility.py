from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rebuttal.common import atomic_json, project_root, selected_problems, sha256


REQUIRED_ASSETS = (
    "data/hgcn_final.pth",
    "data/hgcn_refined.pth",
    "data/node_embeddings.pt",
    "data/id_to_name.pkl.gz",
    "data/node_text_map.pkl.gz",
)


def source_position(text: str, needle: str) -> int | None:
    index = text.find(needle)
    return None if index < 0 else text[:index].count("\n") + 1


def audit(root: Path, dataset: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    assets: dict[str, Any] = {}
    for relative in REQUIRED_ASSETS:
        path = root / relative
        exists = path.is_file()
        entry: dict[str, Any] = {"exists": exists}
        if exists:
            entry.update(size_bytes=path.stat().st_size, sha256=sha256(path))
        assets[relative] = entry

    agent_path = root / "src/system2/lie_search.py"
    train_path = root / "src/system1/preprocess_data.py"
    agent_text = agent_path.read_text(encoding="utf-8")
    train_text = train_path.read_text(encoding="utf-8")

    has_heap_search = "heapq.heappush" in agent_text and "frontier" in agent_text
    has_cone_formula = bool(re.search(r"arcsin|aperture|cone_filter", agent_text, re.I))
    retrieval_line = source_position(agent_text, "def retrieve_theorems")
    llm_line = source_position(agent_text, "self.llm.generate_candidates")
    lie_line = source_position(agent_text, "self.lie.apply_tactic")
    move_after_llm = bool(llm_line and lie_line and lie_line > llm_line)
    retrieval_reencodes_goal = "q = self.goal_encoder.encode(query_text" in agent_text
    explicit_lie_load = bool(
        re.search(r"(lie|tac_to_coeff).{0,80}load_state_dict", agent_text, re.S)
    )
    true_cone_loss = bool(re.search(r"arcsin|angle.*aperture|entailment.*angle", train_text, re.I))

    checks = {
        "single_trajectory": not has_heap_search,
        "entailment_cone_retrieval": has_cone_formula,
        "move_precedes_llm": not move_after_llm,
        "moved_state_drives_retrieval": not retrieval_reencodes_goal,
        "trained_lie_checkpoint_loaded": explicit_lie_load,
        "entailment_cone_training_loss": true_cone_loss,
    }
    explanations = {
        "single_trajectory": "Recovered agent maintains a heap frontier/A* search.",
        "entailment_cone_retrieval": "Recovered retrieval uses distance top-k; no cone aperture/filter was found.",
        "move_precedes_llm": "Recovered code calls the LLM before applying the tactic-conditioned Lie move.",
        "moved_state_drives_retrieval": "Recovered retrieval re-encodes goal text and ignores the moved state passed through the frontier.",
        "trained_lie_checkpoint_loaded": "Recovered agent initializes Lie/tactic heads but does not load their trained weights.",
        "entailment_cone_training_loss": "The file labels a cone loss, but implements norm regularization and distance margin loss.",
    }
    for name, passed in checks.items():
        if not passed:
            findings.append(
                {
                    "severity": "BLOCKS_PAPER_STRICT",
                    "check": name,
                    "detail": explanations[name],
                }
            )

    test_count = len(selected_problems(dataset, "test")) if dataset.exists() else 0
    if test_count != 244:
        findings.append(
            {
                "severity": "ERROR",
                "check": "official_minif2f_test_count",
                "detail": f"Expected 244 official test problems, found {test_count}.",
            }
        )

    checkpoint_info: dict[str, Any] = {}
    try:
        import torch

        for name in ("hgcn_final.pth", "hgcn_refined.pth"):
            value = torch.load(root / "data" / name, map_location="cpu")
            state = value.get("model", {}) if isinstance(value, dict) else {}
            checkpoint_info[name] = {
                "top_level_keys": sorted(value.keys()) if isinstance(value, dict) else [],
                "state_keys": sorted(state.keys()),
                "contains_lie_parameters": any(
                    "lie" in key.lower() or "tac_to_coeff" in key.lower()
                    for key in state
                ),
            }
    except Exception as exc:
        checkpoint_info["inspection_error"] = repr(exc)
        findings.append(
            {
                "severity": "WARNING",
                "check": "checkpoint_inspection",
                "detail": f"Could not inspect checkpoint tensors: {exc!r}",
            }
        )

    strict_ready = (
        all(entry["exists"] for entry in assets.values())
        and test_count == 244
        and all(checks.values())
        and all(
            checkpoint_info.get(name, {}).get("contains_lie_parameters", False)
            for name in ("hgcn_final.pth", "hgcn_refined.pth")
        )
    )

    return {
        "paper_strict_ready": strict_ready,
        "recovered_system_runnable": all(entry["exists"] for entry in assets.values()),
        "dataset": str(dataset),
        "official_test_count": test_count,
        "assets": assets,
        "source_checks": checks,
        "source_locations": {
            "retrieval_function": retrieval_line,
            "llm_call": llm_line,
            "lie_move": lie_line,
        },
        "checkpoint_info": checkpoint_info,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root() / "rebuttal/datasets/minif2f.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-paper-strict", action="store_true")
    args = parser.parse_args()

    report = audit(args.root.resolve(), args.dataset.resolve())
    output = args.output or args.root / "results/rebuttal/audit.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAudit written to {output}")
    if args.require_paper_strict and not report["paper_strict_ready"]:
        print("ERROR: paper-strict execution is blocked by the findings above.", file=sys.stderr)
        return 2
    return 0 if report["recovered_system_runnable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
