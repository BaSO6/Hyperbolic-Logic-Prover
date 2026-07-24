from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CHECKPOINTS = (1, 2, 4, 8, 16, 32)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def selected_problems(
    dataset: Path,
    split: str,
    limit: int = 0,
) -> list[dict[str, Any]]:
    problems = [row for row in load_jsonl(dataset) if row.get("split") == split]
    problems.sort(key=lambda row: row["name"])
    return problems[:limit] if limit else problems


def unique_problem_ids(problems: list[dict[str, Any]]) -> list[str]:
    """Return stable IDs while preserving ordinary unique theorem names."""
    counts = Counter(str(problem["name"]) for problem in problems)
    occurrences: dict[str, int] = defaultdict(int)
    identifiers = []
    for problem in problems:
        name = str(problem["name"])
        occurrences[name] += 1
        identifiers.append(
            name
            if counts[name] == 1
            else f"{name}__occurrence_{occurrences[name]:02d}"
        )
    return identifiers


def validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")


def indexed_problem_shard(
    problems: list[dict[str, Any]],
    num_shards: int,
    shard_index: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Return a deterministic round-robin shard with global problem indices."""
    validate_shard(num_shards, shard_index)
    return [
        (problem_index, problem)
        for problem_index, problem in enumerate(problems)
        if problem_index % num_shards == shard_index
    ]


def sharded_output_path(
    output: Path,
    num_shards: int,
    shard_index: int,
) -> Path:
    """Place parallel writers in separate directories while preserving filenames."""
    validate_shard(num_shards, shard_index)
    if num_shards == 1:
        return output
    shard_dir = f"shard-{shard_index:02d}-of-{num_shards:02d}"
    return output.parent / shard_dir / output.name


def validate_resume_manifest(
    output: Path,
    metadata: dict[str, Any],
    stable_fields: Iterable[str],
) -> None:
    """Refuse to mix existing result rows with a different experiment."""
    if not output.exists():
        return
    manifest_path = output.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"Cannot safely resume {output}: missing {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    if (
        "max_attempts" in previous
        and "max_attempts" in metadata
        and int(metadata["max_attempts"]) < int(previous["max_attempts"])
    ):
        raise ValueError(
            f"Refusing to reduce max_attempts for existing results in {output}: "
            f"{previous['max_attempts']} -> {metadata['max_attempts']}"
        )
    differences = [
        field
        for field in stable_fields
        if previous.get(field) != metadata.get(field)
    ]
    if differences:
        raise ValueError(
            f"Refusing to mix incompatible results in {output}; changed fields: "
            + ", ".join(differences)
        )


def completed_attempts(path: Path, method: str) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    for row in load_jsonl(path):
        if row.get("method") == method:
            completed.add((str(row["problem"]), int(row["attempt"])))
    return completed


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def runtime_manifest() -> dict[str, Any]:
    source_commit_path = project_root() / "rebuttal/SOURCE_COMMIT"
    manifest: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "recovered_source_commit": (
            source_commit_path.read_text(encoding="utf-8").strip()
            if source_commit_path.exists()
            else None
        ),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import torch

        manifest.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            }
        )
        if torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(0)
            manifest["gpu_count"] = torch.cuda.device_count()
    except Exception as exc:
        manifest["torch_import_error"] = repr(exc)
    return manifest


def optional_sha256(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def validate_attempt_bound(max_attempts: int) -> None:
    if max_attempts < 1 or max_attempts > 32:
        raise ValueError("--max-attempts must be in [1, 32]")
