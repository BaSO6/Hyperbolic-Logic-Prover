#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEEPSEEK_COMMIT="2c4ba9119eef74d0d611f494261b2c5bae98c69a"
MATHLIB_COMMIT="2f65ba7f1a9144b20c8e7358513548e317d26de1"

fail=0
for command_name in python nvidia-smi git lake; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "MISSING COMMAND: $command_name"
    fail=1
  fi
done

if [[ ! -d third_party/DeepSeek-Prover-V1.5/.git ]]; then
  echo "MISSING: third_party/DeepSeek-Prover-V1.5"
  fail=1
fi
if [[ ! -f models/DeepSeek-Prover-V1.5-RL/config.json ]]; then
  echo "MISSING: models/DeepSeek-Prover-V1.5-RL"
  fail=1
fi
if [[ ! -f models/all-MiniLM-L6-v2/config.json ]]; then
  echo "MISSING: models/all-MiniLM-L6-v2"
  fail=1
fi
if [[ ! -e data/mathlib4 ]]; then
  echo "MISSING: data/mathlib4"
  fail=1
fi
if [[ -d third_party/DeepSeek-Prover-V1.5/.git ]]; then
  actual_deepseek="$(git -C third_party/DeepSeek-Prover-V1.5 rev-parse HEAD)"
  if [[ "$actual_deepseek" != "$DEEPSEEK_COMMIT" ]]; then
    echo "WRONG DEEPSEEK COMMIT: $actual_deepseek"
    fail=1
  fi
fi
if [[ -e third_party/DeepSeek-Prover-V1.5/mathlib4/.git ]]; then
  actual_mathlib="$(git -C third_party/DeepSeek-Prover-V1.5/mathlib4 rev-parse HEAD)"
  if [[ "$actual_mathlib" != "$MATHLIB_COMMIT" ]]; then
    echo "WRONG MATHLIB COMMIT: $actual_mathlib"
    fail=1
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  echo "Preflight stopped before Python/Lean checks because required inputs are missing."
  exit 2
fi

echo "Disk:"
df -h "$ROOT"
echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

python -m rebuttal.audit_reproducibility
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY

echo "Preflight passed for the recovered-system rebuttal suite."
