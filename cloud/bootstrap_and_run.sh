#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/cloud/bootstrap.sh"
bash "$ROOT/cloud/smoke_gpu.sh"
visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$visible_gpus" -ge 4 && "${FORCE_SINGLE_GPU:-0}" != "1" ]]; then
  echo "Detected $visible_gpus GPUs; starting deterministic four-GPU run."
  bash "$ROOT/cloud/run_rebuttal_4gpu.sh"
else
  echo "Detected $visible_gpus GPU(s); starting single-GPU run."
  bash "$ROOT/cloud/run_rebuttal_n32.sh"
fi
