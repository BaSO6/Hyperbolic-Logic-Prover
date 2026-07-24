#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv-rebuttal/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_INDEX="${SHARD_INDEX:?Set SHARD_INDEX to 0, 1, 2, or 3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-32}"
LIMIT="${LIMIT:-0}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_HLP="${RUN_HLP:-1}"
LEAN_WORKERS_PER_SHARD="${LEAN_WORKERS_PER_SHARD:-4}"

if [[ "$SHARD_INDEX" -lt 0 || "$SHARD_INDEX" -ge "$NUM_SHARDS" ]]; then
  echo "SHARD_INDEX must be in [0, NUM_SHARDS)."
  exit 2
fi

bash cloud/preflight.sh

if [[ "$RUN_NATIVE" == "1" ]]; then
  python -m rebuttal.run_native_n32 \
    --max-attempts "$MAX_ATTEMPTS" \
    --limit "$LIMIT" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --lean-workers "$LEAN_WORKERS_PER_SHARD"
fi

if [[ "$RUN_HLP" == "1" ]]; then
  python -m rebuttal.run_hlp_n32 \
    --max-attempts "$MAX_ATTEMPTS" \
    --max-steps 64 \
    --limit "$LIMIT" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX"
fi

echo "Finished shard $SHARD_INDEX/$NUM_SHARDS."
