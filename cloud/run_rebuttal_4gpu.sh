#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv-rebuttal/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NUM_SHARDS=4
MAX_ATTEMPTS="${MAX_ATTEMPTS:-32}"
LIMIT="${LIMIT:-0}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_HLP="${RUN_HLP:-1}"
LEAN_WORKERS_PER_SHARD="${LEAN_WORKERS_PER_SHARD:-4}"
STOP_FILE="$ROOT/results/rebuttal/GPU_MONITOR_4GPU_STOP"
LOG_DIR="$ROOT/results/rebuttal/logs"

if [[ "$RUN_NATIVE" != "1" && "$RUN_HLP" != "1" ]]; then
  echo "At least one of RUN_NATIVE or RUN_HLP must be 1."
  exit 2
fi

available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$available_gpus" -lt "$NUM_SHARDS" ]]; then
  echo "Need at least $NUM_SHARDS visible GPUs, found $available_gpus."
  exit 2
fi

mkdir -p "$LOG_DIR"
rm -f "$STOP_FILE"
python -m rebuttal.gpu_monitor \
  --output results/rebuttal/gpu_samples_4gpu.csv \
  --stop-file "$STOP_FILE" &
monitor_pid=$!

cleanup() {
  touch "$STOP_FILE"
  wait "$monitor_pid" 2>/dev/null || true
}
interrupt_children() {
  jobs -pr | xargs -r kill 2>/dev/null || true
  cleanup
  exit 130
}
trap cleanup EXIT
trap interrupt_children INT TERM

bash cloud/preflight.sh

wait_for_shards() {
  local phase="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "$phase failed; inspect $LOG_DIR/${phase}_shard_*.log"
    return 1
  fi
}

merge_expected_count="$LIMIT"
if [[ "$merge_expected_count" -eq 0 ]]; then
  merge_expected_count=244
fi

if [[ "$RUN_NATIVE" == "1" ]]; then
  echo "Launching native baseline on four independent GPUs."
  native_pids=()
  for shard_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard_index" \
      python -m rebuttal.run_native_n32 \
        --max-attempts "$MAX_ATTEMPTS" \
        --limit "$LIMIT" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard_index" \
        --lean-workers "$LEAN_WORKERS_PER_SHARD" \
        >"$LOG_DIR/native_shard_${shard_index}.log" 2>&1 &
    native_pids+=("$!")
  done
  wait_for_shards native "${native_pids[@]}"
  python -m rebuttal.merge_shards \
    --input-dir results/rebuttal/native \
    --num-shards "$NUM_SHARDS" \
    --expected-count "$merge_expected_count" \
    --max-attempts "$MAX_ATTEMPTS"
fi

if [[ "$RUN_HLP" == "1" ]]; then
  echo "Launching recovered HLP on four independent GPUs."
  hlp_pids=()
  for shard_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard_index" \
      python -m rebuttal.run_hlp_n32 \
        --max-attempts "$MAX_ATTEMPTS" \
        --max-steps 64 \
        --limit "$LIMIT" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard_index" \
        >"$LOG_DIR/hlp_shard_${shard_index}.log" 2>&1 &
    hlp_pids+=("$!")
  done
  wait_for_shards hlp "${hlp_pids[@]}"
  python -m rebuttal.merge_shards \
    --input-dir results/rebuttal/hlp \
    --num-shards "$NUM_SHARDS" \
    --expected-count "$merge_expected_count" \
    --max-attempts "$MAX_ATTEMPTS"
fi

aggregate_inputs=()
if [[ "$RUN_NATIVE" == "1" ]]; then
  aggregate_inputs+=(results/rebuttal/native/results.jsonl)
fi
if [[ "$RUN_HLP" == "1" ]]; then
  aggregate_inputs+=(results/rebuttal/hlp/results.jsonl)
fi
python -m rebuttal.aggregate "${aggregate_inputs[@]}"
echo "Finished four-GPU run. Results: $ROOT/results/rebuttal/summary"
