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
EXPECTED_COUNT="$LIMIT"
if [[ "$EXPECTED_COUNT" -eq 0 ]]; then
  EXPECTED_COUNT=244
fi
available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$available_gpus" -lt "$NUM_SHARDS" ]]; then
  echo "Need four visible GPUs, found $available_gpus."
  exit 2
fi
if [[ ! -f results/rebuttal/corrected_cone/training_manifest.json ]]; then
  echo "Missing corrected cone artifacts; run cloud/train_corrected_cones.sh first."
  exit 2
fi
bash cloud/preflight.sh

OUTPUT_ROOT="results/rebuttal/corrected_inverse_n32"
STOP_FILE="$OUTPUT_ROOT/GPU_MONITOR_STOP"
mkdir -p results/rebuttal/logs "$OUTPUT_ROOT"
rm -f "$STOP_FILE"
python -m rebuttal.gpu_monitor \
  --output "$OUTPUT_ROOT/gpu_samples_4gpu.csv" \
  --stop-file "$STOP_FILE" &
monitor_pid=$!
pids=()
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
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" \
    python -m rebuttal.run_hlp_n32 \
      --mode corrected_inverse \
      --max-attempts "$MAX_ATTEMPTS" \
      --max-steps 64 \
      --limit "$LIMIT" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_index" \
      --output "$OUTPUT_ROOT/results.jsonl" \
      >"results/rebuttal/logs/corrected_inverse_shard_${shard_index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "Corrected inverse-cone run failed; inspect shard logs."
  exit 1
fi

python -m rebuttal.merge_shards \
  --input-dir "$OUTPUT_ROOT" \
  --num-shards "$NUM_SHARDS" \
  --expected-count "$EXPECTED_COUNT" \
  --max-attempts "$MAX_ATTEMPTS"

aggregate_inputs=("$OUTPUT_ROOT/results.jsonl")
if [[ "$LIMIT" -eq 0 && -f results/rebuttal/native/results.jsonl ]]; then
  aggregate_inputs=(results/rebuttal/native/results.jsonl "${aggregate_inputs[@]}")
fi
python -m rebuttal.aggregate "${aggregate_inputs[@]}" \
  --output-dir results/rebuttal/corrected_frontier_summary

echo "Corrected inverse-cone N=$MAX_ATTEMPTS run complete."
