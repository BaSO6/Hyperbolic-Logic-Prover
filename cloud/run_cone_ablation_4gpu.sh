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

MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
LIMIT="${LIMIT:-0}"
CONE_ARMS_ROOT="${CONE_ARMS_ROOT:-results/rebuttal/cone_arms}"
EXPECTED_COUNT="$LIMIT"
if [[ "$EXPECTED_COUNT" -eq 0 ]]; then
  EXPECTED_COUNT=244
fi

available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$available_gpus" -lt 4 ]]; then
  echo "Need four visible GPUs for the four-arm ablation."
  exit 2
fi
if [[ ! -f results/rebuttal/corrected_cone/training_manifest.json ]]; then
  echo "Missing corrected cone artifacts; run cloud/train_corrected_cones.sh first."
  exit 2
fi
bash cloud/preflight.sh

arms=(
  corrected_distance
  paper_origin_forward
  corrected_apex_forward
  corrected_inverse
)
LOG_DIR="$CONE_ARMS_ROOT/logs"
STOP_FILE="$CONE_ARMS_ROOT/GPU_MONITOR_STOP"
mkdir -p "$LOG_DIR"
rm -f "$STOP_FILE"
python -m rebuttal.gpu_monitor \
  --output "$CONE_ARMS_ROOT/gpu_samples.csv" \
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
for gpu in 0 1 2 3; do
  arm="${arms[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python -m rebuttal.run_hlp_n32 \
      --mode "$arm" \
      --max-attempts "$MAX_ATTEMPTS" \
      --limit "$LIMIT" \
      --output "$CONE_ARMS_ROOT/$arm/results.jsonl" \
      >"$LOG_DIR/${arm}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one cone arm failed; inspect $LOG_DIR/*.log"
  exit 1
fi

inputs=()
for arm in "${arms[@]}"; do
  result="$CONE_ARMS_ROOT/$arm/results.jsonl"
  python -m rebuttal.validate_results "$result" \
    --expected-count "$EXPECTED_COUNT" \
    --max-attempts "$MAX_ATTEMPTS"
  inputs+=("$result")
done
python -m rebuttal.aggregate "${inputs[@]}" \
  --output-dir "$CONE_ARMS_ROOT/summary"

echo "Four-arm corrected cone ablation complete."
