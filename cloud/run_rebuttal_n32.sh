#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv-rebuttal/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

MAX_ATTEMPTS="${MAX_ATTEMPTS:-32}"
LIMIT="${LIMIT:-0}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_HLP="${RUN_HLP:-1}"
STOP_FILE="$ROOT/results/rebuttal/GPU_MONITOR_STOP"

mkdir -p results/rebuttal
rm -f "$STOP_FILE"
python -m rebuttal.gpu_monitor \
  --output results/rebuttal/gpu_samples.csv \
  --stop-file "$STOP_FILE" &
monitor_pid=$!
cleanup() {
  touch "$STOP_FILE"
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

bash cloud/preflight.sh

if [[ "$RUN_NATIVE" == "1" ]]; then
  python -m rebuttal.run_native_n32 \
    --max-attempts "$MAX_ATTEMPTS" \
    --limit "$LIMIT"
fi

if [[ "$RUN_HLP" == "1" ]]; then
  python -m rebuttal.run_hlp_n32 \
    --max-attempts "$MAX_ATTEMPTS" \
    --max-steps 64 \
    --limit "$LIMIT"
fi

python -m rebuttal.aggregate
echo "Finished. Results: $ROOT/results/rebuttal/summary"
