#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/rebuttal

PID_FILE="$ROOT/results/rebuttal/corrected_cone_run.pid"
LOG_FILE="$ROOT/results/rebuttal/corrected_cone_run.log"

if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$previous_pid" ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "A corrected-cone run is already active (PID $previous_pid)."
    exit 2
  fi
fi

nohup bash -c '
  set -euo pipefail
  root="$1"
  bash "$root/cloud/bootstrap.sh"
  CUDA_VISIBLE_DEVICES=0 bash "$root/cloud/train_corrected_cones.sh"
  CONE_ARMS_ROOT=results/rebuttal/smoke/corrected_cone_arms \
    LIMIT=2 MAX_ATTEMPTS=1 \
    bash "$root/cloud/run_cone_ablation_4gpu.sh"
  MAX_ATTEMPTS="${CONE_ABLATION_ATTEMPTS:-1}" \
    bash "$root/cloud/run_cone_ablation_4gpu.sh"
  if [[ "${RUN_NATIVE_FRONTIER:-0}" == "1" ]]; then
    RUN_NATIVE=1 RUN_HLP=0 MAX_ATTEMPTS=32 \
      bash "$root/cloud/run_rebuttal_4gpu.sh"
  fi
  if [[ "${RUN_CORRECTED_INVERSE_N32:-0}" == "1" ]]; then
    MAX_ATTEMPTS=32 bash "$root/cloud/run_corrected_inverse_4gpu.sh"
  fi
' bash "$ROOT" >"$LOG_FILE" 2>&1 &
run_pid=$!
echo "$run_pid" >"$PID_FILE"

echo "Started corrected-cone pipeline as PID $run_pid"
echo "Follow progress with: tail -f $LOG_FILE"
