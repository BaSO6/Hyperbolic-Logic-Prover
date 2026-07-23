#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p results/rebuttal

PID_FILE="$ROOT/results/rebuttal/cloud_run.pid"
LOG_FILE="$ROOT/results/rebuttal/cloud_run.log"

if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$previous_pid" ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "A rebuttal run is already active (PID $previous_pid)."
    echo "Log: $LOG_FILE"
    exit 2
  fi
fi

nohup bash "$ROOT/cloud/bootstrap_and_run.sh" >"$LOG_FILE" 2>&1 &
run_pid=$!
echo "$run_pid" >"$PID_FILE"

echo "Started rebuttal pipeline as PID $run_pid"
echo "Follow progress with: tail -f $LOG_FILE"
