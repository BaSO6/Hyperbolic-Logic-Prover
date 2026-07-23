#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv-rebuttal/bin/activate"
export PATH="$HOME/.elan/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

bash cloud/preflight.sh

python -m rebuttal.run_native_n32 \
  --max-attempts 1 \
  --limit 2 \
  --batch-problems 2 \
  --lean-workers 2 \
  --output results/rebuttal/smoke/native/results.jsonl

python -m rebuttal.run_hlp_n32 \
  --max-attempts 1 \
  --limit 2 \
  --max-steps 64 \
  --max-expansions 50 \
  --output results/rebuttal/smoke/hlp/results.jsonl

python -m rebuttal.aggregate \
  results/rebuttal/smoke/native/results.jsonl \
  results/rebuttal/smoke/hlp/results.jsonl \
  --output-dir results/rebuttal/smoke/summary

python -m rebuttal.estimate_runtime \
  results/rebuttal/smoke/native/results.jsonl \
  results/rebuttal/smoke/hlp/results.jsonl \
  --problems 244 \
  --attempts 32 \
  --output results/rebuttal/smoke/runtime_estimate.json

echo "GPU smoke test passed."
