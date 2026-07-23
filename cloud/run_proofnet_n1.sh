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
  --dataset rebuttal/datasets/proofnet.jsonl \
  --split test \
  --expected-count 186 \
  --max-attempts "${MAX_ATTEMPTS:-1}" \
  --output results/rebuttal/proofnet/native/results.jsonl

python -m rebuttal.run_hlp_n32 \
  --dataset rebuttal/datasets/proofnet.jsonl \
  --split test \
  --expected-count 186 \
  --max-attempts "${MAX_ATTEMPTS:-1}" \
  --max-steps 64 \
  --max-expansions 50 \
  --output results/rebuttal/proofnet/hlp/results.jsonl

python -m rebuttal.aggregate \
  results/rebuttal/proofnet/native/results.jsonl \
  results/rebuttal/proofnet/hlp/results.jsonl \
  --output-dir results/rebuttal/proofnet/summary

echo "ProofNet-test run complete."
