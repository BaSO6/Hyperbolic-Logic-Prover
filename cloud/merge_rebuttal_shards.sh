#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv-rebuttal/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-32}"
EXPECTED_COUNT="${EXPECTED_COUNT:-244}"

python -m rebuttal.merge_shards \
  --input-dir results/rebuttal/native \
  --num-shards "$NUM_SHARDS" \
  --expected-count "$EXPECTED_COUNT" \
  --max-attempts "$MAX_ATTEMPTS"
python -m rebuttal.merge_shards \
  --input-dir results/rebuttal/hlp \
  --num-shards "$NUM_SHARDS" \
  --expected-count "$EXPECTED_COUNT" \
  --max-attempts "$MAX_ATTEMPTS"
python -m rebuttal.aggregate \
  results/rebuttal/native/results.jsonl \
  results/rebuttal/hlp/results.jsonl

echo "Merged and summarized all shards."
