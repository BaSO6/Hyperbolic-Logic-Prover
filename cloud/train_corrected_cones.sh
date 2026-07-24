#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv-rebuttal/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

EPOCHS="${CONE_EPOCHS:-200}"
EDGE_BATCH_SIZE="${CONE_EDGE_BATCH_SIZE:-32768}"
MAX_DIAGNOSTIC_QUERIES="${MAX_DIAGNOSTIC_QUERIES:-500}"

bash cloud/preflight.sh
python -m unittest tests.test_entailment_cones tests.test_rebuttal_common
python -m rebuttal.audit_corrected_cones
python -m rebuttal.train_corrected_cones \
  --epochs "$EPOCHS" \
  --edge-batch-size "$EDGE_BATCH_SIZE"
python -m rebuttal.diagnose_corrected_cones \
  --max-queries "$MAX_DIAGNOSTIC_QUERIES"
python -m rebuttal.audit_corrected_cones

echo "Corrected cone training and four-arm link diagnostics complete."
