#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEEPSEEK_COMMIT="2c4ba9119eef74d0d611f494261b2c5bae98c69a"
MATHLIB_COMMIT="2f65ba7f1a9144b20c8e7358513548e317d26de1"
ENV_DIR="$ROOT/.venv-rebuttal"

cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi
"$PYTHON_BIN" - <<'PY'
import sys
assert (3, 9) <= sys.version_info[:2] < (3, 12), (
    "Python 3.9–3.11 is required, got " + sys.version
)
PY

if [[ ! -d "$ENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.1
python -m pip install --no-build-isolation -r rebuttal/requirements-cloud.txt

if ! command -v elan >/dev/null 2>&1; then
  curl -sSfL https://elan.lean-lang.org/elan-init.sh -o /tmp/elan-init.sh
  sh /tmp/elan-init.sh -y --no-modify-path --default-toolchain none
fi
export PATH="$HOME/.elan/bin:$PATH"

mkdir -p third_party models
if [[ ! -d third_party/DeepSeek-Prover-V1.5/.git ]]; then
  git clone https://github.com/deepseek-ai/DeepSeek-Prover-V1.5.git \
    third_party/DeepSeek-Prover-V1.5
fi
git -C third_party/DeepSeek-Prover-V1.5 fetch origin "$DEEPSEEK_COMMIT"
git -C third_party/DeepSeek-Prover-V1.5 checkout "$DEEPSEEK_COMMIT"
git -C third_party/DeepSeek-Prover-V1.5 submodule update --init mathlib4
actual_mathlib="$(git -C third_party/DeepSeek-Prover-V1.5/mathlib4 rev-parse HEAD)"
if [[ "$actual_mathlib" != "$MATHLIB_COMMIT" ]]; then
  echo "Unexpected Mathlib commit: $actual_mathlib"
  exit 2
fi

if [[ ! -e data/mathlib4 ]]; then
  ln -s "$ROOT/third_party/DeepSeek-Prover-V1.5/mathlib4" data/mathlib4
elif [[ -L data/mathlib4 ]]; then
  target="$(readlink data/mathlib4)"
  if [[ "$target" != "$ROOT/third_party/DeepSeek-Prover-V1.5/mathlib4" ]]; then
    echo "data/mathlib4 points to unexpected target: $target"
    exit 2
  fi
else
  echo "data/mathlib4 already exists and is not a symlink; refusing to overwrite it."
  exit 2
fi

(
  cd third_party/DeepSeek-Prover-V1.5/mathlib4
  lake exe cache get
  lake build
)

if [[ ! -f models/DeepSeek-Prover-V1.5-RL/.download-complete ]]; then
  huggingface-cli download deepseek-ai/DeepSeek-Prover-V1.5-RL \
    --local-dir models/DeepSeek-Prover-V1.5-RL
  touch models/DeepSeek-Prover-V1.5-RL/.download-complete
fi
if [[ ! -f models/all-MiniLM-L6-v2/.download-complete ]]; then
  huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir models/all-MiniLM-L6-v2
  touch models/all-MiniLM-L6-v2/.download-complete
fi

echo "Bootstrap complete."
echo "Activate with: source $ENV_DIR/bin/activate"
