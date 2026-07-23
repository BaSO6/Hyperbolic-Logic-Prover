#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 USER@HOST:/absolute/remote/path"
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="$1"

rsync -az --info=progress2 \
  --exclude '.venv-rebuttal/' \
  --exclude 'models/' \
  --exclude 'third_party/' \
  --exclude 'results/' \
  --exclude 'benchmark_reports/' \
  --exclude 'miniF2F-main/' \
  --exclude '*.pdf' \
  "$ROOT/" "$DESTINATION/"

echo "Upload complete."
echo "SSH to the server and run: bash cloud/launch_huawei.sh"
