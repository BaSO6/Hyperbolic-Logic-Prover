#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/cloud/bootstrap.sh"
bash "$ROOT/cloud/smoke_gpu.sh"
bash "$ROOT/cloud/run_rebuttal_n32.sh"
