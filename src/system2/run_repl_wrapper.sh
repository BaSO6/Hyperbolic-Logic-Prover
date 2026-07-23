#!/bin/bash
# Version: v20.0 (portable, no hardcoded paths)

# Resolve project root from this script's location
# Script lives at: <project_root>/src/system2/run_repl_wrapper.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MATHLIB_ROOT="${HLP_MATHLIB_ROOT:-$PROJECT_ROOT/data/mathlib4}"
REPL_BIN="${HLP_REPL_BIN:-$MATHLIB_ROOT/.lake/packages/REPL/.lake/build/bin/repl}"

# Fall back to the repository-local REPL for legacy installations.  Rebuttal
# runs use DeepSeek-Prover-V1.5's pinned Mathlib and bundled REPL so both
# methods are checked by the same Lean environment.
if [ ! -f "$REPL_BIN" ]; then
    REPL_BIN="$PROJECT_ROOT/tools/repl/.lake/build/bin/repl"
fi

if [ ! -f "$REPL_BIN" ]; then
    echo "{\"error\": \"REPL binary not found at $REPL_BIN\"}" >&2; exit 1
fi

echo "{\"info\": \"[Wrapper] v20.0 portable mode. PROJECT_ROOT=$PROJECT_ROOT\"}" >&2
cd "$MATHLIB_ROOT"
exec lake env "$REPL_BIN"
