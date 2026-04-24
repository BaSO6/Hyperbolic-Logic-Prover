#!/bin/bash
# Version: v20.0 (portable, no hardcoded paths)

# Resolve project root from this script's location
# Script lives at: <project_root>/src/system2/run_repl_wrapper.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MATHLIB_ROOT="$PROJECT_ROOT/data/mathlib4"
REPL_BIN="$PROJECT_ROOT/tools/repl/.lake/build/bin/repl"

if [ ! -f "$REPL_BIN" ]; then
    echo "{\"error\": \"REPL binary not found at $REPL_BIN\"}" >&2; exit 1
fi

echo "{\"info\": \"[Wrapper] v20.0 portable mode. PROJECT_ROOT=$PROJECT_ROOT\"}" >&2
cd "$MATHLIB_ROOT"
exec lake env "$REPL_BIN"