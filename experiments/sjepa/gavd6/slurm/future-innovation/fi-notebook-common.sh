#!/usr/bin/env bash
# Share the existing interpreter, deterministic thread settings and run paths.
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
fi_notebook() {
  local number="$1"
  shift
  uv run --no-sync python \
    "$GAVD6_ROOT/scripts/research_directions/future_innovation/execute_future_innovation_notebook.py" \
    --notebook "$number" --mode execute --run-root "$FI_RUN_ROOT" "$@"
}
