#!/usr/bin/env bash
# Thin entry point; the Python coordinator quotes arguments and records the DAG.
set -euo pipefail
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export GAVD6_ROOT
source "$GAVD6_ROOT/slurm/temporal-gait/common.sh"
exec "$TG_PYTHON" "$GAVD6_ROOT/scripts/research_directions/temporal_gait/submit.py" "$@"
