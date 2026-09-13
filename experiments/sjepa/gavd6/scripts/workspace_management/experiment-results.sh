#!/usr/bin/env bash
# Standalone navigation/migration tool; intentionally outside fitting fingerprints.
set -euo pipefail
GAVD6_ROOT="${GAVD6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export PYTHONPATH="$GAVD6_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${RESULTS_PYTHON:-$GAVD6_ROOT/.venv/bin/python}" \
  -m gavd6_sjepa.shared_infrastructure.result_catalog --project "$GAVD6_ROOT" "$@"
