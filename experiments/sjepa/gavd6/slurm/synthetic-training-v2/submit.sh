#!/usr/bin/env bash
# The same entry point works from any directory after loading a run's session.env.
set -euo pipefail
stv2_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$stv2_checkout/slurm/synthetic-training-v2/study.env"
[[ -x "$STV2_PYTHON" ]] || {
  echo "Study Python not found: $STV2_PYTHON" >&2
  echo "Set STV2_PYTHON to the existing verified study interpreter; see README.md." >&2
  exit 1
}
exec "$STV2_PYTHON" "$stv2_checkout/scripts/research_directions/synthetic_training_v2/haic.py" "$@"
