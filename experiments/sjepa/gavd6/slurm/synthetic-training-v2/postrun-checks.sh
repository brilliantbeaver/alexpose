#!/usr/bin/env bash
# One CPU allocation for all three diagnostics; status never submits work.
set -euo pipefail
stv2_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "$stv2_checkout/slurm/synthetic-training-v2/study.env"
[[ -x "$STV2_PYTHON" ]] || { echo 'Source the saved session.env with its verified Python.' >&2; exit 1; }
stv2_check="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stv2_check" in
  all|calibration|timing|curves)
    exec "$STV2_PYTHON" "$stv2_checkout/scripts/research_directions/synthetic_training_v2/postrun_checks.py" submit --checks "$stv2_check" "$@" ;;
  status)
    exec "$STV2_PYTHON" "$stv2_checkout/scripts/research_directions/synthetic_training_v2/postrun_checks.py" status "$@" ;;
  *) echo 'Usage: postrun-checks.sh [all|calibration|timing|curves|status]' >&2; exit 2 ;;
esac
