#!/usr/bin/env bash
# The original session supplies the Python interpreter, account and asset paths.
set -euo pipefail
: "${STV2_ROOT:?Source the completed pilot session.env first}"
: "${STV2_PYTHON:?Source the completed pilot session.env first}"
: "${STV2_WORK:?Source the completed pilot session.env first}"
cd "$STV2_ROOT"
exec "$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/full_experiment.py "$@"
