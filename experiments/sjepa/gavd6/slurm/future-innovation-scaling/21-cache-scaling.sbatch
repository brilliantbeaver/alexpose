#!/usr/bin/env bash
#SBATCH --job-name=fi-scale-cache
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
: "${GAVD6_ROOT:?}" "${FI_SCALING_ROOT:?}" "${FI_PARENT_ROOT:?}"
cd "$GAVD6_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"${FI_PYTHON:-.venv/bin/python}" scripts/research_directions/future_innovation/run_source_learning_curve.py cache \
  --run-root "$FI_SCALING_ROOT" --parent-root "$FI_PARENT_ROOT" --device cuda
"${FI_PYTHON:-.venv/bin/python}" scripts/research_directions/future_innovation/run_source_learning_curve.py audit-teacher \
  --run-root "$FI_SCALING_ROOT" --parent-root "$FI_PARENT_ROOT" --device cuda
"${FI_PYTHON:-.venv/bin/python}" scripts/research_directions/future_innovation/run_source_learning_curve.py plan \
  --run-root "$FI_SCALING_ROOT" --parent-root "$FI_PARENT_ROOT"
