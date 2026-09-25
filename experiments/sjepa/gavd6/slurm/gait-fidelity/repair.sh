#!/usr/bin/env bash
# Isolated extension: explicit work paths prevent accidentally using the core run.
set -euo pipefail
gf_repair_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$gf_repair_dir/common.sh"
usage() {
  cat <<'EOF'
Usage: bash slurm/gait-fidelity/repair.sh COMMAND WORK [OPTIONS]

setup WORK --response-work COMPLETED_RESPONSE_WORK
  Initialize a new immutable repair extension. Requires a new code release.
fixture WORK
  Run the complete tiny CPU software test, without Slurm or human evidence.
preflight|status|verify|exposure-audit|audit-expansion WORK
launch WORK --stage development|benchmark|confirmation
  Submit one CPU coordinator; at most four GPU workers by default.
lock-confirmation WORK --exposure-ledger CSV --reviewed-by NAME --evidence TEXT
  Require documented original-test people and the fixed completed fit matrix.
evaluate WORK --split development
  Reconstruct development evaluation from retained prediction exports.

See START_REPAIR_03.md. The default cutoff is September 25, 2026, 08:00 Pacific.
EOF
}
[[ $# -gt 0 ]] || { usage; exit 2; }
case "$1" in -h|--help|help) usage; exit 0 ;; esac
[[ $# -ge 2 && "$2" != --* ]] || { usage >&2; exit 2; }
gf_action="$1"; gf_work="$2"; shift 2
if [[ "$gf_action" == setup || "$gf_action" == fixture ]]; then
  gf_python="${GF_PYTHON:-/hai/scratch/$USER/envs/synthetic-training-cu124/bin/python}"
  if [[ "$gf_action" == fixture && ! -x "$gf_python" ]]; then gf_python="$gf_code_root/.venv/bin/python"; fi
  [[ -x "$gf_python" ]] || { printf 'Set GF_PYTHON to the existing study interpreter; missing: %s\n' "$gf_python" >&2; exit 1; }
  export PYTHONPATH="$gf_code_root/src" PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  exec "$gf_python" -u -m gavd6_sjepa.research_directions.gait_fidelity.repair "$gf_action" --work "$gf_work" "$@"
fi
gf_load_session "$gf_work"
case "$gf_action" in
  launch|preflight|status|verify|exposure-audit|audit-expansion|lock-confirmation|evaluate)
    exec "$GF_PYTHON" -u -m gavd6_sjepa.research_directions.gait_fidelity.repair "$gf_action" --work "$GF_WORK" "$@" ;;
  *) usage >&2; exit 2 ;;
esac
