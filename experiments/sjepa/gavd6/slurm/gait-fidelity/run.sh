#!/usr/bin/env bash
# One entry point for both the HAIC study and the local software fixture.
set -euo pipefail
gf_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$gf_script_dir/common.sh"
usage() {
  cat <<'EOF'
Usage: bash slurm/gait-fidelity/run.sh COMMAND [WORK] [OPTIONS]

setup       Initialize a full-manifest AMASS cohort and saved HAIC asset paths.
preflight   Check the saved paths, dependencies and source identity on CPU.
cohort      Read the frozen AMASS selection, exclusions and locked test inventory.
plan        Read the selected core/full experiment plan and shared dependencies.
gavd-plan   Inventory GAVD, group related recordings and save train/dev/test roles.
gavd-evaluate  Apply saved restorers and evaluate real-video observational outcomes.
gavd-lock   Freeze selected checkpoints and reviewed GAVD confirmation exposure.
launch      Submit one CPU coordinator, which dispatches at most eight GPU jobs.
status      Show the coordinator submission and experiment progress.
report      Generate/read the current evidence-labelled report.
verify      Reconstruct and verify completed experiment artifacts.
fixture     Run the small software fixture locally, without Slurm or CUDA.

WORK defaults to GF_WORK, or ASSET_ROOT/outputs/gait-fidelity/study-01.
setup accepts --cohort-preset named_walking|treadmill_walking|all_eligible|reviewed,
  --num-shards N, --experiment-set core|full, and explicit asset/review CSV options.
launch accepts --max-jobs 8 --account mind --partition hai --controller-hours 72.
Add --prepare-only to launch when you want to inspect the shared data before fits.
Use launch --gavd-only after gavd-plan to extract real training/development videos.
GAVD and AMASS stages share one coordinator, eight-GPU cap and allocation budget.
Use a new WORK for a new protocol. Resume by repeating launch for the same WORK.
EOF
}
[[ $# -gt 0 ]] || { usage; exit 2; }
gf_action="$1"; shift
case "$gf_action" in -h|--help|help) usage; exit 0 ;; esac
gf_asset_root="${GAVD6_ROOT:-/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6}"
gf_work="${GF_WORK:-$gf_asset_root/outputs/gait-fidelity/study-01}"
gf_explicit_work=0
if [[ $# -gt 0 && "$1" != --* ]]; then gf_work="$1"; shift; gf_explicit_work=1; fi
if [[ "$gf_action" == setup ]]; then
  gf_fixture=0
  for gf_arg in "$@"; do [[ "$gf_arg" != --fixture ]] || gf_fixture=1; done
  if [[ "$gf_fixture" == 1 && ! -d "$gf_asset_root" ]]; then
    gf_asset_root="$gf_code_root"
    if [[ "$gf_explicit_work" == 0 && -z "${GF_WORK:-}" ]]; then gf_work="$gf_code_root/outputs/gait-fidelity/study-01"; fi
  fi
  gf_session="$gf_asset_root/outputs/synthetic-training-v2/source-smoke-01/session.env"
  if [[ -f "$gf_session" && "$gf_fixture" == 0 ]]; then source "$gf_session"; fi
  gf_python="${GF_PYTHON:-${STV2_PYTHON:-/hai/scratch/$USER/envs/synthetic-training-cu124/bin/python}}"
  if [[ "$gf_fixture" == 1 && ! -x "$gf_python" ]]; then gf_python="$gf_code_root/.venv/bin/python"; fi
  [[ -x "$gf_python" ]] || { printf 'Interpreter missing: %s\nSet GF_PYTHON to an existing compatible interpreter.\n' "$gf_python" >&2; exit 1; }
  export PYTHONPATH="$gf_code_root/src" PYTHONNOUSERSITE=1
  exec "$gf_python" -m gavd6_sjepa.research_directions.gait_fidelity init --work "$gf_work" --root "$gf_asset_root" "$@"
fi
gf_load_session "$gf_work"
case "$gf_action" in
  launch)
    "$GF_PYTHON" -m gavd6_sjepa.research_directions.gait_fidelity preflight --work "$GF_WORK"
    exec "$GF_PYTHON" "$GF_ROOT/slurm/gait-fidelity/submit.py" --work "$GF_WORK" "$@"
    ;;
  fixture)
    exec "$GF_PYTHON" -m gavd6_sjepa.research_directions.gait_fidelity run --work "$GF_WORK" --local "$@"
    ;;
  status)
    if [[ -f "$GF_WORK/control/coordinator.json" ]]; then cat "$GF_WORK/control/coordinator.json"; printf '\n'; fi
    exec "$GF_PYTHON" -m gavd6_sjepa.research_directions.gait_fidelity status --work "$GF_WORK" "$@"
    ;;
  preflight|plan|cohort|report|verify|gavd-plan|gavd-evaluate|gavd-lock|lock-confirmation|evaluate-confirmation)
    exec "$GF_PYTHON" -m gavd6_sjepa.research_directions.gait_fidelity "$gf_action" --work "$GF_WORK" "$@"
    ;;
  *) usage >&2; exit 2 ;;
esac
