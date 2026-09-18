#!/usr/bin/env bash
# Synchronize only the dedicated study environment; never the root project.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash slurm/synthetic-training/setup-environment.sh [--check] [--require-cuda]

Set ST_PYTHON to an absolute .../bin/python path in a dedicated environment.
Default: on a Slurm compute node with CUDA_HOME pointing at toolkit 12.4,
create/reuse Python 3.11, build MMCV for H100, sync the study lockfile, and
verify imports and small CPU operations. See README Step 4 for the batch job.
--check         Check the installed lock and runtime without syncing packages.
--require-cuda  Also require working CUDA operations (use in a GPU allocation).
EOF
}
fail() { echo "Synthetic-training setup: $*" >&2; exit 1; }
check_only=0
require_cuda=0
for option in "$@"; do
  case "$option" in
    --check) check_only=1 ;;
    --require-cuda) require_cuda=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; fail "Unknown option: $option" ;;
  esac
done

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$project_dir/../.." && pwd -P)"
[[ -z "${GAVD6_ROOT:-}" || "$(cd "$GAVD6_ROOT" && pwd -P)" == "$repo_root" ]] ||
  fail "GAVD6_ROOT does not match this checkout: $repo_root"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
  fail "This lock targets Linux x86_64 (HAIC), not this host. No environment was changed."
[[ -n "${ST_PYTHON:-}" ]] || fail "Export ST_PYTHON, for example /hai/scratch/\$USER/envs/synthetic-training-cu124/bin/python."
case "$ST_PYTHON" in
  /*/bin/python|/*/bin/python3|/*/bin/python3.11) ;;
  *) fail "ST_PYTHON must be an absolute path ending in /bin/python (or python3/python3.11), not an environment directory." ;;
esac
for program in uv realpath flock; do
  command -v "$program" >/dev/null || fail "Required command '$program' is missing. See Step 2 of slurm/synthetic-training/README.md."
done
env_root="$(realpath -m "$(dirname "$(dirname "$ST_PYTHON")")")"
root_environment="$(realpath -m "$repo_root/.venv")"
case "$env_root" in
  /|/usr|/usr/local|/opt|"$HOME"|"$repo_root"|"$project_dir"|"$root_environment")
    fail "Choose a dedicated study environment; refusing to synchronize $env_root." ;;
esac
ST_PYTHON="$env_root/bin/$(basename "$ST_PYTHON")"
export ST_PYTHON
[[ -f "$project_dir/uv.lock" ]] || fail "Study uv.lock is missing. Update the checkout; setup will not create a new lock."

# Old manual-install exports must not disable the manifest or select other
# package sources. Keep cache/proxy/TLS/offline preferences available to HAIC.
unset UV_CONSTRAINT UV_OVERRIDE UV_BUILD_CONSTRAINT UV_NO_CONFIG UV_CONFIG_FILE
unset UV_PROJECT UV_WORKING_DIR UV_PROJECT_ENVIRONMENT UV_PYTHON UV_SYSTEM_PYTHON
unset UV_INDEX_URL UV_EXTRA_INDEX_URL UV_INDEX UV_DEFAULT_INDEX UV_FIND_LINKS UV_NO_INDEX
unset UV_NO_SOURCES UV_NO_SOURCES_PACKAGE UV_TORCH_BACKEND UV_EXCLUDE_NEWER
unset UV_NO_BUILD_ISOLATION UV_NO_BUILD UV_NO_BINARY UV_FROZEN
unset UV_ISOLATED UV_NO_PROJECT UV_NO_SYNC UV_NO_INSTALL_PACKAGE UV_NO_INSTALL_LOCAL
unset UV_NO_INSTALL_PROJECT UV_NO_INSTALL_WORKSPACE UV_NO_BINARY_PACKAGE UV_NO_BUILD_PACKAGE
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
export UV_PROJECT_ENVIRONMENT="$env_root"
export PYTHONNOUSERSITE=1

uv_version="$(uv --version)"
read -r _ version_number _ <<< "$uv_version"
IFS=. read -r uv_major uv_minor uv_patch <<< "$version_number"
[[ "$uv_major" == 0 && "$uv_minor" == 12 && "$uv_patch" =~ ^[0-9]+$ && "$uv_patch" -ge 15 ]] ||
  fail "Use uv >=0.12.15,<0.13 (tested: 0.12.15); found $uv_version. See Step 2 of the README."

if [[ -e "$env_root" ]]; then
  [[ -f "$env_root/pyvenv.cfg" && -x "$ST_PYTHON" ]] ||
    fail "$env_root exists but is not a usable virtual environment. Choose a fresh ST_PYTHON path; no files were removed."
  "$ST_PYTHON" -c '
import pathlib, sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This study requires Python 3.11. Choose a fresh ST_PYTHON path; keep the existing environment.")
if sys.prefix == sys.base_prefix or pathlib.Path(sys.prefix).resolve() != pathlib.Path(sys.argv[1]).resolve():
    raise SystemExit("ST_PYTHON does not belong to the declared virtual environment.")
' "$env_root"
elif [[ "$check_only" == 1 ]]; then
  fail "Environment does not exist: $env_root. Submit setup-environment.sbatch as described in README Step 4."
fi

if [[ "$check_only" == 0 ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] ||
    fail "MMCV must be built on a compute node. Submit setup-environment.sbatch as described in README Step 4."
  [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] ||
    fail "CUDA 12.4 compiler unavailable. Use setup-environment.sbatch; it prepares the toolkit before setup."
  "$CUDA_HOME/bin/nvcc" --version | grep -Eq 'release 12\.4,' ||
    fail "CUDA_HOME must contain toolkit 12.4 to match torch==2.6.0+cu124."
  command -v c++ >/dev/null || fail "A C++ compiler is required to build MMCV."
  export MAX_JOBS="${ST_BUILD_JOBS:-4}"
  [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || fail "ST_BUILD_JOBS must be a positive integer."
fi

mkdir -p "$(dirname "$env_root")"
# flock releases automatically on exit, including a failed or interrupted sync.
# Keep its inode in place so all writers lock the same file.
exec 9> "$env_root.synthetic-training-setup.lock"
flock --nonblock 9 || fail "Another setup/check is using $env_root. Retry after it finishes."

echo "Study manifest: $project_dir/pyproject.toml"
echo "Study Python:   $ST_PYTHON"
if [[ "$check_only" == 1 ]]; then
  uv sync --project "$project_dir" --locked --check --offline --python "$ST_PYTHON"
else
  command -v git >/dev/null || fail "git is required to install the pinned source revisions."
  python_request=3.11
  [[ ! -x "$ST_PYTHON" ]] || python_request="$ST_PYTHON"
  uv sync --project "$project_dir" --locked --python "$python_request"
fi
uv pip check --python "$ST_PYTHON"
checker_command=("$ST_PYTHON" "$repo_root/scripts/research_directions/synthetic_training/check_environment.py")
[[ "$require_cuda" == 0 ]] || checker_command+=(--require-cuda)
[[ "$check_only" == 1 ]] || checker_command+=(--json-output "$env_root/synthetic-training-environment.json")
"${checker_command[@]}"
echo "Study package checks passed. Continue with the model/assets steps in the README."
