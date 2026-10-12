#!/usr/bin/env bash
# Copy code into a new immutable release; never overwrite the running checkout.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash slurm/gait-fidelity/sync-to-haic.sh HOST:/absolute/GAVD6_ROOT [--apply]

The default previews a new release and copies nothing. --apply creates a unique
GAVD6_ROOT/releases/gait-fidelity/UTC-ID directory and copies source, launchers,
tutorials and protocol documents. Data, outputs, environments and caches remain
on HAIC. Use the printed release path for setup. No Git publication is required.
EOF
}
destination=''; apply=0
for arg in "$@"; do
  case "$arg" in
    --apply) apply=1 ;;
    --dry-run) apply=0 ;;
    -h|--help) usage; exit 0 ;;
    --*) usage >&2; exit 2 ;;
    *) [[ -z "$destination" ]] || { usage >&2; exit 2; }; destination="$arg" ;;
  esac
done
if [[ ! "$destination" =~ ^([[:alnum:]_][[:alnum:]_.-]*@)?[[:alnum:]][[:alnum:].-]*:/[[:alnum:]_./-]+$ ]]; then
  echo 'Use HOST:/absolute/GAVD6_ROOT with plain letters, digits, _, ., / and -.' >&2; exit 2
fi
remote_host="${destination%%:*}"; asset_root="${destination#*:}"; asset_root="${asset_root%/}"
case "/${asset_root#/}/" in /|*/../*|*/./*|*//*) echo 'Use a normalized absolute directory.' >&2; exit 2 ;; esac
gf_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
command -v python3 >/dev/null || { echo 'python3 is required to package the release.' >&2; exit 1; }
release_id="$(python3 -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])')"
release="$asset_root/releases/gait-fidelity/$release_id"
staging="$(mktemp -d "${TMPDIR:-/tmp}/gait-fidelity-release.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
python3 - "$gf_checkout" "$staging" "$release_id" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path

root, staged = map(Path, sys.argv[1:3])
trees = (
    'src/gavd6_sjepa', 'scripts/research_directions/synthetic_training',
    'scripts/research_directions/synthetic_training_v2',
    'slurm/synthetic-training', 'slurm/synthetic-training-v2', 'slurm/gait-fidelity',
    'notebooks/gait_fidelity', 'docs/studies/gait-fidelity',
    'docs/studies/synthetic-training-v2',
)
skip = {'__pycache__', '.ipynb_checkpoints', 'outputs', 'node_modules', 'wheels', 'profiles', 'settings'}
files = [root / 'pyproject.toml']
for tree in trees:
    folder = root / tree
    if not folder.is_dir():
        raise SystemExit(f'Required source directory is missing: {folder}')
    for path in folder.rglob('*'):
        rel = path.relative_to(root)
        if any(part.startswith('.') or part in skip for part in rel.parts):
            continue
        if path.suffix in {'.pyc', '.pyo', '.env'}:
            continue
        if path.is_symlink():
            raise SystemExit(f'Unexpected release symlink: {rel}')
        if path.is_file():
            files.append(path)
hashes = {}
for path in sorted(set(files)):
    rel = path.relative_to(root)
    target = staged / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    hashes[rel.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
(staged / 'gait-fidelity-release.json').write_text(json.dumps({
    'release_id': sys.argv[3], 'files': hashes,
    'contents': 'code and protocol only; no study data, results or environment',
}, indent=2) + '\n')
print(f'Packaged {len(hashes)} files with SHA256 identities.')
PY
printf 'Asset checkout: %s\nNew release: %s\n' "$asset_root" "$release"
if [[ "$apply" == 0 ]]; then echo 'Preview complete; add --apply to transfer a new release.'; exit 0; fi
command -v rsync >/dev/null || { echo 'rsync is required.' >&2; exit 1; }
# The validated path contains no shell metacharacters. mkdir refuses a collision.
ssh "$remote_host" "test -d '$asset_root' && mkdir -p '$asset_root/releases/gait-fidelity' && mkdir '$release'"
rsync -a --checksum "$staging/" "$remote_host:$release/"
printf '\nRelease copied. In the HAIC shell run:\n\n'
printf 'bash %q/slurm/gait-fidelity/run.sh setup %q/outputs/gait-fidelity/full-manifest-01 --root %q\n' "$release" "$asset_root" "$asset_root"
printf '\nKeep this release unchanged while its jobs are pending or running.\n'
