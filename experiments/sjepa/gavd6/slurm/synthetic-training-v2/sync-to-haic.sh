#!/usr/bin/env bash
# Run on the Mac before initializing a new experiment. Never copies run outputs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash slurm/synthetic-training-v2/sync-to-haic.sh USER@HOST:/absolute/checkout [--apply]

Preview is the default. Add --apply to transfer the listed files to an existing
HAIC checkout. Historical files are verified locally first; replaced remote
files are backed up under outputs/synthetic-training-v2/sync-backup-<unique-id>.
Existing study profiles, run outputs, environments, and caches are not copied.
Run this before initialization: code updates change an experiment's identity.
EOF
}
destination=''
apply=0
for argument in "$@"; do
  case "$argument" in
    --apply) apply=1 ;;
    --dry-run) apply=0 ;;
    -h|--help) usage; exit 0 ;;
    --*) usage >&2; exit 2 ;;
    *) [[ -z "$destination" ]] || { usage >&2; exit 2; }; destination="$argument" ;;
  esac
done
# rsync forwards a remote path to a shell. Restrict it to plain path characters;
# do not interpolate shell expressions or use eval to construct remote commands.
if [[ ! "$destination" =~ ^([[:alnum:]_][[:alnum:]_.-]*@)?[[:alnum:]][[:alnum:].-]*:/[[:alnum:]_./-]+$ ]]; then
  echo 'Use USER@HOST:/absolute/checkout with letters, digits, _, ., /, and - only.' >&2
  exit 2
fi
remote_root="${destination#*:}"
remote_root="${remote_root%/}"
case "/${remote_root#/}/" in
  /|*/../*|*/./*|*//*) echo 'Use a normalized absolute checkout path, not / or a path with . or .. components.' >&2; exit 2 ;;
esac
destination="${destination%%:*}:$remote_root"
command -v python3 >/dev/null || { echo 'python3 is required on the Mac to verify historical hashes.' >&2; exit 1; }
command -v rsync >/dev/null || { echo 'rsync is required on the Mac.' >&2; exit 1; }
stv2_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
file_list="$(mktemp "${TMPDIR:-/tmp}/stv2-sync.XXXXXX")"
trap 'rm -f "$file_list"' EXIT

# Validate the complete preservation manifest before making any remote call.
# A single explicit list prevents accidental transfer of ignored output trees.
python3 - "$stv2_checkout" "$file_list" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

root, listing = map(Path, sys.argv[1:])
manifest = root / 'docs/studies/synthetic-training-v2/preservation-manifest.json'
frozen = json.loads(manifest.read_text())['files']
if not isinstance(frozen, dict) or not frozen:
    raise SystemExit('The historical preservation manifest must name its required files.')
files = set()
failures = []
for name, expected in frozen.items():
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\n' in name or '\0' in name:
        raise SystemExit(f'Unsafe historical path: {name!r}')
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        failures.append(f'Missing or unsafe historical file: {name}')
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(f'Historical SHA256 mismatch: {name}')
    files.add(name)
if failures:
    raise SystemExit('\n'.join(failures) + '\nNo files were transferred. Restore the verified local evidence first.')

trees = (
    'src/gavd6_sjepa',
    'scripts/research_directions/synthetic_training',
    'scripts/research_directions/synthetic_training_v2',
    'slurm/synthetic-training',
    'slurm/synthetic-training-v2',
    'docs/studies/synthetic-training-v2',
    'notebooks/synthetic_training_v2',
)
skip_directories = {'__pycache__', 'wheels', 'outputs', 'settings', 'profiles', 'node_modules'}
for name in trees:
    folder = root / name
    if not folder.is_dir():
        raise SystemExit(f'Required study directory is missing: {name}. No files were transferred.')
    for path in folder.rglob('*'):
        relative = path.relative_to(root)
        if any(part.startswith('.') or part in skip_directories for part in relative.parts):
            continue
        if path.suffix in {'.pyc', '.pyo', '.env'} and relative.as_posix() != 'slurm/synthetic-training-v2/study.env':
            continue
        if path.is_symlink():
            raise SystemExit(f'Unexpected symlink in study files: {relative}. No files were transferred.')
        if path.is_file():
            files.add(relative.as_posix())
listing.write_bytes(b''.join(name.encode() + b'\0' for name in sorted(files)))
print(f'Historical hashes verified: {len(frozen)}. Study files selected: {len(files)}.')
PY

backup_id="$(python3 -c 'from datetime import datetime, timezone; import uuid; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12])')"
backup_dir="$remote_root/outputs/synthetic-training-v2/sync-backup-$backup_id"
options=(-avR --checksum --itemize-changes --backup "--backup-dir=$backup_dir" --from0 "--files-from=$file_list")
if [[ "$apply" == 0 ]]; then
  options+=(--dry-run)
  echo 'Preview only. Add --apply to transfer these files.'
else
  echo "Backup of replaced remote files: $backup_dir"
fi
cd "$stv2_checkout"
rsync "${options[@]}" ./ "$destination/"
