#!/usr/bin/env python3
"""Build a self-contained HAIC installer without publishing or changing Git."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

PAYLOAD_PATHS = (
    'scripts/research_directions/synthetic_training_v2/full_experiment.py',
    'scripts/research_directions/synthetic_training_v2/expanded_inputs.py',
    'scripts/research_directions/synthetic_training_v2/expansion_results.py',
    'scripts/research_directions/synthetic_training_v2/check_results.py',
    'scripts/research_directions/synthetic_training_v2/postrun_checks.py',
    'scripts/research_directions/synthetic_training_v2/diagnostics/__init__.py',
    'scripts/research_directions/synthetic_training_v2/diagnostics/calibration.py',
    'scripts/research_directions/synthetic_training_v2/diagnostics/timing.py',
    'scripts/research_directions/synthetic_training_v2/diagnostics/plots.py',
    'slurm/synthetic-training-v2/full-experiment.sh',
    'slurm/synthetic-training-v2/full-experiment.sbatch',
    'slurm/synthetic-training-v2/postrun-checks.sbatch',
    'slurm/synthetic-training-v2/README.md',
)

INSTALLER = r'''#!/usr/bin/env bash
# Generated installer: exact file allowlist, dependency checks and backups.
# No Git mutations, scientific-source edits or job submissions.
set -euo pipefail
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"
cd "$STV2_ROOT"
"$STV2_PYTHON" - "$STV2_ROOT" "$STV2_WORK" <<'PY'
import base64, gzip, hashlib, json, os, shutil, sys, tempfile
from contextlib import ExitStack
from pathlib import Path

archive = base64.b64decode('__PAYLOAD__')
if hashlib.sha256(archive).hexdigest() != '__ARCHIVE_SHA256__':
    raise SystemExit('Installer payload checksum failed; no files changed.')
package = json.loads(gzip.decompress(archive))
root, pilot = [Path(value).resolve(strict=True) for value in sys.argv[1:]]
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'scripts/research_directions/synthetic_training_v2'))
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import code_identity, sha256_file
import haic

def validate(package):
    if code_identity(root) != package['scientific_code_identity']:
        raise SystemExit('HAIC scientific code differs from this installer. No files changed; preserve the pilot revision.')
    for name, expected in package['dependencies'].items():
        if sha256_file(root / name) != expected:
            raise SystemExit(f'HAIC dependency differs: {name}. No files changed.')
    decoded = {}
    for name, entry in package['files'].items():
        target = (root / name).resolve()
        if name not in package['allowed_paths'] or Path(name).is_absolute() or '..' in Path(name).parts or not target.is_relative_to(root):
            raise SystemExit(f'Unsafe payload path: {name}')
        component = root
        for part in Path(name).parts:
            component = component / part
            if component.is_symlink():
                raise SystemExit(f'Symlink in installation path: {component}; no files changed.')
        data = base64.b64decode(entry['content'])
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise SystemExit(f'Payload checksum failed: {name}')
        if name.endswith('.py'):
            compile(data, name, 'exec')
        if not target.exists() or target.read_bytes() != data:
            decoded[name] = data
    return decoded

with ExitStack() as locks:
    for name in ('automation-launch', 'automation', 'manage', 'full-expansion-manage'):
        locks.enter_context(haic.stage_lock(pilot, name))
    changed = validate(package)
    if changed:
        haic.settled(haic.state_for(pilot))
        # An old expansion must be terminal before any executable is replaced.
        jobs = []
        for path in pilot.parent.glob('*/request.json'):
            request = json.loads(path.read_text())
            if request.get('kind') != 'stv2-full-experiment-v1' or request.get('pilot_work') != str(pilot):
                continue
            submission = json.loads((path.parent / 'submission.json').read_text())
            if not submission.get('job_id'):
                raise SystemExit(f'Unresolved expansion submission: {path.parent}; no files changed.')
            jobs.append({'job_id': submission['job_id']})
        # CPU post-run diagnostics also pin these helper files. Their jobs are
        # deliberately outside the pilot's GPU ledger but still must finish.
        for path in (pilot / 'diagnostics').glob('*/submission.json'):
            submission = json.loads(path.read_text())
            if not submission.get('job_id'):
                raise SystemExit(f'Unresolved diagnostic submission: {path.parent}; no files changed.')
            jobs.append({'job_id': submission['job_id']})
        snapshot = haic.scheduler_snapshot(jobs)
        if any(snapshot.get(job['job_id'], {}).get('state') not in haic.TERMINAL for job in jobs):
            raise SystemExit('An expansion is active or awaiting accounting; no files changed.')
        backup = Path(tempfile.mkdtemp(prefix='full-experiment-install-backup-', dir=pilot))
        for name, data in changed.items():
            target = root / name
            if target.exists():
                saved = backup / name
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, saved)
        for name, data in changed.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=target.name + '.install-', dir=target.parent)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
            os.replace(temporary, target)
        (backup / 'installed-files.json').write_text(json.dumps(
            {name: package['files'][name]['sha256'] for name in changed}, indent=2) + '\n')
        print(f'Installed {len(changed)} files; previous copies: {backup}')
    else:
        print('Matching expansion files already installed; no files changed.')
print('FULL_EXPERIMENT_INSTALLED. Use the launch command in slurm/synthetic-training-v2/README.md.')
PY
'''


def build(output):
    from full_experiment import implementation_files
    from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import code_identity, sha256_file
    files = {}
    for name in PAYLOAD_PATHS:
        content = (ROOT / name).read_bytes()
        files[name] = dict(content=base64.b64encode(content).decode('ascii'),
                           sha256=hashlib.sha256(content).hexdigest())
    dependencies = {name: value for name, value in implementation_files().items() if name not in files}
    dependencies['docs/studies/synthetic-training-v2/protocol.md'] = sha256_file(
        ROOT / 'docs/studies/synthetic-training-v2/protocol.md')
    package = dict(schema=1, scientific_code_identity=code_identity(ROOT),
                   allowed_paths=list(PAYLOAD_PATHS), files=files, dependencies=dependencies)
    compressed = gzip.compress(json.dumps(package, sort_keys=True).encode(), mtime=0)
    text = INSTALLER.replace('__PAYLOAD__', base64.b64encode(compressed).decode('ascii')).replace(
        '__ARCHIVE_SHA256__', hashlib.sha256(compressed).hexdigest())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    print(f'Installer: {output.resolve()}\nSHA256: {hashlib.sha256(output.read_bytes()).hexdigest()}')
    return package


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=ROOT / 'artifacts/synthetic-training-v2/full-experiment-install-20260919.sh')
    build(parser.parse_args().output)
