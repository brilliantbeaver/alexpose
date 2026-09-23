"""Small persistence helpers shared by local tutorials and HAIC workers."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def locked(path, *, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def code_identity(root):
    """Hash all imported project source and launch scripts, excluding caches."""
    root = Path(root)
    paths = list((root / 'src/gavd6_sjepa').rglob('*.py'))
    paths += list((root / 'src/gavd6_sjepa/research_directions/gait_fidelity').glob('*.json'))
    paths += list((root / 'slurm/gait-fidelity').glob('*.sh'))
    paths += list((root / 'slurm/gait-fidelity').glob('*.sbatch'))
    paths += list((root / 'slurm/gait-fidelity').glob('*.py'))
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths) if p.is_file()}


def verify_code(root, expected):
    mismatches = [name for name, h in expected.items()
                  if not (Path(root) / name).is_file() or sha256(Path(root) / name) != h]
    if mismatches:
        raise RuntimeError('Frozen experiment code changed: ' + ', '.join(mismatches[:8]))
