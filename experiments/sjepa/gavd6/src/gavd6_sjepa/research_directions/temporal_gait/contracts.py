"""Atomic artifacts, exact byte identities, role gates and compatible resumes."""
from contextlib import contextmanager
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_bytes(path, payload.encode())


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if Path(temp).exists():
            Path(temp).unlink()


def write_once_json(path, value):
    if Path(path).exists():
        if read_json(path) != json.loads(json.dumps(value, allow_nan=False)):
            raise ValueError(f"Incompatible frozen artifact: {path}; use a new run root")
    else:
        atomic_json(path, value)


def atomic_npz(path, **arrays):
    import io
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError("Object/pickle arrays forbidden")
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    atomic_bytes(path, buf.getvalue())


def code_fingerprint():
    """Hash relevant actual files (including untracked/dirty), not merely HEAD.

    Enumeration is restricted to repository SOFTWARE directories, never data.
    """
    package = Path(__file__).resolve().parent
    checkout = package.parents[3]
    paths = list(package.glob("*.py"))
    for relative in ("scripts/research_directions/temporal_gait", "slurm/temporal-gait"):
        folder = checkout / relative
        paths.extend(p for p in folder.glob("*") if p.is_file() and p.suffix in {".py", ".sh", ".sbatch"})
    paths.extend((checkout / "notebooks/temporal_gait").glob("*.ipynb"))
    paths.extend(p for p in (checkout / "pyproject.toml", checkout / "uv.lock") if p.is_file())
    paths.extend(checkout / relative for relative in GOVERNING_DOCUMENTS)
    return {str(p.relative_to(checkout)): sha256_file(p) for p in sorted(set(paths))}


def runtime():
    import torch
    return {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "cuda": torch.version.cuda}


def input_identity(cfg):
    from .config import MANIFEST_FIELDS
    return {name: {"path": getattr(cfg, name), "sha256": sha256_file(getattr(cfg, name))} for name in MANIFEST_FIELDS if getattr(cfg, name)}


def freeze_run(cfg):
    cfg.validate(check_input_paths=True)
    configuration = cfg.scientific_dict()
    identity = {"config": configuration, "inputs": input_identity(cfg), "code": code_fingerprint(), "runtime": runtime()}
    with stage_lock(cfg.root, "freeze"):
        write_once_json(cfg.root / "config/resolved.json", configuration)
        write_once_json(cfg.root / "config/identity.json", identity)
        checkout = Path(__file__).resolve().parent.parents[3]
        for relative in GOVERNING_DOCUMENTS:
            original = checkout / relative
            snapshot = cfg.root / "config/governing-documents" / original.name
            content = original.read_bytes()
            if snapshot.exists() and snapshot.read_bytes() != content:
                raise ValueError("Frozen governing document differs; new run required")
            if not snapshot.exists():
                atomic_bytes(snapshot, content)
    return digest(identity)


def verify_run(cfg):
    recorded = read_json(cfg.root / "config/identity.json")
    configuration = cfg.scientific_dict()
    current = {"config": configuration, "inputs": input_identity(cfg), "code": code_fingerprint(), "runtime": runtime()}
    if current != recorded:
        raise ValueError("Resume rejected: configuration, manifest, relevant code or environment changed")
    for relative in GOVERNING_DOCUMENTS:
        snapshot = cfg.root / "config/governing-documents" / Path(relative).name
        if not snapshot.is_file() or sha256_file(snapshot) != recorded["code"][relative]:
            raise ValueError("Retained governing protocol snapshot changed or is missing")
    return digest(recorded)


GOVERNING_DOCUMENTS = ("docs/studies/temporal-gait/protocol/protocol.md",
                       "docs/studies/temporal-gait/development/decisions.md")


@contextmanager
def stage_lock(root, name):
    path = Path(root) / "locks/process" / (name.replace("/", "_") + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError(f"Stage already running: {name}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def receipt(cfg, stage, outputs, *, status="complete", roles=(), details=None):
    identity = verify_run(cfg)
    value = {"schema_version": cfg.schema_version, "stage": stage, "mode": cfg.mode, "status": status,
             "identity": identity, "source_partition": list(roles), "seed_policy": cfg.seeds,
             "outputs": {str(Path(p).resolve().relative_to(cfg.root)): sha256_file(p) for p in outputs},
             "details": details or {}}
    atomic_json(cfg.root / "receipts" / f"{stage}.json", value)
    return value


def verify_receipt(cfg, stage):
    item = read_json(cfg.root / "receipts" / f"{stage}.json")
    if item["identity"] != verify_run(cfg) or item["status"] != "complete":
        raise ValueError(f"Missing/incompatible/incomplete stage: {stage}")
    for relative, expected in item["outputs"].items():
        if sha256_file(cfg.root / relative) != expected:
            raise ValueError(f"Changed output: {relative}")
    return item


def require_roles(records, allowed):
    if not records or any(r["role"] not in allowed for r in records):
        raise ValueError(f"Role boundary requires {sorted(allowed)}")


def snapshot_worktree(output):
    """Non-mutating ledger of preexisting edits; not a clean-tree assertion."""
    result = subprocess.run(["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=True)
    atomic_json(output, {"status_porcelain": result.stdout, "note": "Unrelated changes remain user-owned."})
