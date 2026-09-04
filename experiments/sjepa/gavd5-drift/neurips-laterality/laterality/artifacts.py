from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def checkpoint_path(artifact_root: Path, variant: str, fold: int, seed: int) -> Path:
    return artifact_root / "checkpoints" / variant / f"fold_{fold:02d}" / f"seed_{seed}.pt"


def evaluation_path(artifact_root: Path, variant: str, fold: int, seed: int) -> Path:
    return artifact_root / "evaluations" / variant / f"fold_{fold:02d}" / f"seed_{seed}.csv"


def ensure_empty_or_matching(directory: Path, context_digest: str) -> None:
    """Prevent protocol/profile/model contexts from sharing one artifact root."""
    marker = directory / "protocol_snapshot.json"
    if marker.exists():
        current = json.loads(marker.read_text())
        if current.get("context_digest") != context_digest:
            raise RuntimeError(
                f"Artifact root {directory} belongs to context "
                f"{current.get('context_digest')}; choose a new output directory"
            )


def initialize_artifact_root(
    directory: Path,
    protocol: dict[str, Any],
    protocol_digest: str,
    context_digest: str,
    profile: str,
) -> Path:
    """Bind an output directory to one immutable protocol before writing results."""
    ensure_empty_or_matching(directory, context_digest)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "protocol_snapshot.json"
    if not marker.exists():
        atomic_write_json(
            marker,
            {
                "schema": "neurips_laterality_protocol_snapshot/v2",
                "protocol_digest": protocol_digest,
                "context_digest": context_digest,
                "profile": profile,
                "protocol": protocol,
            },
        )
    return marker
