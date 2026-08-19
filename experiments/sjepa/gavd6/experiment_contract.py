"""Small provenance helpers shared by the GAVD S-JEPA notebooks.

These helpers intentionally cover only the scientific identity of an
experiment: cohort membership, model state, and an explicitly chosen run.
They are not a general experiment-management framework.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np


def safe_id(value: str, field: str = "identifier") -> str:
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{field} must contain only letters, digits, '.', '_' or '-': {value!r}")
    return value


def run_id(environment_name: str, mode: str, smoke_default: str) -> str:
    """Return an explicit run ID; real analyses must never silently overwrite."""
    value = os.getenv(environment_name)
    if value:
        return safe_id(value, environment_name)
    if mode == "smoke":
        return safe_id(smoke_default, environment_name)
    raise ValueError(
        f"Set {environment_name} for real runs so outputs are not silently mixed or overwritten."
    )


def cohort_id(records) -> str:
    """Stable short ID from ordered sequence/video/cohort membership."""
    members = [
        {
            "sequence_id": str(record["sequence_id"]),
            "video_id": str(record["video_id"]),
            "cohort": str(record.get("cohort", "canonical")),
        }
        for record in records
    ]
    encoded = json.dumps(sorted(members, key=lambda item: item["sequence_id"]), separators=(",", ":"), sort_keys=True)
    return f"cohort-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"


def model_state_sha256(state_dict) -> str:
    """Hash tensor names, dtypes, shapes, and bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extraction_dir(root: Path, mode: str, cohort: str, extraction_run: str) -> Path:
    return Path(root) / mode / "cohorts" / safe_id(cohort, "cohort") / "runs" / safe_id(extraction_run, "extraction run")


def checkpoint_dir(root: Path, mode: str, cohort: str, state_hash: str, run: str) -> Path:
    return (Path(root) / mode / "cohorts" / safe_id(cohort, "cohort") /
            "checkpoints" / safe_id(state_hash, "state hash") / "runs" / safe_id(run, "run"))


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
