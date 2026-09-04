from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_REVIEWS = (
    "ethics_determination",
    "data_use_review",
    "derived_pose_release_review",
)


def load_governance(path: str | Path) -> dict[str, Any]:
    status_path = Path(path)
    payload = json.loads(status_path.read_text())
    if payload.get("schema") != "neurips_laterality_governance/v1":
        raise ValueError(f"Unsupported governance schema in {status_path}")
    return payload


def submission_readiness(payload: dict[str, Any]) -> dict[str, Any]:
    unresolved: list[str] = []
    malformed: list[str] = []
    for key in REQUIRED_REVIEWS:
        entry = payload.get(key, {})
        if entry.get("status") != "resolved":
            unresolved.append(key)
        elif not entry.get("reference") or not entry.get("date"):
            malformed.append(key)
    return {
        "ready": not unresolved and not malformed,
        "unresolved": unresolved,
        "resolved_but_missing_reference_or_date": malformed,
        "rule": "All three reviews must be resolved and carry an internal reference and date.",
    }
