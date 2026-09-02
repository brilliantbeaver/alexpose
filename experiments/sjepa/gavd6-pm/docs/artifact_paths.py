"""Resolve the artifact root the same way the notebooks do.

Kept separate from make_figures so that tools which need the paths, such as the ledger refresher, can
import them without triggering make_figures' strict input validation at import time.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve_artifacts() -> Path:
    """Return the artifact directory for the active mode.

    The notebooks read GAVD_ARTIFACT_DIR from experiments/sjepa/gavd6-pm/.env and append the mode. This
    module used to be a hardcoded path inside make_figures that no active run wrote to, so figure builds
    failed before drawing anything. Resolution order is the configured root first, then the in-tree work
    directory, then the legacy cache directory. A candidate counts only if it holds a classifier contract.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        load_dotenv(ROOT.parents[2] / ".env", override=False)
    except Exception:
        pass
    mode = os.getenv("GAVD_MODE", "real").strip().lower()
    configured = os.getenv("GAVD_ARTIFACT_DIR")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser() / mode)
    candidates.append(ROOT / "work" / "artifacts" / mode)
    candidates.append(ROOT / "cache" / "artifacts" / mode)
    for candidate in candidates:
        if (candidate / "classifier_contract.json").is_file():
            return candidate
    raise FileNotFoundError(
        "No artifact root holds classifier_contract.json. Run notebooks 04 through 06 first, or set "
        "GAVD_ARTIFACT_DIR in experiments/sjepa/gavd6-pm/.env. Checked:\n  "
        + "\n  ".join(str(candidate) for candidate in candidates)
    )


def load_contract(artifacts: Path) -> dict:
    path = artifacts / "classifier_contract.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run notebooks 04 through 06 before building figures: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not contract.get("curriculum_complete"):
        raise RuntimeError("Classifier contract does not name a complete curriculum")
    expected = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
    if contract.get("conditions_seen") != expected:
        raise RuntimeError(f"Unexpected curriculum order: {contract.get('conditions_seen')}")
    return contract
