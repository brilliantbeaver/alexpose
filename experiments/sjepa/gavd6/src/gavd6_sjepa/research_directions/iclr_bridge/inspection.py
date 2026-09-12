"""Strictly no-fit inspection, separate from CPU model reconstruction."""
from pathlib import Path

from .cached_panel import _load_frozen
from .verification_supplement import expected_sealed_artifacts
from ..future_innovation.fi_contracts import read_json, sha256_file


def inspect_cached_panel(output_root):
    """Check frozen lineage and sealed digests; do not fit or load any model.

    This is file-integrity inspection only. It does not reconstruct predictions,
    score arithmetic, confidence intervals or candidate selection. Those checks
    belong to verify_panel_supplement, which performs read-only CPU refits.
    """
    root = Path(output_root)
    contract, _, _ = _load_frozen(root)
    seal = read_json(root / "reports/completion.json")
    if set(seal.get("artifacts", {})) != expected_sealed_artifacts(contract):
        raise ValueError("Unexpected sealed artifact inventory")
    for path, digest in seal["artifacts"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Sealed digest mismatch: {path}")
    return {"status": "integrity_checked", "read_only": True, "model_refits": False,
            "numerical_reconstruction_performed": False,
            "source_and_parent_digests_checked": True,
            "sealed_artifacts_checked": len(seal["artifacts"]),
            "scientific_status_as_saved": read_json(root / "reports/panel-report.json")["status"],
            "claim": "File integrity and lineage only; no prediction or score reconstruction."}
