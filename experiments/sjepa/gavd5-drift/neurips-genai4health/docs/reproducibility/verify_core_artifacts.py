"""Read-only checks of protocol-v2 counts, hashes, parameters, and predictions.

Run with the experiment's Python environment from any working directory:
    python neurips-genai4health/docs/reproducibility/verify_core_artifacts.py

This script prints a verification summary. It does not train, extract poses, or
write files. It requires the existing local work/artifacts/real tree and the
project's evaluation_protocol module. It complements build_evidence.py by
recomputing registry/manifest digests and examining checkpoint tensor states.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "neurips-brain-body"))
from evaluation_protocol import (  # noqa: E402
    manifest_fingerprint,
    split_fingerprint,
    validate_split_registry,
)

ART = ROOT / "work/artifacts/real"
PROTOCOL = ART / "evaluation_protocol"
PREFIX = "sjepa_outer_fold_0_seed_42_jepa_vicreg"
LABELS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    manifest = pd.read_csv(PROTOCOL / "eligible_sequence_manifest.csv")
    registry = pd.read_csv(PROTOCOL / "source_split_registry.csv")
    contract = json.loads((PROTOCOL / "source_split_contract.json").read_text(encoding="utf-8"))
    validate_split_registry(registry, manifest)
    manifest_sha = manifest_fingerprint(manifest)
    split_sha = split_fingerprint(registry)
    require(manifest_sha == contract["manifest_sha256"], "Manifest digest mismatch")
    require(split_sha == contract["split_sha256"], "Split digest mismatch")

    cohort = {}
    for label, filename in (
        ("raw", "raw_sequence_manifest.csv"),
        ("metadata_public", "metadata_public_sequence_manifest.csv"),
        ("decoded", "eligible_sequence_manifest.csv"),
    ):
        frame = pd.read_csv(PROTOCOL / filename)
        if label == "decoded":
            frame = frame.loc[frame.decoded_frame_eligible.eq(True)]
        cohort[label] = {
            "sequences": len(frame),
            "sources": int(frame.video_id.nunique()),
            "annotated_frame_rows": int(frame.n_annotated_frames.sum()),
        }

    qc = pd.read_csv(PROTOCOL / "pose_qc_eligibility_outer_fold_0.csv")
    eligible = qc.loc[qc.pose_qc_eligible.eq(True)]
    cohort["pose_qc"] = {
        "sequences": len(eligible),
        "sources": int(eligible.video_id.nunique()),
        "annotated_frame_rows": int(eligible.frames.sum()),
    }
    roles = eligible.groupby("split_role").agg(
        sequences=("sequence_id", "size"), sources=("video_id", "nunique")
    )

    checkpoint_path = ART / "checkpoints" / f"{PREFIX}.pt"
    sidecar = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256(checkpoint_path)
    require(checkpoint_sha == sidecar["checkpoint_sha256"], "Checkpoint byte digest mismatch")
    require(sidecar["manifest_sha256"] == manifest_sha, "Checkpoint manifest mismatch")
    require(sidecar["split_sha256"] == split_sha, "Checkpoint split mismatch")
    # The existing local checkpoint contains tensors plus plain metadata.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["test_tensors_loaded"] is False, "Recorded test access during training")
    access = pd.DataFrame(checkpoint["tensor_access_log"])
    require(not access.role.eq("test").any(), "Test entries in training access log")
    role_ids = {key: set(values) for key, values in checkpoint["role_video_ids"].items()}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        require(not role_ids[first] & role_ids[second], f"Source overlap: {first}/{second}")
    parameter_counts = {
        name: sum(tensor.numel() for tensor in state.values())
        for name, state in checkpoint["model_state"].items()
    }

    stages = []
    for entry in sidecar["stage_checkpoints"]:
        # Resolve the local file by basename, not the producing machine's root.
        filename = entry["path"].replace("\\", "/").rsplit("/", 1)[-1]
        stage_path = ART / "checkpoints" / filename
        require(sha256(stage_path) == entry["sha256"], f"Stage digest mismatch: {filename}")
        stage = torch.load(stage_path, map_location="cpu", weights_only=True)
        stages.append({
            "stage": entry["name"],
            "selected_epoch_zero_based": stage["best_epoch"],
            "best_validation_objective": stage["best_validation_loss"],
        })

    readout_prefix = "readout_outer_fold_0_seed_42_jepa_vicreg"
    predictions = pd.read_csv(ART / f"{readout_prefix}_source_predictions.csv")
    saved_metrics = pd.read_csv(ART / f"{readout_prefix}_metrics.csv").set_index("lane")
    reconstructed = {}
    for lane, frame in predictions.groupby("lane"):
        require(set(frame.video_id) == role_ids["test"], f"Test source mismatch in {lane}")
        require(not frame.video_id.duplicated().any(), f"Duplicate test source in {lane}")
        values = {
            "accuracy": accuracy_score(frame.condition, frame.prediction),
            "balanced_accuracy": balanced_accuracy_score(frame.condition, frame.prediction),
            "macro_f1": f1_score(frame.condition, frame.prediction, labels=LABELS,
                                 average="macro", zero_division=0),
        }
        for key, value in values.items():
            require(np.isclose(value, saved_metrics.loc[lane, f"test_source_{key}"],
                               rtol=0, atol=1e-12), f"Metric mismatch: {lane}/{key}")
        reconstructed[lane] = values

    result = {
        "status": "all inspected checks passed",
        "manifest_sha256": manifest_sha,
        "split_sha256": split_sha,
        "checkpoint_sha256": checkpoint_sha,
        "cohort": cohort,
        "pose_qc_role_counts": roles.to_dict(orient="index"),
        "locked_legacy_geometry_caches": int((qc.in_locked_manifest.eq(True)
                                             & qc.resolution_safe_geometry.eq(False)).sum()),
        "component_parameter_counts": parameter_counts,
        "primary_optimized_parameters": sum(parameter_counts[key] for key in
                                             ("online_encoder", "predictor", "projector")),
        "stages": stages,
        "readout_metrics_reconstructed": reconstructed,
        "limitations": [
            "Checks saved artifacts; does not reproduce training.",
            "Digests bind only the fields/bytes actually hashed.",
            "No inference of historical test secrecy, person independence, or clinical validity.",
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
