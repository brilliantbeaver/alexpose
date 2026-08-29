#!/usr/bin/env python3
"""Verify that the historical AMASS Core11 checkpoint can be safely probed.

This program is intentionally a compatibility and integrity gate, not a
downstream evaluator.  It never writes the checkpoint, imports only the exact
historical source revision, and produces a small JSON sidecar describing the
verified load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HISTORICAL_COMMIT = "c2385c125d41698451486cc7cc30f1892d3ec773"
CHECKPOINT_SHA256 = "4741bbbcdb4ba16cf0b75a798b4f7e1fea2c5aeccf41da17250822c3ebd7000b"
SOURCE_SHA256 = {
    "amass_core11_jepa.py": "6dd218b3e26cc09a287ae928ded035d830274185ddf6d5979016c2579dac2886",
    "gait_parity_jepa.py": "2ffce2fbcdc41a8dc85e562807db4445dd1cfefd6f769969aa3bf3a6fe8ab949",
}
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_ROOT = Path("/Users/theodoremui/dev/alexpose-amass-core11-c2385c1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    require(isinstance(payload, dict), f"Expected a JSON object in {path}")
    return payload


def verify_historical_source(legacy_root: Path) -> Path:
    legacy_root = legacy_root.expanduser().resolve()
    require(legacy_root.is_dir(), f"Historical worktree does not exist: {legacy_root}")
    require(
        git_output(legacy_root, "rev-parse", "HEAD") == HISTORICAL_COMMIT,
        "Historical worktree is not at the checkpoint's recorded commit",
    )
    require(
        not git_output(legacy_root, "status", "--porcelain"),
        "Historical worktree is dirty; use an untouched detached worktree",
    )
    source_dir = legacy_root / "experiments" / "sjepa" / "gavd6"
    for name, expected_hash in SOURCE_SHA256.items():
        path = source_dir / name
        require(path.is_file(), f"Missing historical source file: {path}")
        observed_hash = sha256_file(path)
        require(
            observed_hash == expected_hash,
            f"Historical source hash mismatch for {name}: {observed_hash}",
        )
    return source_dir


def feature_summary(features: Any, torch_module: Any) -> dict[str, Any]:
    branches = tuple(features) if isinstance(features, (tuple, list)) else (features,)
    require(branches, "EMA encoder returned no feature branches")
    require(
        all(isinstance(branch, torch_module.Tensor) for branch in branches),
        "EMA encoder returned a non-tensor feature branch",
    )
    return {
        "feature_branch_count": len(branches),
        "feature_shapes": [list(branch.shape) for branch in branches],
        "features_finite": all(bool(torch_module.isfinite(branch).all()) for branch in branches),
    }


def write_report(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing compatibility report: {path}. "
            "Pass --overwrite-report to replace this generated sidecar."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help="Detached historical worktree at commit c2385c1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "outputs" / "archive" / "run1" / "seed-7_standard.pt",
        help="Immutable historical checkpoint to verify.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=PROJECT_DIR / "outputs" / "archive" / "run1" / "run_config.json",
        help="Run provenance sidecar written with the checkpoint.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_DIR / "outputs" / "archive" / "run1" / "pilot_probe_c2385c1" / "compatibility_report.json",
        help="New JSON report path. The checkpoint is never written.",
    )
    parser.add_argument(
        "--overwrite-report",
        action="store_true",
        help="Replace only an existing generated JSON report.",
    )
    arguments = parser.parse_args()

    checkpoint = arguments.checkpoint.expanduser().resolve()
    run_config_path = arguments.run_config.expanduser().resolve()
    require(checkpoint.is_file(), f"Checkpoint does not exist: {checkpoint}")
    require(run_config_path.is_file(), f"Run config does not exist: {run_config_path}")
    checkpoint_hash_before = sha256_file(checkpoint)
    require(
        checkpoint_hash_before == CHECKPOINT_SHA256,
        f"Checkpoint SHA-256 mismatch: {checkpoint_hash_before}",
    )
    run_config = load_json(run_config_path)
    require(
        run_config.get("code_sha256") == SOURCE_SHA256,
        "Run config does not name the expected historical source hashes",
    )

    source_dir = verify_historical_source(arguments.legacy_root)
    sys.path.insert(0, str(source_dir))
    import torch
    from gait_parity_jepa import TrainConfig, VICRegProjector, build_model, lift_orbit

    require(
        Path(sys.modules["gait_parity_jepa"].__file__).resolve() == source_dir / "gait_parity_jepa.py",
        "Imported gait_parity_jepa is not the historical source file",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(
        set(payload) == {"metadata", "model_state", "optimizer_state", "projector_state"},
        "Checkpoint has an unexpected top-level payload schema",
    )
    metadata = payload["metadata"]
    require(metadata["variant"] == "standard", "This pilot loader is for seed-7_standard.pt")
    require(metadata["seed"] == 7, "This pilot loader is for seed 7")
    require(metadata["optimizer_updates"] == 248_600, "Unexpected attempted-update count")
    require(metadata["runtime"]["code_sha256"] == SOURCE_SHA256, "Checkpoint code hashes differ")

    config = TrainConfig(**metadata["train_config"])
    model = build_model(config, metadata["variant"], metadata["seed"])
    model.load_state_dict(payload["model_state"], strict=True)
    projector = VICRegProjector(config.embed_dim)
    projector.load_state_dict(payload["projector_state"], strict=True)
    require(
        not any(parameter.requires_grad for parameter in model.target_encoder.parameters()),
        "Historical EMA target encoder is unexpectedly trainable",
    )
    model.eval()
    canonical = torch.zeros(1, config.frames, config.joints, 3)
    orbit = lift_orbit(canonical, config.mirror_pairs, config.mirror_channel)
    with torch.no_grad():
        features = model.target_encoder(orbit)
    feature_info = feature_summary(features, torch)
    require(feature_info["features_finite"], "Historical EMA encoder emitted non-finite pilot features")

    optimizer_steps = sorted(
        {
            int(state["step"])
            for state in payload["optimizer_state"]["state"].values()
            if "step" in state
        }
    )
    checkpoint_hash_after = sha256_file(checkpoint)
    require(
        checkpoint_hash_after == checkpoint_hash_before,
        "Checkpoint changed during verification; refusing to report success",
    )
    report = {
        "status": "verified_for_feature_extraction_only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "historical_commit": HISTORICAL_COMMIT,
        "historical_source_dir": str(source_dir),
        "historical_source_sha256": SOURCE_SHA256,
        "legacy_worktree_clean": True,
        "checkpoint_variant": metadata["variant"],
        "checkpoint_seed": metadata["seed"],
        "attempted_updates": metadata["optimizer_updates"],
        "successful_optimizer_step_values": optimizer_steps,
        "successful_updates": optimizer_steps[0] if len(optimizer_steps) == 1 else None,
        "torch_runtime_for_verification": str(torch.__version__),
        "checkpoint_torch_runtime": metadata["runtime"]["torch"],
        "strict_model_state_load": True,
        "strict_projector_state_load": True,
        "ema_target_encoder_frozen": True,
        **feature_info,
        "scope": (
            "Compatibility only. Downstream performance requires a separately registered "
            "labelled-data, split, baseline, and probe-fitting protocol."
        ),
    }
    write_report(arguments.report, report, overwrite=arguments.overwrite_report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
