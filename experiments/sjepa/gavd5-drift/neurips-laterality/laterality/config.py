from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parent
DEFAULT_PROTOCOL_PATH = SUITE_ROOT / "config" / "protocol.json"


def canonical_json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    payload = json.loads(protocol_path.read_text())
    if payload.get("schema") != "neurips_laterality_protocol/v2":
        raise ValueError(f"Unsupported protocol schema in {protocol_path}")
    return payload


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer")
    return parsed


def parse_str_list(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one value")
    return parsed


@dataclass(frozen=True)
class ExperimentContext:
    suite_root: Path
    repo_root: Path
    protocol_path: Path
    protocol: dict[str, Any]
    protocol_digest: str
    context_digest: str
    profile: str
    artifact_root: Path
    pose_root: Path
    annotation_root: Path
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    variants: tuple[str, ...]

    @property
    def profile_config(self) -> dict[str, Any]:
        return self.protocol["training"][self.profile]

    @property
    def is_paper(self) -> bool:
        return self.profile == "paper"


def load_context(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    profile: str | None = None,
) -> ExperimentContext:
    resolved_protocol = Path(protocol_path).resolve()
    protocol = load_protocol(resolved_protocol)
    chosen_profile = profile or os.getenv("LATERALITY_PROFILE", "smoke")
    if chosen_profile not in {"smoke", "paper"}:
        raise ValueError("LATERALITY_PROFILE must be 'smoke' or 'paper'")

    protocol_digest = canonical_json_digest(protocol)
    artifact_override = os.getenv("LATERALITY_ARTIFACT_ROOT")
    artifact_root = (
        Path(artifact_override).expanduser().resolve()
        if artifact_override
        else (
            SUITE_ROOT
            / "artifacts"
            / chosen_profile
            / f"protocol_{protocol_digest[:12]}"
        )
    )
    pose_config = Path(protocol["data"]["pose_root"])
    pose_root = pose_config if pose_config.is_absolute() else (SUITE_ROOT / pose_config).resolve()
    annotation_config = Path(protocol["data"]["annotation_root"])
    annotation_root = (
        annotation_config
        if annotation_config.is_absolute()
        else (SUITE_ROOT / annotation_config).resolve()
    )

    context_digest = canonical_json_digest(
        {
            "protocol_digest": protocol_digest,
            "profile": chosen_profile,
            "data_mode": "real_pose_cache" if chosen_profile == "paper" else "synthetic_smoke",
            "model_override": protocol["training"][chosen_profile].get(
                "model_override", {}
            ),
        }
    )

    default_seeds = [int(x) for x in protocol["training"][chosen_profile]["seeds"]]
    seeds = parse_int_list(os.getenv("LATERALITY_SEEDS"), default_seeds)
    n_outer = int(protocol["splits"]["outer_folds"])
    folds = parse_int_list(os.getenv("LATERALITY_FOLDS"), list(range(n_outer)))
    if any(fold < 0 or fold >= n_outer for fold in folds):
        raise ValueError(f"Fold selection must be in [0, {n_outer - 1}]")
    variants = parse_str_list(
        os.getenv("LATERALITY_VARIANTS"),
        list(protocol["training"]["variants"]),
    )
    unknown = sorted(set(variants) - set(protocol["training"]["variants"]))
    if unknown:
        raise ValueError(f"Unknown training variants: {unknown}")

    return ExperimentContext(
        suite_root=SUITE_ROOT,
        repo_root=REPO_ROOT,
        protocol_path=resolved_protocol,
        protocol=protocol,
        protocol_digest=protocol_digest,
        context_digest=context_digest,
        profile=chosen_profile,
        artifact_root=artifact_root,
        pose_root=pose_root,
        annotation_root=annotation_root,
        seeds=tuple(seeds),
        folds=tuple(folds),
        variants=tuple(variants),
    )


def model_config(context: ExperimentContext) -> dict[str, Any]:
    data = context.protocol["data"]
    base = {
        "frames": int(data["frames"]),
        "joints": int(data["joints"]),
        "coordinate_dim": int(data["coordinate_dim"]),
        "segment_length": int(data["segment_length"]),
        "embed_dim": int(context.protocol["model"]["embed_dim"]),
        "encoder_depth": int(context.protocol["model"]["encoder_depth"]),
        "predictor_depth": int(context.protocol["model"]["predictor_depth"]),
        "heads": int(context.protocol["model"]["heads"]),
    }
    base.update(context.profile_config.get("model_override", {}))
    if base["frames"] % base["segment_length"]:
        raise ValueError("frames must be divisible by segment_length")
    return base
