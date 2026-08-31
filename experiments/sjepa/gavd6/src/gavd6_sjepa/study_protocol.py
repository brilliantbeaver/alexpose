"""Common arm/route contract for the decisive Latent Laterality comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .latent_laterality import (
    TwoStateDurationModel,
    apply_block_correction,
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    learned: bool
    encoder_variant: str | None
    correction: str
    posterior: str
    downstream_source: str


ARM_SPECS = {
    item.name: item
    for item in (
        ArmSpec("raw_temporal", False, None, "continuity_map", "map", "raw"),
        ArmSpec("standard_sjepa", True, "standard_sjepa", "none", "none", "encoder"),
        ArmSpec("standard_mirror_aug", True, "standard_mirror_aug", "none", "none", "encoder"),
        ArmSpec("paired_unconstrained", True, "paired_unconstrained", "none", "none", "encoder"),
        ArmSpec("reflection_equivariant", True, "reflection_equivariant", "none", "none", "encoder"),
        ArmSpec("correction_first_sjepa", True, "standard_sjepa", "continuity_map", "map", "encoder"),
        ArmSpec("sg_jepa", True, "reflection_equivariant", "none", "structured_detached", "encoder"),
        ArmSpec("uniform_posterior", True, "reflection_equivariant", "none", "uniform_50_50", "encoder"),
        ArmSpec("oracle_correction", False, None, "oracle", "point", "raw"),
        ArmSpec("raw_downstream", False, None, "none", "none", "raw"),
        ArmSpec("random_encoder", False, "standard_sjepa", "none", "none", "random_encoder"),
    )
}

SOURCE_ROUTES = ("amass-only", "gavd-only", "amass-to-gavd")
SOURCE_SCREEN_ARMS = ("standard_sjepa", "reflection_equivariant")
CONFIRMATION_SEEDS = (7, 19, 31)


def require_benchmark_gate(
    path: Path, *, gauge_manifest_sha256: str | None = None
) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("ready_for_sg_jepa") is not True:
        raise RuntimeError(
            "SG-JEPA is blocked: the sequence benchmark did not pass every "
            "mask-leakage, global-chart, and oracle-versus-continuity gate"
        )
    gated_manifest = payload.get("gauge_manifest_sha256")
    if gauge_manifest_sha256 is not None and gated_manifest != gauge_manifest_sha256:
        raise RuntimeError(
            "SG-JEPA is blocked: the benchmark decision was made for a different "
            "gauge manifest"
        )
    return payload


def source_screen_jobs(seed: int = 7) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (route, arm, int(seed))
        for route in SOURCE_ROUTES
        for arm in SOURCE_SCREEN_ARMS
    )


def confirmatory_jobs(
    strongest_route: str, strongest_baseline: str
) -> tuple[tuple[str, str, int], ...]:
    if strongest_route not in SOURCE_ROUTES:
        raise ValueError(strongest_route)
    if strongest_baseline not in ARM_SPECS or strongest_baseline == "sg_jepa":
        raise ValueError(strongest_baseline)
    return tuple(
        (strongest_route, arm, seed)
        for seed in CONFIRMATION_SEEDS
        for arm in (strongest_baseline, "sg_jepa")
    )


def prepare_arm_observation(
    arm: str,
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    continuity_logits: np.ndarray,
    duration_model: TwoStateDurationModel,
    temperature: float,
    oracle_path: np.ndarray | None = None,
    block_frames: int = 4,
) -> dict:
    """Give every control a single typed input/output preparation interface."""

    if arm not in ARM_SPECS:
        raise ValueError(f"Unknown study arm: {arm}")
    spec = ARM_SPECS[arm]
    posterior = duration_model.infer(
        continuity_logits, temperature=temperature
    )
    output = {
        "coordinates": np.asarray(coordinates).copy(),
        "valid": np.asarray(valid).copy(),
        "map_path": posterior.map_path,
        "block_swap_probability": posterior.block_swap_probability,
        "edge_switch_probability": posterior.edge_switch_probability,
        "posterior_detached": spec.posterior in {"structured_detached", "uniform_50_50"},
    }
    if spec.correction == "continuity_map":
        output["coordinates"], output["valid"] = apply_block_correction(
            coordinates, valid, posterior.map_path, block_frames=block_frames
        )
    elif spec.correction == "oracle":
        if oracle_path is None:
            raise ValueError("Oracle arm requires oracle_path")
        output["coordinates"], output["valid"] = apply_block_correction(
            coordinates, valid, oracle_path, block_frames=block_frames
        )
    if spec.posterior == "uniform_50_50":
        output["block_swap_probability"] = np.full_like(
            posterior.block_swap_probability, 0.5
        )
        output["edge_switch_probability"] = np.full_like(
            posterior.edge_switch_probability, 0.5
        )
    return output
