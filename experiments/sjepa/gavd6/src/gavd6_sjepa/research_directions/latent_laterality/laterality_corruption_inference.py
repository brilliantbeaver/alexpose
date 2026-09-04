"""Sequence-level latent-laterality benchmark and structured inference.

The module keeps three operations typed and separate:

* a sensor-chart reflection changes one coordinate channel only;
* ``P`` exchanges bilateral token-indexed fields only; and
* a global latent-chart bit re-expresses the whole semantic path.

Corruption is generated once per full sequence. Windows are only views of that
already-corrupted sequence, so overlapping windows cannot disagree about a
shared source frame.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from gavd6_sjepa.research_directions.reflection_equivariance.amass_core11_training_pipeline import MIRROR_CHANNEL, MIRROR_PAIRS


PATH_FAMILIES = ("clean", "global_swap", "local_segment", "repeated_switches")


@dataclass(frozen=True)
class SequenceGaugeConfig:
    block_frames: int = 4
    window_frames: int = 64
    window_stride: int = 32
    clean_probability: float = 0.20
    global_probability: float = 0.10
    local_probability: float = 0.50
    repeated_probability: float = 0.20
    local_durations: tuple[int, ...] = (1, 2, 4, 8)
    repeated_switch_rate: float = 0.06
    maximum_repeated_switches: int = 4
    nuisance_boundaries: int = 4
    nuisance_selection: str = "matched"
    boundary_radius_frames: int = 1
    boundary_mode: str = "interpolate"
    noise_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.block_frames < 1 or self.window_frames < self.block_frames:
            raise ValueError("block_frames and window_frames must be positive")
        if self.window_frames % self.block_frames:
            raise ValueError("window_frames must be divisible by block_frames")
        probabilities = np.asarray(
            [
                self.clean_probability,
                self.global_probability,
                self.local_probability,
                self.repeated_probability,
            ],
            dtype=float,
        )
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Path-family probabilities must be nonnegative and sum to one")
        if not self.local_durations or min(self.local_durations) < 1:
            raise ValueError("local_durations must contain positive block counts")
        if not 0.0 <= self.repeated_switch_rate <= 1.0:
            raise ValueError("repeated_switch_rate must be in [0, 1]")
        if self.maximum_repeated_switches < 1:
            raise ValueError("maximum_repeated_switches must be positive")
        if self.nuisance_selection not in {"matched", "independent"}:
            raise ValueError("Nuisance selection must be matched or independent")
        if (
            self.nuisance_selection == "matched"
            and self.nuisance_boundaries < self.maximum_repeated_switches
        ):
            raise ValueError("Matched nuisances must cover every possible true switch")
        if self.boundary_mode not in {"interpolate", "gap"}:
            raise ValueError("boundary_mode must be interpolate or gap")
        if self.boundary_radius_frames < 0 or self.noise_scale < 0:
            raise ValueError("Boundary radius and noise scale must be nonnegative")


def sequence_gauge_config_json(config: SequenceGaugeConfig) -> str:
    """Return the canonical, manifest-safe representation of a gauge config."""

    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def sequence_gauge_config_from_json(value: str) -> SequenceGaugeConfig:
    """Load and validate the exact configuration embedded in a draw manifest."""

    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Sequence gauge configuration must be a JSON object")
    if "local_durations" in payload:
        payload["local_durations"] = tuple(payload["local_durations"])
    return SequenceGaugeConfig(**payload)


@dataclass(frozen=True)
class SequenceGaugeDraw:
    sequence_id: str
    identity: str
    split: str
    corruption_draw: int
    path_family: str
    gauge_path_rle: tuple[tuple[int, int], ...]
    switch_frames: tuple[int, ...]
    semantic_scope: str
    sensor_reflection_bit: int
    latent_chart_bit: int
    nuisance_boundary_frames: tuple[int, ...]
    occlusion_seed: int
    noise_seed: int
    generator_version: str = "sequence-gauge-v1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TemporalPosterior:
    map_path: np.ndarray
    block_swap_probability: np.ndarray
    edge_switch_probability: np.ndarray
    log_partition: float


def stable_seed(seed: int, namespace: str, key: str) -> int:
    payload = f"{seed}:{namespace}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def run_length_encode(path: np.ndarray) -> tuple[tuple[int, int], ...]:
    path = np.asarray(path, dtype=np.int8)
    if path.ndim != 1 or len(path) == 0 or not np.isin(path, (0, 1)).all():
        raise ValueError("Gauge path must be a nonempty binary vector")
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(path) + 1):
        if index == len(path) or path[index] != path[start]:
            result.append((int(path[start]), index - start))
            start = index
    return tuple(result)


def run_length_decode(encoded: Sequence[Sequence[int]]) -> np.ndarray:
    values: list[int] = []
    for bit, count in encoded:
        if int(bit) not in (0, 1) or int(count) < 1:
            raise ValueError("Invalid gauge run-length encoding")
        values.extend([int(bit)] * int(count))
    if not values:
        raise ValueError("Gauge run-length encoding is empty")
    return np.asarray(values, dtype=np.int8)


def semantic_permute(values: np.ndarray) -> np.ndarray:
    """Apply Core11 semantic ``P`` without changing coordinate signs."""

    result = np.asarray(values).copy()
    original = np.asarray(values)
    if original.shape[-1] == 11:
        joint_axis = original.ndim - 1
    elif original.ndim >= 2 and original.shape[-2] == 11:
        joint_axis = original.ndim - 2
    else:
        raise ValueError(f"Cannot locate Core11 joint axis in shape {original.shape}")
    for left, right in MIRROR_PAIRS:
        left_index = [slice(None)] * original.ndim
        right_index = [slice(None)] * original.ndim
        left_index[joint_axis] = left
        right_index[joint_axis] = right
        result[tuple(left_index)] = original[tuple(right_index)]
        result[tuple(right_index)] = original[tuple(left_index)]
    return result


def semantic_permute_by_frame(values: np.ndarray, frame_path: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    frame_path = np.asarray(frame_path, dtype=bool)
    if len(result) != len(frame_path):
        raise ValueError("Frame path and token-indexed array lengths differ")
    if frame_path.any():
        result[frame_path] = semantic_permute(result[frame_path])
    return result


def sensor_reflect(coordinates: np.ndarray, bit: int) -> np.ndarray:
    """Reflect the declared sensor channel without exchanging joint names."""

    result = np.asarray(coordinates, dtype=np.float32).copy()
    if int(bit) not in (0, 1):
        raise ValueError("Sensor reflection bit must be binary")
    if bit:
        result[..., MIRROR_CHANNEL] *= -1.0
    return result


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.ones_like(values)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float) / float(len(values) - 1)


def boundary_ambiguity_scores(
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    block_frames: int,
    pelvis_world: np.ndarray | None = None,
) -> np.ndarray:
    """Rank boundaries using side-symmetric, training-label-free motion cues."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (11, 3):
        raise ValueError("Expected Core11 coordinates [T,11,3]")
    if valid.shape != coordinates.shape[:2]:
        raise ValueError("Validity shape does not match coordinates")
    blocks = len(coordinates) // block_frames
    if blocks < 2:
        return np.empty(0, dtype=np.float64)
    boundaries = np.arange(1, blocks) * block_frames
    crossing = np.zeros(len(boundaries), dtype=float)
    bilateral = np.zeros_like(crossing)
    low_speed = np.zeros_like(crossing)
    turning = np.zeros_like(crossing)
    pelvis = None if pelvis_world is None else np.asarray(pelvis_world, dtype=float)
    for item, boundary in enumerate(boundaries):
        lo, hi = max(0, boundary - block_frames), min(len(coordinates), boundary + block_frames)
        lateral_separations = []
        motion_differences = []
        for left, right in MIRROR_PAIRS:
            pair_valid = valid[lo:hi, left] & valid[lo:hi, right]
            if pair_valid.any():
                delta = coordinates[lo:hi, left] - coordinates[lo:hi, right]
                lateral_separations.append(
                    float(np.median(np.abs(delta[pair_valid, MIRROR_CHANNEL])))
                )
                if pair_valid.sum() > 1:
                    left_speed = np.linalg.norm(
                        np.diff(coordinates[lo:hi, left], axis=0), axis=1
                    )
                    right_speed = np.linalg.norm(
                        np.diff(coordinates[lo:hi, right], axis=0), axis=1
                    )
                    motion_differences.append(float(np.mean(np.abs(left_speed - right_speed))))
        crossing[item] = -min(lateral_separations, default=0.0)
        bilateral[item] = -np.mean(motion_differences) if motion_differences else 0.0
        trajectory = coordinates[:, 0] if pelvis is None else pelvis
        if len(trajectory) > boundary + 1:
            before = trajectory[boundary - 1] - trajectory[max(0, boundary - block_frames)]
            after = trajectory[min(len(trajectory) - 1, boundary + block_frames)] - trajectory[boundary]
            low_speed[item] = -float(np.linalg.norm(before) + np.linalg.norm(after))
            denom = np.linalg.norm(before) * np.linalg.norm(after)
            if denom > 1e-8:
                turning[item] = 1.0 - float(np.clip(np.dot(before, after) / denom, -1.0, 1.0))
    return 0.30 * _rank01(crossing) + 0.30 * _rank01(bilateral) + 0.25 * _rank01(low_speed) + 0.15 * _rank01(turning)


def _choose_weighted_boundary(
    candidates: np.ndarray, scores: np.ndarray, rng: np.random.Generator
) -> int:
    if len(candidates) == 0:
        raise ValueError("No eligible sequence boundary")
    weights = np.exp(3.0 * (scores - np.max(scores)))
    weights /= weights.sum()
    return int(rng.choice(candidates, p=weights))


def generate_sequence_draw(
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    sequence_id: str,
    identity: str,
    split: str,
    corruption_draw: int,
    seed: int,
    config: SequenceGaugeConfig = SequenceGaugeConfig(),
    pelvis_world: np.ndarray | None = None,
) -> tuple[SequenceGaugeDraw, np.ndarray]:
    """Generate one persistent block path for an entire source sequence."""

    blocks = len(coordinates) // config.block_frames
    if blocks < 3:
        raise ValueError("Sequence needs at least three complete gauge blocks")
    key = f"{sequence_id}:{corruption_draw}"
    path_rng = np.random.default_rng(stable_seed(seed, "path", key))
    family = str(
        path_rng.choice(
            PATH_FAMILIES,
            p=(
                config.clean_probability,
                config.global_probability,
                config.local_probability,
                config.repeated_probability,
            ),
        )
    )
    path = np.zeros(blocks, dtype=np.int8)
    scores = boundary_ambiguity_scores(
        coordinates,
        valid,
        block_frames=config.block_frames,
        pelvis_world=pelvis_world,
    )
    boundaries = np.arange(1, blocks)
    if family == "global_swap":
        path[:] = 1
    elif family == "local_segment":
        duration = min(int(path_rng.choice(config.local_durations)), blocks - 2)
        starts = np.arange(1, blocks - duration)
        start_scores = np.asarray(
            [scores[start - 1] + scores[start + duration - 1] for start in starts]
        )
        start = _choose_weighted_boundary(starts, start_scores, path_rng)
        path[start : start + duration] = 1
    elif family == "repeated_switches":
        probabilities = config.repeated_switch_rate * (0.35 + 1.3 * scores)
        probabilities = np.clip(probabilities, 0.0, 0.95)
        chosen = boundaries[path_rng.random(len(boundaries)) < probabilities]
        if len(chosen) == 0:
            chosen = np.asarray([_choose_weighted_boundary(boundaries, scores, path_rng)])
        if len(chosen) > config.maximum_repeated_switches:
            ranked = np.argsort(scores[chosen - 1], kind="stable")
            chosen = np.sort(chosen[ranked[-config.maximum_repeated_switches :]])
        state = 0
        for block in range(blocks):
            if block in set(chosen.tolist()):
                state ^= 1
            path[block] = state

    switch_blocks = np.flatnonzero(path[1:] != path[:-1]) + 1
    switch_frames = tuple(int(item * config.block_frames) for item in switch_blocks)
    nuisance_rng = np.random.default_rng(stable_seed(seed, "boundary", key))
    if config.nuisance_selection == "matched":
        nuisance_blocks = list(map(int, switch_blocks))
        remaining = np.setdiff1d(boundaries, np.asarray(nuisance_blocks, dtype=int))
        needed = min(config.nuisance_boundaries, len(boundaries)) - len(nuisance_blocks)
        if needed > 0:
            pseudo = nuisance_rng.choice(remaining, size=needed, replace=False)
            nuisance_blocks.extend(map(int, pseudo))
    else:
        # Gaps must not be guaranteed at true changes: that would make the
        # validity pattern a weak switch detector.  Draw them with an
        # independent RNG, so a subset of events is obscured while the rest
        # remain visible and mask-only evidence has no label association.
        count = min(config.nuisance_boundaries, len(boundaries))
        nuisance_blocks = list(map(int, nuisance_rng.choice(boundaries, size=count, replace=False)))
    nuisance_frames = tuple(sorted(block * config.block_frames for block in nuisance_blocks))
    bit_rng = np.random.default_rng(stable_seed(seed, "independent-bits", key))
    draw = SequenceGaugeDraw(
        sequence_id=str(sequence_id),
        identity=str(identity),
        split=str(split),
        corruption_draw=int(corruption_draw),
        path_family=family,
        gauge_path_rle=run_length_encode(path),
        switch_frames=switch_frames,
        semantic_scope="core11_all_pairs",
        sensor_reflection_bit=int(bit_rng.integers(0, 2)),
        latent_chart_bit=int(bit_rng.integers(0, 2)),
        nuisance_boundary_frames=nuisance_frames,
        occlusion_seed=stable_seed(seed, "occlusion", key),
        noise_seed=stable_seed(seed, "noise", key),
    )
    return draw, path


def _apply_boundary_nuisance(
    coordinates: np.ndarray,
    valid: np.ndarray,
    boundaries: Iterable[int],
    config: SequenceGaugeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = coordinates.copy()
    valid = valid.copy()
    radius = config.boundary_radius_frames
    if radius == 0:
        return coordinates, valid
    for boundary in boundaries:
        left = max(0, int(boundary) - radius - 1)
        right = min(len(coordinates) - 1, int(boundary) + radius)
        if left >= right:
            continue
        for frame in range(max(0, int(boundary) - radius), min(len(coordinates), int(boundary) + radius)):
            usable = valid[left] & valid[right]
            if config.boundary_mode == "gap":
                coordinates[frame, usable] = 0.0
                valid[frame, usable] = False
            else:
                fraction = float(frame - left) / float(right - left)
                coordinates[frame, usable] = (
                    (1.0 - fraction) * coordinates[left, usable]
                    + fraction * coordinates[right, usable]
                )
                # Interpolation retains the pre-existing validity signature.
                # Thus matched pseudo-boundaries cannot disclose switch labels
                # to a mask-only model.
    coordinates[~valid] = 0.0
    return coordinates, valid


def apply_sequence_draw(
    coordinates: np.ndarray,
    valid: np.ndarray,
    draw: SequenceGaugeDraw,
    *,
    config: SequenceGaugeConfig = SequenceGaugeConfig(),
    confidence: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply typed nuisance and semantic actions to a full sequence."""

    coordinates = np.asarray(coordinates, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != coordinates.shape[:2]:
        raise ValueError("Validity shape does not match coordinates")
    if confidence is not None and np.asarray(confidence).shape != valid.shape:
        raise ValueError("Confidence shape does not match validity")
    path = run_length_decode(draw.gauge_path_rle)
    covered = len(path) * config.block_frames
    if covered > len(coordinates) or len(coordinates) - covered >= config.block_frames:
        raise ValueError("Gauge path does not cover the sequence's complete blocks")
    nuisance_coordinates = sensor_reflect(coordinates, draw.sensor_reflection_bit)
    nuisance_valid = valid.copy()
    if config.noise_scale:
        rng = np.random.default_rng(draw.noise_seed)
        noise = rng.normal(0.0, config.noise_scale, nuisance_coordinates.shape)
        nuisance_coordinates[nuisance_valid] += noise[nuisance_valid].astype(np.float32)
    nuisance_coordinates, nuisance_valid = _apply_boundary_nuisance(
        nuisance_coordinates,
        nuisance_valid,
        draw.nuisance_boundary_frames,
        config,
    )
    frame_path = np.repeat(path ^ draw.latent_chart_bit, config.block_frames)
    if covered < len(coordinates):
        frame_path = np.concatenate(
            [frame_path, np.repeat(frame_path[-1], len(coordinates) - covered)]
        )
    observed_coordinates = semantic_permute_by_frame(nuisance_coordinates, frame_path)
    observed_valid = semantic_permute_by_frame(nuisance_valid, frame_path)
    result = {
        "coordinates": observed_coordinates.astype(np.float32),
        "valid": observed_valid.astype(bool),
        "nuisance_reference": nuisance_coordinates.astype(np.float32),
        "nuisance_valid": nuisance_valid.astype(bool),
        "gauge_path": path,
        "observed_frame_path": frame_path.astype(np.int8),
    }
    if confidence is not None:
        nuisance_confidence = np.asarray(confidence, dtype=np.float32).copy()
        nuisance_confidence[~nuisance_valid] = 0.0
        result["confidence"] = semantic_permute_by_frame(
            nuisance_confidence, frame_path
        ).astype(np.float32)
    return result


def slice_corrupted_windows(
    corrupted: Mapping[str, np.ndarray],
    *,
    config: SequenceGaugeConfig = SequenceGaugeConfig(),
) -> list[dict[str, np.ndarray | int]]:
    """Slice only after full-sequence corruption, preserving shared frames."""

    frames = len(corrupted["coordinates"])
    if frames < config.window_frames:
        return []
    starts = list(range(0, frames - config.window_frames + 1, config.window_stride))
    final = frames - config.window_frames
    if starts[-1] != final:
        starts.append(final)
    output = []
    for start in starts:
        stop = start + config.window_frames
        item: dict[str, np.ndarray | int] = {"start_frame": start}
        for key in ("coordinates", "valid", "nuisance_reference", "nuisance_valid", "observed_frame_path"):
            item[key] = np.asarray(corrupted[key])[start:stop].copy()
        if "confidence" in corrupted:
            item["confidence"] = np.asarray(corrupted["confidence"])[start:stop].copy()
        output.append(item)
    return output


def continuity_edge_logits(
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    block_frames: int = 4,
    scale: float = 1.0,
) -> np.ndarray:
    """Transparent one-boundary same-versus-swapped continuity evidence."""

    if scale <= 0:
        raise ValueError("Continuity scale must be positive")
    blocks = len(coordinates) // block_frames
    logits = []
    for block in range(1, blocks):
        left_frame = block * block_frames - 1
        right_frame = block * block_frames
        usable = valid[left_frame] & valid[right_frame]
        swapped_right = semantic_permute(coordinates[right_frame])
        swapped_valid = semantic_permute(valid[right_frame])
        swapped_usable = valid[left_frame] & swapped_valid
        same = np.mean(
            np.square(coordinates[left_frame, usable] - coordinates[right_frame, usable])
        ) if usable.any() else 0.0
        swapped = np.mean(
            np.square(coordinates[left_frame, swapped_usable] - swapped_right[swapped_usable])
        ) if swapped_usable.any() else 0.0
        logits.append((same - swapped) / scale)
    return np.asarray(logits, dtype=np.float64)


class TwoStateDurationModel:
    """Exact duration-expanded two-state model with MAP and marginals."""

    def __init__(
        self,
        max_duration: int = 16,
        switch_hazard: Sequence[float] | None = None,
    ) -> None:
        if max_duration < 2:
            raise ValueError("max_duration must be at least two")
        self.max_duration = int(max_duration)
        if switch_hazard is None:
            duration = np.arange(1, max_duration + 1, dtype=float)
            hazard = np.clip(0.015 + 0.004 * duration, 0.015, 0.12)
        else:
            hazard = np.asarray(switch_hazard, dtype=float)
            if hazard.shape != (max_duration,):
                raise ValueError("switch_hazard length must equal max_duration")
        if (hazard <= 0).any() or (hazard >= 1).any():
            raise ValueError("Every duration hazard must lie strictly in (0,1)")
        self.switch_hazard = hazard
        self._transition = self._transition_matrix()

    @property
    def states(self) -> int:
        return 2 * self.max_duration

    def _index(self, gauge: int, duration: int) -> int:
        return int(gauge) * self.max_duration + int(duration) - 1

    def _decode(self, state: int) -> tuple[int, int]:
        return state // self.max_duration, state % self.max_duration + 1

    def _transition_matrix(self) -> np.ndarray:
        matrix = np.full((self.states, self.states), -np.inf, dtype=np.float64)
        for state in range(self.states):
            gauge, duration = self._decode(state)
            hazard = self.switch_hazard[duration - 1]
            stay_duration = min(duration + 1, self.max_duration)
            matrix[state, self._index(gauge, stay_duration)] = math.log1p(-hazard)
            matrix[state, self._index(1 - gauge, 1)] = math.log(hazard)
        return matrix

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        finite = np.isfinite(maximum)
        safe_maximum = np.where(finite, maximum, 0.0)
        shifted = np.where(finite, values - safe_maximum, -np.inf)
        total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        with np.errstate(divide="ignore"):
            answer = np.where(finite, maximum + np.log(total), -np.inf)
        return np.squeeze(answer, axis=axis) if axis is not None else answer.squeeze()

    def _edge_factor(self, logit: float, source: int, target: int, temperature: float) -> float:
        source_gauge, _ = self._decode(source)
        target_gauge, _ = self._decode(target)
        sign = 1.0 if source_gauge != target_gauge else -1.0
        return 0.5 * sign * float(logit) / float(temperature)

    def infer(
        self,
        edge_logits: np.ndarray,
        *,
        temperature: float = 1.0,
        anchor: tuple[int, int, float] | None = None,
        root_bit: int | None = None,
    ) -> TemporalPosterior:
        edge_logits = np.asarray(edge_logits, dtype=np.float64)
        if edge_logits.ndim != 1 or temperature <= 0:
            raise ValueError("edge_logits must be one-dimensional and temperature positive")
        blocks = len(edge_logits) + 1
        node_evidence = np.zeros((blocks, self.states), dtype=np.float64)
        if anchor is not None:
            anchor_block, bit, error = anchor
            if not 0 <= int(anchor_block) < blocks or int(bit) not in (0, 1):
                raise ValueError("Anchor block and bit are outside the path")
            if not 0.0 < float(error) < 1.0:
                raise ValueError("Anchor error must lie strictly in (0,1)")
            for state in range(self.states):
                gauge, _ = self._decode(state)
                node_evidence[int(anchor_block), state] = math.log(
                    (1.0 - error) if gauge == bit else error
                )
        if root_bit is not None and int(root_bit) not in (0, 1):
            raise ValueError("root_bit must be zero, one, or None")
        alpha = np.full((blocks, self.states), -np.inf)
        roots = (0, 1) if root_bit is None else (int(root_bit),)
        for gauge in roots:
            state = self._index(gauge, 1)
            alpha[0, state] = -math.log(float(len(roots))) + node_evidence[0, state]
        viterbi = alpha.copy()
        back = np.zeros((blocks, self.states), dtype=np.int64)
        for block in range(1, blocks):
            factors = np.empty_like(self._transition)
            for source in range(self.states):
                for target in range(self.states):
                    factors[source, target] = self._transition[source, target]
                    if np.isfinite(factors[source, target]):
                        factors[source, target] += self._edge_factor(
                            edge_logits[block - 1], source, target, temperature
                        )
            incoming = alpha[block - 1, :, None] + factors
            alpha[block] = self._logsumexp(incoming, axis=0) + node_evidence[block]
            scores = viterbi[block - 1, :, None] + factors
            back[block] = np.argmax(scores, axis=0)
            viterbi[block] = np.max(scores, axis=0) + node_evidence[block]
        log_partition = float(self._logsumexp(alpha[-1]))
        beta = np.zeros_like(alpha)
        for block in range(blocks - 2, -1, -1):
            values = np.empty_like(self._transition)
            for source in range(self.states):
                for target in range(self.states):
                    values[source, target] = self._transition[source, target]
                    if np.isfinite(values[source, target]):
                        values[source, target] += self._edge_factor(
                            edge_logits[block], source, target, temperature
                        ) + node_evidence[block + 1, target] + beta[block + 1, target]
            beta[block] = self._logsumexp(values, axis=1)
        state_probability = np.exp(alpha + beta - log_partition)
        block_probability = np.zeros(blocks, dtype=np.float64)
        for state in range(self.states):
            gauge, _ = self._decode(state)
            if gauge:
                block_probability += state_probability[:, state]
        edge_probability = np.zeros(blocks - 1, dtype=np.float64)
        for block in range(blocks - 1):
            for source in range(self.states):
                source_gauge, _ = self._decode(source)
                for target in range(self.states):
                    target_gauge, _ = self._decode(target)
                    if source_gauge == target_gauge or not np.isfinite(self._transition[source, target]):
                        continue
                    score = (
                        alpha[block, source]
                        + self._transition[source, target]
                        + self._edge_factor(edge_logits[block], source, target, temperature)
                        + node_evidence[block + 1, target]
                        + beta[block + 1, target]
                        - log_partition
                    )
                    edge_probability[block] += math.exp(score)
        last = int(np.argmax(viterbi[-1]))
        state_path = np.empty(blocks, dtype=np.int64)
        state_path[-1] = last
        for block in range(blocks - 1, 0, -1):
            state_path[block - 1] = back[block, state_path[block]]
        map_path = np.asarray([self._decode(state)[0] for state in state_path], dtype=np.int8)
        return TemporalPosterior(
            map_path=map_path,
            block_swap_probability=block_probability,
            edge_switch_probability=edge_probability,
            log_partition=log_partition,
        )

    def equivalence_path_nll(
        self,
        edge_logits: np.ndarray,
        true_path: np.ndarray,
        *,
        temperature: float = 1.0,
    ) -> float:
        true_path = np.asarray(true_path, dtype=np.int8)
        if len(true_path) < 2 or len(edge_logits) != len(true_path) - 1:
            raise ValueError("Path and edge logits have incompatible lengths")
        posterior = self.infer(edge_logits, temperature=temperature)

        def score(path: np.ndarray) -> float:
            duration = 1
            total = -math.log(2.0)
            for edge, (left, right) in enumerate(zip(path[:-1], path[1:])):
                source = self._index(int(left), duration)
                if left == right:
                    next_duration = min(duration + 1, self.max_duration)
                else:
                    next_duration = 1
                target = self._index(int(right), next_duration)
                total += self._transition[source, target]
                total += self._edge_factor(edge_logits[edge], source, target, temperature)
                duration = next_duration
            return total

        first = score(true_path)
        second = score(1 - true_path)
        mass_log = float(self._logsumexp(np.asarray([first, second])))
        return (posterior.log_partition - mass_log) / float(len(true_path) - 1)


def calibrate_duration_temperature(
    model: TwoStateDurationModel,
    edge_logits: Sequence[np.ndarray],
    paths: Sequence[np.ndarray],
) -> float:
    """Fit one temperature on a disjoint calibration identity set."""

    if len(edge_logits) != len(paths) or not paths:
        raise ValueError("Calibration requires matching nonempty logits and paths")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        return float(
            np.mean(
                [
                    model.equivalence_path_nll(logits, path, temperature=temperature)
                    for logits, path in zip(edge_logits, paths)
                ]
            )
        )

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError(f"Temperature calibration failed: {result.message}")
    return math.exp(float(result.x))


def path_hamming_up_to_global_flip(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.int8)
    target = np.asarray(target, dtype=np.int8)
    if predicted.shape != target.shape:
        raise ValueError("Path shapes differ")
    return float(min(np.mean(predicted != target), np.mean((1 - predicted) != target)))


def switch_f1(predicted: np.ndarray, target: np.ndarray, tolerance: int = 1) -> float:
    predicted_edges = list((np.flatnonzero(np.diff(predicted) != 0) + 1).astype(int))
    target_edges = list((np.flatnonzero(np.diff(target) != 0) + 1).astype(int))
    matched: set[int] = set()
    true_positive = 0
    for edge in predicted_edges:
        options = [
            item
            for item, target_edge in enumerate(target_edges)
            if item not in matched and abs(edge - target_edge) <= tolerance
        ]
        if options:
            chosen = min(options, key=lambda item: abs(edge - target_edges[item]))
            matched.add(chosen)
            true_positive += 1
    if not predicted_edges and not target_edges:
        return 1.0
    precision = true_positive / max(len(predicted_edges), 1)
    recall = true_positive / max(len(target_edges), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def structured_parity_prediction_loss(
    predicted_even: torch.Tensor,
    predicted_odd: torch.Tensor,
    target_even: torch.Tensor,
    target_odd: torch.Tensor,
    relative_flip_probability: torch.Tensor,
    *,
    odd_sigma: float = 1.0,
) -> torch.Tensor:
    """Normalized parity loss with posterior weights always detached.

    The edge/duration model is fitted and calibrated separately. Detaching here
    prevents JEPA error from turning its probabilities into an arbitrary loss
    gate. A uniform ``0.5`` tensor implements the requested posterior ablation.
    """

    if odd_sigma <= 0:
        raise ValueError("odd_sigma must be positive")
    if not (
        predicted_even.shape
        == predicted_odd.shape
        == target_even.shape
        == target_odd.shape
    ):
        raise ValueError("Parity prediction and target tensors must share a shape")
    probability = relative_flip_probability.detach()
    while probability.ndim < predicted_odd.ndim:
        probability = probability.unsqueeze(-1)
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    even_loss = torch.mean(torch.square(predicted_even - target_even))
    same = 0.5 * torch.square((target_odd - predicted_odd) / odd_sigma)
    flipped = 0.5 * torch.square((target_odd + predicted_odd) / odd_sigma)
    log_normalizer = math.log(odd_sigma * math.sqrt(2.0 * math.pi))
    log_components = torch.stack(
        [
            torch.log1p(-probability) - same - log_normalizer,
            torch.log(probability) - flipped - log_normalizer,
        ],
        dim=0,
    )
    odd_nll = -torch.logsumexp(log_components, dim=0).mean()
    return even_loss + odd_nll


def apply_block_correction(
    coordinates: np.ndarray,
    valid: np.ndarray,
    block_path: np.ndarray,
    *,
    block_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    frame_path = np.repeat(np.asarray(block_path, dtype=np.int8), block_frames)
    if len(frame_path) < len(coordinates):
        frame_path = np.concatenate(
            [frame_path, np.repeat(frame_path[-1], len(coordinates) - len(frame_path))]
        )
    return (
        semantic_permute_by_frame(coordinates, frame_path),
        semantic_permute_by_frame(valid, frame_path),
    )


def odd_even_motion_targets(
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    block_frames: int = 4,
) -> tuple[float, float]:
    """Right-minus-left and total distal within-block motion energy."""

    left_energy = 0.0
    right_energy = 0.0
    count = 0
    distal_pairs = MIRROR_PAIRS[2:]
    blocks = len(coordinates) // block_frames
    for block in range(blocks):
        start = block * block_frames
        stop = start + block_frames
        for left, right in distal_pairs:
            for joint, accumulator in ((left, "left"), (right, "right")):
                pair_valid = valid[start : stop - 1, joint] & valid[start + 1 : stop, joint]
                if not pair_valid.any():
                    continue
                energy = float(
                    np.mean(
                        np.square(
                            np.diff(coordinates[start:stop, joint], axis=0)[pair_valid]
                        )
                    )
                )
                if accumulator == "left":
                    left_energy += energy
                else:
                    right_energy += energy
                count += 1
    denominator = max(count / 2.0, 1.0)
    left_energy /= denominator
    right_energy /= denominator
    return right_energy - left_energy, right_energy + left_energy


def block_odd_even_motion_targets(
    coordinates: np.ndarray,
    valid: np.ndarray,
    *,
    block_frames: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Blockwise version used for correspondence-sensitive common scoring."""

    blocks = len(coordinates) // block_frames
    odd = np.zeros(blocks, dtype=np.float64)
    even = np.zeros(blocks, dtype=np.float64)
    for block in range(blocks):
        start = block * block_frames
        stop = start + block_frames
        odd[block], even[block] = odd_even_motion_targets(
            coordinates[start:stop], valid[start:stop], block_frames=block_frames
        )
    return odd, even


def relative_reduction(baseline: float, candidate: float) -> float:
    return (float(baseline) - float(candidate)) / max(abs(float(baseline)), 1e-12)
