"""AMASS Core11 manifest, streaming loader, and matched JEPA training helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from gait_parity_jepa import (
    VARIANTS,
    OrbitJEPA,
    TrainConfig,
    VICRegProjector,
    build_model,
    cosine_ema,
    lift_orbit,
    orbit_vicreg,
    sample_mask,
    seed_everything,
    sjepa_cross_entropy,
)


SCHEMA = "core11-v1"
COORDINATE_FRAME = "gait-parity-body-v1"
JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_forefoot",
    "right_forefoot",
)
CHANNEL_NAMES = ("forward", "vertical_up", "mediolateral")
MIRROR_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))
MIRROR_CHANNEL = 2
VALID_FINGERPRINTS = {
    "female": "4664fd77484e9f3da96576b65b8ccd71a2c56d2e46311981d927d3297401c2cb",
    "male": "59ff15eeca81677eaa43c0d70edb0438aabffef0b66313f00dd8dbb7be91680d",
}
WINDOW_FRAMES = 64
STRIDE_FRAMES = 32
TIME_PATCH_FRAMES = 4
MINIMUM_VALID_JOINT_FRACTION = 0.95
REQUIRED_ARCHIVE_KEYS = {
    "coordinates",
    "coordinates_m",
    "valid",
    "canonical_times_s",
    "pelvis_world_m",
    "world_to_body_transform",
    "body_to_world_transform",
    "physical_basis_world",
    "leg_length_m",
    "joint_names",
    "channel_names",
    "provenance_json",
}
REQUIRED_MANIFEST_COLUMNS = {
    "relative_path",
    "tensor_relative_path",
    "source_dataset",
    "subject_id_candidate",
    "identity",
    "split",
    "motion_id",
    "gender",
    "canonical_fps",
    "canonical_frames",
    "source_sha256",
    "source_sha256_verified",
    "body_model_sha256",
    "dmpl_model_sha256",
    "schema",
    "coordinate_frame",
    "conversion_fingerprint",
    "status",
}

__all__ = [
    "CHANNEL_NAMES",
    "COORDINATE_FRAME",
    "JOINT_NAMES",
    "MINIMUM_VALID_JOINT_FRACTION",
    "MIRROR_CHANNEL",
    "MIRROR_PAIRS",
    "REQUIRED_ARCHIVE_KEYS",
    "SCHEMA",
    "STRIDE_FRAMES",
    "TIME_PATCH_FRAMES",
    "VALID_FINGERPRINTS",
    "WINDOW_FRAMES",
    "Core11WindowDataset",
    "FixedBatchPlan",
    "atomic_torch_save",
    "build_window_index",
    "checkpoint_payload",
    "core11_train_config",
    "load_conversion_manifest",
    "make_train_loader",
    "mirror_validity",
    "optimizer_updates",
    "sha256_file",
    "sha256_json",
    "train_streaming_variant",
    "validate_archives",
    "window_index_sha256",
    "window_starts",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _true_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_conversion_manifest(path: Path) -> pd.DataFrame:
    """Load the converted manifest and reject schema, split, or provenance drift."""

    path = Path(path).expanduser().resolve()
    text_columns = REQUIRED_MANIFEST_COLUMNS.difference(
        {"canonical_fps", "canonical_frames", "source_sha256_verified"}
    )
    frame = pd.read_csv(path, dtype={name: str for name in text_columns})
    missing = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Conversion manifest is empty")
    if not frame["tensor_relative_path"].is_unique:
        raise ValueError("tensor_relative_path must be unique")
    if not frame["source_sha256"].is_unique:
        raise ValueError("source_sha256 must be unique")
    if set(frame["status"]) != {"converted"}:
        raise ValueError(f"Expected only converted rows, found {sorted(frame['status'].unique())}")
    if set(frame["schema"]) != {SCHEMA}:
        raise ValueError(f"Expected schema {SCHEMA}, found {sorted(frame['schema'].unique())}")
    if set(frame["coordinate_frame"]) != {COORDINATE_FRAME}:
        raise ValueError("Unexpected coordinate frame")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split set: {sorted(frame['split'].unique())}")
    if not np.allclose(frame["canonical_fps"].to_numpy(float), 30.0, rtol=0.0, atol=0.0):
        raise ValueError("Core11 manifest must use exactly 30 fps")
    if not _true_series(frame["source_sha256_verified"]).all():
        raise ValueError("Every source SHA-256 must be verified")
    expected = frame["gender"].map(VALID_FINGERPRINTS)
    if expected.isna().any() or not expected.equals(frame["conversion_fingerprint"]):
        raise ValueError("Gender-specific conversion fingerprint mismatch")
    if (frame["canonical_frames"].astype(int) < 1).any():
        raise ValueError("canonical_frames must be positive")
    split_counts = frame.groupby("identity")["split"].nunique()
    if (split_counts != 1).any():
        leaked = split_counts[split_counts != 1].index.tolist()[:5]
        raise ValueError(f"Identity leakage across splits: {leaked}")
    return frame.sort_values("tensor_relative_path", kind="stable").reset_index(drop=True)


def window_starts(frames: int) -> list[int]:
    if frames < WINDOW_FRAMES:
        return []
    starts = list(range(0, frames - WINDOW_FRAMES + 1, STRIDE_FRAMES))
    final = frames - WINDOW_FRAMES
    if starts[-1] != final:
        starts.append(final)
    return starts


def build_window_index(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen 64-frame/32-stride, index-only window table."""

    rows = []
    for sequence_index, row in manifest.iterrows():
        immutable = {
            "schema": row["schema"],
            "source_sha256": row["source_sha256"],
            "conversion_fingerprint": row["conversion_fingerprint"],
        }
        for start in window_starts(int(row["canonical_frames"])):
            identity = {**immutable, "start_frame": start, "window_frames": WINDOW_FRAMES}
            rows.append(
                {
                    "window_id": sha256_json(identity),
                    "sequence_index": int(sequence_index),
                    "tensor_relative_path": row["tensor_relative_path"],
                    "identity": row["identity"],
                    "split": row["split"],
                    "source_sha256": row["source_sha256"],
                    "conversion_fingerprint": row["conversion_fingerprint"],
                    "start_frame": int(start),
                    "window_frames": WINDOW_FRAMES,
                }
            )
    index = pd.DataFrame(rows)
    if index.empty or not index["window_id"].is_unique:
        raise ValueError("Window index is empty or contains duplicate IDs")
    return index.reset_index(drop=True)


def window_index_sha256(index: pd.DataFrame) -> str:
    columns = [
        "window_id",
        "tensor_relative_path",
        "identity",
        "split",
        "source_sha256",
        "conversion_fingerprint",
        "start_frame",
        "window_frames",
    ]
    return sha256_json(index[columns].to_dict(orient="records"))


def mirror_validity(valid: torch.Tensor) -> torch.Tensor:
    """Swap Core11 bilateral validity without applying a coordinate sign change."""

    mirrored = valid.clone()
    original = valid.clone()
    for left, right in MIRROR_PAIRS:
        mirrored[..., left] = original[..., right]
        mirrored[..., right] = original[..., left]
    return mirrored


def _safe_tensor_path(root: Path, relative_text: str) -> Path:
    relative = PurePosixPath(str(relative_text))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe tensor_relative_path: {relative_text!r}")
    root = Path(root).expanduser().resolve()
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root)
    return path


def _archive_arrays(path: Path, row: Mapping) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARCHIVE_KEYS.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_ARCHIVE_KEYS}

    frames = int(row["canonical_frames"])
    coordinates = arrays["coordinates"]
    coordinates_m = arrays["coordinates_m"]
    valid = arrays["valid"]
    times = arrays["canonical_times_s"]
    pelvis = arrays["pelvis_world_m"]
    world_to_body = arrays["world_to_body_transform"]
    body_to_world = arrays["body_to_world_transform"]
    physical_basis = arrays["physical_basis_world"]
    leg_length = arrays["leg_length_m"]
    joint_names = tuple(arrays["joint_names"].tolist())
    channel_names = tuple(arrays["channel_names"].tolist())
    provenance = json.loads(str(arrays["provenance_json"].item()))

    expected_shapes = {
        "coordinates": (frames, 11, 3),
        "coordinates_m": (frames, 11, 3),
        "valid": (frames, 11),
        "canonical_times_s": (frames,),
        "pelvis_world_m": (frames, 3),
        "world_to_body_transform": (3, 3),
        "body_to_world_transform": (3, 3),
        "physical_basis_world": (3, 3),
        "leg_length_m": (),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{path}: {name} has shape {arrays[name].shape}, expected {shape}")
    expected_dtypes = {
        "coordinates": np.dtype(np.float32),
        "coordinates_m": np.dtype(np.float32),
        "valid": np.dtype(np.bool_),
        "canonical_times_s": np.dtype(np.float64),
        "pelvis_world_m": np.dtype(np.float32),
        "world_to_body_transform": np.dtype(np.float64),
        "body_to_world_transform": np.dtype(np.float64),
        "physical_basis_world": np.dtype(np.float64),
        "leg_length_m": np.dtype(np.float64),
    }
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"{path}: {name} has dtype {arrays[name].dtype}, expected {dtype}")
    if joint_names != JOINT_NAMES or channel_names != CHANNEL_NAMES:
        raise ValueError(f"{path}: joint or channel order mismatch")
    for name in (
        "coordinates",
        "coordinates_m",
        "canonical_times_s",
        "pelvis_world_m",
        "world_to_body_transform",
        "body_to_world_transform",
        "physical_basis_world",
        "leg_length_m",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: non-finite values in {name}")
    if float(leg_length) <= 0 or not np.all(coordinates[~valid] == 0) or not np.all(coordinates_m[~valid] == 0):
        raise ValueError(f"{path}: invalid scale or nonzero invalid coordinates")
    if not np.allclose(body_to_world, world_to_body.T, atol=1e-8):
        raise ValueError(f"{path}: body/world transforms are not inverses")
    if not np.allclose(world_to_body @ world_to_body.T, np.eye(3), atol=1e-8):
        raise ValueError(f"{path}: world_to_body_transform is not orthogonal")
    if not np.isclose(np.linalg.det(physical_basis), 1.0, atol=1e-8):
        raise ValueError(f"{path}: physical basis is not right-handed")

    source = provenance["source"]
    conversion = provenance["conversion"]
    schema = provenance["schema"]
    resampling = provenance["resampling"]
    body_model = provenance["body_model"]
    expected_source = {
        "relative_path": row["relative_path"],
        "source_dataset": row["source_dataset"],
        "subject_id_candidate": row["subject_id_candidate"],
        "identity": row["identity"],
        "split": row["split"],
        "motion_id": row["motion_id"],
        "gender": row["gender"],
        "sha256": row["source_sha256"],
    }
    for key, expected in expected_source.items():
        if str(source.get(key, "")) != str(expected):
            raise ValueError(f"{path}: provenance source {key} mismatch")
    if source.get("sha256_verified_against_current_file") is not True:
        raise ValueError(f"{path}: source SHA-256 is not marked verified")
    if conversion.get("fingerprint") != row["conversion_fingerprint"]:
        raise ValueError(f"{path}: conversion fingerprint mismatch")
    if schema.get("name") != SCHEMA or tuple(schema.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError(f"{path}: provenance schema or joint order mismatch")
    if tuple(schema.get("channel_names", ())) != CHANNEL_NAMES:
        raise ValueError(f"{path}: provenance channel order mismatch")
    if resampling.get("canonical_fps") != 30 or resampling.get("canonical_frames") != frames:
        raise ValueError(f"{path}: provenance resampling mismatch")
    if body_model.get("body_model_sha256") != row["body_model_sha256"]:
        raise ValueError(f"{path}: body-model hash mismatch")
    if body_model.get("dmpl_model_sha256") != row["dmpl_model_sha256"]:
        raise ValueError(f"{path}: DMPL-model hash mismatch")
    return coordinates.copy(), valid.copy()


class Core11WindowDataset(Dataset):
    """Load Core11 windows lazily with a bounded per-worker sequence cache."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        window_index: pd.DataFrame,
        tensor_root: Path,
        *,
        split: str,
        cache_sequences: int = 2,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(split)
        self.manifest = manifest.to_dict(orient="records")
        selected = window_index.loc[window_index["split"] == split]
        self.windows = selected.to_dict(orient="records")
        self.tensor_root = Path(tensor_root).expanduser().resolve()
        self.split = split
        self.cache_sequences = max(int(cache_sequences), 0)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _sequence(self, sequence_index: int) -> tuple[np.ndarray, np.ndarray]:
        if sequence_index in self._cache:
            value = self._cache.pop(sequence_index)
            self._cache[sequence_index] = value
            return value
        row = self.manifest[sequence_index]
        if row["split"] != self.split:
            raise ValueError("Window/manifest split mismatch")
        path = _safe_tensor_path(self.tensor_root, row["tensor_relative_path"])
        value = _archive_arrays(path, row)
        if self.cache_sequences:
            self._cache[sequence_index] = value
            while len(self._cache) > self.cache_sequences:
                self._cache.popitem(last=False)
        return value

    def __getitem__(self, item: int) -> dict:
        entry = self.windows[item]
        coordinates, valid = self._sequence(int(entry["sequence_index"]))
        start = int(entry["start_frame"])
        stop = start + WINDOW_FRAMES
        coordinates = coordinates[start:stop]
        valid = valid[start:stop]
        if coordinates.shape != (WINDOW_FRAMES, 11, 3):
            raise ValueError(f"Incomplete indexed window {entry['window_id']}")
        valid_patch = valid.reshape(
            WINDOW_FRAMES // TIME_PATCH_FRAMES, TIME_PATCH_FRAMES, 11
        ).mean(axis=1) >= MINIMUM_VALID_JOINT_FRACTION
        return {
            "coordinates": torch.from_numpy(coordinates.copy()),
            "valid": torch.from_numpy(valid_patch),
            "window_id": entry["window_id"],
        }


def validate_archives(
    manifest: pd.DataFrame,
    tensor_root: Path,
    relative_paths: Iterable[str] | None = None,
) -> int:
    selected = manifest
    if relative_paths is not None:
        wanted = set(relative_paths)
        selected = manifest.loc[manifest["tensor_relative_path"].isin(wanted)]
        missing = wanted.difference(selected["tensor_relative_path"])
        if missing:
            raise ValueError(f"Archive paths absent from manifest: {sorted(missing)[:3]}")
    for row in selected.to_dict(orient="records"):
        path = _safe_tensor_path(tensor_root, row["tensor_relative_path"])
        _archive_arrays(path, row)
    return len(selected)


class FixedBatchPlan:
    """Recreate the same cyclic shuffled window order for every model variant."""

    def __init__(self, n_items: int, batch_size: int, updates: int, seed: int) -> None:
        if n_items < 1 or batch_size < 1 or updates < 1:
            raise ValueError("n_items, batch_size, and updates must be positive")
        self.n_items = int(n_items)
        self.batch_size = int(batch_size)
        self.updates = int(updates)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.updates

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(self.n_items)
        cursor = 0
        for _ in range(self.updates):
            if cursor + self.batch_size > self.n_items:
                tail = order[cursor:]
                order = rng.permutation(self.n_items)
                need = self.batch_size - len(tail)
                batch = np.concatenate([tail, order[:need]])
                cursor = need
            else:
                batch = order[cursor : cursor + self.batch_size]
                cursor += self.batch_size
            yield batch.tolist()


def make_train_loader(
    dataset: Core11WindowDataset,
    config: TrainConfig,
    updates: int,
    seed: int,
    *,
    num_workers: int,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_sampler": FixedBatchPlan(len(dataset), config.batch_size, updates, seed + 17),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)


def _augment_canonical(
    x: torch.Tensor,
    max_degrees: float,
    generator: torch.Generator,
) -> torch.Tensor:
    view = x.clone()
    present = view.abs().sum(dim=-1) > 1e-8
    angles = (torch.rand(len(x), generator=generator) * 2 - 1) * math.radians(max_degrees)
    angles = angles.to(x.device)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    forward, lateral = view[..., 0].clone(), view[..., 2].clone()
    view[..., 0] = cosine[:, None, None] * forward + sine[:, None, None] * lateral
    view[..., 2] = -sine[:, None, None] * forward + cosine[:, None, None] * lateral
    return view.masked_fill(~present[..., None], 0.0)


def train_streaming_variant(
    model: OrbitJEPA,
    dataset: Core11WindowDataset,
    config: TrainConfig,
    device: torch.device,
    updates: int,
    seed: int,
    *,
    num_workers: int,
) -> tuple[OrbitJEPA, VICRegProjector, torch.optim.Optimizer, pd.DataFrame, float]:
    """Train one variant on the shared deterministic batch/mask/view schedule."""

    seed_everything(seed)
    model = model.to(device)
    projector = VICRegProjector(config.embed_dim).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    view_generator = torch.Generator(device="cpu").manual_seed(seed + 200_000)
    loader = make_train_loader(dataset, config, updates, seed, num_workers=num_workers)
    history = []
    start = time.perf_counter()
    model.train()
    for step, batch in enumerate(loader):
        canonical = batch["coordinates"].to(device, non_blocking=True)
        valid = batch["valid"]
        orbit_valid = valid & mirror_validity(valid)
        target_mask = sample_mask(
            orbit_valid, config.mask_fraction, mask_generator, config.mask_joints
        ).to(device, non_blocking=True)
        first_orbit = lift_orbit(
            _augment_canonical(canonical, config.max_yaw_degrees, view_generator),
            config.mirror_pairs,
            config.mirror_channel,
        )
        second_orbit = lift_orbit(
            _augment_canonical(canonical, config.max_yaw_degrees, view_generator),
            config.mirror_pairs,
            config.mirror_channel,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predicted, targets = model(first_orbit, target_mask)
            jepa = sjepa_cross_entropy(predicted, targets, model.target_center)
            even_terms, odd_terms = orbit_vicreg(model, projector, first_orbit, second_orbit)
            vicreg = even_terms[0] + config.odd_vicreg_weight * odd_terms[0]
            total = jepa + config.vicreg_weight * vicreg
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss at update {step + 1}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        momentum = cosine_ema(step, updates, config.ema_start)
        model.update_target(momentum)
        model.update_center(targets.detach())
        history.append(
            {
                "update": step + 1,
                "total_loss": float(total.detach().cpu()),
                "jepa_loss": float(jepa.detach().cpu()),
                "vicreg_loss": float(vicreg.detach().cpu()),
                "even_variance": float(even_terms[2].detach().cpu()),
                "odd_variance": float(odd_terms[2].detach().cpu()),
                "ema_momentum": momentum,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    return model.cpu(), projector.cpu(), optimizer, pd.DataFrame(history), time.perf_counter() - start


def core11_train_config(profile: str) -> TrainConfig:
    if profile not in {"smoke", "full"}:
        raise ValueError("AMASS profile must be smoke or full")
    smoke = profile == "smoke"
    return TrainConfig(
        profile=f"amass-{profile}",
        frames=WINDOW_FRAMES,
        stride=STRIDE_FRAMES,
        segment_length=TIME_PATCH_FRAMES,
        embed_dim=32 if smoke else 96,
        encoder_depth=2 if smoke else 4,
        predictor_depth=1 if smoke else 2,
        heads=4 if smoke else 8,
        batch_size=8 if smoke else 32,
        epochs=1 if smoke else 100,
        learning_rate=3e-4 if smoke else 2e-4,
        weight_decay=0.05,
        mask_fraction=0.60,
        ema_start=0.996,
        vicreg_weight=0.05,
        odd_vicreg_weight=1.0,
        max_yaw_degrees=8.0,
        amp=not smoke,
        joints=11,
        mask_joints=tuple(range(11)),
        mirror_pairs=MIRROR_PAIRS,
        mirror_channel=MIRROR_CHANNEL,
    )


def optimizer_updates(config: TrainConfig, train_windows: int, override: int | None = None) -> int:
    return int(override or config.epochs * math.ceil(train_windows / config.batch_size))


def atomic_torch_save(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: OrbitJEPA,
    projector: VICRegProjector,
    optimizer: torch.optim.Optimizer,
    *,
    variant: str,
    seed: int,
    config: TrainConfig,
    manifest_sha256: str,
    window_sha256: str,
    updates: int,
    runtime: Mapping | None = None,
) -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    return {
        "model_state": model.state_dict(),
        "projector_state": projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": {
            "variant": variant,
            "seed": seed,
            "train_config": asdict(config),
            "manifest_sha256": manifest_sha256,
            "window_index_sha256": window_sha256,
            "conversion_fingerprints": VALID_FINGERPRINTS,
            "optimizer_updates": updates,
            "checkpoint_lineage": {
                "stage": "amass_core11_pretraining",
                "parent_checkpoint": None,
            },
            "runtime": dict(runtime or {}),
        },
    }
