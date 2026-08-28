"""AMASS Core11 manifest, streaming loader, and matched JEPA training helpers."""

from __future__ import annotations

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

from .gait_parity_jepa import *


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
WINDOW_FRAMES = 64
STRIDE_FRAMES = 32
TIME_PATCH_FRAMES = 4
MINIMUM_VALID_JOINT_FRACTION = 0.95
REQUIRED_ARCHIVE_KEYS = {
    "coordinates",
    "valid",
    "joint_names",
    "channel_names",
}
REQUIRED_MANIFEST_COLUMNS = {
    "tensor_relative_path",
    "identity",
    "split",
    "canonical_fps",
    "canonical_frames",
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
    "WINDOW_FRAMES",
    "Core11WindowDataset",
    "FixedBatchPlan",
    "SyntheticCore11Dataset",
    "atomic_dataframe_to_csv",
    "atomic_torch_save",
    "build_window_index",
    "checkpoint_payload",
    "configure_worker_tensor_sharing",
    "core11_train_config",
    "evaluate_variant",
    "fit_variant",
    "load_conversion_manifest",
    "make_train_loader",
    "make_synthetic_core11_datasets",
    "mirror_validity",
    "optimizer_updates",
    "train_streaming_variant",
    "validate_archives",
    "window_starts",
]


def load_conversion_manifest(path: Path) -> pd.DataFrame:
    """Load the columns needed to construct safe, split-aware training windows."""

    path = Path(path).expanduser().resolve()
    text_columns = REQUIRED_MANIFEST_COLUMNS.difference({"canonical_fps", "canonical_frames"})
    frame = pd.read_csv(path, dtype={name: str for name in text_columns})
    missing = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Conversion manifest is empty")
    if not frame["tensor_relative_path"].is_unique:
        raise ValueError("tensor_relative_path must be unique")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError(f"Unexpected split set: {sorted(frame['split'].unique())}")
    if not np.allclose(frame["canonical_fps"].to_numpy(float), 30.0, rtol=0.0, atol=0.0):
        raise ValueError("Core11 manifest must use exactly 30 fps")
    frame["canonical_frames"] = frame["canonical_frames"].astype(int)
    if (frame["canonical_frames"] < 1).any():
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
        for start in window_starts(int(row["canonical_frames"])):
            rows.append(
                {
                    "window_id": f"{sequence_index}:{start}",
                    "sequence_index": int(sequence_index),
                    "tensor_relative_path": row["tensor_relative_path"],
                    "identity": row["identity"],
                    "split": row["split"],
                    "start_frame": int(start),
                    "window_frames": WINDOW_FRAMES,
                }
            )
    index = pd.DataFrame(rows)
    if index.empty or not index["window_id"].is_unique:
        raise ValueError("Window index is empty or contains duplicate IDs")
    return index.reset_index(drop=True)


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
    valid = arrays["valid"]
    joint_names = tuple(arrays["joint_names"].tolist())
    channel_names = tuple(arrays["channel_names"].tolist())

    expected_shapes = {
        "coordinates": (frames, 11, 3),
        "valid": (frames, 11),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{path}: {name} has shape {arrays[name].shape}, expected {shape}")
    expected_dtypes = {
        "coordinates": np.dtype(np.float32),
        "valid": np.dtype(np.bool_),
    }
    for name, dtype in expected_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"{path}: {name} has dtype {arrays[name].dtype}, expected {dtype}")
    if joint_names != JOINT_NAMES or channel_names != CHANNEL_NAMES:
        raise ValueError(f"{path}: joint or channel order mismatch")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{path}: non-finite coordinates")
    if not np.all(coordinates[~valid] == 0):
        raise ValueError(f"{path}: invalid joints must have zero coordinates")
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


class SyntheticCore11Dataset(Dataset):
    """Small deterministic, split-specific dataset for pipeline smoke checks only."""

    def __init__(self, split: str, size: int, seed: int) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(split)
        self.split = split
        self.size = int(size)
        generator = np.random.default_rng(seed)
        phase = np.linspace(0, 4 * np.pi, WINDOW_FRAMES, endpoint=False)
        coordinates = []
        for item in range(self.size):
            sample = np.zeros((WINDOW_FRAMES, 11, 3), dtype=np.float32)
            phase_shift = generator.uniform(-np.pi, np.pi)
            scale = generator.uniform(0.6, 1.4)
            for joint in range(11):
                amplitude = scale * generator.uniform(0.4, 1.4, size=3)
                sample[:, joint, 0] = amplitude[0] * np.sin(
                    phase + phase_shift + 0.13 * joint
                )
                sample[:, joint, 1] = amplitude[1] * np.cos(
                    phase * generator.uniform(0.7, 1.4) + 0.09 * joint
                )
                sample[:, joint, 2] = amplitude[2] * np.sin(
                    2 * phase + 0.5 * phase_shift + 0.11 * joint
                )
            sample += generator.normal(0, 0.2, sample.shape).astype(np.float32)
            coordinates.append(sample)
        self.coordinates = torch.from_numpy(np.stack(coordinates))
        self.valid = torch.ones(
            self.size,
            WINDOW_FRAMES // TIME_PATCH_FRAMES,
            11,
            dtype=torch.bool,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, item: int) -> dict:
        return {
            "coordinates": self.coordinates[item],
            "valid": self.valid[item],
            "window_id": f"{self.split}:{item}",
        }


def make_synthetic_core11_datasets() -> dict[str, SyntheticCore11Dataset]:
    return {
        "train": SyntheticCore11Dataset("train", 16, 10_007),
        "validation": SyntheticCore11Dataset("validation", 8, 20_007),
        "test": SyntheticCore11Dataset("test", 8, 30_007),
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
    # Linux defaults to the ``file_descriptor`` sharing strategy.  With workers,
    # every prefetched tensor is then sent over a Unix file descriptor.  This
    # training loop constructs a loader per epoch, so avoiding descriptor-backed
    # tensor transfer prevents worker queues from exhausting the Slurm process
    # descriptor limit.
    if num_workers:
        configure_worker_tensor_sharing()
    kwargs = {
        "dataset": dataset,
        "batch_sampler": FixedBatchPlan(len(dataset), config.batch_size, updates, seed + 17),
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers:
        # A loader is deliberately scoped to one epoch.  Persistent workers are
        # only appropriate when the same loader is reused, and otherwise leave
        # file-descriptor-heavy queues alive across epoch boundaries.
        kwargs.update({"persistent_workers": False, "prefetch_factor": 1})
    return DataLoader(**kwargs)


def configure_worker_tensor_sharing() -> str:
    """Use shared-memory files rather than one file descriptor per tensor."""

    strategies = torch.multiprocessing.get_all_sharing_strategies()
    if "file_system" not in strategies:
        raise RuntimeError(
            "PyTorch does not provide the file_system sharing strategy required "
            "for multi-worker AMASS loading"
        )
    torch.multiprocessing.set_sharing_strategy("file_system")
    return torch.multiprocessing.get_sharing_strategy()


def _loader_batches(loader: DataLoader) -> Iterator:
    """Yield batches and promptly close worker processes on every exit path."""

    iterator = iter(loader)
    try:
        yield from iterator
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()


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


def _variant_views(
    model,
    canonical: torch.Tensor,
    valid: torch.Tensor,
    mask_generator: torch.Generator,
    view_generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build matched augmented views without giving standard S-JEPA a mirror branch."""

    base_target_mask = sample_mask(
        valid,
        model.config.mask_fraction,
        mask_generator,
        model.config.mask_joints,
    ).to(canonical.device, non_blocking=True)
    first = _augment_canonical(canonical, model.config.max_yaw_degrees, view_generator)
    second = _augment_canonical(canonical, model.config.max_yaw_degrees, view_generator)
    if model.variant == "standard_sjepa":
        return first, second, base_target_mask
    target_mask = orbit_closed_target_masks(
        base_target_mask, model.config.mirror_pairs
    )
    return (
        lift_orbit(first, model.config.mirror_pairs, model.config.mirror_channel),
        lift_orbit(second, model.config.mirror_pairs, model.config.mirror_channel),
        target_mask,
    )


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
    projector = VICRegProjector(model.config.embed_dim).to(device)
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
    for step, batch in enumerate(_loader_batches(loader)):
        canonical = batch["coordinates"].to(device, non_blocking=True)
        valid = batch["valid"]
        first_view, second_view, target_mask = _variant_views(
            model, canonical, valid, mask_generator, view_generator
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            terms, representation_terms, targets = _objective_terms(
                model, projector, first_view, second_view, target_mask
            )
            total = terms["total_loss"]
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
                "jepa_loss": float(terms["jepa_cross_entropy"].detach().cpu()),
                "vicreg_loss": float(terms["vicreg_loss"].detach().cpu()),
                "ema_momentum": momentum,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        history[-1].update(
            {
                f"{name}_{term_name}": float(value.detach().cpu())
                for name, values in representation_terms.items()
                for term_name, value in values.items()
                if term_name != "features"
            }
        )
    return model.cpu(), projector.cpu(), optimizer, pd.DataFrame(history), time.perf_counter() - start


def _objective_terms(
    model,
    projector: VICRegProjector,
    first_view: torch.Tensor,
    second_view: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[
    dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]], torch.Tensor
]:
    predicted, targets = model(first_view, target_mask)
    cross_entropy, teacher_entropy, kl_divergence = sjepa_distribution_metrics(
        predicted, targets, model.target_center
    )
    if model.variant == "standard_sjepa":
        first_standard = model.encoder(first_view).mean(1)
        second_standard = model.encoder(second_view).mean(1)
        standard_terms = vicreg_terms(
            projector(first_standard), projector(second_standard)
        )
        vicreg_loss = standard_terms[0]
        representation_terms = {
            "standard": {
                "features": first_standard,
                "invariance": standard_terms[1],
                "variance": standard_terms[2],
                "covariance": standard_terms[3],
            }
        }
    else:
        first_even, first_odd = parity_channels(model.encoder(first_view))
        second_even, second_odd = parity_channels(model.encoder(second_view))
        even_terms = vicreg_terms(projector(first_even), projector(second_even))
        odd_terms = vicreg_terms(projector(first_odd), projector(second_odd))
        vicreg_loss = even_terms[0] + model.config.odd_vicreg_weight * odd_terms[0]
        representation_terms = {
            "even": {
                "features": first_even,
                "invariance": even_terms[1],
                "variance": even_terms[2],
                "covariance": even_terms[3],
            },
            "odd": {
                "features": first_odd,
                "invariance": odd_terms[1],
                "variance": odd_terms[2],
                "covariance": odd_terms[3],
            },
        }
    total_loss = cross_entropy + model.config.vicreg_weight * vicreg_loss
    terms = {
        "total_loss": total_loss,
        "jepa_cross_entropy": cross_entropy,
        "teacher_entropy": teacher_entropy,
        "kl_divergence": kl_divergence,
        "vicreg_loss": vicreg_loss,
    }
    for name, values in representation_terms.items():
        terms.update(
            {
                f"{name}_{term_name}": value
                for term_name, value in values.items()
                if term_name != "features"
            }
        )
    return terms, representation_terms, targets


def _finish_metrics(
    totals: dict[str, float],
    weight: int,
    representation_features: Mapping[str, list[torch.Tensor]],
) -> dict[str, float]:
    metrics = {name: value / max(weight, 1) for name, value in totals.items()}
    for parity, features in representation_features.items():
        health = representation_metrics(torch.cat(features, dim=0))
        metrics.update({f"{parity}_{name}": value for name, value in health.items()})
    return metrics


def _accumulate_terms(
    totals: dict[str, float], terms: dict[str, torch.Tensor], weight: int
) -> None:
    for name, value in terms.items():
        totals[name] = totals.get(name, 0.0) + float(value.detach().float().cpu()) * weight


def _train_epoch(
    model: OrbitJEPA,
    projector: VICRegProjector,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    epoch: int,
    seed: int,
    num_workers: int,
    global_update: int,
    total_updates: int,
) -> tuple[dict[str, float], int]:
    config = model.config
    updates = math.ceil(len(dataset) / config.batch_size)
    loader = make_train_loader(
        dataset,
        config,
        updates,
        seed + epoch * 10_003,
        num_workers=num_workers,
    )
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 100_000 + epoch)
    view_generator = torch.Generator(device="cpu").manual_seed(seed + 200_000 + epoch)
    amp_enabled = bool(config.amp and device.type == "cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable += list(projector.parameters())
    totals: dict[str, float] = {}
    representation_features: dict[str, list[torch.Tensor]] = {}
    weight = 0
    model.train()
    projector.train()
    for batch in _loader_batches(loader):
        canonical = batch["coordinates"].to(device, non_blocking=True)
        valid = batch["valid"]
        first_view, second_view, target_mask = _variant_views(
            model, canonical, valid, mask_generator, view_generator
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            terms, representations, targets = _objective_terms(
                model, projector, first_view, second_view, target_mask
            )
        if not all(torch.isfinite(value) for value in terms.values()):
            raise FloatingPointError(f"Non-finite training metric in epoch {epoch}")
        scaler.scale(terms["total_loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        momentum = cosine_ema(global_update, total_updates, config.ema_start)
        model.update_target(momentum)
        model.update_center(targets.detach())
        global_update += 1
        batch_weight = len(canonical)
        _accumulate_terms(totals, terms, batch_weight)
        for name, values in representations.items():
            representation_features.setdefault(name, []).append(
                values["features"].detach().float().cpu()
            )
        weight += batch_weight
    metrics = _finish_metrics(totals, weight, representation_features)
    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    metrics["ema_momentum"] = float(momentum)
    metrics["window_exposures"] = float(weight)
    return metrics, global_update


@torch.no_grad()
def evaluate_variant(
    model: OrbitJEPA,
    projector: VICRegProjector,
    dataset: Dataset,
    device: torch.device,
    *,
    seed: int,
    split: str,
    num_workers: int = 0,
    draws: int | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate every arm on an identical deterministic mask/view schedule."""

    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    if num_workers:
        configure_worker_tensor_sharing()
    config = model.config
    draws = int(draws or config.evaluation_draws)
    prior_model_mode, prior_projector_mode = model.training, projector.training
    model.eval()
    projector.eval()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
    )
    split_offset = 0 if split == "validation" else 1_000_000
    totals: dict[str, float] = {}
    representation_features: dict[str, list[torch.Tensor]] = {}
    weight = 0
    audit_coordinates = None
    audit_mask = None
    amp_enabled = bool(config.amp and device.type == "cuda")
    for draw in range(draws):
        mask_generator = torch.Generator(device="cpu").manual_seed(
            seed + split_offset + 300_000 + draw
        )
        view_generator = torch.Generator(device="cpu").manual_seed(
            seed + split_offset + 400_000 + draw
        )
        for batch in _loader_batches(loader):
            canonical = batch["coordinates"].to(device, non_blocking=True)
            valid = batch["valid"]
            first_view, second_view, target_mask = _variant_views(
                model, canonical, valid, mask_generator, view_generator
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                terms, representations, _ = _objective_terms(
                    model, projector, first_view, second_view, target_mask
                )
            if not all(torch.isfinite(value) for value in terms.values()):
                raise FloatingPointError(f"Non-finite {split} metric")
            batch_weight = len(canonical)
            _accumulate_terms(totals, terms, batch_weight)
            for name, values in representations.items():
                representation_features.setdefault(name, []).append(
                    values["features"].detach().float().cpu()
                )
            weight += batch_weight
            if audit_coordinates is None:
                audit_coordinates = canonical[:2].detach()
                audit_mask = target_mask[:2].detach()
    metrics = _finish_metrics(totals, weight, representation_features)
    if model.variant != "standard_sjepa":
        audit, layer_audit = complete_commutation_audit(
            model, audit_coordinates, audit_mask, device=device
        )
        metrics.update(audit)
    else:
        layer_audit = pd.DataFrame(columns=["layer", "max_abs", "max_rel", "module"])
    model.train(prior_model_mode)
    projector.train(prior_projector_mode)
    return metrics, layer_audit


def checkpoint_is_eligible(
    variant: str,
    metrics: Mapping[str, float],
    minimum_feature_variance: float,
    commutation_tolerance: float = 1e-5,
) -> bool:
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        return False
    if variant == "standard_sjepa":
        return metrics["standard_feature_variance"] > minimum_feature_variance
    if metrics["even_feature_variance"] <= minimum_feature_variance:
        return False
    if metrics["odd_feature_variance"] <= minimum_feature_variance:
        return False
    if variant == "reflection_equivariant":
        audit_names = (
            "online_encoder_commutation_max_abs",
            "target_encoder_commutation_max_abs",
            "predictor_commutation_max_abs",
            "masked_prediction_commutation_max_abs",
        )
        if any(metrics[name] > commutation_tolerance for name in audit_names):
            return False
    return True


def fit_variant(
    model: OrbitJEPA,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    device: torch.device,
    *,
    seed: int,
    output_dir: Path,
    num_workers: int = 0,
    max_epochs: int | None = None,
    patience: int | None = None,
) -> tuple[OrbitJEPA, VICRegProjector, pd.DataFrame, dict, pd.DataFrame]:
    """Train by epochs and select the lowest eligible validation KL checkpoint."""

    config = model.config
    max_epochs = int(max_epochs or config.epochs)
    patience = int(patience or config.early_stopping_patience)
    seed_everything(seed)
    model = model.to(device)
    projector = VICRegProjector(config.embed_dim).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable += list(projector.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    total_updates = max_epochs * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(config.amp and device.type == "cuda")
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed-{seed}_{model.variant}"
    best_path = output_dir / f"{stem}_best.pt"
    history_path = output_dir / f"{stem}_history.csv"
    history = []
    best_kl = float("inf")
    best_epoch = None
    best_metrics = None
    best_layer_audit = None
    stale_epochs = 0
    global_update = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        train_metrics, global_update = _train_epoch(
            model,
            projector,
            train_dataset,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch=epoch,
            seed=seed,
            num_workers=num_workers,
            global_update=global_update,
            total_updates=total_updates,
        )
        validation_metrics, layer_audit = evaluate_variant(
            model,
            projector,
            validation_dataset,
            device,
            seed=seed,
            split="validation",
            num_workers=num_workers,
        )
        eligible = checkpoint_is_eligible(
            model.variant,
            validation_metrics,
            config.minimum_feature_variance,
        )
        row = {"epoch": epoch, "eligible": eligible}
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update(
            {f"validation_{name}": value for name, value in validation_metrics.items()}
        )
        history.append(row)
        payload = checkpoint_payload(
            model,
            projector,
            optimizer,
            variant=model.variant,
            seed=seed,
            config=config,
            updates=global_update,
            epoch=epoch,
            validation_metrics=validation_metrics,
        )
        if eligible and validation_metrics["kl_divergence"] < best_kl:
            best_kl = validation_metrics["kl_divergence"]
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
            best_layer_audit = layer_audit.copy()
            stale_epochs = 0
            atomic_torch_save(best_path, payload)
        else:
            stale_epochs += 1
        # Persist every completed epoch. If Slurm terminates the job, the CSV
        # retains all epochs completed before the interruption.
        atomic_dataframe_to_csv(pd.DataFrame(history), history_path)
        if stale_epochs >= patience:
            break
    wall_seconds = time.perf_counter() - started
    if best_epoch is None:
        raise RuntimeError(f"{model.variant} produced no eligible validation checkpoint")
    best_model, best_projector, metadata = load_checkpoint(best_path)
    result = {
        "variant": model.variant,
        "seed": seed,
        "selected_epoch": best_epoch,
        "epochs_trained": len(history),
        "optimizer_updates": int(metadata["optimizer_updates"]),
        "wall_seconds": wall_seconds,
        "best_checkpoint": str(best_path),
        "validation": best_metrics,
    }
    return best_model, best_projector, pd.DataFrame(history), result, best_layer_audit


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
        epochs=3 if smoke else 100,
        learning_rate=3e-4 if smoke else 2e-4,
        weight_decay=0.05,
        mask_fraction=0.60,
        ema_start=0.996,
        vicreg_weight=0.05,
        odd_vicreg_weight=1.0,
        max_yaw_degrees=8.0,
        amp=not smoke,
        early_stopping_patience=3 if smoke else 12,
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


def atomic_dataframe_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Replace a CSV atomically so interruption cannot corrupt the prior copy."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def checkpoint_payload(
    model: OrbitJEPA,
    projector: VICRegProjector,
    optimizer: torch.optim.Optimizer,
    *,
    variant: str,
    seed: int,
    config: TrainConfig,
    updates: int,
    epoch: int | None = None,
    validation_metrics: Mapping[str, float] | None = None,
) -> dict:
    if variant not in AMASS_VARIANTS:
        raise ValueError(variant)
    return {
        "model_state": model.state_dict(),
        "projector_state": projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metadata": {
            "variant": variant,
            "seed": seed,
            "train_config": asdict(config),
            "paired_mask_contract": (
                PAIRED_MASK_CONTRACT
                if variant != "standard_sjepa"
                else "single-branch-v1"
            ),
            "optimizer_updates": updates,
            "epoch": epoch,
            "validation_metrics": dict(validation_metrics or {}),
        },
    }
