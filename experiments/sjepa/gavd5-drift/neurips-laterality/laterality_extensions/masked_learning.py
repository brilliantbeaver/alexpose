"""Small, controlled masked-prediction experiments for research tutorials.

This module does not modify the registered laterality protocol or its artifacts.
The default experiment is a synthetic software demonstration. Real experiments
require an explicit setting, reuse source partitions read-only, and produce
exploratory results rather than retroactively extending the registered study.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge

from laterality.config import SUITE_ROOT, canonical_json_digest, load_context
from laterality.data import load_cohort, prepare_cohort
from laterality.evaluation import WeightedScaler, laterality_features
from laterality.geometry import FULL_MIRROR_PAIRS
from laterality.metrics import source_weights, weighted_mae, weighted_r2
from laterality.model import (
    SJEPAGait,
    VICRegProjector,
    anatomical_reflect_tensor,
    authorized_pool,
    geometric_view,
    sjepa_cross_entropy,
    valid_patches,
    vicreg_loss,
)
from laterality.splitting import build_source_splits, get_fold, load_splits
from laterality.training import (
    resolve_device,
    set_reproducible_seed,
    source_balanced_epoch_batches,
)


GAIT_JOINTS = (11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
PROBE_PAIRS = ((11, 12), (25, 26), (27, 28), (29, 30), (31, 32))
SymmetryPenalty = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]
LearningProgress = Callable[[dict[str, Any]], None]
MASKING_RUNNER_API_VERSION = 2
_CACHE_SCHEMA = "neurips_laterality_masking_comparison/v1"


@dataclass(frozen=True)
class LearningSettings:
    seed: int = 7
    fold: int = 0
    steps: int = 8
    batch_size: int = 5
    embed_dim: int = 16
    encoder_depth: int = 1
    predictor_depth: int = 1
    heads: int = 2
    segment_length: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    mask_fraction: float = 0.5
    mask_policy: str = "gait"
    random_subset_seed: int = 31
    ridge_alpha: float = 1.0
    reflection_probability: float = 0.0
    symmetry_weight: float = 0.0
    vicreg_weight: float = 0.01
    ema_momentum: float = 0.99
    device: str = "cpu"
    confirm_real_run: bool = False

    def validate(self) -> None:
        if self.steps < 1 or self.batch_size < 2:
            raise ValueError("Use at least one update and two examples per batch")
        if self.heads < 1 or self.embed_dim < 2 or self.embed_dim % self.heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        if self.segment_length < 1 or self.encoder_depth < 1 or self.predictor_depth < 1:
            raise ValueError("Model depths and segment length must be positive")
        if not 0 < self.mask_fraction < 1:
            raise ValueError("Mask fraction must be strictly between zero and one")
        if self.mask_policy not in {"gait", "uniform", "random_subset"}:
            raise ValueError("Mask policy must be gait, uniform, or random_subset")
        if not 0 <= self.reflection_probability <= 1:
            raise ValueError("Reflection probability must lie in [0, 1]")
        if self.symmetry_weight < 0 or self.vicreg_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.ridge_alpha <= 0:
            raise ValueError("Learning rate and ridge penalty must be positive")
        if not all(np.isfinite(value) for value in (
            self.learning_rate, self.weight_decay, self.ridge_alpha,
            self.symmetry_weight, self.vicreg_weight,
        )):
            raise ValueError("Loss and optimizer settings must be finite")
        if not 0 <= self.ema_momentum < 1:
            raise ValueError("Teacher momentum must lie in [0, 1)")
        if self.device != "auto":
            try:
                device = torch.device(self.device)
            except (RuntimeError, ValueError) as error:
                raise ValueError(f"Invalid learning device: {self.device!r}") from error
            if device.type not in {"cpu", "cuda", "mps"}:
                raise ValueError("Learning device must be auto, cpu, cuda, or mps")


def resolve_learning_device(requested: str = "auto") -> torch.device:
    """Resolve one explicit device and never hide a failed accelerator request."""
    device = resolve_device() if requested == "auto" else torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Learning device must resolve to cpu, cuda, or mps")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            cuda_build = torch.version.cuda
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
            runtime = (
                f"Active interpreter: {sys.executable}. "
                f"PyTorch: {torch.__version__}; compiled CUDA: {cuda_build or 'none (CPU-only build)'}."
            )
            if cuda_build is None:
                remedy = (
                    " This kernel imported a CPU-only PyTorch build. Select the "
                    "'GAVD5 CUDA (PyTorch 2.13)' kernel (or install a CUDA-enabled "
                    "PyTorch build into this exact interpreter), restart the kernel, and rerun."
                )
            else:
                visibility = (
                    f" CUDA_VISIBLE_DEVICES={visible_devices!r}."
                    if visible_devices is not None else ""
                )
                remedy = (
                    " This CUDA-enabled PyTorch build could not initialize a GPU."
                    f"{visibility} Check that the NVIDIA driver is available to this process, "
                    "that CUDA_VISIBLE_DEVICES has not hidden every GPU, then restart the kernel."
                )
            raise RuntimeError(
                "CUDA was requested for masking training but is unavailable. " + runtime + remedy
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested for masking training but is unavailable")
    return device


def learning_hardware_report(requested: str = "auto") -> dict[str, Any]:
    """Describe the effective backend and expose a CPU-only CUDA install.

    Device auto-selection can only see accelerators compiled into PyTorch.  On
    Windows it is therefore possible for ``nvidia-smi`` to see a capable GPU
    while a PyPI CPU wheel makes ``torch.cuda.is_available()`` false.  The
    notebook needs to make that expensive failure mode visible *before* a
    150,000-update grid starts.
    """
    nvidia = []
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for index, line in enumerate(completed.stdout.splitlines()):
                values = [value.strip() for value in line.split(",")]
                if len(values) == 4:
                    name, memory_mib, driver, capability = values
                    nvidia.append({
                        "index": index,
                        "name": name,
                        "memory_mib": int(memory_mib),
                        "driver": driver,
                        "compute_capability": capability,
                    })
        except (OSError, subprocess.SubprocessError, ValueError):
            # PyTorch remains the authority for execution.  A failed optional
            # hardware inventory must not make CPU or MPS notebooks unusable.
            nvidia = []

    error = None
    try:
        resolved = resolve_learning_device(requested)
    except (RuntimeError, ValueError) as caught:
        resolved = None
        error = str(caught)
    report: dict[str, Any] = {
        "requested_device": str(requested),
        "resolved_device": str(resolved) if resolved is not None else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "nvidia_gpus": nvidia,
        "resolution_error": error,
        "precision": "float32",
        "compile": False,
    }
    if nvidia and not torch.cuda.is_available():
        report["status"] = "NVIDIA GPU detected, but this PyTorch build cannot use CUDA"
        report["action"] = (
            "Install a CUDA-enabled PyTorch wheel in this exact notebook environment "
            "using https://pytorch.org/get-started/locally/, restart the kernel, and "
            "verify torch.cuda.is_available() before real training."
        )
    elif error:
        report["status"] = "Requested accelerator is unavailable"
        report["action"] = error
    elif resolved is not None and resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        report["status"] = "CUDA ready"
        report["active_accelerator"] = {
            "index": int(index),
            "name": torch.cuda.get_device_name(index),
            "memory_mib": int(torch.cuda.get_device_properties(index).total_memory // 2**20),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
        }
    elif resolved is not None and resolved.type == "mps":
        report["status"] = "Apple MPS ready"
    else:
        report["status"] = "CPU execution"
        report["action"] = (
            "Real training will be slow on CPU; use an explicit accelerator when available."
        )
    return report


def configure_learning_runtime(
    device: str | torch.device,
    *,
    cpu_threads: int | None = None,
) -> dict[str, Any]:
    """Configure the CPU fallback without changing accelerator-side numerics."""
    resolved = resolve_learning_device(str(device))
    if resolved.type == "cpu":
        requested = cpu_threads
        if requested is None:
            configured = os.getenv("LATERALITY_RESEARCH_CPU_THREADS")
            try:
                requested = int(configured) if configured else min(8, os.cpu_count() or 1)
            except ValueError as error:
                raise ValueError(
                    "LATERALITY_RESEARCH_CPU_THREADS must be a positive integer"
                ) from error
        if int(requested) < 1:
            raise ValueError("LATERALITY_RESEARCH_CPU_THREADS must be a positive integer")
        torch.set_num_threads(int(requested))
    return {
        "device": str(resolved),
        "cpu_threads": torch.get_num_threads() if resolved.type == "cpu" else None,
        "accelerated": resolved.type in {"cuda", "mps"},
    }


@dataclass(frozen=True)
class LearningDataset:
    xyz: np.ndarray
    valid: np.ndarray
    targets: np.ndarray
    source_ids: np.ndarray
    sequence_ids: np.ndarray
    train_sources: tuple[str, ...]
    test_sources: tuple[str, ...]
    synthetic: bool
    fold: int = 0
    dataset_note: str = "synthetic demonstration"
    cohort_digest: str = ""
    split_digest: str = ""

    def validate(self, segment_length: int = 4) -> None:
        n = len(self.xyz)
        if self.xyz.ndim != 4 or self.xyz.shape[2:] != (33, 3):
            raise ValueError("xyz must have shape [sequences, frames, 33, 3]")
        if self.valid.shape != self.xyz.shape[:-1]:
            raise ValueError("Coordinate and validity shapes disagree")
        if self.valid.dtype != np.dtype(bool):
            raise ValueError("Validity must be a boolean array")
        if self.xyz.shape[1] % segment_length:
            raise ValueError("Frame count must be divisible by segment length")
        if any(np.asarray(a).shape != (n,) for a in (self.targets, self.source_ids, self.sequence_ids)):
            raise ValueError("Targets and identifiers need one value per sequence")
        if len(set(map(str, self.sequence_ids))) != n:
            raise ValueError("Sequence identifiers must be unique")
        train, test = set(self.train_sources), set(self.test_sources)
        if not train or not test or train & test:
            raise ValueError("Training and test source sets must be nonempty and disjoint")
        if train | test != set(map(str, self.source_ids)):
            raise ValueError("Declared source roles must cover the dataset exactly")
        if not np.isfinite(self.targets).all() or not np.isfinite(self.xyz[self.valid]).all():
            raise ValueError("Observed coordinates and probe targets must be finite")
        patches = np.asarray(self.valid, bool).reshape(n, -1, segment_length, 33).all(axis=2)
        if np.any(patches[:, :, GAIT_JOINTS].sum(axis=(1, 2)) < 2):
            raise ValueError("Every sequence needs at least two eligible gait patches")

    @property
    def train_rows(self) -> np.ndarray:
        return np.flatnonzero(np.isin(self.source_ids.astype(str), self.train_sources))

    @property
    def test_rows(self) -> np.ndarray:
        return np.flatnonzero(np.isin(self.source_ids.astype(str), self.test_sources))


def load_learning_dataset(*, real: bool = False, fold: int = 0) -> LearningDataset:
    """Load one source-disjoint experiment without writing a cohort or split.

    Setting ``real=True`` deliberately requests the registered local cohort.
    Missing/stale artifacts raise their original validation error; no synthetic
    data are silently substituted and no existing artifact is rebuilt.
    """
    context = load_context(profile="paper" if real else "smoke")
    if real:
        cohort = load_cohort(context)
        splits = load_splits(context, cohort)
    else:
        cohort = prepare_cohort(context)
        splits = build_source_splits(
            cohort.table,
            context.protocol["data"]["conditions"],
            context.protocol["splits"]["outer_folds"],
            context.protocol["splits"]["inner_folds"],
            context.protocol["splits"]["seed"],
        )
    partition = get_fold(splits, fold)
    dataset = LearningDataset(
        xyz=cohort.model_xyz,
        valid=cohort.model_valid,
        targets=cohort.targets,
        source_ids=cohort.table["video_id"].astype(str).to_numpy(),
        sequence_ids=cohort.table["sequence_id"].astype(str).to_numpy(),
        train_sources=tuple(partition["train_sources"]),
        test_sources=tuple(partition["test_sources"]),
        synthetic=not real,
        fold=int(fold),
        dataset_note=(
            "Exploratory reuse of the registered GAVD source partition; no unseen-person claim"
            if real else "Synthetic movement fixture; software demonstration only"
        ),
        cohort_digest=cohort.cohort_digest,
        split_digest=str(splits.get("split_digest") or canonical_json_digest(splits)),
    )
    dataset.validate()
    return dataset


def random_target_subset(seed: int, joints: int = 33) -> tuple[int, ...]:
    return tuple(sorted(map(int, np.random.default_rng(seed).choice(joints, size=12, replace=False))))


def candidate_mask(valid_patch: np.ndarray, policy: str, subset_seed: int = 31) -> np.ndarray:
    valid = np.asarray(valid_patch, dtype=bool)
    if valid.ndim != 3 or valid.shape[-1] != 33:
        raise ValueError("Patch validity must have shape [batch, segments, 33]")
    if policy == "uniform":
        return valid.copy()
    joints = GAIT_JOINTS if policy == "gait" else random_target_subset(subset_seed)
    if policy not in {"gait", "random_subset"}:
        raise ValueError(f"Unsupported mask policy: {policy}")
    allowed = np.zeros(33, dtype=bool)
    allowed[list(joints)] = True
    return valid & allowed[None, None, :]


def matched_target_masks(
    valid_patch: np.ndarray,
    *,
    mask_fraction: float,
    rng: np.random.Generator,
    policies: tuple[str, ...] = ("gait", "uniform"),
    subset_seed: int = 31,
) -> dict[str, np.ndarray]:
    """Give every policy and example the same realized hidden-token count.

    The budget is based on available gait targets and capped by the least
    available policy/example. It always leaves at least one valid token as
    context. Shared random priorities pair draws where candidates overlap.
    """
    if not 0 < mask_fraction < 1 or not policies:
        raise ValueError("Supply a mask fraction in (0, 1) and at least one policy")
    valid = np.asarray(valid_patch, dtype=bool)
    candidates = {p: candidate_mask(valid, p, subset_seed) for p in policies}
    gait_count = candidate_mask(valid, "gait").reshape(len(valid), -1).sum(axis=1)
    available = min(int(c.reshape(len(valid), -1).sum(axis=1).min()) for c in candidates.values())
    context_cap = int(valid.reshape(len(valid), -1).sum(axis=1).min()) - 1
    count = min(max(1, int(np.floor(gait_count.min() * mask_fraction))), available, context_cap)
    if count < 1:
        raise ValueError("The requested policies have no common positive target budget")
    priorities = rng.random(valid.shape)
    output = {}
    for policy, eligible in candidates.items():
        selected = np.zeros_like(valid)
        for row in range(len(valid)):
            candidates_row = np.flatnonzero(eligible[row].ravel())
            order = np.argsort(priorities[row].ravel()[candidates_row], kind="stable")
            selected[row].ravel()[candidates_row[order[:count]]] = True
        output[policy] = selected
    return output


def raw_pose_features(dataset: LearningDataset) -> np.ndarray:
    """Simple direct movement summaries, with missingness kept explicit.

    Speeds here are changes per resized sample, not metres/second. Summaries use
    only valid observations and paired endpoints. Their purpose is to check
    whether a learned representation improves on readily available pose data.
    """
    xyz = np.asarray(dataset.xyz, dtype=np.float64)
    valid = np.asarray(dataset.valid, dtype=bool)
    count = valid.sum(axis=1)
    mean = np.divide(np.where(valid[..., None], xyz, 0).sum(axis=1), count[..., None],
                     out=np.zeros((len(xyz), 33, 3)), where=count[..., None] > 0)
    variance = np.divide(np.where(valid[..., None], (xyz - mean[:, None]) ** 2, 0).sum(axis=1),
                         count[..., None], out=np.zeros_like(mean), where=count[..., None] > 0)
    transition = valid[:, 1:] & valid[:, :-1]
    movement = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    change = np.divide(np.where(transition, movement, 0).sum(axis=1), transition.sum(axis=1),
                       out=np.zeros((len(xyz), 33)), where=transition.sum(axis=1) > 0)
    return np.concatenate((mean.reshape(len(xyz), -1), np.sqrt(variance).reshape(len(xyz), -1),
                           change, valid.mean(axis=1)), axis=1)


def frozen_features(
    encoder: torch.nn.Module,
    dataset: LearningDataset,
    *,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    prepared: _PreparedLearningInputs | None = None,
) -> np.ndarray:
    resolved = resolve_learning_device(str(device))
    encoder.eval()
    encoder.to(resolved)
    if prepared is not None:
        if prepared.device != resolved:
            raise ValueError("Prepared feature tensors are on a different device")
        xyz_all, valid_patch_all = prepared.xyz, prepared.valid_patch
    else:
        clean_xyz = np.where(dataset.valid[..., None], dataset.xyz, 0.0)
        xyz_all = torch.as_tensor(clean_xyz, dtype=torch.float32, device=resolved)
        valid_all = torch.as_tensor(dataset.valid, dtype=torch.bool, device=resolved)
        valid_patch_all = valid_patches(valid_all, encoder.segment_length)
    output = []
    with torch.no_grad():
        for start in range(0, len(dataset.xyz), batch_size):
            xyz = xyz_all[start:start + batch_size]
            patches = valid_patch_all[start:start + batch_size]
            tokens = encoder(xyz, patches).reshape(len(xyz), encoder.segments, 33, encoder.embed_dim)
            output.append(laterality_features(tokens.cpu().numpy(), patches.cpu().numpy(), PROBE_PAIRS))
    return np.concatenate(output)


def evaluate_features(features: np.ndarray, dataset: LearningDataset,
                      *, ridge_alpha: float = 1.0) -> dict[str, Any]:
    """Fit preprocessing and one fixed-penalty probe on training sources only."""
    train, test = dataset.train_rows, dataset.test_rows
    weights = source_weights(dataset.source_ids[train])
    scaler = WeightedScaler(center=True).fit(features[train], weights)
    regression = Ridge(alpha=ridge_alpha, fit_intercept=True)
    regression.fit(scaler.transform(features[train]), dataset.targets[train], sample_weight=weights)
    prediction = regression.predict(scaler.transform(features[test]))
    test_weights = source_weights(dataset.source_ids[test])
    return {
        "r2": weighted_r2(dataset.targets[test], prediction, test_weights),
        "mae": weighted_mae(dataset.targets[test], prediction, test_weights),
        "prediction": prediction,
        "scaler": scaler,
        "readout": regression,
    }


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masking_implementation_digest() -> str:
    """Hash the extension and core modules that determine saved results."""
    laterality_root = Path(__file__).resolve().parents[1] / "laterality"
    paths = [
        Path(__file__).resolve(),
        *[
            laterality_root / name
            for name in (
                "config.py", "data.py", "evaluation.py", "geometry.py",
                "metrics.py", "model.py", "splitting.py", "training.py",
            )
        ],
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dataset_content_digest(dataset: LearningDataset) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("xyz", dataset.xyz),
        ("valid", dataset.valid),
        ("targets", dataset.targets),
    ):
        array = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(json.dumps(list(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    digest.update(json.dumps(dataset.sequence_ids.astype(str).tolist()).encode("utf-8"))
    digest.update(json.dumps(dataset.source_ids.astype(str).tolist()).encode("utf-8"))
    return digest.hexdigest()


def _sampling_schedule(dataset: LearningDataset, settings: LearningSettings) -> list[np.ndarray]:
    table = pd.DataFrame({"video_id": dataset.source_ids.astype(str)})
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 101]))
    per_epoch = math.ceil(len(dataset.train_sources) / settings.batch_size)
    schedule = []
    while len(schedule) < settings.steps:
        for rows, _ in source_balanced_epoch_batches(
            table, list(dataset.train_sources), batch_size=settings.batch_size,
            updates_per_epoch=per_epoch, rng=rng,
        ):
            schedule.append(rows)
            if len(schedule) == settings.steps:
                break
    return schedule


@dataclass(frozen=True)
class _PreparedLearningInputs:
    """Fold-local tensors and paired random choices shared by comparison arms."""

    signature: str
    device: torch.device
    xyz: torch.Tensor
    valid: torch.Tensor
    valid_patch: torch.Tensor
    schedule: np.ndarray
    schedule_rows: torch.Tensor
    target_masks: dict[str, torch.Tensor]
    target_counts: dict[str, list[list[int]]]
    reflection_draws: np.ndarray
    resident_bytes: int


def _prepared_input_signature(
    dataset: LearningDataset,
    settings: LearningSettings,
    budget_policies: tuple[str, ...],
    device: torch.device,
) -> str:
    return canonical_json_digest({
        "fold": dataset.fold,
        "cohort_digest": dataset.cohort_digest,
        "split_digest": dataset.split_digest,
        "data_shape": list(dataset.xyz.shape),
        "train_sources": list(dataset.train_sources),
        "test_sources": list(dataset.test_sources),
        "seed": settings.seed,
        "steps": settings.steps,
        "batch_size": settings.batch_size,
        "segment_length": settings.segment_length,
        "mask_fraction": settings.mask_fraction,
        "random_subset_seed": settings.random_subset_seed,
        "budget_policies": list(budget_policies),
        "device": str(device),
    })


def _prepare_learning_inputs(
    dataset: LearningDataset,
    settings: LearningSettings,
    budget_policies: tuple[str, ...],
    device: torch.device,
) -> _PreparedLearningInputs:
    """Move immutable fold data once and precompute the paired CPU random streams."""
    schedule = np.stack(_sampling_schedule(dataset, settings))
    base_valid_patch = np.asarray(dataset.valid, dtype=bool).reshape(
        len(dataset.xyz), -1, settings.segment_length, 33
    ).all(axis=2)
    mask_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 102]))
    masks_by_policy: dict[str, list[np.ndarray]] = {policy: [] for policy in budget_policies}
    for rows in schedule:
        masks = matched_target_masks(
            base_valid_patch[rows],
            mask_fraction=settings.mask_fraction,
            rng=mask_rng,
            policies=budget_policies,
            subset_seed=settings.random_subset_seed,
        )
        for policy, mask in masks.items():
            masks_by_policy[policy].append(mask)
    reflection_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 103]))
    reflection_draws = reflection_rng.random(schedule.shape)
    clean_xyz = np.where(dataset.valid[..., None], dataset.xyz, 0.0)
    xyz = torch.as_tensor(clean_xyz, dtype=torch.float32, device=device).contiguous()
    valid = torch.as_tensor(dataset.valid, dtype=torch.bool, device=device).contiguous()
    valid_patch = torch.as_tensor(base_valid_patch, dtype=torch.bool, device=device).contiguous()
    schedule_rows = torch.as_tensor(schedule, dtype=torch.long, device=device)
    target_masks = {
        policy: torch.as_tensor(np.stack(policy_masks), dtype=torch.bool, device=device).contiguous()
        for policy, policy_masks in masks_by_policy.items()
    }
    target_counts = {
        policy: [
            mask.reshape(len(mask), -1).sum(axis=1).astype(int).tolist()
            for mask in policy_masks
        ]
        for policy, policy_masks in masks_by_policy.items()
    }
    resident = [xyz, valid, valid_patch, schedule_rows, *target_masks.values()]
    return _PreparedLearningInputs(
        signature=_prepared_input_signature(dataset, settings, budget_policies, device),
        device=device,
        xyz=xyz,
        valid=valid,
        valid_patch=valid_patch,
        schedule=schedule,
        schedule_rows=schedule_rows,
        target_masks=target_masks,
        target_counts=target_counts,
        reflection_draws=reflection_draws,
        resident_bytes=sum(tensor.numel() * tensor.element_size() for tensor in resident),
    )


def _prevalidated_sjepa_forward(
    model: SJEPAGait,
    view: torch.Tensor,
    target: torch.Tensor,
    valid_patch: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the core forward after fold-level mask validation, avoiding a device sync."""
    context = model.view_encoder(view, valid_patch, hide_mask=target_mask)
    predicted = model.predictor(context, target_mask, valid_patch)
    with torch.no_grad():
        target_tokens = model.target_encoder(target, valid_patch)
        selected = target_tokens[target_mask.reshape(len(target), -1)].reshape(
            len(target), -1, target_tokens.shape[-1]
        )
    return predicted, selected


def run_masked_learning(
    dataset: LearningDataset,
    settings: LearningSettings,
    *,
    symmetry_penalty: SymmetryPenalty | None = None,
    budget_policies: tuple[str, ...] = ("gait", "uniform"),
    progress_callback: LearningProgress | None = None,
    _prepared: _PreparedLearningInputs | None = None,
    _control_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a fresh small model; no existing model or result file is touched.

    The symmetry callback receives (view_encoder, coordinates, valid_patches).
    It must return one scalar. The primary evaluated encoder is the EMA target
    encoder, consistently across all variants and its paired initialization.
    """
    settings.validate()
    device = resolve_learning_device(settings.device)
    settings = replace(settings, device=str(device))
    dataset.validate(settings.segment_length)
    if settings.fold != dataset.fold:
        raise ValueError("Settings fold must match the loaded source partition")
    if not dataset.synthetic and not settings.confirm_real_run:
        raise PermissionError("Real-data training requires confirm_real_run=True")
    if settings.symmetry_weight and symmetry_penalty is None:
        raise ValueError("A positive symmetry weight requires a symmetry penalty callback")
    if settings.mask_policy not in budget_policies:
        raise ValueError("The selected mask policy must be included in the shared budget")
    prepared = _prepared or _prepare_learning_inputs(dataset, settings, budget_policies, device)
    expected_prepared = _prepared_input_signature(dataset, settings, budget_policies, device)
    if prepared.signature != expected_prepared or prepared.device != device:
        raise ValueError("Prepared learning inputs do not match this fold, seed, budget, or device")
    set_reproducible_seed(settings.seed)
    model = SJEPAGait(
        frames=dataset.xyz.shape[1], joints=33, coordinate_dim=3,
        segment_length=settings.segment_length, embed_dim=settings.embed_dim,
        encoder_depth=settings.encoder_depth, predictor_depth=settings.predictor_depth,
        heads=settings.heads,
    ).to(device)
    projector = VICRegProjector(settings.embed_dim).to(device)
    initial_encoder = copy.deepcopy(model.target_encoder)
    initial_digest = _state_digest(initial_encoder)
    if _control_features is not None and _control_features["initial_state_digest"] != initial_digest:
        raise AssertionError("Shared initial features do not match this encoder initialization")
    trainable = [*model.view_encoder.parameters(), *model.predictor.parameters(), *projector.parameters()]
    optimizer = torch.optim.AdamW(trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay,
                                 betas=(0.9, 0.95))
    schedule = prepared.schedule
    reflection_rows = (
        torch.as_tensor(
            prepared.reflection_draws < settings.reflection_probability,
            dtype=torch.bool,
            device=device,
        )
        if settings.reflection_probability
        else None
    )
    history = []
    count_history = []
    training_started = time.monotonic()
    model.train()
    projector.train()
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    for step, rows in enumerate(schedule):
        if set(dataset.source_ids[rows].astype(str)) & set(dataset.test_sources):
            raise AssertionError("A test source reached representation training")
        row_index = prepared.schedule_rows[step]
        xyz = prepared.xyz.index_select(0, row_index)
        valid = prepared.valid.index_select(0, row_index)
        patches = prepared.valid_patch.index_select(0, row_index)
        target_mask = prepared.target_masks[settings.mask_policy][step]
        reflected = reflection_rows[step] if reflection_rows is not None else None
        if reflected is not None and reflected.any():
            xyz, valid = anatomical_reflect_tensor(xyz, valid, reflected)
            transformed_mask = target_mask[reflected].clone()
            for left, right in FULL_MIRROR_PAIRS:
                transformed_mask[:, :, [left, right]] = transformed_mask[:, :, [right, left]]
            target_mask = target_mask.clone()
            target_mask[reflected] = transformed_mask
            patches = valid_patches(valid, settings.segment_length)
            if (target_mask & ~patches).any():
                raise AssertionError("Reflection produced an invalid target mask")
        count_history.append(prepared.target_counts[settings.mask_policy][step])
        # A deterministic per-update view stream prevents a callback's random
        # draws from changing the next update's matched augmentations.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(settings.seed + 100_003 * (settings.fold + 1) + step)
            paired_views = geometric_view(
                torch.cat((xyz, xyz), dim=0),
                torch.cat((valid, valid), dim=0),
                max_degrees=8.0,
                translate=0.03,
            )
            view_a, view_b = paired_views.chunk(2, dim=0)
        predicted, target = _prevalidated_sjepa_forward(
            model, view_a, xyz, patches, target_mask,
        )
        prediction_loss = sjepa_cross_entropy(predicted, target, model.target_center)
        # Both unmasked views use the same encoder independently. Stacking them
        # along the batch dimension gives accelerators a larger vectorized call
        # without allowing attention or normalization across examples.
        paired_patches = torch.cat((patches, patches), dim=0)
        paired_tokens = model.view_encoder(
            torch.cat((view_a, view_b), dim=0), paired_patches
        ).reshape(2 * len(rows), model.view_encoder.segments, 33, -1)
        paired_projection = projector(authorized_pool(paired_tokens, paired_patches, GAIT_JOINTS))
        projected_a, projected_b = paired_projection.chunk(2, dim=0)
        regularizer = vicreg_loss(projected_a, projected_b)
        symmetry = xyz.new_zeros(())
        if settings.symmetry_weight:
            symmetry = symmetry_penalty(model.view_encoder, xyz, patches)
            if symmetry.ndim != 0:
                raise ValueError("Symmetry callback must return a scalar")
        loss = prediction_loss + settings.vicreg_weight * regularizer + settings.symmetry_weight * symmetry
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        loss_value, prediction_value, regularizer_value, symmetry_value = torch.stack((
            loss, prediction_loss, regularizer, symmetry,
        )).detach().cpu().tolist()
        if not math.isfinite(loss_value):
            raise FloatingPointError("Non-finite extension training loss")
        if any(p.grad is not None for p in model.target_encoder.parameters()):
            raise AssertionError("Teacher encoder received gradients")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        model.update_target(settings.ema_momentum)
        model.update_center(target, beta=0.9)
        history.append({"step": step + 1, "loss": loss_value,
                        "masked_prediction_loss": prediction_value,
                        "variance_regularizer": regularizer_value,
                        "symmetry_penalty": symmetry_value,
                        "hidden_tokens_per_example": count_history[-1][0]})
        if progress_callback is not None:
            progress_callback({
                "step": step + 1,
                "total_steps": settings.steps,
                "loss": history[-1]["loss"],
            })
    elapsed_training_seconds = time.monotonic() - training_started
    features = {"learned_encoder": frozen_features(
        model.target_encoder, dataset, device=device, prepared=prepared,
    )}
    if _control_features is None:
        features.update({
            "initial_encoder": frozen_features(
                initial_encoder, dataset, device=device, prepared=prepared,
            ),
            "raw_pose": raw_pose_features(dataset),
        })
    else:
        features.update({
            "initial_encoder": _control_features["initial_encoder"],
            "raw_pose": _control_features["raw_pose"],
        })
    metric_rows, prediction_rows = [], []
    for lane, feature in features.items():
        result = evaluate_features(feature, dataset, ridge_alpha=settings.ridge_alpha)
        metric_rows.append({"representation": lane, "r2": result["r2"], "mae": result["mae"]})
        prediction_rows.append(pd.DataFrame({"representation": lane,
            "sequence_id": dataset.sequence_ids[dataset.test_rows],
            "source_id": dataset.source_ids[dataset.test_rows],
            "target": dataset.targets[dataset.test_rows], "prediction": result["prediction"]}))
    training_mean = np.average(dataset.targets[dataset.train_rows],
                               weights=source_weights(dataset.source_ids[dataset.train_rows]))
    mean_prediction = np.full(len(dataset.test_rows), training_mean)
    test_weights = source_weights(dataset.source_ids[dataset.test_rows])
    metric_rows.append({"representation": "training_mean",
        "r2": weighted_r2(dataset.targets[dataset.test_rows], mean_prediction, test_weights),
        "mae": weighted_mae(dataset.targets[dataset.test_rows], mean_prediction, test_weights)})
    prediction_rows.append(pd.DataFrame({"representation": "training_mean",
        "sequence_id": dataset.sequence_ids[dataset.test_rows],
        "source_id": dataset.source_ids[dataset.test_rows],
        "target": dataset.targets[dataset.test_rows], "prediction": mean_prediction}))
    return {"settings": asdict(settings), "model": model, "initial_encoder": initial_encoder,
            "evaluated_encoder": model.target_encoder,
            "features": features, "initial_features": features["initial_encoder"],
            "metrics": pd.DataFrame(metric_rows), "predictions": pd.concat(prediction_rows, ignore_index=True),
            "history": pd.DataFrame(history), "initial_state_digest": initial_digest,
            "source_draw_digest": hashlib.sha256(schedule.tobytes()).hexdigest(),
            "hidden_token_counts": count_history, "synthetic": dataset.synthetic,
            "source_partition": {"fold": dataset.fold,
                                 "train_sources": list(dataset.train_sources),
                                 "test_sources": list(dataset.test_sources),
                                 "sequence_ids": dataset.sequence_ids.astype(str).tolist(),
                                 "source_ids": dataset.source_ids.astype(str).tolist()},
            "data_shape": list(dataset.xyz.shape), "dataset_note": dataset.dataset_note,
            "cohort_digest": dataset.cohort_digest, "split_digest": dataset.split_digest,
            "implementation_digest": masking_implementation_digest(),
            "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "torch": torch.__version__, "sklearn": sklearn.__version__},
            "runtime_device": str(device),
            "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
            "resident_input_bytes": prepared.resident_bytes,
            "elapsed_training_seconds": elapsed_training_seconds,
            "encoder_forward_calls": settings.steps * (4 + 2 * bool(settings.symmetry_weight)),
            "encoder_invocations": settings.steps * (3 + 2 * bool(settings.symmetry_weight)),
            "n_train_sources": len(dataset.train_sources), "n_test_sources": len(dataset.test_sources),
            "n_train_sequences": len(dataset.train_rows), "n_test_sequences": len(dataset.test_rows)}


def run_matched_comparison(
    dataset: LearningDataset,
    settings: LearningSettings,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    symmetry_penalty: SymmetryPenalty | None = None,
    progress_callback: LearningProgress | None = None,
) -> dict[str, Any]:
    """Train paired recipes and fail if a core comparison contract diverges."""
    settings.validate()
    device = resolve_learning_device(settings.device)
    settings = replace(settings, device=str(device))
    variants = dict(variants or {"gait_targets": {"mask_policy": "gait"},
                                 "uniform_targets": {"mask_policy": "uniform"}})
    permitted = {"mask_policy", "random_subset_seed", "reflection_probability", "symmetry_weight"}
    if any(set(change) - permitted for change in variants.values()):
        raise ValueError("Matched variants may change only masking or reflection settings")
    subset_seeds = {int(change.get("random_subset_seed", settings.random_subset_seed))
                    for change in variants.values()}
    if len(subset_seeds) != 1:
        raise ValueError("Compare one random subset per paired experiment; repeat with another seed separately")
    budget = tuple(dict.fromkeys(["gait", *[str(v.get("mask_policy", settings.mask_policy)) for v in variants.values()]]))
    prepared_settings = replace(settings, random_subset_seed=next(iter(subset_seeds)))
    prepared = _prepare_learning_inputs(dataset, prepared_settings, budget, device)
    runs = {}
    shared_controls = None
    variant_count = len(variants)
    for variant_index, (name, change) in enumerate(variants.items()):
        def report_variant_progress(update, *, _name=name, _index=variant_index):
            if progress_callback is not None:
                progress_callback({
                    **update,
                    "variant": _name,
                    "variant_index": _index + 1,
                    "variant_count": variant_count,
                    "comparison_completed_steps": _index * settings.steps + int(update["step"]),
                    "comparison_total_steps": variant_count * settings.steps,
                })

        run = run_masked_learning(
            dataset,
            replace(settings, **dict(change)),
            symmetry_penalty=symmetry_penalty,
            budget_policies=budget,
            progress_callback=report_variant_progress,
            _prepared=prepared,
            _control_features=shared_controls,
        )
        runs[name] = run
        if shared_controls is None:
            shared_controls = {
                "initial_state_digest": run["initial_state_digest"],
                "initial_encoder": run["features"]["initial_encoder"],
                "raw_pose": run["features"]["raw_pose"],
            }
        # Completed arms no longer need accelerator residency. This bounds
        # memory while retaining their full state for atomic artifact saving.
        run["model"].to("cpu")
        run["initial_encoder"].to("cpu")
    pairing = {
        "same_initialization": len({r["initial_state_digest"] for r in runs.values()}) == 1,
        "same_source_draws": len({r["source_draw_digest"] for r in runs.values()}) == 1,
        "same_hidden_token_counts": len({json.dumps(r["hidden_token_counts"]) for r in runs.values()}) == 1,
    }
    if not all(pairing.values()):
        raise AssertionError(f"Matched comparison contract failed: {pairing}")
    comparison = {"runs": runs, "pairing": pairing, "synthetic": dataset.synthetic,
            "variants": variants,
            "metrics": pd.concat([r["metrics"].assign(variant=name) for name, r in runs.items()], ignore_index=True),
            "history": pd.concat([r["history"].assign(variant=name) for name, r in runs.items()], ignore_index=True)}
    comparison["cacheable"] = all(
        float(replace(settings, **change).symmetry_weight) == 0.0
        for change in variants.values()
    )
    comparison["cache_key"] = (
        comparison_cache_key(dataset, settings, variants)
        if comparison["cacheable"]
        else None
    )
    return comparison


def summarize_cross_fitted_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score pooled held-out rows per seed, retaining source-cluster weights.

    This deliberately avoids averaging fold R² values, whose denominators differ.
    It does not generate confidence intervals or account for retraining uncertainty.
    """
    keys = ["seed", "variant", "representation"]
    required = {*keys, "fold", "source_id", "sequence_id", "target", "prediction"}
    if not required <= set(predictions):
        raise ValueError(f"Missing cross-fitted prediction fields: {sorted(required - set(predictions))}")
    rows = []
    reference_sequences = None
    for key, frame in predictions.groupby(keys, sort=True):
        if frame.sequence_id.duplicated().any():
            raise ValueError("A sequence was tested more than once within a seed and representation")
        if (frame.groupby("source_id").fold.nunique() != 1).any():
            raise ValueError("A source crossed held-out fold boundaries")
        sequences = set(frame.sequence_id.astype(str))
        if reference_sequences is None:
            reference_sequences = sequences
        elif sequences != reference_sequences:
            raise ValueError("Every compared recipe and seed must cover identical held-out sequences")
        weights = source_weights(frame.source_id.to_numpy())
        rows.append({**dict(zip(keys, key)), "r2": weighted_r2(frame.target, frame.prediction, weights),
                     "mae": weighted_mae(frame.target, frame.prediction, weights),
                     "test_sources": frame.source_id.nunique(), "test_sequences": len(frame),
                     "folds": frame.fold.nunique()})
    return pd.DataFrame(rows)


def _validated_study_name(study: str) -> str:
    if not study or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in study):
        raise ValueError("Use a simple lowercase study identifier")
    return study


def _normalized_variants(
    variants: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    selected = variants or {
        "gait_targets": {"mask_policy": "gait"},
        "uniform_targets": {"mask_policy": "uniform"},
    }
    normalized = {str(name): dict(changes) for name, changes in selected.items()}
    for label in normalized:
        if not label or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in label):
            raise ValueError("Variant names must be simple lowercase identifiers")
    return normalized


def comparison_cache_key(
    dataset: LearningDataset,
    settings: LearningSettings,
    variants: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Identify an exact fold/seed comparison, including code and runtime lineage."""
    settings.validate()
    dataset.validate(settings.segment_length)
    effective = replace(settings, device=str(resolve_learning_device(settings.device)))
    selected = _normalized_variants(variants)
    if any(float(replace(effective, **change).symmetry_weight) != 0.0
           for change in selected.values()):
        raise ValueError(
            "Content-addressed reuse requires zero symmetry weight; a custom penalty "
            "has no stable cache identity"
        )
    payload = {
        "schema": _CACHE_SCHEMA,
        "implementation_digest": masking_implementation_digest(),
        "dataset_content_digest": _dataset_content_digest(dataset),
        "cohort_digest": dataset.cohort_digest,
        "split_digest": dataset.split_digest,
        "synthetic": dataset.synthetic,
        "fold": dataset.fold,
        "data_shape": list(dataset.xyz.shape),
        "train_sources": list(dataset.train_sources),
        "test_sources": list(dataset.test_sources),
        "settings": asdict(effective),
        # A list preserves execution order even though JSON object keys are sorted.
        "variants": [{"name": name, "changes": change} for name, change in selected.items()],
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
        },
        "cpu_threads": torch.get_num_threads() if effective.device == "cpu" else None,
    }
    return canonical_json_digest(payload)


def comparison_cache_path(
    dataset: LearningDataset,
    settings: LearningSettings,
    *,
    study: str = "masking",
    variants: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    study = _validated_study_name(study)
    key = comparison_cache_key(dataset, settings, variants)
    kind = "synthetic" if dataset.synthetic else "exploratory_real"
    return SUITE_ROOT / "artifacts" / "research_extensions" / study / f"{kind}_{key}"


def _expected_schedule_and_counts(
    dataset: LearningDataset,
    settings: LearningSettings,
    variants: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, list[list[int]]]]:
    schedule = np.stack(_sampling_schedule(dataset, settings))
    source_digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    budget = tuple(dict.fromkeys([
        "gait",
        *[str(change.get("mask_policy", settings.mask_policy)) for change in variants.values()],
    ]))
    subset_seeds = {
        int(change.get("random_subset_seed", settings.random_subset_seed))
        for change in variants.values()
    }
    if len(subset_seeds) != 1:
        raise RuntimeError("Cached comparison declares incompatible random-subset seeds")
    subset_seed = next(iter(subset_seeds))
    base_valid_patch = np.asarray(dataset.valid, dtype=bool).reshape(
        len(dataset.xyz), -1, settings.segment_length, 33
    ).all(axis=2)
    mask_rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.fold, 102]))
    counts = {label: [] for label in variants}
    for rows in schedule:
        masks = matched_target_masks(
            base_valid_patch[rows],
            mask_fraction=settings.mask_fraction,
            rng=mask_rng,
            policies=budget,
            subset_seed=subset_seed,
        )
        for label, change in variants.items():
            policy = str(change.get("mask_policy", settings.mask_policy))
            counts[label].append(masks[policy].reshape(len(rows), -1).sum(axis=1).astype(int).tolist())
    return source_digest, counts


def load_cached_comparison(
    dataset: LearningDataset,
    settings: LearningSettings,
    *,
    study: str = "masking",
    variants: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Load a complete exact-match result; malformed candidates fail closed."""
    selected = _normalized_variants(variants)
    expected_key = comparison_cache_key(dataset, settings, selected)
    study = _validated_study_name(study)
    kind = "synthetic" if dataset.synthetic else "exploratory_real"
    destination = (
        SUITE_ROOT / "artifacts" / "research_extensions" / study
        / f"{kind}_{expected_key}"
    )
    if not destination.exists():
        return None
    if not destination.is_dir():
        raise RuntimeError(f"Masking cache candidate is not a directory: {destination}")
    manifest_path = destination / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Incomplete or malformed masking cache: {destination}") from error
    if manifest.get("schema") != _CACHE_SCHEMA or manifest.get("cache_key") != expected_key:
        raise RuntimeError(f"Masking cache lineage mismatch: {destination}")
    if manifest.get("complete") is not True or manifest.get("synthetic") != dataset.synthetic:
        raise RuntimeError(f"Masking cache is not a complete matching result: {destination}")
    if manifest.get("variants") != selected:
        raise RuntimeError(f"Masking cache variant declaration mismatch: {destination}")
    pairing = manifest.get("pairing")
    expected_pairing = {
        "same_initialization", "same_source_draws", "same_hidden_token_counts",
    }
    if (
        not isinstance(pairing, dict)
        or set(pairing) != expected_pairing
        or not all(value is True for value in pairing.values())
    ):
        raise RuntimeError(f"Cached matched-comparison contract failed: {destination}")
    expected_files = {"metrics.csv", "training.csv"}
    for label in selected:
        expected_files.update({f"{label}_predictions.csv", f"{label}.pt"})
    recorded_files = manifest.get("files")
    optional_files = {"summary.csv", "diagnostics.csv"}
    if (
        not isinstance(recorded_files, dict)
        or not expected_files <= set(recorded_files)
        or set(recorded_files) - expected_files - optional_files
    ):
        raise RuntimeError(f"Masking cache file inventory mismatch: {destination}")
    for name, expected_digest in recorded_files.items():
        path = destination / name
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise RuntimeError(f"Masking cache content digest mismatch: {path}")
    run_metadata = manifest.get("runs")
    if not isinstance(run_metadata, dict) or set(run_metadata) != set(selected):
        raise RuntimeError(f"Masking cache variant inventory mismatch: {destination}")
    effective = replace(settings, device=str(resolve_learning_device(settings.device)))
    expected_partition = {
        "fold": dataset.fold,
        "train_sources": list(dataset.train_sources),
        "test_sources": list(dataset.test_sources),
        "sequence_ids": dataset.sequence_ids.astype(str).tolist(),
        "source_ids": dataset.source_ids.astype(str).tolist(),
    }
    implementation = masking_implementation_digest()
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }
    expected_source_digest, expected_counts = _expected_schedule_and_counts(
        dataset, effective, selected,
    )
    initial_digests = set()
    for label, change in selected.items():
        metadata = run_metadata[label]
        if metadata.get("settings") != asdict(replace(effective, **change)):
            raise RuntimeError(f"Masking cache settings mismatch for {label}: {destination}")
        if metadata.get("source_partition") != expected_partition:
            raise RuntimeError(f"Masking cache source lineage mismatch for {label}: {destination}")
        if metadata.get("cohort_digest") != dataset.cohort_digest:
            raise RuntimeError(f"Masking cache cohort mismatch for {label}: {destination}")
        if metadata.get("split_digest") != dataset.split_digest:
            raise RuntimeError(f"Masking cache split mismatch for {label}: {destination}")
        if metadata.get("implementation_digest") != implementation:
            raise RuntimeError(f"Masking cache implementation mismatch for {label}: {destination}")
        if metadata.get("runtime_versions") != runtime_versions:
            raise RuntimeError(f"Masking cache runtime mismatch for {label}: {destination}")
        if metadata.get("runtime_device") != str(effective.device):
            raise RuntimeError(f"Masking cache device mismatch for {label}: {destination}")
        expected_threads = torch.get_num_threads() if effective.device == "cpu" else None
        if metadata.get("cpu_threads") != expected_threads:
            raise RuntimeError(f"Masking cache CPU-thread mismatch for {label}: {destination}")
        if metadata.get("source_draw_digest") != expected_source_digest:
            raise RuntimeError(f"Masking cache source-draw mismatch for {label}: {destination}")
        if metadata.get("hidden_token_counts") != expected_counts[label]:
            raise RuntimeError(f"Masking cache target-budget mismatch for {label}: {destination}")
        initial_digest = metadata.get("initial_state_digest")
        if not isinstance(initial_digest, str) or len(initial_digest) != 64:
            raise RuntimeError(f"Masking cache initialization digest is malformed for {label}")
        initial_digests.add(initial_digest)
    if len(initial_digests) != 1:
        raise RuntimeError(f"Masking cache paired initialization mismatch: {destination}")
    metrics = pd.read_csv(destination / "metrics.csv")
    history = pd.read_csv(destination / "training.csv")
    if set(metrics.get("variant", ())) != set(selected) or set(history.get("variant", ())) != set(selected):
        raise RuntimeError(f"Masking cache table variants are incomplete: {destination}")
    expected_representations = {"learned_encoder", "initial_encoder", "raw_pose", "training_mean"}
    expected_sequences = dataset.sequence_ids[dataset.test_rows].astype(str).tolist()
    expected_sources = dataset.source_ids[dataset.test_rows].astype(str).tolist()
    expected_targets = dataset.targets[dataset.test_rows]
    runs: dict[str, Any] = {}
    for label in selected:
        predictions = pd.read_csv(
            destination / f"{label}_predictions.csv",
            dtype={"representation": str, "sequence_id": str, "source_id": str},
        )
        if set(predictions.get("representation", ())) != expected_representations:
            raise RuntimeError(f"Masking cache representations are incomplete for {label}")
        for representation, frame in predictions.groupby("representation", sort=False):
            if frame.sequence_id.tolist() != expected_sequences:
                raise RuntimeError(f"Masking cache sequence coverage mismatch for {label}/{representation}")
            if frame.source_id.tolist() != expected_sources:
                raise RuntimeError(f"Masking cache source coverage mismatch for {label}/{representation}")
            if not np.allclose(frame.target.to_numpy(), expected_targets, rtol=0, atol=1e-12):
                raise RuntimeError(f"Masking cache target mismatch for {label}/{representation}")
            if not np.isfinite(frame.prediction.to_numpy()).all():
                raise RuntimeError(f"Masking cache has non-finite predictions for {label}/{representation}")
        run = dict(run_metadata[label])
        run["predictions"] = predictions
        run["metrics"] = metrics[metrics.variant == label].drop(columns="variant").reset_index(drop=True)
        run["history"] = history[history.variant == label].drop(columns="variant").reset_index(drop=True)
        runs[label] = run
    return {
        "runs": runs,
        "pairing": manifest["pairing"],
        "synthetic": dataset.synthetic,
        "variants": selected,
        "metrics": metrics,
        "history": history,
        "cache_key": expected_key,
        "cache_reused": True,
        "artifact_path": destination,
    }


def save_comparison(comparison: dict[str, Any], *, study: str = "masking") -> Path:
    """Atomically save a content-addressed, reusable exploratory comparison.

    No automatic release is implied. Source identifiers remain linkable, so
    this optional output is subject to the project's data-use/release review.
    """
    import tempfile
    study = _validated_study_name(study)
    _normalized_variants(comparison.get("variants"))
    if set(comparison["runs"]) != set(comparison.get("variants", comparison["runs"])):
        raise ValueError("Saved run labels must match the declared comparison variants")
    if comparison.get("cacheable") is not True:
        raise ValueError(
            "This comparison is not cacheable because its custom training callback has no stable identity"
        )
    cache_key = comparison.get("cache_key")
    if not isinstance(cache_key, str) or len(cache_key) != 64:
        raise ValueError("Comparison has no valid content-addressed cache key")
    parent = SUITE_ROOT / "artifacts" / "research_extensions" / study
    parent.mkdir(parents=True, exist_ok=True)
    kind = "synthetic" if comparison["synthetic"] else "exploratory_real"
    destination = parent / f"{kind}_{cache_key}"
    if destination.exists():
        raise FileExistsError(
            f"Comparison cache already exists; validate it with load_cached_comparison: {destination}"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{kind}_{cache_key}.tmp-", dir=parent))
    comparison["metrics"].to_csv(temporary / "metrics.csv", index=False)
    comparison["history"].to_csv(temporary / "training.csv", index=False)
    for name in ("summary", "diagnostics"):
        if name in comparison:
            if not isinstance(comparison[name], pd.DataFrame):
                raise TypeError(f"Optional {name} must be a DataFrame")
            comparison[name].to_csv(temporary / f"{name}.csv", index=False)
    for label, run in comparison["runs"].items():
        run["predictions"].to_csv(temporary / f"{label}_predictions.csv", index=False)
        torch.save({"model_state": run["model"].state_dict(), "settings": run["settings"],
                    "initial_encoder_state": run["initial_encoder"].state_dict(),
                    "synthetic": comparison["synthetic"]}, temporary / f"{label}.pt")
    files = {
        path.name: _sha256_file(path)
        for path in sorted(temporary.iterdir())
        if path.is_file()
    }
    run_fields = (
        "settings", "initial_state_digest", "source_draw_digest",
        "hidden_token_counts", "source_partition", "data_shape", "dataset_note",
        "cohort_digest", "split_digest", "implementation_digest", "runtime_versions",
        "runtime_device", "cpu_threads", "resident_input_bytes",
        "elapsed_training_seconds", "encoder_forward_calls", "encoder_invocations",
        "n_train_sources", "n_test_sources", "n_train_sequences", "n_test_sequences",
    )
    metadata = {
        "schema": _CACHE_SCHEMA,
        "cache_key": cache_key,
        "complete": True,
        "synthetic": comparison["synthetic"],
        "pairing": comparison["pairing"],
        "variants": comparison["variants"],
        "status": (
            "tutorial demonstration"
            if comparison["synthetic"]
            else "exploratory, not registered evidence"
        ),
        "compute_scope": comparison.get(
            "compute_scope", "Matched optimizer updates; vectorized execution is not a FLOP claim"
        ),
        "files": files,
        "runs": {
            label: {key: run[key] for key in run_fields}
            for label, run in comparison["runs"].items()
        },
    }
    # The manifest is written last inside the temporary directory. Renaming the
    # directory then exposes either one complete cache entry or no entry at all.
    (temporary / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.rename(destination)
    return destination
