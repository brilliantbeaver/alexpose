"""Bounded training, explicit stage identities and reproducible inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .masking import uniform_mask, forecast_mask
from .model import ModelConfig, SkeletonJEPA, loss_terms


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 40
    batch_size: int = 8
    learning_rate: float = 3e-4
    mask_ratio: float = 0.5
    forecast_fraction: float = 0.0
    context_frames: int = 8
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    coordinate_weight: float = 0.1
    latent_mse_weight: float = 0.0
    kl_weight: float = 0.01
    seed: int = 42
    regime: str = "normal"
    run_id: str = "normal-uniform-v1"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tensor_inputs(x, valid, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(valid, dtype=torch.bool, device=device)


def _identity(x, valid, labels, sources):
    digest = hashlib.sha256()
    for a in (x.detach().cpu().numpy(), valid.detach().cpu().numpy()):
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    digest.update(json.dumps([list(labels), list(sources)]).encode())
    return digest.hexdigest()


def fit(model: SkeletonJEPA, x, valid, config: TrainConfig | None = None, *,
        labels, source_ids, partitions=None):
    """Fit training records, drawing one window per source per step.

    Normal fitting refuses abnormal examples. Adaptation requires a normal-trained
    parent and normal replay. This modifies the passed model: deepcopy the normal
    reference before adaptation. Every further stage requires a new run_id.
    """
    c = config or TrainConfig()
    if c.steps < 1 or c.batch_size < 2 or not 0 <= c.forecast_fraction <= 1:
        raise ValueError("Need positive steps, batch_size >= 2, forecast fraction in [0,1]")
    if c.regime not in {"normal", "adaptation", "forecast"}:
        raise ValueError("regime must be normal, adaptation or forecast")
    if c.regime != "forecast" and c.forecast_fraction:
        raise ValueError("Future masks require a separately identified forecast stage")
    if not c.run_id.strip():
        raise ValueError("A declared run identity is required")
    labels, source_ids = np.asarray(labels, dtype=str), np.asarray(source_ids, dtype=str)
    if len(labels) != len(x) or len(source_ids) != len(x):
        raise ValueError("labels and source_ids must identify every window")
    if partitions is not None and (len(partitions) != len(x) or any(p != "train" for p in partitions)):
        raise ValueError("Validation and test records cannot enter representation training")
    if any(s == "" for s in source_ids):
        raise ValueError("Every window must have a nonempty source identity")
    allowed = {"normal", "parkinsons", "stroke", "cerebral palsy", "myopathic"}
    if not set(labels).issubset(allowed):
        raise ValueError("Labels fall outside the five-condition study")
    if c.regime in {"normal", "forecast"} and set(labels) != {"normal"}:
        raise ValueError("Normal reference and forecast branch use normal training sources only")
    parent = model.run_metadata.copy()
    if parent and parent.get("run_id") == c.run_id:
        raise ValueError("A further training stage requires a new run_id")
    if c.regime == "adaptation":
        if not parent or parent.get("regime") not in {"normal", "adaptation"}:
            raise ValueError("Adaptation requires an explicitly trained normal parent")
        if "normal" not in labels or set(labels) == {"normal"}:
            raise ValueError("Adaptation requires normal replay and abnormal training examples")
    if c.regime == "forecast" and (not parent or c.forecast_fraction == 0):
        raise ValueError("Forecast stage requires a pretrained parent and forecast masks")
    if len(set(source_ids)) < 2:
        raise ValueError("At least two distinct training sources are needed for VICReg batches")
    seed_everything(c.seed)
    device = next(model.parameters()).device
    x, valid = _tensor_inputs(x, valid, device)
    model._validate(x, valid)
    rng = np.random.default_rng(c.seed)
    mask_generator = torch.Generator(device="cpu").manual_seed(c.seed)
    source_rows = {s: np.flatnonzero(source_ids == s) for s in sorted(set(source_ids))}
    normal_rows = {s: rows[labels[rows] == "normal"] for s, rows in source_rows.items()}
    abnormal_rows = {s: rows[labels[rows] != "normal"] for s, rows in source_rows.items()}
    normal_sources = [s for s, rows in normal_rows.items() if len(rows)]
    abnormal_sources = [s for s, rows in abnormal_rows.items() if len(rows)]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=c.learning_rate)
    records = []
    started = time.perf_counter()
    model.train()
    for step in range(c.steps):
        if c.regime == "adaptation":
            chosen_normal = rng.choice(normal_sources, size=min(len(normal_sources), max(1, c.batch_size // 2)), replace=False).tolist()
            available = [s for s in abnormal_sources if s not in chosen_normal]
            if not available:
                raise ValueError("Need disjoint normal and abnormal source examples for replay")
            chosen_abnormal = rng.choice(available, size=min(len(available), c.batch_size-len(chosen_normal)), replace=False).tolist()
            rows = [rng.choice(normal_rows[s]) for s in chosen_normal] + [rng.choice(abnormal_rows[s]) for s in chosen_abnormal]
        else:
            chosen = rng.choice(list(source_rows), size=min(len(source_rows), c.batch_size), replace=False)
            rows = [rng.choice(source_rows[s]) for s in chosen]
        bx, bv = x[rows], valid[rows]
        use_forecast = rng.random() < c.forecast_fraction
        target = forecast_mask(bv, c.context_frames) if use_forecast else uniform_mask(bv, c.mask_ratio, generator=mask_generator)
        optimizer.zero_grad(set_to_none=True)
        output = model(bx, bv, target, posterior=bool(model.config.latent_dim), sample_latent=bool(model.config.latent_dim))
        terms = loss_terms(model, output, bx, bv, target,
                           variance_weight=c.variance_weight, covariance_weight=c.covariance_weight,
                           coordinate_weight=c.coordinate_weight, latent_mse_weight=c.latent_mse_weight,
                           kl_weight=c.kl_weight)
        if not torch.isfinite(terms["loss"]):
            raise FloatingPointError("Non-finite training objective")
        terms["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite training gradient")
        optimizer.step()
        model.update_teacher()
        records.append({"step": step, "mask": "forecast" if use_forecast else "uniform",
                        "sources_in_batch": len(rows),
                        "student_mean_std": float(output["representation"].detach().std(0, unbiased=False).mean().cpu()),
                        **{k: float(v.detach().cpu()) for k,v in terms.items()}})
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ("model.py", "masking.py", "training.py")}
    model.run_metadata = {"run_id": c.run_id, "regime": c.regime, "config": asdict(c),
                          "data_sha256": _identity(x, valid, labels, source_ids), "code_sha256": code_hashes,
                          "sources": [str(s) for s in sorted(source_rows)], "labels": [str(s) for s in sorted(set(labels))],
                          "n_windows": len(x), "elapsed_seconds": time.perf_counter() - started,
                          "parent_run_id": parent.get("run_id"), "parent_data_sha256": parent.get("data_sha256"),
                          "training_history": records, "torch_version": str(torch.__version__)}
    model.eval()
    return records


@torch.no_grad()
def encode(model, x, valid, batch_size=32):
    """Return numpy embeddings, preserving input window order."""
    device = next(model.parameters()).device
    x, valid = _tensor_inputs(x, valid, device)
    return torch.cat([model.encode(x[i:i+batch_size], valid[i:i+batch_size])
                      for i in range(0, len(x), batch_size)]).cpu().numpy()


@torch.no_grad()
def anomaly_scores(model, x, valid, *, context_frames=8, samples=1, seed=42, batch_size=32):
    """Future-only errors and uncalibrated diversity for each window.

    Teachers see outcomes only to construct scoring targets. Predictions use the
    prefix and prior, never the posterior. Dispersion is not calibration evidence.
    """
    if samples < 1 or batch_size < 1 or len(x) < 1:
        raise ValueError("samples, batch_size and number of windows must be positive")
    if len(x) > batch_size:
        chunks = [anomaly_scores(model, x[i:i+batch_size], valid[i:i+batch_size],
                                 context_frames=context_frames, samples=samples,
                                 seed=seed+i, batch_size=batch_size)
                  for i in range(0, len(x), batch_size)]
        return {key: np.concatenate([chunk[key] for chunk in chunks],
                                   axis=1 if key == "samples" else 0)
                for key in chunks[0]}
    device = next(model.parameters()).device
    x, valid = _tensor_inputs(x, valid, device)
    target = forecast_mask(valid, context_frames)
    old = model.training
    model.eval()
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        outputs = [model(x, valid, target, sample_latent=samples > 1) for _ in range(samples)]
    pred = torch.stack([o["coordinates"] for o in outputs])
    latent = torch.stack([o["latent"] for o in outputs])
    coordinates = pred.mean(0)
    latent_mean = latent.mean(0)
    teacher_probability = ((outputs[0]["teacher"] - model.center) / model.config.target_temperature).softmax(-1)
    ce = -(teacher_probability * (latent_mean/model.config.prediction_temperature).log_softmax(-1)).sum(-1)
    squared = (coordinates - x).square().mean(-1)
    dispersion = pred.var(0, unbiased=False).mean(-1)
    def average(v):
        return torch.stack([v[i][target[i]].mean() for i in range(len(x))]).cpu().numpy()
    result = {"latent_ce": average(ce), "coordinate_mse": average(squared),
              "predictive_variance": average(dispersion), "coordinates": coordinates.cpu().numpy(),
              "samples": pred.cpu().numpy(), "target_mask": target.cpu().numpy()}
    model.train(old)
    return result


def save_checkpoint(model, path, *, overwrite=False):
    """Persist weights and identities; refuse accidental historical overwrites."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_config": asdict(model.config), "state_dict": model.state_dict(),
                "metadata": model.run_metadata}, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path, *, device="cpu"):
    """Load only tensor/basic-container checkpoint contents."""
    saved = torch.load(path, map_location=device, weights_only=True)
    cfg = saved["model_config"]
    cfg["joint_ids"] = tuple(cfg["joint_ids"])
    model = SkeletonJEPA(ModelConfig(**cfg)).to(device)
    model.load_state_dict(saved["state_dict"])
    model.run_metadata = saved["metadata"]
    model.eval()
    return model


@torch.no_grad()
def representation_diagnostics(model, x, valid):
    """Report spread and effective rank; no loss threshold certifies no collapse.

    Call on one independent example per training source. Rank is bounded by the
    number of examples minus one. Near-identical teaching fixtures may have very
    small spread even when gradients and anti-collapse terms are working.
    """
    device = next(model.parameters()).device
    x, valid = _tensor_inputs(x, valid, device)
    if len(x) < 2:
        raise ValueError("Representation diagnostics need at least two examples")
    result = {}
    for name in ("online", "teacher"):
        z = model.encode(x, valid, encoder=name)
        centered = z - z.mean(0)
        eigenvalues = torch.linalg.svdvals(centered).square()
        probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        rank = entropy.exp() if eigenvalues.sum() > 1e-12 else x.new_zeros(())
        result[name] = {"mean_std": float(centered.std(0, unbiased=False).mean().cpu()),
                        "effective_rank": float(rank.cpu()), "n_examples": len(x),
                        "maximum_rank": min(len(x)-1, model.config.dim)}
    return result
