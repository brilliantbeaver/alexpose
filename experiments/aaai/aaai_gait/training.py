"""Explicit normal-only pretraining, progressive replay, and frozen probes."""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from .model import ModelConfig, SJEPA

CONDITIONS = ("normal", "parkinsons", "stroke", "cerebral palsy", "myopathic")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Cross-device floating point equality is not promised. CPU is the default
    # for a reproducible tutorial and tests; GPU/MPS is a runtime choice.


def train_model(model: SJEPA, x: np.ndarray | torch.Tensor, metadata: pd.DataFrame, *,
                allowed_labels: Sequence[str] = ("normal",), epochs: int = 5,
                batch_size: int = 16, learning_rate: float = 3e-4, seed: int = 42,
                device: str = "cpu", mode: str = "masked", ema: float = 0.996,
                weight_decay: float = 0.05) -> list[dict]:
    """Train only rows explicitly marked train, using source/class-balanced draws.

    Within each epoch, choose classes uniformly, source videos uniformly within
    class, and windows uniformly within a source. Every batch includes a normal
    replay window. An epoch has ceil(number of eligible windows / batch_size)
    optimizer steps. Consequently labels regulate sampling, not the JEPA loss.
    No validation or test coordinates enter training or centering statistics.
    """
    required = {"label", "split", "source_id"}
    if not required.issubset(metadata.columns) or len(metadata) != len(x):
        raise ValueError("One metadata row per window with label, split, source_id is required")
    if batch_size < 2 or epochs < 1:
        raise ValueError("batch_size >= 2 and epochs >= 1 are required")
    allowed = tuple(dict.fromkeys(allowed_labels))
    if not allowed or "normal" not in allowed or set(allowed) - set(CONDITIONS):
        raise ValueError("Include normal and use only the five defined conditions")
    # Catch a source-video leak even when the caller only wants train rows.
    if metadata.groupby("source_id")["split"].nunique().max() > 1:
        raise ValueError("A source video occurs in more than one split")
    eligible = np.flatnonzero((metadata["split"].to_numpy() == "train") & metadata["label"].isin(allowed).to_numpy())
    found = set(metadata.iloc[eligible]["label"])
    if found != set(allowed):
        raise ValueError(f"Missing training examples for {set(allowed) - found}")
    if len(eligible) < 2:
        raise ValueError("VICReg requires at least two training examples")
    if set(metadata.split.unique()) - {"train", "validation", "val", "test"}:
        raise ValueError("Unknown split name")
    coordinates = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    pools = {}
    for label in allowed:
        # DataFrame index labels need not be positional indices.
        pools[label] = [eligible[np.flatnonzero((metadata.iloc[eligible].label.to_numpy() == label) &
                                             (metadata.iloc[eligible].source_id.to_numpy() == source))]
                        for source in metadata.iloc[eligible].loc[lambda z: z.label == label, "source_id"].unique()]
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    mask_rng = torch.Generator().manual_seed(seed + 1000)
    positional_rng = torch.Generator().manual_seed(seed + 2000)
    model.to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, weight_decay=weight_decay)
    steps = max(1, int(np.ceil(len(eligible) / batch_size)))
    history = []
    for epoch in range(epochs):
        aggregate: dict[str, list[float]] = {}
        for step in range(steps):
            labels = ["normal"] + rng.choice(allowed, size=batch_size - 1).tolist()
            chosen = []
            for label in labels:
                sources = pools[label]
                source_rows = sources[int(rng.integers(len(sources)))]
                chosen.append(int(rng.choice(source_rows)))
            # At least two distinct observations are needed for meaningful
            # variance; drawing with replacement must not silently defeat it.
            if len(set(chosen)) < 2:
                alternatives = eligible[eligible != chosen[0]]
                chosen[-1] = int(rng.choice(alternatives))
            batch = torch.as_tensor(coordinates[chosen], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(batch, generator=mask_rng, positional_generator=positional_rng, mode=mode)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite loss; inspect coordinates and collapse diagnostics")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0,
                                          error_if_nonfinite=True)
            optimizer.step()
            model.update_target(ema, losses["target_mean"])
            for key in ("loss", "prediction", "vicreg", "invariance", "variance", "covariance", "projection_std", "embedding_std", "observed_token_fraction", "supervised_token_fraction"):
                aggregate.setdefault(key, []).append(float(losses[key].detach().cpu()))
        history.append({"epoch": epoch + 1, "steps": steps, "mode": mode,
                        "conditions": list(allowed), "seed": seed,
                        **{key: float(np.mean(values)) for key, values in aggregate.items()}})
    model.eval()
    return history


def progressive_train(normal_model: SJEPA, x: np.ndarray, metadata: pd.DataFrame, *,
                      stages: Sequence[str] = CONDITIONS[1:], **kwargs) -> tuple[SJEPA, list[dict]]:
    """Copy the normal anchor, then add conditions cumulatively with normal replay.

    The caller should pre-register the stage order. This function never mutates
    the supplied normal model; its anomaly scores remain anchored to normal.
    """
    if set(stages) - set(CONDITIONS[1:]) or len(set(stages)) != len(stages):
        raise ValueError("Stages must be distinct abnormal conditions from CONDITIONS")
    updated = copy.deepcopy(normal_model)
    allowed = ["normal"]
    history = []
    for condition in stages:
        allowed.append(condition)
        stage_history = train_model(updated, x, metadata, allowed_labels=allowed, **kwargs)
        history.extend({**row, "stage": condition} for row in stage_history)
    return updated, history


@torch.no_grad()
def embed_windows(model: SJEPA, x: np.ndarray, *, batch_size: int = 64,
                  device: str = "cpu") -> np.ndarray:
    if len(x) == 0:
        raise ValueError("Cannot embed an empty collection")
    model.to(device).eval()
    chunks = [model.embed(torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)).cpu().numpy()
              for start in range(0, len(x), batch_size)]
    return np.concatenate(chunks)


def fit_frozen_probe(embeddings: np.ndarray, metadata: pd.DataFrame, *, binary: bool = False,
                     c: float = 1.0, seed: int = 42):
    """Train a standardized linear readout on frozen train embeddings only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(metadata) != len(embeddings) or metadata.groupby("source_id").split.nunique().max() > 1:
        raise ValueError("Embedding alignment or source split integrity failed")
    train = metadata.split.to_numpy() == "train"
    labels = metadata.label.to_numpy()
    if binary:
        labels = (labels != "normal").astype(int)
    if len(np.unique(labels[train])) < 2:
        raise ValueError("The readout needs at least two training classes")
    probe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=c, class_weight=None,
                                                               max_iter=2000, random_state=seed))
    # Fit scaling with equal source mass. Balance classes using those weighted
    # masses, rather than raw window frequencies, for the classifier objective.
    source_counts = metadata.loc[train, "source_id"].value_counts()
    weights = metadata.loc[train, "source_id"].map(lambda s: 1 / source_counts[s]).to_numpy(dtype=float, copy=True)
    weights *= len(weights) / weights.sum()
    training_labels = labels[train]
    classes = np.unique(training_labels)
    label_mass = {label: weights[training_labels == label].sum() for label in classes}
    class_weights = np.asarray([weights.sum() / (len(classes) * label_mass[label]) for label in training_labels])
    probe.fit(embeddings[train], labels[train], standardscaler__sample_weight=weights,
              logisticregression__sample_weight=weights * class_weights)
    return probe


def representation_diagnostics(embeddings: np.ndarray) -> dict[str, float]:
    """Inspect actual encoder features, separately from the VICReg projector.

    A low variance penalty alone cannot establish useful representations. A
    constant encoder has rank zero. Effective rank is bounded by min(B-1,D),
    so compare batches of the same size and report this bound alongside it.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("Need finite [B,D] embeddings with B >= 2")
    centered = z - z.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.sum() <= np.finfo(float).eps:
        effective_rank = 0.0
    else:
        probabilities = singular / singular.sum()
        nonzero = probabilities > 0
        effective_rank = float(np.exp(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()))
    return {"mean_dimension_std": float(z.std(axis=0).mean()),
            "minimum_dimension_std": float(z.std(axis=0).min()),
            "effective_rank": effective_rank, "maximum_centered_rank": float(min(len(z) - 1, z.shape[1]))}


def synthetic_windows(n: int = 12, frames: int = 32, *, seed: int = 7) -> np.ndarray:
    """Smooth toy skeletons for executable teaching examples, never research data."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1, 1))
    time = np.linspace(0, 2 * np.pi, frames)[None, :, None]
    offsets = np.arange(16)[None, None, :] * 0.15
    x = 0.2 * np.sin(time + phase + offsets)
    y = np.broadcast_to(np.linspace(-1, 1, 16)[None, None, :], x.shape) + 0.05 * np.cos(time * 2 + phase + offsets)
    return (np.stack([x, y], axis=-1) + rng.normal(0, 0.01, size=(n, frames, 16, 2))).astype(np.float32)
