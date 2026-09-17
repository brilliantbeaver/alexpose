"""Independent, equal-budget pose-adaptation trials and prediction caches."""

from __future__ import annotations

from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .data import BOX_COLUMNS, PoseFrameDataset, context_clip_indices, load_pose_manifest
from .estimators import StudentSpec, load_estimator
from .context_features import ContextSpec, load_context_encoder, simple_context_features
from .measurements import diagnostic_summary, prediction_summary, reference_error, response_summary


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def student_spec(item):
    return StudentSpec(**{f.name: item[f.name] for f in fields(StudentSpec) if f.name in item})


def context_spec(cfg, *, kind=None):
    selected = kind or cfg.context_kind
    return ContextSpec(kind=selected, repository=cfg.context_repo,
                       checkpoint=cfg.image_checkpoint if selected == "image" else cfg.context_checkpoint,
                       builder=cfg.context_builder, checkpoint_key=cfg.context_checkpoint_key,
                       image_size=cfg.context_image_size, frames=cfg.context_frames)


def predict_dataset(estimator, dataset, batch_size=32):
    """Predict in fixed manifest order; context datasets never open labels."""
    outputs = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.get_batch(range(start, min(start + batch_size, len(dataset))))
        outputs.append(estimator.predict(batch["images"], boxes=batch["boxes"], batch_size=batch_size))
    if not outputs:
        raise ValueError("Cannot predict an empty data partition.")
    return np.concatenate(outputs)


def reference_arrays(dataset):
    """Small coordinate arrays only; image decoding remains bounded by batches."""
    targets, visibility, boxes = [], [], []
    for start in range(0, len(dataset), 64):
        batch = dataset.get_batch(range(start, min(start + 64, len(dataset))))
        targets.extend(batch["keypoints"])
        visibility.extend(batch["visible"])
        boxes.extend(batch["boxes"])
    return np.asarray(targets), np.asarray(visibility), np.asarray(boxes)


def adapt(estimator, replay, lesson, cfg, steps, learning_rate, *, seed):
    """Fresh optimizer; identical replay positions across candidate lessons.

    Replay-only fills all image slots with real examples. Every mixed arm uses
    the same real slots and the same number of synthetic slots. Sampling with
    replacement makes fixed update budgets possible for small pilot libraries.
    """
    if len(replay) == 0 or (lesson is not None and len(lesson) == 0):
        raise ValueError("Replay and selected lesson must contain training examples.")
    rng_real, rng_synthetic = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = estimator.make_optimizer(learning_rate, weight_decay=cfg.weight_decay)
    synthetic_count = round(cfg.train_batch_size * cfg.synthetic_fraction) if lesson is not None else 0
    real_count = cfg.train_batch_size - synthetic_count
    losses = []
    started = perf_counter()
    for step in range(steps):
        batch = replay.get_batch(rng_real.integers(len(replay), size=real_count))
        if synthetic_count:
            extra = lesson.get_batch(rng_synthetic.integers(len(lesson), size=synthetic_count))
            batch["images"] += extra["images"]
            for key in ("keypoints", "visible", "boxes"):
                batch[key] = np.concatenate([batch[key], extra[key]])
        loss = estimator.train_batch(batch["images"], batch["keypoints"], batch["visible"],
                                     optimizer, boxes=batch["boxes"])
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite adaptation loss at update {step + 1}.")
        losses.append(float(loss))
    return np.asarray(losses, np.float32), perf_counter() - started


def crop_context_images(images, boxes):
    """Use a fixed padded union crop so the encoder sees the tracked person."""
    height, width = images[0].shape[:2]
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("A context clip must retain one native image geometry.")
    low, high = np.min(boxes[:, :2], axis=0), np.max(boxes[:, 2:], axis=0)
    padding = (high - low) * 0.125
    low = np.maximum(np.floor(low - padding).astype(int), 0)
    high = np.minimum(np.ceil(high + padding).astype(int), [width, height])
    if np.any(high - low < 2):
        raise ValueError("Context person box lies outside the decoded image.")
    return np.stack([im[low[1]:high[1], low[0]:high[0]] for im in images])


def cache_context(cfg, index, path, *, encoders=None):
    """Encode unlabeled context once, shared across all student trials.

    Files contain means over clips within each domain. Frozen image features
    and temporal shuffling are computed only when their required model exists.
    """
    path = Path(path)
    if path.is_file():
        return read_npz(path)
    dataset = PoseFrameDataset(index, labeled=False)
    if encoders is None:
        encoders = (load_context_encoder(context_spec(cfg), cfg.device),
                    load_context_encoder(context_spec(cfg, kind="image"), cfg.device) if cfg.image_checkpoint else None)
    encoder, image_encoder = encoders
    encoded, simple, shuffled, images = [], [], [], []
    for _, indices in context_clip_indices(dataset.index):
        batch = dataset.get_batch(indices)
        clip = crop_context_images(batch["images"], batch["boxes"])
        encoded.append(encoder.encode(clip))
        sizes = batch["boxes"][:, 2:] - batch["boxes"][:, :2]
        # Explicit observation-scale features challenge reliance on easy visual nuisances.
        observed_geometry = np.r_[np.log1p(sizes).mean(axis=0), (sizes[:, 0] / sizes[:, 1]).mean()]
        domain = str(dataset.index.iloc[indices[0]].domain_id)
        view = [float(domain.startswith(prefix)) for prefix in ("front", "oblique", "side")]
        simple.append(np.r_[simple_context_features(clip), observed_geometry, view])
        if cfg.context_kind == "vjepa":
            shuffled.append(encoder.encode(clip, shuffle_seed=cfg.seed))
        if image_encoder is not None:
            images.append(image_encoder.encode(clip))
    if not encoded:
        raise ValueError("No context clips to encode.")
    result = {"context": np.mean(encoded, axis=0).astype(np.float32),
              "simple_context": np.mean(simple, axis=0).astype(np.float32)}
    if shuffled:
        result["shuffled_context"] = np.mean(shuffled, axis=0).astype(np.float32)
    if images:
        result["image_context"] = np.mean(images, axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A direct user-started student job may overlap another context extraction.
    # Readers must see a complete file, even outside the normal staged launcher.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary, **result)
            temporary.flush()
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return result


def context_feature_paths(cfg, context_index, split):
    """Construct per-domain caches without reading target reference labels."""
    result, encoders = {}, None
    for domain, rows in context_index.groupby("domain_id", sort=True):
        if not all(c.isalnum() or c in "_.-" for c in str(domain)):
            raise ValueError("domain_id must be a safe filename component.")
        path = cfg.root / "data/context_features" / f"{split}_{domain}.npz"
        if not path.is_file() and encoders is None:
            encoders = (load_context_encoder(context_spec(cfg), cfg.device),
                        load_context_encoder(context_spec(cfg, kind="image"), cfg.device) if cfg.image_checkpoint else None)
        cache_context(cfg, rows.reset_index(drop=True), path, encoders=encoders)
        result[str(domain)] = path
    return result


def make_episode_features(before, after, context_index, diagnostic_pre, diagnostic_post,
                          loss_history, context_features, cfg, item, budget):
    """Reference-free target measurements plus declared labeled-source diagnostics."""
    boxes = context_index[BOX_COLUMNS].to_numpy(float)
    delta, magnitude = response_summary(before, after, boxes)
    families = sorted({s["family"] for s in cfg.students if s["role"] == "train"})
    family_code = [float(item["family"] == family) for family in families]
    lr = float(item.get("learning_rate", cfg.learning_rate))
    return {
        **context_features,
        "target_pre": prediction_summary(before, boxes),
        "target_post": prediction_summary(after, boxes),
        "target_delta": delta, "magnitude": magnitude,
        "diagnostic_pre": diagnostic_pre, "diagnostic_post": diagnostic_post,
        "diagnostic_delta": diagnostic_post - diagnostic_pre,
        "loss_history": np.log1p(np.maximum(loss_history, 0)).astype(np.float32),
        "descriptors": np.asarray([np.log10(lr), cfg.probe_steps, *family_code], np.float32),
        "budget": np.asarray([np.log1p(budget)], np.float32),
    }


def run_source_student(cfg: RunConfig, student_id: str):
    """Save the actual student×lesson outcome table, never a surrogate gain."""
    item = cfg.student(student_id)
    if item["role"] == "held":
        raise ValueError("Held architectures cannot contribute source teacher outcomes.")
    split = "train" if item["role"] == "train" else "validation"
    index = load_pose_manifest(cfg.root / "data/synthetic.csv")
    contexts = index.loc[index.role.eq(f"{split}_context")].reset_index(drop=True)
    refs = index.loc[index.role.eq(f"{split}_reference")].reset_index(drop=True)
    diagnostic = PoseFrameDataset(index.loc[index.role.eq("diagnostic")])
    probe = PoseFrameDataset(index.loc[index.role.eq("probe")])
    replay = PoseFrameDataset(cfg.root / "data/replay.csv")
    lesson_rows = index.loc[index.role.eq("lesson")].copy()
    lessons = sorted(lesson_rows.lesson_id.unique().tolist())
    if not lessons or contexts.empty or refs.empty or len(diagnostic) == 0:
        raise ValueError("Prepare all lesson, context, diagnostic and reference partitions first.")
    caches = context_feature_paths(cfg, contexts, split)
    if set(contexts.domain_id) != set(refs.domain_id):
        raise ValueError('Every source context domain needs its own disjoint reference domain.')
    root = cfg.root / "source" / student_id
    if (root / "outcomes.csv").exists():
        raise FileExistsError(f"Completed source trials already exist: {root}; use a new run for changed trials.")
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "heads").mkdir(exist_ok=True)
    estimator = load_estimator(student_spec(item), cfg.device)
    original = estimator.head_state()
    contexts_dataset = PoseFrameDataset(contexts, labeled=False)
    reference_dataset = PoseFrameDataset(refs)
    targets, visible, ref_boxes = reference_arrays(reference_dataset)
    diag_targets, diag_visible, diag_boxes = reference_arrays(diagnostic)
    before = predict_dataset(estimator, contexts_dataset, cfg.predict_batch_size)
    diag_before = predict_dataset(estimator, diagnostic, cfg.predict_batch_size)
    diag_pre = diagnostic_summary(diag_before, diag_targets, diag_visible, diag_boxes,
                                  diagnostic.index.lesson_id, lessons,
                                  missing_penalty=cfg.missing_prediction_penalty)
    lr = float(item.get("learning_rate", cfg.learning_rate))
    loss_history, probe_seconds = adapt(estimator, replay, probe, cfg, cfg.probe_steps, lr, seed=cfg.seed)
    probed = estimator.head_state()
    estimator.save_head(root / "heads/probe.pt")
    after = predict_dataset(estimator, contexts_dataset, cfg.predict_batch_size)
    diag_after = predict_dataset(estimator, diagnostic, cfg.predict_batch_size)
    diag_post = diagnostic_summary(diag_after, diag_targets, diag_visible, diag_boxes,
                                   diagnostic.index.lesson_id, lessons,
                                   missing_penalty=cfg.missing_prediction_penalty)
    records = []

    def evaluate_branch(action, budget, seconds):
        predictions = predict_dataset(estimator, reference_dataset, cfg.predict_batch_size)
        np.savez_compressed(root / f"predictions_{action}_{budget}.npz", frame_ids=refs.frame_id.to_numpy(str),
                            predictions=predictions)
        for domain, group in refs.groupby("domain_id", sort=True):
            pos = group.index.to_numpy()
            error = reference_error(predictions[pos], targets[pos], visible[pos], ref_boxes[pos],
                                    group, missing_penalty=cfg.missing_prediction_penalty)
            records.append(dict(student_id=student_id, family=item["family"], split=split,
                                domain_id=str(domain), budget=budget, action=action,
                                error=error, adaptation_seconds=seconds))

    estimator.load_head_state(original)
    evaluate_branch("original", 0, 0.0)
    estimator.load_head_state(probed)
    evaluate_branch("probe", 0, probe_seconds)
    for budget in cfg.adaptation_steps:
        for action in ["replay", *lessons, "pooled"]:
            estimator.load_head_state(probed)
            lesson = None if action == "replay" else PoseFrameDataset(
                lesson_rows if action == "pooled" else lesson_rows.loc[lesson_rows.lesson_id.eq(action)])
            _, seconds = adapt(estimator, replay, lesson, cfg, budget, lr, seed=cfg.seed + 100)
            estimator.save_head(root / "heads" / f"{action}_{budget}.pt")
            evaluate_branch(action, budget, seconds)
        estimator.load_head_state(original)
        _, seconds = adapt(estimator, replay, None, cfg, cfg.probe_steps + budget, lr, seed=cfg.seed)
        evaluate_branch("full_replay", budget, seconds)
        for domain, group in contexts.groupby("domain_id", sort=True):
            pos = group.index.to_numpy()
            features = make_episode_features(before[pos], after[pos], group, diag_pre, diag_post,
                                             loss_history, read_npz(caches[str(domain)]), cfg, item, budget)
            np.savez_compressed(root / "features" / f"{domain}_{budget}.npz", **features)
    table = pd.DataFrame(records)
    table.to_csv(root / "outcomes.csv", index=False)
    write_json(root / "trial.json", {"student": item, "lessons": lessons,
               "probe_seconds": probe_seconds, "probe_losses": loss_history.tolist(),
               "budgets": cfg.adaptation_steps, "seed": cfg.seed,
               "synthetic_examples_per_batch": round(cfg.train_batch_size * cfg.synthetic_fraction),
               "real_examples_per_mixed_batch": cfg.train_batch_size - round(cfg.train_batch_size * cfg.synthetic_fraction)})
    return table
