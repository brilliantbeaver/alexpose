"""Frozen, source-stratified train/validation/test partitions for the full cache.

Split one row per source before expanding back to clips or windows. No model
scores are used to construct, balance, or choose a partition.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .data import load_index, source_id_from_name

LABELS = ["normal", "ms", "pd"]
PARTITIONS = ("train", "validation", "test")
CLASS_DIRS = {"normal": "Normal", "ms": "MS", "pd": "PD"}
FULL_NAME = re.compile(r"[A-Za-z0-9_-]{11}(?:_P\d+(?:_\d+)?)?$")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_table(records):
    """Require unique clips and one unambiguous condition per source."""
    by_source, clips = {}, set()
    for record in records:
        if not record.source_id or record.label not in LABELS:
            raise ValueError(f"Invalid source or label: {record.clip_name}")
        if record.clip_name in clips:
            raise ValueError(f"Duplicate clip: {record.clip_name}")
        clips.add(record.clip_name)
        if record.source_id in by_source and by_source[record.source_id] != record.label:
            raise ValueError(f"Conflicting labels for source {record.source_id}")
        by_source[record.source_id] = record.label
    if set(by_source.values()) != set(LABELS):
        raise ValueError("All three conditions must have usable source videos")
    return by_source


def inventory(records, video_dir, exclusions):
    """Match every cache entry to a raw clip and account for every raw clip."""
    source_table(records)
    raw = {}
    for label, folder in CLASS_DIRS.items():
        for path in sorted((Path(video_dir) / folder).glob("*.mp4")):
            if not FULL_NAME.fullmatch(path.stem) or path.stem in raw:
                raise ValueError(f"Ambiguous or duplicate full-data filename: {path}")
            raw[path.stem] = dict(clip=path.stem, label=label,
                                  source_id=source_id_from_name(path.name),
                                  video=f"{folder}/{path.name}")
    if not raw:
        raise ValueError("No video-data-full inventory found")
    missing = set(raw) - {r.clip_name for r in records}
    if missing != set(exclusions) or any(not str(v).strip() for v in exclusions.values()):
        raise ValueError("Missing caches must exactly match the reviewed exclusions.json; "
                         f"missing={sorted(missing)}, reviewed={sorted(exclusions)}")
    cache = []
    for r in sorted(records, key=lambda r: r.clip_name):
        expected = raw.get(r.clip_name)
        if expected is None or (r.label, r.source_id) != (expected["label"], expected["source_id"]):
            raise ValueError(f"Cache metadata disagrees with raw inventory: {r.clip_name}")
        with np.load(r.path, allow_pickle=False) as z:
            norm, original = z["keypoints_norm"], z["keypoints"]
            if (norm.ndim != 3 or norm.shape[1:] != (33, 3) or len(norm) < 8
                    or original.shape != norm.shape or len(norm) != r.n_frames
                    or not np.isfinite(norm).all() or not np.isfinite(original).all()
                    or int(z["fps"]) != 15):
                raise ValueError(f"Invalid full-data pose cache: {r.path}")
        cache.append(dict(clip=r.clip_name, label=r.label, source_id=r.source_id,
                          n_frames=r.n_frames, cache_file=r.path.name,
                          sha256=file_sha256(r.path)))
    return dict(raw_clips=[raw[c] for c in sorted(raw)], cache=cache,
                exclusions=[dict(**raw[c], reason=exclusions[c]) for c in sorted(missing)])


def make_registry(records, dataset, n_splits=5, inner_splits=4, seed=42):
    """Outer N-fold CV and the first inner fold as a validation holdout.

    StratifiedKFold is applied to unique sources, NOT clips. The inner loop is
    one holdout, not an averaged inner cross-validation search.
    """
    import sklearn
    from sklearn.model_selection import StratifiedKFold

    table = source_table(records)
    sources = np.array(sorted(table))
    labels = np.array([table[s] for s in sources])
    if n_splits < 2 or inner_splits < 2:
        raise ValueError("Outer and inner fold counts must both be at least two")
    if min(Counter(labels).values()) < n_splits:
        raise ValueError("Too few sources per condition for the requested outer folds")
    outer = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (development, test) in enumerate(outer.split(sources, labels)):
        if min(Counter(labels[development]).values()) < inner_splits:
            raise ValueError("Too few development sources per condition for inner folds")
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + 1 + k)
        train, validation = next(inner.split(sources[development], labels[development]))
        fold = {"fold": k, "inner_seed": seed + 1 + k}
        for role, indices in zip(PARTITIONS, (development[train], development[validation], test)):
            selected = set(sources[indices])
            fold[f"{role}_sources"] = sorted(selected)
            fold[f"{role}_clips"] = sorted(r.clip_name for r in records if r.source_id in selected)
        folds.append(fold)
    registry = dict(schema_version=1, dataset="video-data-full", grouping="source_video_not_participant",
                    splitter="StratifiedKFold on one row per source; expanded to clips",
                    n_splits=n_splits, inner_splits=inner_splits, seed=seed,
                    sklearn_version=sklearn.__version__, labels=LABELS,
                    inventory=dataset, dataset_sha256=digest(dataset), folds=folds)
    registry["registry_sha256"] = digest(registry)
    validate_registry(registry, records, dataset)
    return registry


def validate_registry(registry, records, dataset):
    """Fail on changed data, malformed partitions, leakage, or lost OOF coverage."""
    if registry.get("schema_version") != 1 or registry.get("dataset") != "video-data-full":
        raise ValueError("Expected the versioned video-data-full registry")
    unsigned = {k: v for k, v in registry.items() if k != "registry_sha256"}
    if digest(unsigned) != registry.get("registry_sha256"):
        raise ValueError("Registry checksum mismatch")
    if registry["inventory"] != dataset or registry["dataset_sha256"] != digest(dataset):
        raise ValueError("Dataset changed: create a new registry version and retrain; do not reuse checkpoints")
    source_table(records)
    by_clip = {r.clip_name: r for r in records}
    if "cache" in dataset:
        expected = {entry["clip"]: (entry["label"], entry["source_id"], entry["n_frames"], entry["cache_file"])
                    for entry in dataset["cache"]}
        observed = {r.clip_name: (r.label, r.source_id, r.n_frames, r.path.name) for r in records}
        if observed != expected:
            raise ValueError("Records disagree with the frozen cache inventory")
    all_clips = set(by_clip)
    seen = Counter()
    if len(registry["folds"]) != registry["n_splits"] or registry["n_splits"] < 2:
        raise ValueError("Wrong number of outer folds")
    for k, fold in enumerate(registry["folds"]):
        if fold["fold"] != k:
            raise ValueError("Fold numbers must be consecutive")
        clip_sets, source_sets = [], []
        for role in PARTITIONS:
            clips = fold[f"{role}_clips"]
            if not clips or len(clips) != len(set(clips)) or not set(clips) <= all_clips:
                raise ValueError(f"Invalid {role} clips in fold {k}")
            group = {by_clip[c].source_id for c in clips}
            if sorted(group) != fold[f"{role}_sources"]:
                raise ValueError(f"Source list disagrees with clips in fold {k}")
            if {by_clip[c].label for c in clips} != set(LABELS):
                raise ValueError(f"A condition is missing from {role} in fold {k}")
            clip_sets.append(set(clips))
            source_sets.append(group)
        if set.union(*clip_sets) != all_clips:
            raise ValueError(f"Incomplete fold {k}")
        for i in range(3):
            for j in range(i):
                if clip_sets[i] & clip_sets[j] or source_sets[i] & source_sets[j]:
                    raise ValueError(f"Source leakage across partitions in fold {k}")
        seen.update(fold["test_clips"])
    if seen != Counter({clip: 1 for clip in all_clips}):
        raise ValueError("Every usable clip must be tested exactly once")


def load_full_registry(exp_dir, registry_path=None):
    """Load and verify a frozen registry, or create it from reviewed exclusions."""
    exp_dir = Path(exp_dir)
    path = Path(registry_path) if registry_path else exp_dir / "artifacts/eval/full-v1/fold_registry.json"
    exclusions_path = path.with_name("exclusions.json")
    if not exclusions_path.exists():
        raise FileNotFoundError(f"Review missing caches in {exclusions_path} first (use {{}} if none)")
    records = load_index(exp_dir / "artifacts/keypoints-full")
    exclusions = json.loads(exclusions_path.read_text())
    dataset = inventory(records, exp_dir / "video-data-full", exclusions)
    if path.exists():
        registry = json.loads(path.read_text())
        validate_registry(registry, records, dataset)
    else:
        registry = make_registry(records, dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(registry, indent=2) + "\n")
    return records, registry


def partition_records(records, registry, fold_index=0):
    if not 0 <= fold_index < len(registry["folds"]):
        raise ValueError("Fold index is out of range")
    by_clip = {r.clip_name: r for r in records}
    fold = registry["folds"][fold_index]
    return tuple([by_clip[c] for c in fold[f"{role}_clips"]] for role in PARTITIONS)


def split_summary(records, registry):
    """Rows suitable for a readable DataFrame; report clips AND sources."""
    rows = []
    for k in range(registry["n_splits"]):
        for role, recs in zip(PARTITIONS, partition_records(records, registry, k)):
            row = dict(fold=k, partition=role, clips=len(recs), sources=len({r.source_id for r in recs}))
            for label in LABELS:
                group = [r for r in recs if r.label == label]
                row[f"{label}_clips"] = len(group)
                row[f"{label}_sources"] = len({r.source_id for r in group})
            rows.append(row)
    return rows


def checkpoint_context(registry, fold_index, cfg, stage):
    fold = registry["folds"][fold_index]
    return dict(dataset_sha256=registry["dataset_sha256"], registry_sha256=registry["registry_sha256"],
                fold=fold_index, train_sources=fold["train_sources"], train_clips=fold["train_clips"],
                config=cfg.to_dict(), stage=stage)


def load_partition_checkpoint(path, model, cfg, registry, fold_index, stage, device="cpu"):
    """Validate provenance BEFORE installing any checkpoint weights."""
    from .train_v2 import load_checkpoint_v2
    expected = checkpoint_context(registry, fold_index, cfg, stage)
    return load_checkpoint_v2(path, model, map_location=device, expected_context=expected)


def fold_run_dir(exp_dir, registry, cfg, fold_index=0):
    """Separate smoke runs, profiles, configurations, datasets, and folds."""
    path = (Path(exp_dir) / "artifacts/runs/full-v1" / registry["registry_sha256"][:12]
            / f"{cfg.profile}-{digest(cfg.to_dict())[:12]}" / f"fold-{fold_index}")
    path.mkdir(parents=True, exist_ok=True)
    return path
