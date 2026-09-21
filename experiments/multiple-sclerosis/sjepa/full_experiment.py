"""Shared training, validation selection, and OOF evaluation for notebooks 03–06."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import SequenceWindowDataset, sliding_windows
from .eval import evaluate, aggregate_folds
from .splits import (LABELS, checkpoint_context, partition_records,
                     load_partition_checkpoint, validate_registry)


def embed_records(model, records, cfg, device):
    """One vector per clip; fill inference batches across clip boundaries.

    Pool each clip's windows separately, in their original order. A batch holds
    at most cfg.batch_size windows, even when clips are long.
    """
    import torch
    from .masking_v2 import sample_target_mask

    readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,
                                np.random.default_rng(0), target_ratio=0.6)
    mask = torch.from_numpy(readout).to(device)
    model.eval()
    pieces = [[] for _ in records]
    pending, owners = [], []

    def flush():
        batch = torch.from_numpy(np.stack(pending)).to(device)
        embedded = model.embed(batch, mask).cpu().numpy()
        for owner, vector in zip(owners, embedded):
            pieces[owner].append(vector)
        pending.clear()
        owners.clear()

    with torch.inference_mode():
        for i, record in enumerate(records):
            windows = sliding_windows(record.load_norm(), cfg.window_frames, cfg.window_stride)
            if not len(windows):
                raise ValueError(f"No windows for {record.clip_name}")
            for window in windows:
                pending.append(window)
                owners.append(i)
                if len(pending) == cfg.batch_size:
                    flush()
        if pending:
            flush()
    return np.stack([np.stack(clip).mean(axis=0) for clip in pieces])


def fit_probe(embeddings, records):
    """Scaling and the class-balanced linear head see training clips only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
    return probe.fit(embeddings, [r.label for r in records])


def source_weights(records):
    counts = Counter(r.source_id for r in records)
    return np.array([1.0 / counts[r.source_id] for r in records])


def score_records(records, predictions, equal_source=False):
    return evaluate([r.label for r in records], predictions, LABELS,
                    sample_weight=source_weights(records) if equal_source else None)


def nuisance_features(records, kind="visibility"):
    rows = []
    for r in records:
        if kind == "visibility":
            vis = r.load_raw()[:, :, 2]
            row = np.concatenate([vis.mean(0), vis.std(0)])
        elif kind == "mean_pose":
            row = r.load_norm()[:, :, :2].mean(0).ravel()
        else:
            raise ValueError(f"Unknown control: {kind}")
        rows.append(row)
    return np.stack(rows)


def train_checkpoint(model, train_records, cfg, registry, fold_index, stage,
                     updates, device, path, *, log_every=0):
    """Only the registry's training partition is accepted by this entry point."""
    from .train_v2 import train_sjepa_v2, save_checkpoint_v2
    from .experiment_cache import atomic_path

    fold = registry["folds"][fold_index]
    if (sorted(r.clip_name for r in train_records) != fold["train_clips"]
            or sorted({r.source_id for r in train_records}) != fold["train_sources"]):
        raise ValueError("Training records do not match the locked training partition")
    if updates < 1:
        raise ValueError("Training needs a positive update budget")
    ds = SequenceWindowDataset(train_records, cfg.window_frames, cfg.window_stride)
    state = train_sjepa_v2(model, ds, cfg, total_updates=updates, device=device,
                           mask_ratio=0.6, seed=cfg.seed, log_every=log_every)
    if not all(np.isfinite(values).all() for values in
               (state.losses, state.eff_rank, state.emb_std, state.teacher_drift)):
        raise ValueError("Non-finite training diagnostics; checkpoint was not saved")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        save_checkpoint_v2(temporary, model, cfg, train_state=state,
                           extra={"split_context": checkpoint_context(registry, fold_index, cfg, stage),
                                  "updates_this_stage": updates})
    return state


def run_fold(records, registry, fold_index, cfg, device, updates, more_updates, output_dir,
             *, cache_context=None, feature_cache=None, rf_jobs=None, verbose=False):
    """Fresh model per outer fold; select on validation before reading test arrays."""
    from .models import build_model
    from .classical import build_feature_matrix, train_rf_and_predict
    from .experiment_cache import read_manifest, write_manifest, read_arrays, write_arrays
    from .splits import file_sha256

    train, validation, test = partition_records(records, registry, fold_index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A fresh model, teacher, optimizer, and RNG stream for each outer fold.
    model = build_model(cfg, device=device, repaired=True)
    candidates, validation_scores, diagnostics = {}, {}, {}
    parent_checksum = None
    for stage, budget in (("ssl", updates), ("continued", more_updates)):
        started = perf_counter()
        path = output_dir / f"{stage}.pt"
        context = dict(run=cache_context, fold=fold_index, stage=stage, parent=parent_checksum)
        manifest = output_dir / f"{stage}.json"
        saved = read_manifest(manifest, context) if cache_context else None
        if saved is not None:
            load_partition_checkpoint(path, model, cfg, registry, fold_index, stage, device)
            diagnostics[stage] = saved
            if verbose:
                print(f"fold {fold_index} {stage}: checkpoint cache hit", flush=True)
        else:
            if verbose:
                print(f"fold {fold_index} {stage}: training {budget} updates", flush=True)
            kwargs = {"log_every": max(1, budget // 10)} if verbose else {}
            state = train_checkpoint(model, train, cfg, registry, fold_index, stage,
                                     budget, device, path, **kwargs)
            diagnostics[stage] = dict(final_loss=state.losses[-1], effective_rank=state.eff_rank[-1])
            if cache_context:
                write_manifest(manifest, context, diagnostics[stage], [path.name])
        # Readouts belong to these exact checkpoint bytes, not just the stage name.
        embeddings_path = output_dir / f"{stage}-embeddings.npz"
        embeddings_manifest = output_dir / f"{stage}-embeddings.json"
        embedding_context = dict(context, checkpoint=file_sha256(path)) if cache_context else None
        hit = read_manifest(embeddings_manifest, embedding_context) if cache_context else None
        embeddings = read_arrays(embeddings_path) if hit is not None else None
        if embeddings is not None and (
                set(embeddings) != {"train", "validation"}
                or embeddings["train"].shape != (len(train), cfg.encoder_dim)
                or embeddings["validation"].shape != (len(validation), cfg.encoder_dim)):
            embeddings = None
        if embeddings is None:
            embeddings = dict(train=embed_records(model, train, cfg, device),
                              validation=embed_records(model, validation, cfg, device))
            if cache_context:
                write_arrays(embeddings_path, **embeddings)
                write_manifest(embeddings_manifest, embedding_context, True, [embeddings_path.name])
        probe = fit_probe(embeddings["train"], train)
        predictions = probe.predict(embeddings["validation"])
        validation_scores[stage] = score_records(validation, predictions, equal_source=True).macro_f1
        candidates[stage] = (path, probe)
        if cache_context:
            parent_checksum = file_sha256(path)
        if verbose:
            print(f"fold {fold_index} {stage}: validation macro-F1 "
                  f"{validation_scores[stage]:.3f}; {perf_counter() - started:.1f}s", flush=True)
    # Strictly greater: ties keep the shorter, original training run.
    selected = "continued" if validation_scores["continued"] > validation_scores["ssl"] else "ssl"
    chosen_path, chosen_probe = candidates[selected]
    load_partition_checkpoint(chosen_path, model, cfg, registry, fold_index, selected, device)
    # No test features were extracted above this point. Selection is now fixed.
    predictions = {"sjepa": chosen_probe.predict(embed_records(model, test, cfg, device))}
    if verbose:
        print(f"fold {fold_index}: selected {selected}; evaluating test clips and baselines", flush=True)
    if feature_cache is None:
        Xtr, ytr, _, _ = build_feature_matrix(train, fps=cfg.target_fps)
        Xte, _, _, _ = build_feature_matrix(test, fps=cfg.target_fps)
    else:
        train_features, test_features = feature_cache.matrix(train), feature_cache.matrix(test)
        Xtr, Xte, ytr = train_features["rf"], test_features["rf"], [r.label for r in train]
    rf_kwargs = {} if rf_jobs is None else {"n_jobs": rf_jobs}
    predictions["rf"] = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed, **rf_kwargs)
    for kind in ("visibility", "mean_pose"):
        Xtr = nuisance_features(train, kind) if feature_cache is None else train_features[kind]
        Xte = nuisance_features(test, kind) if feature_cache is None else test_features[kind]
        head = fit_probe(Xtr, train)
        predictions[kind] = head.predict(Xte)
    majority = max(LABELS, key=lambda label: sum(r.label == label for r in train))
    predictions["majority"] = [majority] * len(test)
    rows = [dict(clip=r.clip_name, source_id=r.source_id, true=r.label, fold=fold_index,
                 **{f"pred_{name}": str(pred[i]) for name, pred in predictions.items()})
            for i, r in enumerate(test)]
    return rows, dict(fold=fold_index, selected=selected,
                      validation_source_macro_f1=validation_scores, diagnostics=diagnostics)


def summarize_oof(rows, records, registry):
    """Require one correctly attributed test prediction per usable clip."""
    by_clip = {r.clip_name: r for r in records}
    if Counter(row["clip"] for row in rows) != Counter({c: 1 for c in by_clip}):
        raise ValueError("Incomplete or duplicate OOF predictions: cannot publish a full-CV score")
    expected_fold = {clip: f["fold"] for f in registry["folds"] for clip in f["test_clips"]}
    for row in rows:
        record = by_clip[row["clip"]]
        if (row["fold"], row["source_id"], row["true"]) != (expected_fold[row["clip"]], record.source_id, record.label):
            raise ValueError("OOF row disagrees with the frozen test partition")
    ordered = [by_clip[row["clip"]] for row in rows]
    report = {}
    for name in ("rf", "sjepa", "visibility", "mean_pose", "majority"):
        pred = [row[f"pred_{name}"] for row in rows]
        if not set(pred) <= set(LABELS):
            raise ValueError("Invalid predicted class")
        fold_metrics = []
        for fold in registry["folds"]:
            subset = [row for row in rows if row["fold"] == fold["fold"]]
            fold_metrics.append(score_records([by_clip[row["clip"]] for row in subset],
                                               [row[f"pred_{name}"] for row in subset], True))
        report[name] = dict(clip_weighted=score_records(ordered, pred).to_dict(),
                            source_weighted=score_records(ordered, pred, True).to_dict(),
                            source_weighted_fold_mean_std=aggregate_folds(fold_metrics))
    return report


def _execute_fold(records, registry, k, cfg, device, updates, more_updates, out,
                  cache_root, context, feature_cache, cpu_threads, rf_jobs, verbose):
    """Top-level process entry point: isolated RNG, bounded threads, one fold lock."""
    from contextlib import nullcontext
    import shutil
    import torch
    from filelock import FileLock
    from threadpoolctl import threadpool_limits
    from .experiment_cache import read_manifest, write_manifest

    destination = Path(out) / f"fold-{k}"
    work = Path(cache_root) / f"fold-{k}" if cache_root else destination
    work.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(work / ".lock")) if cache_root else nullcontext()
    started = perf_counter()
    with lock:
        manifest = work / "complete.json"
        fold_context = dict(run=context, fold=k)
        cached = read_manifest(manifest, fold_context) if cache_root else None
        if cached is not None:
            rows, selection = cached["rows"], cached["selection"]
        else:
            previous_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(cpu_threads)
                with threadpool_limits(limits=cpu_threads):
                    rows, selection = run_fold(
                        records, registry, k, cfg, device, updates, more_updates, work,
                        cache_context=context if cache_root else None, feature_cache=feature_cache,
                        rf_jobs=rf_jobs, verbose=verbose)
            finally:
                torch.set_num_threads(previous_threads)
            if cache_root:
                write_manifest(manifest, fold_context, dict(rows=rows, selection=selection),
                               ["ssl.pt", "continued.pt"])
        if cache_root:
            # Each evaluation remains self-contained, including both checkpoints.
            shutil.copytree(work, destination, ignore=shutil.ignore_patterns(".lock", ".*"))
    elapsed = perf_counter() - started
    status = "cache hit" if cached is not None else "computed"
    if verbose:
        print(f"fold {k}: {len(rows)} test clips; validation chose {selection['selected']} "
              f"[{status}, {elapsed:.1f}s]", flush=True)
    return rows, selection, dict(fold=k, status=status, seconds=elapsed)


def run_cross_validation(records, registry, cfg, device, updates, more_updates, output_dir,
                          *, cache_dir=None, fold_workers=1, feature_workers=1,
                          cpu_threads=1, rf_jobs=1, verbose=True):
    """Run all folds with optional persistent caches and CPU process parallelism.

    Keep fold_workers=1 on a single MPS/CUDA device. On CPU, independent worker
    processes own their RNG/model state; inner feature/RF workers are then capped
    at one. Re-running with the same cache reuses completed stages and folds.
    An interrupted stage restarts; a completed stage does not need retraining.
    """
    import torch
    from joblib import Parallel, delayed, parallel_config
    from .experiment_cache import cache_identity, FeatureCache, write_json
    from .splits import digest

    for name, value in (("fold_workers", fold_workers), ("feature_workers", feature_workers),
                         ("cpu_threads", cpu_threads), ("rf_jobs", rf_jobs),
                         ("updates", updates), ("more_updates", more_updates)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if fold_workers > 1 and torch.device(device).type != "cpu":
        raise ValueError("Use fold_workers=1 for a single GPU/MPS device; parallel folds require CPU")
    fold_workers = min(fold_workers, registry["n_splits"])
    if fold_workers > 1:
        feature_workers = rf_jobs = 1  # no nested worker pools
    started = perf_counter()
    # Callers obtain the inventory through load_full_registry, which hashes the
    # files. Recheck all structural invariants at the evaluation entry point.
    validate_registry(registry, records, registry["inventory"])
    cache_root = context = feature_cache = None
    if cache_dir is not None:
        identity, feature_identity = cache_identity(records, registry, cfg, device,
                                                    updates, more_updates, cpu_threads)
        context = digest(identity)
        cache_root = Path(cache_dir) / "runs" / context
        cache_root.mkdir(parents=True, exist_ok=True)
        write_json(cache_root / "provenance.json", identity)
        feature_cache = FeatureCache(Path(cache_dir) / "features" / digest(feature_identity),
                                     cfg.target_fps, feature_workers)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    if cache_root:
        write_json(out / "provenance.json", identity)
    if verbose:
        print(f"{registry['n_splits']} folds on {device}; fold workers={fold_workers}; "
              f"CPU threads/worker={cpu_threads}; cache={'on' if cache_root else 'off'}", flush=True)
    tasks = ((records, registry, k, cfg, device, updates, more_updates, out,
              cache_root, context, feature_cache, cpu_threads, rf_jobs, verbose)
             for k in range(registry["n_splits"]))
    if fold_workers == 1:
        completed = [_execute_fold(*args) for args in tasks]
    else:
        with parallel_config(backend="loky", inner_max_num_threads=cpu_threads):
            completed = Parallel(n_jobs=fold_workers)(delayed(_execute_fold)(*args) for args in tasks)
    rows, selections, timings = [], [], []
    for fold_rows, selection, timing in completed:
        rows.extend(fold_rows)
        selections.append(selection)
        timings.append(timing)
    results = dict(dataset="video-data-full", registry_sha256=registry["registry_sha256"],
                   dataset_sha256=registry["dataset_sha256"], config=cfg.to_dict(),
                   smoke=cfg.profile.endswith("smoke"), base_updates=updates,
                   continuation_updates=more_updates, selections=selections,
                   metrics=summarize_oof(rows, records, registry),
                   execution=dict(seconds=perf_counter() - started, folds=timings,
                                  cache_key=context, fold_workers=fold_workers,
                                  feature_workers=feature_workers, cpu_threads=cpu_threads,
                                  rf_jobs=rf_jobs))
    write_json(out / "oof.json", rows)
    write_json(out / "results.json", results)
    return results


def new_evaluation_dir(exp_dir, registry, cfg):
    from .splits import fold_run_dir
    parent = fold_run_dir(exp_dir, registry, cfg).parent
    return parent / ("capstone-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
