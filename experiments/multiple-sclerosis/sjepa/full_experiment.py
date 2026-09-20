"""Shared training, validation selection, and OOF evaluation for notebooks 03–06."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .data import SequenceWindowDataset, sliding_windows
from .eval import evaluate, aggregate_folds
from .splits import (LABELS, checkpoint_context, partition_records,
                     load_partition_checkpoint, validate_registry)


def embed_records(model, records, cfg, device):
    """One vector per clip; fixed seeded readout, bounded batches, no fitting."""
    import torch
    from .masking_v2 import sample_target_mask

    readout = sample_target_mask(cfg.num_joints, cfg.num_time_tokens,
                                np.random.default_rng(0), target_ratio=0.6)
    mask = torch.from_numpy(readout).to(device)
    model.eval()
    vectors = []
    for r in records:
        windows = sliding_windows(r.load_norm(), cfg.window_frames, cfg.window_stride)
        pieces = []
        with torch.no_grad():
            for start in range(0, len(windows), cfg.batch_size):
                batch = torch.from_numpy(windows[start:start + cfg.batch_size]).to(device)
                pieces.append(model.embed(batch, mask).cpu().numpy())
        vectors.append(np.concatenate(pieces).mean(axis=0))
    return np.stack(vectors)


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
                     updates, device, path):
    """Only the registry's training partition is accepted by this entry point."""
    from .train_v2 import train_sjepa_v2, save_checkpoint_v2

    fold = registry["folds"][fold_index]
    if (sorted(r.clip_name for r in train_records) != fold["train_clips"]
            or sorted({r.source_id for r in train_records}) != fold["train_sources"]):
        raise ValueError("Training records do not match the locked training partition")
    if updates < 1:
        raise ValueError("Training needs a positive update budget")
    ds = SequenceWindowDataset(train_records, cfg.window_frames, cfg.window_stride)
    state = train_sjepa_v2(model, ds, cfg, total_updates=updates, device=device,
                           mask_ratio=0.6, seed=cfg.seed)
    if not all(np.isfinite(values).all() for values in
               (state.losses, state.eff_rank, state.emb_std, state.teacher_drift)):
        raise ValueError("Non-finite training diagnostics; checkpoint was not saved")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint_v2(path, model, cfg, train_state=state,
                       extra={"split_context": checkpoint_context(registry, fold_index, cfg, stage),
                              "updates_this_stage": updates})
    return state


def run_fold(records, registry, fold_index, cfg, device, updates, more_updates, output_dir):
    """Fresh model per outer fold; select on validation before reading test arrays."""
    from .models import build_model
    from .classical import build_feature_matrix, train_rf_and_predict

    train, validation, test = partition_records(records, registry, fold_index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # A fresh model, teacher, optimizer, and RNG stream for each outer fold.
    model = build_model(cfg, device=device, repaired=True)
    candidates, validation_scores, diagnostics = {}, {}, {}
    for stage, budget in (("ssl", updates), ("continued", more_updates)):
        path = output_dir / f"{stage}.pt"
        state = train_checkpoint(model, train, cfg, registry, fold_index, stage, budget, device, path)
        probe = fit_probe(embed_records(model, train, cfg, device), train)
        predictions = probe.predict(embed_records(model, validation, cfg, device))
        validation_scores[stage] = score_records(validation, predictions, equal_source=True).macro_f1
        candidates[stage] = (path, probe)
        diagnostics[stage] = dict(final_loss=state.losses[-1], effective_rank=state.eff_rank[-1])
    # Strictly greater: ties keep the shorter, original training run.
    selected = "continued" if validation_scores["continued"] > validation_scores["ssl"] else "ssl"
    chosen_path, chosen_probe = candidates[selected]
    load_partition_checkpoint(chosen_path, model, cfg, registry, fold_index, selected, device)
    # No test features were extracted above this point. Selection is now fixed.
    predictions = {"sjepa": chosen_probe.predict(embed_records(model, test, cfg, device))}
    Xtr, ytr, _, _ = build_feature_matrix(train, fps=cfg.target_fps)
    Xte, _, _, _ = build_feature_matrix(test, fps=cfg.target_fps)
    predictions["rf"] = train_rf_and_predict(Xtr, ytr, Xte, seed=cfg.seed)
    for kind in ("visibility", "mean_pose"):
        head = fit_probe(nuisance_features(train, kind), train)
        predictions[kind] = head.predict(nuisance_features(test, kind))
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


def run_cross_validation(records, registry, cfg, device, updates, more_updates, output_dir):
    """Run all outer folds; publish results only after complete OOF validation."""
    # Callers obtain the inventory through load_full_registry, which hashes the
    # files. Recheck all structural invariants at the evaluation entry point.
    validate_registry(registry, records, registry["inventory"])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rows, selections = [], []
    for k in range(registry["n_splits"]):
        fold_rows, selection = run_fold(records, registry, k, cfg, device, updates, more_updates,
                                         out / f"fold-{k}")
        rows.extend(fold_rows)
        selections.append(selection)
        print(f"fold {k}: {len(fold_rows)} test clips; validation chose {selection['selected']}", flush=True)
    results = dict(dataset="video-data-full", registry_sha256=registry["registry_sha256"],
                   dataset_sha256=registry["dataset_sha256"], config=cfg.to_dict(),
                   smoke=cfg.profile.endswith("smoke"), base_updates=updates,
                   continuation_updates=more_updates, selections=selections,
                   metrics=summarize_oof(rows, records, registry))
    (out / "oof.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def new_evaluation_dir(exp_dir, registry, cfg):
    from .splits import fold_run_dir
    parent = fold_run_dir(exp_dir, registry, cfg).parent
    return parent / ("capstone-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
