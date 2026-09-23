"""Versioned body-12 tracks, physical clocks and independent-person boundaries.

The disk format extends the synthetic-training-v2 bundle.  Inference arrays
contain only estimated coordinates, native scores, availability and timestamps;
references and intervention metadata remain separate.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..synthetic_training_v2.contracts import (
    TrackBundle as _V2TrackBundle, INPUT_KEYS, TARGET_KEYS, JOINTS,
    atomic_json, array_digest, digest, sha256_file, validate_records,
    validate_targets,
)

SCHEMA = "gait-fidelity-body12-v1"
SWAP = np.arange(12).reshape(6, 2)[:, ::-1].ravel()
REQUIRED = {"source_family_id", "pair_id", "movement_state", "movement_magnitude",
            "physical_state", "camera_id", "naming", "observation"}


class IndexedArray:
    """A row selection over disk-backed arrays; indexing loads only selected rows."""

    def __init__(self, source, indices):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices),) + source.shape[1:]
        self.dtype, self.ndim = source.dtype, source.ndim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return self.source[(self.indices[key[0]],) + key[1:]]

    def __array__(self, dtype=None, copy=None):
        value = np.asarray(self.source[self.indices], dtype=dtype)
        return value.copy() if copy else value


def select_rows(bundle, indices):
    """Keep row selections lazy, including references used only by evaluators."""
    indices = np.asarray(indices, dtype=np.int64)
    return TrackBundle({k: IndexedArray(v, indices) for k, v in bundle.inputs.items()},
                       {k: IndexedArray(v, indices) for k, v in bundle.targets.items()},
                       [bundle.records[i] for i in indices], bundle.evidence_status, bundle.provenance)


def _validate_records(bundle, held_extractor):
    if bundle.evidence_status != "technical-source-screen":
        return validate_records(bundle.records, evidence_status=bundle.evidence_status, held_extractor=held_extractor)
    # Reuse the historical identity/duplicate/split checks after checking the
    # new evidence contract. These temporary copies never become saved records.
    checked = []
    for row in bundle.records:
        if row.get("review_mode") != "technical_geometry" or row.get("locomotion_status") not in {"metadata_candidate", "reviewed_locomotion"}:
            raise ValueError("Technical source screening needs explicit candidate/review provenance")
        if row.get("reserved") is not False and row.get("reserved") != "unknown":
            raise ValueError("Protected identities cannot enter the technical source cohort")
        if row["split"] == "confirmation":
            if row.get("reserved") is not False or row.get("exposure") != "unexposed_verified" or not bundle.provenance.get("confirmation_admitted"):
                raise ValueError("Confirmation requires verified exposure and a frozen declaration")
        checked.append(dict(row, locomotion_status="audited_locomotion", reserved=False))
    return validate_records(checked, evidence_status="audited-source", held_extractor=held_extractor)


class TrackBundle(_V2TrackBundle):
    """The established bundle API with an explicit, arbitrary fixed-rate window."""

    def validate(self, held_extractor=None):
        return validate_bundle(self, held_extractor=held_extractor)

    def save(self, folder):
        return save_dataset(self, folder)

    @classmethod
    def load(cls, folder, *, expected_identity=None, allow_confirmation=False):
        result = load_dataset(folder, allow_confirmation=allow_confirmation)
        if expected_identity is not None and result.provenance.get("identity") != expected_identity:
            raise ValueError("Dataset identity does not match the requested version")
        return result

    def subset(self, split):
        if split == "confirmation":
            raise PermissionError("Confirmation requires an explicit locked evaluation loader")
        ix = np.flatnonzero([r["split"] == split for r in self.records])
        return select_rows(self, ix)


def validate_bundle(bundle, *, held_extractor=None):
    """Reject leakage, inconsistent pairs, invented clocks and mislabeled targets."""
    if set(bundle.inputs) != INPUT_KEYS or set(bundle.targets) != TARGET_KEYS:
        raise ValueError("Inference/target field allow-list violation")
    x, c, o, t = (bundle.inputs[k] for k in ("xy", "confidence", "observed", "timestamps"))
    if x.ndim != 4 or x.shape[2:] != (12, 2) or x.shape[1] < 8 or len(x) == 0:
        raise ValueError("Expected nonempty xy[N,T>=8,12,2]")
    if c.shape != x.shape[:-1] or o.shape != c.shape or o.dtype != bool or t.shape != x.shape[:2]:
        raise ValueError("Coordinate, confidence, observed and timestamp shapes disagree")
    hz = float(bundle.provenance.get("hz", 25.0))
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("Physical clock must retain one declared positive fixed sampling rate")
    for start in range(0, len(x), 256):
        sl = slice(start, start + 256)
        xx, cc, oo, tt = x[sl], c[sl], o[sl], t[sl]
        if not np.isfinite(tt).all() or not np.allclose(np.diff(tt, axis=1), 1 / hz, rtol=0, atol=1e-6):
            raise ValueError("Physical clock must retain one declared positive fixed sampling rate")
        if np.any(oo & ~np.isfinite(xx).all(-1)) or np.any(oo & (~np.isfinite(cc) | (cc <= 0))):
            raise ValueError("Observed coordinates require finite positive native estimator scores")
        if np.any(~oo & np.isfinite(xx).all(-1)):
            raise ValueError("Unobserved coordinates must remain missing; no silent interpolation")
        validate_targets({k: v[sl] for k, v in bundle.targets.items()}, {k: v[sl] for k, v in bundle.inputs.items()})
    if len(bundle.records) != len(x):
        raise ValueError("Records and arrays differ")
    summary = _validate_records(bundle, held_extractor)
    families, pairs, naming_groups = {}, defaultdict(list), defaultdict(list)
    for i, r in enumerate(bundle.records):
        if REQUIRED - set(r) or any(str(r.get(k, "")).strip() == "" for k in REQUIRED):
            raise ValueError(f"Missing gait-fidelity provenance: {sorted(REQUIRED - set(r))}")
        if r["naming"] not in {"correct", "global_swap", "temporary_swap"}:
            raise ValueError("Unknown naming intervention")
        if r["physical_state"] not in {"original", "mirrored"}:
            raise ValueError("Unknown physical mirror state")
        if r["observation"] not in {"clear", "occluded"}:
            raise ValueError("Unknown observation intervention")
        if not np.isfinite(float(r["movement_magnitude"])):
            raise ValueError("Movement magnitude must be finite")
        identity = (r["canonical_person_id"], r["split"], r["motion_hash"])
        if families.setdefault(r["source_family_id"], identity) != identity:
            raise ValueError("Source family crosses a person, split or raw-motion identity")
        if r.get("held_intervention", False) and r["split"] == "train":
            raise ValueError("Held intervention entered training")
        if not np.isclose(t[i, 0], float(r["start_s"])) or not np.isclose(t[i, -1], float(r["end_s"])):
            raise ValueError("Record duration disagrees with retained physical timestamps")
        pairs[r["pair_id"]].append(i)
        naming_groups[(r["source_family_id"], r["physical_state"], r["movement_state"],
                       float(r["movement_magnitude"]), r["camera_id"], r["observation"], r["extractor"])].append(i)
    for indices in pairs.values():
        anchor = bundle.records[indices[0]]
        for i in indices[1:]:
            r = bundle.records[i]
            for k in ("source_family_id", "split", "canonical_person_id", "physical_state", "camera_id", "naming", "observation", "extractor"):
                if r[k] != anchor[k]:
                    raise ValueError(f"Movement pair changes undeclared factor {k}")
            if not np.array_equal(t[i], t[indices[0]]):
                raise ValueError("Movement pair timestamps differ")
        baseline = [i for i in indices if bundle.records[i]["movement_state"] == "baseline"]
        if len(baseline) != 1:
            raise ValueError("Each movement pair needs exactly one baseline")
        b = baseline[0]
        for i in indices:
            if bundle.records[i]["movement_state"] == "no_change":
                if not np.array_equal(bundle.targets["xy"][i], bundle.targets["xy"][b], equal_nan=True):
                    raise ValueError("No-change reference differs from its baseline")
    for indices in naming_groups.values():
        ref = indices[0]
        for i in indices[1:]:
            for k in TARGET_KEYS:
                if not np.array_equal(bundle.targets[k][i], bundle.targets[k][ref], equal_nan=True):
                    raise ValueError("Naming interventions must never change anatomical references")
    summary.update(samples=x.shape[1], hz=hz, source_families=len(families), pairs=len(pairs),
                   evidence_status=bundle.evidence_status)
    return summary


def save_dataset(bundle, folder, *, storage="npz"):
    """Publish only a completely serialized, hash-verified immutable directory.

    A failed write leaves a named staging directory and failure record, while
    the requested public path remains absent and can safely be retried.
    """
    validate_bundle(bundle)
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"Completed datasets are immutable: {folder}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{folder.name}-staging-", dir=folder.parent))
    try:
        meta = dict(schema=SCHEMA, joints=list(JOINTS), evidence_status=bundle.evidence_status,
                    records=bundle.records, provenance=bundle.provenance, storage=storage)
        if storage == "npz":
            for group in ("inputs", "targets"):
                np.savez_compressed(staging / f"{group}.npz", **getattr(bundle, group))
                meta[f"{group}_sha256"] = sha256_file(staging / f"{group}.npz")
        elif storage == "npy":
            meta["arrays"] = {}
            for group in ("inputs", "targets"):
                meta["arrays"][group] = {}
                for key, value in getattr(bundle, group).items():
                    path = staging / f"{group}-{key}.npy"
                    dest = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=value.shape)
                    for start in range(0, len(value), 256):
                        dest[start:start + 256] = value[start:start + 256]
                    dest.flush(); del dest
                    meta["arrays"][group][key] = dict(path=path.name, sha256=sha256_file(path))
        else:
            raise ValueError("Dataset storage must be npz or npy")
        atomic_json(staging / "manifest.json", meta)
        # Detect truncated/incompatible archives before publishing the manifest.
        load_dataset(staging, allow_confirmation=True)
        if folder.exists():
            raise FileExistsError(f"Another writer already published {folder}")
        os.rename(staging, folder)
    except Exception as exc:
        atomic_json(staging / "write-failure.json", dict(status="unpublished", destination=str(folder), error=str(exc)))
        raise
    return folder


def load_dataset(folder, *, allow_confirmation=False):
    folder = Path(folder)
    meta = json.loads((folder / "manifest.json").read_text())
    if meta.get("schema") != SCHEMA or meta.get("joints") != list(JOINTS):
        raise ValueError("Incompatible gait-fidelity dataset schema or joint order")
    if any(r["split"] == "confirmation" for r in meta["records"]) and not allow_confirmation:
        raise PermissionError("Confirmation references remain locked to ordinary development loading")
    arrays = []
    for key in ("inputs", "targets"):
        if meta.get("storage", "npz") == "npy":
            group = {}
            for name, entry in meta["arrays"][key].items():
                path = folder / entry["path"]
                if path.parent.resolve() != folder.resolve() or sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"Dataset content hash mismatch: {key}/{name}")
                group[name] = np.load(path, allow_pickle=False, mmap_mode="r")
            arrays.append(group)
        else:
            path = folder / f"{key}.npz"
            if sha256_file(path) != meta[f"{key}_sha256"]:
                raise ValueError(f"Dataset content hash mismatch: {key}")
            with np.load(path, allow_pickle=False) as value:
                arrays.append({k: value[k].copy() for k in value.files})
    result = TrackBundle(*arrays, meta["records"], meta["evidence_status"], meta["provenance"])
    validate_bundle(result)
    return result


def merge_disk_datasets(paths, output, *, provenance=None):
    """Consolidate complete family/shard bundles with bounded array memory.

    References remain locked in ordinary loaders. This preparation-only merge
    validates every source, then publishes one immutable memory-mapped bundle.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("No completed datasets to merge")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Completed datasets are immutable: {output}")
    metas = [json.loads((p / "manifest.json").read_text()) for p in paths]
    if len({m["evidence_status"] for m in metas}) != 1:
        raise ValueError("Dataset evidence statuses differ")
    records = [r for m in metas for r in m["records"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        first = load_dataset(paths[0], allow_confirmation=True)
        dests = {}
        for group in ("inputs", "targets"):
            dests[group] = {k: np.lib.format.open_memmap(staging / f"{group}-{k}.npy", mode="w+", dtype=v.dtype,
                            shape=(len(records),) + v.shape[1:]) for k, v in getattr(first, group).items()}
        position = 0
        for index, path in enumerate(paths):
            part = first if index == 0 else load_dataset(path, allow_confirmation=True)
            for group in ("inputs", "targets"):
                for key, value in getattr(part, group).items():
                    dest = dests[group][key]
                    if value.shape[1:] != dest.shape[1:] or value.dtype != dest.dtype:
                        raise ValueError("Dataset array shape/dtype differs between shards")
                    for start in range(0, len(value), 256):
                        count = min(256, len(value) - start)
                        dest[position + start:position + start + count] = value[start:start + count]
            position += len(part.records)
        meta = dict(schema=SCHEMA, joints=list(JOINTS), evidence_status=metas[0]["evidence_status"],
                    records=records, provenance=provenance if provenance is not None else metas[0]["provenance"],
                    storage="npy", arrays={})
        for group, values in dests.items():
            meta["arrays"][group] = {}
            for key, value in values.items():
                value.flush()
                path = staging / f"{group}-{key}.npy"
                meta["arrays"][group][key] = dict(path=path.name, sha256=sha256_file(path))
        atomic_json(staging / "manifest.json", meta)
        load_dataset(staging, allow_confirmation=True)
        if output.exists():
            raise FileExistsError(f"Another writer already published {output}")
        os.rename(staging, output)
    except Exception as exc:
        atomic_json(staging / "write-failure.json", dict(status="unpublished", destination=str(output), error=str(exc)))
        raise
    return output


def apply_naming(inputs, naming):
    """Swap complete anatomical channels together while keeping times unchanged."""
    if naming not in {"correct", "global_swap", "temporary_swap"}:
        raise ValueError(f"Unknown naming intervention: {naming}")
    result = {k: np.array(v, copy=True) for k, v in inputs.items()}
    n = len(result["xy"])
    frames = np.arange(n) if naming == "global_swap" else np.arange(n // 3, 2 * n // 3) if naming == "temporary_swap" else np.array([], int)
    for k in ("xy", "confidence", "observed"):
        result[k][frames] = result[k][frames][:, SWAP]
    return result


def fixture_bundle(seed=17, *, samples=128, people=5, windows=2, hz=25.0):
    """Articulated analytic SOFTWARE fixture, explicitly unrelated to human data."""
    if samples < 8 or people < 3 or windows < 2:
        raise ValueError("Fixture requires >=8 samples, >=3 people and >=2 source windows per person")
    inputs, targets, records = defaultdict(list), defaultdict(list), []
    for p in range(people):
        split = "train" if p < people - 2 else "development"
        for w in range(windows):
            times = w * (samples / hz + .5) + np.arange(samples) / hz
            phase = 2 * np.pi * (1.0 + .03 * p) * (times - times[0]) + .2 * w
            family = f"fixture-p{p}-w{w}"
            for physical in ("original", "mirrored"):
                for camera, flatten in (("oblique", .85), ("side", 1.)):
                    for state, magnitude in (("baseline", 0.), ("no_change", 0.), ("knee_flexion", 5.), ("knee_flexion", 10.)):
                        xy = np.zeros((samples, 12, 2), np.float32)
                        for side in (0, 1):
                            s = 1 if side == 0 else -1
                            xy[:, side] = [300 + s * 35, 90]
                            xy[:, 2 + side] = xy[:, side] + [s * 20, 55]
                            xy[:, 4 + side] = xy[:, 2 + side] + [s * 10, 55]
                            xy[:, 6 + side] = [300 + s * 20, 230]
                            hip = .25 * np.sin(phase + side * np.pi)
                            knee = .5 + (.3 + (np.deg2rad(magnitude) if side else 0.)) * np.sin(phase + side * np.pi + .5)
                            xy[:, 8 + side] = xy[:, 6 + side] + 75 * np.stack([flatten * np.sin(hip), np.cos(hip)], -1)
                            xy[:, 10 + side] = xy[:, 8 + side] + 75 * np.stack([flatten * np.sin(hip - knee), np.cos(hip - knee)], -1)
                        if physical == "mirrored":
                            xy = xy[:, SWAP].copy(); xy[..., 0] = 600 - xy[..., 0]
                        for observation in ("clear", "occluded"):
                            # Deterministic by physical condition: the no-change
                            # control retains exactly the baseline observation.
                            obs_rng = np.random.default_rng(int(digest([seed, family, physical, camera, magnitude, observation])[:8], 16))
                            raw = xy + obs_rng.normal(0, 1., xy.shape).astype(np.float32)
                            observed = np.ones((samples, 12), bool)
                            visible = observed.copy()
                            if observation == "occluded":
                                observed[samples // 3:samples // 2, 8:] = False
                                visible[samples // 3:samples // 2, 8:] = False
                            raw[~observed] = np.nan
                            native = np.where(observed, .9, np.nan).astype(np.float32)
                            for naming in ("correct", "global_swap", "temporary_swap"):
                                inp = apply_naming(dict(xy=raw, confidence=native, observed=observed, timestamps=times), naming)
                                for k, v in inp.items(): inputs[k].append(v)
                                for k, v in dict(xy=xy, valid=np.ones_like(observed), visible=visible, eval_scale=np.full(samples, 350., np.float32)).items(): targets[k].append(v)
                                pair = digest([family, physical, camera, observation, naming, "fixture-source"])[:24]
                                records.append(dict(person_id=f"fixture-p{p}", canonical_person_id=f"fixture-p{p}",
                                    split=split, original_split="train" if split == "train" else "validation",
                                    exposure="software_fixture", locomotion_status="fixture", audit_reviewer="analytic_generator",
                                    audit_evidence="software_fixture_only", audit_date="2026-09-22", reserved=False,
                                    motion_id=family, motion_hash=digest(family), window_id=family,
                                    source_family_id=family, pair_id=pair, movement_state=state, movement_magnitude=magnitude,
                                    magnitude_deg=magnitude, movement_level_deg=magnitude,
                                    endpoint="baseline" if state == "baseline" else "intervention",
                                    physical_state=physical, camera_id=camera, naming=naming, observation=observation,
                                    variant=f"{physical}-{camera}-{state}-{magnitude:g}-{observation}-{naming}",
                                    extractor="fixture-source", extractor_family="fixture", student_id="fixture-source",
                                    box_source="analytic_fixture", target_kind="software_fixture", held_intervention=False,
                                    start_s=float(times[0]), end_s=float(times[-1])))
    arrays = {k: np.stack(v) for k, v in inputs.items()}
    provenance = dict(kind="analytic software fixture; no human evidence", hz=hz,
                      samples=samples, seed=seed, identity=array_digest(arrays))
    result = TrackBundle(arrays, {k: np.stack(v) for k, v in targets.items()}, records, "fixture-tested", provenance)
    validate_bundle(result)
    return result


def merge_datasets(paths, output, *, allow_confirmation=False):
    """Merge completed preparation shards after their hashes and roster checks."""
    from .preparation import merge_datasets as merge
    return merge(paths, output, allow_confirmation=allow_confirmation)
