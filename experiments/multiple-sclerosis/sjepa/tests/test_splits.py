"""Leakage regressions: related clips, held-out data, stale models, and OOF coverage."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjepa.data import SequenceRecord, save_sequence_npz, source_id_from_name, grouped_kfold
from sjepa.splits import (LABELS, CLASS_DIRS, digest, inventory, make_registry, validate_registry,
                         partition_records, checkpoint_context, load_partition_checkpoint)
from sjepa.full_experiment import (source_weights, summarize_oof, fit_probe,
                                   train_checkpoint)


@pytest.fixture
def records(tmp_path):
    result = []
    for label in LABELS:
        for i in range(8):
            source = f"v{label[0]}{i:09d}"
            for j in range(13 if label == "ms" and i == 0 else 1):
                clip = f"{source}_P{j+1}"
                result.append(SequenceRecord(tmp_path / f"{clip}.npz", label, source, clip, 16))
    return result


@pytest.fixture
def registry(records):
    return make_registry(records, {"fixture": True}, inner_splits=4)


def resign(registry):
    registry["registry_sha256"] = digest({k: v for k, v in registry.items() if k != "registry_sha256"})


def test_source_suffixes_and_uncut_ids():
    for name in ("tsOMPBS277Q_P1.mp4", "tsOMPBS277Q_P5_02.mp4", "tsOMPBS277Q_clip-01.mp4"):
        assert source_id_from_name(name) == "tsOMPBS277Q"
    assert source_id_from_name("abcdefgh_P1.mp4") == "abcdefgh_P1"


def test_related_sequences_stay_together_and_tested_once(records, registry):
    tested = Counter()
    for k in range(5):
        parts = partition_records(records, registry, k)
        groups = [{r.source_id for r in rs} for rs in parts]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for rs in parts:
            assert {r.label for r in rs} == set(LABELS)
        tested.update(r.clip_name for r in parts[2])
    assert tested == Counter({r.clip_name: 1 for r in records})


def test_order_and_clip_multiplicity_do_not_change_source_folds(records, registry):
    shuffled = make_registry(list(reversed(records)), {"fixture": True})
    one_per_source = list({r.source_id: r for r in records}.values())
    thinned = make_registry(one_per_source, {"fixture": True})
    assert registry == shuffled
    for a, b in zip(registry["folds"], thinned["folds"]):
        for role in ("train", "validation", "test"):
            assert a[f"{role}_sources"] == b[f"{role}_sources"]


@pytest.mark.parametrize("fault", ["duplicates", "conflicting_labels", "too_few"])
def test_bad_input_fails(records, fault):
    if fault == "duplicates":
        records.append(records[0])
    elif fault == "conflicting_labels":
        records.append(replace(records[0], clip_name="different", label="pd"))
    else:
        records = [r for r in records if r.label != "pd"] + [next(r for r in records if r.label == "pd")]
    with pytest.raises(ValueError):
        make_registry(records, {})


def test_sparse_legacy_grouped_kfold_does_not_force_two_folds(records):
    sparse = [r for r in records if r.label != "pd"] + [next(r for r in records if r.label == "pd")]
    with pytest.raises(ValueError, match="at least two"):
        list(grouped_kfold(sparse))


def test_resigned_registry_with_source_leak_is_rejected(records, registry):
    bad = deepcopy(registry)
    fold = next(f for f in bad["folds"] if any(c.endswith("_P13") for c in f["train_clips"]))
    clip = next(c for c in fold["train_clips"] if c.endswith("_P13"))
    fold["train_clips"].remove(clip)
    fold["test_clips"].append(clip)
    source = next(r.source_id for r in records if r.clip_name == clip)
    fold["test_sources"] = sorted(fold["test_sources"] + [source])
    resign(bad)
    with pytest.raises(ValueError, match="leakage"):
        validate_registry(bad, records, {"fixture": True})


def test_changed_dataset_and_corrupt_registry_rejected(records, registry):
    with pytest.raises(ValueError, match="Dataset changed"):
        validate_registry(registry, records, {"fixture": False})
    registry["seed"] = 999
    with pytest.raises(ValueError, match="checksum"):
        validate_registry(registry, records, {"fixture": True})


def test_empty_and_infeasible_inner_split_fail(records):
    with pytest.raises(ValueError):
        make_registry([], {})
    with pytest.raises(ValueError, match="inner folds"):
        make_registry(records, {}, inner_splits=8)


def test_inventory_accounts_for_missing_clips_and_changed_bytes(records, tmp_path):
    small = [next(r for r in records if r.label == label) for label in LABELS]
    for i, r in enumerate(small):
        folder = tmp_path / "videos" / CLASS_DIRS[r.label]
        folder.mkdir(parents=True)
        (folder / f"{r.clip_name}.mp4").touch()
        arr = np.ones((16, 33, 3), dtype=np.float32) * (i + 1)
        save_sequence_npz(r.path, arr, arr, 15, r.source_id, r.label, r.clip_name)
    original = inventory(small, tmp_path / "videos", {})
    # Keep all classes present while testing a missing additional clip.
    rejected = tmp_path / "videos/Normal/abcdefghijk_P1.mp4"
    rejected.touch()
    with pytest.raises(ValueError, match="exclusions"):
        inventory(small, tmp_path / "videos", {})
    reviewed = {rejected.stem: "reviewed rejection"}
    assert len(inventory(small, tmp_path / "videos", reviewed)["exclusions"]) == 1
    arr[0, 0, 0] = 987
    r = small[-1]
    save_sequence_npz(r.path, arr, arr, 15, r.source_id, r.label, r.clip_name)
    changed = inventory(small, tmp_path / "videos", reviewed)
    assert original["cache"] != changed["cache"]


@pytest.mark.parametrize("mismatch", ["fold", "stage", "config", "registry", "legacy"])
def test_checkpoint_rejected_before_weights_load(records, registry, tmp_path, mismatch):
    import torch
    from sjepa.config import get_config
    from sjepa.models import build_model
    from sjepa.train_v2 import save_checkpoint_v2
    cfg = get_config("laptop", smoke=True)
    model = build_model(cfg, device="cpu", repaired=True)
    context = checkpoint_context(registry, 0, cfg, "ssl")
    if mismatch == "fold": context["fold"] = 1
    if mismatch == "stage": context["stage"] = "continued"
    if mismatch == "config": context["config"]["window_stride"] += 1
    if mismatch == "registry": context["registry_sha256"] = "wrong"
    path = tmp_path / "checkpoint.pt"
    save_checkpoint_v2(path, model, cfg, extra={} if mismatch == "legacy" else {"split_context": context})
    with torch.no_grad(): model.mask_token.add_(1)
    before = model.mask_token.clone()
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_partition_checkpoint(path, model, cfg, registry, 0, "ssl")
    assert torch.equal(model.mask_token, before)


def test_training_entry_point_rejects_validation(records, registry, tmp_path):
    train, val, _ = partition_records(records, registry)
    with pytest.raises(ValueError, match="training partition"):
        train_checkpoint(None, train + val, None, registry, 0, "ssl", 1, "cpu", tmp_path / "bad.pt")


def test_probe_scaler_never_fits_held_out_data(records):
    train = records[:3]
    train = [replace(r, label=LABELS[i]) for i, r in enumerate(train)]
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    probe = fit_probe(X, train)
    probe.predict(np.array([[1e6, -1e6]]))
    np.testing.assert_allclose(probe[0].mean_, [3, 4])


def test_selection_finishes_before_test_features_are_read(records, registry, tmp_path, monkeypatch):
    """Trace partition use at the orchestration boundary, including a validation win."""
    from types import SimpleNamespace
    import sjepa.full_experiment as experiment
    import sjepa.models as models
    import sjepa.classical as classical
    train, val, test = partition_records(records, registry)
    train_clips = {r.clip_name for r in train}
    val_clips = {r.clip_name for r in val}
    test_clips = {r.clip_name for r in test}
    events = []
    model = SimpleNamespace(stage=None)

    def training(m, recs, cfg, reg, fold, stage, *args):
        assert {r.clip_name for r in recs} == train_clips
        m.stage = stage
        events.append(('train', stage))
        return SimpleNamespace(losses=[1.0], eff_rank=[2.0])

    def embedding(m, recs, *args):
        clips = {r.clip_name for r in recs}
        if clips == test_clips:
            assert ('select', 'continued') in events
        else:
            assert clips == train_clips or clips == val_clips
        events.append(('embed', m.stage, clips == test_clips))
        # Continued head predicts labels correctly; original gets every label wrong.
        return np.array([[LABELS.index(r.label), int(m.stage == 'continued')] for r in recs])

    class Head:
        def predict(self, X):
            return [LABELS[(int(row[0]) + (0 if row[1] else 1)) % 3] for row in X]

    def probe(E, recs):
        assert {r.clip_name for r in recs} == train_clips
        return Head()

    def load(path, m, cfg, reg, fold, stage, device):
        events.append(('select', stage))
        m.stage = stage

    def features(recs, **kwargs):
        clips = {r.clip_name for r in recs}
        assert clips in (train_clips, test_clips)
        assert ('select', 'continued') in events
        return np.array([[LABELS.index(r.label), 1] for r in recs]), [r.label for r in recs], [], []

    monkeypatch.setattr(models, 'build_model', lambda *a, **kw: model)
    monkeypatch.setattr(experiment, 'train_checkpoint', training)
    monkeypatch.setattr(experiment, 'embed_records', embedding)
    monkeypatch.setattr(experiment, 'fit_probe', probe)
    monkeypatch.setattr(experiment, 'load_partition_checkpoint', load)
    monkeypatch.setattr(classical, 'build_feature_matrix', features)
    monkeypatch.setattr(classical, 'train_rf_and_predict', lambda X, y, Z, **kw: Head().predict(Z))
    monkeypatch.setattr(experiment, 'nuisance_features', lambda recs, kind: features(recs)[0])
    rows, selected = experiment.run_fold(records, registry, 0, SimpleNamespace(seed=42, target_fps=15),
                                          'cpu', 4, 2, tmp_path)
    assert selected['selected'] == 'continued'
    assert {row['clip'] for row in rows} == test_clips
    assert all(row['pred_sjepa'] == row['true'] for row in rows)


def test_equal_source_weight_and_oof_coverage(records, registry):
    weights = source_weights(records)
    for source in {r.source_id for r in records}:
        assert sum(w for w, r in zip(weights, records) if r.source_id == source) == pytest.approx(1)
    rows = []
    for k in range(5):
        for r in partition_records(records, registry, k)[2]:
            rows.append(dict(clip=r.clip_name, source_id=r.source_id, true=r.label, fold=k,
                             **{f"pred_{name}": r.label for name in ["rf", "sjepa", "visibility", "mean_pose", "majority"]}))
    assert summarize_oof(rows, records, registry)["sjepa"]["source_weighted"]["macro_f1"] == 1
    for bad in (rows[:-1], rows + rows[:1]):
        with pytest.raises(ValueError, match="Incomplete or duplicate"):
            summarize_oof(bad, records, registry)
    rows[0]["fold"] = (rows[0]["fold"] + 1) % 5
    with pytest.raises(ValueError, match="frozen test partition"):
        summarize_oof(rows, records, registry)
