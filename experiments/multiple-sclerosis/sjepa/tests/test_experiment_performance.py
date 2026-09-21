"""Speedups must preserve losses, split boundaries, and reusable result identity."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sjepa.config import get_config
from sjepa.data import SequenceRecord, save_sequence_npz, sliding_windows
from sjepa.experiment_cache import (atomic_path, cache_identity, read_manifest,
                                    write_manifest, write_json, FeatureCache)
from sjepa.full_experiment import embed_records, run_cross_validation, run_fold
from sjepa.losses import CenteringSharpeningCE
from sjepa.models import build_model
from sjepa.splits import LABELS, make_registry, file_sha256, digest


def test_batched_loss_preserves_unequal_masks_gradients_and_center():
    torch.manual_seed(17)
    pred = torch.randn(4, 11, 7, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(pred, requires_grad=True)
    mask = torch.arange(11)[None] < torch.tensor([1, 3, 7, 10])[:, None]
    old = CenteringSharpeningCE(7, 0.9, 0.1, 0.06).double()
    new = CenteringSharpeningCE(7, 0.9, 0.1, 0.06).double()
    old.center.fill_(0.3)
    new.load_state_dict(old.state_dict())
    expected = torch.stack([old(p[m], t[m], update_center=False)
                            for p, t, m in zip(pred, target, mask)]).mean()
    old.update_center_from(target[mask])
    actual = new.masked_batch(pred, target, mask)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(new.center, old.center)
    expected_grad = torch.autograd.grad(expected, pred)[0]
    actual_grad, target_grad = torch.autograd.grad(actual, (pred, target), allow_unused=True)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert target_grad is None
    assert (actual_grad[~mask] == 0).all()


def test_embedding_fills_batches_and_pools_each_clip_separately():
    cfg = get_config("laptop", smoke=True)
    cfg.batch_size = 4
    records = [SimpleNamespace(load_norm=lambda n=n: np.full((n, 33, 3), n, np.float32),
                               clip_name=str(n)) for n in (8, 16, 48, 64)]

    class Encoder:
        calls = []

        def eval(self):
            pass

        def embed(self, batch, mask):
            self.calls.append(len(batch))
            return batch.mean(dim=(1, 2))

    model = Encoder()
    embeddings = embed_records(model, records, cfg, "cpu")
    expected = [[n] * 3 for n in (8, 16, 48, 64)]
    np.testing.assert_array_equal(embeddings, expected)
    assert all(n == 4 for n in model.calls[:-1])
    assert sum(model.calls) == sum(len(sliding_windows(r.load_norm(), cfg.window_frames,
                                                       cfg.window_stride)) for r in records)


def test_embedding_matches_original_model_readout():
    from sjepa.masking_v2 import sample_target_mask

    cfg = get_config("laptop", smoke=True)
    cfg.batch_size = 4
    rng = np.random.default_rng(2)
    arrays = [rng.normal(size=(n, 33, 3)).astype(np.float32) for n in (8, 16, 48, 64)]
    records = [SimpleNamespace(load_norm=lambda a=a: a, clip_name=str(i))
               for i, a in enumerate(arrays)]
    model = build_model(cfg, device="cpu", repaired=True)
    model.eval()
    mask = torch.from_numpy(sample_target_mask(cfg.num_joints, cfg.num_time_tokens,
                                              np.random.default_rng(0), target_ratio=0.6))
    expected = []
    for a in arrays:
        windows = sliding_windows(a, cfg.window_frames, cfg.window_stride)
        pieces = [model.embed(torch.from_numpy(windows[i:i + 4]), mask).numpy()
                  for i in range(0, len(windows), 4)]
        expected.append(np.concatenate(pieces).mean(axis=0))
    np.testing.assert_allclose(embed_records(model, records, cfg, "cpu"), expected,
                               atol=1e-6, rtol=1e-5)


def test_atomic_write_and_cache_corruption(tmp_path):
    path = tmp_path / "payload.json"
    write_json(path, {"original": True})
    with pytest.raises(RuntimeError):
        with atomic_path(path) as temporary:
            temporary.write_text("partial")
            raise RuntimeError("interrupted")
    assert json.loads(path.read_text()) == {"original": True}
    manifest = tmp_path / "complete.json"
    write_manifest(manifest, "same-run", {"done": True}, [path.name])
    assert read_manifest(manifest, "same-run") == {"done": True}
    assert read_manifest(manifest, "changed-budget") is None
    path.write_text("corrupt")
    assert read_manifest(manifest, "same-run") is None
    manifest.write_text("{partial")
    assert read_manifest(manifest, "same-run") is None


@pytest.fixture
def toy_experiment(tmp_path):
    records = []
    rng = np.random.default_rng(42)
    for label in LABELS:
        for i in range(8):
            name = f"{label}_{i}"
            arr = rng.normal(size=(16 + 16 * (i % 3), 33, 3)).astype(np.float32)
            arr[:, :, 2] = rng.uniform(0.7, 1.0, arr.shape[:2])
            path = tmp_path / f"{name}.npz"
            save_sequence_npz(path, arr, arr, 15, name, label, name)
            records.append(SequenceRecord(path, label, name, name, len(arr)))
    inventory = {"cache": [dict(clip=r.clip_name, label=r.label, source_id=r.source_id,
                                n_frames=r.n_frames, cache_file=r.path.name,
                                sha256=file_sha256(r.path)) for r in records]}
    registry = make_registry(records, inventory, n_splits=2, inner_splits=2)
    return records, registry, get_config("laptop", smoke=True)


def test_identity_rejects_stale_data_and_changes_with_budget_config(toy_experiment):
    records, registry, cfg = toy_experiment
    original, features = cache_identity(records, registry, cfg, "cpu", 1, 1, 1)
    other, same_features = cache_identity(records, registry, cfg, "cpu", 2, 1, 1)
    assert digest(original) != digest(other)
    assert features == same_features  # raw features do not depend on training budget
    changed, _ = cache_identity(records, registry, replace(cfg, seed=123), "cpu", 1, 1, 1)
    assert digest(original) != digest(changed)
    records[0].path.write_bytes(b"changed since registry was loaded")
    with pytest.raises(ValueError, match="cache changed"):
        cache_identity(records, registry, cfg, "cpu", 1, 1, 1)


def test_cached_cv_reuses_work_and_matches_uncached(toy_experiment, tmp_path, monkeypatch):
    import sjepa.classical as classical
    import sjepa.full_experiment as experiment

    records, registry, cfg = toy_experiment
    uncached = run_cross_validation(records, registry, cfg, "cpu", 1, 1,
                                    tmp_path / "uncached", verbose=False)
    calls = []
    extract = classical.sequence_to_feature_vector

    def counted(record, fps):
        calls.append(record.clip_name)
        return extract(record, fps)

    monkeypatch.setattr(classical, "sequence_to_feature_vector", counted)
    cold = run_cross_validation(records, registry, cfg, "cpu", 1, 1, tmp_path / "cold",
                                cache_dir=tmp_path / "cache", verbose=False)
    assert len(calls) == len(records) == len(set(calls))
    assert cold["metrics"] == uncached["metrics"]
    assert cold["selections"] == uncached["selections"]

    def unexpected(*args, **kwargs):
        pytest.fail("warm cache must not train, embed, or extract features")

    monkeypatch.setattr(experiment, "run_fold", unexpected)
    monkeypatch.setattr(classical, "sequence_to_feature_vector", unexpected)
    warm = run_cross_validation(records, registry, cfg, "cpu", 1, 1, tmp_path / "warm",
                                cache_dir=tmp_path / "cache", verbose=False)
    assert warm["metrics"] == cold["metrics"]
    assert all(f["status"] == "cache hit" for f in warm["execution"]["folds"])
    assert (tmp_path / "warm/fold-0/ssl.pt").exists()


def test_completed_stages_survive_interrupted_fold(toy_experiment, tmp_path, monkeypatch):
    import sjepa.full_experiment as experiment

    records, registry, cfg = toy_experiment
    output = tmp_path / "interrupted"
    original = experiment.fit_probe

    def interrupt(*args):
        raise RuntimeError("interrupt after first checkpoint and embeddings")

    monkeypatch.setattr(experiment, "fit_probe", interrupt)
    with pytest.raises(RuntimeError, match="interrupt"):
        run_fold(records, registry, 0, cfg, "cpu", 1, 1, output, cache_context="toy")
    assert (output / "ssl.json").exists()
    monkeypatch.setattr(experiment, "fit_probe", original)
    train = experiment.train_checkpoint
    stages = []

    def counted(model, recs, config, reg, fold, stage, *args, **kwargs):
        stages.append(stage)
        return train(model, recs, config, reg, fold, stage, *args, **kwargs)

    monkeypatch.setattr(experiment, "train_checkpoint", counted)
    run_fold(records, registry, 0, cfg, "cpu", 1, 1, output, cache_context="toy")
    assert stages == ["continued"]


def test_cpu_parallel_folds_match_serial(toy_experiment, tmp_path):
    records, registry, cfg = toy_experiment
    serial = run_cross_validation(records, registry, cfg, "cpu", 1, 1,
                                  tmp_path / "serial", verbose=False)
    parallel = run_cross_validation(records, registry, cfg, "cpu", 1, 1,
                                    tmp_path / "parallel", fold_workers=2,
                                    cache_dir=tmp_path / "parallel-cache", verbose=False)
    assert serial["metrics"] == parallel["metrics"]
    assert serial["selections"] == parallel["selections"]


def test_parallel_feature_cache_preserves_order_and_recovers_corruption(toy_experiment, tmp_path):
    records, _, cfg = toy_experiment
    subset = records[:3]
    serial_cache = FeatureCache(tmp_path / "serial-features", cfg.target_fps)
    serial = serial_cache.matrix(subset)
    parallel = FeatureCache(tmp_path / "parallel-features", cfg.target_fps, jobs=2).matrix(subset)
    for kind in serial:
        np.testing.assert_array_equal(parallel[kind], serial[kind])
    (serial_cache.directory / f"{digest(subset[0].clip_name)}.npz").write_bytes(b"partial")
    recovered = serial_cache.matrix(subset)
    for kind in serial:
        np.testing.assert_array_equal(recovered[kind], serial[kind])


@pytest.mark.parametrize("kwargs", [{"fold_workers": 0}, {"cpu_threads": -1},
                                     {"feature_workers": 0}, {"updates": 0}])
def test_bad_execution_options_fail_before_writes(tmp_path, kwargs):
    arguments = dict(records=[], registry={}, cfg=None, device="cpu", updates=1,
                     more_updates=1, output_dir=tmp_path / "output")
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="positive integer"):
        run_cross_validation(**arguments)
    assert not (tmp_path / "output").exists()


def test_single_accelerator_cannot_run_parallel_folds(tmp_path):
    with pytest.raises(ValueError, match="single GPU/MPS"):
        run_cross_validation([], {}, None, "mps", 1, 1, tmp_path / "output", fold_workers=2)
