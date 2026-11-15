"""Integration checks for the notebook orchestration and evaluation units."""

import json

import numpy as np
import pandas as pd
import torch

from aaai_gait.demo import synthetic_corpus
from aaai_gait.settings import Settings
from aaai_gait.training import fit_frozen_probe
from aaai_gait.workflow import (anomaly_evaluation, capstone, normal_anchor,
                                prepare_corpus, select_probe)


def test_probe_selection_ignores_all_test_values_and_labels():
    x, meta = synthetic_corpus(frames=8)
    embeddings = x.mean(axis=1).reshape(len(x), -1)
    original, choice = select_probe(embeddings, meta)
    altered = embeddings.copy()
    altered[meta.split.eq("test")] = np.nan
    changed = meta.copy()
    changed.loc[changed.split.eq("test"), "label"] = "unknown_test_label"
    candidate, second_choice = select_probe(altered, changed)
    assert choice == second_choice
    np.testing.assert_array_equal(original[-1].coef_, candidate[-1].coef_)


def test_probe_scaling_uses_training_sources_only():
    x, meta = synthetic_corpus(frames=8)
    z = x.mean(axis=1).reshape(len(x), -1)
    # Deliberately oversample one recording with a distinct value. An
    # unweighted scaler would pass this test on the balanced original fixture.
    z = np.concatenate([z, np.repeat(z[:1] + .2, 5, axis=0)])
    meta = pd.concat([meta, meta.iloc[[0] * 5]], ignore_index=True)
    probe = fit_frozen_probe(z, meta)
    train = meta.split.eq("train").to_numpy()
    counts = meta.loc[train, "source_id"].value_counts()
    weights = meta.loc[train, "source_id"].map(lambda s: 1 / counts[s]).to_numpy()
    expected = np.average(z[train], axis=0, weights=weights)
    assert not np.allclose(expected, z[train].mean(axis=0))
    np.testing.assert_allclose(probe.named_steps["standardscaler"].mean_, expected, atol=1e-7)


def test_probe_imputation_fits_only_training_observations():
    x, meta = synthetic_corpus(frames=8)
    z = x.mean(axis=1).reshape(len(x), -1)
    train = meta.split.eq("train").to_numpy()
    first = np.flatnonzero(train)[:4]
    z[first, 0] = np.nan
    z[:, 1] = np.nan  # Preserve an entirely unobserved feature as a zero column.
    z[~train, 0] = 1e6
    probe = fit_frozen_probe(z, meta)
    imputer = probe.named_steps["simpleimputer"]
    assert imputer.statistics_[0] == np.nanmedian(z[train, 0])
    assert imputer.statistics_[1] == 0
    assert np.isfinite(probe.predict_proba(z)).all()


class ScoreFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def latent_residual(self, x, mode="masked"):
        return {"score": x[:, 0, 0, 0]}


def test_anomaly_threshold_counts_sources_not_windows():
    rows, values = [], []
    for source in range(20):
        for sequence in range(2):
            for window in range(3):
                rows.append(dict(source_id=f"v{source}", sequence_id=f"v{source}s{sequence}",
                                 split="val", label="normal"))
                values.append(float(source))
    for source, label, score in (("normal_test", "normal", 5.), ("abnormal_test", "stroke", 20.)):
        rows.append(dict(source_id=source, sequence_id=source, split="test", label=label))
        values.append(score)
    x = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1, 1)
    _, metrics = anomaly_evaluation(ScoreFixture(), x, pd.DataFrame(rows), batch_size=11)
    assert metrics["n_normal_calibration_sources"] == 20
    assert metrics["threshold"] == 19
    assert metrics["false_positive_rate"] == 0
    assert metrics["sensitivity"] == 1


def test_synthetic_capstone_keeps_paired_order_and_separate_artifacts(tmp_path):
    torch.set_num_threads(2)
    settings = Settings(demo=True, epochs=1, batch_size=8, frames=8, artifact_dir=tmp_path)
    x, meta, provenance = prepare_corpus(settings)
    result = capstone(x, meta, provenance, settings, bootstrap_repeats=30)
    assert result["synthetic"] is True
    assert "synthetic_demo" in str(provenance["run_dir"])
    assert len(result["methods"]) == 7
    assert set(result["methods"]) == {"random_forest", "normal_jepa", "progressive_jepa",
                                     "untrained_encoder", "mean_pose", "pose_availability", "prior_baseline"}
    reference = None
    for name in result["methods"]:
        predictions = pd.read_csv(provenance["run_dir"] / f"test_predictions_{name}.csv")
        identity = predictions[["sequence_id", "source_id", "label", "split"]]
        if reference is None:
            reference = identity
        else:
            pd.testing.assert_frame_equal(identity, reference)
        probabilities = predictions.filter(like="p_").to_numpy()
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    anchor, _ = normal_anchor(x, meta, provenance, settings)
    _, metrics = anomaly_evaluation(anchor, x, meta, batch_size=8)
    assert metrics["n_normal_calibration_sources"] == 1
    assert metrics["finite_threshold"] is False
    saved = json.loads((provenance["run_dir"] / "provenance.json").read_text())
    assert saved["synthetic"] is True
    assert saved["corpus_sha256"] == provenance["manifest_hash"]
