"""Falsify the matched experiment's information and sampling boundaries."""
from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from aaai_gait.context_prediction import (
    BRANCHES, TARGET_TYPES, ContextBundle, ContextConfig, context_access,
    fit_context_models, load_context_bundle, make_mask_bank, save_context_bundle,
    score_context_models, _matched_mean, _sample_rows, _groups,
)
from aaai_gait.model import JOINT_INDICES, SJEPA, token_validity
from aaai_gait.training import synthetic_windows


@pytest.fixture
def cfg():
    torch.set_num_threads(2)
    return ContextConfig(pretrain_updates=2, predictor_updates=2, batch_size=4,
                         validation_interval=2, validation_batches=1, dim=16, depth=2, heads=2)


@pytest.fixture
def bundle(cfg):
    torch.manual_seed(5)
    return ContextBundle(SJEPA(cfg.model_config()), cfg).eval()


def test_heads_start_identically_with_independent_parameters(bundle):
    for kind in TARGET_TYPES:
        own = bundle.heads[kind + "_own"]
        for branch in ("patch", "full"):
            other = bundle.heads[kind + "_" + branch]
            for a, b in zip(own.parameters(), other.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                assert a.data_ptr() != b.data_ptr()


def test_own_predictions_ignore_all_excluded_joint_coordinates(bundle, cfg):
    x = torch.tensor(synthetic_windows(3))
    mask = make_mask_bank(cfg, 17, 1).expand(len(x), -1)
    joint = JOINT_INDICES.index(27)
    selected = torch.full((len(x),), joint)
    changed = x.clone()
    others = [j for j in range(16) if j != joint]
    changed[:, :, others] += 100
    first = bundle.predict_joint(x, mask, selected)
    second = bundle.predict_joint(changed, mask, selected)
    for kind in TARGET_TYPES:
        torch.testing.assert_close(first["predictions"][kind + "_own"], second["predictions"][kind + "_own"], rtol=0, atol=0)
    torch.testing.assert_close(first["losses"]["coordinate"]["own"], second["losses"]["coordinate"]["own"], equal_nan=True)
    # Contextual target dependence is real and must not be confused with leakage.
    assert not torch.allclose(bundle.targets(x)["latent"], bundle.targets(changed)["latent"])


def test_patch_predictions_ignore_other_joints_outside_target_patch(bundle, cfg):
    x = torch.tensor(synthetic_windows(1))
    mask = make_mask_bank(cfg, 17, 1)
    slot = JOINT_INDICES.index(27)
    selected = torch.tensor([slot])
    first = bundle.predict_joint(x, mask, selected)
    patch = int(first["eligible"].nonzero()[0, 1])
    changed = x.clone()
    for t in range(32):
        if t // 4 != patch:
            changed[:, t, [j for j in range(16) if j != slot]] += 100
    second = bundle.predict_joint(changed, mask, selected)
    for kind in TARGET_TYPES:
        torch.testing.assert_close(first["predictions"][kind + "_patch"][0], second["predictions"][kind + "_patch"][0], rtol=0, atol=0)
    assert not torch.allclose(first["predictions"]["coordinate_full"][0], second["predictions"]["coordinate_full"][0])


def test_hidden_local_target_cannot_enter_any_predictor(bundle, cfg):
    x = torch.tensor(synthetic_windows(2))
    mask = make_mask_bank(cfg, 23, 1).expand(len(x), -1)
    changed = x.clone()
    changed[mask.reshape(len(x), 8, 16).repeat_interleave(4, 1)] += 80
    selected = torch.tensor([10, 10])
    first, second = bundle.predict_joint(x, mask, selected), bundle.predict_joint(changed, mask, selected)
    for name in first["predictions"]:
        torch.testing.assert_close(first["predictions"][name], second["predictions"][name], rtol=0, atol=0)


def test_context_access_is_nested_and_masked_before_every_layer(bundle, cfg):
    x = torch.tensor(synthetic_windows(2))
    mask = make_mask_bank(cfg, 5, 2)
    valid = token_validity(x, bundle.backbone.cfg)
    index = torch.tensor([3 * 16 + 10, 5 * 16 + 7])
    own, patch, full = [context_access(mask, valid, index, branch) for branch in BRANCHES]
    assert not (own & ~patch).any() and not (patch & ~full).any()
    assert not (full & mask).any()
    # The two encoder layers share the same structural restriction. Information
    # isolation tests above would fail if the first layer were globally encoded.
    assert len(bundle.backbone.view_encoder.transformer.layers) == 2


def test_mask_bank_and_common_support_do_not_depend_on_coordinates(bundle, cfg):
    x = synthetic_windows(2)
    missing = x.copy()
    missing[:, :, 2] = np.nan
    first = score_context_models(bundle, x, mask_repeats=2, joints=(13, 27))
    second = score_context_models(bundle, missing, mask_repeats=2, joints=(13, 27))
    keys = ["window_index", "joint", "patch_index", "mask_id", "target_type"]
    pd.testing.assert_frame_equal(first[keys], second[keys])
    lost = second[second.joint.eq(13)]
    assert not lost.valid.any()
    assert lost[["loss_own", "loss_patch", "loss_full"]].isna().all().all()
    assert second.attrs["requested_targets"] == first.attrs["requested_targets"]
    assert second.attrs["valid_targets"] < first.attrs["valid_targets"]
    assert not second.duplicated(keys).any()


def test_at_least_two_visible_own_patches_without_redraw(bundle, cfg):
    x = torch.tensor(synthetic_windows(1))
    mask = make_mask_bank(cfg, 11, 1).clone()
    slot = 10
    mask[:, slot::16] = True
    mask[:, slot] = False
    original = mask.clone()
    output = bundle.predict_joint(x, mask, torch.tensor([slot]))
    assert output["own_visible_count"].item() == 1
    assert not output["eligible"].any()
    assert torch.equal(mask, original)


def test_every_head_receives_gradient_teacher_and_center_frozen(bundle, cfg):
    x = torch.tensor(synthetic_windows(4))
    mask = make_mask_bank(cfg, 7, 4)
    center = bundle.backbone.target_center.clone()
    output = bundle.predict_joint(x, mask, torch.tensor([2, 3, 10, 11]))
    for kind in TARGET_TYPES:
        for branch in BRANCHES:
            loss = _matched_mean(output["losses"][kind][branch], output["eligible"])
            loss.backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bundle.heads[kind + "_" + branch].parameters())
    assert all(p.grad is None for p in bundle.backbone.parameters())
    torch.testing.assert_close(center, bundle.backbone.target_center, rtol=0, atol=0)


def test_scores_independent_of_batching_and_repeat(bundle):
    x = synthetic_windows(3)
    x[0, 0, 2, 0] = np.nan
    a = score_context_models(bundle, x, mask_repeats=2, batch_size=1, joints=(13, 27))
    b = score_context_models(bundle, x, mask_repeats=2, batch_size=3, joints=(13, 27))
    keys = ["window_index", "joint", "patch_index", "mask_id", "target_type"]
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, rtol=1e-5, atol=1e-5)


def test_fixed_budget_normal_only_and_validation_cannot_update_weights(cfg):
    x = synthetic_windows(12)
    kwargs = dict(config=cfg, seed=8, train_source_ids=["train0"] * 4 + ["train1"] * 4,
                  validation_source_ids=["val0"] * 4, train_labels=["normal"] * 8,
                  validation_labels=["normal"] * 4)
    a, history = fit_context_models(x[:8], x[8:], **kwargs)
    b, _ = fit_context_models(x[:8], x[8:] + .1, **kwargs)
    assert history["pretrain_updates_completed"] == 2
    assert history["predictor_updates_completed_per_head"] == 2
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="overlap"):
        fit_context_models(x[:8], x[8:], **{**kwargs, "validation_source_ids": ["train0"] * 4})
    with pytest.raises(ValueError, match="Only normal"):
        fit_context_models(x[:8], x[8:], **{**kwargs, "train_labels": ["normal"] * 7 + ["stroke"]})


def test_source_balanced_draws_do_not_overweight_long_recording():
    ids = ["long"] * 100 + ["short"]
    rows = _sample_rows(np.random.default_rng(31), _groups(ids, len(ids)), 10000)
    assert .47 < (rows == 100).mean() < .53


def test_provided_backbone_is_immutable_and_missing_training_is_finite(bundle, cfg):
    old = {name: value.clone() for name, value in bundle.backbone.state_dict().items()}
    x = synthetic_windows(12)
    x[:, :, :4] = np.nan
    fitted, history = fit_context_models(x[:8], x[8:], config=cfg, pretrained_model=bundle.backbone)
    assert history["pretrain_updates_completed"] == 0
    assert history["predictor_updates_completed_per_head"] == cfg.predictor_updates
    assert all(np.isfinite(value) for name, value in history["predictors"][-1].items()
               if name.startswith("train_"))
    for name, value in fitted.backbone.state_dict().items():
        torch.testing.assert_close(value, old[name], rtol=0, atol=0)
        torch.testing.assert_close(bundle.backbone.state_dict()[name], old[name], rtol=0, atol=0)
    assert fitted.provenance["frozen_backbone_sha256"]


def test_nonfinite_and_provenance_boundaries(bundle, cfg, tmp_path):
    x = synthetic_windows(2)
    corrupt = x.copy(); corrupt[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        score_context_models(bundle, corrupt, mask_repeats=1)
    with pytest.raises(ValueError, match="unobserved"):
        score_context_models(bundle, np.full_like(x, np.nan), mask_repeats=1)
    # Numerical failure on an eligible target must never become a zero loss.
    assert torch.isnan(_matched_mean(torch.tensor([[float('nan'), 1.]]), torch.tensor([[True, True]])))
    path = save_context_bundle(bundle, tmp_path / "context.pt", provenance={"fold_hash": "a"})
    restored = load_context_bundle(path, expected_provenance={"fold_hash": "a"})
    for name, value in bundle.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    with pytest.raises(ValueError, match="provenance"):
        load_context_bundle(path, expected_provenance={"fold_hash": "b"})
    with pytest.raises(FileExistsError):
        save_context_bundle(bundle, path)
