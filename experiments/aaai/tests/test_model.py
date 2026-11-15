"""Scientific invariants, rather than tests that duplicate implementation."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from aaai_gait.model import (JOINT_INDICES, ModelConfig, SJEPA, forecast_mask,
                             load_checkpoint, save_checkpoint, token_validity,
                             uniform_mask, vicreg)
from aaai_gait.training import (embed_windows, progressive_train, synthetic_windows,
                                train_model)


@pytest.fixture
def small():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    return SJEPA(ModelConfig(frames=8, patch_frames=2, dim=16, heads=2, depth=1))


def test_uniform_mask_is_independent_and_covers_exactly_whitelist():
    # Empirical inclusion probabilities catch biased joint or motion sampling.
    mask = uniform_mask(10000, 64, .5, generator=torch.Generator().manual_seed(10))
    assert torch.all(mask.sum(1) == 32)
    assert torch.all((mask.float().mean(0) - .5).abs() < .025)
    assert not torch.equal(mask[0], mask[1])
    assert JOINT_INDICES == (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    with pytest.raises(ValueError, match="whitelisted"):
        ModelConfig(joints=33)


def test_hidden_coordinates_cannot_reach_prediction(small):
    x = torch.tensor(synthetic_windows(3, 8))
    mask = uniform_mask(3, small.cfg.tokens, generator=torch.Generator().manual_seed(9))
    changed = x.clone()
    # Unflatten the mask to change every coordinate in hidden joint patches.
    hidden = mask.reshape(3, 4, 16).repeat_interleave(2, dim=1)
    changed[hidden] += 100
    original = small(x, mask)
    perturbed = small(changed, mask)
    torch.testing.assert_close(original["predicted"], perturbed["predicted"], rtol=0, atol=0)
    assert not torch.allclose(original["target"], perturbed["target"])


def test_forecast_excludes_all_future_observations(small):
    x = torch.tensor(synthetic_windows(2, 8))
    mask = forecast_mask(2, small.cfg)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 50
    torch.testing.assert_close(small(x, mask)["predicted"], small(changed, mask)["predicted"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="entire configured future"):
        small.loss(x, mask=uniform_mask(2, small.cfg.tokens), mode="forecast")


def test_forecast_context_covers_every_frame_used_for_scale():
    # An odd number of temporal patches must round the prefix upward.
    cfg = ModelConfig(frames=12, patch_frames=4)
    mask = forecast_mask(1, cfg)
    visible_frames = int((~mask[0]).sum()) // cfg.joints * cfg.patch_frames
    assert visible_frames >= cfg.frames / 2
    with pytest.raises(ValueError, match="at least half"):
        ModelConfig(forecast_context=.25)


def test_teacher_no_gradient_and_only_explicit_ema_update(small):
    x = torch.tensor(synthetic_windows(3, 8))
    old = {k: v.clone() for k, v in small.target_encoder.state_dict().items()}
    optimizer = torch.optim.AdamW((p for p in small.parameters() if p.requires_grad), lr=.001)
    result = small.loss(x)
    result["loss"].backward()
    assert all(p.grad is None for p in small.target_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in small.view_encoder.parameters())
    optimizer.step()
    for k, v in small.target_encoder.state_dict().items():
        torch.testing.assert_close(v, old[k], rtol=0, atol=0)
    small.update_target(.9, result["target_mean"])
    for k, v in small.target_encoder.state_dict().items():
        torch.testing.assert_close(v, old[k] * .9 + small.view_encoder.state_dict()[k] * .1)
    small.train()
    assert not small.target_encoder.training


def test_vicreg_variance_has_gradient_and_collapse_penalty():
    a = (torch.randn(8, 4) * .01).requires_grad_()
    b = (torch.randn(8, 4) * .01).requires_grad_()
    result = vicreg(a, b)
    result["variance"].backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert vicreg(torch.zeros(8, 4), torch.zeros(8, 4))["variance"] > .98
    with pytest.raises(ValueError, match="B >= 2"):
        vicreg(torch.zeros(1, 4), torch.zeros(1, 4))


def test_scores_and_embeddings_do_not_depend_on_batching(small):
    x = synthetic_windows(5, 8)
    all_scores = small.latent_residual(torch.tensor(x), masks=5)
    individual = torch.cat([small.latent_residual(torch.tensor(row[None]), masks=5)["score"] for row in x])
    torch.testing.assert_close(all_scores["score"], individual)
    np.testing.assert_allclose(embed_windows(small, x, batch_size=1),
                               embed_windows(small, x, batch_size=5), atol=1e-6)
    repeat = small.latent_residual(torch.tensor(x), masks=5)
    torch.testing.assert_close(all_scores["score"], repeat["score"], rtol=0, atol=0)


def test_stop_noise_is_predictor_only_and_eval_is_deterministic():
    model = SJEPA(ModelConfig(frames=8, patch_frames=2, dim=16, heads=2, depth=1, spe=True))
    x = torch.tensor(synthetic_windows(3, 8))
    mask = uniform_mask(3, model.cfg.tokens)
    a = model(x, mask, stochastic_positions=True, generator=torch.Generator().manual_seed(1))
    b = model(x, mask, stochastic_positions=True, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(a["predicted"], b["predicted"])
    torch.testing.assert_close(a["target"], b["target"], rtol=0, atol=0)
    c = model.latent_residual(x)
    d = model.latent_residual(x)
    torch.testing.assert_close(c["score"], d["score"], rtol=0, atol=0)


def test_checkpoint_protects_anchor_and_dataset_identity(small, tmp_path):
    p = tmp_path / "normal.pt"
    save_checkpoint(small, p, manifest_hash="fresh", split_hash="fixed", stage="normal")
    loaded, info = load_checkpoint(p, manifest_hash="fresh", split_hash="fixed")
    assert info["stage"] == "normal"
    for key, value in small.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    with pytest.raises(FileExistsError):
        save_checkpoint(small, p, manifest_hash="fresh", split_hash="fixed", stage="normal")
    with pytest.raises(ValueError, match="provenance"):
        load_checkpoint(p, manifest_hash="stale", split_hash="fixed")


def metadata(n=8):
    return pd.DataFrame({"label": ["normal"] * 4 + ["parkinsons"] * 4,
                         "split": ["train"] * 3 + ["test"] + ["train"] * 3 + ["test"],
                         "source_id": [f"source_{i}" for i in range(n)]})


def test_training_cannot_use_heldout_coordinates(small):
    x = synthetic_windows(8, 8)
    changed = x.copy()
    changed[[3, 7]] = np.nan  # Any use of heldout coordinates would fail finite check.
    other = copy.deepcopy(small)
    one = train_model(small, x, metadata(), epochs=1, batch_size=3)
    two = train_model(other, changed, metadata(), epochs=1, batch_size=3)
    assert one == two
    assert one[0]["conditions"] == ["normal"]
    for key, value in small.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
    leaked = metadata()
    leaked.loc[3, "source_id"] = leaked.loc[0, "source_id"]
    with pytest.raises(ValueError, match="more than one split"):
        train_model(small, x, leaked, epochs=1)


def test_progressive_training_preserves_normal_anchor(small):
    before = {key: value.clone() for key, value in small.state_dict().items()}
    updated, history = progressive_train(small, synthetic_windows(8, 8), metadata(),
                                          stages=["parkinsons"], epochs=1, batch_size=3)
    assert history[0]["conditions"] == ["normal", "parkinsons"]
    for key, value in small.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert any(not torch.equal(value, before[key]) for key, value in updated.state_dict().items())


def partially_observed(frames=8):
    x = torch.tensor(synthetic_windows(3, frames))
    # One missing coordinate invalidates its whole joint patch. Missing
    # positions occur in both context and target for the forecast mask.
    x[:, 0, 0, 0] = torch.nan
    x[:, frames // 2, 1, 1] = torch.nan
    return x


def test_missing_tokens_excluded_from_loss_and_center(small):
    x = partially_observed()
    mask = forecast_mask(len(x), small.cfg)
    output = small(x, mask)
    losses = small.loss(x, mask, mode="forecast")
    valid = token_validity(x, small.cfg)
    assert torch.all(valid.sum(1) == small.cfg.tokens - 2)
    assert torch.equal(losses["target_mask"], mask & valid)
    teacher = torch.softmax((output["target"] - small.target_center) / small.cfg.target_temperature, dim=-1)
    student = torch.log_softmax(output["predicted"] / small.cfg.student_temperature, dim=-1)
    per_token = -(teacher * student).sum(-1)
    expected = torch.stack([per_token[i][mask[i] & valid[i]].mean() for i in range(len(x))]).mean()
    torch.testing.assert_close(losses["prediction"], expected)
    torch.testing.assert_close(losses["target_mean"], output["target"][mask & valid].mean(0))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in small.parameters())


def test_invalid_patch_finite_components_cannot_influence_valid_outputs(small):
    x = partially_observed()
    changed = x.clone()
    # Preserve the same missing coordinates but replace the remaining observed
    # components of those invalid patches. Zero filling alone would fail this.
    changed[:, :2, 0, :] += 1e30
    changed[:, 4:6, 1, :] -= 1e30
    mask = forecast_mask(len(x), small.cfg)
    original = small(x, mask)
    perturbed = small(changed, mask)
    torch.testing.assert_close(original["predicted"], perturbed["predicted"], rtol=0, atol=0)
    torch.testing.assert_close(original["target"], perturbed["target"], rtol=0, atol=0)
    torch.testing.assert_close(small.embed(x), small.embed(changed), rtol=0, atol=0)


def test_future_missingness_does_not_change_predictor_context(small):
    x = torch.tensor(synthetic_windows(3, 8))
    changed = x.clone()
    changed[:, 4:, :4] = torch.nan
    mask = forecast_mask(len(x), small.cfg)
    original = small(x, mask)
    altered = small(changed, mask)
    # Predictor hidden queries stay present regardless of future observability.
    torch.testing.assert_close(original["predicted"], altered["predicted"], rtol=0, atol=0)
    assert (altered["target_mask"].sum(1) < original["target_mask"].sum(1)).all()


def test_missing_score_coverage_and_batch_determinism(small):
    x = partially_observed()
    output = small.latent_residual(x, masks=6)
    valid = token_validity(x, small.cfg).reshape_as(output["coverage"])
    assert torch.isnan(output["per_token"][~valid]).all()
    assert (output["coverage"][~valid] == 0).all()
    assert torch.isfinite(output["score"]).all()
    separately = torch.cat([small.latent_residual(row[None], masks=6)["score"] for row in x])
    torch.testing.assert_close(output["score"], separately)


def test_missing_context_or_targets_rejected_cleanly(small):
    x = torch.tensor(synthetic_windows(3, 8))
    with pytest.raises(ValueError, match="valid context"):
        small.embed(torch.full_like(x, torch.nan))
    with pytest.raises(ValueError, match="valid context"):
        small.loss(torch.full_like(x, torch.nan))
    context_missing, targets_missing = x.clone(), x.clone()
    context_missing[:, :4] = torch.nan
    targets_missing[:, 4:] = torch.nan
    with pytest.raises(ValueError, match="valid context"):
        small.loss(context_missing, mode="forecast")
    with pytest.raises(ValueError, match="valid target"):
        small.loss(targets_missing, mode="forecast")
    with pytest.raises(ValueError, match="valid target"):
        small.latent_residual(targets_missing, mode="forecast")


def test_spe_uses_separate_rng_and_preserves_all_mask_draws(small):
    from dataclasses import replace
    spe_model = SJEPA(replace(small.cfg, spe=True))
    x = partially_observed()
    first = torch.Generator().manual_seed(57)
    second = torch.Generator().manual_seed(57)
    positions = torch.Generator().manual_seed(99)
    for _ in range(3):
        a = small.loss(x, generator=first)
        b = spe_model.loss(x, generator=second, positional_generator=positions)
        assert torch.equal(a["mask"], b["mask"])
        assert torch.equal(a["target_mask"], b["target_mask"])
        assert torch.equal(first.get_state(), second.get_state())


def test_training_with_partial_observations_stays_finite(small):
    x = synthetic_windows(8, 8)
    x[:, 0, 0, 0] = np.nan
    x[:, 4, 1, 1] = np.nan
    history = train_model(small, x, metadata(), epochs=1, batch_size=3)
    assert np.isfinite(history[0]["loss"])
    assert 0 < history[0]["observed_token_fraction"] < 1
    assert 0 < history[0]["supervised_token_fraction"] < .5
    assert np.isfinite(embed_windows(small, x)).all()
