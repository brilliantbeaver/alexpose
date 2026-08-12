"""Regression tests for the Phase 1 correctness repairs (D1, D2, D7).

Each test is written to FAIL on the legacy model/mask and PASS on the repaired
one, so it demonstrates the defect and then guards the fix. We deliberately avoid
weak checks like "loss decreased" or "teacher != student", which a collapsed or
shortcut model can also satisfy.

Run:  SJEPA_SMOKE=1 python -m pytest sjepa/tests/test_correctness.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_EXP_DIR = Path(__file__).resolve().parents[2]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from sjepa.config import get_config  # noqa: E402
from sjepa.models import build_model  # noqa: E402
from sjepa.masking_v2 import (  # noqa: E402
    sample_target_mask, sample_mask_batch, mask_bank_stats,
)


def _cfg():
    return get_config("laptop", smoke=True)


# ---------------------------------------------------------------------------
# D1: predictor target-position identity
# ---------------------------------------------------------------------------
def test_legacy_predictor_has_no_position_identity():
    """The E0 predictor produces IDENTICAL hidden predictions (documents D1)."""
    cfg = _cfg()
    model = build_model(cfg, device="cpu", repaired=False)
    from sjepa.masking import AnatomicalMaskSampler
    s = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)
    ctx = torch.from_numpy(s.context_mask)
    tgt_idx = torch.from_numpy(s.target_indices).long()
    x = torch.randn(2, cfg.window_frames, cfg.num_joints, cfg.in_channels)
    pred, _ = model(x, x, ctx)
    tgt = pred[:, tgt_idx, :]                        # (B, M, dim)
    std_across_targets = tgt.std(dim=1).mean().item()
    assert std_across_targets < 1e-5, (
        f"legacy predictor unexpectedly varied across targets (std={std_across_targets})")


def test_repaired_predictor_has_position_identity():
    """Repaired predictor makes DIFFERENT predictions per target position."""
    cfg = _cfg()
    model = build_model(cfg, device="cpu", repaired=True)
    rng = np.random.default_rng(0)
    tgt = torch.from_numpy(
        sample_mask_batch(2, cfg.num_joints, cfg.num_time_tokens, rng))
    ctx = ~tgt
    x = torch.randn(2, cfg.window_frames, cfg.num_joints, cfg.in_channels)
    pred, _ = model.forward_repaired(x, x, ctx)
    # Std across the hidden target positions must be clearly non-zero.
    stds = []
    for b in range(2):
        idx = torch.nonzero(tgt[b], as_tuple=False).squeeze(1)
        stds.append(pred[b, idx, :].std(dim=0).mean().item())
    assert min(stds) > 1e-3, f"repaired predictions still near-identical: {stds}"


def test_position_permutation_changes_predictions():
    """Permuting the predictor's position tags changes predictions at HIDDEN slots.

    Hardened per Codex AR-1 P2: we inspect the actual hidden (target) slots (not an
    all-visible sequence), use a DETERMINISTIC non-identity roll permutation (so the
    test can never accidentally be the identity), and check spatial and temporal
    tags independently.
    """
    cfg = _cfg()
    model = build_model(cfg, device="cpu", repaired=True)
    model.eval()
    rng = np.random.default_rng(0)
    tgt = torch.from_numpy(
        sample_mask_batch(1, cfg.num_joints, cfg.num_time_tokens, rng))  # (1, N)
    ctx = ~tgt
    hidden_idx = torch.nonzero(tgt[0], as_tuple=False).squeeze(1)
    x = torch.randn(1, cfg.window_frames, cfg.num_joints, cfg.in_channels)
    with torch.no_grad():
        base, _ = model.forward_repaired(x, x, ctx)
        base_hidden = base[0, hidden_idx, :].clone()

        # Deterministic non-identity roll of the SPATIAL (joint) tags.
        roll = torch.roll(torch.arange(cfg.num_joints), shifts=1)
        model.predictor.spatial_pos.copy_(model.predictor.spatial_pos[:, :, roll, :])
        sp_perm, _ = model.forward_repaired(x, x, ctx)
        assert not torch.allclose(base_hidden, sp_perm[0, hidden_idx, :], atol=1e-4), (
            "hidden predictions unchanged when SPATIAL position tags were permuted")

    # Rebuild and permute the TEMPORAL tags independently.
    model2 = build_model(cfg, device="cpu", repaired=True)
    model2.eval()
    with torch.no_grad():
        base2, _ = model2.forward_repaired(x, x, ctx)
        roll_t = torch.roll(torch.arange(cfg.num_time_tokens), shifts=1)
        model2.predictor.temporal_pos.copy_(model2.predictor.temporal_pos[:, roll_t, :, :])
        tp_perm, _ = model2.forward_repaired(x, x, ctx)
        assert not torch.allclose(base2[0, hidden_idx, :], tp_perm[0, hidden_idx, :], atol=1e-4), (
            "hidden predictions unchanged when TEMPORAL position tags were permuted")


# ---------------------------------------------------------------------------
# D2: every joint gets context gradient over a mask bank
# ---------------------------------------------------------------------------
def test_every_joint_receives_context_gradient():
    """Over several per-example masks, every joint's spatial embedding gets grad."""
    cfg = _cfg()
    model = build_model(cfg, device="cpu", repaired=True)
    from sjepa.losses import CenteringSharpeningCE
    ce = CenteringSharpeningCE(cfg.encoder_dim, cfg.center_beta, cfg.tau_pred, cfg.tau_target)
    rng = np.random.default_rng(1)
    spatial = model.view_encoder.tokenizer.spatial_emb
    got_grad = torch.zeros(cfg.num_joints, dtype=torch.bool)
    for _ in range(8):
        tgt = torch.from_numpy(
            sample_mask_batch(4, cfg.num_joints, cfg.num_time_tokens, rng))
        ctx = ~tgt
        x = torch.randn(4, cfg.window_frames, cfg.num_joints, cfg.in_channels)
        pred, targ = model.forward_repaired(x, x, ctx)
        loss = torch.zeros(())
        for b in range(4):
            idx = torch.nonzero(tgt[b], as_tuple=False).squeeze(1)
            loss = loss + ce(pred[b, idx, :], targ[b, idx, :])
        model.zero_grad()
        loss.backward()
        g = spatial.grad.detach().abs().sum(dim=(0, 1, 3))     # (V,)
        got_grad |= (g > 0)
        model.zero_grad()
    assert got_grad.all(), (
        f"joints never receiving context gradient: "
        f"{torch.nonzero(~got_grad).squeeze(1).tolist()}")


# ---------------------------------------------------------------------------
# Mask bank coverage gates
# ---------------------------------------------------------------------------
def test_mask_bank_meets_coverage_gates():
    cfg = _cfg()
    stats = mask_bank_stats(cfg.num_joints, cfg.num_time_tokens, n_masks=512, seed=0)
    # Every joint visible in >=20% and targeted in >=10% of masks.
    assert stats.joint_visible_frac.min() >= 0.20, (
        f"min joint-visible frac too low: {stats.joint_visible_frac.min():.3f}")
    assert stats.joint_target_frac.min() >= 0.10, (
        f"min joint-target frac too low: {stats.joint_target_frac.min():.3f}")
    # Mean target fraction lands in a sensible band around the requested 0.60.
    assert 0.35 <= stats.mean_target_frac <= 0.80, stats.mean_target_frac


def test_masks_are_per_example_and_diverse():
    cfg = _cfg()
    rng = np.random.default_rng(3)
    batch = sample_mask_batch(16, cfg.num_joints, cfg.num_time_tokens, rng)
    assert batch.shape == (16, cfg.num_tokens)
    # Rows differ (per-example, not one shared pattern).
    uniq = {row.tobytes() for row in batch}
    assert len(uniq) >= 8, f"masks not diverse across examples: {len(uniq)} unique"
    # Every row has non-empty context and non-empty target.
    assert batch.any(axis=1).all() and (~batch).any(axis=1).all()


def test_masks_deterministic_under_seed():
    cfg = _cfg()
    a = sample_target_mask(cfg.num_joints, cfg.num_time_tokens, np.random.default_rng(7))
    b = sample_target_mask(cfg.num_joints, cfg.num_time_tokens, np.random.default_rng(7))
    assert np.array_equal(a, b), "same seed produced different masks"


def test_clinical_joints_targeted_more_often():
    """The clinical bias should make clinical joints targeted at least as often."""
    from sjepa.masking_v2 import CLINICAL_JOINTS
    cfg = _cfg()
    stats = mask_bank_stats(cfg.num_joints, cfg.num_time_tokens, n_masks=512, seed=0)
    clin = np.array(sorted(CLINICAL_JOINTS))
    non_clin = np.array([j for j in range(cfg.num_joints) if j not in CLINICAL_JOINTS])
    assert stats.joint_target_frac[clin].mean() >= stats.joint_target_frac[non_clin].mean()


if __name__ == "__main__":
    test_legacy_predictor_has_no_position_identity(); print("[ok] D1 legacy defect reproduced")
    test_repaired_predictor_has_position_identity(); print("[ok] D1 repaired: position identity")
    test_position_permutation_changes_predictions(); print("[ok] D1 permutation sensitivity")
    test_every_joint_receives_context_gradient(); print("[ok] D2 every joint gets context grad")
    test_mask_bank_meets_coverage_gates(); print("[ok] mask coverage gates")
    test_masks_are_per_example_and_diverse(); print("[ok] D7 per-example diverse masks")
    test_masks_deterministic_under_seed(); print("[ok] masks deterministic")
    test_clinical_joints_targeted_more_often(); print("[ok] clinical target bias")
    print("ALL CORRECTNESS TESTS PASSED")
