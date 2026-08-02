"""Smoke tests for the sjepa package.

Runs with no network and no video decode: it builds the model on a synthetic
window tensor, runs a training step for BOTH the laptop and gpu profiles (shrunk
by SJEPA_SMOKE so it is fast), and checks the properties that matter:

* shapes flow through the two-lane forward pass,
* the loss is finite and decreases over a handful of steps,
* the EMA target tracks but is not identical to the view encoder,
* a checkpoint round-trips,
* the anatomical mask partitions tokens exactly (12 masked joints, no overlap).

Run:  SJEPA_SMOKE=1 python -m pytest sjepa/tests/test_smoke.py -q
Or:   SJEPA_SMOKE=1 python sjepa/tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Make the experiment folder importable when run directly.
_EXP_DIR = Path(__file__).resolve().parents[2]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from sjepa.config import get_config, ANATOMICAL_MASK_IDX
from sjepa.masking import AnatomicalMaskSampler, joint_token_mask
from sjepa.models import build_model, SJEPA
from sjepa.train import train_sjepa, save_checkpoint, load_checkpoint


def _synthetic_dataset(cfg, n=24):
    """A tiny in-memory dataset of random-walk skeletons with 3 fake classes."""
    class DS:
        def __init__(self):
            rng = np.random.default_rng(cfg.seed)
            self.x = []
            self.y = []
            for i in range(n):
                base = rng.normal(0, 1, size=(cfg.num_joints, cfg.in_channels))
                steps = np.cumsum(rng.normal(0, 0.05, size=(cfg.window_frames, cfg.num_joints, cfg.in_channels)), axis=0)
                self.x.append((base[None] + steps).astype(np.float32))
                self.y.append(i % 3)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return torch.from_numpy(self.x[i]), int(self.y[i])

    return DS()


def test_mask_partition():
    cfg = get_config("laptop", smoke=True)
    sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)
    tm = sampler.target_mask
    cm = sampler.context_mask
    assert tm.shape[0] == cfg.num_tokens
    # No overlap, full cover.
    assert not (tm & cm).any()
    assert (tm | cm).all()
    # Exactly the 12 masked joints across every time block.
    expected = len(ANATOMICAL_MASK_IDX) * cfg.num_time_tokens
    assert int(tm.sum()) == expected
    # The masked token pattern repeats the joint pattern each time block.
    per_joint = joint_token_mask(cfg.num_joints, 1)
    assert int(per_joint.sum()) == len(ANATOMICAL_MASK_IDX)


def _run_profile(profile: str):
    cfg = get_config(profile, smoke=True)
    device = "cpu"  # deterministic and always available in CI
    model = build_model(cfg, device=device)

    # Shapes: one forward pass.
    sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)
    context_mask = torch.from_numpy(sampler.context_mask)
    x = torch.randn(4, cfg.window_frames, cfg.num_joints, cfg.in_channels)
    pred, targ = model(x, x, context_mask)
    assert pred.shape == (4, cfg.num_tokens, cfg.encoder_dim)
    assert targ.shape == (4, cfg.num_tokens, cfg.encoder_dim)

    # Snapshot target weights before training.
    before = [p.clone() for p in model.target_encoder.parameters()]

    ds = _synthetic_dataset(cfg)
    state = train_sjepa(model, ds, cfg, epochs=6, use_vicreg=False, device=device)

    assert len(state.losses) > 0
    assert all(np.isfinite(state.losses)), "loss went non-finite"
    early = np.mean(state.losses[: max(1, len(state.losses) // 3)])
    late = np.mean(state.losses[-max(1, len(state.losses) // 3):])
    assert late <= early + 1e-3, f"loss did not decrease ({early:.4f} -> {late:.4f})"

    # EMA target changed (tracked the view encoder) but is not identical to it.
    after = list(model.target_encoder.parameters())
    changed = any(not torch.allclose(a, b) for a, b in zip(after, before))
    assert changed, "EMA target did not update"
    diffs = [torch.norm(tp - vp).item()
             for tp, vp in zip(model.target_encoder.parameters(), model.view_encoder.parameters())]
    assert sum(diffs) > 0, "target encoder identical to view encoder (would collapse)"

    # VICReg path runs without error.
    train_sjepa(model, ds, cfg, epochs=1, use_vicreg=True, class_aware_vicreg=True, device=device)

    # Checkpoint round-trip.
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "m.pt"
        save_checkpoint(ckpt, model, cfg)
        model2 = build_model(cfg, device=device)
        load_checkpoint(ckpt, model2, map_location=device)
        e1 = model.embed(x, torch.from_numpy(sampler.target_mask))
        e2 = model2.embed(x, torch.from_numpy(sampler.target_mask))
        assert torch.allclose(e1, e2, atol=1e-5), "checkpoint did not restore embeddings"


def test_laptop_profile():
    _run_profile("laptop")


def test_gpu_profile():
    # 'gpu' profile shrunk by smoke; still exercises the larger-config code path on CPU.
    _run_profile("gpu")


if __name__ == "__main__":
    os.environ.setdefault("SJEPA_SMOKE", "1")
    test_mask_partition()
    print("[ok] mask partition")
    test_laptop_profile()
    print("[ok] laptop profile: shapes, loss decreases, EMA tracks, VICReg, checkpoint")
    test_gpu_profile()
    print("[ok] gpu profile (smoke-shrunk)")
    print("ALL SMOKE TESTS PASSED")
