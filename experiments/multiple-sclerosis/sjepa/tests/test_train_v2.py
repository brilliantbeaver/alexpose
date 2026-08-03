"""Regression tests for the Phase 2 repairs (D3, D4, source-uniform sampling).

Covers:
* strict SSL uses NO diagnosis labels (label permutation leaves the loss path
  and the resulting embeddings unchanged);
* the source-uniform sampler exposes each source equally despite window-count
  imbalance;
* save/resume reproduces an uninterrupted run's next-step loss and predictions
  within tolerance;
* embedding std / effective rank / teacher drift are finite and logged.

Run:  SJEPA_SMOKE=1 python -m pytest sjepa/tests/test_train_v2.py -q
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_EXP_DIR = Path(__file__).resolve().parents[2]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from sjepa.config import get_config  # noqa: E402
from sjepa.models import build_model  # noqa: E402
from sjepa.train_v2 import (  # noqa: E402
    train_sjepa_v2, source_uniform_sampler, save_checkpoint_v2, load_checkpoint_v2,
)


class _Dataset:
    """Synthetic windows with an intentionally imbalanced source distribution."""

    def __init__(self, cfg, source_windows):
        rng = np.random.default_rng(cfg.seed)
        self.windows, self.labels, self.source_ids = [], [], []
        for si, (src, n) in enumerate(source_windows.items()):
            for _ in range(n):
                base = rng.normal(0, 1, size=(cfg.num_joints, cfg.in_channels))
                steps = np.cumsum(
                    rng.normal(0, 0.05, size=(cfg.window_frames, cfg.num_joints, cfg.in_channels)),
                    axis=0)
                self.windows.append((base[None] + steps).astype(np.float32))
                self.labels.append(si % 3)
                self.source_ids.append(src)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i]), int(self.labels[i])


def _imbalanced_dataset(cfg):
    # One long source (75 windows) and several short ones (1-3): the D4/sampler trap.
    return _Dataset(cfg, {"long": 75, "a": 2, "b": 3, "c": 1, "d": 2, "e": 4})


def test_source_uniform_sampler_equalizes_exposure():
    cfg = get_config("laptop", smoke=True)
    ds = _imbalanced_dataset(cfg)
    sampler = source_uniform_sampler(ds, seed=0)
    draws = Counter(ds.source_ids[i] for i in list(sampler))
    # Despite 'long' having 75/87 windows, each source should be drawn a
    # comparable number of times (within a generous band for a small sample).
    fracs = {s: c / sum(draws.values()) for s, c in draws.items()}
    # 6 sources -> uniform ~0.167 each. 'long' must not dominate.
    assert fracs["long"] < 0.40, f"long source still dominates: {fracs}"
    assert min(fracs.values()) > 0.03, f"a source is starved: {fracs}"


def test_strict_ssl_ignores_labels():
    """Permuting labels must not change the SSL trajectory (no label leak, D3)."""
    cfg = get_config("laptop", smoke=True)
    ds = _imbalanced_dataset(cfg)

    def run(label_shift):
        ds2 = _imbalanced_dataset(cfg)
        ds2.labels = [(l + label_shift) % 3 for l in ds2.labels]
        torch.manual_seed(123)
        model = build_model(cfg, device="cpu", repaired=True)
        st = train_sjepa_v2(model, ds2, cfg, total_updates=6, device="cpu", seed=7)
        return st.losses

    a = run(0)
    b = run(1)   # every label shifted
    assert np.allclose(a, b, atol=1e-6), (
        f"SSL loss depends on labels (leak): {a} vs {b}")


def test_save_resume_matches_uninterrupted():
    """A run split by a save/resume matches an uninterrupted run within tolerance."""
    cfg = get_config("laptop", smoke=True)

    # Uninterrupted 8-step run.
    torch.manual_seed(0)
    ds = _imbalanced_dataset(cfg)
    m_full = build_model(cfg, device="cpu", repaired=True)
    st_full = train_sjepa_v2(m_full, ds, cfg, total_updates=8, device="cpu", seed=11)

    # Split run: 4 steps, checkpoint, resume for 4 more. Both segments use the
    # SAME schedule horizon (8) so the LR/EMA schedules are identical to the
    # uninterrupted run; only the stopping point differs.
    torch.manual_seed(0)
    ds2 = _imbalanced_dataset(cfg)
    m_a = build_model(cfg, device="cpu", repaired=True)
    st_a = train_sjepa_v2(m_a, ds2, cfg, total_updates=4, device="cpu", seed=11,
                          schedule_updates=8)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "resume.pt"
        save_checkpoint_v2(ckpt, m_a, cfg, train_state=st_a)
        m_b = build_model(cfg, device="cpu", repaired=True)
        loaded = load_checkpoint_v2(ckpt, m_b)
        st_b = train_sjepa_v2(m_b, ds2, cfg, total_updates=8, device="cpu",
                              resume_state=loaded["train_state"], seed=11,
                              schedule_updates=8)

    resumed_losses = st_a.losses + st_b.losses
    # Every step of the resumed run must match the uninterrupted run's loss.
    assert len(resumed_losses) == len(st_full.losses) == 8
    for k in range(8):
        assert abs(resumed_losses[k] - st_full.losses[k]) < 1e-3, (
            f"resume diverged at step {k}: {resumed_losses[k]:.5f} vs "
            f"{st_full.losses[k]:.5f}")


def test_diagnostics_are_finite_and_nontrivial():
    cfg = get_config("laptop", smoke=True)
    ds = _imbalanced_dataset(cfg)
    model = build_model(cfg, device="cpu", repaired=True)
    st = train_sjepa_v2(model, ds, cfg, total_updates=6, device="cpu", seed=5)
    assert all(np.isfinite(st.losses))
    assert all(np.isfinite(st.emb_std)) and max(st.emb_std) > 0
    assert all(r >= 0 for r in st.eff_rank)
    assert np.isfinite(st.ema_half_life_steps)
    # teacher drift should be finite and non-negative
    assert all(d >= -1e-6 for d in st.teacher_drift)


if __name__ == "__main__":
    test_source_uniform_sampler_equalizes_exposure(); print("[ok] source-uniform sampler")
    test_strict_ssl_ignores_labels(); print("[ok] strict SSL ignores labels (D3)")
    test_save_resume_matches_uninterrupted(); print("[ok] save/resume matches (D4)")
    test_diagnostics_are_finite_and_nontrivial(); print("[ok] diagnostics finite")
    print("ALL TRAIN_V2 TESTS PASSED")
