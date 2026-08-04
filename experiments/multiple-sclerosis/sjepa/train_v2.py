"""Repaired S-JEPA training (Phase 2): label-free, source-uniform, resumable.

This is the corrected training loop used by R1. It differs from ``train.py`` (kept
for the E0 reference) in the ways the audit and plan require:

* **No diagnosis labels.** The strict self-supervised objective never sees the
  class label. The class-aware VICReg path is gone from this loop; any label use
  belongs in a separate, clearly named supervised-adaptation stage.
* **Source-uniform sampling.** Windows are sampled so every *source video* is
  equally likely, regardless of how many windows it produced, so a long clip
  cannot dominate optimization.
* **Schedules in optimizer updates, not epochs.** ``total_updates`` drives the
  LR cosine, the warmup, and the EMA momentum ramp, so the teacher actually moves
  for the real budget. EMA half-life is reported.
* **Full, resumable training state.** The running center, optimizer, global step,
  EMA schedule position, RNG, and sampler seed are all saved and restored, so an
  interrupted-then-resumed run reproduces an uninterrupted one.
* **Repaired mechanics.** Uses ``model.forward_repaired`` (PredictorV2 target
  positions) and per-example stochastic masks from ``masking_v2``.
* **Collapse diagnostics.** Per-dimension embedding std, effective rank, and
  teacher/student drift are logged so a collapsed representation is visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from .config import SJEPAConfig
from .losses import CenteringSharpeningCE
from .masking_v2 import sample_mask_batch
from .models import SJEPA, ema_tau_schedule
from .augment import random_view


@dataclass
class TrainV2State:
    losses: List[float] = field(default_factory=list)
    emb_std: List[float] = field(default_factory=list)
    eff_rank: List[float] = field(default_factory=list)
    teacher_drift: List[float] = field(default_factory=list)
    source_exposure: Dict[str, int] = field(default_factory=dict)
    total_updates: int = 0
    ema_half_life_steps: float = 0.0


def _cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def source_uniform_sampler(dataset, seed: int) -> WeightedRandomSampler:
    """Weight each window by 1/(#windows in its source) so sources are uniform."""
    counts: Dict[str, int] = {}
    for s in dataset.source_ids:
        counts[s] = counts.get(s, 0) + 1
    weights = torch.tensor([1.0 / counts[s] for s in dataset.source_ids], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                 generator=g)


def _source_uniform_weights(dataset) -> np.ndarray:
    counts: Dict[str, int] = {}
    for s in dataset.source_ids:
        counts[s] = counts.get(s, 0) + 1
    w = np.array([1.0 / counts[s] for s in dataset.source_ids], dtype=np.float64)
    return w / w.sum()


def _draw_batch_indices(weights: np.ndarray, batch_size: int, gen: np.random.Generator):
    """One source-uniform batch of window indices (with replacement)."""
    return gen.choice(len(weights), size=batch_size, replace=True, p=weights)


def _effective_rank(z: torch.Tensor) -> float:
    """Effective rank (entropy of normalized singular values) of a (B, D) batch.

    Computed on CPU because ``torch.linalg.svdvals`` is not implemented on MPS;
    doing the SVD on CPU keeps this collapse diagnostic honest on every device
    instead of silently returning zero.
    """
    if z.shape[0] < 2:
        return 0.0
    zc = (z - z.mean(dim=0, keepdim=True)).detach().to("cpu", dtype=torch.float32)
    try:
        s = torch.linalg.svdvals(zc)
    except Exception:
        return float("nan")
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


def _ema_half_life(start: float, end: float, total: int) -> float:
    """Approximate EMA half-life in steps at the mid-schedule momentum."""
    tau_mid = 0.5 * (start + end)
    if tau_mid >= 1.0:
        return float("inf")
    return math.log(0.5) / math.log(tau_mid)


def train_sjepa_v2(
    model: SJEPA,
    dataset,
    cfg: SJEPAConfig,
    total_updates: int = 1000,
    device: Optional[str] = None,
    mask_ratio: float = 0.6,
    masks_per_sequence: int = 1,
    clinical_bias: float = 1.5,
    log_every: int = 0,
    resume_state: Optional[Dict] = None,
    seed: Optional[int] = None,
    schedule_updates: Optional[int] = None,
) -> TrainV2State:
    """Label-free repaired S-JEPA training for a fixed number of optimizer updates.

    ``dataset`` yields ``(window (T,V,C), label)`` but the label is IGNORED here.
    Pass ``resume_state`` (from :func:`export_train_state`) to continue a run.

    ``schedule_updates`` sets the horizon the LR/EMA/warmup schedules are computed
    against; it defaults to ``total_updates``. Keeping it fixed while stopping
    early at a smaller ``total_updates`` lets a save/resume run reproduce an
    uninterrupted one exactly (the schedules must not change between segments).
    """
    # Resume can carry the full schedule spec so the caller need not re-pass it.
    if resume_state is not None and resume_state.get("schedule"):
        sch = resume_state["schedule"]
        if schedule_updates is None:
            schedule_updates = sch.get("schedule_updates")
        seed = sch.get("seed", seed)
        mask_ratio = sch.get("mask_ratio", mask_ratio)
        masks_per_sequence = sch.get("masks_per_sequence", masks_per_sequence)
        clinical_bias = sch.get("clinical_bias", clinical_bias)
    horizon = schedule_updates if schedule_updates is not None else total_updates
    assert model.repaired, "train_sjepa_v2 requires a model built with repaired=True"
    device = device or next(model.parameters()).device
    seed = cfg.seed if seed is None else seed

    ce = CenteringSharpeningCE(cfg.encoder_dim, cfg.center_beta,
                               cfg.tau_pred, cfg.tau_target).to(device)
    params = (list(model.view_encoder.parameters())
              + list(model.predictor.parameters()) + [model.mask_token])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))

    start_step = 0
    if resume_state is not None:
        ce.load_state_dict(resume_state["ce"])
        opt.load_state_dict(resume_state["opt"])
        start_step = int(resume_state["step"])
        torch.set_rng_state(resume_state["torch_rng"])
        _restore_device_rng(resume_state.get("device_rng"))
        mask_rng = np.random.default_rng()
        mask_rng.bit_generator.state = resume_state["mask_rng"]
        data_rng = np.random.default_rng()
        data_rng.bit_generator.state = resume_state["data_rng"]
    else:
        torch.manual_seed(seed)
        mask_rng = np.random.default_rng(seed)
        data_rng = np.random.default_rng(seed + 10_000)

    # Deterministic, resumable source-uniform data stream (replaces DataLoader so
    # the batch sequence is a pure function of resumable RNG state -> save/resume
    # reproduces an uninterrupted run exactly).
    weights = _source_uniform_weights(dataset)

    def get_batch():
        idx = _draw_batch_indices(weights, cfg.batch_size, data_rng)
        xs = np.stack([dataset[int(i)][0].numpy() for i in idx], axis=0)
        return torch.from_numpy(xs), idx

    state = TrainV2State()
    # True source exposure: accumulated from the windows actually DRAWN each step
    # (AR-5 P2), not from stored window counts, so it reflects the update budget,
    # seed, and replacement-based source-uniform sampling and can reproduce the stream.
    exposure: Dict[str, int] = {}
    state.ema_half_life_steps = _ema_half_life(cfg.ema_start, cfg.ema_end, horizon)
    warmup = max(1, int(0.1 * horizon))
    model.train()

    step = start_step
    while step < total_updates:
        x_np, drawn_idx = get_batch()
        x = x_np.to(device).float()                   # (B, T, V, C)
        B = x.shape[0]
        for i in drawn_idx:
            s = dataset.source_ids[int(i)]
            exposure[s] = exposure.get(s, 0) + 1

        lr = _cosine_lr(step, horizon, cfg.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        # Per-example masks (D7). One target mask row per sequence.
        tgt_np = sample_mask_batch(B, cfg.num_joints, cfg.num_time_tokens, mask_rng,
                                   target_ratio=mask_ratio,
                                   masks_per_sequence=masks_per_sequence,
                                   clinical_bias=clinical_bias)
        tgt = torch.from_numpy(tgt_np).to(device)
        if masks_per_sequence > 1:
            x = x.repeat_interleave(masks_per_sequence, dim=0)
        ctx = ~tgt

        x_view = random_view(x)
        predicted, target = model.forward_repaired(x_view, x, ctx)

        # Per-example CE over each row's own target tokens. The centering EMA must
        # move ONCE per optimizer batch, not once per example, so we score every
        # example against the same center snapshot (update_center=False) and then
        # update the center a single time from all masked target tokens in the batch
        # (AR-5 P1: per-example updates applied the beta EMA B times per step).
        losses = []
        batch_targets = []
        for b in range(predicted.shape[0]):
            idx = torch.nonzero(tgt[b], as_tuple=False).squeeze(1)
            losses.append(ce(predicted[b, idx, :], target[b, idx, :], update_center=False))
            batch_targets.append(target[b, idx, :])
        loss = torch.stack(losses).mean()
        ce.update_center_from(torch.cat(batch_targets, dim=0))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        tau = ema_tau_schedule(step, horizon, cfg.ema_start, cfg.ema_end)
        # teacher drift = 1 - cosine(view, target) averaged over params, pre-update
        with torch.no_grad():
            drift = _teacher_student_drift(model)
        model.update_target(tau)

        # diagnostics on the target embedding
        with torch.no_grad():
            z = model.embed(x)                        # (B', dim) mean over all tokens
            state.emb_std.append(float(z.std(dim=0).mean()))
            state.eff_rank.append(_effective_rank(z))
            state.teacher_drift.append(drift)
        state.losses.append(float(loss.detach().cpu()))
        if log_every and step % log_every == 0:
            print(f"step {step:5d} loss {state.losses[-1]:.4f} lr {lr:.2e} tau {tau:.5f} "
                  f"emb_std {state.emb_std[-1]:.4f} eff_rank {state.eff_rank[-1]:.1f}")
        step += 1

    state.total_updates = step
    state.source_exposure = exposure
    schedule = {
        "schedule_updates": horizon,
        "seed": seed,
        "mask_ratio": mask_ratio,
        "masks_per_sequence": masks_per_sequence,
        "clinical_bias": clinical_bias,
    }
    state._final = export_train_state(  # type: ignore[attr-defined]
        model, ce, opt, step, mask_rng, data_rng, device, schedule=schedule)
    return state


def _teacher_student_drift(model: SJEPA) -> float:
    """Mean (1 - cosine) between view-encoder and target-encoder parameters."""
    sims = []
    for tp, sp in zip(model.target_encoder.parameters(), model.view_encoder.parameters()):
        a, b = tp.flatten(), sp.flatten()
        denom = (a.norm() * b.norm()).clamp_min(1e-9)
        sims.append(1.0 - float((a @ b) / denom))
    return float(np.mean(sims)) if sims else 0.0


def _device_rng_state(device) -> Optional[dict]:
    """Capture the accelerator RNG state so resume is exact off-CPU too.

    ``random_view`` draws on the batch's device, so a CPU-only RNG snapshot is
    insufficient for an MPS/CUDA run (Codex AR-1 P1). We snapshot whichever
    accelerator generator is active.
    """
    dev = str(device)
    try:
        if dev.startswith("mps") and hasattr(torch, "mps"):
            return {"kind": "mps", "state": torch.mps.get_rng_state()}
        if dev.startswith("cuda") and torch.cuda.is_available():
            return {"kind": "cuda", "state": torch.cuda.get_rng_state()}
    except Exception:
        return None
    return None


def _restore_device_rng(dev_state: Optional[dict]) -> None:
    if not dev_state:
        return
    try:
        if dev_state["kind"] == "mps" and hasattr(torch, "mps"):
            torch.mps.set_rng_state(dev_state["state"])
        elif dev_state["kind"] == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(dev_state["state"])
    except Exception:
        pass


def export_train_state(model, ce, opt, step, mask_rng, data_rng, device,
                       schedule: Optional[Dict] = None) -> Dict:
    return {
        "ce": ce.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "device_rng": _device_rng_state(device),
        "mask_rng": mask_rng.bit_generator.state,
        "data_rng": data_rng.bit_generator.state,
        # Full schedule/spec so resume needs no external arguments (Codex AR-1 P1).
        "schedule": schedule or {},
    }


def save_checkpoint_v2(path, model: SJEPA, cfg: SJEPAConfig,
                       train_state: Optional[TrainV2State] = None,
                       extra: Optional[Dict] = None) -> None:
    """Save the repaired model AND the full resumable training state (D4 fix)."""
    payload = {
        "view_encoder": model.view_encoder.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "mask_token": model.mask_token.detach().cpu(),
        "config": cfg.to_dict(),
        "repaired": True,
        "extra": extra or {},
    }
    if train_state is not None and getattr(train_state, "_final", None) is not None:
        payload["train_state"] = train_state._final  # type: ignore[attr-defined]
    torch.save(payload, path)


def load_checkpoint_v2(path, model: SJEPA, map_location: Optional[str] = None) -> Dict:
    """Load the repaired model; returns the checkpoint dict (incl. train_state)."""
    ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model.view_encoder.load_state_dict(ckpt["view_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.predictor.load_state_dict(ckpt["predictor"])
    with torch.no_grad():
        model.mask_token.copy_(ckpt["mask_token"].to(model.mask_token.device))
    return ckpt
