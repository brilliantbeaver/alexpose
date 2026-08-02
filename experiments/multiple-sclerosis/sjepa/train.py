"""Training loops for S-JEPA pretraining and progressive fine-tuning.

Both loops share the same two-lane step. The only difference is that fine-tuning
mixes in the ms and pd windows and can add the VICReg term. Everything is written
to run happily on a laptop CPU or MPS device as well as a GPU; the size comes from
the config profile, not from this code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SJEPAConfig
from .losses import CenteringSharpeningCE, VICRegLoss
from .masking import AnatomicalMaskSampler
from .models import SJEPA, ema_tau_schedule
from .augment import random_view


@dataclass
class TrainState:
    losses: List[float] = field(default_factory=list)
    ce_losses: List[float] = field(default_factory=list)
    vic_losses: List[float] = field(default_factory=list)


def _cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_sjepa(
    model: SJEPA,
    dataset,
    cfg: SJEPAConfig,
    epochs: Optional[int] = None,
    use_vicreg: bool = False,
    class_aware_vicreg: bool = False,
    device: Optional[str] = None,
    log_every: int = 0,
) -> TrainState:
    """Run the two-lane S-JEPA objective over ``dataset`` windows.

    ``dataset`` yields ``(window_tensor (T,V,C), label_int)``. Labels are only used
    when ``class_aware_vicreg`` is on.
    """
    device = device or next(model.parameters()).device
    epochs = epochs if epochs is not None else cfg.pretrain_epochs

    sampler = AnatomicalMaskSampler(cfg.num_joints, cfg.num_time_tokens)
    context_mask = torch.from_numpy(sampler.context_mask).to(device)
    target_mask = torch.from_numpy(sampler.target_mask).to(device)
    target_idx = torch.nonzero(target_mask, as_tuple=False).squeeze(1)

    ce = CenteringSharpeningCE(cfg.encoder_dim, cfg.center_beta,
                               cfg.tau_pred, cfg.tau_target).to(device)
    vic = VICRegLoss(cfg.vicreg_sim, cfg.vicreg_var, cfg.vicreg_cov, cfg.vicreg_gamma).to(device)

    params = list(model.view_encoder.parameters()) + list(model.predictor.parameters()) + [model.mask_token]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    total_steps = max(1, epochs * len(loader))
    state = TrainState()
    step = 0
    model.train()

    for epoch in range(epochs):
        for x, y in loader:
            x = x.to(device).float()                    # (B, T, V, C)
            y = y.to(device)

            lr = _cosine_lr(step, total_steps, cfg.lr, cfg.warmup_epochs * len(loader))
            for g in opt.param_groups:
                g["lr"] = lr

            x_view = random_view(x)                     # transformed view
            predicted, target = model(x_view, x, context_mask)

            # Slice both lanes to the masked (target) tokens for the CE loss.
            B = x.shape[0]
            pred_m = predicted[:, target_idx, :].reshape(-1, cfg.encoder_dim)
            targ_m = target[:, target_idx, :].reshape(-1, cfg.encoder_dim)
            ce_loss = ce(pred_m, targ_m)

            vic_loss = x.new_zeros(())
            if use_vicreg:
                # Pool target-encoder embeddings over masked tokens for VICReg.
                with torch.no_grad():
                    z_full = target[:, target_idx, :].mean(dim=1)   # (B, dim)
                # A second view for the invariance term.
                z_view = model.predictor(
                    model.view_encoder.forward_context(random_view(x), context_mask, model.mask_token)
                )[:, target_idx, :].mean(dim=1)
                labels = y if class_aware_vicreg else None
                vic_loss = vic(z_view, z_full, labels=labels)

            loss = ce_loss + (cfg.vicreg_weight * vic_loss if use_vicreg else 0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients: small models on small batches can otherwise take a
            # single huge step and diverge to NaN.
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()

            tau = ema_tau_schedule(step, total_steps, cfg.ema_start, cfg.ema_end)
            model.update_target(tau)

            state.losses.append(float(loss.detach().cpu()))
            state.ce_losses.append(float(ce_loss.detach().cpu()))
            state.vic_losses.append(float(vic_loss.detach().cpu()) if use_vicreg else 0.0)
            if log_every and step % log_every == 0:
                print(f"epoch {epoch} step {step} loss {state.losses[-1]:.4f} "
                      f"(ce {state.ce_losses[-1]:.4f}) lr {lr:.2e} tau {tau:.5f}")
            step += 1

    return state


def save_checkpoint(path, model: SJEPA, cfg: SJEPAConfig, extra: Optional[Dict] = None) -> None:
    torch.save(
        {
            "view_encoder": model.view_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "mask_token": model.mask_token.detach().cpu(),
            "config": cfg.to_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path, model: SJEPA, map_location: Optional[str] = None) -> Dict:
    ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    model.view_encoder.load_state_dict(ckpt["view_encoder"])
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.predictor.load_state_dict(ckpt["predictor"])
    with torch.no_grad():
        model.mask_token.copy_(ckpt["mask_token"].to(model.mask_token.device))
    return ckpt
