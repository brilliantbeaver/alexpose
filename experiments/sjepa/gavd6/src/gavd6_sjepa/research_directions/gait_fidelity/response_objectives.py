"""Paired residual objectives; references and pair membership are loss-only data."""
from __future__ import annotations

import torch

from ..temporal_gait.objectives import per_example_mean

VARIANTS = {"jepa_delta_v1": "paired_jepa", "jepa_endpoint_v1": "paired_jepa",
            "coordinate_delta_v1": "coordinate"}
SUPPORT_RULE = "both_endpoint_queries_and_every_reference_frame_in_both_patches_v1"
FORMULAS = {
    "jepa_delta_v1": "mean_pair(mean_token(sum_channel((e_b-e_a)^2)/(2D)));e=H(p/tau_s-stopgrad((t-center)/tau_t))",
    "jepa_endpoint_v1": "mean_pair(mean_token(sum_channel(e_a^2+e_b^2)/(2D)));e=H(p/tau_s-stopgrad((t-center)/tau_t))",
    "coordinate_delta_v1": "mean_pair(mean_token(mean_frame(sum_xy((r_b-r_a)^2)/4)));r_i=(s_i/mean_pair(s))*(pred_i-target_i)",
}


def response_support(queries, reference_valid, cfg):
    """Return [pair,patch*joint] support without changing the base-loss support."""
    b, t, j = reference_valid.shape
    if b == 0 or b % 2 or t != cfg.window_size or j != 12 or reference_valid.dtype != torch.bool:
        raise ValueError("Response supervision needs adjacent endpoint pairs and boolean body12 reference support")
    slots = t // cfg.patch_size * j
    if queries.dtype != torch.bool or queries.numel() != b * slots or queries.shape[0] != b:
        raise ValueError("Queries must identify every endpoint patch/joint")
    complete = reference_valid.reshape(b, t // cfg.patch_size, cfg.patch_size, j).all(2).flatten(1)
    return (queries.reshape(b, slots) & complete).reshape(b // 2, 2, slots).all(1)


def reduce_response_tokens(errors, support):
    loss, _, supported = per_example_mean(errors, support)
    return loss, {"support_rule": SUPPORT_RULE, "supported_pairs": int(supported.sum()),
                  "unsupported_pairs": int((~supported).sum()), "supported_tokens": int(support.sum()),
                  "tokens_per_pair": support.sum(1).detach().cpu().tolist(),
                  "loss": float(loss.detach()) if supported.any() else None}


def centered_logit_errors(predicted, teacher, center, student_temperature=.1, teacher_temperature=.06):
    """H(v)=v-mean_channels(v); never unit-normalize an endpoint difference."""
    if predicted.shape != teacher.shape or min(student_temperature, teacher_temperature) <= 0:
        raise ValueError("Matched logits and positive temperatures are required")
    residual = predicted.float() / student_temperature - (teacher.detach().float() - center.detach().float()) / teacher_temperature
    return residual - residual.mean(-1, keepdim=True)


def latent_response_loss(predicted, teacher, queries, reference_valid, cfg, *, variant,
                         center, student_temperature=.1, teacher_temperature=.06):
    if variant not in {"jepa_delta_v1", "jepa_endpoint_v1"}:
        raise ValueError("Unknown latent response variant")
    support = response_support(queries, reference_valid, cfg)
    e = centered_logit_errors(predicted, teacher, center, student_temperature, teacher_temperature)
    if e.shape[:2] != (len(reference_valid), support.shape[1]):
        raise ValueError("Latent tokens differ from the reference query grid")
    a, b = e.reshape(len(support), 2, support.shape[1], -1).unbind(1)
    errors = .5 * ((b - a).square().mean(-1) if variant == "jepa_delta_v1"
                    else a.square().mean(-1) + b.square().mean(-1))
    return reduce_response_tokens(errors, support)


def coordinate_response_loss(predicted, target, queries, reference_valid, scales, cfg):
    support = response_support(queries, reference_valid, cfg)
    if predicted.shape != target.shape or predicted.shape != (*reference_valid.shape, 2):
        raise ValueError("Coordinate responses require aligned body12 coordinates")
    scales = torch.as_tensor(scales, dtype=torch.float32, device=predicted.device)
    if scales.shape != (len(predicted),) or not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Every endpoint needs its positive observation-derived scale")
    common = scales.reshape(-1, 2).mean(1).repeat_interleave(2)
    # Mask invalid references before arithmetic, so excluded NaNs cannot poison backward.
    safe_target = torch.where(reference_valid[..., None], target.detach().float(), 0)
    residual = (predicted.float() - safe_target) * (scales / common)[:, None, None, None]
    a, b = residual.reshape(-1, 2, cfg.window_size, 12, 2).unbind(1)
    frame_errors = (b - a).square().sum(-1) / 4
    errors = frame_errors.reshape(len(support), -1, cfg.patch_size, 12).mean(2).flatten(1)
    return reduce_response_tokens(errors, support)


@torch.no_grad()
def predictive_diagnostics(predicted, teacher, valid, *, center,
                           student_temperature=.1, teacher_temperature=.06):
    """CE, entropy and KL share probabilities, support, and equal-example weights."""
    log_q = ((teacher.detach().float() - center.detach().float()) / teacher_temperature).log_softmax(-1)
    q = log_q.exp()
    log_p = (predicted.detach().float() / student_temperature).log_softmax(-1)
    entropy = -(q * log_q).sum(-1)
    ce = -(q * log_p).sum(-1)
    kl = (q * (log_q - log_p)).sum(-1)
    return {name: float(per_example_mean(value, valid)[0])
            for name, value in (("teacher_entropy", entropy), ("predictive_ce", ce), ("predictive_kl", kl))}
