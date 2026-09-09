"""Explicit numerical policies and synchronization-free motion JEPA kernels.

Parameters, EMA, centers and loss reductions stay FP32, including in BF16 mode.
No process-wide policy is left changed after a training/evaluation call.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .comparative_training import _encode_prevalidated
from .masked_learning import resolve_learning_device


def motion_numerical_policy(device, precision="fp32"):
    device = resolve_learning_device(str(device))
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Motion precision must be fp32 or bf16")
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("BF16 motion training requires a CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("This GPU does not support native BF16")
    return {"precision": precision, "parameters": "float32", "loss_reductions": "float32",
            "evaluation_precision": "float32", "float32_matmul_precision": "highest",
            "compile": False}


@contextmanager
def motion_numerical_context():
    """Make FP32 reductions/readouts independent of another notebook's TF32 toggle."""
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def prediction_resident(model, view, target, valid_patch, target_mask, indices, *, precision):
    """Prevalidated targets use fixed-size gathers instead of CUDA nonzero().

Boolean indexing has a data-dependent output size and synchronizes CUDA with
the host. Indices are computed from the same masks on CPU before optimization.
Ragged clips retain dense per-clip weighting rather than padding/trimming targets.
"""
    flat = target_mask.flatten(1)
    with torch.autocast(device_type=view.device.type, dtype=torch.bfloat16,
                        enabled=precision == "bf16"):
        context = _encode_prevalidated(model.view_encoder, view, valid_patch, hide_mask=target_mask)
        p = model.predictor
        tokens = p.encoder_to_predictor(context)
        tokens = torch.where(flat[..., None], p.mask_token.expand_as(tokens), tokens)
        position = (p.time_pos[:, None] + p.joint_pos[None]).reshape(1, -1, tokens.shape[-1])
        predicted = p.output(p.norm(p.blocks(tokens + position,
            src_key_padding_mask=~valid_patch.flatten(1))))
        with torch.no_grad():
            selected = _encode_prevalidated(model.target_encoder, target, valid_patch)
        if indices is not None:
            gather = indices[..., None].expand(-1, -1, selected.shape[-1])
            predicted = predicted.gather(1, gather)
            selected = selected.gather(1, gather)
    # Temperature sharpening, softmax and reductions need FP32 even under AMP.
    with torch.autocast(device_type=view.device.type, enabled=False):
        selected = selected.float()
        probability = F.softmax((selected - model.target_center) / 0.06, dim=-1).detach()
        loss = -(probability * F.log_softmax(predicted.float() / 0.10, dim=-1)).sum(-1)
        counts = flat.sum(1)
        loss = loss.mean() if indices is not None else ((loss * flat).sum(1) / counts).mean()
    return loss, selected, counts


@torch.no_grad()
def update_teacher_foreach(model, momentum):
    """Two multi-tensor launches replace two launches per teacher parameter."""
    targets = list(model.target_encoder.parameters())
    online = list(model.view_encoder.parameters())
    torch._foreach_mul_(targets, momentum)
    torch._foreach_add_(targets, online, alpha=1.0 - momentum)
