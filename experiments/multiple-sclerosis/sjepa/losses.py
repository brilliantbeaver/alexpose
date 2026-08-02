"""Losses for S-JEPA.

Two pieces live here:

* :class:`CenteringSharpeningCE` is the core S-JEPA objective. It does not compare
  the predicted and target latents directly. Instead it turns both into soft
  probability distributions over feature dimensions with a softmax, sharpens the
  target with a low temperature, subtracts a slowly updated running centre from
  the target, and then measures cross-entropy. The centre and the sharpening are
  what keep training stable and stop the representation from collapsing to a
  constant.

* :class:`VICRegLoss` is an addition, not part of the original S-JEPA. It works on
  the pooled embeddings and has three terms: variance (keep each dimension spread
  out), invariance (two views of the same window should match), and covariance
  (decorrelate the dimensions). We use it in the progressive fine-tune to help the
  normal, ms, and pd clusters stay apart. A class-aware variant removes the class
  mean first so each class stays compact while the variance floor keeps the class
  centres from piling on top of each other.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenteringSharpeningCE(nn.Module):
    """Latent-space cross-entropy with target centering and sharpening.

    ``loss = mean over masked tokens of  -sum_d p_target[d] * log p_pred[d]``

    where ``p_pred = softmax(pred / tau_pred)`` and
    ``p_target = softmax((target - center) / tau_target)`` with a stop-gradient on
    the target. ``center`` is an EMA of the batch mean of the target features.
    """

    def __init__(self, dim: int, center_beta: float, tau_pred: float, tau_target: float):
        super().__init__()
        self.beta = center_beta
        self.tau_pred = tau_pred
        self.tau_target = tau_target
        self.register_buffer("center", torch.zeros(1, dim))

    @torch.no_grad()
    def _update_center(self, target: torch.Tensor) -> None:
        batch_center = target.mean(dim=0, keepdim=True)     # (1, dim)
        self.center.mul_(self.beta).add_(batch_center, alpha=1.0 - self.beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``pred`` and ``target`` are (M, dim), M = masked tokens across the batch."""
        target = target.detach()
        p_pred = F.log_softmax(pred / self.tau_pred, dim=-1)
        with torch.no_grad():
            centered = (target - self.center) / self.tau_target
            p_target = F.softmax(centered, dim=-1)
        loss = -(p_target * p_pred).sum(dim=-1).mean()
        self._update_center(target)
        return loss


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularisation on pooled embeddings."""

    def __init__(self, sim: float = 25.0, var: float = 25.0, cov: float = 1.0,
                 gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.sim, self.var, self.cov, self.gamma, self.eps = sim, var, cov, gamma, eps

    def _variance(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[0] <= 1:
            return z.new_zeros(())
        # unbiased=False avoids the degrees-of-freedom edge case on tiny batches
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(self.gamma - std))

    def _covariance(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        if n <= 1:
            return z.new_zeros(())
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag.pow(2).sum()) / d

    def forward(self, z_a: torch.Tensor, z_b: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the VICReg loss.

        * ``z_a`` are embeddings of one view, shape (B, D).
        * ``z_b`` optional embeddings of a second view of the same windows, used
          for the invariance term. If ``None`` the invariance term is zero.
        * ``labels`` optional (B,) class ids. If given, the variance and
          covariance terms are computed on residuals after removing the per-class
          mean, which keeps each class compact while the variance floor pushes the
          class centres apart.
        """
        inv = z_a.new_zeros(())
        if z_b is not None:
            inv = F.mse_loss(z_a, z_b)

        z_var = z_a
        if labels is not None:
            z_var = _remove_class_means(z_a, labels)

        var = self._variance(z_var)
        cov = self._covariance(z_var)
        return self.sim * inv + self.var * var + self.cov * cov


def _remove_class_means(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Subtract each sample's class mean, so what remains is within-class spread."""
    out = z.clone()
    for c in torch.unique(labels):
        m = labels == c
        out[m] = z[m] - z[m].mean(dim=0, keepdim=True)
    return out
