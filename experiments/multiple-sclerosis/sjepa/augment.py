"""Geometric view augmentation for S-JEPA.

The paper rotates a true 3D skeleton around the body's vertical axis to make
diverse "views" for the view encoder. Our skeletons come from a single monocular
camera, so we only have reliable 2D pixel coordinates (MediaPipe also gives a
weak z, which we keep as a soft channel but do not treat as real depth). We
therefore approximate the paper's 3D rotation with 2D transforms that do not
change what the walk means:

* a small in-plane rotation about the pelvis,
* a horizontal flip (left and right legs swap, which is a valid gait view),
* a small translation and scale jitter.

These operate on the already normalised windows, shape ``(B, T, V, C)``. Only the
x and y channels change; the visibility channel is passed through untouched.
"""

from __future__ import annotations

import torch

_LEFT_HIP, _RIGHT_HIP = 23, 24

# BlazePose-33 left/right pairs, used to relabel joints on a horizontal flip.
_FLIP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _flip_index(num_joints: int) -> torch.Tensor:
    idx = list(range(num_joints))
    for a, b in _FLIP_PAIRS:
        if a < num_joints and b < num_joints:
            idx[a], idx[b] = b, a
    return torch.tensor(idx, dtype=torch.long)


def random_view(
    x: torch.Tensor,
    max_rot_deg: float = 15.0,
    max_translate: float = 0.1,
    scale_jitter: float = 0.1,
    flip_prob: float = 0.5,
) -> torch.Tensor:
    """Return a randomly transformed copy of a batch of windows.

    ``x`` is ``(B, T, V, C)`` normalised coordinates. The same random transform is
    applied to every frame of a given window so the motion stays coherent, but
    each window in the batch gets its own transform.
    """
    B, T, V, C = x.shape
    device = x.device
    out = x.clone()
    xy = out[..., :2]                                   # (B, T, V, 2)

    # Per-window rotation angle.
    ang = (torch.rand(B, device=device) * 2 - 1) * (max_rot_deg * torch.pi / 180.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    rot = torch.stack([torch.stack([cos, -sin], -1),
                       torch.stack([sin, cos], -1)], -2)  # (B, 2, 2)

    # Rotate about the pelvis of each frame.
    pelvis = (xy[:, :, _LEFT_HIP] + xy[:, :, _RIGHT_HIP]) / 2.0   # (B, T, 2)
    centered = xy - pelvis[:, :, None, :]
    rotated = torch.einsum("bij,btvj->btvi", rot, centered)
    xy = rotated + pelvis[:, :, None, :]

    # Per-window scale jitter.
    scale = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * scale_jitter
    xy = xy * scale

    # Per-window translation.
    trans = (torch.rand(B, 1, 1, 2, device=device) * 2 - 1) * max_translate
    xy = xy + trans

    out[..., :2] = xy

    # Horizontal flip for a random subset of windows.
    flip_mask = torch.rand(B, device=device) < flip_prob
    if flip_mask.any():
        flip_idx = _flip_index(V).to(device)
        flipped = out[flip_mask]
        flipped[..., 0] = -flipped[..., 0]              # mirror x
        flipped = flipped[:, :, flip_idx, :]            # relabel left/right joints
        out[flip_mask] = flipped

    return out
