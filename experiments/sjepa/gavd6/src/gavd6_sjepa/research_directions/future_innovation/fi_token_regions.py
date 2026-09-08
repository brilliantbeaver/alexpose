"""Temporal-major region indices and a data-independent target projection."""

import numpy as np

from .fi_contracts import FRAME


def token_index(tubelet, row, col, grid=24):
    return tubelet * grid * grid + row * grid + col


def context_indices(frame=FRAME):
    return np.arange(frame.context_stop_exclusive // frame.tubelet_size * frame.grid**2)


def union_box(boxes):
    boxes = np.asarray(boxes)
    return np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]


def region_masks(box, grid=24, dilation=1):
    box = np.asarray(box)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Invalid person box")
    centers = (np.arange(grid) + 0.5) / grid
    cols = np.flatnonzero((centers >= box[0]) & (centers <= box[2]))
    rows = np.flatnonzero((centers >= box[1]) & (centers <= box[3]))
    if not len(cols) or not len(rows):
        raise ValueError("Empty person patch region")

    def rectangle(extra):
        mask = np.zeros((grid, grid), dtype=bool)
        mask[
            max(0, rows[0] - extra) : min(grid, rows[-1] + extra + 1),
            max(0, cols[0] - extra) : min(grid, cols[-1] + extra + 1),
        ] = True
        return mask.reshape(-1)

    person = rectangle(dilation)
    background = ~rectangle(dilation + 1)
    if not background.any():
        raise ValueError("Empty background region beyond person guard band")
    return person, background


def pool_context(tokens, boxes, frame=FRAME):
    time = frame.context_stop_exclusive // frame.tubelet_size
    tokens = np.asarray(tokens).reshape(time, frame.grid**2, -1)
    background = []
    for t in range(time):
        person_mask, background_mask = region_masks(
            union_box(boxes[2 * t : 2 * t + 2]), frame.grid
        )
        background.append(tokens[t, background_mask])
    return np.concatenate(
        (
            tokens.mean(axis=(0, 1)),
            tokens[-1, person_mask].mean(axis=0),
            np.concatenate(background).mean(axis=0),
        )
    ).astype(np.float32)


def pool_target(tokens, boxes, frame=FRAME):
    t = frame.target_tubelet_start // frame.tubelet_size
    tokens = np.asarray(tokens).reshape(
        frame.frames_per_clip // frame.tubelet_size, frame.grid**2, -1
    )
    person, background = region_masks(
        union_box(
            boxes[frame.target_tubelet_start : frame.target_tubelet_stop_exclusive]
        ),
        frame.grid,
    )
    return tokens[t, person].mean(axis=0), tokens[t, background].mean(axis=0)


def fixed_projection(input_dim, output_dim=256, seed=260905):
    if output_dim > input_dim or output_dim < 1:
        raise ValueError(
            "Projection dimension must be positive and no larger than the input"
        )
    matrix = np.random.default_rng(seed).standard_normal((input_dim, output_dim))
    q, r = np.linalg.qr(matrix)
    # Canonical QR signs remove a LAPACK-dependent ambiguity.
    q *= np.where(np.diag(r) < 0, -1, 1)
    return (q * np.sqrt(input_dim / output_dim)).astype(np.float32)
