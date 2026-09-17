"""Explicit deterministic oscillator CONTRACT fixtures, never replacement GAVD."""
import numpy as np
from .video import TimedPose
from .windows import prepare_bout, concatenate


def fixture_bout(source=0, bout_index=0, role="train"):
    # 100Hz makes exact 250ms original-time endpoints testable independently of 25Hz grid.
    times = np.arange(600) / 100.
    coords = np.zeros((len(times), 33, 2))
    coords[..., 0] = 320
    coords[..., 1] = np.linspace(100, 400, 33)
    for left, right in ((11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)):
        coords[:, left, 0] -= 30
        coords[:, right, 0] += 30
    for j in range(25, 33):
        phase = j % 2 * np.pi + .37 * source + .12 * bout_index
        coords[:, j, 0] += (12 + source) * np.sin(2 * np.pi * .9 * times + phase)
        coords[:, j, 1] += 7 * np.cos(2 * np.pi * .9 * times + phase)
    coords[..., 0] += times[:, None] * (1 + source * .1)
    observed = np.ones((len(times), 33), bool)
    observed[100:110, 30] = False
    pose = TimedPose(times, coords, observed, np.ones_like(observed, float), np.arange(len(times)), 640, 480)
    bout = dict(sequence_id=f"SYNTHETIC-bout-{source}-{bout_index}", video_id=f"SYNTHETIC-video-{source}",
                group_id=f"SYNTHETIC-group-{source}", role=role, start_pts=0., end_pts_exclusive=6.)
    return pose, bout


def dataset(cfg, role):
    if cfg.mode != "synthetic":
        raise ValueError("Synthetic fixture requires explicit mode=synthetic; never a real-data fallback")
    source_ids = {"train": range(4), "development": range(4, 7), "test": range(7, 10), "calibration": range(10, 12)}[role]
    datasets, reports = [], []
    for source in source_ids:
        for b in range(2):
            pose, bout = fixture_bout(source, b, role)
            ds, report = prepare_bout(pose, cfg, bout)
            datasets.append(ds)
            reports.append(report)
    return concatenate(datasets), reports
