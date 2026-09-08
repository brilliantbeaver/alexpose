"""Fixed context-only nuisance schema, including camera and detector quality."""

import cv2
import numpy as np

VIEWS = ("front", "back", "left side", "right side", "unknown")


def view_one_hot(value):
    value = str(value).strip().lower()
    return np.array(
        [name == (value if value in VIEWS else "unknown") for name in VIEWS],
        dtype=np.float32,
    )


def context_nuisance(video, boxes, skeleton, row):
    # Slice at entry: future values cannot influence any statistic below.
    video, boxes, skeleton = video[:32], boxes[:32], skeleton[:32]
    if len(video) != 32 or skeleton.shape != (32, 33, 4):
        raise ValueError("Nuisances require the 32-frame observed context")
    h, w = video.shape[1:3]
    fps = float(row["decoded_fps"])
    duration = (row["sequence_last_frame"] - row["sequence_first_frame"] + 1) / fps
    relative = (row["source_first_frame"] - row["sequence_first_frame"]) / max(
        1, row["sequence_last_frame"] - row["sequence_first_frame"]
    )
    center, size = (boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]
    speed = np.linalg.norm(np.diff(center, axis=0) * fps, axis=1).mean()
    quality = skeleton[..., 2]
    values, names = [], []

    def add(name, value):
        array = np.asarray(value, dtype=float).reshape(-1)
        values.extend(array)
        names.extend(
            [name] if len(array) == 1 else [f"{name}_{i}" for i in range(len(array))]
        )

    add("timing_duration_relative_fps", [duration, relative, fps])
    add("source_width_height_aspect", [w, h, w / h])
    add("view", view_one_hot(row.get("cam_view")))
    add("box_first_center_size_area", np.r_[center[0], size[0], np.prod(size[0])])
    add("box_last_center_size_area", np.r_[center[-1], size[-1], np.prod(size[-1])])
    add("box_log_scale_change", np.log(size[-1] / size[0]))
    add(
        "box_velocity_mean_std",
        np.r_[
            np.diff(center, axis=0).mean(axis=0) * fps,
            np.diff(center, axis=0).std(axis=0) * fps,
        ],
    )
    add(
        "confidence_mean_quantiles",
        np.r_[quality.mean(), np.quantile(quality, [0.1, 0.5, 0.9])],
    )
    add("joint_missingness", 1 - skeleton[..., 3].mean(axis=0))
    # Fixed spatial reduction bounds cost and does not depend on pixel content.
    small = np.stack([cv2.resize(frame, (160, 96)) for frame in video])
    masks = []
    for box in boxes:
        mask = np.ones((96, 160), dtype=bool)
        x0, y0, x1, y1 = (box * [160, 96, 160, 96]).astype(int)
        mask[max(0, y0) : min(96, y1 + 1), max(0, x0) : min(160, x1 + 1)] = False
        masks.append(mask)
    pixels = small[np.stack(masks)].astype(float) / 255
    if not len(pixels):
        raise ValueError("No context background pixels")
    add("background_rgb_mean_std", np.r_[pixels.mean(axis=0), pixels.std(axis=0)])
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in small]
    flows = []
    for t in range(1, 32):
        flow = cv2.calcOpticalFlowFarneback(
            gray[t - 1], gray[t], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mask = masks[t - 1] & masks[t]
        if mask.any():
            flows.append(np.median(flow[mask], axis=0) / [160, 96] * fps)
    if not flows:
        raise ValueError("No background support for context camera motion")
    add(
        "background_flow_mean_std", np.r_[np.mean(flows, axis=0), np.std(flows, axis=0)]
    )
    matching = np.r_[
        fps,
        view_one_hot(row.get("cam_view")),
        duration,
        np.sqrt(np.prod(size, axis=1)).mean(),
        speed,
        quality.mean(),
        skeleton[..., 3].mean(),
    ]
    return np.asarray(values, dtype=np.float32), names, matching.astype(np.float32)
