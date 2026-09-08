"""Exact decoding, frozen crop geometry, and past-only whole-body histories."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gavd6_sjepa.data_foundations.gavd_pose_extraction_primitives import (
    create_landmarker,
    detect_pose_in_crop,
    safe_literal,
)

from .fi_contracts import FRAME


def decode_exact_window(video_path, first_frame, count=64):
    if first_frame < 0 or count < 1:
        raise ValueError(
            "Decode requires nonnegative zero-based index and positive count"
        )
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Non-positive or non-finite FPS")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(first_frame)):
            raise ValueError("Decoder cannot seek by source frame")
        frames = []
        for offset in range(count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Decode failed at source frame {first_frame + offset}"
                )
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if abs(position - (first_frame + offset + 1)) > 0.1:
                raise ValueError(
                    "Decoder frame position disagrees with requested source index"
                )
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames), fps
    finally:
        capture.release()


@dataclass(frozen=True)
class CropGeometry:
    source_height: int
    source_width: int
    resolution: int = 384

    @property
    def resized_hw(self):
        short = int(self.resolution * 256 / 224)
        h, w = self.source_height, self.source_width
        return (int(short * h / w), short) if w < h else (short, int(short * w / h))

    @property
    def offset_xy(self):
        h, w = self.resized_hw
        return np.array(
            [round((w - self.resolution) / 2), round((h - self.resolution) / 2)]
        )

    def pixels(self, video):
        h, w = self.resized_hw
        x, y = self.offset_xy
        return np.stack(
            [
                cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)[
                    y : y + self.resolution, x : x + self.resolution
                ]
                for frame in video
            ]
        )

    def boxes(self, normalized_boxes):
        h, w = self.resized_hw
        shifted = np.asarray(normalized_boxes) * np.array([w, h, w, h]) - np.tile(
            self.offset_xy, 2
        )
        clipped = np.clip(shifted, 0, self.resolution)
        area = np.prod(shifted[:, 2:] - shifted[:, :2], axis=1)
        retention = (
            np.prod(np.maximum(clipped[:, 2:] - clipped[:, :2], 0), axis=1) / area
        )
        return clipped / self.resolution, retention


def annotation_boxes(rows):
    boxes = []
    for row in rows:
        box, info = safe_literal(row["bbox"]), safe_literal(row["vid_info"])
        try:
            left, top, width, height = [
                float(box[k]) for k in ("left", "top", "width", "height")
            ]
            w, h = float(info["width"]), float(info["height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing annotation box/source dimensions") from error
        if min(width, height, w, h) <= 0:
            raise ValueError("Degenerate annotation box")
        normalized = np.array(
            [left / w, top / h, (left + width) / w, (top + height) / h]
        )
        if (
            not np.isfinite(normalized).all()
            or np.any(normalized < 0)
            or np.any(normalized > 1)
        ):
            raise ValueError("Annotation box outside source image")
        boxes.append(normalized)
    return np.asarray(boxes, dtype=np.float32)


def normalize_skeleton(raw, boxes, threshold=0.45):
    """Box coordinates -> per-frame mid-hip -> context median torso scale.

    Frames lacking both hips have no usable centered coordinates. Their original
    detector confidence is retained, and their validity channel is false.
    No interpolation and no future frame is ever used.
    """
    if raw.shape != (32, 33, 4) or boxes.shape != (32, 4):
        raise ValueError(
            "Skeleton normalization requires exactly the observed 32 frames"
        )
    confidence = np.nan_to_num(raw[..., 3], nan=0.0).clip(0, 1)
    valid = np.isfinite(raw[..., :2]).all(axis=-1) & (confidence >= threshold)
    hip_ok = valid[:, 23] & valid[:, 24]
    valid &= hip_ok[:, None]
    xy = (raw[..., :2] - boxes[:, None, :2]) / (boxes[:, None, 2:] - boxes[:, None, :2])
    hip = (xy[:, 23] + xy[:, 24]) / 2
    shoulder = (xy[:, 11] + xy[:, 12]) / 2
    scale_ok = hip_ok & valid[:, 11] & valid[:, 12]
    scales = np.linalg.norm(shoulder[scale_ok] - hip[scale_ok], axis=-1)
    scales = scales[np.isfinite(scales) & (scales > 1e-6)]
    if not len(scales):
        raise ValueError("No visible torso for past-only robust body scale")
    scale = float(np.median(scales))
    xy = (xy - hip[:, None]) / scale
    xy[~valid] = 0
    history = np.concatenate(
        [xy, confidence[..., None], valid[..., None]], axis=-1
    ).astype(np.float32)
    if not np.isfinite(history).all():
        raise ValueError("Non-finite normalized skeleton")
    return history, scale


def extract_history(video, rows, boxes, fps, model_path):
    if len(video) != FRAME.frames_per_clip:
        raise ValueError("Expected 64 decoded frames")
    raw = []
    # Reset tracker at every window; there is no state from another source/future.
    with create_landmarker(model_path) as landmarker:
        for t in range(FRAME.context_stop_exclusive):
            pose, _ = detect_pose_in_crop(
                cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR),
                rows[t],
                landmarker,
                round(t * 1000 / fps),
            )
            raw.append(pose)
    raw = np.stack(raw)
    history, scale = normalize_skeleton(raw, boxes[:32])
    return raw, history, scale


def alignment_sheet(video, raw_pose, boxes, first_frame, destination):
    panels = []
    for t in (0, 15, 30, 31, 32, 38, 39, 63):
        frame = video[t].copy()
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = (boxes[t] * [w, h, w, h]).astype(int)
        color = (255, 220, 0) if t in (38, 39) else (0, 255, 0)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        if t < 32:
            for x, y, _, confidence in raw_pose[t]:
                if np.isfinite([x, y]).all() and confidence >= 0.45:
                    cv2.circle(frame, (int(x * w), int(y * h)), 3, (255, 0, 255), -1)
        frame = cv2.resize(frame, (384, 240))
        label = f"local {t}; source0 {first_frame + t}; " + (
            "OBSERVED" if t < 32 else "HIDDEN"
        )
        if t in (38, 39):
            label += " TARGET"
        cv2.putText(frame, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        panels.append(frame)
    sheet = np.concatenate(
        [np.concatenate(panels[:4], axis=1), np.concatenate(panels[4:], axis=1)]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot save alignment sheet: {destination}")
