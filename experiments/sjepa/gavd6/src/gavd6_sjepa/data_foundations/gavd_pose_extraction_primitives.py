"""Shared annotation geometry and MediaPipe VIDEO-mode extraction primitives.

Extracted from the historical GAVD augmentation pipeline without changing its
crop or landmark convention. Imports are lazy so manifest tools need no model.
"""

from __future__ import annotations

import ast

import cv2
import numpy as np
import pandas as pd


def safe_literal(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def create_landmarker(model_path, visibility_threshold=0.45):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=visibility_threshold,
        min_pose_presence_confidence=visibility_threshold,
        min_tracking_confidence=visibility_threshold,
    )
    return vision.PoseLandmarker.create_from_options(options)


def scaled_crop_bounds(annotation_row, frame_shape, padding=0.15):
    """Verbatim nb02 logic: scale bbox by vid_info, pad, clamp."""
    frame_height, frame_width = frame_shape[:2]
    bbox = safe_literal(annotation_row.get("bbox"))
    source = safe_literal(annotation_row.get("vid_info"))
    source_width = float(source.get("width") or frame_width)
    source_height = float(source.get("height") or frame_height)
    scale_x = frame_width / source_width
    scale_y = frame_height / source_height
    left = float(bbox.get("left", 0.0)) * scale_x
    top = float(bbox.get("top", 0.0)) * scale_y
    width = float(bbox.get("width", source_width)) * scale_x
    height = float(bbox.get("height", source_height)) * scale_y
    left -= padding * width
    top -= padding * height
    width *= 1.0 + 2.0 * padding
    height *= 1.0 + 2.0 * padding
    x0 = max(0, int(np.floor(left)))
    y0 = max(0, int(np.floor(top)))
    x1 = min(frame_width, int(np.ceil(left + width)))
    y1 = min(frame_height, int(np.ceil(top + height)))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, frame_width, frame_height
    return x0, y0, x1, y1


def detect_pose_in_crop(frame_bgr, annotation_row, landmarker, timestamp_ms):
    """Verbatim nb02: crop, detect, write full-frame-normalized (33,4) row."""
    import mediapipe as mp

    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = scaled_crop_bounds(annotation_row, frame_bgr.shape)
    crop = frame_bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
    )
    sequence_row = np.full((33, 4), np.nan, dtype=np.float32)
    sequence_row[:, 3] = 0.0
    if not result.pose_landmarks:
        return sequence_row, (x0, y0, x1, y1)
    crop_height, crop_width = crop.shape[:2]
    for index, landmark in enumerate(result.pose_landmarks[0]):
        sequence_row[index, 0] = (x0 + landmark.x * crop_width) / width
        sequence_row[index, 1] = (y0 + landmark.y * crop_height) / height
        sequence_row[index, 2] = landmark.z * crop_width / width
        sequence_row[index, 3] = float(landmark.visibility or 0.0)
    return sequence_row, (x0, y0, x1, y1)
