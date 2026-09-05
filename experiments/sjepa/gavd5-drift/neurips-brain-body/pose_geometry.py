"""Resolution-safe geometry for GAVD annotation boxes and pose previews."""

from __future__ import annotations

import ast
import math
from typing import Any, Mapping, Sequence

import numpy as np


class PoseGeometryError(ValueError):
    """An annotation or frame cannot define a valid crop rectangle."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError) as error:
        raise PoseGeometryError(f"Could not parse {label}: {value!r}") from error
    if not isinstance(parsed, Mapping):
        raise PoseGeometryError(f"{label} must be a mapping, got {type(parsed).__name__}")
    return parsed


def _finite_number(mapping: Mapping[str, Any], key: str, *, label: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise PoseGeometryError(f"{label} needs numeric {key!r}") from error
    if not math.isfinite(value):
        raise PoseGeometryError(f"{label}.{key} must be finite, got {value!r}")
    return value


def normalized_padded_bounds(
    annotation_row: Mapping[str, Any],
    *,
    padding: float = 0.15,
) -> np.ndarray:
    """Return clipped ``[x0,y0,x1,y1]`` in annotation-normalized space.

    GAVD boxes are expressed in the pixel coordinate system named by
    ``vid_info``. Normalizing there first makes subsequent projection invariant
    to the resolution of a newly downloaded copy of the same YouTube source.
    """

    padding = float(padding)
    if not math.isfinite(padding) or padding < 0:
        raise PoseGeometryError(f"padding must be finite and non-negative: {padding!r}")
    bbox = _mapping(annotation_row.get("bbox"), label="bbox")
    video = _mapping(annotation_row.get("vid_info"), label="vid_info")
    source_width = _finite_number(video, "width", label="vid_info")
    source_height = _finite_number(video, "height", label="vid_info")
    left = _finite_number(bbox, "left", label="bbox")
    top = _finite_number(bbox, "top", label="bbox")
    width = _finite_number(bbox, "width", label="bbox")
    height = _finite_number(bbox, "height", label="bbox")
    if source_width <= 0 or source_height <= 0:
        raise PoseGeometryError(
            f"vid_info dimensions must be positive, got {source_width}x{source_height}"
        )
    if width <= 0 or height <= 0:
        raise PoseGeometryError(
            f"bbox dimensions must be positive, got {width}x{height}"
        )

    x0 = (left - padding * width) / source_width
    y0 = (top - padding * height) / source_height
    x1 = (left + (1.0 + padding) * width) / source_width
    y1 = (top + (1.0 + padding) * height) / source_height
    bounds = np.clip(np.asarray([x0, y0, x1, y1], dtype=np.float64), 0.0, 1.0)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise PoseGeometryError(
            f"bbox lies outside the annotated frame after clipping: {bounds.tolist()}"
        )
    return bounds


def project_normalized_bounds(
    normalized_bounds: Sequence[float],
    frame_shape: Sequence[int],
) -> tuple[int, int, int, int]:
    """Project normalized bounds to a decoded frame using outer rounding."""

    if len(frame_shape) < 2:
        raise PoseGeometryError(f"frame_shape needs height and width: {frame_shape!r}")
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    if frame_width <= 0 or frame_height <= 0:
        raise PoseGeometryError(
            f"decoded frame dimensions must be positive, got {frame_width}x{frame_height}"
        )
    bounds = np.asarray(normalized_bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise PoseGeometryError(f"normalized bounds must be four finite values: {bounds!r}")
    if np.any(bounds < 0.0) or np.any(bounds > 1.0):
        raise PoseGeometryError(f"normalized bounds must lie in [0,1]: {bounds.tolist()}")

    x0 = max(0, int(np.floor(bounds[0] * frame_width)))
    y0 = max(0, int(np.floor(bounds[1] * frame_height)))
    x1 = min(frame_width, int(np.ceil(bounds[2] * frame_width)))
    y1 = min(frame_height, int(np.ceil(bounds[3] * frame_height)))
    if x1 <= x0 or y1 <= y0:
        raise PoseGeometryError(
            f"projected crop is empty at {frame_width}x{frame_height}: "
            f"{[x0, y0, x1, y1]}"
        )
    return x0, y0, x1, y1


def scaled_crop_bounds(
    annotation_row: Mapping[str, Any],
    frame_shape: Sequence[int],
    *,
    padding: float = 0.15,
) -> tuple[int, int, int, int]:
    """Project one GAVD bbox into the actual decoded frame resolution."""

    return project_normalized_bounds(
        normalized_padded_bounds(annotation_row, padding=padding),
        frame_shape,
    )


def normalize_pixel_bounds(
    pixel_bounds: Sequence[int],
    frame_size: Sequence[int],
) -> np.ndarray:
    """Convert pixel bounds to normalized space for resolution-safe caching.

    ``frame_size`` is explicitly ``(width, height)``.
    """

    if len(frame_size) != 2:
        raise PoseGeometryError(f"frame_size must be (width,height): {frame_size!r}")
    width, height = float(frame_size[0]), float(frame_size[1])
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise PoseGeometryError(f"invalid frame_size {frame_size!r}")
    bounds = np.asarray(pixel_bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise PoseGeometryError(f"pixel bounds must be four finite values: {bounds!r}")
    normalized = bounds / np.asarray([width, height, width, height], dtype=np.float64)
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise PoseGeometryError(
            f"pixel bounds {bounds.tolist()} exceed frame size {width}x{height}"
        )
    # Keep float64: down-casting ratios such as 349/360 can move a value just
    # above an integer and make a later ``ceil`` add a spurious pixel.
    return normalized


def bounds_iou(left: Sequence[int], right: Sequence[int]) -> float:
    """Intersection-over-union for two ``[x0,y0,x1,y1]`` rectangles."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,):
        raise PoseGeometryError("IoU operands must each contain four coordinates")
    intersection_width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0
