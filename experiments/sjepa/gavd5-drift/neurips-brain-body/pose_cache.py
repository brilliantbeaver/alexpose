"""Validated pose-cache migration and bounded retry helpers.

The pose arrays are scientific artifacts, while fold assignments and manifest
hashes are provenance.  This module keeps those concerns separate: a legacy
cache can receive current provenance only after its identities, annotated frame
numbers, tensor shapes, and pose-model hash have all been verified.

Retry decisions are similarly explicit.  Only errors labelled transient are
retried; stale schemas, missing prerequisites, and integrity mismatches need a
different action and are never sent through a blind retry loop.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy as np


CURRENT_EXTRACTION_VERSION = "gavd5_pose_v3_split_provenance"
REFRESHABLE_EXTRACTION_VERSIONS = frozenset(
    {
        "gavd3_pose_v2_video_mode",
        "gavd4_pose_v2_video_mode",
        "gavd5_pose_v2_video_mode",
        CURRENT_EXTRACTION_VERSION,
    }
)

LEGACY_REQUIRED_KEYS = frozenset(
    {
        "sequence",
        "frame_numbers",
        "crop_bounds",
        "fps",
        "sequence_id",
        "video_id",
        "condition",
        "source_csv",
        "source_video",
        "extraction_version",
        "pose_model",
        "pose_model_sha256",
        "visibility_threshold",
    }
)
PROVENANCE_REQUIRED_KEYS = frozenset(
    {
        "eligibility_stage",
        "outer_fold",
        "split_role",
        "split_version",
        "split_seed",
        "manifest_sha256",
        "split_sha256",
        "split_roles_json",
    }
)
CURRENT_REQUIRED_KEYS = LEGACY_REQUIRED_KEYS | PROVENANCE_REQUIRED_KEYS


class PoseCacheError(ValueError):
    """Base class for actionable cache-validation failures."""


class StalePoseCacheError(PoseCacheError):
    """The cache is structurally sound but lacks current provenance."""


class PoseCacheIntegrityError(PoseCacheError):
    """The cached scientific arrays do not match their expected source row."""


class PoseExtractionError(RuntimeError):
    """A labelled extraction error used to make retry decisions auditable."""

    retryable = False
    failure_kind = "pose_extraction_error"

    def __init__(self, message: str, *, failure_kind: str | None = None) -> None:
        super().__init__(message)
        if failure_kind is not None:
            self.failure_kind = str(failure_kind)


class TransientPoseExtractionError(PoseExtractionError):
    """A decoder or inference failure that may clear after reopening state."""

    retryable = True
    failure_kind = "transient_pose_runtime"


class PermanentPoseExtractionError(PoseExtractionError):
    """A deterministic source/annotation failure that a retry cannot repair."""

    retryable = False
    failure_kind = "permanent_pose_source_error"


@dataclass(frozen=True)
class FailureDisposition:
    kind: str
    retryable: bool
    action: str


def classify_pose_error(error: BaseException) -> FailureDisposition:
    """Map an exception to a stable audit category and recovery action."""

    if isinstance(error, StalePoseCacheError):
        return FailureDisposition(
            "stale_pose_cache",
            False,
            "validate and migrate provenance, or rebuild the cache",
        )
    if isinstance(error, PoseCacheIntegrityError):
        return FailureDisposition(
            "pose_cache_integrity",
            False,
            "quarantine or rebuild; do not reuse the cached arrays",
        )
    if isinstance(error, FileNotFoundError):
        return FailureDisposition(
            "missing_source_video",
            False,
            "restore the decoded source video before extraction",
        )
    if isinstance(error, PoseExtractionError):
        return FailureDisposition(
            str(error.failure_kind),
            bool(error.retryable),
            "retry with fresh decoder/inference state"
            if error.retryable
            else "repair the source or annotation before rerunning",
        )
    if isinstance(error, (TimeoutError, ConnectionError)):
        return FailureDisposition(
            "transient_io_error",
            True,
            "retry after a short backoff",
        )
    return FailureDisposition(
        "unexpected_pose_error",
        False,
        "inspect the traceback; unexpected errors are not retried automatically",
    )


def _scalar(value: np.ndarray) -> Any:
    if np.asarray(value).shape != ():
        raise PoseCacheIntegrityError("Expected scalar cache metadata")
    return np.asarray(value).item()


def read_pose_payload(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ fully so no open handle remains during atomic replacement."""

    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.array(data[key], copy=True) for key in data.files}
    except PoseCacheError:
        raise
    except Exception as error:
        raise PoseCacheIntegrityError(
            f"Could not read pose cache {path}: {type(error).__name__}: {error}"
        ) from error


def _validate_array_contract(
    payload: Mapping[str, np.ndarray],
    *,
    path: Path,
) -> None:
    missing = LEGACY_REQUIRED_KEYS.difference(payload)
    if missing:
        raise StalePoseCacheError(
            f"Pose cache {path} is missing legacy keys {sorted(missing)}"
        )

    sequence = np.asarray(payload["sequence"])
    frame_numbers = np.asarray(payload["frame_numbers"])
    crop_bounds = np.asarray(payload["crop_bounds"])
    if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
        raise PoseCacheIntegrityError(
            f"Pose cache {path} has sequence shape {sequence.shape}; expected (T, 33, 4)"
        )
    if frame_numbers.ndim != 1 or len(frame_numbers) != len(sequence):
        raise PoseCacheIntegrityError(
            f"Pose cache {path} has {len(sequence)} poses but frame_numbers shape "
            f"{frame_numbers.shape}"
        )
    if crop_bounds.shape != (len(sequence), 4):
        raise PoseCacheIntegrityError(
            f"Pose cache {path} has crop_bounds shape {crop_bounds.shape}; "
            f"expected ({len(sequence)}, 4)"
        )
    geometry_keys = {"frame_sizes", "crop_bounds_normalized"}
    present_geometry_keys = geometry_keys.intersection(payload)
    if present_geometry_keys and present_geometry_keys != geometry_keys:
        raise PoseCacheIntegrityError(
            f"Pose cache {path} has incomplete resolution-safe geometry keys: "
            f"{sorted(present_geometry_keys)}"
        )
    if present_geometry_keys:
        frame_sizes = np.asarray(payload["frame_sizes"])
        normalized_bounds = np.asarray(payload["crop_bounds_normalized"])
        if frame_sizes.shape != (len(sequence), 2):
            raise PoseCacheIntegrityError(
                f"Pose cache {path} has frame_sizes shape {frame_sizes.shape}; "
                f"expected ({len(sequence)}, 2)"
            )
        if normalized_bounds.shape != (len(sequence), 4):
            raise PoseCacheIntegrityError(
                f"Pose cache {path} has crop_bounds_normalized shape "
                f"{normalized_bounds.shape}; expected ({len(sequence)}, 4)"
            )
        if not np.isfinite(frame_sizes).all() or np.any(frame_sizes <= 0):
            raise PoseCacheIntegrityError(
                f"Pose cache {path} has invalid decoded frame dimensions"
            )
        if not np.isfinite(normalized_bounds).all() or np.any(
            (normalized_bounds < 0.0) | (normalized_bounds > 1.0)
        ):
            raise PoseCacheIntegrityError(
                f"Pose cache {path} has invalid normalized crop bounds"
            )
        scales = frame_sizes[:, [0, 1, 0, 1]].astype(np.float64)
        reconstructed = crop_bounds.astype(np.float64) / scales
        tolerance = float(np.max(1.0 / scales)) + 1e-9
        if not np.allclose(normalized_bounds, reconstructed, atol=tolerance):
            raise PoseCacheIntegrityError(
                f"Pose cache {path} normalized and pixel crop bounds disagree"
            )
    fps = float(_scalar(payload["fps"]))
    if not np.isfinite(fps) or fps <= 0:
        raise PoseCacheIntegrityError(f"Pose cache {path} has invalid fps={fps!r}")


def _validate_identity_and_frames(
    payload: Mapping[str, np.ndarray],
    *,
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    for key in ("sequence_id", "video_id", "condition"):
        cached = str(_scalar(payload[key]))
        wanted = str(expected[key])
        if cached != wanted:
            raise PoseCacheIntegrityError(
                f"Pose cache {path} has {key}={cached!r}; expected {wanted!r}"
            )

    cached_frames = np.asarray(payload["frame_numbers"], dtype=np.int64)
    expected_frames = np.asarray(expected["frame_numbers"], dtype=np.int64)
    if not np.array_equal(cached_frames, expected_frames):
        raise PoseCacheIntegrityError(
            f"Pose cache {path} frame numbers do not exactly match the current "
            "annotation CSV"
        )

    expected_model_hash = str(expected["pose_model_sha256"])
    cached_model_hash = str(_scalar(payload["pose_model_sha256"]))
    if cached_model_hash != expected_model_hash:
        raise PoseCacheIntegrityError(
            f"Pose cache {path} used model SHA-256 {cached_model_hash}; "
            f"expected {expected_model_hash}"
        )
    if "visibility_threshold" in expected:
        cached_threshold = float(_scalar(payload["visibility_threshold"]))
        expected_threshold = float(expected["visibility_threshold"])
        if not np.isclose(cached_threshold, expected_threshold, atol=1e-7):
            raise PoseCacheIntegrityError(
                f"Pose cache {path} used visibility threshold "
                f"{cached_threshold}; expected {expected_threshold}"
            )


def validate_current_pose_cache(
    path: str | Path,
    *,
    expected: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return a current payload only after scientific and provenance checks."""

    path = Path(path)
    payload = read_pose_payload(path)
    _validate_array_contract(payload, path=path)
    _validate_identity_and_frames(payload, path=path, expected=expected)

    missing = CURRENT_REQUIRED_KEYS.difference(payload)
    if missing:
        raise StalePoseCacheError(
            f"Pose cache {path} is missing current provenance keys {sorted(missing)}"
        )
    version = str(_scalar(payload["extraction_version"]))
    if version != CURRENT_EXTRACTION_VERSION:
        raise StalePoseCacheError(
            f"Pose cache {path} uses refreshable extraction schema {version!r}"
        )
    for key in PROVENANCE_REQUIRED_KEYS:
        cached = _scalar(payload[key])
        wanted = provenance[key]
        if str(cached) != str(wanted):
            raise StalePoseCacheError(
                f"Pose cache {path} has stale {key}={cached!r}; expected {wanted!r}"
            )
    return payload


def atomic_savez_compressed(
    path: str | Path,
    *,
    validator: Callable[[Path], Any] | None = None,
    **payload: np.ndarray,
) -> None:
    """Write a complete NPZ beside its target and atomically replace the target."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        # Verify that the archive is readable before exposing it as the cache.
        with np.load(temporary, allow_pickle=False) as check:
            if set(check.files) != set(payload):
                raise PoseCacheIntegrityError(
                    f"Temporary pose archive {temporary} did not preserve all keys"
                )
        if validator is not None:
            validator(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def refresh_pose_cache_provenance(
    path: str | Path,
    *,
    expected: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    """Atomically add current provenance to a fully validated legacy cache.

    Returns the previous extraction-version label.  No pose coordinate, frame
    number, crop bound, visibility value, or FPS value is recomputed.
    """

    path = Path(path)
    payload = read_pose_payload(path)
    _validate_array_contract(payload, path=path)
    _validate_identity_and_frames(payload, path=path, expected=expected)
    old_version = str(_scalar(payload["extraction_version"]))
    if old_version not in REFRESHABLE_EXTRACTION_VERSIONS:
        raise PoseCacheIntegrityError(
            f"Pose cache {path} uses unknown extraction version {old_version!r}; "
            "automatic provenance migration is not allowed"
        )

    payload.update(
        {
            "extraction_version": np.asarray(CURRENT_EXTRACTION_VERSION),
            "cache_origin_version": np.asarray(
                str(
                    _scalar(payload["cache_origin_version"])
                    if "cache_origin_version" in payload
                    else old_version
                )
            ),
            "provenance_refreshed_at_utc": np.asarray(
                datetime.now(timezone.utc).isoformat()
            ),
        }
    )
    for key, value in provenance.items():
        payload[key] = np.asarray(value)
    if "source_csv" in expected:
        payload["source_csv"] = np.asarray(str(expected["source_csv"]))

    atomic_savez_compressed(
        path,
        validator=lambda candidate: validate_current_pose_cache(
            candidate,
            expected=expected,
            provenance=provenance,
        ),
        **payload,
    )
    validate_current_pose_cache(path, expected=expected, provenance=provenance)
    return old_version


T = TypeVar("T")


class PoseOperationFailed(RuntimeError):
    """A terminal operation failure carrying every audited attempt."""

    def __init__(
        self,
        error: Exception,
        disposition: FailureDisposition,
        attempts: Sequence[Mapping[str, Any]],
    ) -> None:
        self.original_error = error
        self.disposition = disposition
        self.attempts = [dict(item) for item in attempts]
        super().__init__(
            f"{disposition.kind} after {len(attempts)} attempt(s): "
            f"{type(error).__name__}: {error}"
        )


def run_with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int = 1,
    backoff_seconds: float = 1.0,
    classifier: Callable[[BaseException], FailureDisposition] = classify_pose_error,
    on_retry: Callable[[int, int, FailureDisposition, Exception, float], None]
    | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[T, list[dict[str, Any]]]:
    """Run an operation with bounded exponential backoff for transient errors."""

    max_retries = max(int(max_retries), 0)
    backoff_seconds = max(float(backoff_seconds), 0.0)
    max_attempts = 1 + max_retries
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            result = operation()
        except Exception as error:
            disposition = classifier(error)
            will_retry = disposition.retryable and attempt < max_attempts
            delay = backoff_seconds * (2 ** (attempt - 1)) if will_retry else 0.0
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failure_kind": disposition.kind,
                    "retryable": disposition.retryable,
                    "will_retry": will_retry,
                    "backoff_seconds": delay,
                }
            )
            if not will_retry:
                raise PoseOperationFailed(error, disposition, attempts) from error
            if on_retry is not None:
                on_retry(attempt, max_attempts, disposition, error, delay)
            if delay:
                sleeper(delay)
        else:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "succeeded",
                    "error_type": "",
                    "error": "",
                    "failure_kind": "",
                    "retryable": False,
                    "will_retry": False,
                    "backoff_seconds": 0.0,
                }
            )
            return result, attempts
    raise AssertionError("retry loop terminated without a result or exception")
