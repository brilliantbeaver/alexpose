#!/usr/bin/env python3
"""Build validity/confidence-aware, source-video-split GAVD Core11 tensors."""

from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from gavd6_sjepa.amass_core11_jepa import CHANNEL_NAMES, JOINT_NAMES
from gavd6_sjepa.gavd_core11_probe import AdapterConfig, adapt_mediapipe_sequence


REQUIRED_COLUMNS = {
    "pose_path",
    "sequence_id",
    "video_id",
    "split",
    "fps",
    "aspect_ratio",
}


def _safe_output(root: Path, sequence_id: str) -> tuple[Path, str]:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in sequence_id)
    if not safe:
        raise ValueError(f"Unsafe empty sequence ID after normalization: {sequence_id!r}")
    relative = PurePosixPath(f"{safe}_core11.npz")
    path = (root / relative.name).resolve()
    path.relative_to(root.resolve())
    return path, relative.as_posix()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _adapt_row(row: dict, pose_root: Path, output_root: Path, config: AdapterConfig) -> dict:
    relative = PurePosixPath(str(row["pose_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe pose_path: {row['pose_path']!r}")
    pose_path = pose_root.joinpath(*relative.parts).resolve()
    pose_path.relative_to(pose_root)
    with np.load(pose_path, allow_pickle=False) as archive:
        if "sequence" not in archive.files:
            raise ValueError(f"{pose_path} has no sequence array")
        sequence = np.asarray(archive["sequence"], dtype=np.float32)
    adapted = adapt_mediapipe_sequence(
        sequence,
        source_fps=float(row["fps"]),
        aspect_ratio=float(row["aspect_ratio"]),
        config=config,
    )
    output_path, tensor_relative_path = _safe_output(
        output_root, str(row["sequence_id"])
    )
    _atomic_npz(
        output_path,
        coordinates=adapted["coordinates"],
        valid=adapted["valid"],
        detector_confidence=adapted["detector_confidence"],
        joint_names=np.asarray(JOINT_NAMES),
        channel_names=np.asarray(CHANNEL_NAMES),
        frame_method=np.asarray(adapted["frame_method"]),
        odd_sign_anchored=np.asarray(False),
    )
    return {
        "tensor_relative_path": tensor_relative_path,
        "identity": str(row["video_id"]),
        "video_id": str(row["video_id"]),
        "sequence_id": str(row["sequence_id"]),
        "condition": str(row.get("condition", "")),
        "split": str(row["split"]),
        "canonical_fps": 30.0,
        "canonical_frames": len(adapted["coordinates"]),
        "coordinate_frame": adapted["frame_method"],
        "odd_sign_anchored": False,
        "observed_core_joint_fraction": adapted["observed_core_joint_fraction"],
    }


def build(
    manifest_path: Path,
    output_root: Path,
    output_manifest: Path,
    rejects_path: Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"GAVD pose manifest is missing {sorted(missing)}")
    if not frame.sequence_id.is_unique:
        raise ValueError("sequence_id must be unique")
    if not set(frame.split).issubset({"train", "validation", "test"}):
        raise ValueError("split must contain only train, validation, and test")
    split_counts = frame.groupby("video_id").split.nunique()
    if (split_counts != 1).any():
        leaked = split_counts[split_counts != 1].index.tolist()[:5]
        raise ValueError(f"Source videos cross data splits: {leaked}")
    pose_root = manifest_path.resolve().parent
    config = AdapterConfig(frame_policy="gauge-neutral-travel-or-image")
    rows = []
    rejects = []
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for row in frame.to_dict(orient="records"):
        try:
            rows.append(_adapt_row(row, pose_root, output_root, config))
        except Exception as exc:
            rejects.append(
                {
                    "sequence_id": str(row.get("sequence_id", "")),
                    "video_id": str(row.get("video_id", "")),
                    "split": str(row.get("split", "")),
                    "pose_path": str(row.get("pose_path", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not rows:
        raise ValueError("No GAVD sequence satisfied the Core11 adapter contract")
    output = pd.DataFrame(rows)
    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, output_manifest)
    rejects_path = rejects_path or output_manifest.with_name(
        output_manifest.stem + "-rejects.csv"
    )
    reject_frame = pd.DataFrame(
        rejects, columns=("sequence_id", "video_id", "split", "pose_path", "error")
    )
    reject_temporary = rejects_path.with_suffix(rejects_path.suffix + ".tmp")
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    reject_frame.to_csv(reject_temporary, index=False)
    os.replace(reject_temporary, rejects_path)
    output.attrs["rejects_path"] = str(rejects_path)
    output.attrs["reject_count"] = len(rejects)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--rejects", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build(
        args.pose_manifest.resolve(),
        args.output_root.resolve(),
        args.output_manifest.resolve(),
        args.rejects.resolve() if args.rejects is not None else None,
    )
    print(
        f"Wrote {len(output):,} sequences from {output.video_id.nunique():,} "
        f"source videos to {args.output_manifest}"
    )
    print(
        f"Recorded {output.attrs['reject_count']:,} rejected sequences in "
        f"{output.attrs['rejects_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
