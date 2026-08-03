"""Run-once pose extraction over all videos in video-data/.

Notebook 01 walks a learner through this same logic cell by cell. This script is
the batch version we run during setup so the cached .npz files exist and every
later notebook opens instantly. It is safe to re-run: it skips videos already
cached unless --force is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> experiment dir
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from sjepa.data import (  # noqa: E402
    load_video_sequence, clean_sequence, normalize_sequence,
    save_sequence_npz, source_id_from_name,
)

VIDEO_DIR = EXP_DIR / "video-data"
CACHE_DIR = EXP_DIR / "artifacts" / "keypoints"
CLASSES = ["normal", "ms", "pd"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fps", type=int, default=15)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max videos per class (debug)")
    args = ap.parse_args()

    from ambient.pose.model_management import MediaPipeModelManager
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor

    MediaPipeModelManager().ensure_model_available()
    extractor = SequenceKeypointExtractor()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    for label in CLASSES:
        folder = VIDEO_DIR / label
        videos = sorted(folder.glob("*.mp4"))
        if args.limit:
            videos = videos[: args.limit]
        for vid in videos:
            source_id = source_id_from_name(vid.name)
            clip_name = vid.stem
            out = CACHE_DIR / f"{label}__{source_id}__{clip_name}.npz"
            if out.exists() and not args.force:
                with np.load(out, allow_pickle=True) as z:
                    n = int(z["keypoints_norm"].shape[0])
                rows.append(dict(path=str(out), label=label, source_id=source_id,
                                 clip_name=clip_name, n_frames=n, cached=True))
                print(f"[skip cached] {vid.name}")
                continue

            print(f"[extract] {label}/{vid.name}", flush=True)
            raw = load_video_sequence(vid, target_fps=args.target_fps, extractor=extractor)
            cleaned = clean_sequence(raw)
            if cleaned is None or cleaned.shape[0] < 8:
                skipped.append((label, vid.name, "too few valid frames"))
                print(f"    -> SKIPPED (too few valid frames)")
                continue
            norm = normalize_sequence(cleaned)
            save_sequence_npz(out, cleaned, norm, args.target_fps, source_id, label, clip_name)
            rows.append(dict(path=str(out), label=label, source_id=source_id,
                             clip_name=clip_name, n_frames=int(norm.shape[0]), cached=False))
            print(f"    -> {norm.shape[0]} frames cached")

    index = pd.DataFrame(rows)
    index_path = EXP_DIR / "artifacts" / "keypoints_index.parquet"
    index.to_parquet(index_path, index=False)

    # A grouped manifest that also records the source id for split reproducibility.
    grouped = index[["label", "source_id", "clip_name", "n_frames"]].copy()
    grouped.to_csv(EXP_DIR / "artifacts" / "manifest_grouped.csv", index=False)

    print("\n=== summary ===")
    print(index.groupby("label").agg(videos=("clip_name", "count"),
                                     sources=("source_id", "nunique"),
                                     total_frames=("n_frames", "sum")))
    if skipped:
        print("\nskipped:")
        for s in skipped:
            print("  ", s)
    print(f"\nindex written to {index_path} ({len(index)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
