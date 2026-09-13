"""Shared test paths and fixtures; no test cases or discovery aliases."""
from pathlib import Path
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[1]


def artifact_snapshot(root):
    """Record file contents and modification times for read-only checks."""
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*") if path.is_file()
    }


def source_discovery_fixture(root, *, gap=False, short=False):
    import pandas as pd
    from gavd6_sjepa.research_directions.future_prediction.cohort import deterministic_window_start
    from gavd6_sjepa.research_directions.future_prediction.contracts import initialize_run

    cache = root / "storage"
    (cache / "all").mkdir(parents=True)
    source_ids = [f"video-{i}" for i in range(25)]
    for video_id in source_ids:
        (cache / "all" / f"{video_id}.mp4").write_bytes(b"source fixture; discovery only")
    sequences, frames = [], []
    missing_frame = deterministic_window_start(0, 159, "sequence-0") + 33
    for i in range(50):
        length = 160 if gap and i == 0 else 32 if short and i == 0 else 64
        sequences.append({"sequence_id": f"sequence-{i}", "video_id": source_ids[i // 2],
                          "first_frame": 1, "last_frame": length})
        for frame in range(1, length + 1):
            if gap and i == 0 and frame == missing_frame:
                continue
            frames.append({"seq": f"sequence-{i}", "frame_num": frame, "id": source_ids[i // 2],
                           "bbox": str({"left": 10, "top": 10, "width": 50, "height": 80}),
                           "vid_info": str({"width": 100, "height": 100})})
    sequence_path, video_path, annotation_path = (root / name for name in ("sequences.csv", "videos.csv", "annotations.csv"))
    pd.DataFrame(sequences).to_csv(sequence_path, index=False)
    pd.DataFrame({"video_id": source_ids}).to_csv(video_path, index=False)
    pd.DataFrame(frames).to_csv(annotation_path, index=False)
    model = root / "model"
    model.write_text("fixture")
    run = root / "run"
    initialize_run(run, protocol="legacy-v1", sequence_manifest=sequence_path, video_manifest=video_path,
                   annotations=[annotation_path], pose_model=model, vjepa_root=root,
                   checkpoint=model, synthetic=True, youtube_dir=cache)
    return (run, sequence_path, video_path, [annotation_path], cache), missing_frame
