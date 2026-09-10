"""Label-blind deterministic candidate order, pose eligibility, and cohort freeze."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv,
    sha256_file,
)

from .fi_contracts import (
    check_run,
    read_json,
    save_npz,
    stable_key,
    write_json,
    write_once_json,
)
from .fi_token_regions import region_masks, union_box
from .fi_source_inventory import discover_sources
from .fi_video_pose import (
    CropGeometry,
    alignment_sheet,
    annotation_boxes,
    decode_exact_window,
    extract_history,
)


def assign_source_folds(video_ids, n_folds=5):
    ordered = sorted(
        set(map(str, video_ids)), key=lambda value: stable_key("fold", value)
    )
    return {video: index % n_folds for index, video in enumerate(ordered)}


def deterministic_window_start(first_frame, last_frame, sequence_id):
    choices = last_frame - 63 - first_frame + 1
    if choices < 1:
        raise ValueError("Sequence shorter than 64 frames")
    return first_frame + int(stable_key("window", sequence_id)[:16], 16) % choices


def select_eligible(rows, count=50):
    selected, counts = [], Counter()
    for row in sorted(rows, key=lambda r: stable_key("candidate", r["sequence_id"])):
        if counts[row["video_id"]] < 2:
            selected.append(row)
            counts[row["video_id"]] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"Need exactly {count} eligible windows; found {len(selected)}"
        )
    folds = assign_source_folds([r["video_id"] for r in selected])
    return pd.DataFrame([{**r, "outer_fold": folds[r["video_id"]]} for r in selected])


def validate_cohort(cohort):
    if len(cohort) != 50 or not cohort.window_id.is_unique:
        raise ValueError("Cohort must contain exactly 50 unique windows")
    if cohort.video_id.nunique() < 25 or cohort.groupby("video_id").size().max() > 2:
        raise ValueError("Cohort violates source count/cap")
    if cohort.groupby("video_id").outer_fold.nunique().max() != 1:
        raise ValueError("A source appears in multiple folds")
    if set(cohort.outer_fold) != set(range(5)):
        raise ValueError("Cohort must cover all five outer folds")


def build_candidates(root, sequence_manifest, video_manifest, annotations, youtube_dir, video_roots=()):
    root = Path(root)
    contract = check_run(root)
    if (root / "config/candidates-contract.json").exists():
        saved = read_json(root / "config/candidates-contract.json")
        if sha256_file(root / "manifests/candidates.csv") != saved["manifest_sha256"]:
            raise ValueError("Candidate order changed after freeze")
        for path, digest in saved["artifacts"].items():
            if sha256_file(root / path) != digest:
                raise ValueError(f"Candidate artifact changed: {path}")
        candidates = pd.read_csv(root / "manifests/candidates.csv")
        if len(candidates) < 50:
            raise ValueError(f"Only {len(candidates)} candidates; see exclusions.csv")
        print(f"Reusing {len(candidates)} verified candidates")
        return
    for path in [sequence_manifest, video_manifest, *annotations]:
        if contract["inputs_sha256"].get(str(Path(path).resolve())) != sha256_file(
            path
        ):
            raise ValueError(f"Input was not preregistered or changed: {path}")
    sequences = pd.read_csv(sequence_manifest)
    videos = pd.read_csv(video_manifest)
    if not sequences.sequence_id.is_unique or not videos.video_id.is_unique:
        raise ValueError("Duplicate sequence or source in input manifest")
    if not set(sequences.video_id) <= set(videos.video_id):
        raise ValueError("Sequence source absent from video inventory")
    columns = ["seq", "frame_num", "bbox", "vid_info", "id"]
    # Read only alignment columns, never presentation or diagnosis labels.
    frames = pd.concat(
        [pd.read_csv(p, usecols=columns) for p in annotations], ignore_index=True
    )
    if frames.duplicated(["seq", "frame_num"]).any():
        raise ValueError(
            "Duplicate frame annotations (do not mix overlapping annotation releases)"
        )
    groups = {key: group for key, group in frames.groupby("seq", sort=False)}
    sources = discover_sources(videos, video_manifest, youtube_dir, video_roots)
    atomic_write_dataframe_csv(root / "manifests/source-availability.csv", pd.DataFrame([
        {"video_id": video_id, "available": item["path"] is not None,
         "video_path": str(item["path"] or ""), "method": item["method"], "reason": item["error"]}
        for video_id, item in sorted(sources.items())
    ]))
    candidates, exclusions = [], []
    for row in sorted(
        sequences.to_dict("records"),
        key=lambda r: stable_key("candidate", r["sequence_id"]),
    ):
        sequence = str(row["sequence_id"])
        try:
            if not all(
                re.fullmatch(r"[A-Za-z0-9_-]+", str(value))
                for value in (sequence, row["video_id"])
            ):
                raise ValueError(
                    "Sequence/source identifiers must be safe artifact basenames"
                )
            first, last = int(row["first_frame"]) - 1, int(row["last_frame"]) - 1
            if first < 0:
                raise ValueError(
                    "GAVD source frames must be one-based positive integers"
                )
            source = sources[str(row["video_id"])]
            if source["path"] is None:
                raise ValueError(source["error"])
            # The annotation span, not the duration of the entire source video,
            # bounds a tracked person's window. Never fabricate repeated frames.
            start = deterministic_window_start(first, last, sequence)
            group = groups.get(sequence)
            if group is None:
                raise ValueError("Missing per-frame annotation")
            selected = group.set_index("frame_num").reindex(
                range(start + 1, start + 65)
            )
            if selected[columns[2:]].isna().any().any():
                # A hash landing on a gap must not discard another intact window.
                available = sorted({int(f) for f in group.dropna(subset=columns[2:]).frame_num
                                    if first + 1 <= int(f) <= last + 1})
                starts = [available[i] - 1 for i in range(max(0, len(available) - 63))
                          if available[i + 63] - available[i] == 63]
                if not starts:
                    raise ValueError("No 64 consecutive annotated person boxes in sequence")
                start = starts[int(stable_key("window", sequence)[:16], 16) % len(starts)]
                selected = group.set_index("frame_num").reindex(range(start + 1, start + 65))
                if selected[columns[2:]].isna().any().any():
                    raise ValueError("Window has incomplete box or source metadata")
            if set(selected.id.astype(str)) != {str(row["video_id"])}:
                raise ValueError("Frame annotation source differs from sequence source")
            records = selected.reset_index().to_dict("records")
            boxes = annotation_boxes(records)
            window_id = f"{sequence}-{start:08d}"
            box_path = root / "boxes" / f"{window_id}.npz"
            save_npz(box_path, source_boxes=boxes)
            annotation_path = root / "boxes" / f"{window_id}.json"
            write_json(annotation_path, records)
            video_path = source["path"]
            candidates.append(
                {
                    "window_id": window_id,
                    "sequence_id": sequence,
                    "video_id": str(row["video_id"]),
                    "video_path": str(video_path.resolve()),
                    "source_first_frame": start,
                    "source_last_frame": start + 63,
                    "sequence_first_frame": first,
                    "sequence_last_frame": last,
                    "box_path": str(box_path),
                    "annotation_path": str(annotation_path),
                    "cam_view": str(row.get("cam_view", "unknown")),
                }
            )
        except (ValueError, KeyError) as error:
            exclusions.append(
                {"sequence_id": sequence, "stage": "candidate", "reason": str(error)}
            )
    atomic_write_dataframe_csv(
        root / "manifests/candidates.csv", pd.DataFrame(candidates)
    )
    atomic_write_dataframe_csv(
        root / "manifests/exclusions.csv",
        pd.DataFrame(exclusions, columns=["sequence_id", "stage", "reason"]),
    )
    # A failed discovery is retryable when storage is mounted or completed later.
    # Do not freeze an unusable candidate inventory permanently.
    if len(candidates) < 50:
        raise ValueError(f"Only {len(candidates)} candidates; see source-availability.csv and exclusions.csv")
    usable = sum(min(2, count) for count in Counter(row["video_id"] for row in candidates).values())
    if usable < 50:
        raise ValueError(f"Only {usable} candidate windows after source cap; need 50 from at least 25 sources")
    write_once_json(
        root / "config/candidates-contract.json",
        {
            "manifest_sha256": sha256_file(root / "manifests/candidates.csv"),
            "artifacts": {
                "manifests/source-availability.csv": sha256_file(root / "manifests/source-availability.csv"),
                **{
                str(Path(r[key]).relative_to(root)): sha256_file(r[key])
                for r in candidates
                for key in ("box_path", "annotation_path")
                },
            },
        },
    )


def extract_and_freeze(root):
    root = Path(root)
    contract = check_run(root)
    if (root / "config/cohort-contract.json").exists():
        cohort = load_cohort(root, verify_artifacts=True)
        print(f"Reusing {len(cohort)} verified cohort windows")
        return
    candidate_contract = read_json(root / "config/candidates-contract.json")
    if (
        sha256_file(root / "manifests/candidates.csv")
        != candidate_contract["manifest_sha256"]
    ):
        raise ValueError("Candidate order changed after freeze")
    pose_model = contract["pose_model"]
    if sha256_file(pose_model) != contract["inputs_sha256"][pose_model]:
        raise ValueError("Pose model changed")
    candidates = pd.read_csv(root / "manifests/candidates.csv").to_dict("records")
    exclusions = [
        row
        for row in pd.read_csv(root / "manifests/exclusions.csv").to_dict("records")
        if row["stage"] == "candidate"
    ]
    eligible, counts = [], Counter()
    for row in candidates:
        if len(eligible) == 50:
            exclusions.append(
                {
                    "sequence_id": row["sequence_id"],
                    "stage": "selection",
                    "reason": "after fixed cohort reached 50",
                }
            )
            continue
        if counts[row["video_id"]] >= 2:
            exclusions.append(
                {
                    "sequence_id": row["sequence_id"],
                    "stage": "selection",
                    "reason": "source cap",
                }
            )
            continue
        try:
            for key in ("box_path", "annotation_path"):
                relative = str(Path(row[key]).relative_to(root))
                if sha256_file(row[key]) != candidate_contract["artifacts"][relative]:
                    raise RuntimeError(f"Candidate artifact changed: {relative}")
            video, fps = decode_exact_window(
                row["video_path"], row["source_first_frame"]
            )
            records = read_json(row["annotation_path"])
            with np.load(row["box_path"], allow_pickle=False) as data:
                boxes = data["source_boxes"]
            geometry = CropGeometry(*video.shape[1:3])
            model_boxes, retention = geometry.boxes(boxes)
            if retention[np.r_[0:32, 38:40]].min() < contract["minimum_crop_retention"]:
                raise ValueError("Person crop retention below 90%")
            for t in list(range(16)) + [19]:
                region_masks(union_box(model_boxes[2 * t : 2 * t + 2]))
            raw, history, scale = extract_history(
                video, records, boxes, fps, pose_model
            )
            if history[..., 3].mean() < contract["minimum_pose_coverage"]:
                raise ValueError("Whole-body valid context coverage below 0.45")
            pose_path = root / "poses" / f"{row['window_id']}.npz"
            frame_path = root / "frames" / f"{row['window_id']}.npz"
            save_npz(
                pose_path,
                raw=raw,
                skeleton=history,
                body_scale=np.array(scale),
                source_frames=np.arange(
                    row["source_first_frame"], row["source_first_frame"] + 32
                ),
            )
            # Keep the exactly decoded RGB once under the versioned run root.
            save_npz(frame_path, video=video)
            model_box_path = root / "boxes" / f"{row['window_id']}-model.npz"
            save_npz(model_box_path, model_boxes=model_boxes, crop_retention=retention)
            overlay = root / "qc/alignment-overlays" / f"{row['window_id']}.jpg"
            alignment_sheet(video, raw, boxes, row["source_first_frame"], overlay)
            eligible.append(
                {
                    **row,
                    "decoded_fps": fps,
                    "source_width": video.shape[2],
                    "source_height": video.shape[1],
                    "horizon_seconds": 8 / fps,
                    "pose_path": str(pose_path),
                    "frame_path": str(frame_path),
                    "model_box_path": str(model_box_path),
                    "overlay_path": str(overlay),
                    "eligibility_reason": "all checks passed",
                }
            )
            counts[row["video_id"]] += 1
        except ValueError as error:
            exclusions.append(
                {
                    "sequence_id": row["sequence_id"],
                    "stage": "pose_eligibility",
                    "reason": str(error),
                }
            )
        atomic_write_dataframe_csv(
            root / "manifests/exclusions.csv", pd.DataFrame(exclusions)
        )
    atomic_write_dataframe_csv(
        root / "manifests/exclusions.csv", pd.DataFrame(exclusions)
    )
    cohort = select_eligible(eligible)
    validate_cohort(cohort)
    atomic_write_dataframe_csv(root / "manifests/gate-windows.csv", cohort)
    rows = cohort.to_dict("records")
    audit_rows = sorted(rows, key=lambda r: stable_key("audit", r["window_id"]))[:10]
    audits = []
    for row in audit_rows:
        donor = min(
            (r for r in rows if r["video_id"] != row["video_id"]),
            key=lambda r: stable_key("pixel-donor", row["window_id"], r["window_id"]),
        )
        audits.append(
            {"window_id": row["window_id"], "donor_window_id": donor["window_id"]}
        )
    atomic_write_dataframe_csv(
        root / "manifests/audit-windows.csv", pd.DataFrame(audits)
    )
    # Complete the run contract once, after eligibility but before any teacher
    # features. The initial thresholds/model configuration are unchanged.
    contract["manifest_sha256"] = sha256_file(root / "manifests/gate-windows.csv")
    write_json(root / "config/run-contract.json", contract)
    files = [
        "manifests/gate-windows.csv",
        "manifests/audit-windows.csv",
        "manifests/exclusions.csv",
    ]
    files += [
        str(Path(r[key]).relative_to(root))
        for r in rows
        for key in (
            "pose_path",
            "frame_path",
            "box_path",
            "model_box_path",
            "overlay_path",
        )
    ]
    write_once_json(
        root / "config/cohort-contract.json",
        {
            "run_contract_sha256": sha256_file(root / "config/run-contract.json"),
            "manifest_sha256": sha256_file(root / "manifests/gate-windows.csv"),
            "artifacts": {p: sha256_file(root / p) for p in files},
        },
    )


def load_cohort(root, verify_artifacts=False):
    root = Path(root)
    run = check_run(root)
    contract = read_json(root / "config/cohort-contract.json")
    if (
        sha256_file(root / "config/run-contract.json")
        != contract["run_contract_sha256"]
    ):
        raise ValueError("Run contract changed")
    if sha256_file(root / "manifests/gate-windows.csv") != contract["manifest_sha256"]:
        raise ValueError("Frozen cohort manifest changed")
    if run.get("manifest_sha256") != contract["manifest_sha256"]:
        raise ValueError("Run and cohort manifest fingerprints disagree")
    if verify_artifacts:
        for path, digest in contract["artifacts"].items():
            if sha256_file(root / path) != digest:
                raise ValueError(f"Cohort artifact changed: {path}")
    cohort = pd.read_csv(root / "manifests/gate-windows.csv")
    validate_cohort(cohort)
    return cohort
