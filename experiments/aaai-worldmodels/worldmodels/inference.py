"""Classify one unlabeled local walking sequence using a trusted local run."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from .data import JOINT_IDS, LABELS, sha256_file
from .features import build_features
from .pose import make_windows
from .settings import configure

# These are the prepared-cohort parameters in workflow.prepare.
FRAMES, STRIDE, CONTEXT, FRAME_STEP = 16, 8, 8, 2
CONFIDENCE, MIN_VALID = 0.5, 0.7
CHECKPOINTS = {"jepa_normal": "normal_completion.pt", "jepa_adapted": "adaptation_4.pt",
               "jepa_mixed": "mixed_adaptation.pt", "untrained_probe": "untrained_encoder.pt"}


def extract_video_range(video, pose_model, *, start_frame=1, end_frame, max_windows=None):
    """Decode an inclusive one-based range once and use Ambient's RGB image API.

    Missing detections retain their frame positions. The first detected person is
    used, so the user must review the person and walking interval before relying
    on a score. No source labels or GAVD annotation rows are required.
    """
    import cv2
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    video, pose_model = Path(video), Path(pose_model)
    if not video.is_file() or not pose_model.is_file():
        raise FileNotFoundError("Both video and explicit pose-model files must exist")
    if start_frame < 1 or end_frame < start_frame:
        raise ValueError("Use a nonempty inclusive one-based frame range")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive")
    numbers = np.arange(start_frame, end_frame + 1, FRAME_STEP)
    if max_windows is not None:
        numbers = numbers[:FRAMES + (max_windows - 1) * STRIDE]
    xy = np.full((len(numbers), 33, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(numbers), 33), dtype=np.float32)
    timestamps = np.full(len(numbers), np.nan)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Video could not be opened")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    dimensions = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video has no positive finite nominal frame rate")
    extractor = SequenceKeypointExtractor()
    frame_to_output = {int(frame)-1: i for i, frame in enumerate(numbers)}
    decoded = 0
    detected = 0
    try:
        for decoded_index in range(int(numbers[-1])):
            ok, image = cap.read()
            if not ok:
                break
            if decoded_index not in frame_to_output:
                continue
            i = frame_to_output[decoded_index]
            decoded += 1
            timestamps[i] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            pose = extractor.extract_from_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), str(pose_model))
            if pose is not None and len(pose.keypoints):
                detected += 1
                for kp in pose.keypoints:
                    if 0 <= kp.id < 33:
                        xy[i, kp.id] = [kp.x, kp.y]
                        confidence[i, kp.id] = kp.confidence
    finally:
        cap.release()
        extractor.cleanup()
    video_hash = sha256_file(video)
    identity = dict(video_sha256=video_hash, pose_model_sha256=sha256_file(pose_model),
                    start_frame=int(start_frame), end_frame=int(end_frame), frame_step=FRAME_STEP,
                    processed_end_frame=int(numbers[-1]), decoded_sampled_frames=decoded,
                    detected_sampled_frames=detected, requested_sampled_frames=len(numbers),
                    coordinate_units="pixels", image_dimensions=dimensions,
                    source_fps=fps, sample_fps=fps / FRAME_STEP,
                    person_selection="full_image_first_detection")
    return dict(xy=xy, confidence=confidence, frame_numbers=numbers, decoded_indexes=numbers-1,
                timestamp_seconds=timestamps, fps=np.array(fps),
                sequence_id=np.array("input-sequence"), source_id=np.array(video_hash),
                label=np.array(""), fingerprint=np.array(video_hash), provenance=identity)


def _check_run(run_dir):
    """Fail closed on code/schema/version mismatch before unpickling joblib."""
    run = Path(run_dir)
    result = json.loads((run / "results.json").read_text())
    if result.get("identity") != "medwm-gait-v1":
        raise ValueError("Unsupported run identity")
    for filename in ("data.py", "pose.py", "features.py", "model.py", "masking.py", "training.py", "evaluation.py", "inference.py"):
        recorded = result.get("code_sha256", {}).get(filename)
        if not recorded or recorded != sha256_file(Path(__file__).with_name(filename)):
            raise ValueError(f"Run/current code mismatch: {filename}; use the original run environment")
    for package in ("torch", "numpy", "scikit-learn", "mediapipe"):
        recorded = result.get("versions", {}).get(package)
        if recorded != importlib.metadata.version(package):
            raise ValueError(f"Run/current package mismatch: {package}; restore the recorded version")
    import ambient.pose.joint_angles as shared_angles
    import ambient.pose.keypoint_extractor as shared_extractor
    for name,module in (("joint_angles",shared_angles),("keypoint_extractor",shared_extractor)):
        if result.get("ambient_source_sha256",{}).get(name)!=sha256_file(module.__file__):
            raise ValueError(f"Run/current shared Ambient mismatch: {name}")
    hashes = result.get("artifact_sha256", {})
    required = ["rf_classifier.joblib", "normal_completion.pt", "adaptation_4.pt", "mixed_adaptation.pt",
                "normal_forecast.pt", "jepa_normal_classifier.joblib", "jepa_adapted_classifier.joblib",
                "jepa_mixed_classifier.joblib"]
    for filename in required:
        if not (run / filename).is_file() or hashes.get(filename) != sha256_file(run / filename):
            raise ValueError(f"Missing or changed artifact identity: {filename}")
    return result


def classify_pose(pose, run_dir, *, max_windows=None):
    """Apply saved classifiers to one extracted sequence, with no truth label.

    Joblib can execute code while loading. Only pass runs produced locally by this
    project or otherwise independently trusted. File hashes detect changed bytes;
    they do not establish that a downloaded pickle is safe.
    """
    import joblib
    from .evaluation import align_probabilities
    from .training import load_checkpoint, encode, anomaly_scores
    result = _check_run(run_dir)
    run = Path(run_dir)
    expected_pose_model = result.get("pose_model_sha256")
    actual_pose_model = pose.get("provenance", {}).get("pose_model_sha256")
    if not expected_pose_model or actual_pose_model != expected_pose_model:
        raise ValueError("Input pose-model weights differ from the recorded training extractor")
    x, valid, meta = make_windows(pose, FRAMES, STRIDE, CONTEXT, CONFIDENCE, MIN_VALID)
    if max_windows is not None:
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        x, valid, meta = x[:max_windows], valid[:max_windows], meta.iloc[:max_windows]
    base = dict(schema=1, run=str(run.resolve()), run_status=result.get("status"),
                input=pose.get("provenance", {}), joint_ids=list(JOINT_IDS),
                coordinate_system="pelvis-centered image xy divided by prefix torso scale",
                interpretation="Research predictions of dataset gait labels, not clinical diagnoses.",
                person_review_status="required", n_windows=len(x),
                quality=dict(confidence_threshold=CONFIDENCE, minimum_valid_fraction=MIN_VALID,
                             frames_per_window=FRAMES, stride=STRIDE, context_frames=CONTEXT))
    if not len(x):
        return dict(base, status="quality_failure", reason="No window meets the fixed pose-quality and length requirements",
                    classifiers={}, forecast=None)
    features, names = build_features(x, valid, fps=meta.sample_fps.to_numpy())
    if names != result.get("features"):
        raise ValueError("Feature names or order differ from the trained RF contract")
    classifiers = {}
    unavailable = {}
    for name in ("rf", "jepa_normal", "jepa_adapted", "jepa_mixed", "untrained_probe"):
        matrix = features
        if name != "rf":
            checkpoint = run / CHECKPOINTS[name]
            if not checkpoint.is_file():
                unavailable[name] = "Matching encoder checkpoint was not saved"
                continue
            if result.get("artifact_sha256", {}).get(checkpoint.name) != sha256_file(checkpoint):
                raise ValueError(f"Changed encoder artifact: {checkpoint.name}")
            encoder = load_checkpoint(checkpoint)
            if encoder.config.frames != FRAMES or tuple(encoder.config.joint_ids) != tuple(JOINT_IDS):
                raise ValueError("Encoder token contract differs from preparation settings")
            matrix = encode(encoder, x, valid)
        classifier_path = run / f"{name}_classifier.joblib"
        if result.get("artifact_sha256", {}).get(classifier_path.name) != sha256_file(classifier_path):
            raise ValueError(f"Changed classifier artifact: {classifier_path.name}")
        estimator = joblib.load(classifier_path)
        probability = align_probabilities(estimator, matrix).mean(0)
        classifiers[name] = dict(predicted_label=LABELS[int(np.argmax(probability))],
                                probabilities=dict(zip(LABELS, map(float, probability))),
                                abnormal_probability=float(1 - probability[list(LABELS).index("normal")]),
                                aggregation="arithmetic mean of window probabilities in this sequence")
    forecast = load_checkpoint(run / "normal_forecast.pt")
    scores = anomaly_scores(forecast, x, valid, context_frames=CONTEXT)
    mean_score = float(np.mean(scores["coordinate_mse"]))
    threshold = result.get("anomaly", {}).get("threshold")
    base["quality"].update(mean_valid_fraction=float(meta.valid_fraction.mean()),
                           sampled_fps=sorted(map(float, meta.sample_fps.unique())),
                           first_retained_frame=int(meta.first_frame.min()),
                           last_retained_frame=int(meta.last_frame.max()))
    return dict(base, status="predictions_require_person_review", classifiers=classifiers,
                unavailable=unavailable,
                forecast=dict(coordinate_mse=mean_score,
                              latent_ce=float(np.mean(scores["latent_ce"])),
                              normal_validation_threshold=threshold,
                              exceeds_threshold=bool(mean_score > threshold) if threshold is not None else None,
                              threshold_status=result.get("anomaly", {}).get("threshold_status"),
                              threshold_scope="Calibrated on source means; one new sequence is not a coverage guarantee"))


def classify_video(video, run_dir, pose_model, *, start_frame=1, end_frame, max_windows=None):
    """Classify one local video interval against an immutable trusted run."""
    result = _check_run(run_dir)  # Fail before expensive extraction.
    if result.get("pose_model_sha256") != sha256_file(pose_model):
        raise ValueError("Supplied pose-model weights differ from training")
    pose = extract_video_range(video, pose_model, start_frame=start_frame,
                               end_frame=end_frame, max_windows=max_windows)
    return classify_pose(pose, run_dir, max_windows=max_windows)


def main(argv=None):
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="Trusted local comparison run, containing results.json")
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = classify_video(args.video, args.run, args.pose_model, start_frame=args.start_frame,
                                end_frame=args.end_frame, max_windows=args.max_windows)
        status = 2 if report["status"] == "quality_failure" else 0
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        report = dict(schema=1, status="input_or_compatibility_failure", reason=str(exc), classifiers={}, forecast=None)
        status = 2
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return status


if __name__ == "__main__":
    sys.exit(main())
