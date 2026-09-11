"""Frozen protocol and artifact lineage for the Experiment 0 feasibility gate."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import shutil
import socket
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_json,
    sha256_file,
)

VJEPA_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
ARMS = (
    "real-skeleton",
    "time-shuffle",
    "clip-mismatch",
    "background-target",
    "no-skeleton",
)
LEGACY_PROTOCOL = "legacy-v1"
DIRECT_PROTOCOL = "direct-v2"
DIRECT_ARMS = tuple(arm for arm in ARMS if arm != "background-target")


def protocol_name(run):
    name = run.get("protocol", LEGACY_PROTOCOL)
    if name not in {LEGACY_PROTOCOL, DIRECT_PROTOCOL}:
        raise ValueError(f"Unknown Experiment 0 protocol: {name}")
    return name


def experiment_arms(root):
    run = check_run(root)
    expected = DIRECT_ARMS if protocol_name(run) == DIRECT_PROTOCOL else ARMS
    if tuple(read_json(Path(root) / "config/control-contract.json")["arms"]) != expected:
        raise ValueError("Control arms disagree with the frozen experiment protocol")
    return expected


def audit_summary_path(root):
    name = "readiness-summary.json" if protocol_name(check_run(root)) == DIRECT_PROTOCOL else "validity-summary.json"
    return Path(root) / "qc" / name


@dataclass(frozen=True)
class FrameContract:
    frames_per_clip: int = 64
    context_stop_exclusive: int = 32
    target_horizon_frames: int = 8
    target_tubelet_start: int = 38
    target_tubelet_stop_exclusive: int = 40
    resolution: int = 384
    patch_size: int = 16
    tubelet_size: int = 2

    def __post_init__(self):
        if (
            min(
                self.frames_per_clip,
                self.context_stop_exclusive,
                self.resolution,
                self.patch_size,
                self.tubelet_size,
            )
            <= 0
            or self.frames_per_clip % self.tubelet_size
            or self.context_stop_exclusive % self.tubelet_size
            or self.target_tubelet_start % self.tubelet_size
            or self.target_tubelet_start < self.context_stop_exclusive
            or self.target_tubelet_stop_exclusive - self.target_tubelet_start
            != self.tubelet_size
            or self.target_tubelet_stop_exclusive - self.context_stop_exclusive
            != self.target_horizon_frames
            or self.target_tubelet_stop_exclusive > self.frames_per_clip
            or self.resolution % self.patch_size
        ):
            raise ValueError("Invalid past/future frame or token boundary")

    @property
    def grid(self):
        return self.resolution // self.patch_size


FRAME = FrameContract()


@dataclass(frozen=True)
class GateThresholds:
    real_delta_r2_min: float = 0.05
    real_to_shuffle_min: float = 2.0
    mismatch_delta_r2_max: float = 0.01
    person_ablation_reduction_min: float = 0.50
    motion_to_background_change_min: float = 2.0
    person_edit_direction_fraction_min: float = 0.80
    bootstrap_positive_fraction_min: float = 0.90


@dataclass(frozen=True)
class ModelContract:
    seeds: tuple[int, ...] = (7, 19, 31)
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    weight_decays: tuple[float, ...] = (0.01, 0.1)
    updates: tuple[int, ...] = (25, 50, 100, 200)
    width: int = 64
    learning_rate: float = 0.001
    gradient_clip: float = 1.0
    inner_folds: int = 3
    target_variance_tolerance: float = 1e-10
    bootstrap_repetitions: int = 2000
    seed_aggregation: str = "arithmetic_mean_of_scores"
    stopping_rule: str = "minimum_pooled_inner_source_weighted_mse; ties_choose_first"

    def __post_init__(self):
        if (
            len(self.seeds) != 3
            or len(set(self.seeds)) != 3
            or not self.updates
            or tuple(sorted(set(self.updates))) != tuple(self.updates)
            or min(self.updates) < 1
            or self.width < 1
            or self.inner_folds < 2
            or self.bootstrap_repetitions < 1
            or not self.ridge_alphas
            or min(self.ridge_alphas) <= 0
            or not self.weight_decays
            or min(self.weight_decays) < 0
            or self.learning_rate <= 0
            or self.gradient_clip <= 0
            or self.target_variance_tolerance <= 0
            or self.seed_aggregation != "arithmetic_mean_of_scores"
        ):
            raise ValueError("Invalid frozen model/seed/selection contract")


def stable_key(*values, seed=260905):
    return sha256(":".join(map(str, (seed, *values))).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def stage_lock(root, stage):
    """Reject duplicate CLI/Slurm writers; the OS releases the lock after a crash."""
    path = Path(root) / "logs/locks" / f"{stage}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another job is writing stage {stage}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json(path, payload):
    # Shared writer remains backwards compatible; this experiment forbids NaN JSON.
    json.dumps(payload, allow_nan=False)
    atomic_write_json(Path(path), payload)


def write_once_json(path, payload):
    path = Path(path)
    if path.exists():
        if read_json(path) != json.loads(json.dumps(payload, allow_nan=False)):
            raise ValueError(f"Frozen artifact differs: {path}. Use a new run ID.")
        return
    write_json(path, payload)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def code_fingerprint():
    """Include uncommitted implementation, shared primitives and dependency lock."""
    package = Path(__file__).resolve().parents[2]
    checkout = package.parents[1]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [
        package / "data_foundations/gavd_pose_extraction_primitives.py",
        package / "data_foundations/gavd_video_download_pipeline.py",
        package / "shared_infrastructure/artifact_io_operations.py",
        package / "command_line_interface.py",
        checkout / "pyproject.toml",
        checkout / "uv.lock",
    ]
    paths += sorted((checkout / "slurm/future-innovation").glob("*.sh"))
    paths += sorted((checkout / "slurm/future-innovation").glob("*.sbatch"))
    return stable_key(*[f"{p.name}:{sha256_file(p)}" for p in paths if p.exists()])


def runtime_versions():
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "torch",
        "torchvision",
        "timm",
        "einops",
        "mediapipe",
        "opencv-python",
        "pyarrow",
        "joblib",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.system(), "packages": packages}


def record_run_provenance(root, contract):
    """Record execution identity without rewriting the contracts binding saved data.

    A source hash or package version change is not evidence of corrupt data.
    Record each process/environment combination (including Slurm array tasks),
    while leaving actual configuration, artifact and model checks to their readers.
    """
    runtime = runtime_versions()
    snapshot = {
        "code_sha256": code_fingerprint(),
        "initial_code_sha256": contract["code_sha256"],
        "runtime": runtime,
        "runtime_changed": runtime != read_json(root / "config/runtime-contract.json"),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    snapshot["code_changed"] = snapshot["code_sha256"] != contract["code_sha256"]
    identity = stable_key(json.dumps(snapshot, sort_keys=True))
    path = root / "logs/provenance" / f"{identity}.json"
    if not path.exists():
        write_once_json(path, snapshot)
        if snapshot["code_changed"] or snapshot["runtime_changed"]:
            print(
                f"FI: code/runtime differs from initialization; recorded in {path}. "
                "Continuing with frozen configuration and artifact validation.",
                file=sys.stderr,
            )


def check_run(root):
    root = Path(root)
    contract = read_json(root / "config/run-contract.json")
    for name, digest in contract["config_sha256"].items():
        if sha256_file(root / "config" / name) != digest:
            raise ValueError(f"Frozen configuration changed: {name}")
    record_run_provenance(root, contract)
    return contract


def measurement_complete(decision):
    """Recognize both current decisions and pre-resumption legacy reports."""
    return decision.get("measurement_complete", decision.get("metrics") is not None)


def verified_report_decision(root):
    """Validate a sealed report before reusing it or archiving a legacy STOP."""
    root = Path(root)
    seal = root / "reports/final-report-contract.json"
    if seal.exists():
        for path, digest in read_json(seal)["artifacts"].items():
            if sha256_file(root / path) != digest:
                raise ValueError("Sealed final report changed")
    path = root / "reports/gate-decision.json"
    return read_json(path) if path.exists() else None


def initialize_run(
    root,
    *,
    sequence_manifest,
    video_manifest,
    annotations,
    pose_model,
    vjepa_root,
    checkpoint,
    change_reason="Experiment 0 initialization",
    synthetic=False,
    model=None,
    youtube_dir=None,
    video_roots=(),
    protocol=DIRECT_PROTOCOL,
    cohort_size=None,
):
    root = Path(root).resolve()
    change_reason = str(change_reason or "").strip() or "Experiment 0 initialization"
    protocol_name({"protocol": protocol})
    direct = protocol == DIRECT_PROTOCOL
    if cohort_size is not None and (type(cohort_size) is not int or cohort_size != 50):
        raise ValueError("Experiment 0 requires exactly 50 clips; full GAVD is reserved for the real experiment")
    cohort_size = 50
    if (root / "config/run-contract.json").exists():
        raise ValueError(
            "Run already initialized; resume its stages or choose a new run ID"
        )
    for directory in (
        "config",
        "manifests",
        "poses",
        "boxes",
        "frames",
        "teacher-cache",
        "models",
        "predictions",
        "qc",
        "reports",
        "logs",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    teacher = {
        "commit": VJEPA_COMMIT,
        "repository": "https://github.com/facebookresearch/vjepa2",
        "repository_path": str(Path(vjepa_root).resolve()),
        "builder": "vjepa2_1_vit_base_384",
        "checkpoint_key": "ema_encoder",
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "embedding_dim": 768,
        "checkpoint_loading": "pretrained=False; strict ema_encoder state loading",
        "output": "final layer 11, encoder.norms_block[-1] then fixed non-affine layer_norm(eps=1e-5)",
        "target_normalization": "official target F.layer_norm after learned encoder norm; applied once",
        "crop": "short_side=int(384*256/224)=438; OpenCV bilinear; rounded center crop",
        "layout": "B,C,T,H,W -> temporal-major T,row,col tokens; masks keep tokens before attention",
        "target_is_full_clip_contextual": True,
        "dtype": "float32",
        "synthetic": synthetic,
    }
    config = {
        "frame-contract.json": asdict(FRAME),
        "teacher-contract.json": teacher,
        "model-contract.json": asdict(model or ModelContract()),
        "thresholds.json": asdict(GateThresholds()),
        "runtime-contract.json": runtime_versions(),
        "nuisance-contract.json": {
            "pixel_pose_motion_frames": [0, 31],
            "offline_endpoint_metadata_permitted": True,
            "view": "fixed vocabulary: front, back, left side, right side, unknown",
            "continuous_preprocessing": "source-weighted inner/outer-training imputation and scaling",
            "background_flow": "median Farneback outside union of adjacent context person boxes",
        },
        "control-contract.json": {
            "arms": DIRECT_ARMS if direct else ARMS,
            "shuffle_block": 4,
            "shuffle_seed": 260905,
            "mismatch": "partition-local standardized metadata; Hungarian without replacement; different source",
            "no_skeleton": "zero x,y,confidence; retain validity; identical parameter count",
            "capacity_attribution": ("primary paired real-minus-no-skeleton contrast; validity retained in matched control"
                                     if direct else "paired no-skeleton control is reported with every decision; no manual interpretation gate"),
            "audit_count": 10,
            "pixel_feather_pixels": 8,
            "pixel_person_edit": "different-source donor person ROI at reversed future times; resized into recipient box",
            "pixel_background_edit": "static first-frame donor background with donor person inpainted",
            "pixel_donor": "different source, stable hash order, fixed before teacher features",
        },
    }
    if direct:
        config["protocol-contract.json"] = {
            "name": DIRECT_PROTOCOL,
            "purpose": "incremental skeleton coordinate/confidence history beyond RGB, nuisance and matched validity/capacity",
            "revision": "author-requested after inspection of legacy sensitivity audit; not the original preregistration",
            "removed_prerequisites": ["person/background pixel-edit selectivity", "background-target gain reduction",
                                      "minimum 90% crop retention", "minimum 45% whole-body pose coverage"],
            "retained_checks": ["artifact integrity", "source isolation", "past-only inputs", "teacher repeatability", "person-target variance"],
            "cohort_policy": "50 eligible clips from the available full-GAVD candidate pool; at least 25 sources, at most two clips per source",
            "empty_background": "zero unavailable RGB/flow summaries and pooling vectors; prefix pixel, flow-pair and token support in nuisance inputs",
            "skeleton_increment_min": 0.0,
            "skeleton_increment_comparison": "strictly positive mean real-minus-no-skeleton; positive in every seed and >=90% paired source draws",
        }
        for key in ("motion_to_background_change_min", "person_edit_direction_fraction_min", "person_ablation_reduction_min"):
            config["thresholds.json"].pop(key)
        for key in ("pixel_feather_pixels", "pixel_person_edit", "pixel_background_edit", "pixel_donor"):
            config["control-contract.json"].pop(key)
        config["control-contract.json"]["audit_count"] = 3
    for name, value in config.items():
        write_once_json(root / "config" / name, value)
    lock = Path(__file__).resolve().parents[4] / "uv.lock"
    shutil.copyfile(lock, root / "config/uv.lock")
    inputs = {
        str(Path(p).resolve()): sha256_file(p)
        for p in [sequence_manifest, video_manifest, pose_model, *annotations]
    }
    write_once_json(
        root / "config/run-contract.json",
        {
            "experiment": "future-innovation-experiment-0",
            "protocol": protocol,
            "run_id": root.name,
            "change_reason": change_reason,
            "code_sha256": code_fingerprint(),
            "config_sha256": {
                name: sha256_file(root / "config" / name)
                for name in [*config, "uv.lock"]
            },
            "inputs_sha256": inputs,
            "selection_seed": 260905,
            "projection_seed": 260905,
            "input_paths": {
                "sequence_manifest": str(Path(sequence_manifest).resolve()),
                "video_manifest": str(Path(video_manifest).resolve()),
                "annotations": [str(Path(p).resolve()) for p in annotations],
                "youtube_dir": str(
                    Path(
                        youtube_dir or Path(sequence_manifest).parent.parent / "youtube"
                    ).resolve()
                ),
                "video_roots": [str(Path(p).expanduser().resolve()) for p in video_roots],
            },
            "cohort_size": cohort_size,
            "minimum_sources": 25,
            "source_cap": 2,
            "outer_folds": 5,
            "source_frame_origin": "GAVD one-based converted once to zero-based at candidate construction",
            "pose_model": str(Path(pose_model).resolve()),
            "pose_visibility_threshold": 0.45,
            "minimum_pose_coverage": 0.0 if direct else 0.45,
            "minimum_crop_retention": 0.0 if direct else 0.90,
            "box_eligibility": ("64 aligned source frames; nonempty person token region at context/target; at least one observed joint transition"
                                if direct else "all 64 source frames annotated; retention >=0.9 for context and target"),
            "synthetic": synthetic,
        },
    )
    return root


def equal_source_weights(video_ids):
    _, inverse, counts = np.unique(
        np.asarray(video_ids, dtype=str), return_inverse=True, return_counts=True
    )
    if not len(inverse):
        raise ValueError("Source weights require a nonempty partition")
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def load_model_contract(root):
    return ModelContract(**read_json(Path(root) / "config/model-contract.json"))
