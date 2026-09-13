"""Manifest-backed motions and controlled observations for the preservation study.

The manifests describe available assets; they do not make local copies of the
HAIC data. No synthetic replacement is made when a research input is missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from ...data_foundations.amass_conversion import load_amass_sequence


@dataclass
class MotionParameters:
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    dmpls: np.ndarray
    gender: str
    fps: float
    timestamps: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def load_amass_manifest(manifest_dir, amass_root, seed: int = 17) -> pd.DataFrame:
    """Join audited people to source files, preserving the repository's splits.

    Validation people are split once into calibration and development. Neither
    file availability nor a motion's measured event size changes its person role.
    ``available`` means path existence, not a successfully decoded motion.
    """
    folder = Path(manifest_dir)
    if (folder / "amass").is_dir():
        folder = folder / "amass"
    inventory = pd.read_csv(folder / "amass_raw_inventory_eligible.csv")
    registry = pd.read_csv(folder / "amass_subject_registry.csv")
    splits = pd.read_csv(folder / "amass_subject_splits.csv")
    approved = registry.loc[
        registry.identity_audit_status.eq("approved")
        & ~registry.excluded.astype(str).str.lower().eq("true"),
        ["subject_id_candidate", "identity"],
    ]
    people = splits[["identity", "split"]].drop_duplicates()
    if people.identity.duplicated().any():
        raise ValueError("An audited AMASS person belongs to multiple existing splits")
    table = inventory.loc[inventory.status.eq("ok")].merge(
        approved, on="subject_id_candidate", how="inner", validate="many_to_one"
    ).merge(people, on="identity", how="inner", validate="many_to_one")
    validation = np.array(sorted(people.loc[people.split.eq("validation"), "identity"]))
    np.random.default_rng(seed).shuffle(validation)
    calibration = set(validation[: len(validation) // 2])
    role = {"train": "train", "validation": "development", "test": "final"}
    table["role"] = table.split.map(role)
    table.loc[table.identity.isin(calibration), "role"] = "calibration"
    if table.role.isna().any():
        raise ValueError("Unexpected AMASS split name")
    table = table.rename(columns={"identity": "person_id", "split": "original_split"})
    root = Path(amass_root).expanduser()
    table["raw_path"] = table.relative_path.map(lambda p: str(root / p))
    table["available"] = table.raw_path.map(lambda p: Path(p).is_file())
    table["duration_s"] = (table.num_frames - 1) / table.mocap_framerate
    return table.sort_values(["role", "person_id", "relative_path"]).reset_index(drop=True)


def select_motion_rows(table: pd.DataFrame, role: str, count: int = 128,
                       duration_s: float = 3.2, seed: int = 17) -> pd.DataFrame:
    """Round-robin people before taking second motions; uses metadata only."""
    selected = table.loc[table.role.eq(role) & table.duration_s.ge(duration_s)].copy()
    selected = selected.sample(frac=1, random_state=seed)
    selected["within_person_order"] = selected.groupby("person_id").cumcount()
    return selected.sort_values("within_person_order", kind="stable").head(count).drop(
        columns="within_person_order"
    ).reset_index(drop=True)


def load_motion(row: Mapping[str, Any], start_s: float = 0, duration_s: float = 3.2,
                fps: float = 20) -> MotionParameters:
    """Read one AMASS window and resample rotations on SO(3), not Euler angles.

    Original timestamps are retained. Translation and dynamic shape use linear
    interpolation. This restoration experiment may use the declared full window.
    """
    sequence = load_amass_sequence(Path(row["raw_path"]))
    source_t = np.arange(len(sequence.poses), dtype=np.float64) / sequence.mocap_framerate
    n = int(round(duration_s * fps))
    if fps <= 0 or n < 3 or start_s < 0:
        raise ValueError("Specify positive fps and a window containing at least three frames")
    target_t = start_s + np.arange(n) / fps
    if target_t[-1] > source_t[-1] + 1e-9:
        raise ValueError(f"Requested window extends past {row['raw_path']}")
    if not np.isfinite(sequence.poses).all() or not np.isfinite(sequence.trans).all():
        raise ValueError(f"Nonfinite motion parameters in {row['raw_path']}")
    poses = np.empty((n, 156), dtype=np.float32)
    for joint in range(52):
        rotations = Rotation.from_rotvec(sequence.poses[:, 3 * joint:3 * joint + 3])
        poses[:, 3 * joint:3 * joint + 3] = Slerp(source_t, rotations)(target_t).as_rotvec()
    interp = lambda a: np.stack([np.interp(target_t, source_t, a[:, c])
                                for c in range(a.shape[1])], axis=1).astype(np.float32)
    metadata = {key: str(row[key]) for key in
                ("person_id", "role", "relative_path", "source_dataset") if key in row}
    metadata.update(source_fps=sequence.mocap_framerate, evidence_origin="amass",
                    parameter_coordinates="amass_z_up", start_s=float(start_s))
    return MotionParameters(poses, interp(sequence.trans), sequence.betas.copy(),
                            interp(sequence.dmpls), sequence.gender, float(fps),
                            target_t, metadata)


def load_gavd_manifest(manifest_dir, video_root) -> pd.DataFrame:
    """Resolve annotated GAVD clips without treating recordings as known people.

    GAVD remains an external, observational test. Clinical reference motion and
    participant identity are not supplied by these manifests.
    """
    folder = Path(manifest_dir)
    if (folder / "gavd").is_dir():
        folder = folder / "gavd"
    table = pd.read_csv(folder / "gavd_full_sequences.csv", dtype={"video_id": str})
    root = Path(video_root).expanduser()
    suffixes = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
    def resolve(video_id):
        candidates = [folder / f"{video_id}{suffix}" for folder in (root / "all", root)
                      for suffix in suffixes]
        return str(next((path for path in candidates if path.is_file()), candidates[0]))
    paths = {v: resolve(v) for v in table.video_id.unique()}
    table["video_path"] = table.video_id.map(paths)
    table["available"] = table.video_path.map(lambda p: Path(p).is_file())
    table["group_id"] = table.video_id
    table["group_unit"] = "recording_identity_not_known"
    table["role"] = "external_observational"
    return table


def tracking_noise(shape, seed: int = 17, mechanism: str = "burst",
                   amplitude_m: float = 0.06, joints=(7, 10),
                   center: float = 0.5, width: float = 0.22) -> np.ndarray:
    """Independently sampled coordinate corruption, shared across paired cases.

    Burst and oscillation target the same joint/time region as real events when
    requested. Their amplitudes must be varied and shortcut-tested by the study.
    """
    t, j, c = shape
    if c != 3 or min(joints) < 0 or max(joints) >= j:
        raise ValueError("Noise requires valid 3D joint indices")
    rng = np.random.default_rng(seed)
    phase = np.arange(t) / max(t - 1, 1)
    envelope = np.exp(-0.5 * ((phase - center) / width) ** 2)
    direction = rng.normal(size=(len(joints), 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    if mechanism == "burst":
        wave = envelope
    elif mechanism == "oscillation":
        wave = envelope * np.sin(2 * np.pi * 7 * phase + rng.uniform(0, 2 * np.pi))
    elif mechanism == "drift":
        wave = envelope * np.linspace(-1, 1, t)
    else:
        raise ValueError(f"Unknown tracking corruption: {mechanism}")
    noise = np.zeros(shape, dtype=np.float32)
    noise[:, list(joints)] = amplitude_m * wave[:, None, None] * direction[None]
    return noise


def make_observation_cases(clean_joints, event_joints, noise, metadata=None,
                           observed=None, confidence=None) -> list[dict[str, Any]]:
    """Four factorial cases plus two exactly skeleton-matched explanations.

    The exact pair always observes edited motion plus the same independent noise;
    its RGB/reference differ. Case labels and event support are evaluation targets,
    never input features. Missing coordinates are zero with an explicit mask.
    """
    clean = np.asarray(clean_joints, dtype=np.float32)
    event = np.asarray(event_joints, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if clean.shape != event.shape or noise.shape != clean.shape:
        raise ValueError("Clean, edited and noise arrays must have equal shapes")
    mask = np.ones(clean.shape[:2], dtype=bool) if observed is None else np.asarray(observed, bool)
    quality = np.ones(clean.shape[:2], dtype=np.float32) if confidence is None else np.asarray(confidence, np.float32)
    if mask.shape != clean.shape[:2] or quality.shape != mask.shape:
        raise ValueError("Observed/confidence arrays must have shape [T,J]")
    meta = dict(metadata or {})
    stem = str(meta.get("motion_id", meta.get("relative_path", "motion")))
    cases = []
    def add(fixture, is_event, has_noise, raw, suffix):
        cases.append(dict(case_id=f"{stem}:{fixture}:{suffix}", raw=np.where(mask[..., None], raw, 0).copy(),
                          truth=(event if is_event else clean).copy(), clean=clean.copy(),
                          event_reference=event.copy(), observed=mask.copy(), confidence=quality.copy(),
                          event_present=bool(is_event), noise_present=bool(has_noise), fixture=fixture,
                          metadata=meta.copy()))
    for is_event in (False, True):
        for has_noise in (False, True):
            truth = event if is_event else clean
            add("factorial", is_event, has_noise, truth + (noise if has_noise else 0),
                f"event{int(is_event)}_noise{int(has_noise)}")
    matched = event + noise
    add("matched", True, bool(np.any(noise)), matched, "real")
    add("matched", False, True, matched, "tracking_failure")
    return cases
