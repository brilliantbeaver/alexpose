from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import atomic_write_json, initialize_artifact_root, sha256_file
from .config import ExperimentContext, model_config
from .geometry import anatomical_mirror, missingness_feature, paired_valid_target, prepare_pose


@dataclass(frozen=True)
class PoseRecord:
    sequence_id: str
    video_id: str
    condition: str
    sequence: np.ndarray
    frame_numbers: np.ndarray
    fps: float
    archive_sha256: str
    extraction_version: str


@dataclass(frozen=True)
class PreparedCohort:
    table: pd.DataFrame
    model_xyz: np.ndarray
    model_valid: np.ndarray
    pair_contrasts: np.ndarray
    missingness: np.ndarray
    cohort_digest: str
    attrition: dict[str, Any]

    @property
    def targets(self) -> np.ndarray:
        return self.table["target"].to_numpy(dtype=np.float64)


def _scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise ValueError(f"Pose archive is missing {key!r}")
    value = archive[key]
    if value.ndim != 0:
        raise ValueError(f"Pose archive field {key!r} must be scalar")
    return value.item()


def load_real_pose_records(
    pose_root: Path,
    annotation_root: Path,
    conditions: Iterable[str],
    data_config: dict[str, Any],
) -> tuple[list[PoseRecord], dict[str, Any]]:
    condition_order = list(map(str, conditions))
    contract = data_config["inventory_contract"]
    annotation_inventory = [
        f"{condition}/{path.name}"
        for condition in condition_order
        for path in sorted((annotation_root / condition).glob("*.csv"))
    ]
    pose_paths = [
        (condition, path)
        for condition in condition_order
        for path in sorted((pose_root / condition).glob("*.npz"))
    ]
    pose_inventory = [f"{condition}/{path.stem}" for condition, path in pose_paths]

    annotation_sources: dict[str, str] = {}
    mapping_lines: list[str] = []
    for condition in condition_order:
        for path in sorted((annotation_root / condition).glob("*.csv")):
            annotation = pd.read_csv(path, usecols=["seq", "id", "gait_pat"])
            values: dict[str, str] = {}
            for column in ("seq", "id", "gait_pat"):
                unique = annotation[column].dropna().astype(str).unique()
                if len(unique) != 1:
                    raise RuntimeError(
                        f"Official annotation {path} has non-constant {column!r}"
                    )
                values[column] = str(unique[0])
            normalized_annotation = "".join(
                character
                for character in values["gait_pat"].lower()
                if character.isalnum()
            )
            if values["seq"] != path.stem or normalized_annotation != condition:
                raise RuntimeError(f"Official annotation identity mismatch in {path}")
            key = f"{condition}/{path.stem}"
            annotation_sources[key] = values["id"]
            mapping_lines.append(
                f"{key}\t{values['id']}\t{normalized_annotation}"
            )

    def inventory_digest(values: list[str]) -> str:
        return hashlib.sha256(
            ("\n".join(sorted(values)) + "\n").encode("utf-8")
        ).hexdigest()

    observed_contract = {
        "annotation_count": len(annotation_inventory),
        "annotation_inventory_sha256": inventory_digest(annotation_inventory),
        "pose_archive_count": len(pose_inventory),
        "pose_inventory_sha256": inventory_digest(pose_inventory),
        "annotation_source_mapping_sha256": hashlib.sha256(
            ("\n".join(sorted(mapping_lines)) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    for key, value in observed_contract.items():
        if value != contract[key]:
            raise RuntimeError(
                f"Locked real-data inventory mismatch for {key}: {value!r} != {contract[key]!r}"
            )
    annotation_keys = {item.removesuffix(".csv") for item in annotation_inventory}
    unexpected = sorted(set(pose_inventory) - annotation_keys)
    if unexpected:
        raise RuntimeError("Pose cache contains archives outside the annotation inventory")

    records: list[PoseRecord] = []
    seen_sequences: set[str] = set()
    extraction_versions: Counter[str] = Counter()
    provenance = data_config["extraction_provenance"]
    for condition in condition_order:
        if not (pose_root / condition).is_dir():
            raise FileNotFoundError(
                f"Missing pose-cache condition directory: {pose_root / condition}"
            )
        for _, path in [item for item in pose_paths if item[0] == condition]:
            with np.load(path, allow_pickle=False) as archive:
                sequence = np.asarray(archive["sequence"], dtype=np.float32)
                frame_numbers = np.asarray(archive["frame_numbers"], dtype=np.int64)
                sequence_id = str(_scalar(archive, "sequence_id"))
                video_id = str(_scalar(archive, "video_id"))
                stored_condition = str(_scalar(archive, "condition"))
                fps = float(_scalar(archive, "fps"))
                source_csv = str(_scalar(archive, "source_csv"))
                extraction_version = str(_scalar(archive, "extraction_version"))
                pose_model = str(_scalar(archive, "pose_model"))
                pose_model_sha256 = str(_scalar(archive, "pose_model_sha256"))
                stored_visibility_threshold = float(
                    _scalar(archive, "visibility_threshold")
                )
            if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
                raise ValueError(f"Bad pose shape in {path}: {sequence.shape}")
            if frame_numbers.shape != (len(sequence),) or np.any(np.diff(frame_numbers) <= 0):
                raise ValueError(f"Non-increasing or malformed frame numbers in {path}")
            if stored_condition != condition:
                raise ValueError(f"Folder/archive condition mismatch in {path}")
            expected_video_id = annotation_sources[f"{condition}/{path.stem}"]
            if video_id != expected_video_id:
                raise ValueError(
                    f"Archive video_id disagrees with official annotation id in {path}"
                )
            if sequence_id != path.stem or Path(source_csv).name != f"{path.stem}.csv":
                raise ValueError(f"Archive/annotation sequence provenance mismatch in {path}")
            if Path(source_csv).parent.name != condition:
                raise ValueError(f"Archive source_csv condition mismatch in {path}")
            if pose_model != provenance["pose_model"]:
                raise ValueError(f"Unexpected pose model in {path}")
            if pose_model_sha256 != provenance["pose_model_sha256"]:
                raise ValueError(f"Unexpected pose model digest in {path}")
            if not np.isclose(
                stored_visibility_threshold,
                float(provenance["visibility_threshold"]),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(f"Unexpected extraction visibility threshold in {path}")
            if sequence_id in seen_sequences:
                raise ValueError(f"Duplicate sequence_id {sequence_id}")
            seen_sequences.add(sequence_id)
            extraction_versions[extraction_version] += 1
            records.append(
                PoseRecord(
                    sequence_id=sequence_id,
                    video_id=video_id,
                    condition=condition,
                    sequence=sequence,
                    frame_numbers=frame_numbers,
                    fps=fps,
                    archive_sha256=sha256_file(path),
                    extraction_version=extraction_version,
                )
            )
    if not records:
        raise RuntimeError(f"No pose archives found under {pose_root}")
    expected_versions = {
        str(key): int(value)
        for key, value in provenance["extraction_version_counts"].items()
    }
    if dict(sorted(extraction_versions.items())) != dict(sorted(expected_versions.items())):
        raise RuntimeError(
            "Pose extraction-version census differs from the locked provenance contract"
        )
    audit = {
        **observed_contract,
        "annotations_without_pose_archive": len(annotation_keys - set(pose_inventory)),
        "official_source_videos": len(set(annotation_sources.values())),
        "unexpected_pose_archives": len(unexpected),
        "extraction_version_counts": dict(sorted(extraction_versions.items())),
        "pose_model": provenance["pose_model"],
        "pose_model_sha256": provenance["pose_model_sha256"],
    }
    return records, audit


def _base_skeleton() -> np.ndarray:
    base = np.zeros((33, 3), dtype=np.float64)
    for left, right, height, width in (
        (11, 12, 0.70, 0.34),
        (13, 14, 0.45, 0.42),
        (15, 16, 0.22, 0.47),
        (17, 18, 0.18, 0.50),
        (19, 20, 0.18, 0.48),
        (21, 22, 0.18, 0.46),
        (23, 24, 0.00, 0.24),
        (25, 26, -0.43, 0.25),
        (27, 28, -0.86, 0.26),
        (29, 30, -0.91, 0.28),
        (31, 32, -0.96, 0.30),
    ):
        base[left] = (-width / 2, height, 0.0)
        base[right] = (width / 2, height, 0.0)
    return base


def synthetic_pose_records(
    conditions: Iterable[str],
    *,
    seed: int = 20260904,
) -> list[PoseRecord]:
    """Deterministic non-evidentiary data used only to exercise pipeline mechanics."""
    rng = np.random.default_rng(seed)
    base = _base_skeleton()
    records: list[PoseRecord] = []
    bilateral = ((11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))
    for condition_index, condition in enumerate(conditions):
        for source_index in range(5):
            video_id = f"smoke-{condition_index:02d}-{source_index:02d}"
            latent = float(rng.uniform(-0.38, 0.38))
            sequence_count = 2 + source_index % 3
            for clip_index in range(sequence_count):
                frames = 28 + clip_index
                phase = np.linspace(0.0, 4.0 * np.pi, frames, endpoint=False)
                sequence = np.repeat(base[None, :, :], frames, axis=0)
                sequence += rng.normal(0.0, 0.0015, size=sequence.shape)
                for pair_index, (left, right) in enumerate(bilateral):
                    wave = np.sin(phase + 0.23 * pair_index + 0.11 * clip_index)
                    left_amplitude = 0.025 * (1.0 + latent)
                    right_amplitude = 0.025 * (1.0 - latent)
                    sequence[:, left, 2] += left_amplitude * wave
                    sequence[:, right, 2] += right_amplitude * wave
                visibility = np.full((frames, 33, 1), 0.99, dtype=np.float64)
                # Deterministic short missing stretches test the separate validity lane.
                if (source_index + clip_index) % 3 == 0:
                    visibility[4:6, 15, 0] = 0.1
                raw = np.concatenate((sequence, visibility), axis=-1).astype(np.float32)
                sequence_id = f"{video_id}-clip-{clip_index:02d}"
                records.append(
                    PoseRecord(
                        sequence_id=sequence_id,
                        video_id=video_id,
                        condition=str(condition),
                        sequence=raw,
                        frame_numbers=np.arange(frames, dtype=np.int64) * 2,
                        fps=30.0,
                        archive_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        extraction_version="synthetic_smoke_v1",
                    )
                )
    return records


def _cohort_digest(
    table: pd.DataFrame,
    model_xyz: np.ndarray,
    model_valid: np.ndarray,
    pair_contrasts: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    stable = table.sort_values("sequence_id").reset_index(drop=True)
    # Float CSV parsing is not byte-preserving across pandas versions. Target and
    # coverage are already committed by pair_contrasts/model_valid below.
    stable_columns = [
        "sequence_id",
        "video_id",
        "condition",
        "usable_pair_count",
        "authorized_patch_count",
        "raw_frame_count",
        "archive_sha256",
        "extraction_version",
    ]
    digest.update(stable[stable_columns].to_csv(index=False).encode("utf-8"))
    order = table["sequence_id"].argsort(kind="stable").to_numpy()
    for values in (model_xyz[order], model_valid[order], pair_contrasts[order]):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def prepare_cohort(context: ExperimentContext) -> PreparedCohort:
    protocol = context.protocol
    data_config = protocol["data"]
    target_config = protocol["target"]
    conditions = tuple(data_config["conditions"])
    if context.is_paper:
        records, inventory_audit = load_real_pose_records(
            context.pose_root,
            context.annotation_root,
            conditions,
            data_config,
        )
    else:
        records = synthetic_pose_records(conditions)
        inventory_audit = {
            "synthetic": True,
            "annotation_count": 0,
            "pose_archive_count": len(records),
            "annotations_without_pose_archive": 0,
        }
    frames = int(model_config(context)["frames"])
    segment_length = int(model_config(context)["segment_length"])
    authorized = np.asarray(protocol["model"]["authorized_target_joints"], dtype=int)

    rows: list[dict[str, Any]] = []
    accepted_xyz: list[np.ndarray] = []
    accepted_valid: list[np.ndarray] = []
    accepted_contrasts: list[np.ndarray] = []
    accepted_missingness: list[np.ndarray] = []
    excluded: list[dict[str, str]] = []
    target_contract_checked = 0
    maximum_mirror_error = 0.0
    maximum_invalid_sentinel_error = 0.0
    for record in records:
        try:
            prepared = prepare_pose(
                record.sequence,
                record.frame_numbers,
                record.fps,
                frames=frames,
                visibility_threshold=float(data_config["visibility_threshold"]),
                max_interpolation_gap=int(data_config["max_interpolation_gap"]),
            )
            target = paired_valid_target(
                prepared.target_xyz,
                prepared.target_valid,
                target_config["pairs"],
                int(target_config["minimum_common_transitions_per_pair"]),
                int(target_config["minimum_usable_pairs"]),
                float(target_config["epsilon"]),
                prepared.frame_times,
            )
            if np.isfinite(target.value):
                mirrored_xyz, mirrored_valid = anatomical_mirror(
                    prepared.target_xyz, prepared.target_valid
                )
                mirrored_target = paired_valid_target(
                    mirrored_xyz,
                    mirrored_valid,
                    target_config["pairs"],
                    int(target_config["minimum_common_transitions_per_pair"]),
                    int(target_config["minimum_usable_pairs"]),
                    float(target_config["epsilon"]),
                    prepared.frame_times,
                )
                sentinel_xyz = prepared.target_xyz.copy()
                sentinel_xyz[~prepared.target_valid] = np.asarray(
                    [1e12, -7e11, 3e11]
                )
                sentinel_target = paired_valid_target(
                    sentinel_xyz,
                    prepared.target_valid,
                    target_config["pairs"],
                    int(target_config["minimum_common_transitions_per_pair"]),
                    int(target_config["minimum_usable_pairs"]),
                    float(target_config["epsilon"]),
                    prepared.frame_times,
                )
                mirror_error = abs(target.value + mirrored_target.value)
                sentinel_error = abs(target.value - sentinel_target.value)
                if mirror_error > 1e-10 or sentinel_error > 1e-12:
                    raise AssertionError(
                        f"Target contract failed for {record.sequence_id}: "
                        f"mirror={mirror_error}, sentinel={sentinel_error}"
                    )
                maximum_mirror_error = max(maximum_mirror_error, mirror_error)
                maximum_invalid_sentinel_error = max(
                    maximum_invalid_sentinel_error, sentinel_error
                )
                target_contract_checked += 1
            coverage = float(prepared.model_valid[:, authorized].mean())
            patch_valid = prepared.model_valid.reshape(
                frames // segment_length, segment_length, 33
            ).all(axis=1)
            authorized_patches = int(patch_valid[:, authorized].sum())
            reasons: list[str] = []
            if not np.isfinite(target.value):
                reasons.append("target_not_computable")
            if coverage < float(data_config["minimum_authorized_coverage"]):
                reasons.append("insufficient_authorized_coverage")
            if authorized_patches < int(data_config["minimum_authorized_patches"]):
                reasons.append("insufficient_authorized_patches")
            if reasons:
                excluded.append(
                    {"sequence_id": record.sequence_id, "reason": ";".join(reasons)}
                )
                continue
            rows.append(
                {
                    "sequence_id": record.sequence_id,
                    "video_id": record.video_id,
                    "condition": record.condition,
                    "target": target.value,
                    "usable_pair_count": target.usable_pair_count,
                    "authorized_coverage": coverage,
                    "authorized_patch_count": authorized_patches,
                    "raw_frame_count": len(record.sequence),
                    "archive_sha256": record.archive_sha256,
                    "extraction_version": record.extraction_version,
                }
            )
            accepted_xyz.append(prepared.model_xyz)
            accepted_valid.append(prepared.model_valid)
            accepted_contrasts.append(np.asarray(target.pair_contrasts, dtype=np.float64))
            accepted_missingness.append(
                missingness_feature(prepared.target_valid, target_config["pairs"])
            )
        except (ValueError, FloatingPointError) as error:
            excluded.append({"sequence_id": record.sequence_id, "reason": f"invalid:{error}"})

    if not rows:
        raise RuntimeError("No pose sequences passed the predeclared QC rules")
    table = pd.DataFrame(rows)
    if table["sequence_id"].duplicated().any():
        raise AssertionError("Accepted cohort has duplicate sequence IDs")
    per_source_conditions = table.groupby("video_id")["condition"].nunique()
    if (per_source_conditions != 1).any():
        raise ValueError("A source video maps to more than one dataset annotation")

    xyz_array = np.stack(accepted_xyz).astype(np.float32)
    valid_array = np.stack(accepted_valid).astype(bool)
    contrast_array = np.stack(accepted_contrasts).astype(np.float64)
    missingness_array = np.stack(accepted_missingness).astype(np.float64)
    attrition = {
        "profile": context.profile,
        "input_sequences": len(records),
        "accepted_sequences": len(table),
        "accepted_sources": int(table["video_id"].nunique()),
        "excluded_sequences": len(excluded),
        "exclusions": excluded,
        "synthetic_evidence": not context.is_paper,
        "inventory": inventory_audit,
        "target_contract": {
            "checked_finite_targets": target_contract_checked,
            "maximum_mirror_antisymmetry_error": maximum_mirror_error,
            "maximum_invalid_sentinel_error": maximum_invalid_sentinel_error,
        },
    }
    digest = _cohort_digest(table, xyz_array, valid_array, contrast_array)
    return PreparedCohort(
        table=table,
        model_xyz=xyz_array,
        model_valid=valid_array,
        pair_contrasts=contrast_array,
        missingness=missingness_array,
        cohort_digest=digest,
        attrition=attrition,
    )


def cohort_paths(artifact_root: Path) -> tuple[Path, Path, Path]:
    return (
        artifact_root / "cohort" / "cohort.npz",
        artifact_root / "cohort" / "manifest.csv",
        artifact_root / "cohort" / "metadata.json",
    )


def save_cohort(context: ExperimentContext, cohort: PreparedCohort) -> dict[str, Path]:
    initialize_artifact_root(
        context.artifact_root,
        context.protocol,
        context.protocol_digest,
        context.context_digest,
        context.profile,
    )
    array_path, table_path, metadata_path = cohort_paths(context.artifact_root)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cohort.", suffix=".npz", dir=array_path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary_name,
            model_xyz=cohort.model_xyz,
            model_valid=cohort.model_valid,
            pair_contrasts=cohort.pair_contrasts,
            missingness=cohort.missingness,
            sequence_ids=cohort.table["sequence_id"].to_numpy(dtype=str),
        )
        os.replace(temporary_name, array_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    cohort.table.to_csv(table_path, index=False)
    atomic_write_json(
        metadata_path,
        {
            "schema": "neurips_laterality_cohort/v2",
            "protocol_digest": context.protocol_digest,
            "context_digest": context.context_digest,
            "profile": context.profile,
            "cohort_digest": cohort.cohort_digest,
            "array_sha256": sha256_file(array_path),
            "manifest_sha256": sha256_file(table_path),
            "attrition": cohort.attrition,
        },
    )
    return {"arrays": array_path, "manifest": table_path, "metadata": metadata_path}


def load_cohort(context: ExperimentContext) -> PreparedCohort:
    array_path, table_path, metadata_path = cohort_paths(context.artifact_root)
    import json

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "neurips_laterality_cohort/v2":
        raise RuntimeError("Unsupported cohort artifact")
    if metadata.get("protocol_digest") != context.protocol_digest:
        raise RuntimeError("Cohort protocol digest mismatch; rebuild it")
    if metadata.get("context_digest") != context.context_digest:
        raise RuntimeError("Cohort profile/effective-context mismatch; rebuild it")
    if metadata.get("profile") != context.profile:
        raise RuntimeError("Cohort profile mismatch")
    if metadata.get("array_sha256") != sha256_file(array_path):
        raise RuntimeError("Cohort array artifact is corrupt or stale")
    if metadata.get("manifest_sha256") != sha256_file(table_path):
        raise RuntimeError("Cohort manifest artifact is corrupt or stale")
    table = pd.read_csv(table_path)
    with np.load(array_path, allow_pickle=False) as archive:
        sequence_ids = archive["sequence_ids"].astype(str)
        if list(sequence_ids) != list(table["sequence_id"].astype(str)):
            raise RuntimeError("Cohort table/array row order mismatch")
        cohort = PreparedCohort(
            table=table,
            model_xyz=archive["model_xyz"].astype(np.float32),
            model_valid=archive["model_valid"].astype(bool),
            pair_contrasts=archive["pair_contrasts"].astype(np.float64),
            missingness=archive["missingness"].astype(np.float64),
            cohort_digest=str(metadata["cohort_digest"]),
            attrition=dict(metadata["attrition"]),
        )
    if cohort.attrition.get("profile") != context.profile:
        raise RuntimeError("Cohort attrition profile is inconsistent")
    if bool(cohort.attrition.get("synthetic_evidence")) != (not context.is_paper):
        raise RuntimeError("Cohort synthetic/real evidence marker is inconsistent")
    if _cohort_digest(cohort.table, cohort.model_xyz, cohort.model_valid, cohort.pair_contrasts) != cohort.cohort_digest:
        raise RuntimeError("Cohort content digest mismatch")
    return cohort
