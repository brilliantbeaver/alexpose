"""Fail-closed contract for an optional subject-indexed external cohort.

This module only validates authorization and manifest structure.  It does not
download poses, infer identities, train a model, or create evaluation results.
An :class:`ExternalCohort` is returned only when the repository's complete
governance gate is resolved and the supplied train/test partition is demonstrably
subject-disjoint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import json

from .governance import submission_readiness


REQUIRED_COLUMNS = (
    "dataset_reference",
    "sequence_id",
    "subject_id",
    "pose_path",
    "split",
    "joint_schema",
)
SUPPORTED_JOINT_SCHEMA = "BlazePose33"
SUPPORTED_SPLITS = frozenset({"train", "validation", "test"})


class ExternalEvaluationBlocked(RuntimeError):
    """Raised when governance does not authorize an external evaluation."""


class ExternalManifestError(ValueError):
    """Raised when an external-cohort manifest violates the data contract."""


@dataclass(frozen=True)
class ExternalPoseSequence:
    """One validated sequence supplied by an external data custodian."""

    dataset_reference: str
    sequence_id: str
    subject_id: str
    pose_path: Path
    split: str
    joint_schema: str


@dataclass(frozen=True)
class ExternalCohort:
    """Authorization token for subject-disjoint evaluation on existing poses.

    Instances cannot represent a failed or partial manifest validation: construction
    is internal, and ``manifest_authorized`` is always true. This is still not an
    evaluation result; a caller that receives no instance must not run or report an
    unseen-person evaluation.
    """

    manifest_path: Path
    governance_path: Path
    dataset_reference: str
    rows: tuple[ExternalPoseSequence, ...]
    train_subject_ids: tuple[str, ...]
    validation_subject_ids: tuple[str, ...]
    test_subject_ids: tuple[str, ...]
    manifest_authorized: bool = True

    @property
    def n_sequences(self) -> int:
        return len(self.rows)

    @property
    def n_subjects(self) -> int:
        return len(
            set(self.train_subject_ids)
            | set(self.validation_subject_ids)
            | set(self.test_subject_ids)
        )


def _require_resolved_governance(governance_path: Path) -> str:
    try:
        payload = json.loads(governance_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalEvaluationBlocked(
            f"External governance record is missing or malformed: {governance_path}"
        ) from error
    if payload.get("schema") != "neurips_laterality_external_governance/v1":
        raise ExternalEvaluationBlocked(
            "External evaluation requires an external-dataset-scoped governance record; "
            "the GAVD governance record cannot authorize another dataset."
        )
    if not payload.get("dataset_reference") or payload.get("authorization_scope") != (
        "subject_disjoint_pose_evaluation"
    ):
        raise ExternalEvaluationBlocked(
            "External governance must name the dataset and explicitly scope subject-disjoint pose evaluation."
        )
    readiness = submission_readiness(payload)
    if readiness["ready"]:
        return str(payload["dataset_reference"])
    unresolved = ", ".join(readiness["unresolved"]) or "none"
    malformed = (
        ", ".join(readiness["resolved_but_missing_reference_or_date"])
        or "none"
    )
    raise ExternalEvaluationBlocked(
        "Unseen-person evaluation is disabled until external-dataset governance is fully resolved; "
        f"unresolved={unresolved}; resolved entries missing reference/date={malformed}."
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ExternalManifestError(f"External manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ExternalManifestError(f"External manifest has no header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ExternalManifestError("External manifest has duplicate column names")
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ExternalManifestError(
                f"External manifest is missing required columns: {missing}"
            )
        rows = list(reader)
    if not rows:
        raise ExternalManifestError("External manifest must contain at least one row")
    return rows


def validate_external_manifest(
    manifest_path: str | Path,
    governance_path: str | Path,
    *,
    pose_root: str | Path | None = None,
) -> ExternalCohort:
    """Validate and authorize a subject-indexed external pose manifest.

    Relative ``pose_path`` values are resolved against ``pose_root`` when given,
    otherwise against the manifest's directory.  The supplied ``subject_id`` is
    treated strictly as custodian-provided metadata: this function never derives
    or guesses identity.

    Governance is checked before the manifest is opened.  The function raises on
    every unresolved, malformed, missing, or leakage-prone state and returns no
    placeholder data or result.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    governance = Path(governance_path).expanduser().resolve()
    authorized_dataset = _require_resolved_governance(governance)

    raw_rows = _read_manifest(manifest)
    base = (
        Path(pose_root).expanduser().resolve()
        if pose_root is not None
        else manifest.parent
    )

    validated: list[ExternalPoseSequence] = []
    sequence_ids: set[str] = set()
    pose_paths: set[Path] = set()
    subject_splits: dict[str, set[str]] = {}
    observed_datasets: set[str] = set()

    for row_number, raw in enumerate(raw_rows, start=2):
        values: dict[str, str] = {}
        for column in REQUIRED_COLUMNS:
            value = raw.get(column)
            if value is None or not value.strip():
                raise ExternalManifestError(
                    f"Row {row_number} has an empty required value: {column}"
                )
            values[column] = value.strip()

        sequence_id = values["sequence_id"]
        if sequence_id in sequence_ids:
            raise ExternalManifestError(f"Duplicate sequence_id: {sequence_id}")
        sequence_ids.add(sequence_id)

        dataset_reference = values["dataset_reference"]
        observed_datasets.add(dataset_reference)
        if dataset_reference != authorized_dataset:
            raise ExternalEvaluationBlocked(
                "External manifest dataset_reference does not match its governance record"
            )

        subject_id = values["subject_id"]
        split = values["split"].lower()
        if split not in SUPPORTED_SPLITS:
            raise ExternalManifestError(
                f"Row {row_number} has unsupported split {values['split']!r}; "
                f"expected one of {sorted(SUPPORTED_SPLITS)}"
            )
        subject_splits.setdefault(subject_id, set()).add(split)

        joint_schema = values["joint_schema"]
        if joint_schema != SUPPORTED_JOINT_SCHEMA:
            raise ExternalManifestError(
                f"Row {row_number} uses {joint_schema!r}; only "
                f"{SUPPORTED_JOINT_SCHEMA!r} is supported"
            )

        supplied_pose_path = Path(values["pose_path"]).expanduser()
        resolved_pose_path = (
            supplied_pose_path.resolve()
            if supplied_pose_path.is_absolute()
            else (base / supplied_pose_path).resolve()
        )
        if not resolved_pose_path.is_file():
            raise ExternalManifestError(
                f"Row {row_number} pose_path is not an existing file: "
                f"{resolved_pose_path}"
            )
        if not resolved_pose_path.is_relative_to(base):
            raise ExternalManifestError(
                f"Row {row_number} pose_path escapes the declared pose root: {resolved_pose_path}"
            )
        if resolved_pose_path in pose_paths:
            raise ExternalManifestError(
                f"The same pose file appears more than once: {resolved_pose_path}"
            )
        pose_paths.add(resolved_pose_path)

        validated.append(
            ExternalPoseSequence(
                dataset_reference=dataset_reference,
                sequence_id=sequence_id,
                subject_id=subject_id,
                pose_path=resolved_pose_path,
                split=split,
                joint_schema=joint_schema,
            )
        )

    leaking_subjects = sorted(
        subject_id
        for subject_id, splits in subject_splits.items()
        if len(splits) > 1
    )
    if leaking_subjects:
        raise ExternalManifestError(
            "Subjects must belong to exactly one partition; overlapping subject_id "
            f"values: {leaking_subjects[:10]}"
        )
    if observed_datasets != {authorized_dataset}:
        raise ExternalManifestError("External manifest must name exactly one governed dataset")

    subjects_by_split = {
        split: tuple(
            sorted(
                subject_id
                for subject_id, splits in subject_splits.items()
                if split in splits
            )
        )
        for split in SUPPORTED_SPLITS
    }
    if not subjects_by_split["train"] or not subjects_by_split["test"]:
        raise ExternalManifestError(
            "External manifest must contain non-empty train and test subject partitions"
        )

    # Keep the train/test assertion explicit even though the stronger one-split-per-
    # subject rule above also covers an optional validation partition.
    if set(subjects_by_split["train"]) & set(subjects_by_split["test"]):
        raise ExternalManifestError("External train and test subjects overlap")

    return ExternalCohort(
        manifest_path=manifest,
        governance_path=governance,
        dataset_reference=authorized_dataset,
        rows=tuple(validated),
        train_subject_ids=subjects_by_split["train"],
        validation_subject_ids=subjects_by_split["validation"],
        test_subject_ids=subjects_by_split["test"],
    )


__all__ = [
    "ExternalCohort",
    "ExternalEvaluationBlocked",
    "ExternalManifestError",
    "ExternalPoseSequence",
    "REQUIRED_COLUMNS",
    "SUPPORTED_JOINT_SCHEMA",
    "validate_external_manifest",
]
