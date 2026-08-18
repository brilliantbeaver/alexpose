"""Validate and migrate legacy augmented-pose artifacts into the active root.

Older extraction code wrote the augmentation cohort below
``gavd5/cache/artifacts/real`` even when notebook 04 was configured to read a
different ``GAVD_ARTIFACT_DIR``. This utility migrates only the archives selected
by the legacy extraction report. It is deliberately guarded:

* it runs only in ``GAVD_MODE=real``;
* it is a dry run unless ``--apply`` is supplied;
* the known legacy cohort must contain exactly 63 eligible archives by default;
* every archive is checked against notebook 04's input contract before copy;
* existing destination files must be byte-identical; nothing is overwritten or
  deleted; and
* copied files and the report are revalidated and hash-compared after migration.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


TUTORIAL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TUTORIAL_DIR.parents[2]
load_dotenv(TUTORIAL_DIR / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)

MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")

ARTIFACT_ROOT = Path(
    os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")
).expanduser()
LEGACY_ARTIFACT_ROOT = TUTORIAL_DIR / "cache" / "artifacts"
MIN_NEURO_OBSERVED = 0.45
EXPECTED_LEGACY_ELIGIBLE_COUNT = 63
REQUIRED_KEYS = {
    "sequence", "sequence_id", "video_id", "condition", "frame_numbers",
    "crop_bounds", "fps", "source_csv", "source_video", "pose_model",
    "pose_model_sha256", "extraction_version", "cohort", "bbox_source", "clip_id",
}
COMPATIBLE_EXTRACTION_VERSIONS = {
    "gavd3_pose_v2_video_mode",
    "gavd4_pose_v2_video_mode",
    "gavd5_pose_v2_video_mode",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path, sequence_id: str) -> None:
    """Check the archive can be consumed by notebook 04 before it is copied."""
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
        sequence = data["sequence"]
        if sequence.ndim != 3 or sequence.shape[0] == 0 or sequence.shape[1:] != (33, 4):
            raise ValueError(f"{path} has invalid sequence shape {sequence.shape}")
        if len(data["frame_numbers"]) != len(sequence):
            raise ValueError(f"{path} has mismatched frame_numbers and sequence lengths")
        if data["crop_bounds"].shape != (len(sequence), 4):
            raise ValueError(f"{path} has invalid crop_bounds shape {data['crop_bounds'].shape}")
        if str(data["sequence_id"].item()) != sequence_id:
            raise ValueError(f"{path} sequence_id does not match its filename/report ID")
        if str(data["condition"].item()) != "normal":
            raise ValueError(f"{path} is not a normal-condition augmentation archive")
        if str(data["cohort"].item()) != "augmented_normal":
            raise ValueError(f"{path} has unexpected cohort metadata")
        if str(data["extraction_version"].item()) not in COMPATIBLE_EXTRACTION_VERSIONS:
            raise ValueError(f"{path} has an unsupported extraction version")
        fps = float(data["fps"].item())
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{path} has invalid fps {fps}")


def selected_pose_files(
    source_root: Path, expected_count: int
) -> tuple[pd.DataFrame, dict[str, Path]]:
    report_path = source_root / "augmented_pose_extraction_report.csv"
    pose_dir = source_root / "poses_augmented" / "normal"
    if not report_path.is_file() or not pose_dir.is_dir():
        raise FileNotFoundError(
            "Legacy cohort is incomplete; expected both "
            f"{report_path} and {pose_dir}."
        )
    report = pd.read_csv(report_path)
    required_columns = {"sequence_id", "status", "neuro_observed"}
    missing_columns = required_columns.difference(report.columns)
    if missing_columns:
        raise ValueError(f"Legacy report missing columns: {sorted(missing_columns)}")
    if report["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Legacy report contains duplicate sequence IDs")

    eligible = report.loc[
        ~report["status"].astype(str).str.startswith("error")
        & (pd.to_numeric(report["neuro_observed"], errors="coerce") >= MIN_NEURO_OBSERVED)
    ].copy()
    if len(eligible) != expected_count:
        raise ValueError(
            f"Legacy report has {len(eligible)} eligible archives; expected {expected_count}. "
            "Refusing a partial or unexpected migration."
        )

    available = {path.stem: path for path in pose_dir.glob("*.npz")}
    selected_ids = set(eligible["sequence_id"].astype(str))
    missing_files = sorted(selected_ids.difference(available))
    unexpected_files = sorted(set(available).difference(selected_ids))
    if missing_files or unexpected_files:
        raise ValueError(
            "Legacy pose cache does not match its selection report: "
            f"missing eligible IDs={missing_files[:8]}; unexpected files={unexpected_files[:8]}"
        )
    selected = {sequence_id: available[sequence_id] for sequence_id in sorted(selected_ids)}
    for sequence_id, path in selected.items():
        validate_archive(path, sequence_id)
    return report, selected


def assert_destination_compatible(
    destination_root: Path, source_root: Path, selected: dict[str, Path]
) -> None:
    """Reject mixed or conflicting destination state before any copy occurs."""
    pose_dir = destination_root / "poses_augmented" / "normal"
    if pose_dir.exists():
        existing = {path.stem: path for path in pose_dir.glob("*.npz")}
        unexpected = sorted(set(existing).difference(selected))
        if unexpected:
            raise FileExistsError(
                f"Destination contains unrecognized augmented archives: {unexpected[:8]}"
            )
        for sequence_id, destination in existing.items():
            if sha256(selected[sequence_id]) != sha256(destination):
                raise FileExistsError(
                    f"Refusing to overwrite different destination artifact: {destination}"
                )

    source_report = source_root / "augmented_pose_extraction_report.csv"
    destination_report = destination_root / source_report.name
    if destination_report.exists() and sha256(source_report) != sha256(destination_report):
        raise FileExistsError(
            f"Refusing to overwrite different destination report: {destination_report}"
        )


def migrate(
    source_root: Path, destination_root: Path, expected_count: int, apply: bool
) -> None:
    if MODE != "real":
        raise RuntimeError("Migration is real-data only. Set GAVD_MODE=real and restart.")
    source_root = source_root.expanduser().resolve()
    destination_root = destination_root.expanduser().resolve()
    if source_root == destination_root:
        raise ValueError("Source and destination are the same; nothing to migrate")

    report, selected = selected_pose_files(source_root, expected_count)
    assert_destination_compatible(destination_root, source_root, selected)
    destination_pose_dir = destination_root / "poses_augmented" / "normal"
    pending = [
        sequence_id for sequence_id, source in selected.items()
        if not (destination_pose_dir / source.name).exists()
    ]
    source_report = source_root / "augmented_pose_extraction_report.csv"
    destination_report = destination_root / source_report.name

    print(f"Validated {len(selected)} eligible legacy archives from {source_root}.")
    print(f"Destination: {destination_root}")
    print(f"Archives to copy: {len(pending)}; report to copy: {not destination_report.exists()}")
    if not apply:
        print("Dry run only. Re-run with --apply to copy; no files were changed.")
        return

    destination_pose_dir.mkdir(parents=True, exist_ok=True)
    for sequence_id in pending:
        source = selected[sequence_id]
        shutil.copy2(source, destination_pose_dir / source.name)
    if not destination_report.exists():
        shutil.copy2(source_report, destination_report)

    _, migrated = selected_pose_files(destination_root, expected_count)
    if set(migrated) != set(selected) or any(
        sha256(selected[sequence_id]) != sha256(migrated[sequence_id])
        for sequence_id in selected
    ):
        raise RuntimeError("Post-copy validation failed; source artifacts were preserved")
    if sha256(source_report) != sha256(destination_report):
        raise RuntimeError("Post-copy report validation failed; source artifacts were preserved")
    print(f"Migrated and validated {len(selected)} augmented-normal pose files.")
    print(f"Pose directory: {destination_pose_dir}")
    print(f"Selection report: {destination_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=LEGACY_ARTIFACT_ROOT / "real",
        help="legacy real-mode artifact root (default: gavd5/cache/artifacts/real)",
    )
    parser.add_argument(
        "--destination-root", type=Path, default=ARTIFACT_ROOT / "real",
        help="active real-mode artifact root (default: GAVD_ARTIFACT_DIR/real)",
    )
    parser.add_argument(
        "--expected-count", type=int, default=EXPECTED_LEGACY_ELIGIBLE_COUNT,
        help="required eligible archive count (default: 63)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="perform the copy after validation; without this flag the command is a dry run",
    )
    args = parser.parse_args()
    if args.expected_count <= 0:
        parser.error("--expected-count must be positive")
    migrate(args.source_root, args.destination_root, args.expected_count, args.apply)


if __name__ == "__main__":
    main()
