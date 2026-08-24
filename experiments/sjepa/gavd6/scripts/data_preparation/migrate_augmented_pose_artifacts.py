"""Validate and migrate legacy augmented-pose artifacts into the active root.

Older versions of ``extract_augmented_poses.py`` wrote augmented artifacts under
``cache/artifacts/real``. Notebook 04 now correctly consumes them from
``GAVD_ARTIFACT_DIR/real``. This utility is intentionally explicit: it copies
only the report-selected, neurologically eligible pose files after validating the
report/file contract. It never deletes or overwrites non-identical artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


TUTORIAL_DIR = Path(__file__).resolve().parents[2]
load_dotenv(TUTORIAL_DIR / ".env", override=False)
MIN_NEURO_OBSERVED = 0.45


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_pose_files(source_root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    report_path = source_root / "augmented_pose_extraction_report.csv"
    pose_dir = source_root / "poses_augmented" / "normal"
    if not report_path.is_file() or not pose_dir.is_dir():
        raise FileNotFoundError(
            "Legacy cohort is incomplete; expected both "
            f"{report_path} and {pose_dir}."
        )
    report = pd.read_csv(report_path)
    required = {"sequence_id", "status", "neuro_observed"}
    missing = required.difference(report.columns)
    if missing:
        raise ValueError(f"Legacy report missing columns: {sorted(missing)}")
    if report["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Legacy report contains duplicate sequence IDs")
    eligible = report.loc[
        ~report["status"].astype(str).str.startswith("error")
        & (pd.to_numeric(report["neuro_observed"], errors="coerce") >= MIN_NEURO_OBSERVED)
    ].copy()
    if eligible.empty:
        raise ValueError("Legacy report contains no neurologically eligible sequences")
    available = {path.stem: path for path in pose_dir.glob("*.npz")}
    selected_ids = set(eligible["sequence_id"].astype(str))
    missing_files = sorted(selected_ids.difference(available))
    unexpected_files = sorted(set(available).difference(selected_ids))
    if missing_files or unexpected_files:
        raise ValueError(
            "Legacy pose cache does not match its selection report: "
            f"missing eligible IDs={missing_files[:8]}; "
            f"unexpected files={unexpected_files[:8]}"
        )
    return report, {sequence_id: available[sequence_id] for sequence_id in sorted(selected_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=TUTORIAL_DIR / "cache" / "artifacts" / "real",
        help="legacy real-mode artifact root (default: cache/artifacts/real)",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path(
            os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")
        ).expanduser()
        / "real",
        help="active real-mode artifact root (default: GAVD_ARTIFACT_DIR/real)",
    )
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    destination_root = args.destination_root.expanduser().resolve()
    if source_root == destination_root:
        raise ValueError("Source and destination are the same; nothing to migrate")

    report, selected = selected_pose_files(source_root)
    destination_pose_dir = destination_root / "poses_augmented" / "normal"
    destination_pose_dir.mkdir(parents=True, exist_ok=True)
    for sequence_id, source in selected.items():
        destination = destination_pose_dir / source.name
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise FileExistsError(
                    f"Refusing to overwrite different destination artifact: {destination}"
                )
            continue
        shutil.copy2(source, destination)

    destination_report = destination_root / "augmented_pose_extraction_report.csv"
    if destination_report.exists() and sha256(source_root / destination_report.name) != sha256(destination_report):
        raise FileExistsError(
            f"Refusing to overwrite different destination report: {destination_report}"
        )
    if not destination_report.exists():
        shutil.copy2(source_root / destination_report.name, destination_report)

    # Re-run the selection check on destination and ensure every copy is byte-identical.
    _, migrated = selected_pose_files(destination_root)
    if set(migrated) != set(selected) or any(
        sha256(selected[sequence_id]) != sha256(migrated[sequence_id])
        for sequence_id in selected
    ):
        raise RuntimeError("Post-copy validation failed; source artifacts were preserved")
    print(
        f"Migrated and validated {len(selected)} augmented-normal pose files to "
        f"{destination_root} (report rows preserved: {len(report)})."
    )


if __name__ == "__main__":
    main()
