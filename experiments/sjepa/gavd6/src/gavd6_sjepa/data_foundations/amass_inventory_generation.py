"""Generate the raw AMASS inventory used by Core11 conversion."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


INVENTORY_FIELDS = [
    "relative_path", "source_dataset", "parent_path", "subject_id_candidate",
    "motion_id", "sha256", "npz_keys", "num_frames", "pose_width",
    "trans_frames", "trans_width", "mocap_framerate", "gender", "status",
    "error",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(archive, key: str) -> str:
    if key not in archive.files:
        return ""
    value = archive[key]
    if value.size != 1:
        return str(value.shape)
    item = value.item()
    return item.decode() if isinstance(item, bytes) else str(item)


def generate_inventory(amass_root: Path, run_root: Path) -> tuple[Path, Path, int, int]:
    """Inspect AMASS archives and write the inventory plus reject report."""

    amass_root = Path(amass_root).resolve()
    output_directory = Path(run_root).resolve() / "manifests"
    output_directory.mkdir(parents=True, exist_ok=True)
    inventory_path = output_directory / "amass_raw_inventory.csv"
    rejects_path = output_directory / "amass_inventory_rejects.csv"
    paths = sorted(amass_root.rglob("*_poses.npz"))
    rows: list[dict[str, object]] = []

    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(amass_root)
        parts = relative.parts
        row: dict[str, object] = {
            "relative_path": relative.as_posix(),
            "source_dataset": parts[0],
            "parent_path": Path(*parts[:-1]).as_posix(),
            "subject_id_candidate": f"{parts[0]}::{parts[-2]}",
            "motion_id": path.name.removesuffix("_poses.npz"),
            "sha256": "", "npz_keys": "", "num_frames": "", "pose_width": "",
            "trans_frames": "", "trans_width": "", "mocap_framerate": "",
            "gender": "", "status": "error", "error": "",
        }
        try:
            row["sha256"] = sha256(path)
            with np.load(path, allow_pickle=False) as archive:
                row["npz_keys"] = json.dumps(sorted(archive.files))
                poses = archive["poses"]
                translations = archive["trans"]
                row["num_frames"] = poses.shape[0]
                row["pose_width"] = poses.shape[1] if poses.ndim > 1 else 1
                row["trans_frames"] = translations.shape[0]
                row["trans_width"] = translations.shape[1] if translations.ndim > 1 else 1
                row["mocap_framerate"] = scalar_text(archive, "mocap_framerate")
                row["gender"] = scalar_text(archive, "gender")
            row["status"] = "ok"
        except Exception as exc:  # keep a complete audit instead of stopping early
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

        if index % 100 == 0 or index == len(paths):
            readable = sum(item["status"] == "ok" for item in rows)
            print(f"[{index:,}/{len(paths):,}] ok={readable:,} latest={path.name}", flush=True)

    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with rejects_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(row for row in rows if row["status"] != "ok")

    readable = sum(row["status"] == "ok" for row in rows)
    return inventory_path, rejects_path, len(rows), readable


def main() -> int:
    inventory_path, rejects_path, total, readable = generate_inventory(
        Path(os.environ["AMASS_EXTRACTED_ROOT"]), Path(os.environ["AMASS_RUN_ROOT"])
    )
    print(f"Wrote {inventory_path}")
    print(f"Inventory: {total} files; {readable} readable; {total - readable} rejected")
    print(f"Reject report: {rejects_path}")
    return 0
