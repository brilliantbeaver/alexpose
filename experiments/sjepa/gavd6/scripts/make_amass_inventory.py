from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

amass_root = Path(os.environ["AMASS_EXTRACTED_ROOT"]).resolve()
run_root = Path(os.environ["AMASS_RUN_ROOT"]).resolve()
out_dir = run_root / "manifests"
out_dir.mkdir(parents=True, exist_ok=True)

inventory_path = out_dir / "amass_raw_inventory.csv"
rejects_path = out_dir / "amass_inventory_rejects.csv"

FIELDS = [
    "relative_path",
    "source_dataset",
    "parent_path",
    "subject_id_candidate",
    "motion_id",
    "sha256",
    "npz_keys",
    "num_frames",
    "pose_width",
    "trans_frames",
    "trans_width",
    "mocap_framerate",
    "gender",
    "status",
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


rows = []
for path in sorted(amass_root.rglob("*_poses.npz")):
    relative = path.relative_to(amass_root)
    parts = relative.parts

    # Keep the unmodified hierarchy for later dataset-specific subject parsing.
    source_dataset = parts[0]
    parent_path = Path(*parts[:-1]).as_posix()
    subject_id_candidate = f"{source_dataset}::{parts[-2]}"
    motion_id = path.name.removesuffix("_poses.npz")

    row = {
        "relative_path": relative.as_posix(),
        "source_dataset": source_dataset,
        "parent_path": parent_path,
        "subject_id_candidate": subject_id_candidate,
        "motion_id": motion_id,
        "sha256": "",
        "npz_keys": "",
        "num_frames": "",
        "pose_width": "",
        "trans_frames": "",
        "trans_width": "",
        "mocap_framerate": "",
        "gender": "",
        "status": "error",
        "error": "",
    }

    try:
        row["sha256"] = sha256(path)

        with np.load(path, allow_pickle=False) as archive:
            row["npz_keys"] = json.dumps(sorted(archive.files))
            poses = archive["poses"]
            trans = archive["trans"]

            row["num_frames"] = poses.shape[0]
            row["pose_width"] = poses.shape[1] if poses.ndim > 1 else 1
            row["trans_frames"] = trans.shape[0]
            row["trans_width"] = trans.shape[1] if trans.ndim > 1 else 1
            row["mocap_framerate"] = scalar_text(archive, "mocap_framerate")
            row["gender"] = scalar_text(archive, "gender")

        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

    rows.append(row)

with inventory_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

with rejects_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(row for row in rows if row["status"] != "ok")

ok = sum(row["status"] == "ok" for row in rows)
print(f"Wrote {inventory_path}")
print(f"Inventory: {len(rows)} files; {ok} readable; {len(rows) - ok} rejected")
print(f"Reject report: {rejects_path}")
