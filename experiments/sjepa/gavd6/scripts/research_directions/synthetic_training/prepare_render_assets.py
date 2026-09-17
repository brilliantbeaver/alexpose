#!/usr/bin/env python3
"""Prepare published SMPLitex textures/UVs and COCO photographic backgrounds.

Run with the study environment after exporting ST_TEXTURE_DIR, ST_UV_PATH,
ST_BACKGROUND_DIR, ST_COCO_IMAGE_ROOT, ST_COCO_ANNOTATIONS, and ST_RUN_ROOT.
This downloads appearance assets only. Licensed SMPL-H/DMPL models remain
user-supplied. The GPU preflight checks UV compatibility with those models.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from urllib.request import Request, urlopen

from PIL import Image


UV_URL = (
    "https://raw.githubusercontent.com/dancasas/SMPLitex/"
    "0476822a8273f96ee8ab3463560d6968c46697c7/"
    "sample-data/smpl_uv_20200910/smpl_uv.obj"
)
UV_SHA256 = "66e65e07e347aabfb4f309060863ad4aaa113d62d655e40a4273453ac70d8458"
TEXTURE_URL = (
    "https://raw.githubusercontent.com/dancasas/dancasas.github.io/"
    "2229570de720a8927c6a206ba21976d7d0f2d005/"
    "projects/SMPLitex/SMPLitex-dataset/textures/SMPLitex-texture-{index:05d}.png"
)


def env_path(name: str) -> Path:
    value = os.environ.get(name, "")
    if not value or not Path(value).expanduser().is_absolute():
        raise ValueError(f"Export {name} as a nonempty absolute path first.")
    return Path(value).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_image(path: Path) -> None:
    with Image.open(path) as image:
        if min(image.size) < 64:
            raise ValueError(f"Image is too small to use: {path}")
        image.verify()
    with Image.open(path) as image:
        image.convert("RGB").load()


def download(url: str, destination: Path, *, expected_hash: str | None = None) -> bool:
    """Preserve existing files; atomically publish a complete verified download."""
    def verify(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty asset: {path}")
        if expected_hash and sha256(path) != expected_hash:
            raise ValueError(f"Asset checksum mismatch: {path}")
        if destination.suffix.lower() == ".png":
            check_image(path)

    if destination.exists():
        verify(destination)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "gavd6-research-asset-setup/1"})
            with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify(temporary)
            temporary.replace(destination)
            return True
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def background_candidates(payload: dict, image_root: Path, count: int, seed: int) -> list[dict]:
    """Choose real COCO images with no annotated people, disjoint from pose replay."""
    categories = {int(item["id"]): item["name"] for item in payload.get("categories", [])}
    if categories != {1: "person"}:
        raise ValueError("Use COCO person_keypoints_train2017.json, containing only category person.")
    if not any(len(item.get("keypoints", [])) == 51 for item in payload.get("annotations", [])):
        raise ValueError("The COCO JSON does not contain 17-landmark person annotations.")
    annotated = {int(item["image_id"]) for item in payload["annotations"]}
    candidates = [item for item in sorted(payload["images"], key=lambda row: int(row["id"]))
                  if int(item["id"]) not in annotated
                  and (image_root / item["file_name"]).is_file()]
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f"Need {count} available COCO backgrounds without annotated people; found {len(candidates)}.")
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backgrounds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.backgrounds < 1 or not 1 <= args.workers <= 8:
        parser.error("Require positive backgrounds and 1–8 download workers.")
    run = env_path("ST_RUN_ROOT")
    if (run / "selectors/frozen.joblib").exists() or any((run / "source").glob("*/outcomes.csv")):
        raise RuntimeError("Source results already exist. Use a new run and separate appearance paths for a changed library.")
    textures, backgrounds = env_path("ST_TEXTURE_DIR"), env_path("ST_BACKGROUND_DIR")
    uv = env_path("ST_UV_PATH")
    images, annotation_path = env_path("ST_COCO_IMAGE_ROOT"), env_path("ST_COCO_ANNOTATIONS")
    if uv.suffix.lower() != ".obj":
        raise ValueError("Set ST_UV_PATH to the destination smpl_uv.obj for this asset recipe.")
    if textures == backgrounds or textures in backgrounds.parents or backgrounds in textures.parents:
        raise ValueError("Texture and background folders must be separate, non-nested directories.")
    payload = json.loads(annotation_path.read_text())
    selected = background_candidates(payload, images, args.backgrounds, args.seed)
    expected_backgrounds = {Path(item["file_name"]).name for item in selected}
    expected_textures = {f"SMPLitex-texture-{index:05d}.png" for index in range(250)}
    for folder, expected in ((textures, expected_textures), (backgrounds, expected_backgrounds)):
        unexpected = [path for path in folder.rglob("*")
                      if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                      and (path.parent != folder or path.name not in expected)]
        if unexpected:
            raise ValueError(f"Use a dedicated appearance directory; unexpected images in {folder}: {unexpected[:3]}")
    print("Downloading/verifying the author-published UV template and 250 SMPLitex textures.", flush=True)
    download(UV_URL, uv, expected_hash=UV_SHA256)

    def texture(index):
        target = textures / f"SMPLitex-texture-{index:05d}.png"
        url = TEXTURE_URL.format(index=index)
        retrieved = download(url, target)
        return dict(path=str(target), expected_source_url=url, sha256=sha256(target),
                    retrieval="downloaded_from_pinned_url" if retrieved else "existing_image_validated_origin_not_rechecked")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        texture_records = list(pool.map(texture, range(250)))
    backgrounds.mkdir(parents=True, exist_ok=True)
    background_records = []
    for item in selected:
        source, target = images / item["file_name"], backgrounds / Path(item["file_name"]).name
        check_image(source)
        if target.exists():
            if sha256(source) != sha256(target):
                raise ValueError(f"Existing background differs from its selected COCO source: {target}")
        else:
            temporary = target.with_name(target.name + ".part")
            shutil.copy2(source, temporary)
            temporary.replace(target)
        background_records.append(dict(image_id=item["id"], source=str(source), path=str(target)))
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  description="Published generated SMPL UV textures and real COCO photographic backgrounds",
                  dataset="https://dancasas.github.io/projects/SMPLitex/",
                  uv=dict(path=str(uv), url=UV_URL, sha256=UV_SHA256),
                  textures=texture_records, backgrounds=background_records,
                  coco_annotations=str(annotation_path), seed=args.seed,
                  background_rule="Image ID has no entry in person-keypoint annotations; no claim of exhaustive person detection.")
    provenance = uv.parent / "appearance-provenance.json"
    provenance.write_text(json.dumps(record, indent=2) + "\n")
    print(f"APPEARANCE_ASSETS_PREPARED: 250 textures, {len(selected)} photographic backgrounds")
    print(f"ST_TEXTURE_DIR={textures}\nST_BACKGROUND_DIR={backgrounds}\nST_UV_PATH={uv}")
    print(f"Provenance: {provenance}")
    print("Next: GPU preflight must verify the UV template against your licensed SMPL-H/DMPL model and render it.")


if __name__ == "__main__":
    main()
