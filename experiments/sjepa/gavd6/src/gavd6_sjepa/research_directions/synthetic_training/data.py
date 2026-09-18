"""Small manifest-backed image datasets with explicit scientific data roles.

Coordinates are original full-image pixels. Estimator adapters own the shared
crop/resize transform. Reading unlabeled context never opens a landmark file.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rendering import COCO_INDICES, KEYPOINT_NAMES, LESSON_RECIPES, TexturedBodyRenderer, source_domains


BOX_COLUMNS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
INDEX_COLUMNS = [
    "frame_id", "clip_id", "frame_index", "image_path", "label_path", "label_index",
    *BOX_COLUMNS, "width", "height", "person_id", "motion_id", "recording_id",
    "domain_id", "role", "lesson_id",
]


def load_pose_manifest(path, roles=None) -> pd.DataFrame:
    """Load an image index; relative asset paths resolve beside its CSV."""
    if isinstance(path, pd.DataFrame):
        table, base = path.copy(), Path.cwd()
    else:
        manifest = Path(path).expanduser().resolve()
        table, base = pd.read_csv(manifest, keep_default_na=False), manifest.parent
    required = {"frame_id", "clip_id", "frame_index", "image_path", *BOX_COLUMNS, "role"}
    if not required <= set(table):
        raise ValueError(f"Image index lacks {sorted(required - set(table))}")
    if table.frame_id.duplicated().any():
        raise ValueError("frame_id must uniquely identify each indexed observation")
    for field in ("image_path", "label_path"):
        if field not in table:
            table[field] = ""
        table[field] = table[field].fillna("").map(
            lambda p: str((base / Path(str(p)).expanduser()).resolve()) if str(p) else ""
        )
    if "label_index" not in table:
        table["label_index"] = 0
    for field in ("person_id", "motion_id", "recording_id", "domain_id", "lesson_id"):
        if field not in table:
            table[field] = ""
    if roles is not None:
        roles = [roles] if isinstance(roles, str) else list(roles)
        table = table.loc[table.role.isin(roles)].copy()
    boxes = table[BOX_COLUMNS].to_numpy(float)
    if (not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError("Every input person box must have finite positive width and height")
    return table.reset_index(drop=True)


@lru_cache(maxsize=16)
def _read_labels(path):
    with np.load(path, allow_pickle=False) as labels:
        return {name: np.asarray(labels[name]).copy() for name in ("keypoints", "visible")}


class PoseFrameDataset:
    """Lazy RGB/landmark reader used by both notebook trials and Slurm arrays."""

    def __init__(self, index, labeled=True):
        self.index = load_pose_manifest(index)
        self.labeled = bool(labeled)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, position):
        from PIL import Image

        row = self.index.iloc[int(position)]
        with Image.open(row.image_path) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        result = dict(image=rgb, bbox=row[BOX_COLUMNS].to_numpy(np.float32),
                      frame_id=str(row.frame_id), metadata=row.to_dict())
        if self.labeled:
            if not row.label_path:
                raise ValueError(f"No reference labels for {row.frame_id}; import annotations first")
            labels = _read_labels(row.label_path)
            keypoints, visible = labels["keypoints"], labels["visible"]
            if keypoints.ndim == 3:
                keypoints, visible = keypoints[int(row.label_index)], visible[int(row.label_index)]
            if keypoints.shape != (12, 2) or visible.shape != (12,):
                raise ValueError(f"Expected twelve body landmarks: {row.label_path}")
            visible = visible.astype(bool)
            if not np.isfinite(keypoints[visible]).all():
                raise ValueError(f"Visible references contain nonfinite coordinates: {row.label_path}")
            result.update(keypoints=keypoints.astype(np.float32).copy(), visible=visible.copy())
        return result

    def get_batch(self, indices):
        """Return variable-size RGB images and aligned arrays, without resizing."""
        samples = [self[int(i)] for i in indices]
        if not samples:
            raise ValueError("Cannot load an empty image batch")
        batch = dict(images=[x["image"] for x in samples],
                     boxes=np.stack([x["bbox"] for x in samples]),
                     frame_ids=np.array([x["frame_id"] for x in samples]),
                     metadata=[x["metadata"] for x in samples])
        if self.labeled:
            batch.update(keypoints=np.stack([x["keypoints"] for x in samples]),
                         visible=np.stack([x["visible"] for x in samples]))
        return batch


def load_dataset_arrays(manifest_or_df, labeled=True):
    """Convenience loader for a small panel; trials should use lazy batches."""
    dataset = PoseFrameDataset(manifest_or_df, labeled=labeled)
    return dataset.get_batch(range(len(dataset)))


def context_clip_indices(index):
    """Yield clip ID and positional row indices in source-frame order."""
    table = load_pose_manifest(index)
    for clip_id, group in table.groupby("clip_id", sort=True):
        yield str(clip_id), group.sort_values("frame_index", kind="stable").index.to_numpy()


def load_context_clips(index):
    """Yield unlabeled RGB clips and fixed boxes, never their reference arrays."""
    dataset = PoseFrameDataset(index, labeled=False)
    for clip_id, indices in context_clip_indices(dataset.index):
        yield clip_id, dataset.get_batch(indices)


def _round_robin_rows(table, count, seed):
    selected = table.sample(frac=1, random_state=seed).copy()
    selected["_order"] = selected.groupby("person_id").cumcount()
    selected = selected.sort_values("_order", kind="stable").head(count).drop(columns="_order")
    if len(selected) != count:
        raise ValueError(f"Need {count} separate eligible motions but this data pool has {len(selected)}")
    return selected.reset_index(drop=True)


def assign_amass_roles(table, seed=17):
    """Separate people for libraries, fitting context/reference, and validation.

    Existing validation/test identities are never moved into training. Test
    people remain unopened. Within the support pool, separate source files are
    assigned to probe, lessons, and diagnostics by the preparation function.
    """
    table = table.copy()
    rng = np.random.default_rng(seed)
    train = np.array(sorted(table.loc[table.original_split.eq("train"), "person_id"].unique()))
    validation = np.array(sorted(table.loc[table.original_split.eq("validation"), "person_id"].unique()))
    if len(train) < 6 or len(validation) < 2:
        raise ValueError("Need at least six training and two validation people for independent data roles")
    rng.shuffle(train)
    rng.shuffle(validation)
    support_count = max(2, len(train) // 2)
    remaining = len(train) - support_count
    pools = {
        "support": train[:support_count],
        "train_context": train[support_count:support_count + remaining // 2],
        "train_reference": train[support_count + remaining // 2:],
        "validation_context": validation[:len(validation) // 2],
        "validation_reference": validation[len(validation) // 2:],
    }
    table["teaching_pool"] = "unused_test"
    for role, people in pools.items():
        table.loc[table.person_id.isin(people), "teaching_pool"] = role
    return table


def _write_rendered_clip(folder, result, clip_id, metadata):
    from PIL import Image

    item = folder / "synthetic" / clip_id
    item.mkdir(parents=True, exist_ok=True)
    label_path = item / "labels.npz"
    np.savez_compressed(label_path, keypoints=result["keypoints"], visible=result["visible"])
    rows = []
    for frame_index, rgb in enumerate(result["images"]):
        image_path = item / f"{frame_index:04d}.jpg"
        Image.fromarray(rgb).save(image_path, quality=95, subsampling=0)
        x1, y1, x2, y2 = result["boxes"][frame_index]
        rows.append(dict(
            **metadata, frame_id=f"{clip_id}_f{frame_index:04d}", clip_id=clip_id,
            frame_index=frame_index, image_path=str(image_path), label_path=str(label_path),
            label_index=frame_index, bbox_x1=x1, bbox_y1=y1, bbox_x2=x2, bbox_y2=y2,
            width=rgb.shape[1], height=rgb.shape[0], texture_path=result["texture_path"],
            background_path=result["background_path"], landmark_convention=result["landmark_convention"],
        ))
    (item / "scene.json").write_text(json.dumps({
        "recipe": result["recipe"], "azimuth_deg": result["azimuth_deg"],
        "camera_pose": result["camera_pose"].tolist(),
        "camera_framing": result["camera_framing"],
        "visibility_reference": result["visibility_reference"],
    }, indent=2) + "\n")
    return rows


def prepare_amass_library(cfg):
    """Render fixed lessons and disjoint source contexts/references from manifests.

    Twenty-four scene settings reuse each role's motion clips, so rendering a
    new background does not create a new independent motion. People and motions
    are explicit in the output index for grouped validation.
    """
    from ..motion_preservation.body_geometry import SMPLHBody
    from ..motion_preservation.motion_data import load_amass_manifest, load_motion

    folder = cfg.root / "data"
    folder.mkdir(parents=True, exist_ok=True)
    count = int(cfg.clips_per_lesson)
    if count < 1 or cfg.source_contexts_per_domain < 1 or cfg.clip_frames < 3:
        raise ValueError("Positive clip counts and at least three frames are required")
    duration = cfg.clip_frames / cfg.clip_fps
    table = load_amass_manifest(cfg.amass_manifest_dir, cfg.amass_root, cfg.seed)
    # Role assignment precedes availability filtering and never follows measured errors.
    table = assign_amass_roles(table, cfg.seed)
    table.to_csv(folder / "amass-person-pools.csv", index=False)
    available = table.loc[table.available & table.duration_s.ge(duration)].copy()
    support = _round_robin_rows(available.loc[available.teaching_pool.eq("support")], 17 * count, cfg.seed)
    tasks, offset = [], 0
    balanced = LESSON_RECIPES[-1]
    for role, recipes in (("probe", (balanced,)), ("lesson", LESSON_RECIPES), ("diagnostic", LESSON_RECIPES)):
        for recipe in recipes:
            for row in support.iloc[offset:offset + count].to_dict("records"):
                tasks.append((row, role, recipe, recipe.name if role != "probe" else "probe"))
            offset += count
    for role in ("train_context", "train_reference", "validation_context", "validation_reference"):
        rows = _round_robin_rows(available.loc[available.teaching_pool.eq(role)],
                                int(cfg.source_contexts_per_domain), cfg.seed)
        for row in rows.to_dict("records"):
            for recipe in source_domains():
                tasks.append((row, role, recipe, ""))
    body_model = SMPLHBody(cfg.body_model_root, cfg.dmpl_root, device=cfg.device)
    renderer = TexturedBodyRenderer(cfg.render_uv_path, cfg.render_texture_dir, cfg.render_background_dir,
                                    cfg.render_width, cfg.render_height)
    # The small body cache avoids decoding the same motion for every setting.
    @lru_cache(maxsize=2)
    def body_for(path, person_id):
        row = table.loc[table.raw_path.eq(path)].iloc[0].to_dict()
        motion_seed = cfg.seed + int(hashlib.sha1(str(row["relative_path"]).encode()).hexdigest()[:7], 16)
        # Fixed metadata-seeded windows avoid always taking calibration/standing
        # prefixes. The same source motion keeps its window across all domains.
        start_s = float(np.random.default_rng(motion_seed).uniform(0, max(0, row["duration_s"] - duration)))
        return body_model.forward(load_motion(row, start_s=start_s, duration_s=duration, fps=cfg.clip_fps))

    records, recipe_counts = [], {}
    try:
        for row, role, recipe, lesson_id in tasks:
            motion_id = hashlib.sha1(str(row["relative_path"]).encode()).hexdigest()[:12]
            clip_id = f"amass_{role}_{recipe.name}_{motion_id}"
            body = body_for(row["raw_path"], row["person_id"])
            # Every lesson uses the same ordered appearance draws. Contexts use
            # a motion-keyed draw shared across their camera/corruption settings.
            group = (role, recipe.name)
            within_recipe = recipe_counts.get(group, 0)
            recipe_counts[group] = within_recipe + 1
            appearance_seed = cfg.seed + (within_recipe if role in {"probe", "lesson", "diagnostic"}
                                           else int(motion_id[:7], 16))
            result = renderer.render(body, recipe, seed=appearance_seed)
            records.extend(_write_rendered_clip(folder, result, clip_id, dict(
                person_id=row["person_id"], motion_id=str(row["relative_path"]), recording_id=motion_id,
                domain_id=recipe.name if role.endswith(("context", "reference")) else "support",
                role=role, lesson_id=lesson_id, original_split=row["original_split"],
                source_path=row["raw_path"], source_start_s=float(body.timestamps[0]),
                source_frame_start=float(body.timestamps[0] * row["mocap_framerate"]), fps=cfg.clip_fps,
            )))
    finally:
        renderer.close()
    output = folder / "synthetic.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    return output


def prepare_coco_replay(cfg):
    """Fix a labelled COCO person subset using the official 17-keypoint schema.

    Only visibly labelled COCO landmarks (v=2) supervise this visible-joint task.
    Selection uses annotation completeness, never a student's prediction error.
    The requested count is distinct images, with one eligible person per image.
    """
    annotations = Path(cfg.coco_annotations_json).expanduser()
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO person_keypoints_train2017.json is required: {annotations}")
    payload = json.loads(annotations.read_text())
    images = {int(item["id"]): item for item in payload["images"]}
    candidates = []
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0) or int(annotation.get("category_id", 1)) != 1:
            continue
        points = np.asarray(annotation.get("keypoints", []), dtype=float)
        if points.size != 51:
            continue
        points = points.reshape(17, 3)[COCO_INDICES]
        if (points[:, 2] == 2).sum() < 6:
            continue
        candidates.append(annotation)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(candidates)
    chosen, image_ids = [], set()
    for item in candidates:
        if item["image_id"] not in image_ids:
            chosen.append(item)
            image_ids.add(item["image_id"])
        if len(chosen) == cfg.coco_replay_images:
            break
    if len(chosen) != cfg.coco_replay_images:
        raise ValueError(f"Requested {cfg.coco_replay_images} eligible COCO images; found {len(chosen)}")
    folder = cfg.root / "data" / "replay"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for item in chosen:
        image = images[int(item["image_id"])]
        image_path = Path(cfg.coco_image_root).expanduser() / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO replay image is absent: {image_path}")
        points = np.asarray(item["keypoints"], np.float32).reshape(17, 3)[COCO_INDICES]
        label_path = folder / f"person_{item['id']}.npz"
        np.savez_compressed(label_path, keypoints=points[:, :2], visible=points[:, 2] == 2)
        x, y, width, height = map(float, item["bbox"])
        records.append(dict(
            frame_id=f"coco_{item['image_id']}_{item['id']}", clip_id=f"coco_{item['image_id']}",
            frame_index=0, image_path=str(image_path.resolve()), label_path=str(label_path), label_index=0,
            bbox_x1=max(0, x), bbox_y1=max(0, y), bbox_x2=min(image["width"], x + width),
            bbox_y2=min(image["height"], y + height), width=image["width"], height=image["height"],
            person_id=f"coco_annotation_{item['id']}", motion_id="", recording_id=f"coco_{item['image_id']}",
            domain_id="coco", role="replay", lesson_id="", landmark_convention="coco_body12_visible_v2",
        ))
    output = cfg.root / "data" / "replay.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    return output
