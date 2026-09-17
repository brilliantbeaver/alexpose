#!/usr/bin/env python3
"""Check the real study assets and disposable GPU updates before notebook 01.

Run with the dedicated ST_PYTHON environment, on a GPU allocation unless using
--assets-only. Artifacts stay under ST_RUN_ROOT/preflight/<UTC timestamp>/.
This does not prepare or certify the independent human GAVD evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from gavd6_sjepa.research_directions.motion_preservation.motion_data import (
    load_amass_manifest, load_gavd_manifest, load_motion,
)
from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training.data import (
    PoseFrameDataset, _round_robin_rows, assign_amass_roles, prepare_coco_replay,
)
from gavd6_sjepa.research_directions.synthetic_training.rendering import (
    LESSON_RECIPES, TexturedBodyRenderer, load_uv_topology,
)


def require_path(value, name, *, directory=False):
    if not value:
        raise ValueError(f"{name} is unset; configure it before running preflight")
    path = Path(value).expanduser().resolve()
    if directory:
        if not path.is_dir():
            raise NotADirectoryError(f"{name} must be a directory: {path}")
    elif not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{name} must be a nonempty file: {path}")
    return path


def revision(path):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"commit": commit, "tracked_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_changes": None}


def check_images(folder, label):
    paths = sorted(p for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        raise FileNotFoundError(f"{label} has no PNG/JPG/JPEG images: {folder}")
    for path in paths:
        with Image.open(path) as image:
            image.convert("RGB").load()
            if min(image.size) < 2:
                raise ValueError(f"{label} image is too small: {path}")
    return len(paths)


def selected_motions(cfg):
    table = assign_amass_roles(load_amass_manifest(
        cfg.amass_manifest_dir, cfg.amass_root, cfg.seed), cfg.seed)
    available = table.loc[table.available & table.duration_s.ge(cfg.clip_frames / cfg.clip_fps)]
    counts = {"support": 17 * cfg.clips_per_lesson}
    counts.update({role: cfg.source_contexts_per_domain for role in (
        "train_context", "train_reference", "validation_context", "validation_reference")})
    selected, summary = [], {}
    for role, needed in counts.items():
        pool = available.loc[available.teaching_pool.eq(role)]
        summary[role] = {"eligible_motions": len(pool), "required_motions": needed,
                         "eligible_people": int(pool.person_id.nunique())}
        selected.append(_round_robin_rows(pool, needed, cfg.seed))
    return pd.concat(selected, ignore_index=True), summary


def check_assets(cfg, output, report):
    if cfg.context_kind != "vjepa" or not cfg.image_checkpoint:
        raise ValueError("This complete source preflight requires V-JEPA and the image comparator")
    if {s["role"] for s in cfg.students} != {"train", "validation", "held"}:
        raise ValueError("Configure train, validation and held students before preflight")
    if cfg.clip_fps <= 0 or cfg.clip_frames < 3:
        raise ValueError("Source clips require positive clip_fps and at least three frames")
    directory_fields = ("amass_root", "amass_manifest_dir", "body_model_root",
                        "render_texture_dir", "render_background_dir", "coco_image_root",
                        "gavd_manifest_dir", "gavd_video_root", "context_repo")
    file_fields = ("render_uv_path", "coco_annotations_json", "gavd_reservation_csv",
                   "context_checkpoint", "image_checkpoint")
    report["asset_paths"] = {}
    for name in (*directory_fields, *file_fields):
        path = require_path(getattr(cfg, name), name, directory=name in directory_fields)
        report["asset_paths"][name] = str(path)
    require_path(Path(cfg.context_repo) / "hubconf.py", "context_repo/hubconf.py")
    for item in cfg.students:
        for field in ("config", "checkpoint", "head_checkpoint"):
            if field == "head_checkpoint" and not item.get(field):
                continue
            path = require_path(item.get(field), f"{item['student_id']}:{field}")
            report["asset_paths"][f"{item['student_id']}:{field}"] = str(path)
    roster = pd.read_csv(cfg.gavd_reservation_csv, dtype={"video_id": str}, keep_default_na=False)
    if not {"video_id", "role"} <= set(roster) or roster.empty:
        raise ValueError("ST_GAVD_RESERVATION needs a CSV with video_id and role columns")
    if (not set(roster.role) <= {"development", "confirmation"}
            or roster.video_id.astype(str).str.strip().eq("").any()
            or not roster.role.eq("development").any()):
        raise ValueError("Reservation needs nonempty IDs and development/confirmation roles")
    report["reservation_schema_checked"] = True
    protected = set(roster.loc[roster.role.eq("confirmation"), "video_id"])
    permitted = set(roster.loc[roster.role.eq("development"), "video_id"]) - protected
    gavd = load_gavd_manifest(cfg.gavd_manifest_dir, cfg.gavd_video_root)
    available = gavd.loc[gavd.available & gavd.video_id.isin(permitted)]
    count = int(available.video_id.nunique())
    required = 6 * (cfg.gavd_context_recordings + cfg.gavd_early_recordings
                    + cfg.gavd_confirmation_recordings)
    report["gavd_readiness"] = {
        "available_permitted_unique_video_ids": count, "minimum_before_view_grouping": required,
        "view_crop_and_related_recording_checks_complete": False,
        "note": "This count is necessary but not sufficient; checked view/size cells and related-video grouping remain required",
    }
    if count < required:
        raise ValueError(f"GAVD has {count} available permitted video IDs; the configured six-cell panel needs at least {required}")
    report["decoded_appearance_images"] = {
        "textures": check_images(Path(cfg.render_texture_dir), "textures"),
        "backgrounds": check_images(Path(cfg.render_background_dir), "backgrounds"),
    }
    rows, report["amass_pools"] = selected_motions(cfg)
    motions = {}
    for row in rows.to_dict("records"):
        duration = cfg.clip_frames / cfg.clip_fps
        motion_seed = cfg.seed + int(hashlib.sha1(str(row["relative_path"]).encode()).hexdigest()[:7], 16)
        start_s = float(np.random.default_rng(motion_seed).uniform(0, max(0, row["duration_s"] - duration)))
        motion = load_motion(row, start_s=start_s, duration_s=duration, fps=cfg.clip_fps)
        if len(motion.poses) != cfg.clip_frames:
            raise ValueError(f"Unexpected decoded frame count: {row['raw_path']}")
        motions.setdefault(motion.gender, (row, motion))
    rows.to_csv(output / "selected-amass-motions.csv", index=False)
    report["decoded_selected_amass_motions"] = len(rows)
    from gavd6_sjepa.research_directions.motion_preservation.body_geometry import SMPLHBody

    body = SMPLHBody(cfg.body_model_root, cfg.dmpl_root, device="cpu")
    report["body_assets"] = {}
    for gender in sorted(motions):
        model_path = require_path(body.body_root / gender / "model.npz", f"SMPL-H {gender}")
        dmpl_path = require_path(body.dmpl_root / gender / "model.npz", f"DMPL {gender}")
        with np.load(model_path, allow_pickle=False) as model:
            faces = np.asarray(model["f"])
            load_uv_topology(cfg.render_uv_path, faces)
        with np.load(dmpl_path, allow_pickle=False) as dmpl:
            if "eigvec" not in dmpl or not np.isfinite(dmpl["eigvec"]).all():
                raise ValueError(f"DMPL needs finite eigvec coefficients: {dmpl_path}")
        report["body_assets"][gender] = {"model": str(model_path), "dmpl": str(dmpl_path),
                                         "uv_faces_match": True}
    # Use the production deterministic subset, but keep all labels/indexes here.
    replay_path = prepare_coco_replay(replace(cfg, run_root=str(output)))
    replay = PoseFrameDataset(replay_path)
    for i in range(len(replay)):
        replay[i]  # Decode every selected real image and validate its visible labels.
    report["decoded_coco_replay_images"] = len(replay)
    return motions, replay


def save_preview(rendered, path):
    frames = np.unique(np.rint(np.linspace(0, len(rendered["images"]) - 1, 6)).astype(int))
    previews = []
    for i in frames:
        frame = Image.fromarray(rendered["images"][i]).copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle(tuple(map(float, rendered["boxes"][i])), outline="cyan", width=2)
        for j, (x, y) in enumerate(rendered["keypoints"][i]):
            if np.isfinite([x, y]).all():
                color = "lime" if rendered["visible"][i, j] else "red"
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                draw.text((x + 3, y + 3), str(j + 1), fill=color)
        previews.append(frame)
    width, height = previews[0].size
    sheet = Image.new("RGB", (width * len(previews), height))
    for i, frame in enumerate(previews):
        sheet.paste(frame, (i * width, 0))
    sheet.save(path)


def check_gpu(cfg, output, report, motions, replay):
    import torch
    from gavd6_sjepa.research_directions.motion_preservation.body_geometry import SMPLHBody
    from gavd6_sjepa.research_directions.synthetic_training.context_features import load_context_encoder
    from gavd6_sjepa.research_directions.synthetic_training.estimators import load_estimator
    from gavd6_sjepa.research_directions.synthetic_training.trials import (
        context_spec, crop_context_images, student_spec,
    )

    if os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise ValueError("Export PYOPENGL_PLATFORM=egl before starting the GPU preflight")
    if not str(cfg.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Full preflight needs ST_DEVICE=cuda and an allocated CUDA GPU")
    report["gpu"] = torch.cuda.get_device_name(torch.device(cfg.device))
    body = SMPLHBody(cfg.body_model_root, cfg.dmpl_root, device=cfg.device)
    renderer = TexturedBodyRenderer(cfg.render_uv_path, cfg.render_texture_dir,
                                    cfg.render_background_dir, cfg.render_width, cfg.render_height)
    rendered = None
    try:
        for gender, (row, motion) in motions.items():
            print(f"GPU: body, UV and EGL render for {gender}", flush=True)
            result = renderer.render(body.forward(motion), LESSON_RECIPES[-1], seed=cfg.seed)
            if not result["visible"].any() or not np.isfinite(result["keypoints"]).all():
                raise ValueError(f"Rendered {gender} motion has unusable landmark labels")
            save_preview(result, output / f"render-{gender}-overlay.png")
            if rendered is None:
                rendered = result
    finally:
        renderer.close()
    del renderer, body, result
    gc.collect()
    torch.cuda.empty_cache()
    report["rendered_genders"] = list(motions)
    clip = crop_context_images(list(rendered["images"]), rendered["boxes"])
    report["context_features"] = {}
    for kind in ("vjepa", "image"):
        print(f"GPU: {kind} frozen context forward", flush=True)
        encoder = load_context_encoder(context_spec(cfg, kind=kind), cfg.device)
        vector = encoder.encode(clip)
        if not vector.size or not np.isfinite(vector).all():
            raise FloatingPointError(f"{kind} context vector is empty/nonfinite")
        report["context_features"][kind] = {"shape": list(vector.shape)}
        if kind == "vjepa":
            shuffled = encoder.encode(clip, shuffle_seed=cfg.seed)
            if shuffled.shape != vector.shape or not np.isfinite(shuffled).all():
                raise FloatingPointError("Shuffled V-JEPA context vector is invalid")
            report["context_features"][kind]["shuffled_checked"] = True
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    n_synthetic = round(cfg.train_batch_size * cfg.synthetic_fraction)
    rng = np.random.default_rng(cfg.seed)
    batch = replay.get_batch(rng.integers(len(replay), size=cfg.train_batch_size - n_synthetic))
    positions = np.random.default_rng(cfg.seed + 1).integers(len(rendered["images"]), size=n_synthetic)
    batch["images"] += [rendered["images"][i] for i in positions]
    for key in ("keypoints", "visible", "boxes"):
        batch[key] = np.concatenate([batch[key], rendered[key][positions]])
    prediction_positions = np.arange(cfg.predict_batch_size) % len(batch["images"])
    prediction_images = [batch["images"][i] for i in prediction_positions]
    prediction_boxes = batch["boxes"][prediction_positions]
    report["students"] = {}
    for item in cfg.students:
        sid = item["student_id"]
        print(f"GPU: {sid} checkpoint, prediction and one disposable mixed update", flush=True)
        torch.manual_seed(cfg.seed)
        estimator = load_estimator(student_spec(item), cfg.device)
        before = {name: p.detach().cpu().clone() for name, p in estimator.model.head.named_parameters()}
        prediction = estimator.predict(prediction_images, boxes=prediction_boxes,
                                       batch_size=cfg.predict_batch_size)
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"{sid} returned nonfinite initial predictions")
        optimizer = estimator.make_optimizer(float(item.get("learning_rate", cfg.learning_rate)), cfg.weight_decay)
        loss = estimator.train_batch(batch["images"], batch["keypoints"], batch["visible"],
                                     optimizer, boxes=batch["boxes"])
        changed = any(not torch.equal(before[name], p.detach().cpu())
                      for name, p in estimator.model.head.named_parameters())
        prediction = estimator.predict(prediction_images, boxes=prediction_boxes,
                                       batch_size=cfg.predict_batch_size)
        if not np.isfinite(loss) or not changed or not np.isfinite(prediction).all():
            raise FloatingPointError(f"{sid} failed finite-loss/prediction/changed-parameter checks")
        report["students"][sid] = {"loss_finite": True, "head_parameters_changed": changed,
                                    "training_batch_size": len(batch["images"]),
                                    "prediction_batch_size": len(prediction_images)}
        del estimator, optimizer, before
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-only", action="store_true", help="CPU paths, schemas, image and motion decoding only")
    args = parser.parse_args()
    cfg = RunConfig.from_env()
    output = cfg.root / "preflight" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "mode": "assets" if args.assets_only else "gpu",
              "configuration": asdict(cfg), "source": revision(ROOT),
              "preflight_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "context_repository": revision(cfg.context_repo) if cfg.context_repo else None,
              "python": sys.executable, "python_version": sys.version,
              "output_directory": str(output), "real_evaluation_ready": False,
              "real_evaluation_requires": "Checked GAVD views, eligible disjoint panel, human references and confirmation review"}
    path = output / "report.json"
    print(f"Preflight artifacts: {output}", flush=True)
    try:
        print("CPU: validate paths, rendering images, selected AMASS motions, UV and COCO replay", flush=True)
        motions, replay = check_assets(cfg, output, report)
        if not args.assets_only:
            check_gpu(cfg, output, report, motions, replay)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {path}", flush=True)
    if report["status"] != "passed":
        return 1
    print("ASSET_PREFLIGHT_PASSED" if args.assets_only else "SOURCE_PREFLIGHT_PASSED", flush=True)
    print("Inspect render-*-overlay.png after GPU success. Human GAVD evaluation remains a separate stage.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
