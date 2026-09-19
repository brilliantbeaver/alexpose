"""Audited AMASS -> paired render -> score-preserving tracks, with explicit assets."""
from __future__ import annotations

import ast
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .contracts import (TrackBundle, atomic_json, digest, sha256_file, code_identity,
                        array_digest, SCHEMA, PREPROCESSING)
from .data import audit_pair
from .extraction import extract_tracks


def _booleans(values):
    strings = values.astype(str).str.lower().str.strip()
    if not strings.isin(["true", "false", "1", "0", "yes", "no"]).all():
        raise ValueError("Reservation flags must be explicit true/false")
    return strings.isin(["true", "1", "yes"])


def audited_motion_windows(manifest_dir, amass_root, audit_csv, reservation_csv=None, *, review_mode="human_audited"):
    """Join locomotion audit to registry/reservation roles before motion loading.

    The optional reservation argument supports isolated software tests. Actual
    source preparation mandates it. No body/motion arrays are opened here.
    """
    if review_mode not in {"human_audited", "automated_development"}:
        raise ValueError("Unknown source review_mode")
    automated = review_mode == "automated_development"
    if automated and reservation_csv is None:
        raise ValueError("Automated development requires explicit reservation/unknown records")
    from ..motion_preservation.motion_data import load_amass_manifest
    base = load_amass_manifest(manifest_dir, amass_root)
    audit = pd.read_csv(audit_csv, keep_default_na=False)
    required = {"relative_path", "start_s", "locomotion_status", "audit_reviewer",
                "audit_evidence", "audit_date", "exposure", "canonical_person_id"}
    if required - set(audit):
        raise ValueError(f"Locomotion audit lacks {required - set(audit)}")
    if set(audit) & {"person_id", "original_split", "raw_path", "available", "duration_s", "split"}:
        raise ValueError("Audit must not override registry identity, split or source metadata")
    if audit[["relative_path", "start_s"]].duplicated().any():
        raise ValueError("Duplicate audited windows")
    joined = audit.merge(base, on="relative_path", how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Audited motion absent from approved existing identity registry")
    audit_reserved = _booleans(joined["reserved"]) if "reserved" in joined else pd.Series(False, index=joined.index)
    reservation_unknown = np.zeros(len(joined), dtype=bool)
    if reservation_csv is not None:
        reservations = pd.read_csv(reservation_csv, keep_default_na=False)
        columns = {"person_id", "canonical_person_id", "original_split", "reserved", "exposure"}
        if columns - set(reservations):
            raise ValueError(f"Reservation manifest lacks {columns - set(reservations)}")
        if reservations.person_id.duplicated().any() or reservations[list(columns)].astype(str).apply(lambda x: x.str.strip().eq("")).any().any():
            raise ValueError("Reservation requires one complete row per registry person")
        reservations = reservations[sorted(columns)].copy()
        flags = reservations["reserved"].astype(str).str.lower().str.strip()
        reservations["reservation_unknown"] = flags.eq("unknown") if automated else False
        reservations["reserved"] = _booleans(flags.mask(reservations["reservation_unknown"], "false"))
        for _, group in reservations.groupby("canonical_person_id"):
            if group.original_split.nunique() != 1:
                raise ValueError("Reservation aliases cross existing registry splits")
        reservations["reserved"] = reservations.groupby("canonical_person_id").reserved.transform("any")
        reservations["reservation_unknown"] = reservations.groupby("canonical_person_id").reservation_unknown.transform("any")
        authority = reservations.rename(columns={k: f"authority_{k}" for k in (columns | {"reservation_unknown"}) - {"person_id"}})
        joined = joined.merge(authority, on="person_id", how="left", validate="many_to_one")
        if joined.authority_canonical_person_id.isna().any():
            raise ValueError("Every audited registry person requires an explicit reservation/exposure record")
        for name in ("original_split", "canonical_person_id", "exposure"):
            if not joined[name].astype(str).eq(joined[f"authority_{name}"].astype(str)).all():
                raise ValueError(f"Audit/registry disagrees with authoritative {name}")
        reserved = audit_reserved.to_numpy() | joined.authority_reserved.to_numpy(bool)
        reservation_unknown = joined.authority_reservation_unknown.to_numpy(bool) & ~reserved
    else:
        if not joined.canonical_person_id.eq(joined.person_id).all():
            raise ValueError("Locomotion audit cannot override registry canonical identity")
        reserved = audit_reserved.to_numpy()
    joined["reserved"] = reserved.astype(object)
    joined.loc[reservation_unknown, "reserved"] = "unknown"
    joined["exclusion_reason"] = np.where(reserved, "reserved_identity",
                                         np.where(joined.original_split.eq("test"), "existing_test_identity", ""))
    usable = joined.original_split.isin(["train", "validation"]) & ~reserved
    allowed, rejected = joined.loc[usable].copy(), joined.loc[~usable].copy()
    for field in required - {"start_s"}:
        if allowed[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty audit field {field}")
    expected_status = "algorithm_screened_locomotion" if automated else "audited_locomotion"
    if not allowed.locomotion_status.eq(expected_status).all():
        raise ValueError(f"Explicit {expected_status} records required for {review_mode}")
    allowed["review_mode"] = review_mode
    if automated:
        for row in allowed.to_dict("records"):
            path = Path(row["audit_evidence"])
            if not path.is_absolute() or not path.is_file():
                raise ValueError("Automated audit_evidence must name an existing absolute evidence JSON path")
            evidence = json.loads(path.read_text())
            if (row["audit_reviewer"] != "stv2-kinematic-screen-v1"
                    or evidence.get("screen_version") != row["audit_reviewer"]
                    or evidence.get("reviewed_by") != "algorithm" or evidence.get("status") != "pass"
                    or evidence.get("review_mode") != review_mode
                    or evidence.get("relative_path") != row["relative_path"]
                    or evidence.get("canonical_person_id") != row["canonical_person_id"]
                    or not np.isclose(float(evidence.get("start_s", np.nan)), float(row["start_s"]), rtol=0, atol=1e-9)
                    or not isinstance(evidence.get("metrics"), dict)
                    or not isinstance(evidence.get("thresholds"), dict)
                    or not isinstance(evidence.get("source_sha256"), str)
                    or len(evidence["source_sha256"]) != 64):
                raise ValueError(f"Machine evidence does not support this exact window: {path}")
    if not allowed.available.all():
        raise FileNotFoundError("Audited AMASS source files unavailable")
    allowed["split"] = allowed.original_split.map({"train": "train", "validation": "development"})
    for _, group in allowed.groupby("canonical_person_id"):
        if group.split.nunique() != 1:
            raise ValueError("Person aliases cross existing splits")
    starts = pd.to_numeric(allowed.start_s, errors="raise").to_numpy(float)
    durations = allowed.duration_s.to_numpy(float)
    if not np.isfinite(starts).all() or not np.isfinite(durations).all() or np.any(starts < 0) or np.any(starts + 63 / 25 > durations):
        raise ValueError("Audited window exceeds finite physical source duration")
    for _, group in allowed.groupby("relative_path"):
        if len(group) > 1 and np.any(np.diff(np.sort(group.start_s.to_numpy(float))) < 64 / 25):
            raise ValueError("Initial source windows must not overlap")
    return allowed, rejected


def _file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required source asset is unavailable: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _tree(path, suffixes=None):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Required asset directory is unavailable: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes))
    if not paths:
        raise FileNotFoundError(f"Required asset directory is empty: {root}")
    return {"root": str(root), "files": {str(p.relative_to(root)): sha256_file(p) for p in paths}}


def _config_files(path, seen=None):
    """Bind literal MMEngine inheritance without executing model configuration."""
    path = Path(path).expanduser().resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return {}
    seen.add(path)
    result = {str(path): _file(path)["sha256"]}
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError("Relative config imports require an explicitly flattened reviewed configuration")
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "_base_" for target in targets):
            try:
                bases = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError("Dynamic configuration inheritance is unsupported; supply a flattened reviewed config") from exc
            bases = [bases] if isinstance(bases, str) else bases
            if not isinstance(bases, (list, tuple)) or not all(isinstance(x, str) for x in bases):
                raise ValueError("Configuration bases must be literal paths")
            for base in bases:
                if "::" in base:
                    raise ValueError("Package configuration inheritance must be flattened and reviewed")
                result.update(_config_files(path.parent / base, seen))
    return result


def preparation_provenance(config, table, repo):
    """Hash actual producer inputs; filenames/adjacent metadata are not identity."""
    manifest = Path(config["manifest_dir"]).expanduser().resolve()
    if (manifest / "amass").is_dir():
        manifest /= "amass"
    body_root = Path(config["body_model_root"]).expanduser().resolve()
    body = body_root / "smplh" if (body_root / "smplh").is_dir() else body_root
    dmpl = Path(config["dmpl_root"]).expanduser().resolve() if config.get("dmpl_root") else (
        body_root / "dmpls" if body != body_root else body_root.parent / "dmpls")
    estimates = {}
    for spec in config["estimators"]:
        if spec["student_id"] in estimates:
            raise ValueError("Duplicate source extractor ID")
        estimates[spec["student_id"]] = dict(family=spec["family"],
            configuration_files=_config_files(spec["config"]), checkpoint=_file(spec["checkpoint"]),
            head_checkpoint=_file(spec["head_checkpoint"]) if spec.get("head_checkpoint") else None)
    if not estimates:
        raise ValueError("At least one explicit source extractor is required")
    assets = {
        "locomotion_audit": _file(config["locomotion_audit"]),
        "reservation_exposure": _file(config["reservation_csv"]),
        "scope_config": _file(config["scope_config"]),
        "registry": {name: _file(manifest / name) for name in
                     ("amass_raw_inventory_eligible.csv", "amass_subject_registry.csv", "amass_subject_splits.csv")},
        "uv": _file(config["uv_path"]), "textures": _tree(config["texture_dir"], {".png", ".jpg", ".jpeg"}),
        "backgrounds": _tree(config["background_dir"], {".png", ".jpg", ".jpeg"}),
        "body_model": _tree(body, {".npz"}), "dmpl": _tree(dmpl, {".npz"}),
        "estimators": estimates,
        # Protected excluded motions are never read or hashed.
        "motions": {str(Path(path).expanduser().resolve()): _file(path)["sha256"] for path in sorted(set(table.raw_path))},
    }
    if config.get("review_mode") == "automated_development":
        assets["algorithm_screen_evidence"] = {
            str(Path(path).expanduser().resolve()): _file(path)["sha256"]
            for path in sorted(set(table.audit_evidence))
        }
        for row in table.to_dict("records"):
            evidence = json.loads(Path(row["audit_evidence"]).read_text())
            source = str(Path(row["raw_path"]).expanduser().resolve())
            if evidence["source_sha256"] != assets["motions"][source]:
                raise ValueError(f"Motion changed since algorithm screening: {source}")
    versions = {}
    for package in ("numpy", "scipy", "pandas", "torch", "torchvision", "mmcv", "mmengine",
                    "mmpose", "mmpretrain", "human-body-prior", "pyrender", "trimesh", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    payload = dict(configuration=config, schema=SCHEMA, preprocessing=PREPROCESSING,
                   code_hash=code_identity(repo), assets=assets, package_versions=versions)
    return {**payload, "identity": digest(payload)}


def held_family(specs, held):
    """Resolve a present, unambiguous held family without losing training input."""
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError("An explicit nonempty extractor roster is required")
    students = {}
    for spec in specs:
        if not isinstance(spec, dict) or any(
            not isinstance(spec.get(key), str) or not spec[key].strip()
            or spec[key] != spec[key].strip() for key in ("student_id", "family")
        ):
            raise ValueError("Every extractor needs a nonempty student_id and family")
        if spec["student_id"] in students:
            raise ValueError(f"Duplicate source extractor ID: {spec['student_id']}")
        students[spec["student_id"]] = spec["family"]
    families = set(students.values())
    if not isinstance(held, str) or not held.strip():
        raise ValueError("Specify a held extractor ID or family from the roster")
    matches = ({students[held]} if held in students else set()) | ({held} if held in families else set())
    if not matches:
        raise ValueError(f"Requested held extractor {held!r} is absent from the roster")
    if len(matches) != 1:
        raise ValueError(f"Requested held extractor {held!r} is ambiguous between an ID and a family")
    family = matches.pop()
    if not families - {family}:
        raise ValueError("Every configured extractor is held; no training extractor remains")
    return family


def prepare_source(config, output, repo):
    """Prepare explicitly authorized source assets; no fixture or asset fallback."""
    from .config import RunConfig
    if not config.get("reservation_csv"):
        raise ValueError("Source preparation requires authoritative reservation_csv")
    scope = RunConfig.load(config["scope_config"])
    if scope.device != "cuda":
        raise ValueError("Body/render/extraction requires explicit CUDA scope")
    scope.require_gpu_scope()
    table, rejected = audited_motion_windows(config["manifest_dir"], config["amass_root"],
                                             config["locomotion_audit"], config["reservation_csv"],
                                             review_mode=config.get("review_mode", "human_audited"))
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a unique preparation run; finished bundles are immutable")
    output.mkdir(parents=True)
    table.to_csv(output / "eligible-windows.csv", index=False)
    rejected.to_csv(output / "protected-excluded.csv", index=False)
    counts = dict(audited_windows=len(table) + len(rejected), admitted_windows=len(table),
                  excluded_windows=len(rejected), independent_people=int(table.canonical_person_id.nunique()),
                  train_windows=int(table.split.eq("train").sum()), development_windows=int(table.split.eq("development").sum()),
                  exclusion_reasons={str(k): int(v) for k, v in rejected.exclusion_reason.value_counts().items()})
    shortage = not counts["train_windows"] or not counts["development_windows"]
    atomic_json(output / "preparation-status.json", dict(status="insufficient_evidence" if shortage else "metadata_ready",
                counts=counts, reason="nonempty_training_and_development_required" if shortage else "runtime_pending"))
    if shortage:
        raise ValueError("No adequate train/development source windows; see preparation-status.json")
    excluded_family = held_family(config["estimators"], scope.held_extractor)
    provenance = preparation_provenance(config, table, repo)
    atomic_json(output / "preparation-provenance.json", provenance)
    from .runtime import require_haic_runtime
    require_haic_runtime()
    from ..motion_preservation.motion_data import load_motion
    from ..motion_preservation.body_geometry import SMPLHBody
    from ..synthetic_training.rendering import TexturedBodyRenderer, RenderRecipe, project_points, SMPL_INDICES
    from ..synthetic_training.estimators import StudentSpec, load_estimator
    inputs = {k: [] for k in ("xy", "confidence", "observed", "timestamps")}
    targets = {k: [] for k in ("xy", "valid", "visible", "eval_scale")}
    records, pairs, failures = [], [], []
    started, renderer = time.perf_counter(), None
    variants = [("clean", 0., 0.), ("blur", 2., 0.), ("obstruction", 0., .15), ("blur_obstruction", 2., .15)]
    try:
        body_model = SMPLHBody(config["body_model_root"], config.get("dmpl_root"), device="cuda")
        renderer = TexturedBodyRenderer(config["uv_path"], config["texture_dir"], config["background_dir"], 640, 480)
        estimators = {s["student_id"]: (s["family"], load_estimator(StudentSpec(**s), device="cuda")) for s in config["estimators"]}
        for row in table.to_dict("records"):
            path = str(Path(row["raw_path"]).expanduser().resolve())
            motion_hash = provenance["assets"]["motions"][path]
            if sha256_file(path) != motion_hash:
                raise ValueError("Motion changed after preparation identity was frozen")
            motion = load_motion(row, start_s=float(row["start_s"]), duration_s=64 / 25, fps=25)
            shape_hash = array_digest({"betas": np.asarray(motion.betas)})
            body = body_model.forward(motion)
            window_id = digest([motion_hash, row["start_s"]])[:20]
            seed = int(config.get("seed", 17)) + int(window_id[:6], 16)
            clean_factors = None
            for variant, blur, obstruction in variants:
                if variant == "blur_obstruction" and row["split"] == "train":
                    continue
                recipe = RenderRecipe(name=variant, azimuth_deg=0., person_height_fraction=.70,
                                      blur_px=blur, occlusion_fraction=obstruction)
                result = renderer.render(body, recipe, seed=seed)
                _, depth = project_points(body.joints[:, SMPL_INDICES], result["camera_pose"],
                                          640, 480, np.deg2rad(50.))
                factors = dict(motion_hash=motion_hash, shape_hash=shape_hash,
                    timestamps_hash=digest(body.timestamps.tolist()), camera_hash=digest(result["camera_pose"].tolist()),
                    background_hash=sha256_file(result["background_path"]),
                    lighting_hash=digest({"directional_intensity": 2.5, "ambient": .35}),
                    appearance_hash=sha256_file(result["texture_path"]), render_seed=seed,
                    blur_px=blur, occlusion_fraction=obstruction, target_hash=digest(result["keypoints"].tolist()))
                if clean_factors is None:
                    clean_factors = factors
                else:
                    audit_pair(clean_factors, factors, [k for k in ("blur_px", "occlusion_fraction") if clean_factors[k] != factors[k]])
                pairs.append(dict(window_id=window_id, variant=variant, **factors, camera_framing=result["camera_framing"],
                    person_pixel_height=[float(x) for x in result["boxes"][:, 3] - result["boxes"][:, 1]],
                    clipping=bool(np.any(result["boxes"][:, :2] <= 0) or np.any(result["boxes"][:, 2:] >= np.array([640, 480]))),
                    joint_depth=np.asarray(depth).tolist(), depth_min=float(np.min(depth)),
                    depth_max=float(np.max(depth)),
                    landmark_convention=result["landmark_convention"], visibility_reference=result["visibility_reference"]))
                if len(pairs) <= 12:
                    save_overlay(result, output / f"overlay-{window_id}-{variant}.png")
                for name, (family, estimator) in estimators.items():
                    if family == excluded_family and row["split"] == "train":
                        continue
                    track = extract_tracks(estimator, result["images"], result["boxes"], body.timestamps,
                                           box_source="renderer_foreground_privileged")
                    if track.status_counts["unsupported_scores"]:
                        raise ValueError("Extractor has unsupported confidence scores")
                    valid = np.isfinite(result["keypoints"]).all(-1)
                    for k, v in track.inputs().items():
                        inputs[k].append(v)
                    for k, v in dict(xy=result["keypoints"], valid=valid, visible=result["visible"] & valid,
                                    eval_scale=np.linalg.norm(result["boxes"][:, 2:] - result["boxes"][:, :2], axis=-1)).items():
                        targets[k].append(v)
                    records.append({**{k: str(row[k]) for k in ("person_id", "canonical_person_id", "relative_path",
                        "original_split", "split", "exposure", "locomotion_status", "audit_reviewer", "audit_evidence", "audit_date")},
                        "motion_id": row["relative_path"], "motion_hash": motion_hash, "window_id": window_id,
                        "variant": variant, "extractor": name, "extractor_family": family, "family": family,
                        "box_source": track.box_source, "reserved": row["reserved"],
                        "review_mode": row["review_mode"], "target_kind": "synthetic_proxy", "seed": seed,
                        "start_s": float(body.timestamps[0]), "end_s": float(body.timestamps[-1]), "extraction_status": track.status_counts})
        if preparation_provenance(config, table, repo)["identity"] != provenance["identity"]:
            raise ValueError("Producer assets or code changed during source preparation")
    except Exception as exc:
        failures.append({"status": "invalid_or_unavailable", "error": str(exc)})
        atomic_json(output / "preparation-status.json", dict(status="insufficient_evidence", counts=counts,
                    completed_track_rows=len(records), failures=failures))
        raise
    finally:
        if renderer is not None:
            renderer.close()
        atomic_json(output / "preparation-cost.json", dict(stage="body_render_extraction",
                    gpu_seconds=time.perf_counter() - started, failures=failures))
        atomic_json(output / "pairs.json", pairs)
    if not records:
        raise ValueError("No supported extracted tracks; no empty successful bundle")
    evidence_status = "automated-source-screen" if config.get("review_mode") == "automated_development" else "source-run"
    bundle = TrackBundle({k: np.stack(v) for k, v in inputs.items()},
                         {k: np.stack(v) for k, v in targets.items()}, records, evidence_status, provenance)
    bundle.save(output / "bundle")
    atomic_json(output / "preparation-status.json", dict(status="source_prepared", counts=counts,
                completed_track_rows=len(records), held_extractor_family=excluded_family,
                landmark_claim="synthetic_proxy_only", evidence_status=evidence_status,
                review_mode=config.get("review_mode", "human_audited")))
    return output / "bundle"


def save_overlay(result, path):
    from PIL import Image, ImageDraw
    positions = [0, len(result["images"]) // 2, len(result["images"]) - 1]
    canvas = Image.new("RGB", (len(positions) * result["images"].shape[2], result["images"].shape[1]))
    for col, index in enumerate(positions):
        frame = Image.fromarray(result["images"][index])
        draw = ImageDraw.Draw(frame)
        for j, (x, y) in enumerate(result["keypoints"][index]):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow")
            draw.text((x + 3, y), str(j), fill="red")
        canvas.paste(frame, (col * frame.width, 0))
    canvas.save(path)
