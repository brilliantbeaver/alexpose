"""Notebook-sized stages for the movement-preservation experiment.

Small NPZ caches keep expensive frozen extraction separate from fast gate
experiments. The final role is constructed only after training and calibration.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
import torch

from .config import RunConfig
from . import motion_data as data
from . import body_geometry as geometry
from . import repair_models as learning
from . import preservation_metrics as metrics
from .pretrained_models import FlowResult, MoMaskPrior, OpticalFlowEstimator, PriorResult, joint_transport_diagnostics, load_external_prior
from .rendering import Camera, render_sequence


DEVELOPMENT_ROLES = ("train", "calibration", "development")
FAMILIES = {"train": "arm_leg_timing", "calibration": "foot_clearance",
            "development": "foot_clearance", "final": "trunk_pelvis_timing"}


def config_from_environment():
    return RunConfig.from_env()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def default(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=default) + "\n")


def _npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {k: arrays[k] for k in arrays.files}


def _safe(text):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(text))


def _same_settings(path, settings):
    """Do not silently relabel cached measurements after a configuration edit."""
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError(f"Cached settings differ at {path}. Use a new run root (or new prior_id for a second prior).")
    if not path.exists():
        _json(path,settings)


def _asset(path):
    if not path:
        return None
    p=Path(path).expanduser().resolve()
    # Adding final-case exports to an external prediction directory is normal;
    # only changes to a concrete checkpoint/configuration file invalidate reuse.
    return {"path":str(p),"mtime_ns":p.stat().st_mtime_ns if p.is_file() else None}


def require_fitted(cfg):
    """Do this before opening final motion, not after inspecting its result."""
    training = cfg.root / "models/training.json"
    calibration = cfg.root / "calibration/locked.json"
    if not training.is_file() or not calibration.is_file():
        raise RuntimeError("Run notebook03 training and calibration before opening final cases.")
    trained, locked = json.loads(training.read_text()), json.loads(calibration.read_text())
    if trained["created"] != locked["training_created"]:
        raise RuntimeError("The gate changed after calibration. Recalibrate on calibration people before evaluating.")
    return trained, locked


def inventory(cfg):
    """Actual manifest counts and runtime asset availability; no video is decoded."""
    cfg.root.mkdir(parents=True, exist_ok=True)
    if not (cfg.root / "config.json").exists():
        cfg.save()
    amass = data.load_amass_manifest(cfg.amass_manifest_dir, cfg.amass_root, cfg.seed)
    gavd = data.load_gavd_manifest(cfg.gavd_manifest_dir, cfg.gavd_video_root)
    amass.to_csv(cfg.root / "amass_inventory.csv", index=False)
    gavd.to_csv(cfg.root / "gavd_inventory.csv", index=False)
    paths = {"AMASS raw": cfg.amass_root, "SMPL-H and DMPL": cfg.body_model_root,
             "MoMask code": cfg.momask_repo, "MoMask checkpoint directory": cfg.momask_checkpoint,
             "Flow checkpoint": cfg.flow_checkpoint, "GAVD video": cfg.gavd_video_root}
    availability = pd.DataFrame([dict(asset=k, path=v, exists=bool(v) and Path(v).expanduser().exists()) for k,v in paths.items()])
    availability["experiment_mode"] = cfg.mode
    return {"amass": amass, "gavd": gavd, "availability": availability}


def _demo_rows(cfg, role):
    return pd.DataFrame([dict(person_id=f"demo_{role}_person{i}", role=role,
                             source_dataset="SIMULATED_MECHANICS_ONLY", relative_path=f"demo/{role}/{i}",
                             raw_path="", duration_s=cfg.duration_s) for i in range(cfg.max_motions_per_role)])


def build_pairs(cfg, roles=DEVELOPMENT_ROLES):
    """Construct grouped angle edits, a factorial and exactly matched observations.

    Four scene conditions rotate across source motions: textured, textureless,
    moving illumination and occlusion. No scene property depends on the label.
    Reference flow is saved separately and never supplied to the learned gate.
    """
    roles = tuple(roles)
    if "final" in roles:
        require_fitted(cfg)
    if not set(roles) <= set(FAMILIES):
        raise ValueError(f"Unknown role in {roles}")
    cfg.root.mkdir(parents=True, exist_ok=True)
    if not (cfg.root / "config.json").exists():
        cfg.save()
    _same_settings(cfg.root/"pair_settings.json",{key:getattr(cfg,key) for key in
                   ("mode","amass_root","body_model_root","dmpl_root","fps","duration_s","image_size","seed","noise_std_m",
                    "max_motions_per_role","max_motions_per_person")})
    if "final" in roles:
        _json(cfg.root/"final_opened.json",{"training_created":require_fitted(cfg)[0]["created"]})
    for directory in ("cases", "scenes"):
        (cfg.root / directory).mkdir(exist_ok=True)
    table = None if cfg.mode == "demo" else data.load_amass_manifest(cfg.amass_manifest_dir, cfg.amass_root, cfg.seed)
    body_model = None
    if cfg.mode == "real":
        if not cfg.body_model_root:
            raise FileNotFoundError("Set MP_BODY_MODEL_ROOT (or AMASS_BODY_MODEL_ROOT) to the licensed SMPL-H/DMPL tree.")
        body_model = geometry.SMPLHBody(cfg.body_model_root, cfg.dmpl_root, cfg.device)
    index_path = cfg.root / "cases.csv"
    existing = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    completed_roles = set(existing.role) if len(existing) else set()
    records, skipped = [], []
    for role in roles:
        if role in completed_roles:
            continue
        if cfg.mode == "demo":
            chosen = _demo_rows(cfg, role)
        else:
            available = table.loc[table.available]
            chosen = data.select_motion_rows(available, role, cfg.max_motions_per_role, cfg.duration_s, cfg.seed)
            chosen = chosen.groupby("person_id", sort=False).head(cfg.max_motions_per_person)
            if not len(chosen):
                raise FileNotFoundError(f"No available {role} AMASS motions of {cfg.duration_s}s under {cfg.amass_root}.")
        for number, (_, source) in enumerate(chosen.iterrows()):
            family = FAMILIES[role]
            stem = f"{role}_{number:04d}_{_safe(source.person_id)}"
            started = perf_counter()
            if cfg.mode == "demo":
                clean, event = geometry.demo_motion(int(round(cfg.duration_s*cfg.fps)), cfg.fps, family)
                # Distinct simulated instances remain mechanics tests, never new
                # independent AMASS people or clinical evidence.
                shift = np.array([number*.015, 0, 0], dtype=np.float32)
                clean.joints += shift; clean.vertices += shift
                event.joints += shift; event.vertices += shift
            else:
                available_start = max(0, float(source.duration_s)-cfg.duration_s)
                start_s = min(available_start/2, 2.0)
                motion = data.load_motion(source, start_s, cfg.duration_s, cfg.fps)
                clean = body_model.forward(motion)
                event = body_model.forward(geometry.edit_motion(motion, family))
            magnitude = geometry.descriptor(event.joints, family)-geometry.descriptor(clean.joints, family)
            if abs(magnitude) < metrics.MIN_EVENT_DELTA[family]:
                skipped.append(dict(role=role, person_id=source.person_id, source=source.relative_path,
                                    reason="event_below_prespecified_descriptor_margin", event_magnitude=magnitude))
                continue
            yaw = {"train": 15, "calibration": 15, "development": 40, "final": -35}[role]
            target = clean.joints[:, 0].mean(axis=0) + [0, .15, 0]
            camera = Camera.look_at(target=target, width=cfg.image_size, height=cfg.image_size, yaw_degrees=yaw)
            appearance = ("textured", "textureless", "moving_shadow", "textured")[number % 4]
            occlusion = (.35, .4, .7, .8) if number % 4 == 3 else None
            scene_ids = {}
            for is_event, body in ((False, clean), (True, event)):
                sid = f"{stem}_event{int(is_event)}"
                rendered = render_sequence(body, camera, appearance, occlusion, seed=cfg.seed+number)
                np.savez_compressed(cfg.root / "scenes" / f"{sid}.npz", rgb=rendered.rgb,
                                    reference_flow=rendered.flow, reference_valid=rendered.flow_valid,
                                    foreground=rendered.foreground, joint_visible=rendered.joint_visible,
                                    camera_json=json.dumps(camera.as_dict()))
                scene_ids[is_event] = sid
            noise_kind = "drift" if role == "final" else ("burst", "oscillation")[(number // 4) % 2]
            joints = (7, 10) if family == "foot_clearance" else ((16, 18, 20) if family == "arm_leg_timing" else (12, 16, 17))
            noise = data.tracking_noise(clean.joints.shape, cfg.seed+number, noise_kind,
                                       amplitude_m=cfg.noise_std_m*(1+.3*(number%3)), joints=joints)
            rng = np.random.default_rng(cfg.seed+number)
            confidence = np.full(clean.joints.shape[:2], .8, np.float32)
            # The same missing-data mask is shared by every explanation and
            # does not tell the model which corruption or event is present.
            observed = np.ones(clean.joints.shape[:2], bool)
            if number % 5 == 4:
                observed[len(observed)//3:len(observed)//3+3, joints] = False
            meta = dict(motion_id=stem, person_id=source.person_id, role=role, source_dataset=source.source_dataset,
                        relative_path=source.relative_path, event_family=family)
            cases = data.make_observation_cases(clean.joints, event.joints, noise, meta, observed, confidence)
            # Full-input ambiguity fixture: both hidden worlds have the same
            # tracker, metadata and completely occluded RGB. Balanced labels.
            sid = f"{stem}_occluded"
            np.savez_compressed(cfg.root / "scenes" / f"{sid}.npz",
                                rgb=np.full((len(clean.joints),cfg.image_size,cfg.image_size,3),128,np.uint8),
                                reference_flow=np.zeros((len(clean.joints)-1,cfg.image_size,cfg.image_size,2),np.float32),
                                reference_valid=np.zeros((len(clean.joints)-1,cfg.image_size,cfg.image_size),bool),
                                foreground=np.zeros((len(clean.joints),cfg.image_size,cfg.image_size),bool),
                                joint_visible=np.zeros(clean.joints.shape[:2],bool), camera_json=json.dumps(camera.as_dict()))
            for original in cases[-2:].copy():
                duplicate = dict(original)
                duplicate["case_id"] = original["case_id"]+"_ambiguous"
                duplicate["fixture"] = "ambiguous"
                cases.append(duplicate)
            for case in cases:
                cid = _safe(case["case_id"])
                case_path = cfg.root / "cases" / f"{cid}.npz"
                np.savez_compressed(case_path, **{key: case[key] for key in
                                    ("raw", "truth", "clean", "event_reference", "observed", "confidence")},
                                    timestamps=clean.timestamps)
                scene_id = sid if case["fixture"] == "ambiguous" else scene_ids[case["event_present"]]
                records.append(dict(case_id=cid, role=role, person_id=source.person_id, identity=source.person_id,
                                    event_family=family, family=family, fixture=case["fixture"],
                                    event_present=case["event_present"], noise_present=case["noise_present"],
                                    source_dataset=source.source_dataset, relative_path=source.relative_path,
                                    scene_id=scene_id, path=str(case_path), corruption=noise_kind,
                                    appearance=appearance, fps=cfg.fps, event_magnitude=magnitude,
                                    mode=cfg.mode, render_seconds=perf_counter()-started))
            print(f"{role}: built {number+1}/{len(chosen)} motions ({len(cases)} cases), {perf_counter()-started:.1f}s", flush=True)
    result = pd.concat([existing, pd.DataFrame(records)], ignore_index=True) if len(existing) else pd.DataFrame(records)
    if not len(result):
        raise RuntimeError("No event passed the prespecified descriptor margin. Inspect source motions; do not infer erasure.")
    result.to_csv(index_path, index=False)
    if skipped:
        path = cfg.root / "excluded_motions.csv"
        pd.DataFrame(skipped).to_csv(path, index=False)
    return result.loc[result.role.isin(roles)].reset_index(drop=True)


def _fill_missing(raw, observed):
    result = np.array(raw, copy=True)
    times = np.arange(len(raw))
    for joint in range(raw.shape[1]):
        good = np.asarray(observed[:, joint], bool) & np.isfinite(raw[:, joint]).all(axis=-1)
        if not good.any():
            raise ValueError(f"Joint {joint} is entirely missing. Supply a training-fitted initializer before using the motion prior.")
        for axis in range(3):
            result[:, joint, axis] = np.interp(times, times[good], raw[good, joint, axis])
    return result


def _demo_flow(rgb):
    import cv2
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    def estimate(a,b):
        return cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
    return FlowResult(np.stack([estimate(a,b) for a,b in zip(gray[:-1],gray[1:])]),
                      np.stack([estimate(b,a) for a,b in zip(gray[:-1],gray[1:])]),
                      metadata={"model":"Farneback", "pretrained":False, "scope":"demo_only"})


def _camera(scene):
    values = json.loads(str(scene["camera_json"].item()))
    return Camera(np.asarray(values["eye"]), np.asarray(values["rotation"]), values["width"], values["height"], values["focal_px"])


def _propagate_flow(raw, camera, flow, observed=None, neighborhood_radius=3, reanchor=.2):
    """Propagate at the *current* estimate, with weak observed-track reanchoring.

    Local medians reject isolated flow outliers. Sampling the original raw path
    after propagation has moved away would integrate the wrong material's flow.
    Missing flow holds the current image position unless a new track measurement
    is available. Metric camera depth remains the permitted raw/imputed estimate;
    this is an image-transport baseline, not independent 3D reconstruction.
    """
    from .pretrained_models import bilinear_sample
    import warnings

    xy, depth = camera.project(raw)
    available = np.isfinite(xy).all(-1) & (depth > 0)
    if observed is not None:
        available &= np.asarray(observed, dtype=bool)
    axis = np.arange(-neighborhood_radius, neighborhood_radius+1)
    offsets = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    tracked = xy.copy()
    for t in range(len(raw)-1):
        neighborhood = tracked[t, :, None] + offsets[None]
        transport, support = bilinear_sample(flow.forward[t], neighborhood)
        endpoint = neighborhood+transport
        _, target_supported = bilinear_sample(flow.forward[t], endpoint)
        support &= target_supported
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            displacement = np.nanmedian(np.where(support[..., None], transport, np.nan), axis=1)
        reliable = (support.mean(axis=1) >= .5) & np.isfinite(displacement).all(-1)
        candidate = tracked[t] + np.nan_to_num(displacement)
        # An interpolated coordinate is not a fresh tracking observation.
        anchored = np.where(available[t+1, :, None],
                            (1-reanchor)*candidate+reanchor*xy[t+1], candidate)
        fallback = np.where(available[t+1, :, None], xy[t+1], tracked[t])
        tracked[t+1] = np.where(reliable[:, None], anchored, fallback)
    centered = tracked - [(camera.width-1)/2, (camera.height-1)/2]
    xyz = np.stack([centered[...,0]*depth/camera.focal_px,
                    -centered[...,1]*depth/camera.focal_px, depth], axis=-1)
    return (xyz @ camera.rotation + camera.eye).astype(np.float32)


def cache_predictions(cfg, roles=DEVELOPMENT_ROLES):
    """Cache actual frozen-prior reconstruction and estimated flow independently.

    Exact renderer flow is retained only for flow-error auditing. It is not used
    for training features, calibration, propagation or restoration predictions.
    """
    if "final" in roles:
        require_fitted(cfg)
    index = pd.read_csv(cfg.root / "cases.csv")
    index = index.loc[index.role.isin(roles)]
    if not len(index):
        raise ValueError("Build the requested role before extracting predictions")
    destination = cfg.root / "predictions" / _safe(cfg.prior_id)
    destination.mkdir(parents=True, exist_ok=True)
    settings={"mode":cfg.mode,"prior_backend":cfg.prior_backend,"prior_id":cfg.prior_id,
              "prior_checkpoint":_asset(Path(cfg.momask_checkpoint)/"model/net_best_fid.tar") if cfg.momask_checkpoint else None,
              "prior_predictions":_asset(cfg.prior_predictions),"flow_backend":cfg.flow_backend,
              "flow_checkpoint":_asset(cfg.flow_checkpoint),"flow_config":_asset(cfg.flow_config)}
    _same_settings(destination/"settings.json",settings)
    flow_folder = cfg.root / "flow" / _safe(cfg.flow_backend)
    flow_folder.mkdir(parents=True, exist_ok=True)
    _same_settings(flow_folder/"settings.json",{key:settings[key] for key in ("mode","flow_backend","flow_checkpoint","flow_config")})
    prior, estimator = None, None
    if cfg.mode == "real":
        if cfg.prior_backend == "momask":
            prior = MoMaskPrior(cfg.momask_repo, cfg.momask_checkpoint, cfg.target_skeleton_path, cfg.device)
        elif cfg.prior_backend != "external":
            raise ValueError("Real prior_backend must be momask or external. Demo smoothing is not a research prior.")
        estimator = OpticalFlowEstimator(cfg.flow_backend, cfg.flow_checkpoint, repo_dir=cfg.flow_repo,
                                         config_path=cfg.flow_config, device=cfg.device)
    records = []
    for number, row in index.iterrows():
        path = destination / f"{row.case_id}.npz"
        if path.is_file():
            old = _npz(path)
            records.append({**row.to_dict(), "prior_id": cfg.prior_id, "prior_backend": cfg.prior_backend,
                            "flow_backend": cfg.flow_backend, "cache_path": str(path),
                            "bridge_roundtrip_error_m": float(old["bridge_error"])})
            continue
        case, scene = _npz(row.path), _npz(cfg.root / "scenes" / f"{row.scene_id}.npz")
        filled = _fill_missing(case["raw"], case["observed"])
        if cfg.mode == "demo":
            result = PriorResult(gaussian_filter1d(filled, 1.5, axis=0), np.arange(len(filled)),
                                 bridge_roundtrip_joints=filled.copy(), bridge_roundtrip_error_m=0,
                                 metadata={"model":"DEMO_smoothing", "pretrained":False})
        elif cfg.prior_backend == "external":
            if not cfg.prior_predictions:
                raise ValueError("External prior needs MP_PRIOR_PREDICTIONS with per-case NPZ files")
            result = load_external_prior(Path(cfg.prior_predictions)/f"{row.case_id}.npz")
            if result.metadata["coordinate_system"] not in {"y_up", "world_y_up"}:
                raise ValueError("Export external prior in the same metric world_y_up coordinates")
        else:
            result = prior.reconstruct(filled)
        frames = result.frame_indices
        if len(frames) < 3 or frames[-1] >= len(filled) or not np.array_equal(np.diff(frames), np.ones(len(frames)-1)):
            raise ValueError("This 20 Hz experiment requires an aligned contiguous reconstruction window")
        training_file=cfg.root/"models/training.json"
        if cfg.prior_backend=="external" and training_file.exists():
            original=json.loads(training_file.read_text())["prior_id"]
            source=cfg.root/"predictions"/_safe(original)/f"{row.case_id}.npz"
            if not source.is_file():
                raise ValueError("Cache the original prior on these same cases before comparing a second prior")
            if not np.array_equal(frames,_npz(source)["frame_indices"]):
                raise ValueError("Second-prior frame_indices must exactly match the original prior's cached window")
        arrays = {key: value[frames] for key,value in case.items()}
        arrays["raw"] = filled[frames]
        arrays["prior"] = result.joints
        arrays["bridge"] = result.bridge_roundtrip_joints if result.bridge_roundtrip_joints is not None else filled[frames]
        arrays["bridge_error"] = np.asarray(result.bridge_roundtrip_error_m if result.bridge_roundtrip_error_m is not None else np.nan)
        arrays["frame_indices"] = frames
        arrays["bone_lengths"] = geometry.estimate_bone_lengths(arrays["raw"], arrays["observed"])
        rgb = scene["rgb"][frames]
        flow_path = flow_folder / f"{row.scene_id}_{frames[0]}_{frames[-1]}.npz"
        if flow_path.exists():
            saved = _npz(flow_path)
            flow = FlowResult(saved["forward"], saved["backward"], saved.get("uncertainty"))
        else:
            flow = _demo_flow(rgb) if cfg.mode == "demo" else estimator.estimate(rgb)
            payload = dict(forward=flow.forward, backward=flow.backward)
            if flow.uncertainty is not None:
                payload["uncertainty"] = flow.uncertainty
            np.savez_compressed(flow_path, **payload)
        camera = _camera(scene)
        raw_xy, _ = camera.project(arrays["raw"])
        prior_xy, _ = camera.project(arrays["prior"])
        # Observed confidence and RGB-based diagnostics are inputs. Renderer
        # visibility would be privileged truth, so it is not supplied here.
        rd = joint_transport_diagnostics(flow, raw_xy, rgb=rgb, valid=arrays["observed"])
        pdg = joint_transport_diagnostics(flow, prior_xy, rgb=rgb)
        arrays["features"] = learning.make_features(arrays["raw"], arrays["prior"], arrays["confidence"], arrays["observed"], rd, pdg, cfg.image_size)
        pad = lambda a: np.concatenate([a,np.zeros_like(a[:1])],axis=0)
        arrays["raw_transport"] = pad(rd["transport_error_px"])
        arrays["prior_transport"] = pad(pdg["transport_error_px"])
        arrays["flow_valid"] = pad(rd["evidence_valid"] & pdg["evidence_valid"])
        arrays["flow_propagated"] = _propagate_flow(arrays["raw"], camera, flow, arrays["observed"])
        arrays["event_support"] = (np.linalg.norm(arrays["event_reference"]-arrays["clean"],axis=-1)>.005).astype(np.float32)
        ref = scene["reference_flow"][frames[:-1]]
        visible = scene["reference_valid"][frames[:-1]]
        error = np.linalg.norm(flow.forward-ref,axis=-1)
        arrays["flow_reference_epe"] = np.asarray(np.mean(error[visible]) if visible.any() else np.nan)
        arrays["foreground_fraction"] = np.asarray(scene["foreground"][frames].mean())
        arrays["flow_energy"] = np.asarray(np.linalg.norm(flow.forward,axis=-1).mean())
        arrays["prior_metadata_json"] = np.asarray(json.dumps(result.metadata))
        np.savez_compressed(path, **arrays)
        records.append({**row.to_dict(), "prior_id":cfg.prior_id, "prior_backend":cfg.prior_backend,
                        "flow_backend":cfg.flow_backend, "cache_path":str(path),
                        "bridge_roundtrip_error_m":float(arrays["bridge_error"])})
        if len(records)%8 == 0:
            print(f"Cached {len(records)}/{len(index)} cases with {cfg.prior_id}",flush=True)
    previous = pd.read_csv(destination/"index.csv") if (destination/"index.csv").exists() else pd.DataFrame()
    output = pd.concat([previous,pd.DataFrame(records)],ignore_index=True).drop_duplicates("case_id",keep="last")
    output.to_csv(destination/"index.csv",index=False)
    return output.loc[output.role.isin(roles)].reset_index(drop=True)


def _load_cases(cfg, role):
    path = cfg.root / "predictions" / _safe(cfg.prior_id) / "index.csv"
    if not path.exists():
        raise FileNotFoundError("Run notebook02 frozen extraction first")
    index = pd.read_csv(path)
    index = index.loc[index.role.eq(role)].reset_index(drop=True)
    if not len(index):
        raise ValueError(f"No cached {role} cases. Check excluded_motions.csv and the available person groups.")
    cases = [_npz(row.cache_path) for row in index.itertuples()]
    return index, cases


def _stack(index, cases):
    keys = ("features", "raw", "prior", "truth", "bone_lengths", "observed", "event_support")
    try:
        result = {key: np.stack([case[key] for case in cases]) for key in keys}
    except ValueError as exc:
        raise ValueError("Use one common duration and contiguous frame selection across cases") from exc
    result["event_present"] = index.event_present.astype(float).to_numpy(copy=True)
    return result


def train_gate(cfg):
    """Train the small gate and its coordinate/random/shuffled controls on train only."""
    import joblib
    if (cfg.root / "final_opened.json").exists():
        raise RuntimeError("This run has opened its final evaluation. Use a new run root for further fitting.")
    index, cases = _load_cases(cfg, "train")
    arrays = _stack(index, cases)
    folder = cfg.root / "models"
    folder.mkdir(exist_ok=True)
    cheap = learning.fit_linear_flow_gate(arrays)
    joblib.dump(cheap, folder/"linear_flow_gate.joblib")
    histories, models = [], []
    for mode in cfg.feature_modes:
        for seed in cfg.seeds:
            name = f"gate_{mode}_seed{seed}"
            payload, history = learning.fit_gate(arrays, seed=seed, mode=mode, epochs=cfg.epochs,
                                                 batch_size=cfg.batch_size, hidden_dim=cfg.hidden_dim,
                                                 learning_rate=cfg.learning_rate, device=cfg.device)
            torch.save(payload, folder/f"{name}.pt")
            histories.extend(history)
            models.append(name)
            print(f"Fitted {name}: final loss={history[-1]['loss']:.4f}",flush=True)
    training = dict(created=datetime.now(timezone.utc).isoformat(), prior_id=cfg.prior_id,
                    mode=cfg.mode, models=models, train_people=sorted(index.person_id.unique()),
                    input_features=arrays["features"].shape[-1], frame_count=arrays["raw"].shape[1],
                    seeds=cfg.seeds, primary_method=f"gate_full_seed{cfg.seeds[0]}",
                    config=asdict(cfg))
    if training["primary_method"] not in models:
        raise ValueError("Include feature_modes=['full', ...] so the primary method is prespecified")
    _json(folder/"training.json",training)
    result = pd.DataFrame(histories)
    result.to_csv(folder/"training_history.csv",index=False)
    return result


def _predictions(cfg, index, cases):
    """Return candidate trajectories and logits; truth is used only by labeled oracle."""
    import joblib
    candidates, logits = {}, {}
    base = [learning.baseline_candidates(case) for case in cases]
    for method in base[0]:
        candidates[method] = [value[method] for value in base]
    features = np.stack([c["features"] for c in cases])
    model_path = cfg.root / "models/training.json"
    if model_path.exists():
        training = json.loads(model_path.read_text())
        if features.shape[-1] != training["input_features"]:
            raise ValueError("Cached feature layout differs from the trained gate")
        if features.shape[1] != training["frame_count"]:
            raise ValueError("Cross-prior comparisons must use the same frame window as the trained gate")
        cheap = joblib.load(cfg.root / "models/linear_flow_gate.joblib")
        weights, prediction_logits = learning.infer_linear_flow_gate(cheap, features)
        candidates["calibrated_flow_gate"] = [geometry.project_bone_lengths(
            c["prior"]+(w*c["observed"])[...,None]*(c["raw"]-c["prior"]),c["bone_lengths"]) for c,w in zip(cases,weights)]
        logits["calibrated_flow_gate"] = prediction_logits
        for name in training["models"]:
            payload = torch.load(cfg.root / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
            weights, prediction_logits = learning.infer_gate(payload,features,device=cfg.device)
            candidates[name] = [geometry.project_bone_lengths(
                c["prior"]+(w*c["observed"])[...,None]*(c["raw"]-c["prior"]),c["bone_lengths"]) for c,w in zip(cases,weights)]
            logits[name] = prediction_logits
    for method, directory in cfg.external_methods.items():
        if method in candidates or method == "oracle_mixture":
            raise ValueError(f"External method name collides with implemented method {method}")
        values = []
        for row,case in zip(index.itertuples(),cases):
            result = load_external_prior(Path(directory)/f"{row.case_id}.npz")
            if not np.array_equal(result.frame_indices,case["frame_indices"]):
                raise ValueError(f"External {method} prediction must align every evaluated frame")
            if result.metadata["coordinate_system"] not in {"y_up","world_y_up"}:
                raise ValueError("External comparisons must use metric world_y_up coordinates")
            values.append(geometry.project_bone_lengths(result.joints,case["bone_lengths"]))
        candidates[method] = values
    oracle = []
    for case in cases:
        delta = case["raw"]-case["prior"]
        weight = np.clip(np.sum((case["truth"]-case["prior"])*delta,axis=-1) /
                         np.maximum(np.sum(delta**2,axis=-1),1e-8),0,1)*case["observed"]
        oracle.append(geometry.project_bone_lengths(case["prior"]+weight[...,None]*delta,case["bone_lengths"]))
    candidates["oracle_mixture"] = oracle
    return candidates, logits


def baseline_report(cfg, split="development"):
    """Pre-training diagnostic at full strength. It is not a matched-repair claim."""
    index,cases = _load_cases(cfg,split)
    rows=[]
    for (_,record),case in zip(index.iterrows(),cases):
        for method,prediction in learning.baseline_candidates(case).items():
            rows.append(metrics.score_case(prediction,case,record,method))
    scores=pd.DataFrame(rows)
    return dict(scores=scores,summary=metrics.summarize(scores),decision={"status":"diagnostic_unmatched_strength","mode":cfg.mode})


def calibrate(cfg):
    """Choose operating points and probability thresholds on calibration people."""
    if (cfg.root / "final_opened.json").exists():
        raise RuntimeError("Final evaluation was opened. Start a new run before changing calibration.")
    training_path=cfg.root / "models/training.json"
    if not training_path.is_file():
        raise RuntimeError("Train the gate before calibrating it")
    training=json.loads(training_path.read_text())
    if cfg.prior_id != training["prior_id"]:
        raise ValueError("Calibrate on the training prior only. Cross-prior evaluation keeps this calibration fixed.")
    index,cases=_load_cases(cfg,"calibration")
    candidates,logits=_predictions(cfg,index,cases)
    probability={method:metrics.calibrate_probability(value,index.event_present.astype(float)) for method,value in logits.items()}
    rows=[]
    for method,values in candidates.items():
        fixed={"raw","oracle_mixture","prior_unprojected","conversion_only_unprojected"}
        grid=[1.] if method in fixed else cfg.strength_grid
        for strength in grid:
            for i,((_,record),case,value) in enumerate(zip(index.iterrows(),cases,values)):
                prediction=value if method in fixed else learning.mix_at_strength(case["raw"],value,strength,case["bone_lengths"],case["observed"])
                params=probability.get(method)
                p=expit(params["a"]*logits[method][i]+params["b"]) if params else np.nan
                rows.append(metrics.score_case(prediction,case,record,method,strength,p,params["threshold"] if params else .5))
    scores=pd.DataFrame(rows)
    points=metrics.lock_operating_points(scores,cfg.target_noise_removal)
    # The comparator is selected before development outcomes are seen. Other
    # strengths and methods remain visible as diagnostics, not retrospective wins.
    comparator=points.loc[~points.method.str.startswith("gate_full_") & ~points.method.eq("oracle_mixture") & points.feasible]
    if len(comparator):
        best=str(comparator.sort_values("calibration_retention",ascending=False).iloc[0].method)
    else:
        best="calibrated_flow_gate"
    locked=dict(training_created=training["created"],primary_method=training["primary_method"],
                comparator=best,prior_id=cfg.prior_id,probability=probability,
                target_noise_removal=cfg.target_noise_removal,removal_tolerance=cfg.removal_tolerance,
                retention_gain=cfg.retention_gain,mode=cfg.mode,
                calibration_people=sorted(index.person_id.unique()), points=points.to_dict("records"))
    _json(cfg.root / "calibration/locked.json",locked)
    points.to_csv(cfg.root / "calibration/operating_points.csv",index=False)
    scores.to_csv(cfg.root / "calibration/strength_curve.csv",index=False)
    return points


def _shortcut_audit(cfg, index, cases):
    """Fit a nuisance-only event classifier on training matched pairs, then test.

    High performance is a warning to redesign or match the construction. This
    test never certifies the absence of all possible shortcuts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    def features(table,values):
        rows=[];labels=[];people=[]
        for record,case in zip(table.itertuples(),values):
            if record.fixture!="matched": continue
            raw=case["raw"]
            rows.append([len(raw),float(np.linalg.norm(raw[-1,0]-raw[0,0])),
                         float(np.mean(np.linalg.norm(np.diff(raw,axis=0),axis=-1))),
                         float(case["foreground_fraction"]),float(case["flow_energy"]),
                         float(case["confidence"].mean()),float(case["observed"].mean())])
            labels.append(int(record.event_present));people.append(record.person_id)
        return np.asarray(rows),np.asarray(labels),np.asarray(people)
    train_index,train_cases=_load_cases(cfg,"train")
    xt,yt,_=features(train_index,train_cases)
    x,y,people=features(index,cases)
    if len(set(yt))<2 or len(set(y))<2:
        return {"status":"insufficient_cases"}
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500)).fit(xt,yt)
    predicted=model.predict(x)
    per_person=[balanced_accuracy_score(y[people==p],predicted[people==p]) for p in np.unique(people)]
    accuracy=float(np.mean(per_person))
    return dict(status="warning_match_nuisances" if accuracy>.7 else "no_observed_shortcut_in_this_probe",
                balanced_accuracy=accuracy,n_people=len(per_person),
                features=["duration","centroid_drift","joint_speed","foreground_fraction","flow_energy","confidence","missingness"])


def evaluate(cfg, split="development"):
    """Apply locked calibration, report all conditions and evaluate the stop rule."""
    if split not in {"development","final"}:
        raise ValueError("Evaluate development or final; fit uses train and calibration separately")
    training,locked=require_fitted(cfg)
    index,cases=_load_cases(cfg,split)
    candidates,logits=_predictions(cfg,index,cases)
    points={row["method"]:row for row in locked["points"]}
    rows=[]
    for method,values in candidates.items():
        if method not in points:
            raise ValueError(f"{method} has no saved calibration. Add methods before opening final evaluation.")
        strength=points[method]["strength"]
        params=locked["probability"].get(method)
        for i,((_,record),case,value) in enumerate(zip(index.iterrows(),cases,values)):
            prediction=value if method in {"raw","oracle_mixture","prior_unprojected","conversion_only_unprojected"} else learning.mix_at_strength(case["raw"],value,strength,case["bone_lengths"],case["observed"])
            p=expit(params["a"]*logits[method][i]+params["b"]) if params else np.nan
            rows.append(metrics.score_case(prediction,case,record,method,strength,p,params["threshold"] if params else .5))
    scores=pd.DataFrame(rows)
    summary=metrics.summarize(scores)
    primary,comparator=locked["primary_method"],locked["comparator"]
    interval=metrics.paired_person_interval(scores,primary,comparator,samples=cfg.bootstrap_samples,seed=cfg.seed)
    by_method=summary.set_index("method")
    a,b=by_method.loc[primary],by_method.loc[comparator]
    repair_match=bool(abs(a.noise_removal-b.noise_removal)<=locked["removal_tolerance"])
    useful_repair=bool(a.noise_removal>=locked["target_noise_removal"] and b.noise_removal>=locked["target_noise_removal"])
    useful_gain=bool(interval["mean"]>=locked["retention_gain"] and interval["low"]>0)
    same_clip=bool(a.event_and_noise_removal>=locked["target_noise_removal"] and
                   abs(a.event_and_noise_removal-b.event_and_noise_removal)<=locked["removal_tolerance"])
    seed_gains={name:float(by_method.loc[name,"retention"]-b.retention) for name in training["models"]
                if name.startswith("gate_full_")}
    seed_consistent=all(value>0 for value in seed_gains.values())
    shortcut=_shortcut_audit(cfg,index,cases) if cfg.prior_id==training["prior_id"] else {"status":"primary_prior_audit_applies"}
    status="continue_pilot" if repair_match and useful_repair and useful_gain and same_clip and seed_consistent else "development_stop"
    if shortcut["status"]=="warning_match_nuisances":
        status="redesign_nuisance_matching"
    if cfg.mode=="demo":
        status="demo_only_no_research_decision"
    decision=dict(status=status,split=split,mode=cfg.mode,prior_id=cfg.prior_id,trained_prior=training["prior_id"],
                  primary=primary,calibration_selected_comparator=comparator,retention_difference=interval,
                  matched_achieved_repair=repair_match,minimum_repair_met=useful_repair,
                  primary_noise_removal=float(a.noise_removal),comparator_noise_removal=float(b.noise_removal),
                  same_clip_event_and_noise_repair=same_clip,seed_retention_gains=seed_gains,
                  final_shift="event family, camera and corruption jointly held out" if split=="final" else None,
                  oracle_note="Reference-informed pointwise mixture before projection; not a certified optimum after projection.",
                  shortcut_probe=shortcut,seed_methods=[m for m in training["models"] if m.startswith("gate_full_")],
                  interpretation="A pilot decision, not evidence of clinical validity or publication readiness.",
                  missing_exact_comparisons=[m for m in ("HTD-Refine","MFTIQ","H-MoRe","two_tracker") if m not in cfg.external_methods])
    folder=cfg.root / "results"
    folder.mkdir(exist_ok=True)
    label=split if cfg.prior_id==training["prior_id"] else f"{split}_{_safe(cfg.prior_id)}"
    scores.to_csv(folder/f"{label}_scores.csv",index=False)
    summary.to_csv(folder/f"{label}_summary.csv",index=False)
    _json(folder/f"{label}_decision.json",decision)
    return dict(scores=scores,summary=summary,decision=decision)


def gavd_stress(cfg):
    from .gavd_stress import gavd_stress as inspect
    return inspect(cfg)
