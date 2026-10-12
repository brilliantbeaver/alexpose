"""Audited source motion to paired RGB, projected references and fresh MMPose tracks.

Knee edits are bounded kinematic stress tests. They are not simulations of a
disease. Physical edits are always re-forwarded through SMPL-H and rendered;
only the explicitly labeled naming interventions act on extracted coordinates.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .data import TrackBundle, apply_naming, fixture_bundle, save_dataset, load_dataset, merge_disk_datasets
from ..synthetic_training_v2.contracts import atomic_json, digest, array_digest, sha256_file

BODY_SWAP = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20])


def technical_geometry_screen(body):
    """Check usable geometry without demanding normal or symmetric movement.

    Ankle excursion, anticorrelation and alternating crossings are descriptive
    diagnostics only in the full-manifest study. A stationary or asymmetric
    interval is not rejected because it differs from normal treadmill walking.
    """
    joints, vertices = np.asarray(body.joints), np.asarray(body.vertices)
    failures = []
    if joints.ndim != 3 or joints.shape[1:] != (22, 3) or len(joints) < 8:
        failures.append('invalid_joint_shape')
    if not np.isfinite(joints).all() or not np.isfinite(vertices).all():
        failures.append('nonfinite_geometry')
    if not failures:
        lengths = np.stack([np.linalg.norm(joints[:, a] - joints[:, b], axis=-1)
                            for a, b in ((1, 2), (1, 4), (4, 7), (2, 5), (5, 8))])
        if np.any(lengths < 1e-5): failures.append('degenerate_lower_body_geometry')
    return dict(passed=not failures, failures=failures,
        screen_version='gf-technical-geometry-v1', reviewed_by='algorithm',
        locomotion_diagnostics=locomotion_screen(joints),
        limitation='Technical geometry checks only; motion labels and clinical status are not validated')


def locomotion_screen(joints):
    """Apply the inherited heuristic rules to every 64-frame part of a window.

    These are the published v2 development-screen bounds, reproduced locally to
    remove a script-directory import dependency. They are not a validated gait
    classifier or independent human review.
    """
    x = np.asarray(joints)
    if x.ndim != 3 or x.shape[1:] != (22, 3) or len(x) < 64 or not np.isfinite(x).all():
        return dict(passed=False, failures=["nonfinite_or_short_geometry"], segments=[])
    starts = sorted(set(list(range(0, len(x) - 63, 64)) + [len(x) - 64]))
    segments = []
    bounds = dict(hip_width_m=[.08, .60], leg_segment_m=[.15, .85],
                  pelvis_head_height_m=[.25, 1.5], ankle_excursion_m_min=.08,
                  ankle_correlation_max=-.15, alternating_crossings=[2, 16],
                  ankle_vertical_excursion_m_max=.65)
    span = lambda v: float(np.percentile(v, 95) - np.percentile(v, 5))
    for start in starts:
        q = x[start:start + 64]
        lateral = q[0, 2] - q[0, 1]; lateral[1] = 0
        length = np.linalg.norm(lateral)
        if length < 1e-6:
            segments.append(dict(start_frame=start, passed=False, failures=["degenerate_initial_hips"], metrics={})); continue
        forward = np.cross(lateral / length, [0., 1., 0.])
        ankles = (q[:, [7, 8]] - q[:, [0]]) @ forward
        excursions = [span(ankles[:, j]) for j in (0, 1)]
        corr = float(np.corrcoef(ankles.T)[0, 1]) if min(excursions) > 1e-8 else 1.
        diff = ankles[:, 0] - ankles[:, 1]
        signs = np.sign(diff[np.abs(diff) > .02])
        crosses = int(np.sum(signs[1:] != signs[:-1]))
        hip = np.linalg.norm(q[:, 1] - q[:, 2], axis=-1)
        legs = np.stack([np.linalg.norm(q[:, a] - q[:, b], axis=-1) for a, b in ((1, 4), (4, 7), (2, 5), (5, 8))])
        height = float(np.median(q[:, 15, 1] - q[:, 0, 1]))
        vertical = [span(q[:, j, 1]) for j in (7, 8)]
        failures = []
        if hip.min() < .08 or hip.max() > .60: failures.append("hip_width")
        if legs.min() < .15 or legs.max() > .85: failures.append("leg_segment")
        if not .25 <= height <= 1.5: failures.append("pelvis_head_height")
        if min(excursions) < .08: failures.append("ankle_excursion")
        if not np.isfinite(corr) or corr > -.15: failures.append("ankle_correlation")
        if not 2 <= crosses <= 16: failures.append("alternating_crossings")
        if max(vertical) > .65: failures.append("ankle_vertical_excursion")
        metrics = dict(hip_width_min_m=float(hip.min()), hip_width_max_m=float(hip.max()),
                       leg_segment_min_m=float(legs.min()), leg_segment_max_m=float(legs.max()),
                       pelvis_head_height_m=height, ankle_excursion_m=excursions,
                       ankle_correlation=corr if np.isfinite(corr) else None,
                       alternating_crossings=crosses, ankle_vertical_excursion_m=vertical)
        segments.append(dict(start_frame=start, passed=not failures, failures=failures, metrics=metrics))
    return dict(passed=all(s["passed"] for s in segments),
                failures=sorted({f for s in segments for f in s["failures"]}), segments=segments,
                thresholds=bounds, screen_version="gf-extended-kinematic-screen-v1",
                limitation="Algorithmic development screen; human full-interval review pending")


def select_intervals(rows, samples, hz):
    """Extend the same frozen motion roster with deterministic nonoverlap starts."""
    candidates, exclusions, end_by_motion = [], [], {}
    for original in rows:
        row = dict(original)
        parent_start = float(row["start_s"])
        start = max(parent_start, end_by_motion.get(row["relative_path"], -np.inf))
        row["parent_audited_start_s"] = parent_start
        row["start_s"] = start
        if start + (samples - 1) / hz > float(row["duration_s"]) + 1e-8:
            exclusions.append(dict(relative_path=row["relative_path"], parent_start_s=parent_start,
                                   attempted_start_s=start, reason="insufficient_duration_for_nonoverlap_interval"))
            continue
        end_by_motion[row["relative_path"]] = start + samples / hz
        candidates.append(row)
    return candidates, exclusions


def mirror_body(body, *, origin=None, normal=None):
    """Reflect geometry and anatomical sides; reverse triangle winding for normals."""
    origin = np.median(body.joints[:, 0], axis=0) if origin is None else np.asarray(origin)
    normal = body.joints[0, 2] - body.joints[0, 1] if normal is None else np.array(normal, float)
    normal[1] = 0
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Mirror plane needs a nondegenerate horizontal pelvis axis")
    normal = normal / length
    transform = np.eye(3) - 2 * np.outer(normal, normal)
    reflect = lambda x: ((np.asarray(x) - origin) @ transform.T + origin).astype(np.float32)
    return replace(body, joints=reflect(body.joints)[:, BODY_SWAP], vertices=reflect(body.vertices),
                   faces=body.faces[:, ::-1].copy(),
                   metadata={**body.metadata, "physical_mirror": True, "mirror_normal": normal.tolist(),
                             "mirror_origin": origin.tolist()})


def knee_intervention(motion, reference_body, magnitude_deg, *, side="right"):
    """Compose a local flexion rotation during reference-supported swing frames.

    The gate is zero for the lower 40% of ankle heights and rises smoothly over
    the next 40%; a cosine window leaves interval boundaries unchanged. Contacts
    are subsequently checked on the resulting geometry, not assumed from this
    gate. Zero magnitude returns an exact parameter copy.
    """
    if side not in {"left", "right"} or not np.isfinite(magnitude_deg) or not 0 <= magnitude_deg <= 20:
        raise ValueError("Specify left/right and a finite kinematic edit in [0,20] degrees")
    knee, ankle = (4, 7) if side == "left" else (5, 8)
    z = reference_body.joints[:, ankle, 1]
    lower, upper = np.quantile(z, [.4, .8])
    gate = np.clip((z - lower) / max(float(upper - lower), 1e-5), 0, 1)
    gate = gate * gate * (3 - 2 * gate)
    gate *= np.sin(np.linspace(0, np.pi, len(gate))) ** 2
    poses = motion.poses.copy().reshape(len(motion.poses), 52, 3)
    if magnitude_deg:
        delta = np.zeros((len(poses), 3)); delta[:, 0] = np.deg2rad(magnitude_deg) * gate
        poses[:, knee] = (Rotation.from_rotvec(poses[:, knee]) * Rotation.from_rotvec(delta)).as_rotvec()
    return replace(motion, poses=poses.reshape(-1, 156), metadata={**motion.metadata,
        "intervention": "swing_gated_local_knee_flexion", "side": side,
        "magnitude_deg": float(magnitude_deg), "gate": gate.tolist(),
        "claim": "kinematic stress test; no dynamic or clinical validation"})


def geometry_screen(reference, changed, *, max_contact_displacement_m=.025,
                    max_extra_penetration_m=.015, max_angle_deg=178.,
                    max_joint_step_m=.20, minimum_changed_mm=.01):
    """Reference-only geometric checks with retained diagnostics and fixed bounds."""
    a, b = np.asarray(reference.joints), np.asarray(changed.joints)
    if a.shape != b.shape or a.shape[1:] != (22, 3) or not np.array_equal(reference.timestamps, changed.timestamps):
        raise ValueError("Geometry screens require paired 22-joint arrays on the same physical clock")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(changed.vertices).all():
        return dict(pass_=False, reasons=["nonfinite_geometry"])
    failures = []
    angles = []
    for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
        u, v = b[:, hip] - b[:, knee], b[:, ankle] - b[:, knee]
        lengths = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        if np.any(lengths < 1e-8): failures.append("degenerate_leg")
        interior = np.rad2deg(np.arccos(np.clip(np.sum(u * v, -1) / np.maximum(lengths, 1e-8), -1, 1)))
        # Interior angles near 180 degrees are straight knees and legitimate;
        # the fixed screen limits pathological over-folding to >=2 degrees.
        flexion = 180 - interior
        angles.extend(flexion.tolist())
        if np.any(flexion > max_angle_deg): failures.append("excessive_geometric_knee_flexion")
    displacement = np.linalg.norm(b - a, axis=-1)
    contact_displacements = []
    for ankle in (7, 8):
        floor = np.quantile(a[:, ankle, 1], .1)
        low = a[:, ankle, 1] <= floor + .02
        contact_displacements.extend(displacement[low, ankle].tolist())
    contact = max(contact_displacements, default=0.)
    if contact > max_contact_displacement_m: failures.append("reference_low_foot_displacement")
    penetration = float(max(0., np.min(reference.vertices[..., 1]) - np.min(changed.vertices[..., 1])))
    if penetration > max_extra_penetration_m: failures.append("additional_ground_penetration")
    step = float(np.max(np.linalg.norm(np.diff(b, axis=0), axis=-1)))
    if step > max_joint_step_m: failures.append("joint_step_discontinuity")
    changed_mm = float(np.max(displacement) * 1000)
    if minimum_changed_mm and changed_mm < minimum_changed_mm: failures.append("intervention_had_no_measurable_effect")
    return dict(pass_=not failures, reasons=sorted(set(failures)),
                max_reference_low_foot_displacement_m=contact, extra_ground_penetration_m=penetration,
                max_joint_step_m=step, max_joint_change_mm=changed_mm,
                max_geometric_knee_flexion_deg=float(max(angles)),
                thresholds=dict(max_contact_displacement_m=max_contact_displacement_m,
                                max_extra_penetration_m=max_extra_penetration_m,
                                max_angle_deg=max_angle_deg, max_joint_step_m=max_joint_step_m,
                                minimum_changed_mm=minimum_changed_mm),
                limitation="Low-foot proxy is not force-measured contact; geometric checks do not establish dynamics")


def _render_fixed(renderer, body, recipe, seed, camera, original_faces):
    """Use the established textured renderer with one externally frozen camera.

    Rendering is reproduced here to avoid monkey-patching a shared historical
    module. UV triangle corners follow the reversed mesh winding under reflection.
    """
    import pyrender
    import trimesh
    from PIL import Image, ImageOps
    from ..synthetic_training.rendering import load_uv_topology, project_points, SMPL_INDICES
    rng = np.random.default_rng(seed)
    texture_path = renderer.textures[int(rng.integers(len(renderer.textures)))]
    background_path = renderer.backgrounds[int(rng.integers(len(renderer.backgrounds)))]
    with Image.open(texture_path) as im: texture = im.convert("RGB").copy()
    with Image.open(background_path) as im: background = np.asarray(ImageOps.fit(im.convert("RGB"), (renderer.width, renderer.height))).copy()
    uv, uv_faces = load_uv_topology(renderer.uv_path, original_faces)
    if body.metadata.get("physical_mirror"):
        uv_faces = uv_faces[:, ::-1]
    yfov = np.deg2rad(50.)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[.35] * 3)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=renderer.width / renderer.height, znear=.05), pose=camera)
    light_pose = camera.copy(); light_pose[:3, 3] += [1.5, 2., 0.]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=light_pose)
    if renderer._renderer is None: renderer._renderer = pyrender.OffscreenRenderer(renderer.width, renderer.height)
    joints, depth = project_points(body.joints[:, SMPL_INDICES], camera, renderer.width, renderer.height, yfov)
    projected, vertex_depth = project_points(body.vertices, camera, renderer.width, renderer.height, yfov)
    if np.any(vertex_depth <= .05): raise ValueError("Paired fixed camera puts body behind near plane")
    blocker = np.zeros((renderer.height, renderer.width), bool)
    if recipe.occlusion_fraction:
        # One fixed image-space obstacle for every physical movement level.
        # Computing its width from each edited mesh would change two factors.
        half = max(1, int(renderer.width * recipe.occlusion_fraction / 2))
        center = renderer.width // 2
        blocker[renderer.height // 2:, max(0, center - half):min(renderer.width, center + half)] = True
    rgbs, masks, boxes = [], [], []
    triangles = np.arange(body.faces.size).reshape(-1, 3)
    for frame, vertices in enumerate(body.vertices):
        base = trimesh.Trimesh(vertices=vertices, faces=body.faces, process=False)
        mesh = trimesh.Trimesh(vertices=vertices[body.faces.ravel()], faces=triangles,
            vertex_normals=base.vertex_normals[body.faces.ravel()], process=False,
            visual=trimesh.visual.texture.TextureVisuals(uv=uv[uv_faces.ravel()], image=texture))
        node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
        rgba, zbuffer = renderer._renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(node)
        foreground = zbuffer > 0
        if not foreground.any(): raise ValueError(f"No rendered body pixels in frame {frame}")
        yy, xx = np.where(foreground)
        if min(xx.min(), yy.min()) <= 0 or xx.max() >= renderer.width - 1 or yy.max() >= renderer.height - 1:
            raise ValueError("Fixed camera clipped a physical variant")
        pixels = background.copy(); pixels[foreground] = rgba[..., :3][foreground]
        pixels[blocker] = np.roll(background, renderer.width // 3, axis=1)[blocker]
        visible = np.zeros(12, bool)
        for j, (x, y) in enumerate(joints[frame]):
            ix, iy = int(round(float(x))), int(round(float(y)))
            if 0 <= ix < renderer.width and 0 <= iy < renderer.height and not blocker[iy, ix]:
                nearby = zbuffer[max(0, iy - 1):iy + 2, max(0, ix - 1):ix + 2]
                nearby = nearby[nearby > 0]
                visible[j] = bool(len(nearby) and abs(float(depth[frame, j] - nearby.min())) < .16)
        rgbs.append(pixels); masks.append(visible)
        boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
    return dict(images=np.stack(rgbs), keypoints=joints, visible=np.stack(masks),
                boxes=np.asarray(boxes, np.float32), camera_pose=camera,
                texture_path=str(texture_path), background_path=str(background_path),
                visibility_reference="synthetic_depth_proxy_0.16m_not_real_annotation",
                landmark_convention="projected_smplh_joint_centers_approximate_coco_body12")


def _source_rows(config):
    if config.get('data', {}).get('source_selection') == 'repair_benchmark':
        from .repair_profile import load_benchmark_cohort
        return load_benchmark_cohort(config)
    if config.get('data', {}).get('source_selection') == 'repair_confirmation':
        from .repair_cohort import load_slim_cohort
        return load_slim_cohort(config)
    if config.get('data', {}).get('source_selection', 'legacy_roster') == 'full_manifest':
        from .cohort import load_cohort
        return load_cohort(config, partition=config['data'].get('partition', 'development'))
    from ..synthetic_training_v2.preparation import audited_motion_windows
    prep = config["preparation"]
    old_path = Path(config["source_bundle"]) / "manifest.json"
    old = json.loads(old_path.read_text())
    if any(r["split"] not in {"train", "development"} for r in old["records"]):
        raise ValueError("Historical roster must contain training/development only")
    old_roles, old_motion_hashes = {}, {}
    for r in old["records"]:
        if old_roles.setdefault(r["canonical_person_id"], r["split"]) != r["split"]:
            raise ValueError("Historical roster contains person leakage")
        if old_motion_hashes.setdefault(r["relative_path"], r["motion_hash"]) != r["motion_hash"]:
            raise ValueError("Historical motion has conflicting content hashes")
    table, rejected = audited_motion_windows(prep["manifest_dir"], prep["amass_root"],
        prep["locomotion_audit"], prep["reservation_csv"], review_mode=prep.get("review_mode", "automated_development"))
    old_windows = {(r["relative_path"], float(r["start_s"])) for r in old["records"]}
    selected = table.loc[[(r["relative_path"], float(r["start_s"])) in old_windows for r in table.to_dict("records")]].copy()
    if set(selected.canonical_person_id) != set(old_roles):
        raise ValueError("Source audit no longer covers the exact frozen historical person roster")
    for r in selected.to_dict("records"):
        if r["split"] != old_roles[r["canonical_person_id"]]: raise ValueError("Historical person role changed")
    selected["historical_motion_sha256"] = selected.relative_path.map(old_motion_hashes)
    return selected.sort_values(["canonical_person_id", "relative_path", "start_s"]), rejected, sha256_file(old_path)


def prepare(config, output):
    """Prepare one deterministic shard; no real data are replaced by fixtures.

    Full-manifest runs use a frozen identity-aware interval plan. The explicit
    legacy mode retains the old v2 roster. Source-family chunks bound memory;
    synthetic variants never increase the number of independent people.
    """
    output = Path(output)
    if output.exists(): raise FileExistsError("Preparation outputs are immutable; choose a fresh attempt directory")
    output.mkdir(parents=True)
    options = config.get("data", {})
    full_manifest = options.get('source_selection', 'legacy_roster') in {'full_manifest', 'repair_confirmation', 'repair_benchmark'}
    confirmation = options.get('partition', 'development') == 'confirmation'
    samples, hz = int(options.get("samples", 128)), float(options.get("hz", 25.))
    if config.get("mode") == "fixture":
        bundle = fixture_bundle(int(config.get("seed", 17)), samples=samples,
                                people=int(options.get("fixture_people", 5)), hz=hz)
        path = save_dataset(bundle, output / "bundle")
        atomic_json(output / "preparation-status.json", dict(status="complete", evidence_status="fixture-tested", bundle=str(path), counts=bundle.validate()))
        return path
    if config.get("mode") != "source": raise ValueError("Set mode explicitly to source or fixture")
    if hz != 25. or samples < 64: raise ValueError("Source preparation requires 25 Hz and at least 64 frames")
    prep = config["preparation"]
    levels = [float(v) for v in options.get("movement_levels_deg", [0., 5., 10., 15.])]
    if len(set(levels)) != len(levels) or 0. not in levels or len(levels) < 3 or any(v < 0 or v > 20 for v in levels):
        raise ValueError("Use unique movement levels in [0,20] including zero and >=2 nonzero levels")
    held_level = float(options.get("held_level_deg", max(levels)))
    if held_level not in levels or held_level == 0: raise ValueError("Held level must be a declared nonzero magnitude")
    cameras = options.get("cameras", [{"id": "oblique", "azimuth_deg": 45.}, {"id": "side", "azimuth_deg": 90.}])
    if len(cameras) != 2 or len({c["id"] for c in cameras}) != 2: raise ValueError("The declared core requires two uniquely named cameras")
    if not all(np.isfinite(float(c["azimuth_deg"])) for c in cameras): raise ValueError("Camera azimuth must be finite")
    shard, shards = int(options.get("shard_index", 0)), int(options.get("num_shards", 1))
    if not 0 <= shard < shards: raise ValueError("Invalid preparation shard index/count")
    table, protected, old_hash = _source_rows(config)
    if full_manifest:
        # The cohort planner already froze exact nonoverlap starts inside each
        # reviewed interval. Applying the legacy extension here would change
        # the declared population and could move a window outside its review.
        candidates, exclusions = table.to_dict('records'), []
    else:
        candidates, exclusions = select_intervals(table.to_dict("records"), samples, hz)
    roster = set(table.canonical_person_id)
    if {r["canonical_person_id"] for r in candidates} != roster:
        atomic_json(output / "preparation-status.json", dict(status="insufficient_evidence", reason="Longer intervals remove a roster person", exclusions=exclusions))
        raise ValueError("Frozen interval length removes an entire person; review reference windows before training")
    train_counts = {person: sum(r["canonical_person_id"] == person and r["split"] == "train" for r in candidates)
                    for person in table.loc[table.split.eq("train"), "canonical_person_id"].unique()}
    if any(n < 2 for n in train_counts.values()):
        atomic_json(output / "preparation-status.json", dict(status="insufficient_evidence", training_windows=train_counts, exclusions=exclusions))
        raise ValueError("Shuffled-reference controls need >=2 nonoverlapping training windows per person; source duration is insufficient")
    rows = [row for i, row in enumerate(candidates) if i % shards == shard]
    common_options = {k: v for k, v in options.items() if k not in {"shard_index", "resume_preparation_paths"}}
    identity = digest(dict(preparation=prep, data=common_options, old_manifest_sha256=old_hash,
                           people=sorted(roster), schema="gait-fidelity-body12-v1"))
    receipt = dict(identity=identity, shard_index=shard, num_shards=shards,
                   assigned_windows=len(rows), total_windows=len(candidates), frozen_people=sorted(roster),
                   historical_manifest_sha256=None if full_manifest else old_hash,
                   cohort_identity=old_hash if full_manifest else None,
                   source_selection=options.get('source_selection', 'full_manifest') if full_manifest else 'legacy_roster',
                   confirmation_admitted=confirmation, exclusions=exclusions)
    if full_manifest:
        receipt['cohort_exclusions'] = protected
    if confirmation:
        receipt['confirmation_lock_sha256'] = sha256_file(config['cohort']['confirmation_lock'])
    atomic_json(output / "preparation-plan.json", receipt)
    if not rows:
        atomic_json(output / "preparation-status.json", {**receipt, "status": "no_work"})
        return output / "preparation-status.json"
    from ..synthetic_training_v2.runtime import require_haic_runtime
    from ..synthetic_training_v2.preparation import held_family, _config_files
    from ..motion_preservation.motion_data import load_motion
    from ..motion_preservation.body_geometry import SMPLHBody
    from ..synthetic_training.rendering import TexturedBodyRenderer, RenderRecipe, fitted_camera_pose
    from ..synthetic_training.estimators import StudentSpec, load_estimator
    from ..synthetic_training_v2.extraction import extract_tracks
    from .visualization import save_video, save_contact_sheet
    runtime = require_haic_runtime()
    held = held_family(prep["estimators"], config.get("held_extractor", "vitpose"))
    assets = {str(Path(prep["uv_path"]).resolve()): sha256_file(prep["uv_path"])}
    for source in Path(__file__).parent.glob("*.py"):
        assets[str(source.resolve())] = sha256_file(source)
    for spec in prep["estimators"]:
        assets.update(_config_files(spec["config"]))
        assets[str(Path(spec["checkpoint"]).resolve())] = sha256_file(spec["checkpoint"])
    renderer = None; started = time.perf_counter()
    arrays = {k: [] for k in ("xy", "confidence", "observed", "timestamps")}
    targets = {k: [] for k in ("xy", "valid", "visible", "eval_scale")}
    records, checks, admitted, chunks, review_index = [], [], [], [], []
    reusable = {}
    if full_manifest:
        for previous in options.get('resume_preparation_paths', []):
            for manifest in sorted(Path(previous).glob('chunks/*/bundle/manifest.json')):
                metadata = json.loads(manifest.read_text())
                prior = metadata.get('provenance', {})
                if prior.get('identity') != identity: continue
                families = prior.get('source_families', [])
                if len(families) != 1: raise ValueError('A resumable chunk must contain one complete source family')
                reusable[families[0]] = manifest.parent
    # Select review media from metadata before seeing predictions or outcomes.
    review_families = set()
    count_by_person = {}
    for row in candidates:
        person = row['canonical_person_id']
        rank = count_by_person.get(person, 0); count_by_person[person] = rank + 1
        if not full_manifest or rank < int(options.get('review_videos_per_person', 2)):
            review_families.add(digest([row['historical_motion_sha256'], float(row['start_s']), samples, hz])[:24])
    try:
        body_model = SMPLHBody(prep["body_model_root"], prep.get("dmpl_root"), device="cuda")
        renderer = TexturedBodyRenderer(prep["uv_path"], prep["texture_dir"], prep["background_dir"], 640, 480)
        estimators = {s["student_id"]: (s["family"], load_estimator(StudentSpec(**s), device="cuda")) for s in prep["estimators"]}
        fixed_assets = dict(assets)
        for row in rows:
            if options.get('source_selection') in {'repair_confirmation', 'repair_benchmark'}:
                from .repair_cohort import _check_deadline
                _check_deadline(config.get('_repair_lock_config', config.get('_repair_benchmark_config', config)))
            family_started = time.perf_counter()
            motion_hash = sha256_file(row["raw_path"])
            if motion_hash != row["historical_motion_sha256"]:
                raise ValueError("Raw AMASS motion changed since the frozen source authority")
            if not full_manifest and prep.get("review_mode", "automated_development") == "automated_development":
                source_evidence = json.loads(Path(row["audit_evidence"]).read_text())
                if source_evidence["source_sha256"] != motion_hash:
                    raise ValueError("Raw motion changed since locomotion screening")
            family_id = digest([motion_hash, float(row["start_s"]), samples, hz])[:24]
            if family_id in reusable:
                completed = load_dataset(reusable[family_id], allow_confirmation=confirmation)
                if {r['source_family_id'] for r in completed.records} != {family_id}:
                    raise ValueError('Resumable chunk family provenance differs')
                for asset, expected in completed.provenance.get('assets', {}).items():
                    if sha256_file(asset) != expected: raise ValueError(f'Resumable source asset changed: {asset}')
                    assets[asset] = expected
                if not completed.provenance.get('assets'):
                    raise ValueError('Resumable chunk lacks source asset hashes')
                chunks.append(reusable[family_id]); admitted.append(family_id)
                checks.extend(completed.provenance.get('geometry_checks', []))
                review_index.extend(completed.provenance.get('review_index', []))
                atomic_json(output/'family-costs'/f'{family_id}.json', dict(source_family_id=family_id,
                    gpu_seconds=time.perf_counter()-family_started, reused_chunk=str(reusable[family_id]),
                    original_gpu_seconds=completed.provenance.get('family_gpu_seconds')))
                continue
            motion = load_motion(row, start_s=float(row["start_s"]), duration_s=samples / hz, fps=hz)
            body = body_model.forward(motion)
            screening = technical_geometry_screen(body) if full_manifest else locomotion_screen(body.joints)
            screen_path = output / "screening" / f"{family_id}.json"
            atomic_json(screen_path, dict(screening, source_sha256=motion_hash,
                canonical_person_id=row["canonical_person_id"], relative_path=row["relative_path"],
                start_s=float(row["start_s"]), end_s=float(body.timestamps[-1]),
                parent_audited_start_s=float(row["parent_audited_start_s"]),
                source_geometry_sha256=array_digest({"joints": body.joints}), reviewed_by="algorithm"))
            if not screening["passed"]:
                exclusions.append(dict(source_family_id=family_id, canonical_person_id=row["canonical_person_id"], reason="technical_geometry_screen_failed" if full_manifest else "extended_locomotion_screen_failed", screening=str(screen_path)))
                continue
            # Audit actual body-model files, including the selected gender.
            family_assets = {str(Path(row['raw_path']).resolve())}
            for asset in (body_model.body_root / motion.gender / "model.npz", body_model.dmpl_root / motion.gender / "model.npz"):
                assets[str(asset.resolve())] = sha256_file(asset)
                family_assets.add(str(asset.resolve()))
            assets[str(Path(row["raw_path"]).resolve())] = motion_hash
            use_levels = [v for v in levels if row["split"] != "train" or v != held_level]
            geometries = {0.: body}; family_checks = []
            base_check = geometry_screen(body, body, minimum_changed_mm=0.)
            family_checks.append(dict(source_family_id=family_id, magnitude_deg=0., **base_check))
            for level in use_levels:
                if level == 0: continue
                changed = body_model.forward(knee_intervention(motion, body, level))
                check = geometry_screen(body, changed, **options.get("geometry_thresholds", {}))
                family_checks.append(dict(source_family_id=family_id, magnitude_deg=level, **check))
                geometries[level] = changed
            checks.extend(family_checks)
            if not all(r["pass_"] for r in family_checks):
                exclusions.append(dict(source_family_id=family_id, canonical_person_id=row["canonical_person_id"], reason="paired_geometry_screen_failed"))
                continue
            mirrors = {level: mirror_body(value, origin=np.median(body.joints[:, 0], axis=0),
                                         normal=body.joints[0, 2] - body.joints[0, 1]) for level, value in geometries.items()}
            # Fit once to the union, then retain the exact pose across all levels.
            union = np.concatenate([b.vertices for b in list(geometries.values()) + list(mirrors.values())])
            camera_poses = {c["id"]: fitted_camera_pose(union, body.joints, float(c["azimuth_deg"]), .70, 640, 480, np.deg2rad(50.))[0] for c in cameras}
            seed = int(config.get("seed", 17)) + int(family_id[:6], 16)
            for physical, bodies in (("original", geometries), ("mirrored", mirrors)):
                for camera_id, camera_pose in camera_poses.items():
                    for magnitude, changed in bodies.items():
                        state = "baseline" if magnitude == 0 else "knee_flexion"
                        for observation, cover in (("clear", 0.), ("occluded", float(options.get("occlusion_fraction", .15)))):
                            render = _render_fixed(renderer, changed, RenderRecipe(observation, occlusion_fraction=cover), seed, camera_pose, body.faces)
                            for asset in (render["texture_path"], render["background_path"]):
                                name = str(Path(asset).resolve()); assets[name] = sha256_file(asset); family_assets.add(name)
                            render_id = f"{family_id}-{physical}-{camera_id}-{magnitude:g}-{observation}"
                            media = output / "media" / f"{render_id}.mp4"
                            keep_video = options.get('keep_videos', True) and family_id in review_families
                            if keep_video: save_video(render["images"], media, fps=hz)
                            if magnitude == 0 and observation == "clear" and family_id in review_families:
                                image_path = output / 'images' / f'{render_id}.png'
                                save_contact_sheet(render, image_path)
                                review_index.append(dict(source_family_id=family_id, canonical_person_id=row['canonical_person_id'],
                                    relative_path=row['relative_path'], start_s=row['start_s'], split=row['split'],
                                    motion_label=row.get('motion_label', 'historical_screened_locomotion'),
                                    physical_state=physical, camera_id=camera_id, image=str(image_path), video=str(media) if keep_video else ''))
                            for extractor, (extractor_family, estimator) in estimators.items():
                                if row["split"] == "train" and extractor_family == held: continue
                                track = extract_tracks(estimator, render["images"], render["boxes"], body.timestamps,
                                    box_source="renderer_foreground_privileged", batch_size=int(options.get("extraction_batch_size", 32)))
                                if track.status_counts["unsupported_scores"]: raise ValueError("Extractor omitted native scores")
                                for naming in ("correct", "global_swap", "temporary_swap"):
                                    for movement_state in ([state, "no_change"] if magnitude == 0 else [state]):
                                        inp = apply_naming(track.inputs(), naming)
                                        for k, v in inp.items(): arrays[k].append(v)
                                        valid = np.isfinite(render["keypoints"]).all(-1)
                                        for k, v in dict(xy=render["keypoints"], valid=valid, visible=valid & render["visible"],
                                            eval_scale=np.linalg.norm(render["boxes"][:, 2:] - render["boxes"][:, :2], axis=-1)).items(): targets[k].append(v)
                                        record = {k: str(row[k]) for k in ("person_id", "canonical_person_id", "relative_path", "original_split", "split", "exposure", "locomotion_status", "audit_reviewer", "audit_evidence", "audit_date")}
                                        record.update(motion_id=row["relative_path"], motion_hash=motion_hash, window_id=family_id,
                                            source_family_id=family_id, pair_id=digest([family_id, physical, camera_id, observation, naming, extractor])[:24],
                                            movement_state=movement_state, movement_magnitude=magnitude,
                                            magnitude_deg=magnitude, movement_level_deg=magnitude,
                                            endpoint="baseline" if movement_state == "baseline" else "intervention",
                                            physical_state=physical, camera_id=camera_id, camera_hash=digest(camera_pose.tolist()),
                                            naming=naming, observation=observation, target_kind="synthetic_proxy", seed=seed,
                                            variant=f"{physical}-{camera_id}-{movement_state}-{magnitude:g}-{observation}-{naming}",
                                            extractor=extractor, student_id=extractor, extractor_family=extractor_family,
                                            box_source=track.box_source, reserved=row["reserved"] if isinstance(row["reserved"], str) else bool(row["reserved"]),
                                            review_mode="technical_geometry" if full_manifest else "automated_development", held_intervention=magnitude == held_level,
                                            locomotion_status=row['locomotion_status'] if full_manifest else "algorithm_screened_locomotion",
                                            audit_reviewer="gf-technical-geometry-v1" if full_manifest else "gf-extended-kinematic-screen-v1",
                                            audit_evidence=str(screen_path.resolve()), parent_audit_evidence=str(row["audit_evidence"]),
                                            parent_audited_start_s=float(row["parent_audited_start_s"]),
                                            start_s=float(body.timestamps[0]), end_s=float(body.timestamps[-1]), extraction_status=track.status_counts,
                                            video_path=str(media.resolve()) if keep_video else "",
                                            render_id=render_id, extended_interval_review="reference_geometry_screen_only; human overlay review pending")
                                        if full_manifest:
                                            record.update(source_dataset=row['source_dataset'], motion_label=row['motion_label'],
                                                label_source=row['label_source'], cohort_identity=old_hash)
                                        records.append(record)
            admitted.append(family_id)
            if full_manifest:
                chunk_provenance = dict(**receipt, hz=hz, samples=samples, held_extractor_family=held,
                    source_families=[family_id], evidence_boundary='Synthetic projected references; metadata walking candidates and technical QC',
                    assets={k:v for k,v in assets.items() if k in fixed_assets or k in family_assets},
                    geometry_checks=family_checks, review_index=[r for r in review_index if r['source_family_id']==family_id],
                    family_gpu_seconds=time.perf_counter()-family_started)
                bundle = TrackBundle({k:np.stack(v) for k,v in arrays.items()}, {k:np.stack(v) for k,v in targets.items()},
                    records, 'technical-source-screen', chunk_provenance)
                chunks.append(save_dataset(bundle, output/'chunks'/family_id/'bundle', storage='npy'))
                arrays = {k:[] for k in arrays}; targets = {k:[] for k in targets}; records = []
            atomic_json(output/'family-costs'/f'{family_id}.json', dict(source_family_id=family_id,
                gpu_seconds=time.perf_counter()-family_started, reused_chunk=None))
        atomic_json(output / "geometry-checks.json", checks)
        atomic_json(output / "coverage.json", dict(exclusions=exclusions, admitted=admitted, assigned=len(rows)))
        atomic_json(output / 'review-index.json', review_index)
        if not records and not chunks:
            if full_manifest:
                atomic_json(output/'preparation-status.json', {**receipt, 'status':'no_work', 'reason':'all_assigned_windows_failed_technical_or_intervention_checks', 'exclusions':exclusions})
                return output/'preparation-status.json'
            raise ValueError("No source families passed reference checks; inspect geometry-checks.json")
        for path, expected in assets.items():
            if sha256_file(path) != expected: raise ValueError(f"Source asset changed during preparation: {path}")
        evidence = "technical-source-screen" if full_manifest else "automated-source-screen"
        provenance = dict(**receipt, hz=hz, samples=samples, assets=assets, runtime=runtime,
            held_extractor_family=held, interventions=common_options,
            evidence_boundary="Synthetic proxy / kinematic stress tests; human full-interval review pending",
            source_families=admitted)
        if full_manifest:
            path = merge_disk_datasets(chunks, output/'bundle', provenance=provenance)
            counts = load_dataset(path, allow_confirmation=confirmation).validate()
        else:
            bundle = TrackBundle({k: np.stack(v) for k, v in arrays.items()}, {k: np.stack(v) for k, v in targets.items()}, records, evidence, provenance)
            path = save_dataset(bundle, output / "bundle")
            counts = bundle.validate()
        atomic_json(output / "preparation-status.json", {**receipt, "status": "complete", "bundle": str(path), "counts": counts, "exclusions": exclusions})
        return path
    except Exception as exc:
        atomic_json(output / "preparation-status.json", {**receipt, "status": "failed", "error": str(exc)})
        raise
    finally:
        if renderer is not None: renderer.close()
        atomic_json(output / "preparation-cost.json", dict(gpu_seconds=time.perf_counter() - started, stage="render_extract", runtime=runtime))


def merge_datasets(paths, output, *, allow_confirmation=False):
    """Verify every declared shard, preserve source receipts and merge once."""
    bundles, bundle_paths, receipts, source_receipts, shards, identities = [], [], [], [], set(), set()
    for path in map(Path, paths):
        if path.suffix == ".json":
            row = json.loads(path.read_text())
            if row.get("status") != "no_work": raise ValueError(f"Noncompleted preparation shard: {path}")
        else:
            bundle = load_dataset(path, allow_confirmation=allow_confirmation)
            bundles.append(bundle); bundle_paths.append(path); row = bundle.provenance
        index = int(row["shard_index"])
        if index in shards: raise ValueError("Duplicate preparation shard")
        shards.add(index); identities.add(row["identity"])
        source_receipts.append(row)
        receipts.append(dict(path=str(path.resolve()), shard_index=index,
                             manifest_sha256=sha256_file(path if path.suffix == ".json" else path / "manifest.json")))
    if not bundles or len(identities) != 1: raise ValueError("No nonempty bundles or incompatible shard configurations")
    expected = int(bundles[0].provenance["num_shards"])
    if shards != set(range(expected)): raise ValueError("Every declared preparation shard must finish before training")
    if len({b.evidence_status for b in bundles}) != 1: raise ValueError("Mixed source evidence levels across shards")
    provenance = dict(bundles[0].provenance, shard_index="merged", shard_receipts=receipts)
    assets = {}
    for b in bundles:
        for name, value in b.provenance.get("assets", {}).items():
            if assets.setdefault(name, value) != value:
                raise ValueError("An asset changed between preparation shards")
    provenance["assets"] = assets
    provenance["source_families"] = sorted({r["source_family_id"] for b in bundles for r in b.records})
    records = [r for b in bundles for r in b.records]
    actual_people = {r["canonical_person_id"] for r in records}
    if actual_people != set(provenance["frozen_people"]) and provenance.get('source_selection') not in {'full_manifest', 'repair_confirmation', 'repair_benchmark'}:
        raise ValueError("Reference screening removed a frozen roster person; review coverage before fitting")
    provenance['coverage'] = dict(planned_people=sorted(provenance['frozen_people']),
        admitted_people=sorted(actual_people), people_without_admitted_families=sorted(set(provenance['frozen_people'])-actual_people),
        exclusions=[r for receipt in source_receipts for r in receipt.get('exclusions', [])])
    training_families = {}
    for r in records:
        if r["split"] == "train": training_families.setdefault(r["canonical_person_id"], set()).add(r["source_family_id"])
    if any(len(families) < 2 for families in training_families.values()):
        raise ValueError("Reference screening left fewer than two training source families per person; shuffled-reference controls cannot run")
    benchmark_only = provenance.get('source_selection') == 'repair_benchmark' and {r['split'] for r in records} == {'development'}
    if not allow_confirmation and {r['split'] for r in records} != {'train', 'development'} and not benchmark_only:
        raise ValueError('Preparation must retain both training and development populations')
    if provenance.get('source_selection') in {'full_manifest', 'repair_confirmation', 'repair_benchmark'}:
        return merge_disk_datasets(bundle_paths, output, provenance=provenance)
    bundle = TrackBundle({k:np.concatenate([b.inputs[k] for b in bundles]) for k in bundles[0].inputs},
        {k:np.concatenate([b.targets[k] for b in bundles]) for k in bundles[0].targets},
        records, bundles[0].evidence_status, provenance)
    return save_dataset(bundle, output)
