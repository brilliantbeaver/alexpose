"""CPU analysis of existing repair caches, without fitting or changing calibration.

Reference-informed variants isolate mechanisms. They are explicitly labelled
diagnostics and must not be reported as deployable repair methods.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter

from . import body_geometry as geometry
from . import preservation_metrics as metrics
from . import repair_models as learning
from . import workflow
from . import flow_diagnostics
from .rendering import Camera


ALLOWED_ROLES = {"train", "calibration", "development"}
STRENGTHS = (0., .01, .025, .05, .1, .25, .5, 1.)
PRIVILEGED = {"oracle_mixture_unprojected", "oracle_mixture_projected",
              "raw_projected_reference_lengths", "truth_projected_observed_lengths",
              "truth_projected_reference_lengths"}
CONDITIONS = ["role", "method", "fixture", "event_present", "noise_present", "privileged"]
MEASURES = ["retention", "noise_removal", "mse_m2", "observed_mse_m2",
            "raw_observed_mse_m2", "completion_mse_m2", "missing_fraction",
            "descriptor_abs_error", "descriptor_signed_error", "signed_event_error",
            "affected_mse_m2", "unaffected_mse_m2", "affected_raw_mse_m2",
            "affected_fraction", "mean_displacement_m"]


def diagnostic_candidates(case):
    """Matched projected/unprojected candidates, plus explicitly privileged probes."""
    raw, prior, truth = (np.asarray(case[k]) for k in ("raw", "prior", "truth"))
    lengths = case["bone_lengths"]
    project = lambda x: geometry.project_bone_lengths(x, lengths)
    reference_lengths = geometry.estimate_bone_lengths(truth)
    candidates = {"raw": raw.copy(), "projected_raw": project(raw),
                  "prior_unprojected": prior.copy(), "prior_projected": project(prior)}
    # An imported prior without a representation bridge must not earn credit
    # for the workflow's raw-input placeholder as a measured round trip.
    if np.isfinite(float(case.get("bridge_error", np.nan))):
        candidates.update(bridge_unprojected=case["bridge"].copy(),
                          bridge_projected=project(case["bridge"]))
    for name, value in (("gaussian", gaussian_filter1d(raw, 1.2, axis=0)),
                        ("median", median_filter(raw, size=(5, 1, 1)))):
        candidates[f"{name}_unprojected"] = value
        candidates[f"{name}_projected"] = project(value)
    candidates["legacy_prior_strength0"] = learning.mix_at_strength(
        raw, prior, 0., lengths, case["observed"])
    # Pointwise closest point on the raw-prior line segment before projection.
    # Because raw is an endpoint, this oracle cannot worsen unprojected MSE.
    delta = prior-raw
    squared = np.sum(delta**2, axis=-1)
    fraction = np.divide(np.sum((truth-raw)*delta, axis=-1), squared,
                         out=np.zeros_like(squared), where=squared > 1e-12)
    oracle = raw+np.clip(fraction, 0, 1)[..., None]*delta
    candidates.update(
        oracle_mixture_unprojected=oracle,
        oracle_mixture_projected=project(oracle),
        raw_projected_reference_lengths=geometry.project_bone_lengths(raw, reference_lengths),
        truth_projected_observed_lengths=project(truth),
        truth_projected_reference_lengths=geometry.project_bone_lengths(truth, reference_lengths))
    return candidates


def score_prediction(prediction, case, metadata, method, strength=1.):
    """Use original observed-joint scoring; decompose factorial tracking error."""
    row = metrics.score_case(prediction, case, metadata, method, strength)
    observed = np.asarray(case["observed"], bool)
    error = np.sum((prediction-case["truth"])**2, axis=-1)
    raw_error = np.sum((case["raw"]-case["truth"])**2, axis=-1)
    affected = observed & (raw_error > 1e-12)
    unaffected = observed & ~affected
    average = lambda values, mask: float(values[mask].mean()) if mask.any() else np.nan
    factorial = metadata["fixture"] == "factorial"
    row.update(privileged=method in PRIVILEGED,
               affected_mse_m2=average(error, affected) if factorial else np.nan,
               unaffected_mse_m2=average(error, unaffected) if factorial else np.nan,
               affected_raw_mse_m2=average(raw_error, affected) if factorial else np.nan,
               affected_fraction=float(affected.sum()/observed.sum()) if factorial and observed.any() else np.nan,
               affected_count=int(affected.sum()) if factorial else 0,
               unaffected_count=int(unaffected.sum()) if factorial else 0,
               mean_displacement_m=float(np.linalg.norm(prediction-case["raw"], axis=-1).mean()))
    # Signed event units distinguish suppression (-1 returns to the unedited
    # descriptor), preservation (0), and overshoot (>0) for either edit sign.
    row.update(reference_descriptor=geometry.descriptor(case["truth"], metadata["event_family"]),
               output_descriptor=geometry.descriptor(prediction, metadata["event_family"]))
    return row


def summarize_cases(rows, extra_groups=()):
    """Equal weight per person within each explicitly named experimental condition."""
    group = CONDITIONS+list(extra_groups)
    per_person = rows.groupby(group+["person_id"], dropna=False)[MEASURES].mean()
    summary = per_person.groupby(level=group, dropna=False).mean()
    summary["n_people"] = per_person.groupby(level=group, dropna=False).size()
    summary["n_cases"] = rows.groupby(group, dropna=False).size()
    return summary.reset_index()


def bone_length_rows(case, metadata):
    """Compare estimated lengths with reference variation without changing inputs."""
    raw, truth, observed = case["raw"], case["truth"], np.asarray(case["observed"], bool)
    rows = []
    for joint in range(1, len(geometry.PARENTS)):
        parent = geometry.PARENTS[joint]
        reference = np.linalg.norm(truth[:, joint]-truth[:, parent], axis=-1)
        valid = observed[:, joint] & observed[:, parent]
        measured = np.linalg.norm(raw[:, joint]-raw[:, parent], axis=-1)[valid]
        estimate = float(case["bone_lengths"][joint])
        median = float(np.median(reference))
        rows.append({**{k: metadata[k] for k in ("case_id", "person_id", "role", "fixture", "event_present", "noise_present")},
                     "joint": joint, "joint_name": geometry.JOINT_NAMES[joint],
                     "reference_length_mean_m": float(reference.mean()),
                     "reference_length_median_m": median,
                     "reference_length_std_m": float(reference.std()),
                     "reference_length_range_m": float(np.ptp(reference)),
                     "observed_length_std_m": float(measured.std()) if len(measured) else np.nan,
                     "estimated_length_m": estimate, "bias_m": estimate-median,
                     "observed_pair_count": int(valid.sum())})
    return rows


def _camera_for_case(root, metadata):
    scene = root/"scenes"/f"{metadata['scene_id']}.npz"
    if not scene.is_file():
        return None
    # Load only camera parameters; do not decompress all RGB frames for a table.
    with np.load(scene, allow_pickle=False) as values:
        camera = json.loads(str(values["camera_json"]))
    camera["eye"] = np.asarray(camera["eye"])
    camera["rotation"] = np.asarray(camera["rotation"])
    return Camera(**camera)


def run_diagnostics(cfg, roles=("calibration", "development"), output_dir=None,
                    max_trace_cases=6):
    """Analyze existing caches on CPU. Never construct final cases or load weights."""
    roles = tuple(dict.fromkeys(roles))
    if not roles or not set(roles) <= ALLOWED_ROLES:
        raise ValueError("Diagnostics allow train, calibration and development only; final must remain reserved.")
    if int(max_trace_cases) < 0:
        raise ValueError("max_trace_cases must be nonnegative")
    folder = Path(output_dir).expanduser().resolve() if output_dir else cfg.root/"diagnostics/repair-mechanism"
    if folder in {cfg.root, *(cfg.root/name for name in ("models", "calibration", "results", "predictions", "scenes", "flow"))}:
        raise ValueError("Choose a dedicated diagnostic output folder, separate from original experiment outputs.")
    case_rows, curve_rows, bones, flows, inventory, trace_choices = [], [], [], [], [], []
    for role in roles:
        try:
            index, cases = workflow._load_cases(cfg, role)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Missing {role} prediction cache. Point MP_RUN_ROOT at the existing HAIC pilot "
                "with predictions/<prior_id>/index.csv and its NPZ files; notebook copies alone are insufficient."
            ) from error
        inventory.append(dict(role=role, n_cases=len(index), n_people=index.person_id.nunique(),
                              prior_id=cfg.prior_id, mode=cfg.mode))
        print(f"Diagnosing {role}: {len(index)} cached cases from {index.person_id.nunique()} people", flush=True)
        for (_, record), case in zip(index.iterrows(), cases):
            metadata = record.to_dict()
            metadata["motion_id"] = re.sub(r"_(?:event[01]|occluded)$", "", str(metadata["scene_id"]))
            candidates = diagnostic_candidates(case)
            for method, value in candidates.items():
                case_rows.append(score_prediction(value, case, metadata, method))
            bones.extend(bone_length_rows(case, metadata))
            flows.append(flow_diagnostics.analyze_flow_case(case, metadata, _camera_for_case(cfg.root, metadata)))
            if metadata["fixture"] == "factorial":
                for method in ("prior", "gaussian", "median"):
                    candidate = candidates[f"{method}_unprojected"]
                    for strength in STRENGTHS:
                        mixed = case["raw"]+strength*(candidate-case["raw"])
                        for projection, value in (("none", mixed), ("fixed_lengths", geometry.project_bone_lengths(mixed, case["bone_lengths"]))):
                            row = score_prediction(value, case, metadata, method, strength)
                            row["projection"] = projection
                            curve_rows.append(row)
            if metadata["fixture"] == "factorial" and metadata["event_present"] and metadata["noise_present"]:
                selected = {"truth": case["truth"], "clean": case["clean"], "event_reference": case["event_reference"],
                            "raw": case["raw"], "projected_raw": candidates["projected_raw"], "prior": case["prior"]}
                if "bridge_unprojected" in candidates:
                    selected["bridge"] = candidates["bridge_unprojected"]
                trace_choices.append(dict(case_id=metadata["case_id"], role=role, person_id=metadata["person_id"],
                                          event_family=metadata["event_family"],
                                          timestamps=case.get("timestamps", np.asarray(case["frame_indices"])/cfg.fps),
                                          observed=case["observed"],
                                          affected_mask=(np.linalg.norm(case["raw"]-case["truth"], axis=-1)>1e-6)&np.asarray(case["observed"], bool),
                                          series=selected))
    scores, curves = pd.DataFrame(case_rows), pd.DataFrame(curve_rows)
    flow_cases = pd.DataFrame(flows)
    flow_pairs = flow_diagnostics.paired_flow_cases(flow_cases)
    traces = sorted(trace_choices, key=lambda x: (x["role"] != "development", x["case_id"]))[:int(max_trace_cases)]
    notes = dict(mode=cfg.mode, analysis_only=True, roles=list(roles), prior_id=cfg.prior_id,
                 noise_removal_scope="observed_joints", reference_informed_methods=sorted(PRIVILEGED),
                 final_evaluated=False, fits_or_calibration_changed=False,
                 strengths=list(STRENGTHS), bridge_available_cases=int(scores.method.eq("bridge_unprojected").sum()),
                 interpretation="Development diagnostics only. Curves do not select or replace official operating points.",
                 affected_mask="Factorial only: observed positions with raw-reference displacement above one micrometer.",
                 signed_event_error="0 preserves the descriptor; -1 returns to the unedited descriptor; positive overshoots in the edit direction.",
                 flow_contrast="Paired transport means use each video's available support. Inspect both coverage values; unequal support can confound the difference.",
                 projection_note="Reference-length projection is a privileged mechanism probe, not a certified optimal repair.")
    report = dict(output_dir=folder, case_scores=scores, summary=summarize_cases(scores),
                  strength_curve=summarize_cases(curves, ("projection", "strength")),
                  strength_case_scores=curves, bone_lengths=pd.DataFrame(bones), flow_cases=flow_cases,
                  flow_pairs=flow_pairs, flow_summary=flow_diagnostics.paired_flow_summary(flow_cases),
                  inventory=pd.DataFrame(inventory), traces=traces, notes=notes)
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in report.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(folder/f"{name}.csv", index=False)
    workflow._json(folder/"diagnostic_notes.json", notes)
    # Small selected traces make the diagnosis reviewable without copying the
    # full motion/render cache. They are never consumed by fitting functions.
    trace_folder = folder/"traces"
    trace_folder.mkdir(exist_ok=True)
    trace_index = []
    for trace in traces:
        path = trace_folder/f"{workflow._safe(trace['case_id'])}.npz"
        np.savez_compressed(path, timestamps=trace["timestamps"], observed=trace["observed"],
                            affected_mask=trace["affected_mask"], **trace["series"])
        trace_index.append({key: trace[key] for key in ("case_id", "role", "person_id", "event_family")})
    workflow._json(trace_folder/"index.json", trace_index)
    print(f"Diagnostic tables and {len(traces)} selected traces saved to {folder}", flush=True)
    return report
