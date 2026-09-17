"""Counterfactual information-boundary and measurement-support gates."""
import copy
from collections import defaultdict
import numpy as np
from .preprocessing import context_input
from .windows import CONTEXT_KEYS


def assert_future_boundary(pose, cfg, issue):
    """Mutations remain AFTER b; exact boundary frame is changed as well.

    This tests downstream preparation, not the honesty of an upstream pose
    extractor's causality declaration. No frame-dependent crop metadata is accepted.
    """
    baseline = context_input(pose, cfg, issue)
    modified = copy.deepcopy(pose)
    future = modified.times >= issue
    if not future.any():
        return {"status": "not_applicable", "reason": "no_post_boundary_observations"}
    modified.coords[future] = 100000.
    modified.observed[future] = ~modified.observed[future]
    modified.confidence[future] = 0.
    after = modified.times > issue
    modified.times[after] += .001  # still strictly after b; cannot enter permitted history
    result = context_input(modified, cfg, issue)
    for key in (*CONTEXT_KEYS, "origin", "scale", "scale_valid"):
        np.testing.assert_array_equal(baseline[key], result[key], err_msg=f"Future mutation changed {key}")
    return {"status": "passed", "mutated_frames": int(future.sum()), "keys": list(CONTEXT_KEYS),
            "limitation": "Does not establish upstream pose/tracker causality from its declaration."}


def support_report(dataset, cfg):
    if not len(dataset):
        return {"windows": 0, "primary_windows": 0, "videos": 0, "groups": 0}
    h = cfg.horizons.index(.5)
    mask = dataset.arrays["endpoint_valid"][:, h, 25:33].reshape(len(dataset), 4, 2).all(2)
    supported = (mask.sum(1) >= 3) & dataset.arrays["scale_valid"]
    return {"windows": len(dataset), "primary_windows": int(supported.sum()),
            "primary_videos": len({r["video_id"] for r, ok in zip(dataset.records, supported) if ok}),
            "videos": len({r["video_id"] for r in dataset.records}), "groups": len({r["group_id"] for r in dataset.records}),
            "context_observed_fraction": float(dataset.arrays["context_valid"].mean()),
            "endpoint_observed_fraction": float(dataset.arrays["endpoint_valid"].mean()),
            "scale_failure_fraction": float(1 - dataset.arrays["scale_valid"].mean()),
            "last_query_offset_seconds": 1 / cfg.grid_hz}


def measurement_differences(dataset):
    """Descriptive prefix-only raw/index/time-grid proxy differences.

    Common finite support is fixed across both comparisons. Windows average
    within bouts, bouts within source videos, then videos equally. This is NOT
    an encoder evaluation, clinical target, significance test or training gate.
    The underlying proxy averages adjacent-observation speeds equally, rather
    than weighting them by segment duration on a VFR clock.
    """
    keys = ["diagnostic_" + name + "_signed_speed_contrast"
            for name in ("raw", "timegrid", "indexgrid")]
    availability = {key: 0 for key in keys}
    by_video = defaultdict(lambda: defaultdict(list))
    matched = 0
    for record in dataset.records:
        values = [record.get(key) for key in keys]
        finite = [isinstance(value, (float, int, np.number)) and np.isfinite(value)
                  for value in values]
        for key, ok in zip(keys, finite):
            availability[key] += int(ok)
        if not all(finite):
            continue
        raw, timed, indexed = values
        row = np.array([abs(timed - raw), abs(indexed - raw), timed - raw, indexed - raw], float)
        by_video[record["video_id"]][record["sequence_id"]].append(row)
        matched += 1
    video_rows = []
    for video_id, bouts in sorted(by_video.items()):
        values = np.mean([np.mean(rows, axis=0) for rows in bouts.values()], axis=0)
        video_rows.append({"video_id": video_id, "bouts": len(bouts),
                           "matched_windows": sum(len(rows) for rows in bouts.values()),
                           "timegrid_absolute_difference": float(values[0]),
                           "indexgrid_absolute_difference": float(values[1]),
                           "timegrid_signed_difference": float(values[2]),
                           "indexgrid_signed_difference": float(values[3])})
    metrics = ("timegrid_absolute_difference", "indexgrid_absolute_difference",
               "timegrid_signed_difference", "indexgrid_signed_difference")
    averages = {name: float(np.mean([row[name] for row in video_rows])) if video_rows else None
                for name in metrics}
    averages["time_minus_index_absolute_difference"] = (
        averages["timegrid_absolute_difference"] - averages["indexgrid_absolute_difference"]
        if video_rows else None)
    return {"status": "descriptive_only" if matched else "no_common_measurement_support",
            "windows": len(dataset.records), "matched_windows": matched,
            "excluded_windows": len(dataset.records) - matched, "availability": availability,
            "videos": len(video_rows), "video_equal_means": averages, "per_video": video_rows,
            "aggregation": "windows_within_bout_then_bouts_within_video_then_equal_videos",
            "definition": "signed speed contrast; equal valid adjacent-observation segments, not duration-weighted VFR",
            "used_future_targets": False, "used_for_training_gate_or_model_selection": False,
            "uncertainty": "not_estimated; descriptive preprocessing proxy, not clinical evidence"}


def audit_datasets(train, development, cfg):
    if any(r["role"] != "train" for r in train.records) or any(r["role"] != "development" for r in development.records):
        raise ValueError("Information audit requires train/development only")
    if {r["group_id"] for r in train.records} & {r["group_id"] for r in development.records}:
        raise ValueError("Training and development connected components overlap")
    train_report, development_report = support_report(train, cfg), support_report(development, cfg)
    passed = all(r["groups"] >= 2 and r.get("primary_videos", 0) >= 2 for r in (train_report, development_report))
    return {"status": "passed" if passed else "stop_insufficient_measurement", "measurement_gate": passed,
            "mode": cfg.mode, "train": train_report, "development": development_report,
            "prefix_measurement_differences": {"train": measurement_differences(train),
                                               "development": measurement_differences(development)},
            "scientific_readiness": False,
            "reason": "Endpoint/scale support permits development comparisons; does not certify JEPA utility."}
