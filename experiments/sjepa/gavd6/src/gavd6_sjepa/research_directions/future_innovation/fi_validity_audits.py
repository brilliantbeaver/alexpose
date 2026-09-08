"""Pixel intervention audits must pass before any model fitting."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv,
    sha256_file,
)

from .fi_cohort import load_cohort, validate_alignment_review
from .fi_contracts import (
    GateThresholds,
    check_run,
    equal_source_weights,
    load_model_contract,
    read_json,
    stable_key,
    write_once_json,
)
from .fi_feature_cache import load_cache, load_window
from .fi_token_regions import pool_target


def normalized_change(original, edited):
    if (
        not np.isfinite(original).all()
        or not np.isfinite(edited).all()
        or np.linalg.norm(original) <= 1e-8
    ):
        raise ValueError("Non-finite or degenerate sensitivity target")
    return float(np.linalg.norm(edited - original) / np.linalg.norm(original))


def pixel_edits(video, boxes, donor, donor_boxes, feather=8):
    person, background = video.copy(), video.copy()
    h, w = video.shape[1:3]
    dh, dw = donor.shape[1:3]

    def bounds(box, width, height):
        a = np.floor(box[:2] * [width, height]).astype(int)
        b = np.ceil(box[2:] * [width, height]).astype(int)
        return max(0, a[0]), max(0, a[1]), min(width, b[0]), min(height, b[1])

    # Remove the donor person before turning its scene into a static background.
    donor_mask = np.zeros((dh, dw), dtype=np.uint8)
    dx0, dy0, dx1, dy1 = bounds(donor_boxes[0], dw, dh)
    donor_mask[dy0:dy1, dx0:dx1] = 255
    static = cv2.inpaint(donor[0], donor_mask, 5, cv2.INPAINT_TELEA)
    static = cv2.resize(static, (w, h)).astype(float)
    for t in range(32, 64):
        x0, y0, x1, y1 = bounds(boxes[t], w, h)
        source_t = 95 - t  # 63..32: a different source with reversed future timing.
        dx0, dy0, dx1, dy1 = bounds(donor_boxes[source_t], dw, dh)
        if min(x1 - x0, y1 - y0, dx1 - dx0, dy1 - dy0) <= 0:
            raise ValueError("Pixel audit requires boxes throughout the hidden frames")
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        # Matched, inward feather: person edit never alters outside pixels;
        # background edit never alters person pixels.
        alpha_person = np.minimum(
            cv2.distanceTransform(mask, cv2.DIST_L2, 3) / feather, 1
        )[..., None]
        alpha_background = np.minimum(
            cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3) / feather, 1
        )[..., None]
        replacement = video[t].astype(float)
        replacement[y0:y1, x0:x1] = cv2.resize(
            donor[source_t, dy0:dy1, dx0:dx1], (x1 - x0, y1 - y0)
        )
        person[t] = (
            np.rint(video[t] * (1 - alpha_person) + replacement * alpha_person)
            .clip(0, 255)
            .astype(np.uint8)
        )
        background[t] = (
            np.rint(video[t] * (1 - alpha_background) + static * alpha_background)
            .clip(0, 255)
            .astype(np.uint8)
        )
    return person, background


def future_pixel_leakage_test(adapter, video, rng):
    changed = video.copy()
    changed[32:] = rng.integers(0, 256, changed[32:].shape, dtype=np.uint8)
    original = adapter.encode_past_context(video)
    repeat = adapter.encode_past_context(video)
    modified = adapter.encode_past_context(changed)
    repeat_error = float(np.max(np.abs(original - repeat)))
    diff = np.abs(original - modified)
    tolerance = max(1e-6, 2 * repeat_error)
    return {
        "repeat_max_abs": repeat_error,
        "max_abs_difference": float(diff.max()),
        "mean_abs_difference": float(diff.mean()),
        "tolerance": tolerance,
        "passed": bool(diff.max() <= tolerance),
        "stable": repeat_error <= 1e-6,
    }


def audit_binding(root):
    root = Path(root)
    return stable_key(
        *[
            sha256_file(root / p)
            for p in (
                "config/cache-contract.json",
                "config/cohort-contract.json",
                "qc/alignment-review.json",
            )
        ]
    )


def run_audits(root, adapter):
    root = Path(root)
    cohort = load_cohort(root, verify_artifacts=True)
    validate_alignment_review(root, cohort)
    _, cache = load_cache(root)
    model = load_model_contract(root)
    variance_valid = {}
    for target_name in ("person", "background"):
        masks = []
        for fold in range(5):
            train = cohort.outer_fold.to_numpy() != fold
            weights = equal_source_weights(cohort.loc[train, "video_id"])
            values = cache[target_name][train].astype(float)
            mean = np.average(values, axis=0, weights=weights)
            variance = np.average((values - mean) ** 2, axis=0, weights=weights)
            masks.append(variance > model.target_variance_tolerance)
        variance_valid[target_name] = bool(np.stack(masks).all(axis=0).any())
    audits = pd.read_csv(root / "manifests/audit-windows.csv")
    if len(audits) != 10 or not audits.window_id.is_unique:
        raise ValueError("Expected ten prespecified audit windows")
    rows = cohort.set_index("window_id").to_dict("index")
    projection = np.load(root / "config/projection-256.npy", allow_pickle=False)
    stability, leakage, sensitivity = [], [], []
    sheets = []
    for audit in audits.to_dict("records"):
        window_id = audit["window_id"]
        row, donor_row = rows[window_id], rows[audit["donor_window_id"]]
        if row["video_id"] == donor_row["video_id"]:
            raise ValueError("Audit donor belongs to recipient source")
        video, boxes, model_boxes, _ = load_window(row)
        donor, donor_boxes, _, _ = load_window(donor_row)
        leak = future_pixel_leakage_test(
            adapter,
            video,
            np.random.default_rng(int(stable_key("leakage", window_id)[:16], 16)),
        )
        leakage.append({"window_id": window_id, **leak})
        original_tokens = adapter.encode_full_target(video)
        repeat_tokens = adapter.encode_full_target(video)
        target_repeat_error = float(np.max(np.abs(original_tokens - repeat_tokens)))
        target = pool_target(original_tokens, model_boxes)[0] @ projection
        cache_index = np.flatnonzero(cohort.window_id.to_numpy() == window_id)[0]
        cache_error = float(np.max(np.abs(target - cache["person"][cache_index])))
        stability.append(
            {
                "window_id": window_id,
                "context_max_abs": leak["repeat_max_abs"],
                "target_max_abs": target_repeat_error,
                "cache_max_abs": cache_error,
                "passed": bool(
                    leak["stable"]
                    and target_repeat_error <= 1e-6
                    and cache_error <= 1e-6
                ),
            }
        )
        edited_person, edited_background = pixel_edits(video, boxes, donor, donor_boxes)
        person_target = (
            pool_target(adapter.encode_full_target(edited_person), model_boxes)[0]
            @ projection
        )
        background_target = (
            pool_target(adapter.encode_full_target(edited_background), model_boxes)[0]
            @ projection
        )
        person_change = normalized_change(target, person_target)
        background_change = normalized_change(target, background_target)
        sensitivity.append(
            {
                **audit,
                "person_change": person_change,
                "background_change": background_change,
                "person_larger": person_change > background_change,
            }
        )
        panels = []
        for variant, label in (
            (video, "Original"),
            (edited_person, "Future person edit"),
            (edited_background, "Static background edit"),
        ):
            strips = []
            for t in (31, 32, 38, 39, 63):
                panel = cv2.resize(variant[t], (256, 160))
                cv2.putText(
                    panel,
                    f"{label}; t={t}",
                    (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 220, 0),
                    1,
                )
                strips.append(panel)
            panels.append(np.concatenate(strips, axis=1))
        sheet = root / "qc/pixel-edit-contact-sheets" / f"{window_id}.jpg"
        sheet.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(sheet), cv2.cvtColor(np.concatenate(panels), cv2.COLOR_RGB2BGR)
        ):
            raise OSError(f"Cannot save pixel-edit sheet: {sheet}")
        sheets.append(str(sheet.relative_to(root)))
        # Preserve partial QC evidence even if a later window fails.
        for filename, records in (
            ("teacher-stability.csv", stability),
            ("causal-leakage.csv", leakage),
            ("target-sensitivity.csv", sensitivity),
        ):
            atomic_write_dataframe_csv(root / "qc" / filename, pd.DataFrame(records))
    thresholds = GateThresholds(**read_json(root / "config/thresholds.json"))
    ratio = float(
        np.median([r["person_change"] for r in sensitivity])
        / max(np.median([r["background_change"] for r in sensitivity]), 1e-8)
    )
    direction = float(np.mean([r["person_larger"] for r in sensitivity]))
    checks = {
        "teacher_stable": all(r["passed"] for r in stability),
        "causal_leakage_absent": all(r["passed"] for r in leakage),
        "target_audit_complete": len(sensitivity) == 10,
        "target_sensitivity": ratio >= thresholds.motion_to_background_change_min,
        "person_edit_consistency": direction
        >= thresholds.person_edit_direction_fraction_min,
        "data_contract_valid": True,
        "target_variance_valid": all(variance_valid.values()),
    }
    files = [
        "qc/teacher-stability.csv",
        "qc/causal-leakage.csv",
        "qc/target-sensitivity.csv",
        *sheets,
    ]
    write_once_json(
        root / "qc/validity-summary.json",
        {
            "binding": audit_binding(root),
            "passed": all(checks.values()),
            "checks": checks,
            "motion_to_background_change_ratio": ratio,
            "person_edit_direction_fraction": direction,
            "artifacts": {p: sha256_file(root / p) for p in files},
        },
    )
    return all(checks.values())


def require_audits(root):
    root = Path(root)
    run = check_run(root)
    summary = read_json(root / "qc/validity-summary.json")
    required = {
        "teacher_stable",
        "causal_leakage_absent",
        "target_audit_complete",
        "target_sensitivity",
        "person_edit_consistency",
        "data_contract_valid",
        "target_variance_valid",
    }
    if set(summary.get("checks", {})) != required:
        raise ValueError("Required validity checks are missing or unexpected")
    if summary["binding"] != audit_binding(root):
        raise ValueError("Validity audit lineage changed")
    for path, digest in summary["artifacts"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"Validity audit changed: {path}")
    if summary.get("passed") is not True or not all(
        value is True for value in summary["checks"].values()
    ):
        raise ValueError(
            "Validity audit failed; fitting is prohibited; build the STOP report"
        )
    if not run["synthetic"]:
        cohort = load_cohort(root)
        validate_alignment_review(root, cohort)
        planned = pd.read_csv(root / "manifests/audit-windows.csv")
        sensitivity = pd.read_csv(root / "qc/target-sensitivity.csv")
        leakage = pd.read_csv(root / "qc/causal-leakage.csv")
        stability = pd.read_csv(root / "qc/teacher-stability.csv")
        for table in (sensitivity, leakage, stability):
            if (
                len(table) != 10
                or not table.window_id.is_unique
                or set(table.window_id) != set(planned.window_id)
            ):
                raise ValueError(
                    "Validity report omits or duplicates a prespecified audit window"
                )
        planned = planned.set_index("window_id")
        for row in sensitivity.itertuples():
            if row.donor_window_id != planned.loc[row.window_id, "donor_window_id"]:
                raise ValueError("Pixel-audit donor changed after preregistration")
            sheet = f"qc/pixel-edit-contact-sheets/{row.window_id}.jpg"
            if sheet not in summary["artifacts"]:
                raise ValueError("Missing pixel-edit contact sheet")
        if not np.isfinite(
            sensitivity[["person_change", "background_change"]].to_numpy()
        ).all():
            raise ValueError("Non-finite sensitivity report")
        ratio = float(
            np.median(sensitivity.person_change)
            / max(np.median(sensitivity.background_change), 1e-8)
        )
        direction = float(
            np.mean(sensitivity.person_change > sensitivity.background_change)
        )
        thresholds = GateThresholds(**read_json(root / "config/thresholds.json"))
        if (
            not np.isclose(
                ratio,
                summary["motion_to_background_change_ratio"],
                rtol=1e-10,
                atol=1e-12,
            )
            or direction != summary["person_edit_direction_fraction"]
            or ratio < thresholds.motion_to_background_change_min
            or direction < thresholds.person_edit_direction_fraction_min
        ):
            raise ValueError(
                "Sensitivity summary does not match its per-window evidence or thresholds"
            )
        numeric_leakage = leakage[
            ["max_abs_difference", "repeat_max_abs", "tolerance"]
        ].to_numpy()
        numeric_stability = stability[
            ["context_max_abs", "target_max_abs", "cache_max_abs"]
        ].to_numpy()
        expected_tolerance = np.maximum(1e-6, 2 * leakage.repeat_max_abs.to_numpy())
        if (
            not np.isfinite(numeric_leakage).all()
            or not np.isfinite(numeric_stability).all()
            or np.any(numeric_leakage < 0)
            or np.any(numeric_stability < 0)
            or np.any(numeric_stability > 1e-6)
            or not np.allclose(
                leakage.tolerance, expected_tolerance, atol=1e-15, rtol=0
            )
            or np.any(leakage.max_abs_difference > leakage.tolerance)
        ):
            raise ValueError(
                "Teacher stability/leakage evidence does not pass frozen tolerances"
            )
    return summary
