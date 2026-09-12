"""Pixel intervention audits must pass before any model fitting."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv,
    sha256_file,
)

from .fi_cohort import load_cohort
from .fi_contracts import (
    GateThresholds,
    DIRECT_PROTOCOL,
    protocol_name,
    audit_summary_path,
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
            )
        ]
    )


def run_audits(root, adapter):
    root = Path(root)
    cohort = load_cohort(root, verify_artifacts=True)
    _, cache = load_cache(root)
    model = load_model_contract(root)
    variance_valid = target_variance_checks(cohort, cache, model)
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


class ValidityAuditRejected(ValueError):
    """Intact, recomputed audit evidence fails a frozen scientific criterion."""

    def __init__(self, root, summary):
        self.root = Path(root).resolve()
        self.summary = summary
        self.summary_sha256 = sha256_file(audit_summary_path(self.root))
        self.failed_checks = [name for name, passed in summary["checks"].items() if not passed]
        thresholds = GateThresholds(**read_json(self.root / "config/thresholds.json"))
        detail = ", ".join(self.failed_checks)
        if "target_sensitivity" in self.failed_checks:
            detail += (f"; person/background sensitivity ratio "
                       f"{summary['motion_to_background_change_ratio']:.3f} < "
                       f"{thresholds.motion_to_background_change_min:g}")
        super().__init__(f"Validity audit failed: {detail}. Fitting is prohibited; "
                         "the predictive measurement is incomplete.")


def target_variance_checks(cohort, cache, model, targets=("person", "background")):
    """Use the same training-source variance criterion for creation and reuse."""
    valid = {}
    for target_name in targets:
        masks = []
        for fold in range(5):
            train = cohort.outer_fold.to_numpy() != fold
            weights = equal_source_weights(cohort.loc[train, "video_id"])
            values = cache[target_name][train].astype(float)
            mean = np.average(values, axis=0, weights=weights)
            variance = np.average((values - mean) ** 2, axis=0, weights=weights)
            if not np.isfinite(variance).all():
                raise ValueError("Non-finite training-target variance")
            masks.append(variance > model.target_variance_tolerance)
        valid[target_name] = bool(np.stack(masks).all(axis=0).any())
    return valid


def _check_boolean_column(table, column, expected):
    values = table[column].to_numpy()
    if values.dtype.kind != "b" or not np.array_equal(values, expected):
        raise ValueError(f"Audit column {column} disagrees with its numeric evidence")


def verify_audits(root):
    """Verify both passed and rejected evidence; corruption is always an error.

    This checks retained measurements, contracts and hashes, without rerunning
    the teacher. Synthetic fixtures use the same verifier and remain synthetic.
    """
    root = Path(root)
    check_run(root)
    summary = read_json(root / "qc/validity-summary.json")
    if not isinstance(summary, dict):
        raise ValueError("Validity summary must be a JSON object")
    required = {"teacher_stable", "causal_leakage_absent", "target_audit_complete",
                "target_sensitivity", "person_edit_consistency", "data_contract_valid",
                "target_variance_valid"}
    checks = summary.get("checks", {})
    if (not isinstance(checks, dict) or set(checks) != required
            or any(type(value) is not bool for value in checks.values())):
        raise ValueError("Required validity checks must be complete boolean flags")
    if type(summary.get("passed")) is not bool or summary["passed"] != all(checks.values()):
        raise ValueError("Validity passed flag disagrees with its checks")
    if summary["binding"] != audit_binding(root):
        raise ValueError("Validity audit lineage changed")
    artifacts = summary["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("Validity artifacts must be a hash mapping")
    csvs = {"qc/teacher-stability.csv", "qc/causal-leakage.csv", "qc/target-sensitivity.csv"}
    if not csvs.issubset(artifacts):
        raise ValueError("Validity audit omits required CSV hashes")
    for path, digest in artifacts.items():
        if Path(path).is_absolute() or not (root / path).resolve().is_relative_to(root.resolve()):
            raise ValueError("Validity artifact must be inside the run")
        if sha256_file(root / path) != digest:
            raise ValueError(f"Validity audit changed: {path}")
    cohort = load_cohort(root, verify_artifacts=True)
    contract = read_json(root / "config/cohort-contract.json")
    if "manifests/audit-windows.csv" not in contract["artifacts"]:
        raise ValueError("Cohort contract omits the frozen audit plan")
    _, cache = load_cache(root)
    planned = pd.read_csv(root / "manifests/audit-windows.csv")
    sensitivity = pd.read_csv(root / "qc/target-sensitivity.csv")
    leakage = pd.read_csv(root / "qc/causal-leakage.csv")
    stability = pd.read_csv(root / "qc/teacher-stability.csv")
    if len(planned) != 10 or not planned.window_id.is_unique:
        raise ValueError("Expected ten prespecified audit windows")
    for table in (sensitivity, leakage, stability):
        if (len(table) != 10 or not table.window_id.is_unique
                or set(table.window_id) != set(planned.window_id)):
            raise ValueError("Validity report omits or duplicates a prespecified audit window")
    planned = planned.set_index("window_id")
    sources = cohort.set_index("window_id").video_id
    for row in sensitivity.itertuples():
        if row.donor_window_id != planned.loc[row.window_id, "donor_window_id"]:
            raise ValueError("Pixel-audit donor changed after preregistration")
        if sources.loc[row.window_id] == sources.loc[row.donor_window_id]:
            raise ValueError("Audit donor belongs to recipient source")
        if f"qc/pixel-edit-contact-sheets/{row.window_id}.jpg" not in artifacts:
            raise ValueError("Missing pixel-edit contact sheet")
    distances = sensitivity[["person_change", "background_change"]].to_numpy()
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Sensitivity distances must be finite and nonnegative")
    person_larger = sensitivity.person_change > sensitivity.background_change
    _check_boolean_column(sensitivity, "person_larger", person_larger)
    ratio = float(np.median(sensitivity.person_change)
                  / max(np.median(sensitivity.background_change), 1e-8))
    direction = float(np.mean(person_larger))
    for name in ("motion_to_background_change_ratio", "person_edit_direction_fraction"):
        if type(summary.get(name)) not in (float, int) or not np.isfinite(summary[name]):
            raise ValueError("Validity summary measurements must be finite numeric scalars")
    if not np.isfinite(ratio) or not np.isfinite(direction):
        raise ValueError("Non-finite derived sensitivity measurement")
    if (not np.isclose(ratio, summary["motion_to_background_change_ratio"], rtol=1e-10, atol=1e-12)
            or direction != summary["person_edit_direction_fraction"]):
        raise ValueError("Sensitivity summary does not match its per-window evidence")
    leak_values = leakage[["max_abs_difference", "mean_abs_difference", "repeat_max_abs", "tolerance"]].to_numpy()
    stable_values = stability[["context_max_abs", "target_max_abs", "cache_max_abs"]].to_numpy()
    expected_tolerance = np.maximum(1e-6, 2 * leakage.repeat_max_abs.to_numpy())
    if (not np.isfinite(leak_values).all() or not np.isfinite(stable_values).all()
            or np.any(leak_values < 0) or np.any(stable_values < 0)
            or np.any(leakage.mean_abs_difference > leakage.max_abs_difference)
            or not np.allclose(leakage.tolerance, expected_tolerance, atol=1e-15, rtol=0)):
        raise ValueError("Malformed teacher stability/leakage evidence")
    repeats = leakage.set_index("window_id").repeat_max_abs
    if not np.array_equal(stability.context_max_abs, repeats.loc[stability.window_id]):
        raise ValueError("Stability and leakage repeat measurements disagree")
    stable = np.all(stable_values <= 1e-6, axis=1)
    causal = leakage.max_abs_difference <= leakage.tolerance
    _check_boolean_column(stability, "passed", stable)
    _check_boolean_column(leakage, "passed", causal)
    _check_boolean_column(leakage, "stable", leakage.repeat_max_abs <= 1e-6)
    thresholds = GateThresholds(**read_json(root / "config/thresholds.json"))
    expected = {
        "teacher_stable": bool(stable.all()), "causal_leakage_absent": bool(causal.all()),
        "target_audit_complete": True, "data_contract_valid": True,
        "target_sensitivity": ratio >= thresholds.motion_to_background_change_min,
        "person_edit_consistency": direction >= thresholds.person_edit_direction_fraction_min,
        "target_variance_valid": all(target_variance_checks(cohort, cache, load_model_contract(root)).values()),
    }
    if checks != expected:
        raise ValueError("Validity checks disagree with recomputed evidence and frozen thresholds")
    return summary


def require_audits(root):
    if read_json(Path(root) / "config/run-contract.json").get("protocol") == "direct-v3":
        from .fi_cache_reuse import verify_reused_readiness
        return verify_reused_readiness(root)
    if protocol_name(check_run(root)) == DIRECT_PROTOCOL:
        from .fi_readiness import verify_readiness
        summary = verify_readiness(root)
    else:
        summary = verify_audits(root)
    if not summary["passed"]:
        raise ValidityAuditRejected(root, summary)
    return summary
