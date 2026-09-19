"""Training-only CPU calibration controls for immutable post-run diagnostics.

These controls fit normalized residuals using the study's input-only, per-window
normalization. They do not fill missing observations, tune on evaluation labels,
or change the original run. The affine control is jointwise, not temporal.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import (
    JOINTS, PREPROCESSING, array_digest, digest, validate_inputs,
    validate_records, validate_targets,
)
from gavd6_sjepa.research_directions.synthetic_training_v2.data import normalize_inputs


SCHEMA = "stv2-postrun-calibration-v1"
FAMILIES = {"rtmpose_m": "rtmpose", "hrnet_w32": "hrnet", "vitpose_base": "vitpose"}
WEIGHTING = "equal people/motions/windows/variants/extractors; equal supported frames within row; per joint"


def _family(record, fixture=False):
    extractor = str(record.get("extractor", "")).strip()
    explicit = {str(record[key]).strip() for key in ("extractor_family", "family")
                if key in record and str(record[key]).strip()}
    if len(explicit) > 1:
        raise ValueError("Conflicting extractor-family metadata")
    family = next(iter(explicit), FAMILIES.get(extractor, extractor if fixture else ""))
    if not extractor or not family:
        raise ValueError("Training records require extractor ID and explicit/known family")
    if extractor in FAMILIES and family != FAMILIES[extractor]:
        raise ValueError("Extractor ID conflicts with its known family")
    return family


def _safe_records(bundle, held_extractor):
    if not isinstance(held_extractor, str) or not held_extractor.strip():
        raise ValueError("An explicit excluded extractor family or ID is required")
    held = held_extractor.strip()
    if (bundle.evidence_status != "fixture-tested"
            and held not in set(FAMILIES) | set(FAMILIES.values())):
        raise ValueError("Unresolvable held extractor: source calibration requires a known family or ID from the study roster")
    records = bundle.records
    if not records:
        raise ValueError("Calibration requires nonempty training records")
    # Reject forbidden roles/exposure before reading any target arrays.
    for record in records:
        if record.get("split") != "train" or record.get("original_split") != "train":
            raise ValueError("Calibration fits only explicit historical training records")
    validate_records(records, evidence_status=bundle.evidence_status, held_extractor=held)
    families = [_family(record, bundle.evidence_status == "fixture-tested") for record in records]
    excluded = {held, FAMILIES.get(held, held)}
    excluded.update(family for record, family in zip(records, families)
                    if record["extractor"] == held)
    if any(record["extractor"] in excluded or family in excluded
           for record, family in zip(records, families)):
        raise ValueError("Held extractor ID or family entered calibration fit")
    return records, families


def _weights(records, support):
    """Balance only eligible training observations at every nested level."""
    counts = support.sum(axis=1)
    result = np.zeros(support.shape, dtype=np.float64)
    keys = ("canonical_person_id", "motion_hash", "window_id", "variant", "extractor")

    def distribute(indices, level, mass):
        if level == len(keys):
            if len(indices) != 1:
                raise ValueError("Duplicate calibration sampling unit")
            index = indices[0]
            result[index, support[index]] = mass / counts[index]
            return
        groups = defaultdict(list)
        for index in indices:
            groups[str(records[index][keys[level]])].append(index)
        for key in sorted(groups):
            distribute(groups[key], level + 1, mass / len(groups))

    eligible = np.flatnonzero(counts).tolist()
    if not eligible:
        raise ValueError("Unsupported training joint; calibration will not invent an offset")
    distribute(eligible, 0, 1.)
    return result


def fit_calibrations(train_bundle, held_extractor, ridge=1e-3):
    """Return JSON-serializable ``joint_offset`` and ``joint_affine`` models.

    ``train_bundle`` must already contain only training rows. Passing a complete
    train/development bundle fails, rather than silently selecting safe rows.
    Source evidence requires an excluded ID/family from the fixed study roster;
    an unrecognized ID cannot silently be interpreted as a different family.
    The fixed ridge penalizes affine slopes, never the intercept; its objective
    is weighted mean squared residual error plus ridge times squared slopes.
    """
    if isinstance(ridge, bool) or not math.isfinite(float(ridge)) or float(ridge) <= 0:
        raise ValueError("Ridge must be finite and strictly positive")
    ridge = float(ridge)
    records, families = _safe_records(train_bundle, held_extractor)
    count = validate_inputs(train_bundle.inputs)
    if count != len(records):
        raise ValueError("Training records and arrays differ")
    validate_targets(train_bundle.targets, train_bundle.inputs)
    normalized, normalization = normalize_inputs(train_bundle.inputs)
    inputs = np.asarray(normalized["xy"], np.float64)
    truth = np.asarray(normalization.apply(train_bundle.targets["xy"]), np.float64)
    support = np.asarray(train_bundle.inputs["observed"]) & np.asarray(train_bundle.targets["valid"])
    offsets, affine, support_report = [], [], []
    for joint, name in enumerate(JOINTS):
        eligible = support[:, :, joint]
        weights = _weights(records, eligible)[eligible]
        x = inputs[:, :, joint][eligible]
        residual = truth[:, :, joint][eligible] - x
        offsets.append(np.sum(weights[:, None] * residual, axis=0).tolist())
        design = np.column_stack((x, np.ones(len(x))))
        penalty = np.diag([ridge, ridge, 0.])
        coefficients = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                                       design.T @ (weights[:, None] * residual))
        affine.append(coefficients.tolist())
        supporting_rows = np.flatnonzero(eligible.any(axis=1))
        support_report.append(dict(joint=name, frames=int(eligible.sum()), rows=len(supporting_rows),
            people=len({records[index]["canonical_person_id"] for index in supporting_rows})))
    provenance = dict(
        split="train", original_split="train", held_extractor=held_extractor,
        pooled_extractors=sorted({str(record["extractor"]) for record in records}),
        pooled_families=sorted(set(families)),
        canonical_people=sorted({str(record["canonical_person_id"]) for record in records}),
        rows=len(records), weighting=WEIGHTING, joint_support=support_report,
        inputs_sha256=array_digest(train_bundle.inputs), targets_sha256=array_digest(train_bundle.targets),
        records_sha256=digest(records), source_identity=train_bundle.provenance.get("identity"),
        evidence_status=train_bundle.evidence_status, target_support="observed input AND valid synthetic target",
        evaluation_labels_used=False, missing_observations="retained as NaN; no inpainting",
    )
    common = dict(schema=SCHEMA, normalization=PREPROCESSING, joints=list(JOINTS), provenance=provenance)
    return {
        "joint_offset": dict(common, method="joint_offset", offset=offsets),
        "joint_affine": dict(common, method="joint_affine", coefficients=affine, ridge=ridge,
                             features=["normalized_x", "normalized_y", "intercept"]),
    }


def apply_calibration(model, inputs):
    """Apply a frozen calibration using inference inputs only; return pixel xy."""
    if (model.get("schema") != SCHEMA or model.get("normalization") != PREPROCESSING
            or model.get("joints") != list(JOINTS)):
        raise ValueError("Incompatible calibration schema, normalization, or joint order")
    normalized, normalization = normalize_inputs(inputs)
    xy = np.asarray(normalized["xy"], np.float64)
    if model.get("method") == "joint_offset":
        offset = np.asarray(model["offset"], np.float64)
        if offset.shape != (12, 2) or not np.isfinite(offset).all():
            raise ValueError("Invalid joint offsets")
        correction = offset[None, None]
    elif model.get("method") == "joint_affine":
        coefficients = np.asarray(model["coefficients"], np.float64)
        if (coefficients.shape != (12, 3, 2) or not np.isfinite(coefficients).all()
                or model.get("features") != ["normalized_x", "normalized_y", "intercept"]):
            raise ValueError("Invalid affine calibration coefficients")
        design = np.concatenate((xy, np.ones((*xy.shape[:-1], 1))), axis=-1)
        correction = np.einsum("ntjf,jfc->ntjc", design, coefficients)
    else:
        raise ValueError("Unknown calibration method")
    output = normalization.invert(xy + correction)
    output = np.where(np.asarray(inputs["observed"])[..., None], output, np.nan)
    if np.any(np.asarray(inputs["observed"]) & ~np.isfinite(output).all(-1)):
        raise ValueError("Calibration produced nonfinite observed coordinates")
    return output
