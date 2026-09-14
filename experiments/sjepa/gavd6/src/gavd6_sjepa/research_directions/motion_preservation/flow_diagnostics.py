"""Read cached image evidence at reference-defined event locations.

These are post hoc diagnostics, not inputs to a repair method. Reference event
support tells us where to inspect measured transport errors. It does not make
those image measurements correct or establish that a 3D event was visible.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .rendering import Camera


REFERENCE_FIELDS = ("flow_reference_epe", "flow_reference_coverage_all_pixels",
                    "flow_reference_coverage_foreground")
PAIR_GROUPS = ("role", "prior_id", "flow_backend", "person_id", "motion_id")
PAIR_VALUES = ("real_event_flow_coverage", "failure_event_flow_coverage",
               "real_event_transport_gap_mean_px", "failure_event_transport_gap_mean_px",
               "paired_gap_difference_px", "both_explanations_favor_expected_path")


def _mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _event_label(value):
    text = str(value).lower()
    if text in {"true", "1", "1.0"}:
        return True
    if text in {"false", "0", "0.0"}:
        return False
    return None


def _fingerprint(case, names):
    """Compact equality check for scientific pairing, never a model feature."""
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        if name not in case:
            digest.update(b"absent")
            continue
        # Match numeric values across harmless float32/float64 cache exports.
        value = np.array(case[name], dtype=np.float64, copy=True, order="C")
        value[value == 0] = 0.0
        value[np.isnan(value)] = np.nan
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _motion_id(metadata):
    explicit = metadata.get("motion_id")
    if isinstance(explicit, str) and explicit:
        return explicit
    case_id = str(metadata.get("case_id", ""))
    stem = re.split(r"(?:_|:)(?:matched|factorial)(?:_|:)", case_id, maxsplit=1)
    if len(stem) == 2:
        return stem[0]
    scene_id = str(metadata.get("scene_id", ""))
    if re.search(r"_event[01]$", scene_id):
        return re.sub(r"_event[01]$", "", scene_id)
    relative_path = metadata.get("relative_path")
    return str(relative_path) if isinstance(relative_path, str) and relative_path else ""


def analyze_flow_case(case: dict, metadata: Mapping[str, Any], camera: Camera | None = None) -> dict:
    """Return scalar diagnostics without loading models or changing predictions.

    Transport arrays may contain T-1 pairs or the cache's T rows. Only the first
    T-1 rows are considered, so the padded last row never becomes evidence.
    Positive ``event_transport_gap_mean_px`` favors the observed trajectory over
    the prior. A gap is undefined when no event transition has valid evidence.

    Projected event displacement compares clean and edited reference positions
    in the same frame. It is a geometric image separation, not measured optical
    flow, temporal displacement, or a test of mesh visibility under occlusion.
    """
    raw, prior, clean, event = [np.asarray(case[name]) for name in
                                ("raw", "prior", "clean", "event_reference")]
    if (raw.ndim != 3 or raw.shape[-1] != 3 or len(raw) < 2
            or any(value.shape != raw.shape for value in (prior, clean, event))):
        raise ValueError("Flow diagnostics need aligned [T,J,3] raw/prior/clean/event arrays, T >= 2")
    frames, joints = raw.shape[:2]
    observed = np.asarray(case.get("observed", np.ones((frames, joints), bool)), dtype=bool)
    if observed.shape != (frames, joints):
        raise ValueError("Observed mask must align with [T,J] joints")

    def transitions(name):
        value = np.asarray(case[name])
        if value.shape not in {(frames-1, joints), (frames, joints)}:
            raise ValueError(f"{name} must have [T-1,J] or padded [T,J] shape")
        return value[:frames-1]

    raw_error, prior_error = transitions("raw_transport"), transitions("prior_transport")
    validity = transitions("flow_valid")
    available = np.isfinite(validity) & validity.astype(bool)
    available &= observed[:-1] & observed[1:]
    available &= np.isfinite(raw_error) & np.isfinite(prior_error)
    event_support = np.isfinite(event-clean).all(axis=-1) & (np.linalg.norm(event-clean, axis=-1) > 1e-6)
    event_transitions = event_support[:-1] | event_support[1:]
    usable = available & event_transitions
    count, valid_count = int(event_transitions.sum()), int(usable.sum())

    output = {key: metadata.get(key, "") for key in
              ("case_id", "person_id", "role", "fixture", "event_present", "event_family", "prior_id", "flow_backend")}
    output.update(
        motion_id=_motion_id(metadata),
        event_frame_joint_count=int(event_support.sum()),
        event_transition_count=count,
        event_observed_transition_count=int((event_transitions & observed[:-1] & observed[1:]).sum()),
        event_flow_valid_count=valid_count,
        event_flow_coverage=valid_count/count if count else np.nan,
        event_raw_transport_mean_px=_mean(raw_error[usable]),
        event_prior_transport_mean_px=_mean(prior_error[usable]),
        event_transport_gap_mean_px=_mean((prior_error-raw_error)[usable]),
        observation_fingerprint=_fingerprint(case, ("raw", "prior", "observed", "confidence", "frame_indices", "timestamps")),
        reference_event_fingerprint=_fingerprint(case, ("clean", "event_reference")),
        event_projected_in_frame_count=np.nan,
        event_projected_in_frame_fraction=np.nan,
        event_displacement_mean_px=np.nan,
        event_displacement_max_px=np.nan,
    )
    if camera is not None:
        clean_xy, clean_depth = camera.project(clean)
        event_xy, event_depth = camera.project(event)
        projected_valid = event_support & (clean_depth > 0) & (event_depth > 0)
        for xy in (clean_xy, event_xy):
            projected_valid &= np.isfinite(xy).all(axis=-1)
            projected_valid &= (xy[..., 0] >= 0) & (xy[..., 0] <= camera.width-1)
            projected_valid &= (xy[..., 1] >= 0) & (xy[..., 1] <= camera.height-1)
        separation = np.linalg.norm(event_xy-clean_xy, axis=-1)[projected_valid]
        output.update(
            event_projected_in_frame_count=int(projected_valid.sum()),
            event_projected_in_frame_fraction=float(projected_valid.sum()/event_support.sum()) if event_support.any() else np.nan,
            event_displacement_mean_px=_mean(separation),
            event_displacement_max_px=float(separation.max()) if separation.size else np.nan,
        )
    for key in REFERENCE_FIELDS:
        value = np.asarray(case.get(key, np.nan))
        output[key] = float(value) if value.size == 1 else np.nan
    return output


def paired_flow_cases(rows: pd.DataFrame) -> pd.DataFrame:
    """Check exactly skeleton-matched explanations, returning one row per pair.

    The contrast is the real-case gap minus the failure-case gap, using each
    video's available event locations. It is not restricted to the intersection
    of valid pixels in the two videos; inspect both coverages alongside it.
    An incomplete, duplicated, mismatched or unsupported pair yields no contrast.
    Ambiguous and factorial fixtures are excluded rather than paired by labels.
    """
    columns = [*PAIR_GROUPS, "status", "real_case_id", "failure_case_id", *PAIR_VALUES]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    selected = rows.loc[rows.fixture.eq("matched")].copy()
    records = []
    for values, group in selected.groupby(list(PAIR_GROUPS), dropna=False, sort=False):
        record = dict(zip(PAIR_GROUPS, values))
        record.update(status="incomplete_pair", real_case_id="", failure_case_id="",
                      **{key: np.nan for key in PAIR_VALUES})
        flags = group.event_present.map(_event_label)
        real, failure = group.loc[flags.eq(True)], group.loc[flags.eq(False)]
        if flags.isna().any():
            record["status"] = "invalid_event_labels"
        elif not record["motion_id"]:
            record["status"] = "unidentified_motion"
        elif len(real) > 1 or len(failure) > 1:
            record["status"] = "duplicate_explanation"
        elif len(real) == len(failure) == 1:
            a, b = real.iloc[0], failure.iloc[0]
            record.update(real_case_id=a.case_id, failure_case_id=b.case_id)
            same_input = a.observation_fingerprint == b.observation_fingerprint
            same_reference = a.reference_event_fingerprint == b.reference_event_fingerprint
            if not same_input:
                record["status"] = "unmatched_skeleton_inputs"
            elif not same_reference:
                record["status"] = "unmatched_reference_event"
            else:
                record.update(real_event_flow_coverage=a.event_flow_coverage,
                              failure_event_flow_coverage=b.event_flow_coverage,
                              real_event_transport_gap_mean_px=a.event_transport_gap_mean_px,
                              failure_event_transport_gap_mean_px=b.event_transport_gap_mean_px)
                if np.isfinite(a.event_transport_gap_mean_px) and np.isfinite(b.event_transport_gap_mean_px):
                    record.update(status="valid_available_support_pair",
                                  paired_gap_difference_px=a.event_transport_gap_mean_px-b.event_transport_gap_mean_px,
                                  both_explanations_favor_expected_path=float(a.event_transport_gap_mean_px > 0 and b.event_transport_gap_mean_px < 0))
                else:
                    record["status"] = "unresolved_flow_support"
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def paired_flow_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Average matched-case diagnostics within person, then equally over people.

    Output groups are role/prior/flow backend. Counts retain rejected pairs;
    coverages include verified pairs even when their flow is unresolved. Scalar
    contrasts and the fraction favoring both expected paths include only pairs
    with available evidence in both videos. No confidence or clinical conclusion
    is inferred from the contrast's sign.
    """
    pairs = paired_flow_cases(rows)
    groups = ["role", "prior_id", "flow_backend"]
    columns = [*groups, "n_pairs", "n_valid_pairs", "n_people", "n_valid_people", *PAIR_VALUES]
    records = []
    for values, subset in pairs.groupby(groups, dropna=False, sort=False):
        valid = subset.loc[subset.status.eq("valid_available_support_pair")]
        verified = subset.loc[subset.status.isin(("valid_available_support_pair", "unresolved_flow_support"))]
        person = verified.groupby("person_id")[list(PAIR_VALUES)].mean()
        record = dict(zip(groups, values))
        record.update(n_pairs=len(subset), n_valid_pairs=len(valid),
                      n_people=subset.person_id.nunique(), n_valid_people=valid.person_id.nunique())
        record.update({key: _mean(person[key]) for key in PAIR_VALUES})
        records.append(record)
    return pd.DataFrame(records, columns=columns)
